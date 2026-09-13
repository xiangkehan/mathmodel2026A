"""run_mech_refit.py：A04（热算子）+A05（Z 推进）修复后的重标定 + 结构性上限分析。

用法：python run_mech_refit.py
依据：other/论文审计意见-de5b9ab-20260912.md（A04/A05）；
     数据/力学升级_修复重标定.md（本脚本产出笔记）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
A_DIR = HERE.parent
sys.path.insert(0, str(A_DIR / "代码"))
sys.path.insert(0, str(A_DIR / "数据"))

from solver_q1 import C0
from run_mech_mainline import P_fit, coupled_run, degradation, rmse_of
from run_mech_A import t_att, R_att, T_DRY, RHO_S, sC0

TH = 0.15
BOUNDS = ([0.0, 0.02, -3.0, 0.0], [4.0, 0.98, 3.0, 10.0])   # x=[log10 p*, β, log10 Π_τ, tau_exp]


def fit_once(x0, dt=1200.0):
    def obj(x):
        P = P_fit(10 ** x[0], x[1], 10 ** x[2], x[3])
        out = coupled_run(P, t_att[-1], dt=dt, M=16, detect=False)
        return np.interp(t_att, out["t"], out["R"]) - R_att
    return least_squares(obj, x0, bounds=BOUNDS, xtol=1e-3, ftol=1e-6, max_nfev=60)


def seg_err(Rp, a, b):
    m = (t_att >= a) & (t_att < b)
    return float(np.sqrt(np.mean((Rp[m] - R_att[m]) ** 2)))


def terminal_J(beta, w_end=0.0128, s0=0.1743):
    """大 p* 极限下末端平衡 J：解 J·g(S(J);β)=g(S0;β)（弹性体、内变量松弛）。"""
    import sys as _s
    _s.path.insert(0, str(A_DIR / "代码"))
    from mech_shrinkage import MechParams
    P = MechParams(rho_d0=1.0, rho_s=RHO_S, p_star=1.0, ret_model="power", q_pow=beta)
    gS0 = float(P.pbar_c(0.7907))
    from scipy.optimize import brentq

    def F(J):
        S = min(w_end / max(J - s0, 1e-12), 1.0)
        return J * float(P.pbar_c(S)) - gS0
    lo = s0 + w_end + 1e-9
    if F(lo) > 0:
        return lo
    return brentq(F, lo, 1.0, xtol=1e-12)


def main():
    t_start = time.perf_counter()
    print("=" * 64, flush=True)
    print("[重标定] A04 热算子 + A05 Z 推进修复后求解器，β 下界 0.02，多起点 …", flush=True)
    best = None
    for x0 in ([1.0, 0.15, 0.0, 0.1], [0.5, 0.05, 0.0, 0.5],
               [2.0, 0.5, -1.0, 4.0], [1.5, 0.08, -1.0, 2.0]):
        t0 = time.perf_counter()
        sol = fit_once(x0)
        rm = float(np.sqrt(np.mean(sol.fun ** 2)))
        print(f"  起点 {x0}: p*={10**sol.x[0]:.3g} β={sol.x[1]:.4f} Π_τ={10**sol.x[2]:.3g} "
              f"τexp={sol.x[3]:.3f} RMSE={rm*1000:.3f} mm ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        if best is None or rm < best[1]:
            best = (sol, rm)
    sol, rm_fit = best
    p_star, beta, Pi_tau, tau_exp = 10 ** sol.x[0], sol.x[1], 10 ** sol.x[2], sol.x[3]
    print(f"[重标定] 最优: p*={p_star:.4g}, β={beta:.4f}, Π_τ={Pi_tau:.4g}, "
          f"τexp={tau_exp:.4f}, RMSE={rm_fit*1000:.3f} mm", flush=True)

    print("[生产] 拟合参数全程运行（dt=600 s，积分到 72 h，不首次达标即停）…", flush=True)
    P = P_fit(p_star, beta, Pi_tau, tau_exp)
    out = coupled_run(P, t_att[-1], dt=600.0, M=16, detect=True)
    Rp = np.interp(t_att, out["t"], out["R"])
    res = Rp - R_att
    rmse = float(np.sqrt(np.mean(res ** 2)))
    r0 = res - res.mean()
    lag1 = float(np.sum(r0[1:] * r0[:-1]) / np.sum(r0 ** 2))
    print(f"[生产] RMSE={rmse*1000:.3f} mm, max|偏差|={np.max(np.abs(res))*1000:.2f} mm, "
          f"lag-1={lag1:.3f}", flush=True)
    print(f"       分段: 急缩 {seg_err(Rp,0,3.5*3600)*1000:.3f} / 缓缩 {seg_err(Rp,3.5*3600,21*3600)*1000:.3f}"
          f" / 平台 {seg_err(Rp,21*3600,1e12)*1000:.3f} mm", flush=True)
    tc = out["cross"]
    t_f_h = tc[1] / 3600 if tc else float("nan")
    if tc:
        w1 = t_att <= tc[0]
        w2 = t_att >= tc[1]
        print(f"       首次达标 t_f = {t_f_h:.4f} h（区间 [{tc[0]/3600:.2f},{tc[1]/3600:.2f}] h）；"
              f"误差分窗: 0–达标 {np.sqrt(np.mean(res[w1]**2))*1000:.3f} / "
              f"达标–72h {np.sqrt(np.mean(res[w2]**2))*1000:.3f} / 全程 {rmse*1000:.3f} mm",
              flush=True)
    i60 = int(np.argmin(np.abs(out["t"] - 60 * 3600)))
    print(f"       平台: pred {Rp[-1]*100:.4f} vs data {R_att[-1]*100:.3f} cm；"
          f"R(60h)={out['R'][i60]*100:.4f} → R(72h)={out['R'][-1]*100:.4f} cm"
          f"（72h 内再降 {(out['R'][i60]-out['R'][-1])*1000:.2f} mm，"
          f"{'平台保持' if out['R'][i60]-out['R'][-1] < 0.001 else '末端仍在收缩'}）；"
          f"t_f 对照 半经验 50.5369 / 附件2外生 50.82 h", flush=True)
    print(f"       g_min = {out['gmin'].min():.4f}；牛顿未收敛步数 = {out['newton_fail']}，"
          f"最大平衡残差 = {out['max_res']:.2e}", flush=True)

    print("[退化] 7 条退化检查 …", flush=True)
    dg = degradation(P)
    dg["vii_gmin"] = float(out["gmin"].min())
    for k, v in dg.items():
        print(f"  {k} = {v:.3e}", flush=True)

    print("[结构上限] 末端平衡 J(β) 曲线与平台可达性 …", flush=True)
    rows = []
    for b_ in (0.02, 0.05, 0.10, 0.15, 0.30, 0.50, 0.85):
        Jt = terminal_J(b_)
        rows.append((b_, Jt))
        print(f"  β={b_:.2f}: 末端 J_eq = {Jt:.4f} → R = {2*np.sqrt(Jt)*100:.3f} cm"
              f"{'  ← 平台 0.36/1.198 可达' if Jt >= 0.355 else ''}", flush=True)
    print(f"  附件 2: J = 0.36, R = 1.198 cm；拟合 β={beta:.3f} 的末端 J = "
          f"{terminal_J(beta):.4f} → R = {2*np.sqrt(terminal_J(beta))*100:.3f} cm", flush=True)

    print("[留出] 奇偶折半 …", flush=True)
    for name, tr, te in (("偶→奇", slice(0, None, 2), slice(1, None, 2)),
                         ("奇→偶", slice(1, None, 2), slice(0, None, 2))):
        def obj_h(x):
            P_h = P_fit(10 ** x[0], x[1], 10 ** x[2], x[3])
            o = coupled_run(P_h, t_att[-1], dt=1800.0, M=16, detect=False)
            return np.interp(t_att[tr], o["t"], o["R"]) - R_att[tr]
        s_h = least_squares(obj_h, sol.x, bounds=BOUNDS, xtol=1e-3, ftol=1e-6, max_nfev=60)
        P_h = P_fit(10 ** s_h.x[0], s_h.x[1], 10 ** s_h.x[2], s_h.x[3])
        o_h = coupled_run(P_h, t_att[-1], dt=1800.0, M=16, detect=False)
        e_tr = float(np.sqrt(np.mean((np.interp(t_att[tr], o_h["t"], o_h["R"]) - R_att[tr]) ** 2)))
        e_te = float(np.sqrt(np.mean((np.interp(t_att[te], o_h["t"], o_h["R"]) - R_att[te]) ** 2)))
        print(f"  {name}: 训练 {e_tr*1000:.3f} / 留出 {e_te*1000:.3f} mm，"
              f"β={s_h.x[1]:.4f}", flush=True)

    print("[bootstrap] 块 B=15，块长 24，dt=1800 …", flush=True)
    rng = np.random.default_rng(20260913)
    boots = []
    n = len(res)
    for b_ in range(15):
        starts = rng.integers(0, n, n // 24 + 1)
        resamp = np.concatenate([np.roll(res, s) for s in starts])[:n]
        R_syn = Rp + resamp - res.mean()
        def obj_b(x):
            P_b = P_fit(10 ** x[0], x[1], 10 ** x[2], x[3])
            o = coupled_run(P_b, t_att[-1], dt=1800.0, M=16, detect=False)
            return np.interp(t_att, o["t"], o["R"]) - R_syn
        s_b = least_squares(obj_b, sol.x, bounds=BOUNDS, xtol=1e-3, ftol=1e-6, max_nfev=40)
        boots.append([10 ** s_b.x[0], s_b.x[1], 10 ** s_b.x[2], s_b.x[3]])
    boots = np.array(boots)
    for i, nm in enumerate(("p*", "β", "Π_τ", "τexp")):
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        print(f"  {nm}: 标定 {[p_star, beta, Pi_tau, tau_exp][i]:.4g}，95% CI [{lo:.3g}, {hi:.3g}]",
              flush=True)

    with open(HERE / "mechRefit.csv", "w", encoding="utf-8") as f:
        f.write("t_s,R_data_cm,R_pred_cm,residual_mm\n")
        for i in range(len(t_att)):
            f.write(f"{t_att[i]:.0f},{R_att[i]*100:.4f},{Rp[i]*100:.4f},{res[i]*1000:.4f}\n")
    np.savez(HERE / "mechRefit_fields.npz", t=out["t"], R=out["R"], Um=out["Um"],
             DeS=out["DeS"], gS=out["gS"], X=out["X"], r=out["r"], U=out["U"],
             fit_params=np.array([p_star, beta, Pi_tau, tau_exp]), boots=boots,
             terminal=np.array(rows), t_f_h=t_f_h,
             newton_fail=out["newton_fail"], max_res=out["max_res"])
    print(f"[完成] 总耗时 {time.perf_counter()-t_start:.0f} s", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1]))
    th = t_att / 3600
    ax1.plot(th, R_att * 100, "k-", lw=2, label="附件 2（实测）")
    ax1.plot(th, Rp * 100, "r--", lw=1.8,
             label=f"重标定（β={beta:.3f}，RMSE={rmse*1000:.2f} mm）")
    ax1.set_ylabel("R (cm)")
    ax1.legend(fontsize=9)
    ax1.set_title("A04/A05 修复后重标定（幂律保水，膨压关闭）")
    ax2.plot(th, res * 1000, "g-")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("残差 (mm)")
    ax2.set_xlabel("t (h)")
    fig.tight_layout()
    fig.savefig(A_DIR / "08-临时" / "mechRefit.png", dpi=150)
    print("written: 08-临时/mechRefit.png", flush=True)


if __name__ == "__main__":
    main()
