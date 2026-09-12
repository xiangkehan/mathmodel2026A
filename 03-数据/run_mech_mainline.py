"""run_mech_mainline.py：力学驱动内生收缩主线版——幂律保水本构标定到附件 2 + 全套验证。

用法：python run_mech_mainline.py
阶段：① 标定（p*, q, Π_τ 三参数，least_squares，多起点）；
     ② 验证（7 条退化检查、留出法、bootstrap CI、三段/交叉/平台、
              末期孔隙率、耦合 t_f、与半经验闭合对比表）。
产物：mechMain_fit.csv、mechMain_validation.csv、mechMain_fields.npz、
      ../08-临时/mechMain_fit.png。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
A_DIR = HERE.parent
sys.path.insert(0, str(A_DIR / "02-代码"))
sys.path.insert(0, str(A_DIR / "03-数据"))

from solver_q1 import R0, T0_K, C0
from solver_q4 import D_of4, k_of4, rho_cp4
from mech_shrinkage import (MechParams, elem_quantities, solve_mechanics,
                            mass_heat_step, volume_check)
from run_mech_A import (P_of, make_X, env_hold, t_att, R_att, mech_only_run,
                        T_DRY, RHO_S, sC0)

PHI_G0 = 0.1726                      # 体积审计初始孔隙率下界
RHO_D0 = (1 - PHI_G0) / sC0
TH = 0.15


def P_fit(p_star, q_pow, Pi_tau, tau_exp=4.0):
    return MechParams(rho_d0=RHO_D0, rho_s=RHO_S, Kinf=1.0, Ginf=1 / 3,
                      K1=1.0, G1=1 / 3, p_star=p_star, tau0=Pi_tau * T_DRY,
                      tau_exp=tau_exp, ret_model="power", q_pow=q_pow)


def coupled_run(P, t_end, dt=1200.0, M=16, detect=True):
    """力学→几何→传质两轮迭代耦合；detect 时记录 umax 穿越区间。"""
    X = make_X(M)
    r = X.copy()
    Z = np.zeros((M, 3))
    U = np.full(M, C0)
    T = np.full(M, T0_K)
    active = None
    log_t, log_R, log_um, log_Us, log_gmin, log_gS, log_DeS, log_vol = [], [], [], [], [], [], [], []
    cross = None
    n_steps = int(round(t_end / dt))
    rec_every = max(1, n_steps // 240)
    n_newton_fail, max_res = 0, 0.0
    for k in range(1, n_steps + 1):
        tt = k * dt
        Ta, Ce_ = env_hold(tt)
        # 预测-校正耦合（每区间仅一次传质推进）：预测与校正都必须以同一份
        # 历史内变量 Z^n 为旧态（solve_mechanics 内部会推进 Z），
        # 只在接受该时间步时提交一次 Z^{n+1}（A05 修复：此前校正从 Z_p 出发，
        # 同一步松弛被推进两次，(1+Δt/τ)^{-1} 变 ^{-2}）。
        mech = solve_mechanics(r, Z, U, float(np.mean(T)), dt, X, P, active0=active)
        r_p, act_p = mech["r"], mech["active"]
        U_new, T_new, Us, Fm, Fh = mass_heat_step(U, T, dt, Ta, Ce_, r_p, X, mech["J"], P,
                                                  D_of4, k_of4, rho_cp4)
        mech2 = solve_mechanics(r_p, Z, U_new, float(np.mean(T_new)), dt, X, P,
                                active0=act_p)
        r, Z, active, U, T = mech2["r"], mech2["Z"], mech2["active"], U_new, T_new
        for m_ in (mech, mech2):
            max_res = max(max_res, m_["res_norm"])
            if not m_["newton_ok"]:
                n_newton_fail += 1
                if n_newton_fail <= 20:
                    print(f"  [warn] t={tt:.0f}s 力学牛顿未收敛 res_norm={m_['res_norm']:.2e}",
                          flush=True)
        umax = float(np.max(U))
        if detect and cross is None and umax < TH:
            cross = (tt - dt, tt)
        if k % rec_every == 0 or k == n_steps:
            _, _, J, w, g, Pore, S = elem_quantities(r, X, U, P)
            lhs, rhs = volume_check(r, X, U, P)
            log_t.append(tt)
            log_R.append(float(r[-1]))
            log_um.append(float(np.sum(U * np.diff((X / R0) ** 2))))
            log_Us.append(float(Us))
            log_gmin.append(float(np.min(g)))
            log_gS.append(float(g[-1]))
            log_DeS.append(float(P.tau(max(Us, 0.02)) / T_DRY))
            log_vol.append(float(abs(lhs - rhs) / max(lhs, 1e-300)))
    return dict(t=np.array(log_t), R=np.array(log_R), Um=np.array(log_um),
                Us=np.array(log_Us), gmin=np.array(log_gmin), gS=np.array(log_gS),
                DeS=np.array(log_DeS), volerr=np.array(log_vol),
                X=X, r=r, U=U, T=T, Z=Z, g=g, S=S, cross=cross,
                newton_fail=n_newton_fail, max_res=max_res)


def rmse_of(out, idx=None):
    Rp = np.interp(t_att, out["t"], out["R"])
    e = Rp - R_att
    if idx is not None:
        e = e[idx]
    return float(np.sqrt(np.mean(e ** 2)))


def fit_once(x0):
    """x = [log10(p*), q, log10(Π_τ), tau_exp]。"""
    def obj(x):
        P = P_fit(10 ** x[0], x[1], 10 ** x[2], x[3])
        out = coupled_run(P, t_att[-1], dt=1200.0, M=16, detect=False)
        Rp = np.interp(t_att, out["t"], out["R"])
        return Rp - R_att
    return least_squares(obj, x0, bounds=([0.0, 0.15, -3.0, 0.0], [4.0, 0.98, 3.0, 10.0]),
                         xtol=1e-3, ftol=1e-6, max_nfev=60)


def degradation(P, label=""):
    """7 条退化检查（方案 §15）。"""
    M = 16
    X = make_X(M)
    res = {}
    dt, t_end = 60.0, 4 * 3600.0
    Udec = lambda tt: np.full(M, C0 - 1.0 * tt / t_end)
    # (i) 不干燥 + 初始总应力平衡 ⇒ 无人为跳变
    _, r1, _, _ = mech_only_run(P, lambda tt: np.full(M, C0), 301.15, dt, t_end, M=M)
    res["i_nojump"] = float(np.max(np.abs(r1 - X)))
    # (ii) 关闭黏弹分支 ⇒ 纯弹性
    P2 = P_fit(P.p_star, P.q_pow, P.tau0 / T_DRY, P.tau_exp)
    P2.G1 = P2.K1 = 0.0
    _, r2, _, _ = mech_only_run(P2, Udec, 323.315, dt, t_end, M=M)
    res["ii_elastic"] = float(r2[-1])
    # (iii) τ≪ ⇒ 充分松弛 ≈ (ii)
    P3 = P_fit(P.p_star, P.q_pow, 1e-9, P.tau_exp)
    _, r3, _, _ = mech_only_run(P3, Udec, 323.315, dt, t_end, M=M)
    res["iii_vs_ii"] = float(np.max(np.abs(r3 - r2)))
    # (iv) τ≫ ⇒ 内变量近冻结
    P4 = P_fit(P.p_star, P.q_pow, 1e9, P.tau_exp)
    _, r4, Z4, _ = mech_only_run(P4, Udec, 323.315, dt, t_end, M=M)
    Emax = float(np.max(np.abs(np.log(np.maximum(np.diff(r4) / np.diff(X), 1e-300)))))
    res["iv_frozen"] = float(np.max(np.abs(Z4))) / max(Emax, 1e-300)
    # (v) 仿射映射 ⇒ 恢复 4qD/R²
    lam = 0.8
    r5 = lam * X
    q5 = (X / R0) ** 2
    rqf = (r5[2:] - r5[:-2]) / (q5[2:] - q5[:-2])
    qf = 0.25 * (np.sqrt(q5[2:]) + np.sqrt(q5[:-2])) ** 2
    res["v_affine"] = float(np.max(np.abs(1.0 / rqf ** 2 - 4.0 * qf / (lam * R0) ** 2)
                                  / (4.0 * qf / (lam * R0) ** 2)))
    # (vi) 固定外形失水 ⇒ g 按恒等式增长
    U6a = np.full(M, C0)
    U6b = np.full(M, C0 - 0.5)
    _, _, _, _, g6a, *_ = elem_quantities(X, X, U6a, P)
    _, _, _, _, g6b, *_ = elem_quantities(X, X, U6b, P)
    res["vi_growth"] = float(np.max(np.abs((g6b - g6a) - P.rho_d0 / 1000.0 * 0.5)
                                  / (P.rho_d0 / 1000.0 * 0.5)))
    return res


def main():
    t_start = time.perf_counter()
    print("=" * 64, flush=True)
    print("[标定] 幂律保水本构 p_c=p*(1−S)/S^q，参数 (p*, q, Π_τ)，多起点 …", flush=True)
    best = None
    for x0 in ([1.0, 0.8, 0.0, 4.0], [2.0, 0.5, -1.0, 6.0],
               [0.5, 0.9, 1.0, 8.0], [1.5, 0.6, 0.0, 2.0]):
        t0 = time.perf_counter()
        sol = fit_once(x0)
        rm = float(np.sqrt(np.mean(sol.fun ** 2)))
        print(f"  起点 {x0}: p*={10**sol.x[0]:.3g} q={sol.x[1]:.4f} Π_τ={10**sol.x[2]:.3g} "
              f"tau_exp={sol.x[3]:.3f} RMSE={rm*1000:.2f} mm ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        if best is None or rm < best[1]:
            best = (sol, rm)
    sol, rm_fit = best
    p_star, q_pow, Pi_tau, tau_exp = 10 ** sol.x[0], sol.x[1], 10 ** sol.x[2], sol.x[3]
    print(f"[标定] 最优: p*={p_star:.4g}, q={q_pow:.4f}, Π_τ={Pi_tau:.4g}, "
          f"tau_exp={tau_exp:.4f}, RMSE={rm_fit*1000:.2f} mm", flush=True)

    print("[生产] 拟合参数全程运行（dt=600 s）…", flush=True)
    P = P_fit(p_star, q_pow, Pi_tau, tau_exp)
    out = coupled_run(P, t_att[-1], dt=600.0, M=16, detect=True)
    Rp = np.interp(t_att, out["t"], out["R"])
    res = Rp - R_att
    rmse = float(np.sqrt(np.mean(res ** 2)))
    seg = lambda a, b: float(np.sqrt(np.mean(res[(t_att >= a) & (t_att < b)] ** 2)))
    r0 = res - res.mean()
    lag1 = float(np.sum(r0[1:] * r0[:-1]) / np.sum(r0 ** 2))
    print(f"[生产] RMSE={rmse*1000:.2f} mm，max|偏差|={np.max(np.abs(res))*1000:.2f} mm，"
          f"lag-1={lag1:.3f}", flush=True)
    print(f"       分段 RMSE: 急缩 {seg(0, 3.5*3600)*1000:.2f} / 缓缩 {seg(3.5*3600, 21*3600)*1000:.2f}"
          f" / 平台 {seg(21*3600, 1e12)*1000:.2f} mm", flush=True)
    # 滞后交叉（m=R/R_ideal 由 <1 转 >1）
    um_att = np.interp(t_att, out["t"], out["Um"])
    R_ideal = R0 * np.sqrt((1 + um_att) / (1 + C0))
    m_d = R_att / R_ideal
    m_p = Rp / R_ideal
    cross_d = t_att[np.argmax(m_d > 1.0)] / 3600
    cross_p = t_att[np.argmax(m_p > 1.0)] / 3600 if np.any(m_p > 1.0) else float("nan")
    print(f"       滞后交叉: data≈{cross_d:.1f} h, pred≈{cross_p:.1f} h；"
          f"平台: data {R_att[-1]*100:.3f} vs pred {Rp[-1]*100:.4f} cm", flush=True)
    print(f"       t_f(耦合): 穿越区间 [{out['cross'][0]:.0f}, {out['cross'][1]:.0f}] s = "
          f"{out['cross'][1]/3600:.4f} h（半经验内生 50.5369 / 附件2外生 50.8230 h）", flush=True)
    # 末期孔隙率
    _, _, J_end_arr, *_ = elem_quantities(out["r"], out["X"], out["U"], P)
    phi_pore_end = 1.0 - P.s0 / J_end_arr[-1]
    print(f"       末期孔隙率(表面单元): {phi_pore_end*100:.1f}%（附件2 独立推 ~31%）", flush=True)

    print("[验证] 7 条退化检查（拟合本构）…", flush=True)
    dg = degradation(P)
    dg["vii_gmin"] = float(np.min(out["gmin"]))
    for k, v in dg.items():
        print(f"  {k} = {v:.3e}", flush=True)

    print("[验证] 留出法（奇偶折半）…", flush=True)
    ho = {}
    for name, tr, te in (("偶→奇", slice(0, None, 2), slice(1, None, 2)),
                         ("奇→偶", slice(1, None, 2), slice(0, None, 2))):
        def obj_h(x):
            P_h = P_fit(10 ** x[0], x[1], 10 ** x[2], x[3])
            o = coupled_run(P_h, t_att[-1], dt=1800.0, M=16, detect=False)
            return np.interp(t_att[tr], o["t"], o["R"]) - R_att[tr]
        s_h = least_squares(obj_h, sol.x, bounds=([0.0, 0.15, -3.0, 0.0], [4.0, 0.98, 3.0, 10.0]),
                            xtol=1e-3, ftol=1e-6, max_nfev=60)
        P_h = P_fit(10 ** s_h.x[0], s_h.x[1], 10 ** s_h.x[2], s_h.x[3])
        o_h = coupled_run(P_h, t_att[-1], dt=1800.0, M=16, detect=False)
        e_tr = float(np.sqrt(np.mean((np.interp(t_att[tr], o_h["t"], o_h["R"]) - R_att[tr]) ** 2)))
        e_te = float(np.sqrt(np.mean((np.interp(t_att[te], o_h["t"], o_h["R"]) - R_att[te]) ** 2)))
        ho[name] = (s_h, e_tr, e_te)
        print(f"  {name}: 训练 {e_tr*1000:.2f} / 留出 {e_te*1000:.2f} mm，"
              f"参数 p*={10**s_h.x[0]:.3g} q={s_h.x[1]:.3f} Π_τ={10**s_h.x[2]:.3g} "
              f"tau_exp={s_h.x[3]:.2f}", flush=True)

    print("[验证] 块 bootstrap（B=20，块长 24，dt=1800 粗算）…", flush=True)
    rng = np.random.default_rng(20260912)
    boots = []
    res_full = res
    n = len(res_full)
    for b_ in range(20):
        starts = rng.integers(0, n, n // 24 + 1)
        resamp = np.concatenate([np.roll(res_full, s) for s in starts])[:n]
        R_syn = Rp + resamp - res_full.mean()
        def obj_b(x):
            P_b = P_fit(10 ** x[0], x[1], 10 ** x[2], x[3])
            o = coupled_run(P_b, t_att[-1], dt=1800.0, M=16, detect=False)
            return np.interp(t_att, o["t"], o["R"]) - R_syn
        s_b = least_squares(obj_b, sol.x, bounds=([0.0, 0.15, -3.0, 0.0], [4.0, 0.98, 3.0, 10.0]),
                            xtol=1e-3, ftol=1e-6, max_nfev=40)
        boots.append([10 ** s_b.x[0], s_b.x[1], 10 ** s_b.x[2], s_b.x[3]])
    boots = np.array(boots)
    for i, nm in enumerate(("p*", "q", "Π_τ", "tau_exp")):
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        print(f"  {nm}: 标定 {[p_star, q_pow, Pi_tau, tau_exp][i]:.4g}，95% CI [{lo:.3g}, {hi:.3g}]",
              flush=True)

    # ---------- 落数据 ----------
    with open(HERE / "mechMain_fit.csv", "w", encoding="utf-8") as f:
        f.write("t_s,R_data_cm,R_pred_cm,residual_mm\n")
        for i in range(len(t_att)):
            f.write(f"{t_att[i]:.0f},{R_att[i]*100:.4f},{Rp[i]*100:.4f},{res[i]*1000:.4f}\n")
    with open(HERE / "mechMain_validation.csv", "w", encoding="utf-8") as f:
        f.write("item,value\n")
        f.write(f"p_star,{p_star:.6g}\nq_pow,{q_pow:.6f}\nPi_tau,{Pi_tau:.6g}\ntau_exp,{tau_exp:.6g}\n")
        f.write(f"rmse_mm,{rmse*1000:.4f}\nmaxdev_mm,{np.max(np.abs(res))*1000:.4f}\n")
        f.write(f"lag1,{lag1:.4f}\n")
        f.write(f"seg_early_mm,{seg(0, 3.5*3600)*1000:.4f}\n")
        f.write(f"seg_mid_mm,{seg(3.5*3600, 21*3600)*1000:.4f}\n")
        f.write(f"seg_late_mm,{seg(21*3600, 1e12)*1000:.4f}\n")
        f.write(f"cross_data_h,{cross_d:.2f}\ncross_pred_h,{cross_p:.2f}\n")
        f.write(f"tf_s,{out['cross'][1]:.0f}\n")
        f.write(f"phi_pore_end_pct,{phi_pore_end*100:.2f}\n")
        for k, v in dg.items():
            f.write(f"deg_{k},{v:.6e}\n")
        for name, (s_h, e_tr, e_te) in ho.items():
            f.write(f"holdout_{name}_train_mm,{e_tr*1000:.4f}\n")
            f.write(f"holdout_{name}_test_mm,{e_te*1000:.4f}\n")
    np.savez(HERE / "mechMain_fields.npz", t=out["t"], R=out["R"], Um=out["Um"],
             DeS=out["DeS"], gS=out["gS"], X=out["X"], r=out["r"], U=out["U"],
             fit_params=np.array([p_star, q_pow, Pi_tau, tau_exp]), boots=boots)
    print(f"[完成] 总耗时 {time.perf_counter()-t_start:.0f} s", flush=True)

    # ---------- 图 ----------
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
             label=f"力学主线（幂律保水，RMSE={rmse*1000:.2f} mm）")
    ax1.plot(th, R_ideal * 100, "b:", lw=1.2, label="失水理想收缩基线")
    for x in (3.5, 21.0):
        ax1.axvline(x, color="gray", alpha=0.4, ls=":")
    ax1.set_ylabel("R (cm)")
    ax1.legend(fontsize=9)
    ax1.set_title("力学驱动内生收缩（van Genuchten 类幂律保水）标定到附件 2")
    ax2.plot(th, res * 1000, "g-")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("残差 (mm)")
    ax2.set_xlabel("t (h)")
    fig.tight_layout()
    fig.savefig(A_DIR / "08-临时" / "mechMain_fit.png", dpi=150)
    print("written: 08-临时/mechMain_fit.png", flush=True)


if __name__ == "__main__":
    main()
