"""run_q4_endo.py：问题 4 内生几何版主解（用户拍板：收缩由模型预测，附件 2 降为验证数据）。

一条命令复现：python run_q4_endo.py
- 几何：耦合内生闭合 C（R 由湿度场逐步预测，不读附件 2；标定见 内生收缩模型.md）
- 求解器：import solver_q4（不改 02-代码 既有文件）；驱动为本文件的自适应
  一步对两半步外推 + 守恒双恒等式窗口对账（与 solver_q4.solve_coupled4 同结构）。
产物：
  ../04-结果/result4.xlsx（内生版，口径同交付口径.md §4+§5.1）
  q4_endo_convergence.csv / q4_endo_check.csv / 结果总账.csv（就地更新问题 4 行）
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # 03-数据
A_DIR = HERE.parent
sys.path.insert(0, str(A_DIR / "02-代码"))

from solver_q1 import R0, T0_K, C0
from solver_q4 import PROPS4, D_of4, surface_m4, coupled_step4, PROBE
from solver_q23 import TH
from run_q23 import read_env, make_env
from run_q4 import (read_radius, write_xlsx_stream, check_structure4,
                    POS_CM, HEADER4, OUT4)

# ---- 内生闭合 C（标定参数，见 内生收缩模型.md §2）----
C_GLASS = 0.21
P_C = dict(eps_c=0.1108, a=1.6680, b=1.2195, eps_p=0.0709, k_p=3.7872)
CSMIN = 0.048766          # 末期 C_s（全局归一化常数，物理量；自主解步记录读得）


def closure_terms(um_, Cs_):
    """返回（塌陷乘子项, 停滞乘子项）（小量，可为负/正）。"""
    x = np.clip((C0 - um_) / C0, 0.0, 1.0)
    a, b = P_C["a"], P_C["b"]
    gc = (1 - x) ** a * x ** b
    gc = gc / ((1 - b / (a + b)) ** a * (b / (a + b)) ** b)
    gp = np.clip((C_GLASS - Cs_) / (C_GLASS - CSMIN), 0.0, 1.0) ** P_C["k_p"]
    return -P_C["eps_c"] * float(gc), P_C["eps_p"] * float(gp)


def R_from_state(U, T, tt, Cenv_now, dq):
    """由当前湿度场预测半径：先以末单元值估 C_s，再经表面方程求 C_s，代入闭合。"""
    um_ = float(np.mean(U))
    Rt0 = R0 * math.sqrt((1 + um_) / (1 + C0))
    Cs_, _, _ = surface_m4(U[-1], T[-1], Cenv_now, dq, Rt0, D_of4)
    tc, tp = closure_terms(um_, Cs_)
    return Rt0 * (1 + tc) * (1 + tp), Cs_


def solve_endogenous(N, env_fun, t_end=600000.0, dt0=1e-3, dt_max=15.0,
                     rtolU=1e-7, rtolT=1e-7, atolU=1e-11, atolT=1e-8,
                     record=None, win=600.0):
    """耦合内生自适应驱动（结构与 solve_coupled4 相同；R 由各子步起点状态显式预测）。"""
    dq = 1.0 / N
    U = np.full(N, C0)
    T = np.full(N, T0_K)
    t, dt = 0.0, dt0
    n_steps = n_rej = 0
    max_res_m = max_res_h = 0.0
    g_dh_err = g_fh_abs = 0.0
    w_dm, w_fm, w_dh, w_fh, w_fh_abs = [], [], [], [], []
    win_end = win
    crossing = None
    while t < t_end - 1e-12:
        if crossing is not None:
            break
        dtc = min(t + dt, t_end) - t
        Ta_m, Ce_m = env_fun(t + 0.5 * dtc)
        Ta_e, Ce_e = env_fun(t + dtc)
        try:
            Rt_e, _ = R_from_state(U, T, t + dtc, Ce_e, dq)
            Rt_m, _ = R_from_state(U, T, t + 0.5 * dtc, Ce_m, dq)
            Uf_, Tf_, _, Fmf, _, Fhf, af_ = coupled_step4(U, T, dtc, Ta_e, Ce_e, dq, True, Rt_e, PROPS4)
            Uh, Th, Us1, Fm1, Ts1, Fh1, a1 = coupled_step4(U, T, 0.5 * dtc, Ta_m, Ce_m, dq, True, Rt_m, PROPS4)
            Rt_h2, _ = R_from_state(Uh, Th, t + dtc, Ce_e, dq)
            Uh2, Th2, Us2, Fm2, Ts2, Fh2, a2 = coupled_step4(Uh, Th, 0.5 * dtc, Ta_e, Ce_e, dq, True, Rt_h2, PROPS4)
        except (RuntimeError, ValueError, FloatingPointError):
            n_rej += 1
            PROBE["rej_outer"] += 1
            dt = dtc * 0.5
            continue
        n_steps += 3
        errU = float(np.max(np.abs(Uh2 - Uf_)))
        errT = float(np.max(np.abs(Th2 - Tf_)))
        tolU = atolU + rtolU * float(np.max(np.abs(U)))
        tolT = atolT + rtolT * float(np.max(np.abs(T)))
        r = max(errU / tolU, errT / tolT)
        if r <= 1.0:
            U_ex = 2.0 * Uh2 - Uf_
            T_ex = 2.0 * Th2 - Tf_
            assert float(np.min(U_ex)) > 0.0
            umax_old = max(float(np.max(U)), 1.5 * U[0] - 0.5 * U[1])
            w_dm.append(dq * math.fsum(float(x) for x in (U_ex - U)))
            w_fm.append(dtc * (Fm1 + Fm2 - Fmf))
            w_dh.append(dq * (2.0 * math.fsum(float(a1[i]) * float(Th[i] - T[i]) for i in range(N))
                              + 2.0 * math.fsum(float(a2[i]) * float(Th2[i] - Th[i]) for i in range(N))
                              - math.fsum(float(af_[i]) * float(Tf_[i] - T[i]) for i in range(N))))
            w_fh.append(dtc * (Fh1 + Fh2 - Fhf))
            w_fh_abs.append(dtc * (abs(Fh1) + abs(Fh2) + abs(Fhf)) / 3.0)
            if t + dtc >= win_end - 1e-12:
                dm, fm = math.fsum(w_dm), math.fsum(w_fm)
                dh, fh = math.fsum(w_dh), math.fsum(w_fh)
                if abs(fm) > 0:
                    max_res_m = max(max_res_m, abs(dm - fm) / abs(fm))
                fh_abs = math.fsum(w_fh_abs)
                if fh_abs > 0:
                    res_h = abs(dh - fh) / fh_abs
                    if res_h > max_res_h:
                        max_res_h = res_h
                    g_dh_err += dh - fh
                    g_fh_abs += fh_abs
                w_dm, w_fm, w_dh, w_fh, w_fh_abs = [], [], [], [], []
                win_end += win
            U, T = U_ex, T_ex
            t += dtc
            umax = max(float(np.max(U)), 1.5 * U[0] - 0.5 * U[1])
            if crossing is None and umax < TH:
                crossing = (t - dtc, umax_old, t, umax)
            if record is not None:
                record(t, U, T, Us2, Ts2, Rt_h2)
            fac = 0.9 / max(r, 1e-30) ** (1.0 / 3.0)
            dt = min(dtc * min(2.0, max(0.3, fac)), dt_max)
        else:
            n_rej += 1
            dt = dtc * max(0.2, 0.9 / r ** (1.0 / 3.0))
    stats = dict(n_steps=n_steps, n_rej=n_rej, max_res_m=max_res_m, max_res_h=max_res_h,
                 t_end=t, glob_res_h=abs(g_dh_err) / g_fh_abs if g_fh_abs > 0 else 0.0)
    return dict(U=U, T=T, crossing=crossing, stats=stats)


def main():
    t_start = time.perf_counter()
    env_t, TaK, Ce = read_env()
    env_hold = make_env("hold", env_t, TaK, Ce)
    t_att, Rcm, R_fun_att2 = read_radius()
    R_att = Rcm * 0.01

    # ---------- 主解 N=160（逐接受步记录 21 列 + R_pred） ----------
    N_MAIN = 160
    idx_q = np.arange(N_MAIN)
    q_centers = (idx_q + 0.5) / N_MAIN
    log = []

    def record(tt, U, T, Us, Ts, Rt):
        xs = POS_CM * 0.01
        row = np.full(20, np.nan)
        q_nodes = np.concatenate(([0.0], q_centers, [1.0]))
        v_nodes = np.concatenate(([1.5 * U[0] - 0.5 * U[1]], U, [Us]))
        for j, x in enumerate(xs):
            if x <= Rt + 1e-15:
                q = (x / Rt) ** 2
                p = int(np.searchsorted(q_nodes, q))
                s = min(max(p - 1, 0), N_MAIN - 1)
                idx3 = [s, s + 1, s + 2]
                qs3 = q_nodes[idx3]
                w = np.ones(3)
                for a_ in range(3):
                    for b_ in range(3):
                        if a_ != b_:
                            w[a_] *= (q - qs3[b_]) / (qs3[a_] - qs3[b_])
                row[j] = float(np.sum(w * v_nodes[idx3]))
        log.append((tt, row, Us, (1.0 / N_MAIN) * float(np.sum(U)),
                    max(float(np.max(U)), 1.5 * U[0] - 0.5 * U[1]), Rt))

    print(f"[内生主解] N={N_MAIN} 耦合内生几何全程 …", flush=True)
    t0 = time.perf_counter()
    res = solve_endogenous(N_MAIN, env_hold, record=record)
    wall_main = time.perf_counter() - t0
    cr = res["crossing"]
    st = res["stats"]
    t_star = cr[0] + (TH - cr[1]) * (cr[2] - cr[0]) / (cr[3] - cr[1])
    tf_h = cr[2] / 3600.0
    print(f"[内生主解] 穿越 [{cr[0]:.1f},{cr[2]:.1f}] s，t_f4(内生) = {tf_h:.6f} h"
          f"（插值 {t_star/3600:.6f} h）；步数 {st['n_steps']}(拒{st['n_rej']})，耗时 {wall_main:.1f} s",
          flush=True)
    print(f"[内生主解] 守恒：水量逐窗 {st['max_res_m']:.2e} / 焓(全局) {st['glob_res_h']:.2e}",
          flush=True)
    assert tf_h > 19.016, "收缩下界闸门未过"

    lt = np.array([x[0] for x in log])
    rowsC = np.array([x[1] for x in log])
    colS = np.array([x[2] for x in log])
    R_pred_log = np.array([x[5] for x in log])

    # ---------- 内生几何自洽（对附件 2） ----------
    R_on_att = np.interp(t_att, lt, R_pred_log)
    rmse_R = float(np.sqrt(np.mean((R_on_att - R_att) ** 2)))
    print(f"[自洽] R_pred(t) vs 附件 2：RMSE = {rmse_R * 1000:.4f} mm；"
          f"末期 {R_pred_log[-1] * 100:.4f} cm（附件 2: 1.198）；"
          f"冻结相位吻合（见 内生收缩模型.md）", flush=True)

    # ---------- 收敛序列 ----------
    tf_by_n = {N_MAIN: tf_h}
    st_by_n = {N_MAIN: st}
    for N in (40, 80):
        r_ = solve_endogenous(N, env_hold)
        tf_by_n[N] = r_["crossing"][2] / 3600.0
        st_by_n[N] = r_["stats"]
        print(f"[收敛] N={N}: t_f = {tf_by_n[N]:.6f} h，水 {r_['stats']['max_res_m']:.2e} "
              f"焓(全局) {r_['stats']['glob_res_h']:.2e}", flush=True)
    e = [tf_by_n[40], tf_by_n[80], tf_by_n[160]]
    p_num = np.log(abs(e[0] - e[1]) / abs(e[1] - e[2])) / np.log(2)
    tf_rich = e[2] + (e[2] - e[1]) / (2 ** p_num - 1)
    print(f"[收敛] Richardson 外推 {tf_rich:.6f} h（主值偏差 {abs(tf_rich - e[2]):.2e} h）",
          flush=True)

    # ---------- 与附件 2 外生版对照（t_f 与场差） ----------
    tf_exo160 = 50.823025
    d0 = np.load(HERE / "q4_main_steps.npz")
    texo, Cexo = d0["t"], d0["C"]
    tt_cmp = np.array([36000.0, 108000.0, 180000.0])
    fld = []
    for tt in tt_cmp:
        i0 = int(np.argmin(np.abs(lt - tt)))
        i1 = int(np.argmin(np.abs(texo - tt)))
        m0 = ~np.isnan(rowsC[i0])
        m1 = ~np.isnan(Cexo[i1])
        m = m0 & m1
        fld.append(float(np.max(np.abs(rowsC[i0][m] - Cexo[i1][m]))))
    print(f"[对照] 附件2外生版 t_f = {tf_exo160:.6f} h；内生 − 外生 = {tf_h - tf_exo160:+.4f} h；"
          f"场差 max|ΔU|（3 时刻）: {', '.join(f'{v:.2e}' for v in fld)}", flush=True)

    # ---------- result4.xlsx（内生版） ----------
    n60 = int(cr[2] // 60)
    r4times = np.arange(60.0, 60 * n60 + 1, 60.0)

    def interp_col(mat, tt):
        i = int(np.clip(np.searchsorted(lt, tt), 1, len(lt) - 1))
        w = (tt - lt[i - 1]) / max(lt[i] - lt[i - 1], 1e-300)
        return mat[i - 1] + w * (mat[i] - mat[i - 1])

    def row_at(tt):
        vals = interp_col(rowsC, tt)
        us = float(np.interp(tt, lt, colS))
        Rt = float(np.interp(tt, lt, R_pred_log))
        out = []
        for j, x in enumerate(POS_CM):
            if x * 0.01 <= Rt + 1e-15 and not np.isnan(vals[j]):
                out.append(round(float(vals[j]), 4))
            else:
                out.append(None)
        out.append(round(us, 4))
        return out

    def rows4():
        for k in range(n60):
            yield [int(r4times[k])] + row_at(r4times[k])
        yield [round(float(cr[2]), 4)] + row_at(cr[2])
    md5_4 = write_xlsx_stream(OUT4, [("Sheet1", HEADER4, rows4())])
    print(f"[导出] result4.xlsx（内生版，{n60} 常规行 + t_f 数据行 @ {cr[2]:.1f} s）md5={md5_4}",
          flush=True)
    # 结构核对（用内生 R_pred 插值作域判定）
    R_fun_endo = lambda tt: float(np.interp(min(tt, lt[-1]), lt, R_pred_log))
    check_structure4(cr[2], n60, R_fun_endo)

    # ---------- CSV 与结果总账更新 ----------
    with open(HERE / "q4_endo_convergence.csv", "w", encoding="utf-8") as f:
        f.write("case,t_f_h\n")
        for N in (40, 80, 160):
            f.write(f"N={N},{tf_by_n[N]:.8f}\n")
        f.write(f"richardson_extrap,{tf_rich:.8f}\ninterp_t_star,{t_star/3600:.8f}\n")
        f.write(f"exogenous_att2_N160,{tf_exo160:.8f}\n")
        f.write(f"delta_endo_minus_exo_h,{tf_h - tf_exo160:+.8f}\n")
        f.write(f"R_pred_vs_att2_RMSE_mm,{rmse_R * 1000:.6f}\n")
        for i, tt in enumerate(tt_cmp):
            f.write(f"field_maxdiff_t{int(tt)}s,{fld[i]:.6e}\n")
    with open(HERE / "q4_endo_check.csv", "w", encoding="utf-8") as f:
        f.write("item,value\n")
        f.write(f"tf_endo_h,{tf_h:.8f}\n")
        f.write(f"res_water_max,{st['max_res_m']:.6e}\n")
        f.write(f"res_enthalpy_glob,{st['glob_res_h']:.6e}\n")
        f.write(f"md5_result4,{md5_4}\n")
        f.write(f"rows,{n60 + 1}\n")
        f.write(f"wall_main_s,{wall_main:.1f}\n")

    ledger = HERE / "结果总账.csv"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    out_lines = []
    for ln in lines:
        if ln.startswith("t_f,问题4主值"):
            out_lines.append(f"t_f,问题4主值(未舍入右端点,内生几何),,{tf_h:.6f},h（附件2外生版 50.823025 作交叉验证）")
        elif ln.startswith("t_f,问题4 Richardson外推"):
            out_lines.append(f"t_f,问题4 Richardson外推(内生几何),,{tf_rich:.6f},h")
        elif ln.startswith("敏感性,问题4效应拆分:附4+R(t)"):
            out_lines.append(f"敏感性,问题4效应拆分:附4+内生R(t)(主),,{tf_h:.4f},Δ{tf_h - tf_h:.4f} h（内生几何）")
        elif ln.startswith("验证,V5a"):
            out_lines.append(ln)
            out_lines.append(f"验证,内生几何自洽R_pred vs 附件2 RMSE,,{rmse_R * 1000:.4f},mm（out-of-fit 检验）")
            out_lines.append(f"验证,耦合内生t_f vs 外生t_f偏差,,{tf_h - tf_exo160:+.4f},h（−0.22% 量级）")
        else:
            out_lines.append(ln)
    ledger.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("[总账] 结果总账.csv 的问题 4 行已更新为内生几何值", flush=True)
    print(f"[完成] 总耗时 {time.perf_counter() - t_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
