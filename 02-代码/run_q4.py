"""问题 4 一条龙：移动域主解 + T6 效应拆分 + V5 退化 + T7 一致性 + result4.xlsx。

用法：  python run_q4.py
产物：
  ../04-结果/result4.xlsx        （Sheet1，60 s × 21 列[0…1.9,药材表面]，x>R(t) 留空，
                                  末行 = t_f 数据行）
  ../03-数据/q4_convergence.csv  （② 主解 t_f 随 N 收敛序列）
  ../03-数据/q4_split.csv        （T6 效应拆分三 t_f）
  ../03-数据/q4_v5.csv           （V5 退化对照）
  ../03-数据/q4_conservation.csv
  ../03-数据/q4_bound.csv        （收缩下界复算）
  ../03-数据/q4_t7.csv           （T7 反推对比）
  ../03-数据/q4_summary.json, q4_main_steps.npz
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from solver_q1 import R0, H, HM, T0_K, C0, build_interp_weights, apply_interp
from analytic_heat import RobinCylinder, robin_eigs
from solver_q23 import TH
from solver_q4 import PROPS4, D_of4, solve_coupled4
from run_q23 import read_env, make_env, interp_rows, write_xlsx_stream

A_DIR = CODE_DIR.parent
ATT2 = A_DIR / "01-题目" / "原始文件" / "附件2.xlsx"
TPL4 = A_DIR / "01-题目" / "原始文件" / "附件3" / "result4.xlsx"
OUT4 = A_DIR / "04-结果" / "result4.xlsx"
DATA_DIR = A_DIR / "03-数据"

N_MAIN = 160
POS_CM = np.arange(0.0, 1.91, 0.1)          # 20 固定列 0…1.9
HEADER4 = ["时间\\到药材中心的距离"] + [round(0.1 * i, 10) for i in range(20)] + ["药材表面"]


def read_radius():
    wb = openpyxl.load_workbook(ATT2, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [(r[0], r[1]) for r in ws.iter_rows(min_row=2, values_only=True)
            if r[0] is not None]
    wb.close()
    t = np.array([r[0] for r in rows], dtype=float)
    Rcm = np.array([r[1] for r in rows], dtype=float)
    assert len(t) == 145 and t[0] == 0 and t[-1] == 259200
    assert np.all(Rcm > 0), "半径必须为正"
    assert np.all(np.diff(Rcm) <= 1e-12), "半径不得反向增大"
    pchip = PchipInterpolator(t, Rcm * 0.01, extrapolate=False)

    def R_fun(tt):
        return float(pchip(min(tt, t[-1])))     # 35 h 后末值平台延拓（声明假设）

    return t, Rcm, R_fun


def q4_bound(R_fun, t_att, Rcm):
    """收缩比较下界：∫₀^t D_max·λ₁(Bi(s))²/R(s)² ds = ln[A₁(C₀−C_e)/(C_th−C_e)]，
    Bi(s)=h_m R(s)/D_max；另给固定 R0 版与 R_min 版对照。"""
    env_t, TaK, Ce = read_env()
    Ta_max = max(float(TaK.max()), T0_K)
    D_max = float(D_of4(C0, Ta_max))
    Ce_end = float(Ce[-1])
    bi_grid = np.linspace(0.3, 8.0, 400)
    lam_grid = np.array([robin_eigs(b, 1)[0] for b in bi_grid])

    def lam1(Bi):
        return float(np.interp(Bi, bi_grid, lam_grid))

    Bi_min = HM * float(Rcm.min() * 0.01) / D_max
    A1_min = None
    rc = RobinCylinder(Bi_min, D_max, R0, n_modes=10)
    A1_min = rc.A[0]
    target = np.log(A1_min * (C0 - Ce_end) / (TH - Ce_end))
    # 时间积分版（沿真实 R(s) 轨迹）
    ss = np.linspace(0, 300000, 30001)
    Rs = np.array([R_fun(x) for x in ss])
    mu = D_max * np.array([lam1(HM * r / D_max) for r in Rs])**2 / Rs**2
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (mu[1:] + mu[:-1]) * np.diff(ss))))
    t_int = float(np.interp(target, cum, ss)) if cum[-1] > target else float("nan")
    # R_min 常值版
    lam_min = lam1(Bi_min)
    t_rmin = Rcm.min()**2 * 1e-4 / (D_max * lam_min**2) * target
    # 固定 R0 版（case ① 的闸门）
    Bi0 = HM * R0 / D_max
    lam0 = lam1(Bi0)
    rc0 = RobinCylinder(Bi0, D_max, R0, n_modes=10)
    t_r0 = R0**2 / (D_max * lam0**2) * np.log(rc0.A[0] * (C0 - Ce_end) / (TH - Ce_end))
    return dict(D_max=D_max, Bi_min=Bi_min, t_int_h=t_int / 3600,
                t_rmin_h=float(t_rmin / 3600), t_r0_h=float(t_r0 / 3600),
                t_film_h=float(Rcm.min() * 0.01 / (2 * HM) *
                               np.log((C0 - Ce_end) / (TH - Ce_end)) / 3600))


def run_q4_full(N, env, R_fun, P=PROPS4, dt_max=15.0, margin=0.0):
    """全程：逐接受步记录 (t, 20固定列U(nan 留空), U_s, Umean, umax, R)。"""
    idx_q = np.arange(N)
    q_centers = (idx_q + 0.5) / N
    log = []

    def record(tt, U, T, Us, Ts):
        Rt = R_fun(tt)
        # 固定列：x ≤ R(t) 时按 q=(x/Rt)² 二次重构，否则留空（nan）
        xs = POS_CM * 0.01
        row = np.full(20, np.nan)
        q_nodes = np.concatenate(([0.0], q_centers, [1.0]))
        v_nodes = np.concatenate(([1.5 * U[0] - 0.5 * U[1]], U, [Us]))
        for j, x in enumerate(xs):
            if x <= Rt + 1e-15:
                q = (x / Rt) ** 2
                p = int(np.searchsorted(q_nodes, q))
                s = min(max(p - 1, 0), N - 1)
                idx3 = [s, s + 1, s + 2]
                qs3 = q_nodes[idx3]
                w = np.ones(3)
                for a_ in range(3):
                    for b_ in range(3):
                        if a_ != b_:
                            w[a_] *= (q - qs3[b_]) / (qs3[a_] - qs3[b_])
                row[j] = float(np.sum(w * v_nodes[idx3]))
        log.append((tt, row, Us, (1.0 / N) * float(np.sum(U)),
                    max(float(np.max(U)), 1.5 * U[0] - 0.5 * U[1]), Rt))

    res = solve_coupled4(N, env, R_fun, 600000.0, P=P, dt_max=dt_max,
                         margin=margin, record=record, record_each=True)
    return res, log


def main():
    DATA_DIR.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    env_t, TaK, Ce = read_env()
    env_hold = make_env("hold", env_t, TaK, Ce)
    t_att, Rcm, R_fun = read_radius()
    print(f"附件2: {len(t_att)} 点, R {Rcm[0]:.3f}→{Rcm[-1]:.3f} cm, 平台起点约 "
          f"{t_att[np.argmax(Rcm <= Rcm.min() + 1e-12)] / 3600:.1f} h", flush=True)

    # ---------- 收缩下界 ----------
    bnd = q4_bound(R_fun, t_att, Rcm)
    print(f"[下界] D_max={bnd['D_max']:.3e}; 时间积分版 {bnd['t_int_h']:.3f} h; "
          f"R_min 常值版 {bnd['t_rmin_h']:.3f} h; 固定R0版 {bnd['t_r0_h']:.3f} h; "
          f"膜界(R_min) {bnd['t_film_h']:.3f} h", flush=True)
    with open(DATA_DIR / "q4_bound.csv", "w", encoding="utf-8") as f:
        f.write("quantity,value\n")
        for k, v in bnd.items():
            f.write(f"{k},{v:.10g}\n")

    # ---------- ② 主解：附4 + R(t) ----------
    print(f"[②主解] N={N_MAIN} 附4+R(t) …", flush=True)
    t0 = time.perf_counter()
    res2, log2 = run_q4_full(N_MAIN, env_hold, R_fun, dt_max=15.0)
    wall_main = time.perf_counter() - t0
    cr = res2["crossing"]
    t_star = cr[0] + (TH - cr[1]) * (cr[2] - cr[0]) / (cr[3] - cr[1])
    tf4_h = cr[2] / 3600.0
    st = res2["stats"]
    print(f"[②主解] 穿越 [{cr[0]:.0f},{cr[2]:.0f}] s, t_f4 = {tf4_h:.6f} h "
          f"(插值 {t_star/3600:.6f} h), 步数 {st['n_steps']}(拒{st['n_rej']}), "
          f"耗时 {wall_main:.1f} s", flush=True)
    print(f"[②主解] 守恒 水量 {st['max_res_m']:.2e} / 焓(全局) {st['glob_res_h']:.2e}", flush=True)
    assert tf4_h > bnd["t_int_h"], "收缩下界闸门未过"

    # ---------- ② 收敛序列 ----------
    tf2_by_n = {N_MAIN: tf4_h}
    st2_by_n = {N_MAIN: st}
    for N in (40, 80):
        print(f"[②收敛] N={N} …", flush=True)
        r_, _ = run_q4_full(N, env_hold, R_fun)
        tf2_by_n[N] = r_["crossing"][2] / 3600.0
        st2_by_n[N] = r_["stats"]
        print(f"[②收敛] N={N}: t_f = {tf2_by_n[N]:.6f} h, 水 {st2_by_n[N]['max_res_m']:.2e} "
              f"焓(全局) {st2_by_n[N]['glob_res_h']:.2e}", flush=True)
    e2 = [tf2_by_n[40], tf2_by_n[80], tf2_by_n[160]]
    p2 = np.log(abs(e2[0] - e2[1]) / abs(e2[1] - e2[2])) / np.log(2)
    tf2_rich = e2[2] + (e2[2] - e2[1]) / (2**p2 - 1)
    print(f"[②V6] 阶估计 {p2:.3f}，Richardson {tf2_rich:.6f} h，主值偏差 "
          f"{abs(tf2_rich - e2[2]):.2e} h", flush=True)

    # ---------- ① 对照：附4 + 固定 R0 ----------
    print("[①对照] N=160 附4+固定R0 …", flush=True)
    R_const = lambda tt: R0
    res1, log1 = run_q4_full(N_MAIN, env_hold, R_const)
    tf1_h = res1["crossing"][2] / 3600.0
    print(f"[①对照] t_f(附4固定R) = {tf1_h:.6f} h, 守恒 水 "
          f"{res1['stats']['max_res_m']:.2e} 焓(全局) {res1['stats']['glob_res_h']:.2e}", flush=True)
    assert tf1_h > bnd["t_r0_h"], "固定R0下界闸门未过"

    # ---------- V5a：移动域代码 R≡R0 vs 独立固定域代码（附4物性） ----------
    print("[V5a] 移动域R≡R0(N=80) vs 独立固定域(附4, N=80) …", flush=True)
    import solver_q23 as sq23
    keep = (sq23.rho_cp, sq23.k_of, sq23.D_of, sq23.dD_dC)
    sq23.rho_cp, sq23.k_of = PROPS4["rho_cp"], PROPS4["k_of"]
    sq23.D_of, sq23.dD_dC = PROPS4["D_of"], PROPS4["dD_dC"]
    res_fix = sq23.solve_coupled(80, env_hold, 600000.0, out_times=(), dt_max=15.0)
    sq23.rho_cp, sq23.k_of, sq23.D_of, sq23.dD_dC = keep
    res_mov, _ = run_q4_full(80, env_hold, R_const)
    tf_fix = res_fix["crossing"][2]
    tf_mov = res_mov["crossing"][2]
    dU_v5 = float(np.max(np.abs(res_fix["U"] - res_mov["U"])))
    print(f"[V5a] t_f 固定域 {tf_fix/3600:.6f} h vs 移动域R≡R0 {tf_mov/3600:.6f} h, "
          f"Δt_f = {abs(tf_fix-tf_mov):.2f} s, 末态场差 max|ΔU| = {dU_v5:.2e}", flush=True)

    # ---------- V5b：D 冻结常数 → Bessel 级数首模 ----------
    print("[V5b] D=const 退化 …", flush=True)
    D0 = float(D_of4(C0, 323.315))
    Bi0 = HM * R0 / D0
    ana = RobinCylinder(Bi0, D0, R0, n_modes=300)
    P_const = dict(PROPS4, D_of=lambda C, T: np.full_like(np.asarray(C, float), D0),
                   dD_dC=lambda C, T: np.zeros_like(np.asarray(C, float)))
    Ce0 = float(Ce[-1])
    t_test = np.array([1e5, 2e5, 4e5])
    r160 = R0 * np.sqrt((np.arange(160) + 0.5) / 160)
    C_ana = ana.eval(r160, t_test, [0.0, t_test[-1]], [Ce0 - C0, Ce0 - C0], C0)
    res_d = solve_coupled4(160, lambda tt: (323.315, Ce0), R_const, t_test[-1],
                           P=P_const, dt_max=60.0, record_each=False, detect=False)
    # 末态与级数末点比对 + 首模商
    C_num_end = res_d["U"]
    d_end = float(np.max(np.abs(C_num_end - C_ana[-1])))
    mean_num = float(np.mean(C_num_end))
    ratio = (mean_num - Ce0) / (C0 - Ce0)
    first = ana.A[0] * ana.mean_mode[0] * np.exp(-ana.mu[0] * t_test[-1])
    print(f"[V5b] D0={D0:.3e}, Bi_m={Bi0:.3f}; t={t_test[-1]:.0f}s 末态逐点最大偏差 "
          f"{d_end:.2e}; 平均残余比 {ratio:.3e} vs 首模 {first:.3e}, 商 {ratio/first:.5f}",
          flush=True)

    # ---------- T6 效应拆分 ----------
    tf3_h = 57.179925
    shrink_eff = tf1_h - tf4_h
    total_eff = tf3_h - tf4_h
    print(f"[T6] t_f: ①附4固定R {tf1_h:.4f} | ②附4+R(t) {tf4_h:.4f} | ③问题3 {tf3_h:.4f} h",
          flush=True)
    print(f"[T6] 收缩效应 ①−② = {shrink_eff:+.4f} h（{shrink_eff/tf1_h*100:.1f}% of ①）；"
          f"公式+收缩 ③−② = {total_eff:+.4f} h（{total_eff/tf3_h*100:.1f}% of ③）", flush=True)

    # ---------- T7 收缩停止一致性 ----------
    t_freeze = float(t_att[np.argmax(Rcm <= 1.200 + 1e-12)])  # 首达 1.200 cm 的时刻
    um = np.array([x[3] for x in log2])
    lt = np.array([x[0] for x in log2])
    um_35 = float(np.interp(126000.0, lt, um))
    um_end = um[-1]
    uminf = lambda tt: (1 + C0) * (R_fun(tt) / R0) ** 2 - 1.0
    t7_rows = [(1800.0, uminf(1800.0), float(np.interp(1800.0, lt, um))),
               (126000.0, uminf(126000.0), um_35),
               (259200.0, uminf(259200.0), float(np.interp(259200.0, lt, um)))]
    print(f"[T7] 收缩约 {t_freeze/3600:.1f} h 冻结(1.200 cm)，t_f4 = {tf4_h:.1f} h，"
          f"早 {(tf4_h*3600 - t_freeze)/3600:.1f} h", flush=True)
    for tt, ui, umo in t7_rows:
        print(f"[T7] t={tt/3600:6.2f} h: 反推Ū={ui:.4f} vs 模型Ū={umo:.4f} "
              f"(比值 {umo/ui if ui>0 else float('nan'):.3f})", flush=True)

    # ---------- result4.xlsx ----------
    print("[导出] result4.xlsx …", flush=True)
    n60 = int(cr[2] // 60)
    r4times = np.arange(60.0, 60 * n60 + 1, 60.0)
    rowsC = np.array([x[1] for x in log2])      # (n_steps, 20) 含 nan
    colS = np.array([x[2] for x in log2])
    lt2 = np.array([x[0] for x in log2])

    def row_at(tt):
        i = int(np.clip(np.searchsorted(lt2, tt), 1, len(lt2) - 1))
        w = (tt - lt2[i - 1]) / max(lt2[i] - lt2[i - 1], 1e-300)
        Rt = R_fun(tt)
        us = float(np.interp(tt, lt2, colS))
        out = []
        for j, x in enumerate(POS_CM):
            if x * 0.01 > Rt + 1e-15:
                out.append(None)
                continue
            # R 单调不增 ⇒ x ≤ R(tt) ≤ R(前一步) ⇒ 前一步必有值；后一步可能留空
            v0, v1 = rowsC[i - 1, j], rowsC[i, j]
            if np.isnan(v0) and np.isnan(v1):
                out.append(round(us, 4))      # x≈Rt 的极限边缘：该处即表面
            elif np.isnan(v1):
                out.append(round(float(v0), 4))
            elif np.isnan(v0):
                out.append(round(float(v1), 4))
            else:
                out.append(round(float(v0 + w * (v1 - v0)), 4))
        out.append(round(us, 4))
        return out

    def rows4():
        for k in range(n60):
            yield [int(r4times[k])] + row_at(r4times[k])
        yield [round(float(cr[2]), 4)] + row_at(cr[2])
    md5_4 = write_xlsx_stream(OUT4, [("Sheet1", HEADER4, rows4())])
    print(f"[导出] result4.xlsx（{n60} 常规行 + t_f 数据行 @ {cr[2]:.1f} s）md5={md5_4}",
          flush=True)

    # ---------- 结构核对 ----------
    check_structure4(cr[2], n60, R_fun)

    # ---------- CSV / summary ----------
    with open(DATA_DIR / "q4_convergence.csv", "w", encoding="utf-8") as f:
        f.write("case,t_f_h\n")
        for N in (40, 80, 160):
            f.write(f"N={N},{tf2_by_n[N]:.8f}\n")
        f.write(f"richardson_extrap,{tf2_rich:.8f}\ninterp_t_star,{t_star/3600:.8f}\n")
    with open(DATA_DIR / "q4_split.csv", "w", encoding="utf-8") as f:
        f.write("case,t_f_h\nappendix4_fixedR,{:.8f}\nappendix4_R(t),{:.8f}\nproblem3,{:.8f}\n"
                .format(tf1_h, tf4_h, tf3_h))
        f.write(f"shrink_effect_h,{shrink_eff:.8f}\nshrink_pct_of_fixedR,"
                f"{shrink_eff/tf1_h*100:.4f}\nformula_plus_shrink_h,{total_eff:.8f}\n"
                f"formula_plus_shrink_pct_of_q3,{total_eff/tf3_h*100:.4f}\n")
    with open(DATA_DIR / "q4_v5.csv", "w", encoding="utf-8") as f:
        f.write("check,value\n")
        f.write(f"V5a_tf_fixed_s,{tf_fix:.4f}\nV5a_tf_moving_R0_s,{tf_mov:.4f}\n")
        f.write(f"V5a_dtf_s,{abs(tf_fix-tf_mov):.6f}\nV5a_maxdU,{dU_v5:.6e}\n")
        f.write(f"V5b_maxdev_end,{d_end:.6e}\nV5b_first_mode_quotient,{ratio/first:.6f}\n")
    with open(DATA_DIR / "q4_conservation.csv", "w", encoding="utf-8") as f:
        f.write("N,max_rel_residual_water,glob_rel_residual_enthalpy\n")
        for N in (40, 80, 160):
            s_ = st2_by_n[N]
            f.write(f"{N},{s_['max_res_m']:.6e},{s_['glob_res_h']:.6e}\n")
    with open(DATA_DIR / "q4_t7.csv", "w", encoding="utf-8") as f:
        f.write("t_s,U_inferred_from_R,U_model_mean\n")
        for tt, ui, umo in t7_rows:
            f.write(f"{tt:.0f},{ui:.8f},{umo:.8f}\n")
    np.savez(DATA_DIR / "q4_main_steps.npz", t=lt2, C=rowsC, Cs=colS, um=um,
             crossing=np.array(cr))
    summary = dict(tf4_right_s=cr[2], tf4_h=tf4_h, tf4_star_h=t_star / 3600,
                   tf4_bracket=list(cr), tf2_by_n={str(k): v for k, v in tf2_by_n.items()},
                   tf2_rich_h=float(tf2_rich), tf1_fixedR_h=tf1_h, tf3_h=tf3_h,
                   shrink_eff_h=shrink_eff, split_pct=dict(
                       shrink_of_fixedR=shrink_eff / tf1_h * 100,
                       formula_plus_shrink_of_q3=total_eff / tf3_h * 100),
                   bound=bnd, md5_result4=md5_4,
                   wall_main_s=wall_main, wall_total_s=time.perf_counter() - t_start,
                   n_main=N_MAIN, rows4=n60 + 1,
                   stats_main={k: float(v) for k, v in st.items() if np.isscalar(v)})
    (DATA_DIR / "q4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[完成] 总耗时 {time.perf_counter() - t_start:.1f} s", flush=True)


def check_structure4(t_right, n60, R_fun):
    def chk(cond, msg):
        print(("  [OK] " if cond else "  [FAIL] ") + msg, flush=True)
        assert cond, msg
    print("[核对] result4 结构（对照交付口径.md §4+§5.1）：")
    tpl = openpyxl.load_workbook(TPL4)
    out = openpyxl.load_workbook(OUT4, read_only=True)
    chk(out.sheetnames == tpl.sheetnames == ["Sheet1"], f"sheet: {out.sheetnames}")
    ws = out["Sheet1"]
    chk(ws.cell(1, 1).value == "时间\\到药材中心的距离", "A1 表头")
    hdr = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1, min_col=2, max_col=22))]
    chk(all(abs(hdr[i] - 0.1 * i) < 1e-12 for i in range(20)) and hdr[20] == "药材表面",
        f"列头 0,0.1,…,1.9,药材表面（末列 {hdr[20]!r}）")
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    chk(rows[0][0] == 60 and rows[1][0] == 120, "A 列 60,120 起")
    chk(len(rows) == n60 + 1, f"行数 {len(rows)} == {n60}+1")
    chk(abs(rows[-1][0] - t_right) < 5e-4, f"末行 A={rows[-1][0]} == t_f {t_right:.4f}（数据行）")
    # 留空规则：t=1800 行（第 30 行）1.9 列应已留空；t=60 行 1.9 列应有值
    r1800 = rows[29]
    chk(r1800[20] is None, "t=1800 s 行 1.9 cm 列留空（R(1800)<1.9）")
    chk(rows[0][20] is not None, "t=60 s 行 1.9 cm 列有值")
    chk(all(r[21] is not None for r in rows[::50]), "药材表面列始终有值")
    bad = 0
    for r in rows:
        for j, v in enumerate(r[1:]):
            Rt = R_fun(r[0])
            if v is None:
                if j < 20 and (0.1 * j * 0.01) <= Rt + 1e-15:
                    bad += 1   # 域内却留空
            elif abs(v - round(v, 4)) > 1e-12:
                bad += 1
    chk(bad == 0, f"留空/四位小数逐格核对（异常 {bad}）")
    out.close()


if __name__ == "__main__":
    main()
