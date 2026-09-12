"""run_q4_endo.py：问题 4 内生几何版主解（用户拍板：收缩由模型预测，附件 2 降为验证数据）。

一条命令复现：python run_q4_endo.py
- 几何：耦合内生闭合 C（R 由湿度场逐步预测，不读附件 2；标定见 内生收缩模型.md）
- 求解器：import solver_q4（不改 02-代码 既有文件）；驱动为本文件的自适应
  一步对两半步外推 + 守恒双恒等式窗口对账（与 solver_q4.solve_coupled4 同结构）。
产物：
  ../04-结果/result4.xlsx（内生版，口径同交付口径.md §4+§5.1）
  q4_endo_convergence.csv / q4_endo_check.csv
  q4_split.csv（内生主口径 T6 效应拆分；外生行保留并标注“附件2外生几何（交叉验证）”）
  结果总账.csv（就地更新问题 4 行；以 (类别,项目,子项) 为键覆盖/去重，幂等）

幂等性：本脚本对总账与 q4_split 的写入均按“键覆盖 + 去重”，重复运行不产生重复行。
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # 03-数据
A_DIR = HERE.parent
sys.path.insert(0, str(A_DIR / "02-代码"))

from solver_q1 import R0, H, T0_K, C0
from solver_q4 import PROPS4, D_of4, k_of4, surface_m4, coupled_step4, PROBE
from solver_q23 import TH
from run_q23 import read_env, make_env
from run_q4 import (read_radius, write_xlsx_stream, check_structure4,
                    POS_CM, HEADER4, OUT4)

# ---- 内生闭合 C（标定参数，见 内生收缩模型.md §2）----
C_GLASS = 0.21
P_C = dict(eps_c=0.1108, a=1.6680, b=1.2195, eps_p=0.0709, k_p=3.7872)
CSMIN = 0.048766          # 末期 C_s（全局归一化常数，物理量；自主解步记录读得）


# ====================== CSV 读写（幂等：键覆盖 + 去重） ======================
def read_csv_rows(path):
    """读 CSV → (表头, [行列表])。"""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], [r for r in rows[1:] if r]


def write_csv(path, header, rows):
    """写 CSV（csv 模块引号封装：含逗号的单元格不再被拆列）。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def ledger_upsert(path, updates):
    """按 (类别,项目,子项) 为键覆盖/追加「结果总账.csv」；同时清除历史重复键。

    新键插入到同「类别」末行之后（无同类则追加），保证文档分组不变；
    重复运行同一 updates 不改变文件内容（幂等）。
    """
    header, body = read_csv_rows(path)
    key = lambda r: (r[0], r[1], r[2] if len(r) > 2 else "")
    seen, merged = {}, []
    for r in body:                       # 先按首次出现顺序去重（保留最后一个同键值）
        k = key(r)
        if k in seen:
            merged[seen[k]] = r
        else:
            seen[k] = len(merged)
            merged.append(r)
    for row in updates:
        row = [str(x) for x in row]
        k = key(row)
        if k in seen:
            merged[seen[k]] = row        # 覆盖
        else:
            last = max([i for i, r in enumerate(merged) if r[0] == row[0]], default=None)
            if last is None:
                merged.append(row)
            else:
                merged.insert(last + 1, row)
            seen = {key(r): i for i, r in enumerate(merged)}
    write_csv(path, header, merged)
    return len(merged)


def update_q4_split(path, tf_endo_h, tf_fixed_h, tf_q3_h):
    """把 q4_split.csv 更新为内生主口径（外生行保留并标注交叉验证）；幂等。

    行：appendix4_endogenous_R(t)（内生主解） + 由其派生的收缩/净效应；
    外生行 appendix4_R(t) 备注列标注「附件2外生几何（交叉验证）」。
    """
    header, body = read_csv_rows(path)
    shrink = tf_fixed_h - tf_endo_h
    net = tf_q3_h - tf_endo_h
    add = {"appendix4_endogenous_R(t)": (f"{tf_endo_h:.8f}", "内生几何主解（收缩由模型预测）"),
           "shrink_effect_endo_h": (f"{shrink:.8f}", "= appendix4_fixedR − 内生主解"),
           "shrink_pct_of_fixedR_endo": (f"{shrink / tf_fixed_h * 100:.4f}",
                                         "收缩效应占固定 R 对照"),
           "net_effect_endo_h": (f"{net:.8f}", "= problem3 − 内生主解"),
           "net_pct_of_q3_endo": (f"{net / tf_q3_h * 100:.4f}", "净效应占问题 3")}
    out = []
    for r in body:
        if len(r) < 2:
            continue
        case, val = r[0], r[1]
        if case in add:                       # 去重（幂等）：内生行随后统一追加
            continue
        note = r[2] if len(r) > 2 else ""
        if case == "appendix4_R(t)":
            note = "附件2外生几何（交叉验证）"
        out.append([case, val, note])
    out += [[k, v, n] for k, (v, n) in add.items()]
    write_csv(path, ["case", "t_f_h", "note"], out)
    return shrink, net



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
                     record=None, win=600.0, stop_at_cross=True):
    """耦合内生自适应驱动（结构与 solve_coupled4 相同；R 由各子步起点状态显式预测）。

    stop_at_cross=False 时首次达标后仍推进到 t_end（crossing 照常记录）——
    A07 修复：主解须满 72 h 积分，避免评价段被 np.interp 常值填充。"""
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
        if crossing is not None and stop_at_cross:
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
    diag_r, diag_ts = [], []          # A07：R_from_state 代数闭合残差诊断

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
        # ---- A07 诊断（只量化评估，不进主解）----
        # (i) 代数闭合残差：R_ideal 下求 C_s 代入修正式得 R1；再以 R1 重解 C_s
        #     迭代一次得 R2，|R2−R1| 即未收敛残差的量级估计；
        Ta_, Ce_ = env_hold(tt)
        um_ = float(np.mean(U))
        Rt0 = R0 * math.sqrt((1 + um_) / (1 + C0))
        dq = 1.0 / N_MAIN
        Cs1, _, _ = surface_m4(U[-1], T[-1], Ce_, dq, Rt0, D_of4)
        tc1, tp1 = closure_terms(um_, Cs1)
        R1 = Rt0 * (1 + tc1) * (1 + tp1)
        Cs2, _, _ = surface_m4(U[-1], T[-1], Ce_, dq, R1, D_of4)
        tc2, tp2 = closure_terms(um_, Cs2)
        R2 = Rt0 * (1 + tc2) * (1 + tp2)
        diag_r.append(abs(R2 - R1))
        # (ii) 末单元温度 vs 重建表面温度（coupled_step4 同款重建公式）；
        ks_ = float(k_of4(0.5 * (Us + U[-1])))
        Rh_ = Rt ** 2 * dq / (8.0 * ks_) + Rt / (2.0 * H)
        Ts_rec = T[-1] + ((Ta_ - T[-1]) / Rh_) * Rt ** 2 * dq / (8.0 * ks_)
        diag_ts.append(abs(Ts_rec - T[-1]))

    print(f"[内生主解] N={N_MAIN} 耦合内生几何全程（满 72 h 积分，首次达标单独记录）…", flush=True)
    t0 = time.perf_counter()
    res = solve_endogenous(N_MAIN, env_hold, t_end=259200.0, stop_at_cross=False,
                           record=record)
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
    assert tf_h > 19.016, "瞬时首模近似参考闸门未过"

    lt = np.array([x[0] for x in log])
    rowsC = np.array([x[1] for x in log])
    colS = np.array([x[2] for x in log])
    R_pred_log = np.array([x[5] for x in log])

    # ---------- 内生几何自洽（对附件 2，真实模型轨迹满 72 h，无常值填充） ----------
    R_on_att = np.interp(t_att, lt, R_pred_log)
    res_att = R_on_att - R_att
    rmse_R = float(np.sqrt(np.mean(res_att ** 2)))
    w_pre = t_att <= cr[0]
    w_post = t_att >= cr[2]
    e_pre = float(np.sqrt(np.mean(res_att[w_pre] ** 2)))
    e_post = float(np.sqrt(np.mean(res_att[w_post] ** 2)))
    print(f"[自洽] R_pred(t) vs 附件 2（满 72 h 真实轨迹）：全程 RMSE = {rmse_R*1000:.4f} mm；"
          f"0–达标 {e_pre*1000:.4f} / 达标–72h {e_post*1000:.4f} mm；"
          f"R_pred(72h) = {R_pred_log[-1]*100:.4f} cm（附件 2: 1.198）", flush=True)
    print(f"[自洽] 末端平台核查：R_pred(60h→72h) = "
          f"{float(np.interp(216000.0, lt, R_pred_log))*100:.4f} → {R_pred_log[-1]*100:.4f} cm",
          flush=True)
    # ---------- A07 诊断汇总（R_from_state 代数闭合残差，只评估不改主解） ----------
    diag_r = np.array(diag_r)
    diag_ts = np.array(diag_ts)
    print(f"[诊断] 代数闭合残差 |R2−R1|：max {diag_r.max()*1000:.4f} mm / "
          f"mean {diag_r.mean()*1000:.4f} mm；末单元温度 vs 重建表面温度差："
          f"max {diag_ts.max():.4f} ℃ / mean {diag_ts.mean():.4f} ℃", flush=True)

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
    # A07：三段误差与代数闭合诊断（真实 72 h 轨迹，无常值填充）
    with open(HERE / "q4_endo_eval.csv", "w", encoding="utf-8") as f:
        f.write("item,value\n")
        f.write(f"tf_first_crossing_h,{tf_h:.8f}\n")
        f.write(f"rmse_all_mm,{rmse_R * 1000:.6f}\n")
        f.write(f"rmse_0_to_crossing_mm,{e_pre * 1000:.6f}\n")
        f.write(f"rmse_crossing_to_72h_mm,{e_post * 1000:.6f}\n")
        f.write(f"R_pred_72h_cm,{R_pred_log[-1] * 100:.6f}\n")
        f.write(f"R_pred_60h_cm,{float(np.interp(216000.0, lt, R_pred_log)) * 100:.6f}\n")
        f.write(f"R_att_end_cm,{R_att[-1] * 100:.4f}\n")
        f.write(f"R_closure_resid_max_mm,{diag_r.max() * 1000:.6f}\n")
        f.write(f"R_closure_resid_mean_mm,{diag_r.mean() * 1000:.6f}\n")
        f.write(f"Ts_recon_vs_cell_max_C,{diag_ts.max():.6f}\n")
        f.write(f"Ts_recon_vs_cell_mean_C,{diag_ts.mean():.6f}\n")
    with open(HERE / "q4_endo_check.csv", "w", encoding="utf-8") as f:
        f.write("item,value\n")
        f.write(f"tf_endo_h,{tf_h:.8f}\n")
        f.write(f"res_water_max,{st['max_res_m']:.6e}\n")
        f.write(f"res_enthalpy_glob,{st['glob_res_h']:.6e}\n")
        f.write(f"md5_result4,{md5_4}\n")
        f.write(f"rows,{n60 + 1}\n")
        f.write(f"wall_main_s,{wall_main:.1f}\n")

    # ---------- q4_split.csv：切到内生主口径（T6 效应拆分） ----------
    split_path = HERE / "q4_split.csv"
    hdr_s, body_s = read_csv_rows(split_path)
    val = {r[0]: r[1] for r in body_s if len(r) >= 2}
    tf_fixed_h = float(val.get("appendix4_fixedR", "129.10470888"))
    tf_q3_h = float(val.get("problem3", "57.179925"))
    shrink, net = update_q4_split(split_path, tf_h, tf_fixed_h, tf_q3_h)
    print(f"[q4_split] 内生主口径已写入：内生 {tf_h:.6f} h；收缩效应 {shrink:.4f} h"
          f"（{shrink / tf_fixed_h * 100:.2f}%）；净效应 {net:.4f} h"
          f"（{net / tf_q3_h * 100:.2f}%）；外生行标注交叉验证", flush=True)

    # ---------- 结果总账：以 (类别,项目,子项) 为键覆盖/去重（幂等） ----------
    pct_shrink = shrink / tf_fixed_h * 100.0
    pct_net = net / tf_q3_h * 100.0
    updates = [
        ("t_f", "问题4主值(未舍入右端点,内生几何)", "", f"{tf_h:.6f}",
         f"h（附件2外生版 {tf_exo160:.6f} 作交叉验证）"),
        ("t_f", "问题4 Richardson外推(内生几何)", "", f"{tf_rich:.6f}", "h"),
        ("敏感性", "问题4效应拆分:附4固定R(M3引用)", "", f"{tf_fixed_h:.4f}",
         f"Δ{shrink:+.4f} h（对内生主解，收缩效应 {pct_shrink:.1f}%）"),
        ("敏感性", "问题4效应拆分:附4+内生R(t)(主)", "", f"{tf_h:.4f}", "—（内生几何主解）"),
        ("敏感性", "问题4效应拆分:问题3(M2引用)", "", f"{tf_q3_h:.4f}",
         f"Δ{net:+.4f} h（公式+收缩效应 {pct_net:.1f}%）"),
        ("验证", "内生几何自洽R_pred vs 附件2 RMSE", "", f"{rmse_R * 1000:.4f}",
         "mm（满72h真实轨迹；三段见q4_endo_eval.csv）"),
        ("验证", "内生自洽三段误差0-达标/达标-72h", "", f"{e_pre * 1000:.4f}/{e_post * 1000:.4f}",
         "mm（A07口径）"),
        ("验证", "耦合内生t_f vs 外生t_f偏差", "", f"{tf_h - tf_exo160:+.4f}",
         "h（−0.22% 量级）"),
    ]
    n_led = ledger_upsert(HERE / "结果总账.csv", updates)
    print(f"[总账] 结果总账.csv 问题 4 行已按内生几何覆盖（键去重，共 {n_led} 行；重复运行不变）",
          flush=True)
    print(f"[完成] 总耗时 {time.perf_counter() - t_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
