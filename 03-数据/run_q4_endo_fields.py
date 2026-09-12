"""run_q4_endo_fields.py：内生问题 4 的完整时空场导出（供 3D 演化图/截面图）。

复用 run_q4_endo.py 的耦合内生求解器（不改动其任何既有行为与 result4）。
一条命令：python run_q4_endo_fields.py
产物：q4_endo_fields.npz + q4_endo_fields_README.md；并做与 result4.xlsx 的一致性抽查。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
A_DIR = HERE.parent
sys.path.insert(0, str(A_DIR / "02-代码"))

from run_q4_endo import solve_endogenous
from run_q23 import read_env, make_env

N = 160
DT_FRAME = 360.0          # 帧间隔（s），~506 帧
DR = 0.02                 # 径向步长（cm），固定物理网格 0–2.0 cm，101 点，域外 NaN
R_GRID = np.arange(0.0, 2.0 + 1e-12, DR)


def main():
    t_start = time.perf_counter()
    env_t, TaK, Ce = read_env()
    env_hold = make_env("hold", env_t, TaK, Ce)

    log = []

    def record(tt, U, T, Us, Ts, Rt):
        log.append((tt, U.copy(), T.copy(), Us, Ts, Rt))

    print("[场导出] 耦合内生求解（N=160，逐步记录）…", flush=True)
    res = solve_endogenous(N, env_hold, record=record)
    cr = res["crossing"]
    tf_s = cr[2]
    print(f"[场导出] t_f = {tf_s:.1f} s（{tf_s/3600:.4f} h），接受步 {len(log)} 步", flush=True)

    lt = np.array([x[0] for x in log])
    frames = np.arange(0.0, tf_s, DT_FRAME)
    frames = np.append(frames, tf_s)
    n_t = len(frames)
    n_r = len(R_GRID)
    C = np.full((n_t, n_r), np.nan)
    T = np.full((n_t, n_r), np.nan)
    R_t = np.zeros(n_t)
    Cs_s = np.zeros(n_t)
    Cc_s = np.zeros(n_t)

    q_centers = (np.arange(N) + 0.5) / N
    for k, tau in enumerate(frames):
        i = int(np.clip(np.searchsorted(lt, tau), 1, len(lt) - 1))
        w = (tau - lt[i - 1]) / max(lt[i] - lt[i - 1], 1e-300)
        _, U0, T0, Us0, Ts0, Rt0 = log[i - 1]
        _, U1, T1, Us1, Ts1, Rt1 = log[i]
        U = U0 + w * (U1 - U0)
        Tk = T0 + w * (T1 - T0)                       # 开尔文
        Us = Us0 + w * (Us1 - Us0)
        Ts = Ts0 + w * (Ts1 - Ts0)
        Rt = Rt0 + w * (Rt1 - Rt0)
        assert 200.0 < Tk.min() and Tk.max() < 400.0, "温度必须开尔文"
        assert 200.0 < Ts < 400.0
        # 二次重构：节点 = 中心正则外推 + 单元中心 + 表面
        q_nodes = np.concatenate(([0.0], q_centers, [1.0]))
        vC = np.concatenate(([1.5 * U[0] - 0.5 * U[1]], U, [Us]))
        vT = np.concatenate(([1.5 * Tk[0] - 0.5 * Tk[1]], Tk, [Ts]))
        for j, rcm in enumerate(R_GRID):
            rm = rcm * 0.01
            if rm > Rt + 1e-15:
                continue                                # 域外留 NaN
            q = (rm / Rt) ** 2
            p = int(np.searchsorted(q_nodes, q))
            s = min(max(p - 1, 0), N - 1)
            idx3 = [s, s + 1, s + 2]
            qs3 = q_nodes[idx3]
            wq = np.ones(3)
            for a_ in range(3):
                for b_ in range(3):
                    if a_ != b_:
                        wq[a_] *= (q - qs3[b_]) / (qs3[a_] - qs3[b_])
            C[k, j] = float(np.sum(wq * vC[idx3]))
            T[k, j] = float(np.sum(wq * vT[idx3])) - 273.15     # 开尔文→℃
        R_t[k] = Rt * 100.0
        Cs_s[k] = Us
        Cc_s[k] = 1.5 * U[0] - 0.5 * U[1]
    assert not np.any(np.isnan(C[:, 0])), "中心列不得有 NaN"
    n_in = np.isfinite(C)
    print(f"[场导出] 帧数 {n_t} × 径向 {n_r} 点；域内填充率 {n_in.mean()*100:.1f}%", flush=True)

    np.savez(HERE / "q4_endo_fields.npz",
             t_s=frames, r_cm=R_GRID, C=C, T=T, R_t=R_t, Cs=Cs_s, Cc=Cc_s)
    size = (HERE / "q4_endo_fields.npz").stat().st_size
    print(f"[场导出] q4_endo_fields.npz 写出，{size/1e6:.2f} MB", flush=True)

    # ---------- 一致性抽查（对 result4.xlsx 内生版） ----------
    wb = openpyxl.load_workbook(A_DIR / "04-结果" / "result4.xlsx", read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    ok = True
    for (tt, rcm) in ((36000.0, 1.0), (72000.0, 0.5), (126000.0, 0.3)):
        row = [r for r in rows if r[0] == tt][0]
        j_ref = int(round(rcm / 0.1))                 # 列 j: x=0.1j cm
        v_ref = row[1 + j_ref]
        k = int(np.argmin(np.abs(frames - tt)))
        j = int(np.argmin(np.abs(R_GRID - rcm)))
        v_new = C[k, j]
        d_ = abs(v_new - v_ref)
        ok &= d_ < 1e-3
        print(f"[抽查] t={tt:.0f} s, r={rcm} cm: result4={v_ref} vs npz={v_new:.4f} "
              f"（|Δ|={d_:.2e}）", flush=True)
    print(f"[抽查] 末帧中心 Cc = {Cc_s[-1]:.4f}（达标阈值 0.15，应略低）", flush=True)
    assert ok, "一致性抽查失败"
    assert Cc_s[-1] < 0.15 + 1e-3 and Cc_s[-1] > 0.13, "末帧中心值异常"
    print(f"[完成] 全部通过，总耗时 {time.perf_counter() - t_start:.1f} s", flush=True)

    (HERE / "q4_endo_fields_README.md").write_text(f"""# q4_endo_fields.npz 说明

内生问题 4（耦合内生几何，R 由湿度场经闭合 C 预测）的完整时空场。
复现：`cd 03-数据 && python run_q4_endo_fields.py`（约 {time.perf_counter()-t_start:.0f} s）。

| 键 | 形状 | 单位 | 说明 |
|---|---|---|---|
| t_s | ({n_t},) | s | 帧时刻，0 → t_f={tf_s:.1f} s，帧间隔 {DT_FRAME:.0f} s |
| r_cm | ({n_r},) | cm | **固定物理网格** 0–2.0 cm、步长 {DR} cm（非材料网格） |
| C | ({n_t},{n_r}) | kg/kg（干基） | 水分浓度场；r>R(t) 处 NaN |
| T | ({n_t},{n_r}) | ℃ | 温度场（内部开尔文计算、导出转 ℃）；域外 NaN |
| R_t | ({n_t},) | cm | 每帧外半径 R_pred(t)（闭合 C 预测） |
| Cs | ({n_t},) | kg/kg | 表面含水率 |
| Cc | ({n_t},) | kg/kg | 中心含水率（末帧 {Cc_s[-1]:.4f}≈达标） |

网格说明：径向为固定物理网格（r>R(t) 域外填 NaN）；空间重构为 q 内三点二次
（中心正则外推+单元中心+表面串联阻力值），时间帧由接受步线性插值。
""", encoding="utf-8")
    print("written: q4_endo_fields_README.md", flush=True)


if __name__ == "__main__":
    main()
