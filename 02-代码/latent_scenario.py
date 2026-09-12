"""latent_scenario.py：潜热对照情景（审计 A01 处理，2026-09-12）。

无潜热基线（solver_q23.coupled_step，现行主模型）vs 表面相变汇情景：
同一物性（附录 3）、同一边界数据（附件 1 + 末值保持）、同一数值框架
（BE + 局部外推 + 表面串联阻力结构；潜热汇 Picard 线性化，随外场迭代更新）。

情景定义（诚实声明，均为情景假设而非题目给定）：
- 水通量用干密度定义 j_w = ρ_d,s·h_m·(U_s−U_e) [kg 水/(m²·s)]，
  ρ_d,s = ρ_附3(U_s)/(1+U_s) = (650+128·U_s)/(1+U_s)：表面局部干物质密度
  （单位总体积干物质质量，随表面含水率变化；注意与附录 ρ(C) 湿密度口径区别，
  本文 ρ(C) 不作质量闭合，故此干密度只用于本情景的潜热诊断）。
- 表面能量边界 −k∂_rT|_R = h(T_s−T_a) + L_v·j_w，L_v=2.4e6 J/kg（近似值）。
  离散上与基线同一串联阻力结构：表面汇等价于把环境驱动温度降低
  L_v·j_w/h（R_ext·F_lat，Picard 冻结上一外场迭代的 j_w）。
- 全部相变发生在表面、忽略表面储能；内部相变源情景不在本脚本求解
  （量级讨论见 03-数据/latent_scenario.csv 头部注释与论文 §2.3-3）。

产物：../03-数据/latent_scenario.csv（两模型 T_s/T_center/Um 逐时对照 + t_f）。
内置断言：基线 t_f 复现 57.1799 h（容差 0.02 h，与生产 run_q23 同码同参）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import solver_q23
from solver_q1 import R0, H, HM, T0_K, C0
from solver_q23 import TH, rho_cp, k_of, D_of, dD_dC, surface_m, solve_coupled
from run_q23 import read_env, make_env

LV = 2.4e6          # 近似汽化潜热 J/kg（情景声明值）
TF_BASE_REF = 57.1799     # 生产基线 t_f3（run_q23，N=160）
A_DIR = CODE_DIR.parent
OUT_CSV = A_DIR / "03-数据" / "latent_scenario.csv"


def rho_d_s(Us):
    """表面局部干物质密度 kg/m³（附录 3 湿密度 ρ(C)=650+128C 除以 1+C）。"""
    return (650.0 + 128.0 * Us) / (1.0 + Us)


def coupled_step_latent(U, T, dt, Ta, Cenv, dq, newton):
    """coupled_step 的潜热情景变体：签名与 solver_q23.coupled_step 完全一致。

    差异仅在热场表面行：rhs 增加 −dt·R_ext·F_lat/(Rh·a·dq)，
    F_lat=2·L_v·j_w/R0（q 形式通量，与 F_h 同单位），j_w 取上一外场迭代值
    （Picard 线性化）；返回的 Fs_h 为进入表面单元的总热流（已扣潜热汇）。
    """
    N = U.size
    qf = np.arange(1, N) * dq
    U_m, T_m = U.copy(), T.copy()
    Us_m, _, _ = surface_m(U_m[-1], T_m[-1], Cenv, dq)
    j_w = rho_d_s(Us_m) * HM * (Us_m - Cenv)          # kg 水/(m²·s)
    a = rho_cp(U_m)
    for _ in range(40):
        a = rho_cp(U_m)
        kf = k_of(0.5 * (U_m[:-1] + U_m[1:]))
        ch = 4.0 * qf * kf * dt / (R0**2 * dq**2)
        ks = float(k_of(0.5 * (Us_m + U_m[-1])))
        Rh = R0**2 * dq / (8.0 * ks) + R0 / (2.0 * H)
        F_lat = 2.0 * LV * j_w / R0                    # q 形式潜热汇
        diag = 1.0 + (np.concatenate(([0.0], ch)) + np.concatenate((ch, [0.0]))) / a
        diag[-1] += dt / (Rh * a[-1] * dq)
        rhs = T.copy()
        rhs[-1] += dt * (Ta - (R0 / (2.0 * H)) * F_lat) / (Rh * a[-1] * dq)
        ab = np.zeros((3, N))
        ab[0, 1:] = -ch / a[:-1]
        ab[1, :] = diag
        ab[2, :-1] = -ch / a[1:]
        T_new = solve_banded((1, 1), ab, rhs)
        Fs_h = (Ta - (R0 / (2.0 * H)) * F_lat - T_new[-1]) / Rh
        Ts = T_new[-1] + Fs_h * R0**2 * dq / (8.0 * ks)
        U_new = U_m.copy()
        Tf = 0.5 * (T_new[:-1] + T_new[1:])
        for _ in range(30):
            Uf = 0.5 * (U_new[:-1] + U_new[1:])
            Df = D_of(Uf, Tf)
            b = 4.0 * qf * Df * dt / (R0**2 * dq**2)
            if newton:
                e = 0.5 * dD_dC(Uf, Tf) / Df * (U_new[1:] - U_new[:-1])
            else:
                e = np.zeros(N - 1)
            Cs, Fs_m, Ds = surface_m(U_new[-1], Ts, Cenv, dq)
            Rs = R0**2 * dq / (8.0 * Ds) + R0 / (2.0 * HM)
            gs = dt / (Rs * dq)
            fl = b * (U_new[1:] - U_new[:-1])
            G = U_new - U - np.concatenate((fl, [gs * (Cenv - U_new[-1])])) \
                + np.concatenate(([0.0], fl))
            bn = np.concatenate((b, [0.0]))
            en = np.concatenate((e, [0.0]))
            bp = np.concatenate(([0.0], b))
            ep = np.concatenate(([0.0], e))
            J = np.zeros((3, N))
            J[0, 1:] = -bn[:-1] * (1.0 + en[:-1])
            J[1, :] = 1.0 + bn * (1.0 - en) + bp * (1.0 + ep)
            J[1, -1] += gs
            J[2, :-1] = -bn[:-1] * (1.0 - en[:-1])
            dU = solve_banded((1, 1), J, -G)
            U_new = U_new + dU
            if np.max(np.abs(dU)) < 1e-11:
                break
        dUmax = float(np.max(np.abs(U_new - U_m)))
        dTmax = float(np.max(np.abs(T_new - T_m)))
        U_m, T_m = U_new, T_new
        if not (np.all(np.isfinite(U_m)) and np.all(np.isfinite(T_m))
                and U_m.min() > -1.0 and U_m.max() < 10.0
                and T_m.min() > 200.0 and T_m.max() < 700.0):
            raise RuntimeError("迭代越界，应缩小步长")
        Us_m, _, _ = surface_m(U_m[-1], Ts, Cenv, dq)
        j_w = rho_d_s(Us_m) * HM * (Us_m - Cenv)       # Picard 更新水通量
        if dUmax < 1e-10 and dTmax < 1e-8:
            break
    else:
        raise RuntimeError("耦合外场未收敛，应缩小步长")
    Us, Fs_m, _ = surface_m(U_m[-1], Ts, Cenv, dq)
    return U_m, T_m, Us, Fs_m, Ts, Fs_h, a


def run_scenario(N, env, latent, hours=76):
    """全程求解（复用 solver_q23.solve_coupled 驱动，潜热步经模块函数临时替换注入）。

    返回 dict：t_f（crossing 右端点，与生产口径一致）、逐时
    (t_h, Ts, T_center, Um, Us) 记录。
    """
    dq = 1.0 / N
    log = {k: [] for k in ("t", "Ts", "Tc", "Um", "Us")}

    def record(tt, U, T, Us, Ts):
        log["t"].append(tt)
        log["Ts"].append(float(Ts) - 273.15)
        log["Tc"].append(float(1.5 * T[0] - 0.5 * T[1]) - 273.15)
        log["Um"].append(float(dq * np.sum(U)))
        log["Us"].append(float(Us))

    orig = solver_q23.coupled_step
    if latent:
        solver_q23.coupled_step = coupled_step_latent
    try:
        res = solve_coupled(N, env, 300000.0,
                            out_times=tuple(np.arange(3600.0, hours * 3600.0, 3600.0)),
                            dt_max=10.0, record=record)
    finally:
        solver_q23.coupled_step = orig
    t_f = res["crossing"][2] / 3600.0 if res["crossing"] else float("nan")
    return dict(t_f=t_f, log=log, stats=res["stats"])


def main():
    t0 = time.perf_counter()
    t_env, TaK, Ce = read_env()
    env_hold = make_env("hold", t_env, TaK, Ce)
    N = 160
    print(f"[基线] N={N} 无潜热全程 …", flush=True)
    base = run_scenario(N, env_hold, latent=False)
    print(f"  t_f = {base['t_f']:.4f} h（生产口径 57.1799 h）", flush=True)
    assert abs(base["t_f"] - TF_BASE_REF) < 0.02, \
        f"基线断言失败：{base['t_f']:.4f} vs {TF_BASE_REF}"
    print("  基线断言通过（|Δ|<0.02 h）", flush=True)

    print(f"[情景] N={N} 表面相变汇（L_v={LV:.2g}，ρ_d,s 表面局部）…", flush=True)
    lat = run_scenario(N, env_hold, latent=True)
    print(f"  t_f = {lat['t_f']:.4f} h（Δt_f = {lat['t_f']-base['t_f']:+.4f} h）", flush=True)

    # ---- 逐时对照表 ----
    n = min(len(base["log"]["t"]), len(lat["log"]["t"]))
    dTs = np.abs(np.array(lat["log"]["Ts"][:n]) - np.array(base["log"]["Ts"][:n]))
    dTc = np.abs(np.array(lat["log"]["Tc"][:n]) - np.array(base["log"]["Tc"][:n]))
    dUm = np.abs(np.array(lat["log"]["Um"][:n]) - np.array(base["log"]["Um"][:n]))
    print(f"  全场最大逐时 |ΔT_s|={dTs.max():.3f} ℃、|ΔT_center|={dTc.max():.3f} ℃、"
          f"|ΔUm|={dUm.max():.4f} kg/kg", flush=True)

    # ---- 潜热通量级诊断（t=1 h，潜热解的表面量，同审计口径）----
    Us1 = lat["log"]["Us"][0]
    Ts1 = lat["log"]["Ts"][0]
    Ta1 = float(env_hold(3600.0)[0]) - 273.15
    Ce1 = float(env_hold(3600.0)[1])
    jw1 = rho_d_s(Us1) * HM * (Us1 - Ce1)
    print(f"  t=1h 通量诊断：U_s={Us1:.4f}, T_s={Ts1:.2f} ℃, j_w={jw1:.3e} kg/(m²·s), "
          f"L·j_w={LV*jw1:.1f} W/m², h(T_a−T_s)={H*(Ta1-Ts1):.1f} W/m², "
          f"比值={LV*jw1/max(H*(Ta1-Ts1),1e-300):.2f}", flush=True)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("# 潜热对照情景（审计 A01）：表面相变汇 vs 无潜热基线；同一附录 3 物性、"
                "同一边界数据、同一 BE+局部外推框架；L_v=2.4e6 J/kg（近似值），\n")
        f.write("# j_w=rho_d,s*h_m*(U_s-C_env)，rho_d,s=(650+128*U_s)/(1+U_s) 表面局部干密度；"
                "情景假设，非题目给定。内部相变源情景未求解（量级见论文 §2.3-3）。\n")
        f.write(f"# t_f_base_h={base['t_f']:.6f}\n# t_f_latent_h={lat['t_f']:.6f}\n"
                f"# delta_t_f_h={lat['t_f']-base['t_f']:+.6f}\n"
                f"# max_dTs_C={dTs.max():.4f}\n# max_dTcenter_C={dTc.max():.4f}\n"
                f"# max_dUm={dUm.max():.6f}\n"
                f"# diag_1h: U_s={Us1:.4f},T_s_C={Ts1:.4f},j_w={jw1:.6e},"
                f"LjW_Wm2={LV*jw1:.2f},hDT_Wm2={H*(Ta1-Ts1):.2f}\n")
        f.write("t_h,Ts_base_C,Ts_latent_C,Tcenter_base_C,Tcenter_latent_C,Um_base,Um_latent\n")
        for i in range(n):
            f.write(f"{base['log']['t'][i]/3600:.1f},{base['log']['Ts'][i]:.4f},"
                    f"{lat['log']['Ts'][i]:.4f},{base['log']['Tc'][i]:.4f},"
                    f"{lat['log']['Tc'][i]:.4f},{base['log']['Um'][i]:.6f},"
                    f"{lat['log']['Um'][i]:.6f}\n")
    print(f"written: 03-数据/latent_scenario.csv（{n} 行逐时对照）", flush=True)
    print(f"[完成] 总耗时 {time.perf_counter()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
