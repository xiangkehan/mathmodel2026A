"""问题 2/3：双向耦合质量坐标有限体积求解器（附录 3 物性，固定 R0）。

与问题 1 的差别（00-最终方案.md §1 开关表）：物性全部随状态变、D=D(U,Θ) 双向耦合。

方程（q ∈ [0,1]，r = R0·√q，见 §2.1）：
    U_t = ∂_q(4qD(U,Θ)/R0² · U_q)
    a_eff(U)·Θ_t = ∂_q(4qk(U)/R0² · Θ_q)，a_eff = ρ(U)·c_p(U)（有效体积热容，ρ 不进湿分方程）
附录 3（逐点取值，不得常数化）：
    ρ = 650+128C；c_p = 1450+2736·C/(C+1)；k = 0.21+0.38·C/(C+1)
    D = 2.4e-3·exp(−0.45/C)·exp(−3850/T)，C 取局部干基、T 取局部开尔文。
h=25、h_m=8e-7 从附录 2 延用（声明假设，§2.2）。

离散：单元中心 FV；q=0 面通量系数为 0（正则对称自动满足）；
表面半单元内阻 + 外对流阻串联，U_s/Θ_s 由 Newton 联立重建。
时间：后向欧拉 + 一步 vs 两半步自适应；非线性：外场 Gauss-Seidel 耦合，
热场线性求解，湿场 Picard 起步、max_q U < 0.6 后切换含 ∂D/∂C 的 Newton（尾段 D 变化 16.8 倍）。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from solver_q1 import R0, H, HM, T0_K, C0, build_interp_weights, apply_interp  # noqa: F401

TH = 0.15           # 达标阈值 kg/kg（问题 3）
PROBE = {"outer": 0, "calls": 0, "rej_outer": 0}   # 诊断计数器


# ---- 附录 3 物性（逐点取值）----
def rho_cp(C):
    """有效体积热容 a_eff = ρ·c_p，J/(m³·K)。"""
    C = np.asarray(C, dtype=float)
    return (650.0 + 128.0 * C) * (1450.0 + 2736.0 * C / (C + 1.0))


def k_of(C):
    C = np.asarray(C, dtype=float)
    return 0.21 + 0.38 * C / (C + 1.0)


def D_of(C, T):
    """附录 3：D = 2.4e-3·exp(−0.45/C)·exp(−3850/T) m²/s。T 必须开尔文。

    迭代中间态可能短暂越出物理区间，钳制自变量防止 exp 溢出；
    收敛解必在物理区间内（外围 assert 保证），钳制不影响收敛值。
    """
    C = np.clip(np.asarray(C, dtype=float), 1e-3, 20.0)
    T = np.clip(np.asarray(T, dtype=float), 250.0, 700.0)
    return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / T)


def dD_dC(C, T):
    C = np.asarray(C, dtype=float)
    return D_of(C, T) * 0.45 / C**2


def surface_m(CN, Ts, Cenv, dq):
    """湿分表面：F = 8·D_eff·(C_s−CN)/(R0²·dq) = 2h_m(Cenv−C_s)/R0，Brent 保护求根。

    D_eff = D((C_s+CN)/2, T_s)：半单元中点取值，与内面通量 D_f=D((U_i+U_{i+1})/2) 同阶。
    附录 3 的 D ∝ e^{−0.45/C} 强变，若用端点 D(C_s)，C_s→Cenv 时 D→0 会产生
    "内通量被人为归零"的伪根（表面误判为瞬间干透）；中点取值消除伪根。
    返回 (C_s, F_s, D_eff)。
    """
    def g(Cs):
        De = float(D_of(0.5 * (Cs + CN), Ts))
        return 8.0 * De * (Cs - CN) / (R0**2 * dq) - 2.0 * HM * (Cenv - Cs) / R0

    hi = CN
    lo = max(1e-6, min(CN, Cenv) * 1e-3)
    if g(hi) <= 0.0:                      # Cenv ≥ CN（本情景不出现）：表面贴近环境
        Cs = CN
    else:
        Cs = brentq(g, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=100)
    De = float(D_of(0.5 * (Cs + CN), Ts))
    Fs = 2.0 * HM * (Cenv - Cs) / R0
    return Cs, Fs, De


def coupled_step(U, T, dt, Ta, Cenv, dq, newton):
    """一个耦合后向欧拉步。返回 (Unew, Tnew, Us, Fs_m, Ts, Fs_h, a_fr)。

    外场 Gauss-Seidel：热场（系数冻结在 U_m，线性）→ 湿场（Θ 冻结，Newton/Picard）。
    a_fr 为热场求解实际使用的冻结体积热容（供总焓恒等式逐项对账）。
    """
    N = U.size
    qf = np.arange(1, N) * dq
    U_m, T_m = U.copy(), T.copy()
    Us_m, _, _ = surface_m(U_m[-1], T_m[-1], Cenv, dq)
    a = rho_cp(U_m)
    for _ in range(40):
        PROBE["outer"] += 1
        # ---------- 热场（线性，系数取 U_m）----------
        a = rho_cp(U_m)
        kf = k_of(0.5 * (U_m[:-1] + U_m[1:]))
        ch = 4.0 * qf * kf * dt / (R0**2 * dq**2)        # 面通量系数（不含 a_i）
        ks = float(k_of(0.5 * (Us_m + U_m[-1])))         # 半单元中点取值（同湿分口径）
        Rh = R0**2 * dq / (8.0 * ks) + R0 / (2.0 * H)
        diag = 1.0 + (np.concatenate(([0.0], ch)) + np.concatenate((ch, [0.0]))) / a
        diag[-1] += dt / (Rh * a[-1] * dq)
        rhs = T.copy()
        rhs[-1] += dt * Ta / (Rh * a[-1] * dq)
        ab = np.zeros((3, N))
        ab[0, 1:] = -ch / a[:-1]
        ab[1, :] = diag
        ab[2, :-1] = -ch / a[1:]
        T_new = solve_banded((1, 1), ab, rhs)
        Fs_h = (Ta - T_new[-1]) / Rh
        Ts = T_new[-1] + Fs_h * R0**2 * dq / (8.0 * ks)
        # ---------- 湿场（Θ = T_new 冻结，Newton 或 Picard）----------
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
            fl = b * (U_new[1:] - U_new[:-1])            # 内面通量（dt/dq 已乘）
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
        if dUmax < 1e-10 and dTmax < 1e-8:
            break
    else:
        raise RuntimeError("耦合外场未收敛，应缩小步长")
    Us, Fs_m, _ = surface_m(U_m[-1], Ts, Cenv, dq)
    return U_m, T_m, Us, Fs_m, Ts, Fs_h, a


def solve_coupled(N, env_fun, t_end, out_times=(), dt0=1e-3, dt_max=1.0,
                  atolU=1e-11, rtolU=1e-7, atolT=1e-8, rtolT=1e-7,
                  newton_below=0.6, margin=0.0, record=None, win=600.0,
                  progress=None, record_each=False):
    """双向耦合自适应求解。

    env_fun(t) → (Ta_K, Cenv)；out_times 需精确落点记录（record 回调收 21 列之外的原场）。
    达标检测：每个接受步扫描 max_q U（含中心正则外推），首次 <0.15 后记录穿越区间，
    继续推进 margin 秒后停止。record(t, U, T, Us, Ts) 在落点时刻调用。
    返回 dict：crossing=(t_-, u_-, t_+, u_+)，stats（守恒双恒等式、步数），
    umax_locus（抽样时刻的 argmax 检验）。
    """
    dq = 1.0 / N
    U = np.full(N, C0)
    T = np.full(N, T0_K)
    assert 200.0 < T0_K < 400.0, "初温必须开尔文"
    t, dt = 0.0, dt0
    out_list = sorted(out_times)
    i_out = 0
    n_steps = n_rej = 0
    max_res_m = max_res_h = 0.0
    g_dh_err = 0.0               # 全程 Σ(dh − fh)（焓恒等式全局结算分子）
    g_fh_abs = 0.0               # 全程 Σ|fh|（总吞吐，分母）
    w_dm, w_fm, w_dh, w_fh, w_fh_abs = [], [], [], [], []
    win_end = win
    crossing = None
    t_prev = 0.0
    umax_prev = C0
    locus = []
    worst_h = []
    while t < t_end - 1e-12:
        if crossing is not None and t >= crossing[2] + margin - 1e-12:
            break
        target = t + dt
        if i_out < len(out_list):
            target = min(target, out_list[i_out])
        target = min(target, t_end)
        dtc = target - t
        if dtc <= 1e-15:
            i_out += 1
            continue
        nw = float(np.max(U)) < newton_below
        Ta_m, Ce_m = env_fun(t + 0.5 * dtc)
        Ta_e, Ce_e = env_fun(t + dtc)
        assert 200.0 < Ta_m < 700.0 and 200.0 < Ta_e < 700.0, "环境温度必须开尔文"
        # 一步 vs 两半步；接受外推解 2·半对 − 整步（局部外推，BE → 二阶）。
        # 外推态的守恒通量 = (F_h1 + F_h2 − F_f)·dt（由各 BE 恒等式线性组合，仍严格守恒）。
        try:
            Uf_, Tf_, _, Fmf, _, Fhf, af_ = coupled_step(U, T, dtc, Ta_e, Ce_e, dq, nw)
            Uh, Th, Us1, Fm1, Ts1, Fh1, a1 = coupled_step(U, T, 0.5 * dtc, Ta_m, Ce_m, dq, nw)
            Uh2, Th2, Us2, Fm2, Ts2, Fh2, a2 = coupled_step(Uh, Th, 0.5 * dtc, Ta_e, Ce_e, dq, nw)
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
            assert float(np.min(U_ex)) > 0.0, "外推产生非正水分"
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
                if fh_abs > 0:      # 环境回落段净通量可近零，分母用总吞吐防病态
                    res_h = abs(dh - fh) / fh_abs
                    if res_h > max_res_h:
                        max_res_h = res_h
                        worst_h[:] = [(res_h, win_end, dh, fh, fh_abs, len(w_dh))]
                    g_dh_err += dh - fh
                    g_fh_abs += fh_abs
                w_dm, w_fm, w_dh, w_fh, w_fh_abs = [], [], [], [], []
                win_end += win
            U, T = U_ex, T_ex
            t += dtc
            umax = max(float(np.max(U)), 1.5 * U[0] - 0.5 * U[1])
            if crossing is None and umax < TH:
                crossing = (t - dtc, umax_old, t, umax)
            if record_each and record is not None:
                record(t, U, T, Us2, Ts2)
            if i_out < len(out_list) and abs(t - out_list[i_out]) < 1e-12:
                if record is not None:
                    record(t, U, T, Us2, Ts2)
                if len(locus) < 40:
                    locus.append((out_list[i_out], int(np.argmax(U))))
                i_out += 1
            fac = 0.9 / max(r, 1e-30) ** (1.0 / 3.0)
            dt = min(dtc * min(2.0, max(0.3, fac)), dt_max)
            if progress is not None and n_steps % 3000 == 0:
                progress(t, dtc, r, float(np.max(U)), float(np.min(U)))
        else:
            n_rej += 1
            dt = dtc * max(0.2, 0.9 / r ** (1.0 / 3.0))
    stats = dict(n_steps=n_steps, n_rej=n_rej, max_res_m=max_res_m, max_res_h=max_res_h,
                 t_end=t, worst_h=worst_h,
                 glob_res_h=abs(g_dh_err) / g_fh_abs if g_fh_abs > 0 else 0.0)
    return dict(U=U, T=T, crossing=crossing, stats=stats, locus=locus)
