"""问题 4：仿射收缩移动域（材料坐标）+ 附录 4 物性；兼作 T6 对照与 V5 退化的通用求解器。

移动域方程（00-最终方案.md §2.1，材料坐标 q，r(q,t)=R(t)·√q，长度不变、内部等比为声明闭合）：
    U_t = ∂_q(4qD/R(t)²·U_q)；a_eff·Θ_t = ∂_q(4qk/R(t)²·Θ_q)
与固定域同形，R₀→R(t)；仿射 + 干固体守恒下无伪对流项；
移动表面 Robin 边界形式不变（跳条件中 ṘC_s 项恒等相消，不另补）。

附录 4（逐点取值）：ρ=760+90C、c_p=1850+2150·C/(C+1)、k=0.12+0.20·C/(C+1)、
D=4.2e-4·e^(−0.30/C)·e^(−3850/T)。h=25、h_m=8e-7 自附录 2 延用（声明假设）。

数值框架与 solver_q23 相同（单元中心 FV、表面半单元中点取值+Brent、BE+局部外推、
外场 Gauss-Seidel、湿场 Picard→Newton 切换）；物性包与 R(t) 经参数注入，
使本文件同时承担：② 主解（附4+R(t)）、① 对照（附4+R≡R0）、V5 退化（R≡R0、D 冻结）。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from solver_q1 import R0, H, HM, T0_K, C0, build_interp_weights, apply_interp  # noqa: F401
from solver_q23 import TH, PROBE  # noqa: F401


# ---- 附录 4 物性（逐点取值）----
def rho_cp4(C):
    C = np.asarray(C, dtype=float)
    return (760.0 + 90.0 * C) * (1850.0 + 2150.0 * C / (C + 1.0))


def k_of4(C):
    C = np.asarray(C, dtype=float)
    return 0.12 + 0.20 * C / (C + 1.0)


def D_of4(C, T):
    C = np.clip(np.asarray(C, dtype=float), 1e-3, 20.0)
    T = np.clip(np.asarray(T, dtype=float), 250.0, 700.0)
    return 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / T)


def dD_dC4(C, T):
    C = np.asarray(C, dtype=float)
    return D_of4(C, T) * 0.30 / C**2


PROPS4 = dict(rho_cp=rho_cp4, k_of=k_of4, D_of=D_of4, dD_dC=dD_dC4)


def surface_m4(CN, Ts, Cenv, dq, Rt, Dfun):
    """湿分表面（移动表面 Robin，形式不变）：半单元中点 D_eff + Brent 保护求根。"""
    def g(Cs):
        De = float(Dfun(0.5 * (Cs + CN), Ts))
        return 8.0 * De * (Cs - CN) / (Rt**2 * dq) - 2.0 * HM * (Cenv - Cs) / Rt

    hi = CN
    lo = max(1e-6, min(CN, Cenv) * 1e-3)
    if g(hi) <= 0.0:
        Cs = CN
    else:
        Cs = brentq(g, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=100)
    De = float(Dfun(0.5 * (Cs + CN), Ts))
    Fs = 2.0 * HM * (Cenv - Cs) / Rt
    return Cs, Fs, De


def coupled_step4(U, T, dt, Ta, Cenv, dq, newton, Rt, P):
    """一个耦合后向欧拉步（系数取步末 R(t)）。返回 (Unew,Tnew,Us,Fs_m,Ts,Fs_h,a_fr)。"""
    N = U.size
    qf = np.arange(1, N) * dq
    rho_cp, k_of, D_of, dD_dC = P["rho_cp"], P["k_of"], P["D_of"], P["dD_dC"]
    U_m, T_m = U.copy(), T.copy()
    Us_m, _, _ = surface_m4(U_m[-1], T_m[-1], Cenv, dq, Rt, D_of)
    a = rho_cp(U_m)
    for _ in range(40):
        PROBE["outer"] += 1
        a = rho_cp(U_m)
        kf = k_of(0.5 * (U_m[:-1] + U_m[1:]))
        ch = 4.0 * qf * kf * dt / (Rt**2 * dq**2)
        ks = float(k_of(0.5 * (Us_m + U_m[-1])))
        Rh = Rt**2 * dq / (8.0 * ks) + Rt / (2.0 * H)
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
        Ts = T_new[-1] + Fs_h * Rt**2 * dq / (8.0 * ks)
        U_new = U_m.copy()
        Tf = 0.5 * (T_new[:-1] + T_new[1:])
        for _ in range(30):
            Uf = 0.5 * (U_new[:-1] + U_new[1:])
            Df = D_of(Uf, Tf)
            b = 4.0 * qf * Df * dt / (Rt**2 * dq**2)
            if newton:
                e = 0.5 * dD_dC(Uf, Tf) / Df * (U_new[1:] - U_new[:-1])
            else:
                e = np.zeros(N - 1)
            Cs, Fs_m, Ds = surface_m4(U_new[-1], Ts, Cenv, dq, Rt, D_of)
            Rs = Rt**2 * dq / (8.0 * Ds) + Rt / (2.0 * HM)
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
        Us_m, _, _ = surface_m4(U_m[-1], Ts, Cenv, dq, Rt, D_of)
        if dUmax < 1e-10 and dTmax < 1e-8:
            break
    else:
        raise RuntimeError("耦合外场未收敛，应缩小步长")
    Us, Fs_m, _ = surface_m4(U_m[-1], Ts, Cenv, dq, Rt, D_of)
    return U_m, T_m, Us, Fs_m, Ts, Fs_h, a


def solve_coupled4(N, env_fun, R_fun, t_end, P=PROPS4, out_times=(), dt0=1e-3,
                   dt_max=10.0, atolU=1e-11, rtolU=1e-7, atolT=1e-8, rtolT=1e-7,
                   newton_below=0.6, margin=0.0, record=None, win=600.0,
                   record_each=False, detect=True):
    """移动域自适应求解。R_fun(t) → 当前半径（m）。守恒口径同 solver_q23。"""
    dq = 1.0 / N
    U = np.full(N, C0)
    T = np.full(N, T0_K)
    assert 200.0 < T0_K < 400.0
    t, dt = 0.0, dt0
    out_list = sorted(out_times)
    i_out = 0
    n_steps = n_rej = 0
    max_res_m = max_res_h = 0.0
    g_dh_err = g_fh_abs = 0.0
    w_dm, w_fm, w_dh, w_fh, w_fh_abs = [], [], [], [], []
    win_end = win
    crossing = None
    while t < t_end - 1e-12:
        if detect and crossing is not None and t >= crossing[2] + margin - 1e-12:
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
        Rt_e = R_fun(t + dtc)                    # 系数取步末半径（BE 全隐式口径）
        Rt_m = R_fun(t + 0.5 * dtc)
        Ta_m, Ce_m = env_fun(t + 0.5 * dtc)
        Ta_e, Ce_e = env_fun(t + dtc)
        assert 200.0 < Ta_m < 700.0 and 200.0 < Ta_e < 700.0, "环境温度必须开尔文"
        try:
            Uf_, Tf_, _, Fmf, _, Fhf, af_ = coupled_step4(U, T, dtc, Ta_e, Ce_e, dq, nw, Rt_e, P)
            Uh, Th, Us1, Fm1, Ts1, Fh1, a1 = coupled_step4(U, T, 0.5 * dtc, Ta_m, Ce_m, dq, nw, Rt_m, P)
            Uh2, Th2, Us2, Fm2, Ts2, Fh2, a2 = coupled_step4(Uh, Th, 0.5 * dtc, Ta_e, Ce_e, dq, nw, Rt_e, P)
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
            if detect and crossing is None and umax < TH:
                crossing = (t - dtc, umax_old, t, umax)
            if record_each and record is not None:
                record(t, U, T, Us2, Ts2)
            if i_out < len(out_list) and abs(t - out_list[i_out]) < 1e-12:
                i_out += 1
            fac = 0.9 / max(r, 1e-30) ** (1.0 / 3.0)
            dt = min(dtc * min(2.0, max(0.3, fac)), dt_max)
        else:
            n_rej += 1
            dt = dtc * max(0.2, 0.9 / r ** (1.0 / 3.0))
    stats = dict(n_steps=n_steps, n_rej=n_rej, max_res_m=max_res_m, max_res_h=max_res_h,
                 t_end=t, glob_res_h=abs(g_dh_err) / g_fh_abs if g_fh_abs > 0 else 0.0)
    return dict(U=U, T=T, crossing=crossing, stats=stats)
