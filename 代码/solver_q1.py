"""问题 1：质量坐标 (q ∈ [0,1], r = R0·√q) 有限体积求解器。

依据 00-最终方案.md §2.1 / §4：
- 统一形式  U_t = ∂_q(K U_q)，K = 4qD/R0²；a_eff·Θ_t = ∂_q(4qk/(ρcpR0²)·ρcp·Θ_q)。
- 空间：单元中心 FV（两端等价半单元），共享通量；q=0 面通量系数为 0（正则对称自动满足）。
- 表面：半单元内阻 + 外对流阻力串联，表面值 C_s / T_s 由两方程联立显式重建，
  绝不以末单元平均代替表面值。
- 热场（线性，ρ/cp/k 常数，D 不依赖 T ⇒ 与湿分完全解耦）：
  半离散对称三对角系统，一次性谱分解，分段线性环境下用 φ 函数精确时间积分
  （时间积分无截断误差，V1 对拍时隔离纯空间误差）。
- 湿分（非线性，D = D(C)）：后向欧拉 + 系数冻结 Picard（表面 D_s 内层 Newton），
  一步 vs 两半步对分估计局部误差，自适应步长，拒绝不截断负值、靠缩小步长。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded

# ---- 附录 2 物性（问题 1，全常数）----
R0 = 0.02          # 半径 m
RHO = 820.0        # kg/m^3
CP = 2600.0        # J/(kg·K)
K = 0.36           # W/(m·K)
H = 25.0           # W/(m^2·K)
HM = 8e-7          # m/s
T0_K = 301.15      # 初始温度 K (= 28 ℃)
C0 = 2.55          # 初始干基含水率 kg/kg

ALPHA = K / (RHO * CP)          # 热扩散率 m^2/s
BI_H = H * R0 / K               # 热 Bi 数 = 1.3889


def D_of_C(C):
    """附录 2：D = 7e-9·exp(-0.89/C) m²/s，只依赖 C（不依赖 T）。"""
    return 7e-9 * np.exp(-0.89 / C)


def dD_dC(C):
    return D_of_C(C) * 0.89 / C**2


# ======================================================================
# 热场：半离散线性系统 + 谱分解精确时间积分
# ======================================================================

def heat_modal(N: int):
    """返回 (lam, V, w)：dT/dt = A T + gs·Ta·e_{N-1}，A = V diag(lam) Vᵀ。

    w = gs·V[N-1,:] 为环境温度的模态激励权重。
    """
    dq = 1.0 / N
    qf = np.arange(1, N) * dq                     # 内面 q_{i+1/2}
    af = 4.0 * qf * K / R0**2                     # 面通量系数 4qk/R0²
    off = af / (RHO * CP * dq**2)                 # 对称三对角副对角元
    diag = np.zeros(N)
    diag[:-1] -= off
    diag[1:] -= off
    # 表面：半单元内阻 R0²·dq/(8k) 与外阻 R0/(2h) 串联
    Rs = R0**2 * dq / (8.0 * K) + R0 / (2.0 * H)
    gs = 1.0 / (Rs * RHO * CP * dq)
    diag[-1] -= gs
    lam, V = eigh_tridiagonal(diag, off)
    w = V[-1, :] * gs
    return lam, V, w


def heat_advance(y, lam, w, dt, Ta0, Ta1):
    """模态坐标精确推进 dt，环境温度在 [t, t+dt] 内线性：Ta(0)=Ta0, Ta(dt)=Ta1。"""
    e = np.expm1(lam * dt)
    E1 = e / lam                          # ∫₀^dt e^{λ(dt-s)} ds
    E2 = (e - lam * dt) / lam**2          # ∫₀^dt e^{λ(dt-s)} s ds
    m = (Ta1 - Ta0) / dt
    return y * np.exp(lam * dt) + w * (Ta0 * E1 + m * E2)


def heat_surface(T_last, Ta, N):
    """由末单元中心值与串联阻力重建表面温度（不用末单元平均冒充表面值）。"""
    dq = 1.0 / N
    Rs_int = R0**2 * dq / (8.0 * K)
    Rs = Rs_int + R0 / (2.0 * H)
    Fs = (Ta - T_last) / Rs
    return T_last + Fs * Rs_int


# ======================================================================
# 湿分：后向欧拉 + Picard + 自适应步长
# ======================================================================

def _surface_state(CN, Cenv, dq, Dfun, dDfun):
    """给定末单元中心值 CN，联立求解表面值 C_s 与通量 F_s（内阻+外阻串联）。

    F = 8 D(C_s)(C_s−CN)/(R0²·dq) = 2 h_m (Cenv−C_s)/R0，对 C_s 做 Newton。
    """
    Cs = CN
    for _ in range(60):
        Ds = Dfun(Cs)
        g = 8.0 * Ds * (Cs - CN) / (R0**2 * dq) - 2.0 * HM * (Cenv - Cs) / R0
        gp = 8.0 * (dDfun(Cs) * (Cs - CN) + Ds) / (R0**2 * dq) + 2.0 * HM / R0
        d = g / gp
        Cs -= d
        if abs(d) < 1e-13:
            break
    Ds = Dfun(Cs)
    Fs = 2.0 * HM * (Cenv - Cs) / R0
    return Cs, Fs, Ds


def be_step(C, dt, Cenv, dq, Dfun, dDfun):
    """一个后向欧拉步：系数冻结 Picard 迭代至 max|ΔC| < 1e-11。

    返回 (Cnew, Cs, Fs)：Cs/Fs 为收敛状态下独立 Newton 重建的表面值与 Robin 通量，
    供守恒恒等式校核使用。
    """
    N = C.size
    qf = np.arange(1, N) * dq
    Cnew = C.copy()
    for _ in range(60):
        Cf = 0.5 * (Cnew[:-1] + Cnew[1:])
        Df = Dfun(Cf)
        af = 4.0 * qf * Df / R0**2
        b = af / dq**2                              # 长度 N-1
        _, _, Ds = _surface_state(Cnew[-1], Cenv, dq, Dfun, dDfun)
        Rs = R0**2 * dq / (8.0 * Ds) + R0 / (2.0 * HM)
        gs = 1.0 / (Rs * dq)
        diag = 1.0 + dt * (np.concatenate(([0.0], b)) + np.concatenate((b, [0.0])))
        diag[-1] += dt * gs
        rhs = C.copy()
        rhs[-1] += dt * gs * Cenv
        ab = np.zeros((3, N))
        ab[0, 1:] = -dt * b
        ab[1, :] = diag
        ab[2, :-1] = -dt * b
        Cnext = solve_banded((1, 1), ab, rhs)
        dmax = float(np.max(np.abs(Cnext - Cnew)))
        Cnew = Cnext
        if dmax < 1e-11:
            break
    else:
        raise RuntimeError("Picard 未收敛，应缩小步长重试")
    Cs, Fs, _ = _surface_state(Cnew[-1], Cenv, dq, Dfun, dDfun)
    return Cnew, Cs, Fs


def solve_moisture(N, cenv_fun, t_end, out_times=(), Dfun=D_of_C, dDfun=dD_dC,
                   atol=1e-11, rtol=3e-10, dt0=1e-4):
    """湿分场自适应求解。

    cenv_fun(t) → C_env(t)；out_times 为需要精确落点记录的时刻序列。
    返回 dict：outputs {t: C}, surface {t: (Cs, Fs)}, stats（步数/拒步/守恒残差）。
    """
    dq = 1.0 / N
    C = np.full(N, C0)
    t = 0.0
    dt = dt0
    out_list = sorted(out_times)
    i_out = 0
    outputs, surface = {}, {}
    n_steps = n_rej = 0
    max_cons_res = 0.0        # 每个输出窗口 |ΣΔm − Σ∫F_s dt| / |Σ∫F_s dt|
    win_dm, win_fl = [], []   # fsum 累加器（窗口 = 相邻记录时刻之间）
    win_end = out_list[0] if out_list else min(1.0, t_end)
    while t < t_end - 1e-12:
        target = t + dt
        if i_out < len(out_list):
            target = min(target, out_list[i_out])
        target = min(target, t_end)
        dtc = target - t
        if dtc <= 1e-15:
            outputs[out_list[i_out]] = C.copy()
            i_out += 1
            continue
        Cenv_mid = cenv_fun(t + 0.5 * dtc)
        Cenv_end = cenv_fun(t + dtc)
        # 一步 vs 两半步
        Cf, _, _ = be_step(C, dtc, Cenv_end, dq, Dfun, dDfun)
        Ch, _, Fs1 = be_step(C, 0.5 * dtc, Cenv_mid, dq, Dfun, dDfun)
        Ch2, Cs2, Fs2 = be_step(Ch, 0.5 * dtc, Cenv_end, dq, Dfun, dDfun)
        n_steps += 3
        err = float(np.max(np.abs(Ch2 - Cf)))       # ≈ 半对结果的局部误差（BE 一阶）
        tol = atol + rtol * float(np.max(np.abs(C)))
        if err <= tol:
            win_dm.append(dq * math.fsum(float(x) for x in (Ch2 - C)))
            win_fl.append(0.5 * dtc * float(Fs1))
            win_fl.append(0.5 * dtc * float(Fs2))
            if t + dtc >= win_end - 1e-12:          # 窗口关闭，结算守恒残差
                dm_w = math.fsum(win_dm)
                fl_w = math.fsum(win_fl)
                if abs(fl_w) > 0:
                    max_cons_res = max(max_cons_res, abs(dm_w - fl_w) / abs(fl_w))
                win_dm, win_fl = [], []
                k = i_out + 1
                win_end = out_list[k] if k < len(out_list) else min(win_end + 1.0, t_end)
            C = Ch2
            t += dtc
            if i_out < len(out_list) and abs(t - out_list[i_out]) < 1e-12:
                outputs[out_list[i_out]] = C.copy()
                surface[out_list[i_out]] = (Cs2, Fs2)
                i_out += 1
            fac = 0.9 * np.sqrt(tol / err) if err > 0 else 2.0
            dt = dtc * min(2.0, max(0.3, fac))
        else:
            n_rej += 1
            dt = dtc * max(0.2, 0.9 * np.sqrt(tol / err))
    stats = dict(n_steps=n_steps, n_rej=n_rej, max_cons_res=max_cons_res)
    return dict(C=C, outputs=outputs, surface=surface, stats=stats)


# ======================================================================
# 输出重建：单元中心值 + 正则性中心值 + 串联阻力表面值，q 内二次插值
# ======================================================================

def build_interp_weights(N, q_out, v_surf=None):
    """为每个输出 q* 预计算 3 节点二次 Lagrange 权重。

    节点集：q=0（中心，权重处理在 apply 中）、单元中心 q_i=(i+0.5)/N、q=1（表面）。
    返回 (idx, W)：v(q*) ≈ Σ_k W[:,k] · values[idx[:,k]]，idx=-1 表示中心节点，
    idx=N 表示表面节点。
    """
    dq = 1.0 / N
    q_nodes = np.concatenate(([0.0], (np.arange(N) + 0.5) * dq, [1.0]))
    idx = np.zeros((len(q_out), 3), dtype=int)
    W = np.zeros((len(q_out), 3))
    for j, q in enumerate(q_out):
        p = int(np.searchsorted(q_nodes, q))
        s = min(max(p - 1, 0), len(q_nodes) - 3)
        cand = [s, s + 1, s + 2]
        qs = q_nodes[cand]
        w = np.ones(3)
        for a in range(3):
            for b_ in range(3):
                if a != b_:
                    w[a] *= (q - qs[b_]) / (qs[a] - qs[b_])
        idx[j] = [c - 1 for c in cand]   # 节点 0 → -1（中心），节点 N+1 → N（表面）
        W[j] = w
    return idx, W


def apply_interp(values, v_surf, idx, W):
    """values (..., N) 单元中心值；v_surf (...) 表面值。返回 (..., len(idx))。"""
    N = values.shape[-1]
    n_out = idx.shape[0]
    lead = values.shape[:-1]
    out = np.zeros(lead + (n_out,))
    for j in range(n_out):
        acc = np.zeros(lead)
        for k in range(3):
            i = idx[j, k]
            if i == -1:      # 中心：正则性外推 v(0) = 1.5 v_0 − 0.5 v_1
                v = 1.5 * values[..., 0] - 0.5 * values[..., 1]
            elif i == N:     # 表面
                v = v_surf
            else:
                v = values[..., i]
            acc = acc + W[j, k] * v
        out[..., j] = acc
    return out
