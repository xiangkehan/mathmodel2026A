"""问题 1 热场（以及常 D 退化湿分场）的解析解：圆柱 Robin 特征展开 + Duhamel 卷积。

方程：T_t = diff·(1/r)∂_r(r ∂_r T)，r∈[0,R]；中心正则；
表面 −coef·T_r = h_ext·(T_s − g_env(t))（Robin，Bi = h_ext·R/coef）。

特征值：λ J₁(λ) = Bi·J₀(λ)，特征函数 J₀(λ_n r/R)，μ_n = diff·λ_n²/R²。
令 g(t) = g_env(t) − T(0)，则

    T(r,t) = T(0) + Σ_n c_n(t)·J₀(λ_n r/R)，
    c_n(t) = K_n·J_n(t)，  K_n = 2 μ_n Bi / (J₀(λ_n)(λ_n²+Bi²))，
    J_n(t) = ∫₀^t e^{−μ_n(t−s)} g(s) ds。

阶跃响应系数：A_n = 2Bi²/(λ_n J₁(λ_n)(Bi²+λ_n²))（中心列），
表面列 A_n J₀(λ_n) = 2Bi/(Bi²+λ_n²)。自检：ΣA_n = 1、ΣA_nJ₀(λ_n) = 1。

g 分段线性时 J_n 有精确递推（每段一步，无时间截断误差），
与求解器的精确时间积分对齐 ⇒ V1 对拍隔离纯空间误差。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv


def robin_eigs(Bi: float, n_modes: int) -> np.ndarray:
    """λ J₁(λ) − Bi·J₀(λ) = 0 的前 n_modes 个正根（稠密扫描 + brentq）。"""
    def f(x):
        return x * jv(1, x) - Bi * jv(0, x)

    xmax = (n_modes + 1) * np.pi + 1.0
    xs = np.linspace(1e-9, xmax, int(xmax / 0.002) + 1)
    fs = f(xs)
    roots = []
    for i in range(len(xs) - 1):
        if fs[i] == 0.0:
            roots.append(xs[i])
        elif fs[i] * fs[i + 1] < 0:
            roots.append(brentq(f, xs[i], xs[i + 1], xtol=1e-15, rtol=1e-15))
        if len(roots) >= n_modes:
            break
    roots = np.array(roots[:n_modes])
    assert len(roots) == n_modes, f"只找到 {len(roots)} 个根"
    d = np.diff(roots)
    assert np.all((d > 2.5) & (d < 3.7)), "特征根间距异常，可能漏根"
    return roots


class RobinCylinder:
    """圆柱 Robin 问题的特征展开解（Duhamel 卷积，分段线性驱动精确积分）。"""

    def __init__(self, Bi: float, diff: float, radius: float, n_modes: int = 300):
        self.Bi, self.diff, self.R = Bi, diff, radius
        self.lam = robin_eigs(Bi, n_modes)
        self.mu = diff * self.lam**2 / radius**2
        self.J0l = jv(0, self.lam)
        J1l = jv(1, self.lam)
        self.A = 2 * Bi**2 / (self.lam * J1l * (self.lam**2 + Bi**2))
        self.Kn = 2 * self.mu * Bi / (self.J0l * (self.lam**2 + Bi**2))
        # 自检：阶跃响应趋于 1（中心列与表面列）
        self.sum_center = float(self.A.sum())
        self.sum_surface = float((self.A * self.J0l).sum())
        self.mean_mode = 2.0 * jv(1, self.lam) / self.lam   # J₀(λ√q) 的体积平均

    def _advance(self, J, dt, g0, g1):
        e = np.expm1(-self.mu * dt)
        E1 = -e / self.mu                       # (1−e^{−μdt})/μ
        E2 = (e + self.mu * dt) / self.mu**2    # (e^{−μdt}−1+μdt)/μ²
        m = (g1 - g0) / dt
        return J * np.exp(-self.mu * dt) + g0 * E1 + m * E2

    def _state(self, env_t, env_g, times):
        """把模态卷积状态 J_n 精确推进到各时刻；返回 (g(times), J(times))。"""
        times = np.asarray(times, dtype=float)
        J = np.zeros(len(self.lam))
        t_cur, seg = float(env_t[0]), 0
        g_out = np.zeros(len(times))
        J_out = np.zeros((len(times), len(self.lam)))

        def g_at(t):                            # 段内线性（段界已对齐，无跨段）
            return env_g[seg] + (env_g[seg + 1] - env_g[seg]) * \
                (t - env_t[seg]) / (env_t[seg + 1] - env_t[seg])

        for j, tau in enumerate(times):
            while t_cur < tau - 1e-12:
                t_nxt = min(tau, float(env_t[seg + 1]))
                J = self._advance(J, t_nxt - t_cur, g_at(t_cur), g_at(t_nxt))
                t_cur = t_nxt
                if t_nxt >= float(env_t[seg + 1]) - 1e-12 and seg + 1 < len(env_t) - 1:
                    seg += 1
            g_out[j] = g_at(tau)
            J_out[j] = J
        return g_out, J_out

    def eval(self, r, times, env_t, env_g, offset):
        """r（m，数组）、times（s，升序数组）处的解。

        env_t/env_g：驱动 g(t) = 环境 − 初值 的分段线性采样点（须含 t=0）。
        采用稳态减法加速级数收敛：
            T = offset + g(t) − Σ_n (K_n/μ_n)·J_n'(t)·J₀(λ_n r/R)，J_n' = g − μ_n J_n，
        其中 Σ(K_n/μ_n)J₀ ≡ 1（阶跃稳态为均匀 g）为精确恒等式，
        尾巴项衰减从 λ⁻²（非交错）提到 λ⁻⁷⁄²（交错）。
        """
        r = np.atleast_1d(np.asarray(r, dtype=float))
        basis = jv(0, np.outer(self.lam, r / self.R))       # (M, len(r))
        g_t, J_t = self._state(np.asarray(env_t, float), np.asarray(env_g, float), times)
        coef = self.Kn / self.mu
        out = np.zeros((len(times), len(r)))
        for j in range(len(times)):
            Jp = g_t[j] - self.mu * J_t[j]
            out[j] = offset + g_t[j] - (coef * Jp) @ basis
        return out

    def mean(self, times, env_t, env_g, offset):
        """体积平均（∫₀¹ T(q)dq）随时间（同一稳态减法形式）。"""
        g_t, J_t = self._state(np.asarray(env_t, float), np.asarray(env_g, float), times)
        coef = self.Kn / self.mu * self.mean_mode
        out = np.zeros(len(times))
        for j in range(len(times)):
            Jp = g_t[j] - self.mu * J_t[j]
            out[j] = offset + g_t[j] - float(coef @ Jp)
        return out
