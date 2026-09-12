"""A 级（几何反馈）力学驱动内生收缩模型（按 other/力学驱动内生收缩_方程与实施方案.md §3–§10）。

有限应变（非仿射）：材料点 X∈[0,R0]，现时 r(X,t)，λ_r=r_X、λ_θ=r/X、λ_z=1、
J=λ_rλ_θ=2r r_q/R0²（q=X²/R0²，ρ_d=ρ_d0/J）。不强行 r=R√q。
孔隙非负进方程：g=J−s0−w≥0（s0=ρ_d0/ρ_s，w=ρ_d0U/ρ_w），乘子 ζ 互补
（g≥0, ζ≥0, ζg=0；活动集法，绝不先算负再截断）。
毛细储能：p_c=p*(1−S)（**明确标注的低阶检验本构**，非真实保水定律）；
ψ_cap=(J−s0)∫_S^1 p_c ds，p̄_c=(p*/2)(1−S²)。
单松弛黏弹骨架：对数主应变 E=diag(lnλ_r,lnλ_θ,0)，
ψ_el=G∞‖devE‖²+K∞/2(trE−e*)²+G1‖dev(E−Z)‖²+K1/2[tr(E−Z)]²，
Ż=(E−Z)/τ(U,T)（后向欧拉局部消元）。初始平衡 e*=p̄_c(S0)/K∞ ⇒ 初始总应力为零。
总应力 σ_i=H_i/J+p̄_c−ζ；准静态径向平衡弱式（X 向线性元，轴心正则极限）；
R(t)=r(R0,t) 是输出。
传质（A 级最小衔接）：U_t=∂_q(D/r_q² U_q)，湿边界外系数 h_m/r_q(1,t)；
热方程显热近似 a_eff T_t=(1/(r r_q))∂_q(rk/r_q T_q)。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from solver_q1 import R0, H, HM, T0_K, C0

RHO_W = 1000.0


class MechParams:
    """材料/情景参数。压力单位任意（A 级静力绝对尺度不可辨识，只扫/标定比值）。

    ret_model：
    - 'linear'：p_c=p*(1−S)（A 级低阶检验本构，吸力随 S→0 饱和于 p*）；
    - 'power'：p_c=p*(1−S)/S^q（van Genuchten 类幂律保水形式，
      土壤/食品干燥常用：吸力随 S→0 以 S^{-q} 持续增大（0<q<1），
      且毛细能量 ∫p_c dS 有界；p̄_c(0)=p*/[(1−q)(2−q)] 大而有限）。
    """

    def __init__(self, rho_d0, rho_s, Kinf=1.0, Ginf=1.0 / 3, K1=1.0, G1=1.0 / 3,
                 p_star=1.0, tau0=1.0e6, tau_exp=4.0, ret_model="linear", q_pow=0.8,
                 Pi_t0=0.0, k_tur=2.0):
        self.rho_d0 = rho_d0
        self.rho_s = rho_s
        self.Kinf, self.Ginf = Kinf, Ginf
        self.K1, self.G1 = K1, G1
        self.p_star = p_star
        self.tau0 = tau0
        self.tau_exp = tau_exp      # τ(U)=tau0·(C0/U)^tau_exp（情景形式，声明为假设）
        self.ret_model = ret_model
        self.q_pow = q_pow
        self.Pi_t0 = Pi_t0          # 初始膨压（K∞ 单位；取 K∞=1 MPa 时按 MPa 解读）
        self.k_tur = k_tur          # 膨压松弛指数（随失水单调下降）
        self.s0 = rho_d0 / rho_s

    def tau(self, U, T=323.0):
        return self.tau0 * (C0 / np.maximum(np.asarray(U, float), 0.02)) ** self.tau_exp

    def turgor(self, U):
        """细胞膨压 Π_t(U)=Π_t0·(U/C0)^k_tur（植物生理：渗透压维持细胞饱满，
        失水时膨压松弛 → 基体整体压缩；与毛细项各司其职，不重复归因）。"""
        return self.Pi_t0 * (np.asarray(U, float) / C0) ** self.k_tur

    def pc(self, S):
        """保水关系 p_c(S)（吸力，≥0，随 S 增大而降低）。"""
        S = np.maximum(np.asarray(S, float), 1e-12)
        if self.ret_model == "linear":
            return self.p_star * (1.0 - S)
        return self.p_star * (1.0 - S) / S ** self.q_pow

    def pbar_c(self, S):
        """p̄_c = ∂ψ_cap/∂J = f(S)+S·p_c(S)，f(S)=∫_S^1 p_c(s)ds（解析）。"""
        S = np.asarray(S, float)
        if self.ret_model == "linear":
            return 0.5 * self.p_star * (1.0 - S**2)
        q = self.q_pow
        Sc = np.maximum(S, 1e-12)
        f = self.p_star * ((1.0 - Sc ** (1.0 - q)) / (1.0 - q)
                           - (1.0 - Sc ** (2.0 - q)) / (2.0 - q))
        return f + self.p_star * (1.0 - Sc) * Sc ** (1.0 - q)

    @property
    def e_star(self):
        w0 = self.rho_d0 * C0 / RHO_W
        S0 = w0 / (1.0 - self.s0)
        # 初始总应力为零：弹性预应变平衡（毛细吸力 − 膨压），不重复归因
        return (float(self.pbar_c(S0)) - self.Pi_t0) / self.Kinf


def elem_quantities(r, X, U, P):
    """逐单元 λ_r、λ_θ（轴心正则极限）、J、w、g、Pore、S、rq。X 为节点（M+1 个）。"""
    M = len(X) - 1
    dX = np.diff(X)
    lam_r = np.diff(r) / dX
    Xm = 0.5 * (X[:-1] + X[1:])
    rm = 0.5 * (r[:-1] + r[1:])
    lam_t = rm / Xm
    lam_t[0] = lam_r[0]                     # 正则极限 λ_θ(0)=λ_r(0)
    J = lam_r * lam_t
    w = P.rho_d0 * U / RHO_W
    g = J - P.s0 - w
    Pore = J - P.s0
    S = np.clip(w / np.maximum(Pore, 1e-300), 0.0, 1.0)
    return lam_r, lam_t, J, w, g, Pore, S


def solve_mechanics(r_guess, Z_old, U, T, dt, X, P, active0=None, tol=1e-10):
    """受孔隙非负约束的准静态径向平衡（活动集 + FD-Newton；局部消元 Z）。

    未知量：r_1..r_M（r_0=0 Dirichlet）+ 活动接触单元的 ζ。
    返回 dict(r, Z, zeta, g, S, J, sigma_rr_elem, sigma_tt_elem, active)。
    """
    M = len(X) - 1
    dX = np.diff(X)
    e_ = P.e_star
    r = r_guess.copy()
    active = set() if active0 is None else set(active0)
    Z_old = np.asarray(Z_old, float).reshape(M, 3)

    def eval_state(ry, zeta_map):
        lam_r, lam_t, J, w, g, Pore, S = elem_quantities(ry, X, U, P)
        E = np.column_stack([np.log(np.maximum(lam_r, 1e-300)),
                             np.log(np.maximum(lam_t, 1e-300)), np.zeros(M)])
        tau = P.tau(U, T)
        al = dt / np.maximum(tau, 1e-300)
        Z = (Z_old + al[:, None] * E) / (1.0 + al[:, None])
        trE = E.sum(1)
        devE = E - (trE / 3.0)[:, None]
        EmZ = E - Z
        trEmZ = EmZ.sum(1)
        devEmZ = EmZ - (trEmZ / 3.0)[:, None]
        H = (2.0 * P.Ginf * devE + P.Kinf * (trE - e_)[:, None]
             + 2.0 * P.G1 * devEmZ + P.K1 * trEmZ[:, None])
        zeta = np.array([zeta_map.get(e, 0.0) for e in range(M)])
        sig = H / J[:, None] + P.pbar_c(S)[:, None] - P.turgor(U)[:, None] - zeta[:, None]
        return lam_r, lam_t, J, g, S, Z, sig

    def residual(y, active_list):
        ry = np.empty(M + 1)
        ry[0] = 0.0
        ry[1:] = y[:M]
        zmap = {e: y[M + k] for k, e in enumerate(active_list)}
        lam_r, lam_t, J, g, S, Z, sig = eval_state(ry, zmap)
        Pr = J * sig[:, 0] / lam_r          # 名义应力 P_r=Jσ_rr/λ_r
        Pt = J * sig[:, 1] / lam_t          # P_θ=Jσ_θθ/λ_θ
        F = np.zeros(M + len(active_list))
        # 内部节点 j=1..M-1：右单元 e=j（右端）+ 左单元 e=j+1（左端）
        for j in range(1, M):
            eR, eL = j - 1, j
            F[j - 1] = (Pr[eR] * 0.5 * (X[eR] + X[eR + 1])
                        - Pr[eL] * 0.5 * (X[eL] + X[eL + 1])
                        + 0.5 * dX[eR] * Pt[eR] + 0.5 * dX[eL] * Pt[eL])
        # 表面节点 M（自然边界 σ_rr=0 ⇒ 仅左单元贡献）
        F[M - 1] = Pr[M - 1] * 0.5 * (X[M - 1] + X[M]) + 0.5 * dX[M - 1] * Pt[M - 1]
        # 接触单元：g=0
        for k, e in enumerate(active_list):
            F[M + k] = g[e]
        return F

    for _as in range(60):
        active_list = sorted(active)
        y = np.concatenate([r[1:], np.zeros(len(active_list))])
        for _nt in range(60):
            F = residual(y, active_list)
            if np.linalg.norm(F, np.inf) < tol:
                break
            n = len(y)
            Jf = np.zeros((len(F), n))
            for j in range(n):
                dy = 1e-7 * max(1.0, abs(y[j]))
                yp = y.copy(); yp[j] += dy
                ym = y.copy(); ym[j] -= dy
                Jf[:, j] = (residual(yp, active_list) - residual(ym, active_list)) / (2 * dy)
            try:
                delta = np.linalg.solve(Jf, -F)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(Jf, -F, rcond=None)[0]
            for damp in (1.0, 0.5, 0.25, 0.1, 0.02):
                yt = y + damp * delta
                rt = np.empty(M + 1); rt[0] = 0.0; rt[1:] = yt[:M]
                _, _, Jt, *_ = elem_quantities(rt, X, U, P)
                if np.all(Jt > 1e-8):
                    y = yt
                    break
            else:
                y = y + 0.005 * delta
        r[1:] = y[:M]
        zmap = {e: y[M + k] for k, e in enumerate(active_list)}
        ry = np.empty(M + 1); ry[0] = 0.0; ry[1:] = y[:M]
        lam_r, lam_t, J, g, S, Z, sig = eval_state(ry, zmap)
        changed = False
        for e in range(M):
            if e not in active and g[e] < -1e-9:
                active.add(e); changed = True
            elif e in active and zmap.get(e, 0.0) < -1e-9:
                active.discard(e); changed = True
        if not changed:
            break
    zeta = np.array([zmap.get(e, 0.0) for e in range(M)])
    return dict(r=ry, Z=Z, zeta=zeta, g=g, S=S, J=J,
                sig_r=sig[:, 0], sig_t=sig[:, 1], active=sorted(active), n_as=_as)


# ======================================================================
# 传质 + 显热（非仿射几何系数，BE + Picard）
# ======================================================================

def mass_heat_step(U, T, dt, Ta, Cenv, r, X, Jc, P, Dof, kof, acp):
    """一个 BE 步。U_t=∂_q(D/r_q²U_q)；a_eff T_t=(1/(r r_q))∂_q(rk/r_q T_q)。

    湿表面：内阻 rq_s²Δq/(2D_s) 与外阻 rq_s/h_m 串联（外系数 h_m/r_q(1)）；
    热表面：内阻 rq_sΔq/(2k_s r_s) 与外阻 1/(h r_s) 串联。
    返回 (Unew, Tnew, Us, Fm, Fh)。
    """
    M = len(X) - 1
    q = (X / R0) ** 2
    dqe = np.diff(q)                       # 单元宽 Δq_e（非均匀）
    rs = r[-1]
    rq_s = (r[-1] - r[-2]) / dqe[-1] if M > 1 else (r[-1] / q[-1])
    # 面量（面 j=1..M-1）：中心距 dc、跨面差商 rqf
    dc = 0.5 * (dqe[:-1] + dqe[1:])
    rqf = (r[2:] - r[:-2]) / (q[2:] - q[:-2])
    Un, Tn = U.copy(), T.copy()
    for _ in range(30):
        # ---- 湿 ----
        Df = np.array([float(Dof(0.5 * (Un[j - 1] + Un[j]), 0.5 * (Tn[j - 1] + Tn[j])))
                       for j in range(1, M)])
        bR = Df * dt / (rqf ** 2 * dc * dqe[:-1])     # 行 e 右面系数
        bL = Df * dt / (rqf ** 2 * dc * dqe[1:])      # 行 e 左面系数
        Us = Un[-1]
        for _s in range(40):
            Ds = float(Dof(0.5 * (Us + Un[-1]), Tn[-1]))   # 与参考求解器一致：半单元中点 D
            R_int = rq_s ** 2 * (dqe[-1] / 2.0) / Ds
            R_ext = rq_s / HM
            Us2 = (Un[-1] / R_int + Cenv / R_ext) / (1.0 / R_int + 1.0 / R_ext)
            if abs(Us2 - Us) < 1e-13:
                Us = Us2
                break
            Us = Us2
        Ds = float(Dof(0.5 * (Us + Un[-1]), Tn[-1]))
        gs = dt / ((rq_s ** 2 * (dqe[-1] / 2.0) / Ds + rq_s / HM) * dqe[-1])
        diag = 1.0 + np.concatenate(([0.0], bL)) + np.concatenate((bR, [0.0]))
        diag[-1] += gs
        rhs = U.copy()
        rhs[-1] += gs * Cenv
        A = np.zeros((3, M))
        A[0, 1:] = -bR
        A[1, :] = diag
        A[2, :-1] = -bL
        Un2 = solve_banded((1, 1), A, rhs)
        # ---- 热（q 形式通量 Φ=(k/r_q²)T_q，与湿分同构；单元权重 wc=R0²J_e/2）----
        a = acp(Un2)
        kf = np.array([float(kof(0.5 * (Un2[j - 1] + Un2[j]))) for j in range(1, M)])
        wc = (R0 ** 2 / 2.0) * Jc
        chR = kf * dt / (rqf ** 2 * dc * dqe[:-1] * wc[:-1] * a[:-1])
        chL = kf * dt / (rqf ** 2 * dc * dqe[1:] * wc[1:] * a[1:])
        ks = float(kof(0.5 * (Us + Un2[-1])))
        Rh = rq_s ** 2 * (dqe[-1] / 2.0) / ks + rq_s / H   # 内阻 rq²Δq/(2k) + 外阻 rq/h
        chs = dt / (Rh * wc[-1] * a[-1])
        diagT = 1.0 + np.concatenate(([0.0], chL)) + np.concatenate((chR, [0.0]))
        diagT[-1] += chs
        rhsT = T.copy()
        rhsT[-1] += chs * Ta
        AT = np.zeros((3, M))
        AT[0, 1:] = -chR
        AT[1, :] = diagT
        AT[2, :-1] = -chL
        Tn2 = solve_banded((1, 1), AT, rhsT)
        dm = max(float(np.max(np.abs(Un2 - Un))), float(np.max(np.abs(Tn2 - Tn))))
        Un, Tn = Un2, Tn2
        if dm < 1e-10:
            break
    Fm = HM * (Cenv - Us) / rq_s
    Fh = (Ta - Tn[-1]) / Rh
    return Un, Tn, Us, Fm, Fh


def volume_check(r, X, U, P):
    """全局体积核验式：R²/R0² = s0 + (ρ_d0/ρ_w)Ū + ∫g dq（只核验一致性）。"""
    q = (X / R0) ** 2
    dq = np.diff(q)
    _, _, J, w, g, *_ = elem_quantities(r, X, U, P)
    Um = float(np.sum(U * dq))
    rhs = P.s0 + P.rho_d0 / RHO_W * Um + float(np.sum(g * dq))
    return (r[-1] / R0) ** 2, rhs
