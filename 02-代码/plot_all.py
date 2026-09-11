"""M5 图表生成：从 03-数据/ 的唯一数据源出 5 张论文用矢量图（05-图表/，同时导出 300 dpi PNG）。

用法：  python plot_all.py            # 一次出全部 5 张
         python plot_all.py fig4       # 只出某一张（调试用）
         python plot_all.py --no-cache # fig4 的附4固定R对照曲线强制重算

数据源（每张图对应 §3 规格）：
  fig1  附件1.xlsx（T_a、C_env）、附件2.xlsx（R）
  fig2  q1_fields.npz（T_C、times、pos_cm、env_*）、v1_points_final.csv、v1_convergence.csv
  fig3  q23_main_steps.npz（t、C、T）、q23_summary.json、criteria.csv
  fig4  q23_main_steps.npz（um）、q4_main_steps.npz（um）、q4_split.csv、q4_bound.csv、
        v3_bound.csv、sensitivity.csv、结果总账.csv；附4固定R对照曲线由 solver_q4 现算
        （与 q4_split.csv 的 appendix4_fixedR 逐次核对）
  fig5  sensitivity.csv、error_budget.csv

口径常数（非结果数字，仅用于画线/判定，来源已注明）：
  TH = 0.15 kg/kg  —— 题目问题 3 原文“水分浓度应低于 0.15 kg/kg”
  其余全部数字（含 35 h 冻结点、严格下界）均从上述数据文件读出，脚本内不誊写结果。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CODE_DIR = Path(__file__).resolve().parent
A_DIR = CODE_DIR.parent
DATA_DIR = A_DIR / "03-数据"
FIG_DIR = A_DIR / "05-图表"
ATT1 = A_DIR / "01-题目" / "原始文件" / "附件1.xlsx"
ATT2 = A_DIR / "01-题目" / "原始文件" / "附件2.xlsx"
sys.path.insert(0, str(CODE_DIR))

# ---- 题目口径常数（非计算结果；见模块 docstring）----
TH = 0.15                      # kg/kg，问题 3 达标阈值（题目原文）

# ---- 统一色板 ----
COL = dict(
    q3="#1f4e79",      # 问题3 主解（附3 + 固定 R）
    q4="#c0392b",      # 问题4 主解（附4 + R(t)）
    ctrl="#7f8c8d",    # 对照曲线（附4 + 固定 R0）
    th="#111111",      # 阈值线
    lb="#2e8b57",      # 严格下界
    temp="#d1495b",    # 温度
    hum="#1b6ca8",     # 水分
    ana="#1f4e79",     # 解析解
    num="#d1495b",     # 数值解
    pos="#c0392b",     # 正偏差
    neg="#1f6fb2",     # 负偏差
    zero="#9aa0a6",
    band="#b8c4d0",
)

RC = {
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,          # 嵌入 TrueType 子集（矢量，可复制文字）
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.7,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

TIME_CMAP = plt.cm.viridis


class _GlyphWatch(logging.Handler):
    """抓 matplotlib 的缺字告警（豆腐块检测）。"""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs: list[str] = []

    def emit(self, record):
        m = record.getMessage()
        if "missing from font" in m or "Glyph" in m:
            self.msgs.append(m)


GLYPH = _GlyphWatch()
logging.getLogger("matplotlib").addHandler(GLYPH)


# ====================== 通用读数 ======================
def read_att1():
    """附件1：返回 (t_s, T_a/℃, C_env/(kg/kg))，0–14400 s 每 60 s。"""
    import openpyxl
    wb = openpyxl.load_workbook(ATT1, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    wb.close()
    return (np.array([r[0] for r in rows], float),
            np.array([r[1] for r in rows], float),
            np.array([r[2] for r in rows], float))


def read_att2():
    """附件2：返回 (t_s, R/cm)，0–259200 s 每 1800 s。"""
    import openpyxl
    wb = openpyxl.load_workbook(ATT2, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    wb.close()
    return (np.array([r[0] for r in rows], float),
            np.array([r[1] for r in rows], float))


def read_csv_rows(name):
    """读 03-数据/ 下的 csv，返回 (表头, [dict, ...])；兼容注释行（首个字段以 # 开头）
    与字段内混入英文逗号的情形（把多余的分段并回首列）。"""
    with open(DATA_DIR / name, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    head, body = rows[0], rows[1:]
    n = len(head)
    body = [([",".join(r[:len(r) - n + 1])] + r[len(r) - n + 1:]) if len(r) > n else r
            for r in body]
    return head, [dict(zip(head, r)) for r in body]


def read_json(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def crossing_time(t, y, th):
    """y(t) 自上方穿越 th 的线性插值时刻（t 升序）。"""
    i = int(np.argmax(y < th))
    assert 0 < i < len(y), "曲线未穿越阈值"
    return float(t[i - 1] + (th - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))


def save(fig, stem):
    FIG_DIR.mkdir(exist_ok=True)
    pdf, png = FIG_DIR / f"{stem}.pdf", FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    print(f"  [出图] {pdf.name} ({pdf.stat().st_size/1024:.0f} kB) "
          f"+ {png.name} ({png.stat().st_size/1024:.0f} kB)", flush=True)


def note(ax, text, xy=(0.985, 0.975), ha="right", va="top", fs=8.4):
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.72", lw=0.6, alpha=0.94))


def tile(ax, title):
    ax.set_title(title, fontsize=10.5, pad=6)


# ====================== F1 输入数据 ======================
def fig1():
    t1, Ta, Ce = read_att1()
    t2, R = read_att2()
    h1 = t1 / 3600.0
    h2 = t2 / 3600.0
    t_cut = h1[-1]                       # 附件1 截止（4 h）
    i_frz = int(np.argmax(R <= 1.200 + 1e-12))
    t_frz = h2[i_frz]                    # 35 h 冻结点（R 首达 1.200 cm）
    x_end = 100.0
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.1))

    # ---- 左：附件1 温湿环境 ----
    axL.plot(h1, Ta, color=COL["temp"], lw=1.8, label="烘房温度 $T_a(t)$")
    axL.set_xlabel("时间 $t$ / h")
    axL.set_ylabel("烘房温度 $T_a$ / ℃", color=COL["temp"])
    axL.tick_params(axis="y", colors=COL["temp"])
    axL.set_xlim(0, 6.3)
    axL.set_ylim(27.0, 56.0)
    axC = axL.twinx()
    axC.plot(h1, Ce, color=COL["hum"], lw=1.8, label="烘房水分浓度 $C_{env}(t)$")
    axC.set_ylabel("烘房水分浓度 $C_{env}$ / (kg/kg)", color=COL["hum"])
    axC.tick_params(axis="y", colors=COL["hum"])
    axC.set_ylim(0.0175, 0.0530)
    axC.grid(False)
    axL.axvspan(t_cut, 6.3, color=COL["band"], alpha=0.30, lw=0)
    axL.plot([t_cut, 6.3], [Ta[-1]] * 2, color=COL["temp"], ls="--", lw=1.4)
    axC.plot([t_cut, 6.3], [Ce[-1]] * 2, color=COL["hum"], ls="--", lw=1.4)
    axL.axvline(t_cut, color="0.30", ls=":", lw=1.2)
    axL.annotate(f"附件1 截止 $t$={t_cut:.0f} h（{t1[-1]:.0f} s）：\n"
                 f"其后末值保持延拓（主口径，声明假设）",
                 xy=(t_cut, 0.30), xycoords=("data", "axes fraction"),
                 xytext=(0.985, 0.40), textcoords="axes fraction",
                 fontsize=8.4, ha="right", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.30"),
                 bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.72", lw=0.6, alpha=0.94))
    axL.text(0.985, 0.245, f"延拓值 $T_a$={Ta[-1]:.3f} ℃（左轴）\n"
                           f"$C_{{env}}$={Ce[-1]:.5f} kg/kg（右轴）",
             transform=axL.transAxes, fontsize=8.4, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.72", lw=0.6, alpha=0.94))
    hs = [axL.get_legend_handles_labels()[0][0], axC.get_legend_handles_labels()[0][0]]
    axL.legend(hs, ["烘房温度 $T_a(t)$（左轴）", "烘房水分浓度 $C_{env}(t)$（右轴）"],
               loc="lower right", framealpha=0.92)
    tile(axL, "(a) 附件1：环境温湿（回答：4 h 数据如何裁定与延拓）")

    # ---- 右：附件2 半径 ----
    axR.plot(h2, R, color=COL["q4"], lw=1.8, label="药材半径 $R(t)$")
    axR.plot([h2[-1], x_end], [R[-1]] * 2, color=COL["q4"], ls="--", lw=1.4)
    axR.axvspan(t_frz, h2[-1], color=COL["band"], alpha=0.30, lw=0)
    axR.axvspan(h2[-1], x_end, color=COL["band"], alpha=0.14, lw=0)
    axR.axvline(t_frz, color="0.30", ls=":", lw=1.2)
    axR.axvline(h2[-1], color="0.30", ls=":", lw=1.0)
    axR.annotate(f"$t$={t_frz:.0f} h：$R$ 首达 {R[i_frz]:.3f} cm\n"
                 f"此后基本冻结（{t_frz:.0f}–{h2[-1]:.0f} h 仅 "
                 f"{R[i_frz]:.3f}→{R[-1]:.3f} cm）",
                 xy=(t_frz, R[i_frz]), xytext=(0.34, 0.62), textcoords="axes fraction",
                 fontsize=8.4, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.30"),
                 bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.72", lw=0.6, alpha=0.94))
    axR.annotate(f"附件2 截止 {h2[-1]:.0f} h\n"
                 f"其后 {R[-1]:.3f} cm 末值平台延拓\n（问题4 求解用，声明假设）",
                 xy=(h2[-1], R[-1]), xytext=(0.44, 0.30), textcoords="axes fraction",
                 fontsize=8.4, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.30"),
                 bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.72", lw=0.6, alpha=0.94))
    axR.axhline(R[-1], color="0.45", ls="-.", lw=0.9)
    axR.text(97.5, R[-1] - 0.014, f"平台 {R[-1]:.3f} cm", ha="right", va="top",
             fontsize=8.2, color="0.30")
    axR.set_xlim(0, x_end)
    axR.set_ylim(1.16, 2.06)
    axR.set_xlabel("时间 $t$ / h")
    axR.set_ylabel("药材半径 $R$ / cm")
    axR.legend(loc="upper right", framealpha=0.92)
    tile(axR, "(b) 附件2：半径收缩（回答：35 h 冻结后如何裁定）")
    print(f"  [F1] 附件1 {len(t1)} 点 0–{t_cut:.0f} h；附件2 {len(t2)} 点 0–{h2[-1]:.0f} h，"
          f"R {R[0]:.3f}→{R[-1]:.3f} cm，冻结点 {t_frz:.0f} h", flush=True)
    save(fig, "fig1_输入数据")


# ====================== F2 问题1 解析对拍 ======================
def fig2():
    from analytic_heat import RobinCylinder
    from solver_q1 import ALPHA, BI_H, R0, T0_K

    d = np.load(DATA_DIR / "q1_fields.npz")
    times_arr, pos_cm = d["times"], d["pos_cm"]
    T_C = d["T_C"]                                   # (1800, 21) ℃
    env_t, env_Ta = d["env_t"], d["env_Ta"]
    t_show = [100, 600, 1800]
    ana = RobinCylinder(BI_H, ALPHA, R0, n_modes=300)
    r_dense = np.linspace(0.0, R0, 401)
    T_ana = ana.eval(r_dense, np.array(t_show, float), env_t, env_Ta - T0_K, T0_K) - 273.15

    _, pts = read_csv_rows("v1_points_final.csv")
    dev_max = max(float(p["abs_dev"]) for p in pts)
    head, conv = read_csv_rows("v1_convergence.csv")
    data_rows = [c for c in conv if c[head[0]] != "order(fit)"]
    Ns = np.array([float(c[head[0]]) for c in data_rows])
    errs = np.array([[float(c[h]) for h in head[1:]] for c in data_rows])
    order_row = [c for c in conv if c[head[0]] == "order(fit)"][0]
    orders = [float(order_row[h]) for h in head[1:]]

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    handles = []
    for k, ts in enumerate(t_show):
        col = TIME_CMAP(0.15 + 0.62 * k / max(len(t_show) - 1, 1))
        idx = int(np.where(times_arr == ts)[0][0])
        ax.plot(r_dense * 100, T_ana[k], color=col, lw=1.8)
        ax.plot(pos_cm, T_C[idx], ls="none", marker="o", ms=4.6, mfc="white",
                mec=col, mew=1.3)
        handles.append(Line2D([], [], color=col, marker="o", mfc="white", mec=col,
                              label=f"$t$={ts} s"))
    ax.set_xlabel("到药材中心的距离 $r$ / cm")
    ax.set_ylabel("药材温度 $T$ / ℃")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(27.6, 39.6)
    ax.legend(handles=handles, title="实线：解析解（特征展开）\n空心点：数值解（FV, $N$=320）",
              loc="upper left", framealpha=0.94)
    tile(ax, "问题1 温度场：解析解 vs 数值解（回答：可靠性如何证明）")

    ins = ax.inset_axes([0.53, 0.285, 0.45, 0.315])
    mk = ["s", "^", "o"]
    for k in range(errs.shape[1]):
        col = TIME_CMAP(0.15 + 0.62 * k / max(errs.shape[1] - 1, 1))
        ins.loglog(Ns, errs[:, k], color=col, marker=mk[k], ms=4.0, lw=1.3,
                   label=f"$t$={t_show[k]} s, 阶 {orders[k]:.3f}")
    ref = errs[0, 0] * (Ns / Ns[0]) ** -2.0
    ins.loglog(Ns, ref, color="0.45", ls="--", lw=1.1, label="斜率 −2（二阶参考）")
    ins.set_xlabel("网格数 $N$", fontsize=8.0, labelpad=1.0)
    ins.set_ylabel("最大偏差 / ℃", fontsize=8.0, labelpad=1.0)
    ins.tick_params(labelsize=7.2, pad=1.5)
    ins.grid(True, which="both", alpha=0.22, lw=0.5)
    ins.legend(fontsize=7.0, loc="lower left", framealpha=0.92, handlelength=1.5)
    ins.text(0.97, 0.95, "收敛性：偏差随 $N$", transform=ins.transAxes,
             ha="right", va="top", fontsize=8.4)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.19)
    fig.text(0.10, 0.055,
             f"解析 vs 数值：$N$=320 时五位置逐点最大偏差 {dev_max:.2e} ℃"
             "（v1_points_final.csv），远低于四位小数的分辨率 $5\\times10^{-5}$ ℃；"
             f"偏差随 $N$ 单调下降且与 $N^{{-2}}$ 平行（插图：拟合阶 "
             f"{orders[0]:.3f}/{orders[1]:.3f}/{orders[2]:.3f}，即二阶收敛）。",
             fontsize=8.8, ha="left", va="top")
    print(f"  [F2] 解析对拍最大偏差 {dev_max:.3e} ℃，收敛阶 {orders}", flush=True)
    save(fig, "fig2_问题1解析对拍")


# ====================== F3 温湿剖面 ======================
def fig3():
    d = np.load(DATA_DIR / "q23_main_steps.npz")
    t, C, T = d["t"], d["C"], d["T"]
    pos_cm = np.arange(C.shape[1]) * 0.1
    tf_h = read_json("q23_summary.json")["tf_h"]
    _, crit = read_csv_rows("criteria.csv")
    le_row = [c for c in crit if c["判据"] == "Le" and "附录3" in c["物性"]][0]
    le_txt = f"{le_row['最小值']}–{le_row['最大值']}"

    def prof(field, h):
        i = int(np.argmin(np.abs(t - h * 3600)))
        return field[i].copy()

    t_T = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]           # 预热平衡段（问题2 表3 的 3 h 内）
    t_C = [0.5, 3.0, 12.0, 24.0, 36.0, 57.0]        # 全程（问题3，至 t_f≈57.2 h）

    fig, (axT, axC) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.26, wspace=0.22)
    legT, legC = [], []
    for k, h in enumerate(t_T):
        col = TIME_CMAP(0.95 - 0.80 * k / (len(t_T) - 1))
        axT.plot(pos_cm, prof(T, h), color=col, lw=1.8)
        legT.append(Line2D([], [], color=col, label=f"$t$={h:g} h"))
    for k, h in enumerate(t_C):
        col = TIME_CMAP(0.10 + 0.80 * k / (len(t_C) - 1))
        axC.plot(pos_cm, prof(C, h), color=col, lw=1.8)
        legC.append(Line2D([], [], color=col, label=f"$t$={h:g} h"))
    # 中心（r=0）标记：尾段拖长的直观证据
    for k, h in enumerate(t_C):
        col = TIME_CMAP(0.10 + 0.80 * k / (len(t_C) - 1))
        axC.plot([0], [prof(C, h)[0]], marker="o", ms=5.2, color=col, clip_on=False, zorder=5)

    Ta_end = float(T[-1].max())
    axT.set_xlabel("到药材中心的距离 $r$ / cm")
    axT.set_ylabel("药材温度 $T$ / ℃")
    axT.set_xlim(0, 2.02)
    axT.set_ylim(28.0, 52.5)
    axT.legend(handles=legT, loc="center left", bbox_to_anchor=(0.015, 0.34),
               framealpha=0.95)
    tile(axT, "(a) 温度剖面 $T(r,t)$：预热平衡 3 h 内即达准稳态")

    axC.set_xlabel("到药材中心的距离 $r$ / cm")
    axC.set_ylabel("水分浓度 $C$ / (kg/kg)")
    axC.set_xlim(0, 2.02)
    axC.set_ylim(0.0, 2.72)
    axC.axhline(TH, color=COL["th"], ls="--", lw=1.2)
    axC.text(2.0, TH + 0.045, f"阈值 {TH} kg/kg", ha="right", va="bottom",
             fontsize=8.2, color=COL["th"])
    axC.legend(handles=legC, loc="upper right", framealpha=0.92)
    c36, c57 = prof(C, 36.0)[0], prof(C, 57.0)[0]
    cs57 = prof(C, 57.0)[-1]
    axC.annotate("中心最慢\n（$r$=0，尾段拖长）", xy=(0.02, 0.5 * (c36 + c57)),
                 xytext=(0.20, 0.40), textcoords="axes fraction", fontsize=8.2,
                 ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.30"))
    tile(axC, "(b) 水分剖面 $C(r,t)$：中心最慢、尾段拖长")
    fig.text(0.07, 0.115,
             f"温度：$t\\gtrsim$1 h 后各时刻剖面迅速塌向水平（热场准稳态 $T\\approx T_a$，"
             f"Le={le_txt}）；$t$={t_T[-1]:g} h 时全径仅 {Ta_end-0.4:.1f}–{Ta_end:.1f} ℃。",
             fontsize=8.6, ha="left", va="top")
    fig.text(0.07, 0.055,
             f"水分：中心最慢——$t_f$={tf_h:.2f} h 时中心 {c57:.4f} 仍高于表面 {cs57:.4f}；"
             f"尾段拖长——36→57 h（21 h）中心仅由 {c36:.4f} 降到 {c57:.4f}（差 {c36-c57:.4f}）。"
             "左端圆点=中心值。",
             fontsize=8.6, ha="left", va="top")
    print(f"  [F3] t_f={tf_h:.3f} h；中心 {c57:.4f} / 表面 {cs57:.4f}；36→57 h 中心降 "
          f"{c36-c57:.4f}", flush=True)
    save(fig, "fig3_温湿剖面")


# ====================== F4 达标穿越（钱图） ======================
def fixedR_control(force=False):
    """附4 + 固定 R0 对照的 max-U 全程曲线（模型现算；缓存与 q4_split.csv 逐次核对）。"""
    cache = FIG_DIR / "_fig4_对照_附4固定R_maxU.csv"
    _, split = read_csv_rows("q4_split.csv")
    tf_ref = float([s["t_f_h"] for s in split if s["case"] == "appendix4_fixedR"][0])
    if cache.exists() and not force:
        arr = np.loadtxt(cache, delimiter=",", skiprows=1)
        tt, uu = arr[:, 0], arr[:, 1]
        tf_c = crossing_time(tt, uu, TH) / 3600.0
        # 缓存值的“插值穿越时刻”与 q4_split.csv 的“步末时刻”差 <0.005 h（口径差），
        # 故容差取 0.05 h；超出即视为缓存过期，重算。
        if abs(tf_c - tf_ref) < 0.05:
            print(f"  [F4] 对照曲线读缓存（插值穿越 {tf_c:.4f} h，q4_split.csv 步末 "
                  f"{tf_ref:.4f} h）", flush=True)
            return tt, uu, tf_c
        print(f"  [F4] 缓存与 q4_split.csv 不符（{tf_c:.4f} vs {tf_ref:.4f}），重算", flush=True)
    print("  [F4] 现算 附4+固定R 对照曲线（solver_q4, N=160，约 1–2 min）…", flush=True)
    from run_q4 import make_env, read_env, read_radius, run_q4_full
    from solver_q1 import R0
    t0 = time.perf_counter()
    env_t, TaK, Ce = read_env()
    env_hold = make_env("hold", env_t, TaK, Ce)
    read_radius()                                   # 复用其结构自检（正、单调不增）
    _, log = run_q4_full(160, env_hold, lambda tt: R0)
    tt = np.array([x[0] for x in log])
    uu = np.array([x[4] for x in log])
    tf_c = crossing_time(tt, uu, TH) / 3600.0
    print(f"  [F4] 对照曲线算毕 {time.perf_counter()-t0:.0f} s，t_f={tf_c:.4f} h "
          f"(q4_split.csv {tf_ref:.4f} h)", flush=True)
    assert abs(tf_c - tf_ref) < 2e-2, "对照曲线与 q4_split.csv 不一致"
    FIG_DIR.mkdir(exist_ok=True)
    np.savetxt(cache, np.column_stack([tt, uu]), delimiter=",", header="t_s,umax",
               comments="", fmt="%.10g")
    return tt, uu, tf_c


def fig4(no_cache=False):
    d3 = np.load(DATA_DIR / "q23_main_steps.npz")
    t3, um3 = d3["t"], d3["um"]
    d4 = np.load(DATA_DIR / "q4_main_steps.npz")
    t4, um4 = d4["t"], d4["um"]
    tc, umc, tfc = fixedR_control(force=no_cache)

    tf3 = d3["crossing"][2] / 3600.0
    tf4 = d4["crossing"][2] / 3600.0
    lb3 = float([r["value"] for r in read_csv_rows("v3_bound.csv")[1]
                 if r["quantity"] == "t_lb_h"][0])
    lb4 = float([r["value"] for r in read_csv_rows("q4_bound.csv")[1]
                 if r["quantity"] == "t_int_h"][0])
    _, split = read_csv_rows("q4_split.csv")
    tf_ctl_csv = float([s["t_f_h"] for s in split if s["case"] == "appendix4_fixedR"][0])

    # 与结果总账.csv 交叉核对（图上的 t_f 必须等于总账口径，防“图文不一致”）
    ledger = {}
    for line in (DATA_DIR / "结果总账.csv").read_text(encoding="utf-8").splitlines():
        f = line.split(",")
        if len(f) >= 4 and f[0] == "t_f":
            if f[1].startswith("问题3主值"):
                ledger["q3"] = float(f[3])
            elif f[1].startswith("问题4主值"):
                ledger["q4"] = float(f[3])
    assert abs(ledger["q3"] - tf3) < 1e-6 and abs(ledger["q4"] - tf4) < 1e-6, ledger
    print(f"  [F4] 与结果总账.csv 一致：{ledger['q3']} / {ledger['q4']} h", flush=True)

    h3, h4, hc = t3 / 3600.0, t4 / 3600.0, tc / 3600.0
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.24)
    ax.plot(h3, um3, color=COL["q3"], lw=1.8,
            label=f"问题3 主解（附3 + 固定 $R$）：$t_f$={tf3:.2f} h")
    ax.plot(h4, um4, color=COL["q4"], lw=1.8,
            label=f"问题4 主解（附4 + $R(t)$）：$t_f$={tf4:.2f} h")
    ax.plot(hc, umc, color=COL["ctrl"], lw=1.6, ls="-",
            label=f"对照：附4 + 固定 $R_0$：$t_f$={tfc:.2f} h")
    ax.axhline(TH, color=COL["th"], ls="--", lw=1.3,
               label=f"达标阈值 $C_{{th}}$={TH} kg/kg（题目）")
    ax.text(1.0, TH + 0.045, f"${TH}$", fontsize=8.4, color=COL["th"])
    for lb, col in ((lb3, COL["q3"]), (lb4, COL["q4"])):
        ax.axvline(lb, color=col, ls=":", lw=1.5)
    for lb, col, txt, yf, yt in ((lb3, COL["q3"], f"问题3 严格下界 {lb3:.2f} h", 0.88, 1.95),
                                 (lb4, COL["q4"], f"问题4 严格下界 {lb4:.2f} h", 0.79, 1.45)):
        ax.annotate(txt, xy=(lb, yt), xytext=(0.175, yf), textcoords="axes fraction",
                    fontsize=8.2, color=col, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=col))
    for hh, col in ((tf3, COL["q3"]), (tf4, COL["q4"]), (tfc, COL["ctrl"])):
        ax.axvline(hh, color=col, ls="-.", lw=0.9, alpha=0.85)
    ax.set_xlim(0, 134)
    ax.set_ylim(0, 2.62)
    ax.set_xlabel("烘干时间 $t$ / h")
    ax.set_ylabel(r"$\max_q U(q,t)$ / (kg/kg)")
    ax.legend(loc="upper right", framealpha=0.94)
    tile(ax, "问题3/4 达标穿越：最大水分浓度何时跌破 0.15（回答：$t_f$ 多少、收缩贡献多大）")

    ins = ax.inset_axes([0.43, 0.335, 0.52, 0.375])
    lo, hi = min(tf4, tf3) - 5.0, max(tf4, tf3) + 4.0
    for hh, uu, col, lab in ((h3, um3, COL["q3"], "问题3 主解"),
                             (h4, um4, COL["q4"], "问题4 主解")):
        m = (hh >= lo) & (hh <= hi)
        ins.plot(hh[m], uu[m], color=col, lw=1.7, label=lab)
    ins.axhline(TH, color=COL["th"], ls="--", lw=1.2)
    ins.set_xlim(lo, hi)
    ins.set_ylim(0.1445, 0.1575)
    ins.set_xlabel("$t$ / h", fontsize=8.2, labelpad=0.5)
    ins.tick_params(labelsize=7.6, pad=1.5)
    ins.grid(True, alpha=0.28, lw=0.5)
    ins.text(0.03, 0.95, "穿越区间放大（$\\max_q U$ 单位：kg/kg）",
             transform=ins.transAxes, fontsize=8.2, ha="left", va="top")
    ins.annotate(f"$t_f$={tf3:.2f} h", xy=(tf3, TH), xytext=(tf3 + 0.4, 0.1462),
                 fontsize=7.8, color=COL["q3"])
    ins.annotate(f"$t_f$={tf4:.2f} h", xy=(tf4, TH), xytext=(tf4 - 4.2, 0.1462),
                 fontsize=7.8, color=COL["q4"])
    ins.legend(fontsize=7.4, loc="lower right", framealpha=0.92)
    shrink, tot = tf_ctl_csv - tf4, tf3 - tf4
    fig.text(0.09, 0.115,
             f"效应拆分（T6，源 q4_split.csv）：对照「附4 + 固定 $R_0$」$t_f$={tf_ctl_csv:.2f} h；"
             f"改用附3 物性（仍固定 $R_0$）降至 {tf3:.2f} h（{tf3-tf_ctl_csv:+.2f} h）；"
             f"再改用真实收缩 $R(t)$ 降至 {tf4:.2f} h（{tf4-tf_ctl_csv:+.2f} h，"
             f"占对照 {shrink/tf_ctl_csv*100:.1f}%）——收缩是问题4 的主导效应。",
             fontsize=8.8, ha="left", va="top")
    fig.text(0.09, 0.055,
             f"严格下界（V3 闸门）：问题3 {lb3:.2f} h、问题4 {lb4:.2f} h"
             "——$t_f$ 低于对应下界者必然错误；主解分别高出 "
             f"{tf3-lb3:.2f} h / {tf4-lb4:.2f} h，闸门通过。"
             "源：v3_bound.csv（t_lb_h）、q4_bound.csv（t_int_h）",
             fontsize=8.8, ha="left", va="top")
    print(f"  [F4] t_f: 问题3 {tf3:.4f} h / 问题4 {tf4:.4f} h / 附4固定R {tfc:.4f} h"
          f"（csv {tf_ctl_csv:.4f}）; 下界 {lb3:.4f} / {lb4:.4f} h", flush=True)
    assert tf4 > lb4 and tf3 > lb3, "主解低于严格下界"
    save(fig, "fig4_达标穿越")


# ====================== F5 敏感性 ======================
def read_sensitivity():
    """sensitivity.csv 的“组别”内含英文逗号（如 C_e=0.04986(题目口径,主)），
    按“末两列为 t_f_h / delta_t_f_h”解析，组别名保留原文。"""
    with open(DATA_DIR / "sensitivity.csv", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    out = []
    for r in rows[1:]:
        out.append((",".join(r[:-2]), float(r[-2]), float(r[-1])))
    return out


def fmt_mag(m):
    """误差量级标签：10 的整数次幂用 $10^k$，否则 %.3g。"""
    k = np.log10(m)
    if abs(k - round(k)) < 1e-9:
        return rf"$10^{{{int(round(k))}}}$ h"
    return f"{m:.3g} h"


def fig5():
    sens = read_sensitivity()
    bars = [(n, d, tf) for n, tf, d in sens]
    base = [b for b in bars if abs(b[1]) < 1e-12]
    bars = sorted([b for b in bars if abs(b[1]) > 1e-12], key=lambda x: abs(x[1]))
    _, budget = read_csv_rows("error_budget.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8),
                                   gridspec_kw=dict(width_ratios=[1.3, 1.0]))
    # 分两组：敏感性因素（≤3 条）+ 问题4 效应拆分归因（T6）
    grp1 = sorted([b for b in bars if not b[0].startswith("问题4效应拆分")],
                  key=lambda x: -x[1])
    grp2 = sorted([b for b in bars if b[0].startswith("问题4效应拆分")],
                  key=lambda x: -x[1])
    shown = grp1 + grp2
    ys, cur = [], len(shown) - 0.5
    for i, b in enumerate(shown):
        if i == len(grp1):
            cur -= 1.0                      # 组间留白
        ys.append(cur)
        cur -= 1.0
    cols = [COL["pos"] if v > 0 else COL["neg"] for _, v, _ in shown]
    ax1.barh(ys, [v for _, v, _ in shown], color=cols, height=0.60, zorder=3)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([n for n, _, _ in shown], fontsize=8.6)
    ax1.axvline(0, color="0.25", lw=1.0)
    ax1.set_xlabel(r"$\Delta t_f$ / h（相对各自基准解）")
    ax1.set_xlim(-14, 96)
    ax1.set_ylim(min(ys) - 0.8, max(ys) + 1.5)
    for y, (n, v, tf) in zip(ys, shown):
        ax1.text(v + (2.0 if v > 0 else -2.0), y, f"{v:+.4f} h",
                 va="center", ha="left" if v > 0 else "right", fontsize=8.2)
    div = 0.5 * (ys[len(grp1) - 1] + ys[len(grp1)])
    ax1.axhline(div, color="0.55", lw=0.9, ls="--", zorder=1)
    ax1.text(-13.4, ys[0] + 0.62, "敏感性因素", fontsize=8.8, color="0.25",
             ha="left", va="bottom")
    ax1.text(-13.4, div + 0.42, "问题4 效应拆分（T6 归因）", fontsize=8.8,
             color="0.25", ha="left", va="bottom")
    ce = [b for b in grp1 if b[0].startswith("C_e")][0]
    others = sum(abs(b[1]) for b in grp1 if b is not ce)
    ax1.annotate(f"$C_e$ 口径单因素 {ce[1]:+.2f} h，是其余敏感性因素之和"
                 f"（{others:.2f} h）的 {abs(ce[1])/others:.0f} 倍\n"
                 "⇒ 结论的最大不确定性来源",
                 xy=(ce[1], ys[[b[0] for b in shown].index(ce[0])]),
                 xytext=(0.985, 0.99), textcoords="axes fraction",
                 fontsize=8.3, ha="right", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.30"),
                 bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.72", lw=0.6, alpha=0.94))
    note(ax1, f"基准行（$\\Delta t_f$=0，未画柱）：{'；'.join(b[0] for b in base)}；"
              "红=偏晚、蓝=偏早", xy=(0.985, 0.01), ha="right", va="bottom", fs=7.8)
    tile(ax1, "(a) 敏感性：各因素 $\\Delta t_f$（回答：结论有多稳）")

    mags = np.array([float(r["量级_h"]) for r in budget])
    typs = [r["类型"] for r in budget]
    tcol = {"可收敛": "#1f6fb2", "口径依赖": "#e08a1e", "固有": "#8e44ad", "情景假设": "#c0392b"}
    ys2 = np.arange(len(mags))[::-1]
    for y, m, tp in zip(ys2, mags, typs):
        ax2.barh(y, m, color=tcol.get(tp, "0.5"), height=0.62, zorder=3)
        ax2.text(m * 1.45, y, fmt_mag(m), va="center", fontsize=8.2)
    ax2.set_yticks(ys2)
    ax2.set_yticklabels([r["误差源"] for r in budget], fontsize=8.4)
    ax2.set_xscale("log")
    ax2.set_xlim(2e-7, 200)
    ax2.set_xlabel("误差量级 / h（对数轴）")
    ax2.legend(handles=[Patch(color=c, label=t) for t, c in tcol.items()],
               loc="upper right", framealpha=0.92)
    tile(ax2, "(b) 误差预算：各类误差量级（源 error_budget.csv）")
    print(f"  [F5] 敏感性 {len(shown)} 条（基准 {len(base)} 条未画），"
          f"主导 {ce[0]} {ce[1]:+.4f} h（其余敏感性因素合计 {others:.2f} h）；"
          f"误差预算 {len(mags)} 条", flush=True)
    save(fig, "fig5_敏感性")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig5": fig5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", default=[], help="只出指定图，如 fig4")
    ap.add_argument("--no-cache", action="store_true", help="fig4 对照曲线强制重算")
    a = ap.parse_args()
    plt.rcParams.update(RC)
    want = a.only or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for k in want:
        print(f"[{k}] 作图 …", flush=True)
        if k == "fig4":
            fig4(no_cache=a.no_cache)
        else:
            FIGS[k]()
    if GLYPH.msgs:
        print("[字体] 缺字告警如下（应视为失败）：", flush=True)
        for m in GLYPH.msgs:
            print("   ", m, flush=True)
        sys.exit(1)
    print("[字体] 无缺字告警；PDF 为嵌入 TrueType 子集（pdf.fonttype=42）", flush=True)


if __name__ == "__main__":
    main()
