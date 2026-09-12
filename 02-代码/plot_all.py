"""M5 图表生成（出版级样式重构版）：从 03-数据/ 的唯一数据源出 6 张论文用矢量图。

用法：  python plot_all.py            # 一次出全部 10 张（PDF + 300dpi PNG + 灰度预览）
         python plot_all.py fig4       # 只出某一张（调试用）
         python plot_all.py --no-cache # fig4 的附4固定R对照曲线强制重算

样式规范（同 05-图表/样式重构说明.md，供论文侧协调尺寸）：
  * 按最终物理尺寸设计，导出后不再缩放：单面板宽 5.4 in（= 0.86×6.3 in 正文文本宽），
    双子图宽 6.3 in（= \\linewidth 6.3 in），纵横比 0.50–0.62。
  * Okabe-Ito 色盲安全色板 + 语义色映射 ROLE（同一变量跨图同色）+ 线型/标记冗余编码。
  * 字号 ≥7.5 pt（正文标签 8、刻度 7.5、图例 7.5），白底、去顶/右脊柱、图例无框、网格极浅。
  * 面板标签 (a)/(b) 粗体置于左上外侧；标题 ≤28 字符。
  * **图内不放说明性长文字**：数据源、口径、偏差数字等论述一律由论文 \\caption 承载；
    脚本把每张图的关键数字打印到 stdout（[F1]…[F6] 行），论文侧据此写图注。

数据源（每张图对应 §3 规格）：
  fig1  附件1.xlsx（T_a、C_env）、附件2.xlsx（R）
  fig2  q1_fields.npz（T_C、times、pos_cm、env_*）、v1_points_final.csv、v1_convergence.csv
  fig3  q23_main_steps.npz（t、C、T）、q23_summary.json、criteria.csv
  fig4  q23_main_steps.npz（um）、**q4_endo_fields.npz（C → 逐帧 nanmax 得 maxU，内生几何主解）**、
        q4_split.csv、q4_bound.csv、v3_bound.csv、sensitivity.csv、结果总账.csv；
        附4固定R对照曲线由 solver_q4 现算（与 q4_split.csv 的 appendix4_fixedR 逐次核对）
  fig5  sensitivity.csv（敏感性因素）、q4_split.csv（问题4 效应拆分，基准＝内生几何主解）、
        error_budget.csv
  fig6  endogenous_shrinkage.csv（附件2 时刻网格上的各闭合 R_pred、Ū_model、Ū_inf；
        其驱动场为**附件2 几何（外生 R）**的求解结果，见 03-数据/内生收缩分析.md）；
        缺列时回退 附件2.xlsx、q4_main_steps.npz
  fig7  endogenous_calibrated.csv（附件2 网格上的 R_data、R_pred、残差、失水基线；
        标定用的驱动场同为附件2 几何，见 03-数据/内生收缩模型.md）
  fig8  q4_endo_fields.npz（内生几何 C、T、R_t → 3D 时空曲面）
  fig9  q4_endo_fields.npz（同上 → 6 个时刻的圆盘截面）
  fig10 mechRefit.csv（145 点 R_data/R_pred/残差）、mechRefit_fields.npz（重标定参数、
        全程场量 ū/De/g、末端平衡表 terminal、t_f_h）；口径见
        03-数据/力学升级_修复重标定.md（A04/A05 修复后重标定的幂律吸力力学模型，
        取代此前的线性检验本构情景族；旧文 力学升级_重标定.md 的数值一律作废）

口径常数（非结果数字，仅用于画线/判定，来源已注明）：
  TH = 0.15 kg/kg  —— 题目问题 3 原文“水分浓度应低于 0.15 kg/kg”
  fig6 H_SEG / C_GLASS / RHO_SK —— 附件2 三段分界、玻璃化含水率、骨架密度，
    来源 03-数据/内生收缩分析.md §0–§1（图内所有半径/含水率/交叉点数字仍从数据读出）
  其余全部数字（含 35 h 冻结点、严格下界）均从上述数据文件读出，脚本内不誊写结果。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

CODE_DIR = Path(__file__).resolve().parent
A_DIR = CODE_DIR.parent
DATA_DIR = A_DIR / "03-数据"
FIG_DIR = A_DIR / "05-图表"
QA_DIR = FIG_DIR / "_qa"                     # 灰度预览等自检产物（不入 git）
ATT1 = A_DIR / "01-题目" / "原始文件" / "附件1.xlsx"
ATT2 = A_DIR / "01-题目" / "原始文件" / "附件2.xlsx"
sys.path.insert(0, str(CODE_DIR))

# ---- 题目口径常数（非计算结果；见模块 docstring）----
TH = 0.15                      # kg/kg，问题 3 达标阈值（题目原文）

CI = 120                       # 屏幕渲染 dpi（1 pt = dpi/72 px）
CLIP_TOL = 1.0                 # 文字超出画布容差（px）：仅 >1 px 视为真裁切

# ---- 设计尺寸（英寸；论文实际包含尺寸，导出后不再缩放）----
W_SINGLE = 5.4                 # 单面板：0.86 × 6.3 in 正文文本宽
W_DOUBLE = 6.3                 # 双子图：\linewidth（需论文侧改为 width=\linewidth）
ASPECT = (0.50, 0.62)          # 高/宽 允许区间
# 例外：fig5 为“长中文类别名”的两行竖排（用户规范允许的排法），纵横比需 0.66 才能让
# 7.5 pt 类别名不互相压盖——压到 0.62 时实测刻度标签重叠（本条即审计的硬门槛之一）。
ASPECT_EXEMPT = {"fig5_敏感性": (0.50, 0.70)}

# ---- Okabe-Ito 色盲安全色板 ----
OI = dict(
    blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
    magenta="#CC79A7", sky="#56B4E9", grey="#6B7280", dark="#222222",
)

# ---- 语义色映射（同一变量跨图同色；见 05-图表/样式重构说明.md 的映射表）----
ROLE = dict(
    data=OI["dark"],        # 附件/实测数据（附件1 温湿、附件2 R(t)）
    model=OI["blue"],       # 模型主解（问题4 主解；fig6 最接近的零拟合闭合）
    model2=OI["sky"],       # 模型次解（问题3 主解，固定 R 口径）
    ref=OI["vermillion"],   # 对照/参考解/基线（附4+固定R0；fig6 全局理想基线）
    analytic=OI["green"],   # 解析解
    aux1=OI["orange"],      # 次要物理量：温度 T
    aux2=OI["magenta"],     # 次要物理量：环境水分 C_env；fig6 第三机制（成孔）
    guide=OI["grey"],       # 阈值/严格下界/平台线（一律虚线/点线）
)

# ---- 出版级样式基线 ----
RC = {
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,          # 嵌入 TrueType 子集（矢量，可复制文字）
    "ps.fonttype": 42,
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.2,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": None,        # 按 figsize 精确出图，不裁切（保证设计宽度即最终宽度）
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "lines.dash_capstyle": "round",
}
MIN_FONT = 7.5                 # 最终尺寸下最小字号（pt）
TITLE_MAX = 28                 # 面板标题字符上限
LEG_MAX = 5                    # 图例项上限
# 例外：fig3 的两个面板各 6 条曲线，颜色编码的是“有序连续时间族”（非 6 个并列类别）。
# 该族末段曲线在 r→2 处重合（24/36/57 h 的 C 相差 <0.011 kg/kg），直接标注会互相压盖，
# 故保留 6 项 ncol=2 图例（checklist 的“>5 项改直接标注”对连续族不适用）。
LEG_MAX_EXEMPT = {"fig3_温湿剖面": 6}

TIME_CMAP = plt.cm.viridis     # 时间族专用（感知均匀；禁 jet/rainbow）
GRID_KW = dict(color="0.86", lw=0.4, alpha=1.0)


class _GlyphWatch(logging.Handler):
    """抓 matplotlib 的缺字告警（豆腐块检测，logging 通道）。"""

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


def log_minor_off(ax, which="both"):
    """关闭对数轴的副刻度标签：matplotlib 会给落在视图外的副刻度也生成标签文本（axes=None），
    审计会把它们算作“裁切文字”。视觉上无变化（副刻度标签默认极密，正式图不用）。"""
    for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        if which in ("both", name):
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def crossing_time(t, y, th):
    """y(t) 自上方穿越 th 的线性插值时刻（t 升序）。"""
    i = int(np.argmax(y < th))
    assert 0 < i < len(y), "曲线未穿越阈值"
    return float(t[i - 1] + (th - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))


# ====================== 样式构件 ======================
def style_axis(ax, grid="y"):
    """统一的出版级坐标轴：白底、去顶/右脊柱、极浅网格（可选）。"""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(RC["axes.linewidth"])
    if grid == "y":
        ax.yaxis.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
    elif grid == "x":
        ax.xaxis.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
    elif grid == "both":
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)


def panel(ax, tag, dx=-0.10):
    """面板标签 (a)/(b)：粗体小写，左上外侧（3D 轴须用 text2D，否则被当数据坐标）。"""
    kw = dict(transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom",
              ha="left", color="0.10")
    if hasattr(ax, "get_zlim"):          # mplot3d
        ax.text2D(dx, 1.03, tag, **kw)
    else:
        ax.text(dx, 1.03, tag, **kw)


def tile(ax, title):
    """面板标题（≤TITLE_MAX 字符；长论述进图注）。"""
    assert len(title) <= TITLE_MAX, f"标题过长({len(title)}): {title}"
    ax.set_title(title, fontsize=8.5, pad=4)


def note(ax, text, xy=(0.98, 0.97), ha="right", va="top", fs=7.5):
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.80", lw=0.5, alpha=0.95))


def fit_left(fig, axes, pad_px=6.0):
    """把左边距设成“y 轴刻度文字 + ylabel”实测所需的最小值（当前 left 够用时不动），
    防止长中文类别名被裁切（bbox 出图为精确 figsize，不裁切，故必须自己留够边距）。
    旧实现无条件在原 left 上再加一份标签宽度，fig5 的长类别名因此白占约 2 in 绘图区。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    w_px = fig.get_size_inches()[0] * fig.dpi
    left_cur = fig.subplotpars.left * w_px
    need = 0.0
    for ax in np.atleast_1d(axes).ravel():
        labs = [t for t in ax.get_yticklabels() if t.get_text().strip()]
        if not labs:
            continue
        left_px = min([t.get_window_extent(ren).x0 for t in labs]
                      + ([ax.yaxis.label.get_window_extent(ren).x0]
                         if ax.get_ylabel() else []))
        need = max(need, ax.get_window_extent(ren).x0 - left_px + pad_px)
    if need > left_cur:
        fig.subplots_adjust(left=need / w_px)
    return fig.subplotpars.left


# ====================== 审计 ======================
AUDIT: list[dict] = []


def tick_texts(axis):
    """仅返回**视图范围内**可见的刻度文字（get_xticklabels 会带上范围外、
    尺寸为 None 的隐藏刻度，拿去判裁切会得到假阳性）。3D 轴（Axis3D）走回退分支。

    容差必须**相对**（2026-09-12 修正）：原实现用绝对 1e-9，对量级 1e-11 的坐标轴
    （如 fig15(c) 的守恒残差）等于无穷大，会把视图外一个数量级的对数刻度也算成
    “已绘制”，进而被误报为“裁切文字”。"""
    try:
        lo, hi = sorted(axis.get_view_interval())
        tol = 1e-9 * max(abs(lo), abs(hi), 1e-300)
        out = []
        for tick in axis.get_major_ticks():
            loc = float(tick.get_loc())
            if lo - tol <= loc <= hi + tol:
                labs = [tick.label1] + ([tick.label2]
                                        if tick.label2.get_text().strip() else [])
                out += [t for t in labs if t.get_visible()]
        return out
    except (AttributeError, TypeError):          # mplot3d Axis3D
        return [t for t in axis.get_ticklabels() if t.get_text().strip() and t.get_visible()]


def _unreliable_extent(t):
    """mplot3d 的刻度文字与 ax.text 的 window extent 经三维投影后不可靠，
    不能用于裁切判定（3D 面板的文字位置只能目视核对）。"""
    ax = t.axes
    if ax is None or not hasattr(ax, "get_zlim"):
        return False
    if t in list(ax.texts):
        return True
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if t in tick_texts(axis):
            return True
    return False


def all_axes(fig):
    """图中**全部**坐标轴，含 inset（`ax.inset_axes` 建的子轴不在 `fig.axes` 里，
    fig8 的竖排 colorbar 就是 inset；不纳入审计会漏检它的 label 与文字裁切）。"""
    out = list(fig.axes)
    for ax in list(fig.axes):
        out += [c for c in getattr(ax, "child_axes", []) if c not in out]
    return out


def drawn_texts(fig):
    """图中**实际绘制**的文字（白名单遍历）——fig.findobj(Text) 会连带返回
    刻度/数学排版留下的孤立 Text（axes=None），不能用来判裁切与字号。"""
    out = list(fig.texts)
    for ax in all_axes(fig):
        out += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        out += tick_texts(ax.xaxis) + tick_texts(ax.yaxis)
        lg = ax.get_legend()
        if lg is not None:
            out += list(lg.get_texts())
            if lg.get_title():
                out.append(lg.get_title())
    return [t for t in out if t.get_text().strip() and t.get_visible()]


def _extent(txt):
    return txt.get_window_extent(renderer=txt.figure.canvas.get_renderer())


def _overlap(a, b, tol=0.5):
    return (a.x1 > b.x0 + tol and b.x1 > a.x0 + tol
            and a.y1 > b.y0 + tol and b.y1 > a.y0 + tol)


def _tick_overlaps(ax):
    bad = []
    for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        labs = [t for t in tick_texts(axis) if t.get_text().strip()]
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                if _overlap(_extent(labs[i]), _extent(labs[j])):
                    bad.append(f"{name}:{labs[i].get_text()}/{labs[j].get_text()}")
    return bad


def audit_figure(fig, stem, pdf_path):
    """逐张图的形式合规审计（尺寸/字号/标题/图例/柱基线/colorbar/刻度重叠/冗余编码）。"""
    rec = dict(stem=stem, w=round(fig.get_size_inches()[0], 3),
               h=round(fig.get_size_inches()[1], 3), titles=[], legends=[], cb=0, ax3d=0,
               bar_base=[], tick_bad=[], dense_marker=0, min_font=99.0, embed_img=0,
               clipped=[], cap_hit=[], cb_labels=[], cb_bad=[])
    rec["aspect"] = round(rec["h"] / rec["w"], 3)
    ren = fig.canvas.get_renderer()
    fw, fh = float(ren.width), float(ren.height)
    # 图内不得有 figure 级文字块（数据源/口径/偏差等论述一律由论文 \caption 承载）
    rec["figtext"] = [t.get_text()[:16] for t in fig.texts if t.get_text().strip()]

    texts = drawn_texts(fig)
    for t in texts:
        rec["min_font"] = min(rec["min_font"], float(t.get_fontsize()))

    # 图注（fig.text）不得压住任何面板的 x 轴标签
    for t in fig.texts:
        tb = t.get_window_extent(renderer=ren)
        for ax in all_axes(fig):
            lab = ax.xaxis.label
            if lab.get_text().strip() and _overlap(tb, lab.get_window_extent(renderer=ren),
                                                   tol=0.0):
                rec["cap_hit"].append(f"{t.get_text()[:12]}↔{lab.get_text()[:12]}")

    for ax in all_axes(fig):
        if ax.get_title():
            rec["titles"].append(ax.get_title())
        lg = ax.get_legend()
        if lg is not None:
            rec["legends"].append(len(lg.get_texts()))
        # colorbar 轴：内部创建的 label 为 "<colorbar>"，用户传入 cax 时只有 _colorbar 属性
        if ax.get_label() == "<colorbar>" or getattr(ax, "_colorbar", None) is not None:
            rec["cb"] += 1
            lab = (ax.get_ylabel().strip() or ax.get_xlabel().strip())   # 横/竖 colorbar
            rec["cb_labels"].append(lab)
            # 连续色阶必须配 label（含单位）；无 label 的 colorbar 视为冗余
            if not lab:
                rec["cb_bad"].append("colorbar 缺 label")
        if hasattr(ax, "get_zlim"):
            rec["ax3d"] += 1
        for cont in ax.containers:
            # 柱状图基线必须落在 0：水平柱查 x 基线，竖直柱查 y 基线（有符号条形允许
            # 负向，但基线仍须是 0；对数轴由 matplotlib 贴住轴下限，x 仍为 0）
            horiz = getattr(cont, "orientation", "vertical") == "horizontal"
            for p in getattr(cont, "patches", []):
                if not isinstance(p, Rectangle):
                    continue
                if horiz:
                    if p.get_width() != 0 and min(abs(p.get_x()),
                                                  abs(p.get_x() + p.get_width())) > 1e-9:
                        rec["bar_base"].append(round(float(p.get_x()), 4))
                elif p.get_height() != 0 and min(abs(p.get_y()),
                                                 abs(p.get_y() + p.get_height())) > 1e-9:
                    rec["bar_base"].append(round(float(p.get_y()), 4))
        rec["tick_bad"] += _tick_overlaps(ax)

    for ln in fig.findobj(Line2D):
        x = np.asarray(ln.get_xdata())
        if len(x) > 25 and ln.get_marker() not in ("None", None, " ", "") \
                and ln.get_markevery() is None:
            rec["dense_marker"] += 1

    rec["embed_img"] = pdf_path.read_bytes().count(b"/Subtype /Image")
    # 场图（imshow 热图/圆盘）本质是栅格数据，允许每张场图面板嵌入 1 张位图；
    # colorbar 的色带也用一条窄图绘制。除此之外的嵌入位图 = 图被栅格化，判 FAIL。
    rec["raster_ax"] = sum(1 for ax in all_axes(fig) if ax.get_images())
    rec["img_ok"] = rec["embed_img"] <= rec["raster_ax"] + rec["cb"]
    # 文字裁切检查：bbox=None 精确出图，超出画布的文字会被静默切掉
    ren = fig.canvas.get_renderer()
    fw, fh = float(ren.width), float(ren.height)
    rec["clipped"] = []
    for t in texts:
        if _unreliable_extent(t):
            continue                      # 3D 文字 extent 不可用，改由目视核对
        bb = t.get_window_extent(renderer=ren)
        over = max(-bb.x0, bb.x1 - fw, -bb.y0, bb.y1 - fh)   # >0 即超出画布
        if over > CLIP_TOL:
            rec["clipped"].append((t.get_text()[:16], round(float(over), 1)))
    rec["font_ok"] = rec["min_font"] >= MIN_FONT - 1e-9
    rec["title_ok"] = all(len(t) <= TITLE_MAX for t in rec["titles"])
    rec["legend_ok"] = all(n <= LEG_MAX_EXEMPT.get(stem, LEG_MAX) for n in rec["legends"])
    rec["tick_ok"] = len(rec["tick_bad"]) == 0
    rec["bar_ok"] = not rec["bar_base"]
    rec["clip_ok"] = not rec["clipped"] and not rec["cap_hit"]
    # 场图（fig8/fig9）必须有带 label 的 colorbar；其余图不得出现无 label 的 colorbar
    rec["cb_ok"] = not rec["cb_bad"]
    rec["ok"] = (rec["font_ok"] and rec["title_ok"] and rec["legend_ok"] and rec["tick_ok"]
                 and rec["bar_ok"] and rec["clip_ok"] and not rec["figtext"]
                 and rec["cb_ok"] and rec["img_ok"] and rec["dense_marker"] == 0
                 and ASPECT_EXEMPT.get(stem, ASPECT)[0] <= rec["aspect"]
                 <= ASPECT_EXEMPT.get(stem, ASPECT)[1])
    AUDIT.append(rec)
    return rec


def save(fig, stem):
    """导出 矢量 PDF + 300dpi PNG + 灰度预览，并逐张审计。"""
    FIG_DIR.mkdir(exist_ok=True)
    QA_DIR.mkdir(exist_ok=True)
    pdf, png = FIG_DIR / f"{stem}.pdf", FIG_DIR / f"{stem}.png"
    fig.canvas.draw()                       # 先渲染，供刻度/字号 bbox 审计使用

    # 1) 矢量 PDF（fonttype 42）＋ 2) 300 dpi PNG；warnings 通道捕获缺字
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        fig.savefig(pdf, format="pdf")
        fig.savefig(png, format="png", dpi=300)
    for w in wlist:
        m = str(w.message)
        if "missing from font" in m or "Glyph" in m:
            GLYPH.msgs.append(f"[warnings] {stem}: {m}")

    # 3) 灰度预览（同一渲染的灰阶，用于“不靠颜色也能区分”核验）
    from PIL import Image
    Image.open(png).convert("L").save(QA_DIR / f"{stem}_gray.png")

    fig.canvas.draw()          # savefig 会临时改 dpi；重绘回 fig.dpi 再审计（bbox 一致性）
    rec = audit_figure(fig, stem, pdf)
    flag = "PASS" if rec["ok"] else "FAIL"
    print(f"  [审计] {flag} {stem}: {rec['w']:.1f}×{rec['h']:.1f} in(纵横比 {rec['aspect']:.2f}) "
          f"最小字号 {rec['min_font']:.1f} pt 标题 {len(rec['titles'])} 图例 {rec['legends']} "
          f"刻度重叠 {len(rec['tick_bad'])} 嵌入位图 {rec['embed_img']} "
          f"裁切文字 {len(rec['clipped'])} 图注压轴标 {len(rec['cap_hit'])} "
          f"图内文字块 {len(rec['figtext'])}", flush=True)
    if rec["clipped"] or rec["cap_hit"]:
        print(f"      [裁切/压轴标] {rec['clipped']} {rec['cap_hit']}", flush=True)
    if rec["ax3d"]:
        print(f"      [3D] {rec['ax3d']} 个 3D 面板：其刻度/轴内文字的位置须目视核对"
              "（extent 不可用，程序无法判裁切）", flush=True)
    print(f"  [出图] {pdf.name} ({pdf.stat().st_size/1024:.0f} kB) "
          f"+ {png.name} ({png.stat().st_size/1024:.0f} kB) "
          f"+ _qa/{stem}_gray.png", flush=True)
    plt.close(fig)
    return rec


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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4))
    fig.subplots_adjust(left=0.098, right=0.935, top=0.865, bottom=0.155, wspace=0.50)

    # ---- (a) 附件1 温湿环境 ----
    axL.plot(h1, Ta, color=ROLE["aux1"], lw=1.3, label="温度 $T_a(t)$（左轴）")
    axL.set_xlabel("时间 $t$ / h")
    axL.set_ylabel("温度 $T_a$ / ℃", color=ROLE["aux1"])
    axL.tick_params(axis="y", colors=ROLE["aux1"])
    axL.set_xlim(0, 6.3)
    axL.set_ylim(27.0, 58.0)
    axL.set_xticks([0, 1, 2, 3, 4, 5, 6])
    axL.set_yticks([30, 35, 40, 45, 50, 55])
    axC = axL.twinx()
    axC.plot(h1, Ce, color=ROLE["aux2"], lw=1.3, ls="--", label="水分浓度 $C_{env}(t)$（右轴）")
    axC.set_ylabel("水分浓度 $C_{env}$ / (kg/kg)", color=ROLE["aux2"])
    axC.tick_params(axis="y", colors=ROLE["aux2"])
    axC.set_ylim(0.0175, 0.0555)
    axC.spines["right"].set_visible(True)          # 双 y 轴：保留右脊柱承载右轴
    axC.spines["right"].set_linewidth(RC["axes.linewidth"])
    axL.axvspan(t_cut, 6.3, color="0.93", lw=0, zorder=0)
    axL.plot([t_cut, 6.3], [Ta[-1]] * 2, color=ROLE["aux1"], ls=":", lw=1.1)
    axC.plot([t_cut, 6.3], [Ce[-1]] * 2, color=ROLE["aux2"], ls=":", lw=1.1)
    axL.axvline(t_cut, color=ROLE["guide"], ls=":", lw=0.9)
    axL.annotate(f"$t$={t_cut:.0f} h 数据截止\n其后末值延拓", xy=(t_cut, 0.80),
                 xycoords=("data", "axes fraction"), xytext=(0.97, 0.62),
                 textcoords="axes fraction", fontsize=7.5, ha="right", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axL.text(0.97, 0.42, f"延拓值（左/右轴）\n$T_a$={Ta[-1]:.3f} ℃\n$C_{{env}}$={Ce[-1]:.5f} kg/kg",
             transform=axL.transAxes, fontsize=7.5, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    hs = [Line2D([], [], color=ROLE["aux1"], lw=1.3),
          Line2D([], [], color=ROLE["aux2"], lw=1.3, ls="--")]
    axL.legend(hs, ["温度 $T_a$（左轴）", "水分 $C_{env}$（右轴）"],
               loc="lower left", bbox_to_anchor=(0.02, 0.02), handlelength=1.6)
    panel(axL, "(a)")
    tile(axL, "附件1：环境温湿与延拓")

    # ---- (b) 附件2 半径 ----
    # 时间口径与论文统一：附件2 约 21 h 起进入 1.198 cm 近平台段（H_SEG 的 B/C 分界），
    # 35 h 只是 R 在三位小数下“首达 1.200 cm”的时刻，不再用作“冻结点”。
    t_plat = 21.0
    axR.plot(h2, R, color=ROLE["data"], lw=1.5, label="附件2 实测 $R(t)$")
    axR.plot([h2[-1], x_end], [R[-1]] * 2, color=ROLE["data"], ls="--", lw=1.1)
    axR.axvspan(t_plat, h2[-1], color="0.93", lw=0, zorder=0)
    axR.axvspan(h2[-1], x_end, color="0.96", lw=0, zorder=0)
    axR.axvline(t_plat, color=ROLE["guide"], ls=":", lw=0.9)
    axR.axvline(h2[-1], color=ROLE["guide"], ls=":", lw=0.7)
    axR.axhline(R[-1], color=ROLE["guide"], ls=":", lw=0.9)
    axR.annotate(f"≈{t_plat:.0f} h 起进入近平台段\n（$t$={t_frz:.0f} h 首达 {R[i_frz]:.3f} cm）",
                 xy=(t_plat, R[int(np.argmin(np.abs(h2 - t_plat)))]),
                 xytext=(0.30, 0.72), textcoords="axes fraction",
                 fontsize=7.5, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axR.annotate(f"附件2 截止 {h2[-1]:.0f} h：\n{R[-1]:.3f} cm 平台延拓",
                 xy=(h2[-1], R[-1]), xytext=(0.40, 0.33), textcoords="axes fraction",
                 fontsize=7.5, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axR.text(98.0, R[-1] + 0.014, f"平台 {R[-1]:.3f} cm", ha="right", va="bottom",
             fontsize=7.5, color="0.30")
    axR.set_xlim(0, x_end)
    axR.set_ylim(1.16, 2.10)
    axR.set_xlabel("时间 $t$ / h")
    axR.set_ylabel("药材半径 $R$ / cm")
    panel(axR, "(b)")
    tile(axR, "附件2：半径收缩与近平台")

    print(f"  [F1] 附件1 {len(t1)} 点 0–{t_cut:.0f} h；附件2 {len(t2)} 点 0–{h2[-1]:.0f} h，"
          f"R {R[0]:.3f}→{R[-1]:.3f} cm，≈{t_plat:.0f} h 起近平台"
          f"（{t_frz:.0f} h 首达 {R[i_frz]:.3f} cm；其后至 {h2[-1]:.0f} h 仅降 "
          f"{R[i_frz]-R[-1]:.3f} cm）", flush=True)
    save(fig, "fig1_输入数据")


# ====================== F2 问题1 解析对拍 ======================
def fig2():
    """(a) 特征位置温度时程：中心 r=0 与表面 r=R₀ 的数值解 vs 半解析解；
    (b) 偏差 |T_num−T 解析| 时程（对数纵轴），标出四位小数输出分辨率 5e-5 ℃。

    数据源 03-数据/q1_fields.npz（times/T_C/Ts，1 s 网格 0–1800 s）、
    03-数据/v1_points_final.csv（对拍点的 T_num/T_analytic/abs_dev）；
    半解析解由 02-代码/analytic_heat.py 的 Robin 谱解现算（300 模，0–1800 s 同网格）。
    """
    from analytic_heat import RobinCylinder
    from solver_q1 import ALPHA, BI_H, R0, T0_K

    d = np.load(DATA_DIR / "q1_fields.npz")
    t_s, pos_cm, T_C, Ts = d["times"], d["pos_cm"], d["T_C"], d["Ts"]
    env_t, env_Ta = d["env_t"], d["env_Ta"]
    _, pts = read_csv_rows("v1_points_final.csv")
    dev_max = max(float(p["abs_dev"]) for p in pts)

    ana = RobinCylinder(BI_H, ALPHA, R0, n_modes=300)
    T_ana = ana.eval(np.array([0.0, R0]), t_s, env_t, env_Ta - T0_K, T0_K) - 273.15
    T_ana_c, T_ana_s = T_ana[:, 0], T_ana[:, 1]
    num_c, num_s = T_C[:, 0], Ts - 273.15        # npz 的 Ts 为开尔文，统一换 ℃
    dev_c, dev_s = np.abs(num_c - T_ana_c), np.abs(num_s - T_ana_s)
    res = 5e-5                                   # 四位小数输出分辨率 ℃

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(W_SINGLE, 3.30), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.30, 1.0]))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.885, bottom=0.155, hspace=0.22)
    m = t_s / 60.0
    for num, ana_, col, lab in ((num_c, T_ana_c, ROLE["model"], "中心 $r$=0"),
                                (num_s, T_ana_s, ROLE["aux1"], "表面 $r$=$R_0$")):
        axA.plot(m, ana_, color=col, lw=1.1, ls="--", label=f"{lab}：解析解")
        axA.plot(m, num, color=col, lw=1.2, label=f"{lab}：数值解")
    t_pt = [float(p["t_s"]) / 60.0 for p in pts if abs(float(p["r_cm"])) < 1e-9]
    y_pt = [float(p["T_num_C"]) for p in pts if abs(float(p["r_cm"])) < 1e-9]
    axA.plot(t_pt, y_pt, ls="none", marker="o", ms=3.0, mfc="white", mec=ROLE["model"],
             mew=0.9, label="对拍点（表 4 的 V1）")
    axA.set_ylabel("温度 $T$ / ℃")
    axA.set_ylim(27.9, 37.6)
    style_axis(axA, grid="y")
    axA.legend(loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.0,
               labelspacing=0.24)
    panel(axA, "(a)", dx=-0.115)
    tile(axA, "特征位置温度时程：解析 vs 数值")

    for dev_, col, lab in ((dev_c, ROLE["model"], "中心 $r$=0"),
                           (dev_s, ROLE["aux1"], "表面 $r$=$R_0$")):
        axB.semilogy(m, dev_, color=col, lw=1.1, label=lab)
    axB.axhline(res, color=ROLE["guide"], ls=":", lw=1.0)
    axB.text(29.4, res * 1.30, "输出分辨率 $5\\times10^{-5}$ ℃", fontsize=7.5, color="0.30",
             ha="right", va="bottom")
    axB.set_xlabel("时间 $t$ / min")
    axB.set_ylabel("偏差 / ℃")
    axB.set_xlim(0, 30)
    style_axis(axB, grid="y")
    axB.legend(loc="lower left", handlelength=1.6, labelspacing=0.24)
    panel(axB, "(b)", dx=-0.115)
    tile(axB, "偏差：全程低于 $3.2\\times10^{-6}$ ℃")

    print(f"  [F2] 解析对拍：对拍点最大偏差 {dev_max:.3e} ℃（{len(pts)} 点，四位小数分辨率 "
          f"{res:.0e} ℃，低 {res/dev_max:.0f} 倍）；1 s 网格 1800 点全程：中心 max "
          f"{dev_c.max():.2e}、表面 max {dev_s.max():.2e} ℃；中心 1800 s 数值 {num_c[-1]:.4f} "
          f"/ 解析 {T_ana_c[-1]:.4f} ℃；表面 1800 s 数值 {num_s[-1]:.4f} / 解析 {T_ana_s[-1]:.4f} ℃",
          flush=True)
    save(fig, "fig2_问题1解析对拍")


# ====================== F3 温湿剖面 ======================
def fig3():
    """多时刻径向剖面：温度（3 h 内）与水分浓度（全程）——线色按时间 viridis 映射，
    每面板配竖直 colorbar（标"时间 (h)"），剖面用"散点+连线"（每 4 点一个标记）。

    数据源 03-数据/q23_main_steps.npz（t、C、T）；t_f 与判据区间取自 q23_summary.json /
    criteria.csv。定量读数是"时间色阶 + colorbar"，图例仅列 4 个代表时刻。
    """
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

    t_T = [0.25, 1.0, 2.0, 3.0]          # 预热平衡段（问题 2 表 3 的 3 h 内）
    t_C = [0.5, 12.0, 36.0, 57.0]        # 全程（问题 3，至 t_f≈57.2 h）

    fig = plt.figure(figsize=(W_DOUBLE, 3.4))
    axT = fig.add_axes([0.068, 0.170, 0.298, 0.690])
    caxT = fig.add_axes([0.374, 0.170, 0.014, 0.690])
    axC = fig.add_axes([0.565, 0.170, 0.298, 0.690])
    caxC = fig.add_axes([0.871, 0.170, 0.014, 0.690])

    for ax, cax, tt, field, ylab, ttl, tag in (
            (axT, caxT, t_T, T, "药材温度 $T$ / ℃", "温度剖面：3 h 内近准稳态", "(a)"),
            (axC, caxC, t_C, C, "水分浓度 $C$ / (kg/kg)", "水分剖面：中心最慢", "(b)")):
        norm = matplotlib.colors.Normalize(vmin=min(tt), vmax=max(tt))
        handles = []
        for h in tt:
            col = TIME_CMAP(norm(h))
            ax.plot(pos_cm, prof(field, h), color=col, lw=1.2, marker="o", ms=2.0,
                    markevery=4, mfc="white", mec=col, mew=0.7)
            handles.append(Line2D([], [], color=col, lw=1.2, label=f"$t$={h:g} h"))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=TIME_CMAP), cax=cax)
        cb.set_label("时间 $t$ / h", fontsize=7.5)
        cb.ax.tick_params(labelsize=7.5)
        cb.set_ticks(tt)
        ax.set_xlabel("到药材中心的距离 $r$ / cm")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 2.02)
        style_axis(ax, grid="y")
        ax.legend(handles=handles, loc="upper left", ncol=2, handlelength=1.4,
                  columnspacing=0.9, labelspacing=0.26)
        panel(ax, tag)
        tile(ax, ttl)
    axT.set_ylim(27.0, 58.5)
    axC.set_ylim(0.0, 3.15)
    axC.axhline(TH, color=ROLE["guide"], ls="--", lw=1.1)
    # 阈值线直接进 y 刻度（0.15 这个刻度即达标阈值），避免图内文字压曲线
    axC.set_yticks([0.0, TH, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    c36, c57 = prof(C, 36.0)[0], prof(C, 57.0)[0]
    cs57 = prof(C, 57.0)[-1]
    axC.annotate("中心最慢\n（$r$=0，尾段拖长）", xy=(0.02, 0.5 * (c36 + c57)),
                 xytext=(0.13, 0.30), textcoords="axes fraction", fontsize=7.5,
                 ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"))
    Ta_end = float(T[-1].max())
    print(f"  [F3] t_f={tf_h:.3f} h；中心 {c57:.4f} / 表面 {cs57:.4f}；36→57 h 中心降 "
          f"{c36-c57:.4f}；Le={le_txt}；t={t_T[-1]:g} h 全径 {Ta_end-0.4:.1f}–{Ta_end:.1f} ℃；"
          f"时间色阶 viridis（4 时刻/面板）+ 竖直 colorbar", flush=True)
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
    # 问题4 主解 = 内生几何版：max_q U(q,t) 由内生时空场逐帧全域（r≤R(t)，域外 NaN）取最大值
    de = np.load(DATA_DIR / "q4_endo_fields.npz")
    t4, um4 = de["t_s"], np.nanmax(de["C"], axis=1)
    tc, umc, tfc = fixedR_control(force=no_cache)

    tf3 = d3["crossing"][2] / 3600.0
    tf4 = crossing_time(t4, um4, TH) / 3600.0
    lb3 = float([r["value"] for r in read_csv_rows("v3_bound.csv")[1]
                 if r["quantity"] == "t_lb_h"][0])
    lb4 = float([r["value"] for r in read_csv_rows("q4_bound.csv")[1]
                 if r["quantity"] == "t_int_h"][0])
    _, split = read_csv_rows("q4_split.csv")
    tf_ctl_csv = float([s["t_f_h"] for s in split if s["case"] == "appendix4_fixedR"][0])

    # 与结果总账.csv 交叉核对：图上 问题4 主解 t_f 必须等于总账的**内生几何主值**（找不到判失败）。
    # 总账该行写的是末帧时刻（步末口径 50.5369 h），本图用线性插值穿越时刻（50.5362 h），
    # 二者差异 ~0.0007 h，故容差取 0.01 h。
    # （按"类别=t_f / 项目前缀"取"值"列：总账的项目名里含英文逗号与引号，不能按逗号裸切）
    _, ledger_rows = read_csv_rows("结果总账.csv")
    ledger = {}
    for r in ledger_rows:
        if r["类别"] != "t_f":
            continue
        for key, pat in (("q3", "问题3主值"), ("q4", "问题4主值")):
            if r["项目"].startswith(pat):
                ledger[key] = float(r["值"])
    assert set(ledger) >= {"q3", "q4"}, ledger
    assert abs(ledger["q3"] - tf3) < 1e-6, ledger
    assert abs(ledger["q4"] - tf4) < 0.01, \
        f"图上问题4主解 t_f={tf4:.4f} h 与总账内生主值 {ledger['q4']} h 不符"
    print(f"  [F4] 与结果总账.csv 一致：问题3 {ledger['q3']} h / 问题4（内生几何主值）"
          f"{ledger['q4']} h（本图插值穿越 {tf4:.4f} h）", flush=True)

    h3, h4, hc = t3 / 3600.0, t4 / 3600.0, tc / 3600.0
    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.2))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.875, bottom=0.17)
    ax.plot(h3, um3, color=ROLE["model2"], lw=1.3,
            label=f"问题3 主解（附3 + 固定 $R$）：$t_f$={tf3:.2f} h")
    ax.plot(h4, um4, color=ROLE["model"], lw=1.5,
            label=f"问题4 主解（内生几何 + $R_{{pred}}(t)$）：$t_f$={tf4:.2f} h")
    ax.plot(hc, umc, color=ROLE["ref"], lw=1.2, ls="--",
            label=f"对照：附4 + 固定 $R_0$：$t_f$={tfc:.2f} h")
    ax.axhline(TH, color=ROLE["guide"], ls="--", lw=1.1,
               label=f"达标阈值 $C_{{th}}$={TH} kg/kg（题目）")
    ax.text(1.0, TH + 0.05, f"${TH}$", fontsize=7.5, color="0.30")
    for lb, col in ((lb3, ROLE["model2"]), (lb4, ROLE["model"])):
        ax.axvline(lb, color=col, ls=":", lw=1.2)
    for lb, col, txt, yf, yt in ((lb3, ROLE["model2"], f"问题3 严格下界 {lb3:.2f} h", 0.86, 1.95),
                                 (lb4, ROLE["model"], f"问题4 严格下界 {lb4:.2f} h", 0.77, 1.42)):
        ax.annotate(txt, xy=(lb, yt), xytext=(0.185, yf), textcoords="axes fraction",
                    fontsize=7.5, color=col, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=col))
    for hh, col in ((tf3, ROLE["model2"]), (tf4, ROLE["model"]), (tfc, ROLE["ref"])):
        ax.axvline(hh, color=col, ls="-.", lw=0.8, alpha=0.85)
    ax.set_xlim(0, 134)
    ax.set_ylim(0, 2.62)
    ax.set_xlabel("烘干时间 $t$ / h")
    ax.set_ylabel(r"$\max_q U(q,t)$ / (kg/kg)")
    style_axis(ax, grid="y")
    ax.legend(loc="upper right", handlelength=1.8, labelspacing=0.30)
    panel(ax, "(a)", dx=-0.115)
    tile(ax, "达标穿越：问题3 vs 问题4")

    ins = ax.inset_axes([0.40, 0.20, 0.555, 0.36])
    lo, hi = min(tf4, tf3) - 5.0, max(tf4, tf3) + 4.0
    for hh, uu, col, lab, sty in ((h3, um3, ROLE["model2"], "问题3 主解", "-"),
                                  (h4, um4, ROLE["model"], "问题4 主解", "-")):
        m = (hh >= lo) & (hh <= hi)
        ins.plot(hh[m], uu[m], color=col, lw=1.3, ls=sty, label=lab)
    ins.axhline(TH, color=ROLE["guide"], ls="--", lw=1.0)
    ins.set_xlim(lo, hi)
    ins.set_ylim(0.1445, 0.1575)
    ins.set_xlabel("$t$ / h", fontsize=7.5, labelpad=0.5)
    ins.tick_params(labelsize=7.5, pad=1.5)
    ins.grid(True, **GRID_KW)
    ins.set_axisbelow(True)
    ins.annotate(f"$t_f$={tf3:.2f} h", xy=(tf3, TH), xytext=(tf3 + 0.8, 0.1565),
                 fontsize=7.5, color=ROLE["model2"], va="bottom")
    ins.annotate(f"$t_f$={tf4:.2f} h", xy=(tf4, TH), xytext=(tf4 + 0.8, 0.1565),
                 fontsize=7.5, color=ROLE["model"], va="bottom")
    print(f"  [F4] t_f: 问题3 {tf3:.4f} h / 问题4（内生几何）{tf4:.4f} h / 附4固定R {tfc:.4f} h"
          f"（csv {tf_ctl_csv:.4f}）; 下界 {lb3:.4f} / {lb4:.4f} h", flush=True)
    i_tf = int(np.argmin(np.abs(h4 - tf4)))
    print(f"  [F4] maxU 抽查（源 q4_endo_fields.npz 的 nanmax(C)）：帧 {i_tf} t={h4[i_tf]:.4f} h "
          f"maxU={um4[i_tf]:.6f}（阈值 {TH}）；末帧 t={h4[-1]:.4f} h maxU={um4[-1]:.6f}", flush=True)
    print(f"  [F4] 效应拆分：① 附4+固定 $R_0$ = {tf_ctl_csv:.2f} h；"
          f"② 附4+内生 $R(t)$ = {tf4:.2f} h；③ 问题3（附3+固定 $R$） = {tf3:.2f} h；"
          f"收缩效应 ①−② = {tf_ctl_csv-tf4:.2f} h（{(tf_ctl_csv-tf4)/tf_ctl_csv*100:.1f}%）；"
          f"净效应 ③−② = {tf3-tf4:.2f} h（{(tf3-tf4)/tf3*100:.1f}%）；"
          f"严格下界余量 {tf3-lb3:.2f} / {tf4-lb4:.2f} h", flush=True)
    assert tf4 > lb4 and tf3 > lb3, "主解低于严格下界"
    save(fig, "fig4_达标穿越")


# ====================== F5 敏感性 ======================
# 误差预算的类别（与论文 表~tab:预算 的三类分报一致）：error_budget.csv 的“类别/适用”列
# 记的是适用范围（问题3/问题4/两问），类别改由误差源前缀判定 → (前缀, 图例名, 颜色)
BUDGET_CAT = [("数值误差", "数值误差", ROLE["model"]),
              ("环境情景", "环境情景", ROLE["ref"]),
              ("口径依赖", "口径依赖", OI["orange"]),
              ("模型结构差异", "模型结构差异", ROLE["aux2"]),
              ("已检查因素", "累计量级", OI["grey"])]


def read_sensitivity():
    """sensitivity.csv 的“组别”内含英文逗号（如 C_e=0.04986(题目口径,主)），
    按“末两列为 t_f_h / delta_t_f_h”解析，组别名保留原文。"""
    with open(DATA_DIR / "sensitivity.csv", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    out = []
    for r in rows[1:]:
        out.append((",".join(r[:-2]), float(r[-2]), float(r[-1])))
    return out


def read_q4_split():
    """q4_split.csv：问题4 效应拆分的唯一数据源（基准＝内生几何主解 t_f）。

    返回各 case 的 t_f/h；csv 自带的派生行（shrink_effect_endo_h、net_effect_endo_h）
    用 t_f 现算并逐条核对，防止“csv 改了、图却没跟着改”。"""
    _, rows = read_csv_rows("q4_split.csv")
    d = {r["case"]: float(r["t_f_h"]) for r in rows}
    base = d["appendix4_endogenous_R(t)"]
    for k, v in (("shrink_effect_endo_h", d["appendix4_fixedR"] - base),
                 ("net_effect_endo_h", d["problem3"] - base)):
        assert abs(d[k] - v) < 1e-6, (k, d[k], v)
    return d


def fmt_mag(m):
    """误差量级标签：10 的整数次幂用 $10^k$，否则 %.3g。"""
    k = np.log10(m)
    if abs(k - round(k)) < 1e-9:
        return rf"$10^{{{int(round(k))}}}$ h"
    return f"{m:.3g} h"


def fig5():
    sens = read_sensitivity()
    q4 = read_q4_split()
    q4_base = q4["appendix4_endogenous_R(t)"]
    # 问题4 效应拆分一律读 q4_split.csv（基准＝内生几何主解）；sensitivity.csv 里
    # 三个旧外生拆分行（问题4效应拆分:*）不再进图
    bars = ([(n, d, tf) for n, tf, d in sens if not n.startswith("问题4效应拆分")]
            + [("问题4效应拆分:附录4固定R(对主解)", q4["appendix4_fixedR"] - q4_base,
                q4["appendix4_fixedR"]),
               ("问题4效应拆分:问题3(对主解)", q4["problem3"] - q4_base,
                q4["problem3"]),
               ("问题4效应拆分:附录4+外生R(t)(交叉验证)", q4["appendix4_R(t)"] - q4_base,
                q4["appendix4_R(t)"])])
    base = [b for b in bars if abs(b[1]) < 1e-12]
    bars = sorted([b for b in bars if abs(b[1]) > 1e-12], key=lambda x: abs(x[1]))
    _, budget = read_csv_rows("error_budget.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_DOUBLE, 4.35),
                                   gridspec_kw=dict(height_ratios=[1.0, 1.15]))
    fig.subplots_adjust(left=0.30, right=0.985, top=0.93, bottom=0.17, hspace=0.45)
    # 分两组：敏感性因素 + 问题4 效应拆分归因（基准＝q4_split.csv 的内生几何主解）
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
    cols = [ROLE["ref"] if v > 0 else ROLE["model"] for _, v, _ in shown]
    ax1.barh(ys, [v for _, v, _ in shown], color=cols, height=0.58, zorder=3)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([n for n, _, _ in shown], fontsize=7.5)
    vals = [v for _, v, _ in shown]
    x_lo = -14.0 if min(vals) < 0 else 0.0          # 无负向柱时轴从 0 起
    ax1.axvline(0, color="0.25", lw=0.8)
    ax1.set_xlabel(r"$\Delta t_f$ / h（相对各自基准解）")
    ax1.set_xlim(x_lo, 105)
    ax1.set_ylim(min(ys) - 0.8, max(ys) + 2.0)
    style_axis(ax1, grid="x")
    for y, (n, v, tf) in zip(ys, shown):
        ax1.text(v + 2.0 if v > 0 else 1.0, y, f"{v:+.4f} h",
                 va="center", ha="left", fontsize=7.5)
    div = 0.5 * (ys[len(grp1) - 1] + ys[len(grp1)])
    ax1.axhline(div, color="0.65", lw=0.8, ls="--", zorder=1)
    ax1.text(103.0, ys[0] + 0.60, "敏感性因素", fontsize=7.5, color="0.25",
             ha="right", va="bottom")
    ax1.text(103.0, div, "问题4 效应拆分（基准：内生主解）", fontsize=7.5,
             color="0.25", ha="right", va="center")
    ce = [b for b in grp1 if b[0].startswith("C_e")][0]
    others = sum(abs(b[1]) for b in grp1 if b is not ce)
    panel(ax1, "(a)")
    tile(ax1, "敏感性：各因素 $\\Delta t_f$")

    # (b) 误差预算：error_budget.csv 的“类别/适用”列已是问题口径（问题3/问题4/两问），
    # 因此类别按误差源前缀现取，与论文 表~\\ref{tab:预算} 的三类分报一一对应
    mags = np.array([float(r["量级_h"]) for r in budget])
    cats = [(k, lab, c) for k, lab, c in BUDGET_CAT
            if any(r["误差源"].startswith(k) for r in budget)]
    ycol = lambda src: next((c for k, _, c in cats if src.startswith(k)), "0.5")
    ys2 = np.arange(len(mags))[::-1]
    for y, m, r in zip(ys2, mags, budget):
        ax2.barh(y, m, color=ycol(r["误差源"]), height=0.60, zorder=3)
        ax2.text(m * 1.45, y, fmt_mag(m), va="center", fontsize=7.5)
    ax2.set_yticks(ys2)
    ax2.set_yticklabels([r["误差源"] for r in budget], fontsize=7.5)
    ax2.set_xscale("log")
    ax2.set_xlim(2e-7, 300)
    ax2.set_xlabel("误差量级 / h（对数轴）")
    style_axis(ax2, grid="x")
    ax2.legend(handles=[Patch(color=c, label=lab) for _, lab, c in cats],
               loc="upper center", bbox_to_anchor=(0.50, -0.22), ncol=len(cats),
               handlelength=1.2, columnspacing=1.0, labelspacing=0.28)
    panel(ax2, "(b)", dx=-0.10)
    tile(ax2, "误差预算：各类误差量级")
    fit_left(fig, (ax1, ax2))
    print(f"  [F5] 敏感性 {len(shown)} 条（基准 {len(base)} 条未画："
          f"{'；'.join(b[0] for b in base)}），红=偏晚、蓝=偏早；"
          f"主导 {ce[0]} {ce[1]:+.4f} h 是其余敏感性因素之和（{others:.2f} h）的 "
          f"{abs(ce[1])/others:.0f} 倍；误差预算 {len(mags)} 条 / "
          + "/".join(f"{lab} {sum(1 for r in budget if r['误差源'].startswith(k))} 条"
                     for k, lab, _ in cats) + "；"
          f"问题4 拆分基准（内生主解）{q4_base:.4f} h → "
          + "；".join(f"{n.split(':')[1]} {v:+.4f} h" for n, v, _ in grp2), flush=True)
    save(fig, "fig5_敏感性")


# ====================== F6 收缩机制分析 ======================
def fig6():
    """附件2 收缩曲线是“多机制复合”的诊断：三条零拟合闭合 vs 附件2（不改主模型）。

    数字全部读自 03-数据/endogenous_shrinkage.csv（附件2 时刻网格上的各闭合 R_pred）；
    缺列时按 03-数据/内生收缩分析.md 的口径回退到 附件2.xlsx / q4_main_steps.npz。
    """
    from solver_q1 import C0, R0        # 初始干基含水率 kg/kg / 初始半径 m（题目物性）

    head, rows = read_csv_rows("endogenous_shrinkage.csv")
    t_s = np.array([float(r["t_s"]) for r in rows])
    h = t_s / 3600.0

    def pick(name):
        return np.array([float(r[name]) for r in rows]) if name in head else None

    R_att2 = pick("R_att2_cm")
    if R_att2 is None:
        _, R_att2 = read_att2()
    U_inf = pick("U_inferred_att2")
    if U_inf is None:                       # 等容失水反推（内生收缩分析.md §0）
        U_inf = (1.0 + C0) * (R_att2 / (R0 * 100.0)) ** 2 - 1.0
    U_mod = pick("U_mean_model")
    if U_mod is None:                       # 回退：附件2 几何（外生 R）的问题4 步记录
        d4 = np.load(DATA_DIR / "q4_main_steps.npz")
        U_mod = np.interp(t_s, d4["t"], d4["um"])
    R_ideal, R_crust, R_pore = (pick("R1_ideal_mean_cm"), pick("R3b_crust_mean_cm"),
                                pick("R5_pore_cm"))
    # 表面触发结皮（+毛细塌陷，论文记为“表面触发，平台约 1.345 cm”）：该列 t=0 值为 1.900
    # （分析脚本的常数塌陷起值），与初始条件 R(0)=2.000 cm 不符，按初始条件修正首点。
    R_surf = pick("R4b_crust_collapse_cm")
    if R_surf is not None:
        R_surf = R_surf.copy()
        R_surf[0] = R_att2[0]
    miss = [n for n, v in (("R_att2_cm", R_att2), ("U_inferred_att2", U_inf),
                           ("U_mean_model", U_mod), ("R1_ideal_mean_cm", R_ideal),
                           ("R3b_crust_mean_cm", R_crust), ("R5_pore_cm", R_pore),
                           ("R4b_crust_collapse_cm", R_surf))
            if v is None]
    assert not miss, f"endogenous_shrinkage.csv 缺列（且无回退）: {miss}"

    # ---- 口径常数（非计算结果；来源：03-数据/内生收缩分析.md）----
    H_SEG = (3.5, 21.0)     # §0 附件2 三段形态：A/B、B/C 分界（h）
    C_GLASS = 0.21          # §1 闭合3 玻璃化转变含水率（Gordon–Taylor 反解）kg/kg
    RHO_SK = 1468.0         # §1 闭合5 骨架密度（文献值）kg/m³

    R_plat = float(R_att2[-1])                      # 附件2 末值平台（1.198 cm）
    dev_crust = (R_crust[-1] - R_plat) / R_plat * 100.0
    dev_ideal = (R_ideal[-1] - R_plat) / R_plat * 100.0
    k_frz = int(np.argmax(np.abs(np.diff(R_crust)) < 1e-12))
    h_frz_crust = float(h[k_frz + 1])               # 结皮闭合冻结时刻（23.5 h）
    dfrz_crust = (h_frz_crust - H_SEG[1]) / H_SEG[1] * 100.0
    i_segA = int(np.argmin(np.abs(h - H_SEG[0])))
    share_A = (R_att2[0] - R_att2[i_segA]) / (R_att2[0] - R_plat)
    exc = R_ideal - R_att2                          # 塌陷超额（附件2 少缩的部分）
    k_exc = int(np.argmax(exc))
    gap = U_inf - U_mod                             # 成孔缺口
    gap35 = float(np.interp(35.0, h, gap))
    gap72 = float(np.interp(72.0, h, gap))
    i_c = next(i for i in range(1, len(gap)) if gap[i] > 0 > gap[i - 1])
    t_cross = float(h[i_c - 1] + (0.0 - gap[i_c - 1]) * (h[i_c] - h[i_c - 1])
                    / (gap[i_c] - gap[i_c - 1]))
    u_cross = float(np.interp(t_cross, h, U_mod))
    assert 15.0 < t_cross < 17.0, f"交叉点 {t_cross:.2f} h 与口径 16 h 不符"
    assert abs(dev_crust) < 5.0 and 8.0 < dfrz_crust < 15.0, (dev_crust, dfrz_crust)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.8))
    fig.subplots_adjust(left=0.088, right=0.985, top=0.865, bottom=0.155, wspace=0.24)

    # ---- 左：半径对比（附件2 vs 三条零拟合闭合）----
    for x0, x1 in ((0.0, H_SEG[0]), (H_SEG[0], H_SEG[1]), (H_SEG[1], h[-1])):
        axL.axvspan(x0, x1, color="0.93", lw=0, zorder=0)
    for xb in H_SEG:
        axL.axvline(xb, color="0.72", ls=":", lw=0.8, zorder=1)
    axL.text(0.5 * H_SEG[0], 2.115, "A 急缩", ha="center", va="center",
             fontsize=7.5, color="0.30")
    axL.text(0.5 * H_SEG[0], 2.040, f"占 {share_A*100:.0f}%", ha="center",
             va="center", fontsize=7.5, color="0.40")
    axL.text(0.5 * (H_SEG[0] + H_SEG[1]), 2.115, "B 缓缩", ha="center", va="center",
             fontsize=7.5, color="0.30")
    axL.text(26.0, 2.115, "C 近平台段", ha="center", va="center", fontsize=7.5,
             color="0.30")
    axL.plot(h, R_att2, color=ROLE["data"], lw=1.6, zorder=5,
             label="附件2 实测")
    axL.plot(h, R_ideal, color=ROLE["ref"], ls="--", lw=1.2,
             label="闭合1 全局理想")
    axL.plot(h, R_surf, color=ROLE["analytic"], lw=1.2,
             label="闭合4b 表面触发结皮")
    axL.plot(h, R_crust, color=ROLE["model"], lw=1.5,
             label="闭合3' 均值触发结皮")
    axL.plot(h, R_pore, color=ROLE["aux2"], ls="-.", lw=1.2,
             label="闭合5 成孔演化")
    axL.axhline(R_plat, color=ROLE["guide"], ls=":", lw=1.0)
    # 三条闭合的平台值与相对附件2 平台（1.198 cm）的偏差，直标在各自平台一端
    for val, col, y0 in ((R_surf[-1], ROLE["analytic"], 0.010),
                         (R_pore[-1], ROLE["aux2"], 0.008),
                         (R_crust[-1], ROLE["model"], -0.032)):
        dev = (val - R_plat) / R_plat * 100.0
        axL.text(h[-1] - 1.0, val + y0, f"{val:.3f}（{dev:+.0f}%）" if abs(dev) >= 10
                 else f"{val:.3f}（{dev:+.1f}%）",
                 ha="right", va="bottom" if y0 > 0 else "top", fontsize=7.5, color=col)
    axL.text(h[-1] - 1.0, R_plat + 0.014, f"平台 {R_plat:.3f} cm", ha="right", va="bottom",
             fontsize=7.5, color="0.30")
    axL.annotate(f"早期塌陷\n峰值超额 {exc[k_exc]:.3f} cm",
                 xy=(h[k_exc], 0.5 * (R_att2[k_exc] + R_ideal[k_exc])),
                 xytext=(0.16, 0.56), textcoords="axes fraction",
                 fontsize=7.5, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axL.annotate("后期成孔 → 平台",
                 xy=(30.0, R_pore[-1] + 0.002), xytext=(0.06, 0.22),
                 textcoords="axes fraction", fontsize=7.5, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axL.set_xlim(0, h[-1] + 2.0)
    axL.set_ylim(1.05, 2.22)
    axL.set_xlabel("烘干时间 $t$ / h")
    axL.set_ylabel("药材半径 $R$ / cm")
    style_axis(axL, grid="y")
    axL.legend(loc="upper right", handlelength=1.4, labelspacing=0.22)
    panel(axL, "(a)")
    tile(axL, "半径对比：四条零拟合闭合")

    # ---- 右：交叉诊断（附件2 等效含水率 vs 模型平均含水率）----
    axR.axvspan(0.0, t_cross, color=ROLE["ref"], alpha=0.05, lw=0)
    axR.axvspan(t_cross, h[-1], color=ROLE["model"], alpha=0.05, lw=0)
    axR.axvline(t_cross, color=ROLE["ref"], ls=":", lw=1.1)
    axR.plot(h, U_inf, color=ROLE["data"], lw=1.4, zorder=5,
             label=r"附件2 $\bar U_{inf}(t)$（由 $R$ 反推）")
    axR.plot(h, U_mod, color=ROLE["model"], lw=1.3,
             label=r"模型 $\bar U_{model}(t)$（附件2 几何求解）")
    axR.plot([t_cross], [u_cross], marker="o", ms=5.4, mfc="white", mec=ROLE["ref"],
             mew=1.3, zorder=6)
    axR.annotate(f"交叉 ≈ {t_cross:.1f} h", xy=(t_cross, u_cross),
                 xytext=(t_cross + 3.0, u_cross + 0.50), fontsize=7.5,
                 color=ROLE["ref"], ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ROLE["ref"]),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axR.text(1.2, 2.05, "附件2 更干（塌陷）", ha="left", va="top",
             fontsize=7.5, color=ROLE["ref"])
    axR.text(38.0, 2.05, "附件2 更湿\n（结皮/成孔）", ha="left", va="top",
             fontsize=7.5, color=ROLE["model"])
    axR.set_xlim(0, h[-1] + 2.0)
    axR.set_ylim(0.0, 2.78)
    axR.set_xlabel("烘干时间 $t$ / h")
    axR.set_ylabel(r"含水率 $\bar U$ / (kg/kg)")
    style_axis(axR, grid="y")
    axR.legend(loc="upper right", handlelength=1.4, labelspacing=0.26)
    panel(axR, "(b)")
    tile(axR, "交叉诊断：等效 vs 模型含水率")

    print(f"  [F6] 平台 {R_plat:.4f} cm；结皮闭合（均值触发）{R_crust[-1]:.4f} cm"
          f"（{dev_crust:+.2f}%）、冻结 {h_frz_crust:.1f} h（{dfrz_crust:+.1f}%，"
          f"口径分界 {H_SEG[1]:g} h）；表面触发结皮 {R_surf[-1]:.4f} cm"
          f"（{(R_surf[-1]-R_plat)/R_plat*100:+.1f}%，首点按初始条件修正为 "
          f"{R_surf[0]:.4f} cm）；理想闭合 {R_ideal[-1]:.4f} cm"
          f"（{dev_ideal:+.1f}%）、成孔闭合 {R_pore[-1]:.4f} cm（骨架 {RHO_SK:.0f}，"
          f"{(R_pore[-1]-R_plat)/R_plat*100:+.1f}%）；"
          f"超额峰值 {exc[k_exc]:.4f} cm @ {h[k_exc]:.1f} h（应变 "
          f"{exc[k_exc]/R_att2[0]*100:.1f}%）；C_glass={C_GLASS}；"
          f"交叉 {t_cross:.2f} h；缺口 {gap35:.4f}→{gap72:.4f} kg/kg", flush=True)
    save(fig, "fig6_收缩机制分析")


# ====================== F7 内生收缩贴合 ======================
def fig7():
    """内生收缩模型（闭合 C，5 参数标定）对附件2 的贴合度：R(t) 对比 + 残差带。

    数字全部读自 03-数据/endogenous_calibrated.csv（附件2 时刻网格上的 R_data / R_pred /
    R_ideal / 残差）；模型结构与标定口径见 03-数据/内生收缩模型.md（复现脚本
    08-临时/endogenous_calibrated.py，只读）。
    """
    head, rows = read_csv_rows("endogenous_calibrated.csv")
    t_s = np.array([float(r["t_s"]) for r in rows])
    h = t_s / 3600.0
    R_data = np.array([float(r["R_data_cm"]) for r in rows])
    R_pred = np.array([float(r["R_pred_cm"]) for r in rows])
    R_ideal = np.array([float(r["R_ideal_cm"]) for r in rows])
    res = np.array([float(r["residual_mm"]) for r in rows])      # R_pred − R_data（mm）

    rmse = float(np.sqrt(np.mean(res ** 2)))
    env = float(np.max(np.abs(res)))                             # 残差包络 ±env mm
    k_env = int(np.argmax(np.abs(res)))
    k_min = int(np.argmin(res))

    def cross(m):
        """m=R/R_ideal 自 <1 穿回 >1 的线性插值时刻（滞后交叉）。"""
        i = next(j for j in range(1, len(m)) if m[j] > 1.0 > m[j - 1])
        return float(h[i - 1] + (1.0 - m[i - 1]) * (h[i] - h[i - 1]) / (m[i] - m[i - 1]))

    t_cross_pred = cross(R_pred / R_ideal)
    t_cross_data = cross(R_data / R_ideal)
    H_SEG = (3.5, 21.0)          # 三段分界（口径常数：内生收缩分析.md §0）
    mB = (h >= H_SEG[0]) & (h <= H_SEG[1])
    res_B = float(np.max(np.abs(res[mB])))
    assert rmse < 0.1 and env < 0.2, (rmse, env)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(W_DOUBLE, 3.8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.55, 1.0]))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.875, bottom=0.155, hspace=0.30)

    # ---- (a) 贴合 ----
    for x0, x1 in ((0.0, H_SEG[0]), (H_SEG[0], H_SEG[1]), (H_SEG[1], h[-1])):
        axA.axvspan(x0, x1, color="0.93", lw=0, zorder=0)
    for xb in H_SEG:
        axA.axvline(xb, color="0.72", ls=":", lw=0.8, zorder=1)
    axA.text(0.5 * H_SEG[0], 2.125, "A 急缩", ha="center", va="center",
             fontsize=7.5, color="0.30")
    axA.text(0.5 * (H_SEG[0] + H_SEG[1]), 2.125, "B 缓缩", ha="center", va="center",
             fontsize=7.5, color="0.30")
    axA.text(26.0, 2.125, "C 近平台段", ha="center", va="center", fontsize=7.5,
             color="0.30")
    axA.plot(h, R_data, color=ROLE["data"], lw=1.6, zorder=5, label="附件2 实测 $R(t)$")
    axA.plot(h, R_pred, color=ROLE["model"], ls="--", lw=1.3,
             label=f"物理启发半经验闭合 $R_{{pred}}(t)$（RMSE={rmse:.3f} mm）")
    axA.plot(h, R_ideal, color=ROLE["ref"], ls=":", lw=1.2,
             label="失水理想基线 $R_{ideal}(t)$")
    axA.axhline(R_data[-1], color=ROLE["guide"], ls=":", lw=1.0)
    axA.text(h[-1] - 1.0, R_data[-1] + 0.013, f"平台 {R_data[-1]:.3f} cm", ha="right",
             va="bottom", fontsize=7.5, color="0.30")
    axA.plot([t_cross_pred], [float(np.interp(t_cross_pred, h, R_ideal))], marker="o",
             ms=4.6, mfc="white", mec=ROLE["ref"], mew=1.1, zorder=6)
    axA.annotate(f"滞后交叉 ≈ {t_cross_pred:.0f} h", xy=(t_cross_pred, 1.213),
                 xytext=(21.0, 1.30), fontsize=7.5, color=ROLE["ref"], ha="left",
                 va="bottom", arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ROLE["ref"]))
    axA.set_ylim(1.05, 2.20)
    axA.set_ylabel("药材半径 $R$ / cm")
    style_axis(axA, grid="y")
    axA.legend(loc="upper right", handlelength=1.8, labelspacing=0.28)
    panel(axA, "(a)", dx=-0.075)
    tile(axA, "贴合：附件2 vs 内生模型")

    # ---- (b) 残差 ----
    for x0, x1 in ((0.0, H_SEG[0]), (H_SEG[0], H_SEG[1]), (H_SEG[1], h[-1])):
        axB.axvspan(x0, x1, color="0.96", lw=0, zorder=0)
    for xb in H_SEG:
        axB.axvline(xb, color="0.80", ls=":", lw=0.7, zorder=1)
    axB.axhline(0.0, color="0.25", lw=0.8)
    for s in (1.0, -1.0):
        axB.axhline(s * env, color=ROLE["guide"], ls="--", lw=0.8)
    axB.plot(h, res, color=ROLE["model"], lw=1.1, zorder=5)
    axB.text(h[-1] - 1.0, env + 0.012, f"包络 ±{env:.3f} mm", ha="right", va="bottom",
             fontsize=7.5, color="0.30")
    axB.annotate(f"最大偏差 {res[k_env]:+.3f} mm @ {h[k_env]:.1f} h",
                 xy=(h[k_env], res[k_env]), xytext=(0.32, 0.97), textcoords="axes fraction",
                 fontsize=7.5, ha="left", va="top", color="0.15",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axB.text(0.5 * (H_SEG[0] + H_SEG[1]), -0.245,
             f"B 段过渡区弱欠拟合（±{env:.2f} mm）", ha="center", va="center",
             fontsize=7.5, color="0.30")
    axB.set_ylim(-0.32, 0.32)
    axB.set_yticks([-0.2, 0.0, 0.2])
    axB.set_xlim(0, h[-1] + 2.0)
    axB.set_xlabel("烘干时间 $t$ / h")
    axB.set_ylabel("残差 / mm")
    style_axis(axB, grid="y")
    panel(axB, "(b)", dx=-0.075)
    tile(axB, "残差（预测−实测）")

    print(f"  [F7] RMSE={rmse:.4f} mm；max|残差|={env:.4f} mm @ {h[k_env]:.1f} h"
          f"（min {res[k_min]:+.4f} @ {h[k_min]:.1f} h）；B 段(3.5–21 h)|残差|max="
          f"{res_B:.4f} mm；平台 data {R_data[-1]:.3f} / pred {R_pred[-1]:.4f} cm；"
          f"滞后交叉 pred {t_cross_pred:.2f} h / data {t_cross_data:.2f} h", flush=True)
    save(fig, "fig7_内生收缩贴合")


# ====================== F8 药柱 3D 时空演化 ======================
def fig8():
    """问题 4 内生版药柱的时空场：两张 3D 曲面，定量读数交给 fig12/fig13/fig16。

    (a) 水分浓度场 $C(s,t)$：$s=r/R(t)$ 归一化（拉格朗日）坐标——移动域被拉直，
        壳核结构与表面轨迹一眼可见（viridis）；
    (b) 温度场 $T(r,t)$：固定物理网格 $r$（域外 $r>R(t)$ 处为 NaN，自由边即半径收缩；
        coolwarm）。低仰角视角、竖排 colorbar 带单位。
    数据源 03-数据/q4_endo_fields.npz（t_s/r_cm/C/T/R_t/Cs）。
    """
    d = np.load(DATA_DIR / "q4_endo_fields.npz")
    t_s, r_cm = d["t_s"], d["r_cm"]
    C, T, R_t, Cs = d["C"], d["T"], d["R_t"], d["Cs"]
    h = t_s / 3600.0

    ti = np.arange(0, len(h), 4)          # 507 帧抽稀到 127，保持 R(t) 边界分辨率
    ri = np.arange(0, len(r_cm), 2)       # 101 网格抽稀到 51
    hh, rr = h[ti], r_cm[ri]
    TT, RR = np.meshgrid(hh, rr, indexing="ij")
    ZT = T[np.ix_(ti, ri)]
    ZC = C[np.ix_(ti, ri)]
    SS = (r_cm / R_t[:, None])[np.ix_(ti, ri)]     # s = r/R(t) ∈ [0,1]（域外为 NaN）

    c_lo, c_hi = float(np.nanmin(C)), float(np.nanmax(C))
    t_lo, t_hi = float(np.nanmin(T)), float(np.nanmax(T))
    probe = [(len(h) // 3, int(np.sum(~np.isnan(C[len(h) // 3]))) - 1),
             (2 * len(h) // 3, int(np.sum(~np.isnan(C[2 * len(h) // 3]))) - 1)]

    fig = plt.figure(figsize=(W_DOUBLE, 3.8))
    rects = [(0.030, 0.250, 0.395, 0.610), (0.545, 0.250, 0.395, 0.610)]
    specs = ((SS, ZC, "viridis", c_lo, c_hi, "(a)", "水分场：拉格朗日坐标 $s$",
              "水分浓度 $C$ / (kg/kg)", "$s = r/R(t)$"),
             (RR, ZT, "coolwarm", t_lo, t_hi, "(b)", "温度场：固定网格 $r$",
              "温度 $T$ / ℃", "$r$ / cm"))
    for (X, Z, cmap_name, lo, hi, tag, title, cbl, xlab), rect in zip(specs, rects):
        ax = fig.add_axes(rect, projection="3d")
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0.0)           # 域外（NaN）不画
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
        ax.plot_surface(X, TT, Z, cmap=cmap, norm=norm, rcount=Z.shape[1],
                        ccount=Z.shape[0], linewidth=0, antialiased=False, shade=False)
        if tag == "(a)":
            ax.plot(np.ones_like(hh), hh, Cs[ti], color=ROLE["data"], lw=1.4, zorder=10)
            ax.text(0.86, 40.0, float(np.nanmax(Cs)) * 0.62, " 表面轨迹", fontsize=7.5,
                    color=ROLE["data"], zorder=20)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        else:
            ax.plot(R_t[ti], hh, zs=lo, zdir="z", color=ROLE["data"], lw=1.2, zorder=10)
            ax.text(1.66, 16.0, lo, " $R(t)$", fontsize=7.5, color=ROLE["data"], zorder=20)
            ax.set_xticks([0.5, 1.0, 1.5, 2.0])
        ax.set_xlim(0, 1.0 if tag == "(a)" else 2.0)
        ax.set_ylim(0, float(h[-1]))
        ax.set_zlim(lo, hi)
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_zticks([np.round(v, 2) for v in np.linspace(lo, hi, 3)])
        ax.set_xlabel(xlab, labelpad=3)
        ax.set_ylabel("$t$ / h", labelpad=3)
        ax.set_zlabel(cbl.split(" / ")[0] + " / " + cbl.split(" / ")[1], labelpad=2)
        ax.tick_params(labelsize=7.5, pad=1)
        ax.view_init(elev=20, azim=-118 if tag == "(a)" else -122)
        ax.set_box_aspect((1.0, 1.5, 0.95), zoom=1.26)
        # 竖排 colorbar 放在面板内右侧（外部放会与相邻 3D 面板的 z 轴标签/刻度相撞）
        cax = ax.inset_axes([0.885, 0.20, 0.042, 0.58])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="vertical")
        cb.set_label(cbl, fontsize=7.5)
        cb.ax.tick_params(labelsize=7.5)
        panel(ax, tag, dx=0.005)
        tile(ax, title)

    print(f"  [F8] C 全域 {c_lo:.4f}–{c_hi:.4f} kg/kg；T 全域 {t_lo:.2f}–{t_hi:.2f} ℃；"
          f"R {R_t[0]:.4f}→{R_t[-1]:.4f} cm（{h[0]:.2f}–{h[-1]:.2f} h，{len(h)} 帧）；"
          f"拉格朗日坐标 $s$=r/R(t)∈[0,1]（面板 a）、温度用固定 $r$（面板 b）；"
          f"抽查 C[i={probe[0][0]}, r={r_cm[probe[0][1]]:.2f}]={C[probe[0]]:.5f}、"
          f"C[i={probe[1][0]}, r={r_cm[probe[1][1]]:.2f}]={C[probe[1]]:.5f}；"
          f"表面末值 $C_s$={Cs[-1]:.4f} kg/kg", flush=True)
    save(fig, "fig8_药柱3D演化")


# ====================== F9 截面演化 ======================
def fig9():
    """问题4 内生版药柱的截面热图：6 个代表时刻的圆盘截面（半径随时间收缩）。

    数据源同 fig8：每个时刻把 C(r,t) 插到 Cartesian 网格上、r>R(t) 处留白，
    6 盘共享同一色标（全域 min–max）与同一个 colorbar。
    """
    d = np.load(DATA_DIR / "q4_endo_fields.npz")
    t_s, r_cm = d["t_s"], d["r_cm"]
    C, R_t = d["C"], d["R_t"]
    h = t_s / 3600.0
    c_lo, c_hi = float(np.nanmin(C)), float(np.nanmax(C))

    t_show = [0.0, 5.0, 15.0, 25.0, 40.0, 50.0]
    idx = [int(np.argmin(np.abs(h - th))) for th in t_show]
    xg = np.linspace(-2.2, 2.2, 221)
    XX, YY = np.meshgrid(xg, xg)
    Rn = np.hypot(XX, YY)

    fig = plt.figure(figsize=(W_DOUBLE, 3.9))
    fig.subplots_adjust(left=0.045, right=0.885, top=0.885, bottom=0.105,
                        wspace=0.10, hspace=0.30)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")                 # 域外留白
    im = None
    for k, (th, i) in enumerate(zip(t_show, idx)):
        ax = fig.add_subplot(2, 3, k + 1)
        prof = C[i]
        ok = ~np.isnan(prof)
        Z = np.interp(Rn, r_cm[ok], prof[ok])          # 按半径插值到圆盘网格
        Z = np.where(Rn <= R_t[i], Z, np.nan)          # 域外（r>R(t)）留白
        im = ax.imshow(Z, origin="lower", extent=[xg[0], xg[-1], xg[0], xg[-1]],
                       cmap=cmap, vmin=c_lo, vmax=c_hi, interpolation="nearest")
        ax.add_patch(plt.Circle((0, 0), R_t[0], fill=False, ec="0.55", lw=0.6, ls=":"))
        ax.set_xlim(xg[0], xg[-1])
        ax.set_ylim(xg[0], xg[-1])
        ax.set_aspect("equal")
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.tick_params(labelsize=7.5, pad=1, length=0)
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("$x$ / cm", labelpad=1)
        if k % 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("$y$ / cm", labelpad=1)
        tile(ax, f"$t$={th:g} h，$R$={R_t[i]:.2f} cm")
    cax = fig.add_axes([0.905, 0.245, 0.016, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("水分浓度 $C$ / (kg/kg)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7.5)
    # 抽查：对同一帧，比较 npz 网格原值与“插值到圆盘 Cartesian 网格”的值
    i_chk, i2_chk = idx[3], idx[1]
    prof, okp = C[i_chk], ~np.isnan(C[i_chk])
    prof2, okp2 = C[i2_chk], ~np.isnan(C[i2_chk])
    chk = []
    for rr in (0.0, 0.5, 1.0):
        v_npz = float(prof[okp][int(np.argmin(np.abs(r_cm[okp] - rr)))])
        v_disk = float(np.interp(rr, r_cm[okp], prof[okp]))   # 圆盘网格用的同一插值
        chk.append(f"C({rr:.1f})={v_disk:.4f}(npz {v_npz:.4f})")
    v_srf = float(prof[okp][-1])
    v_srf5 = float(prof2[okp2][-1])
    print(f"  [F9] 6 时刻 t={t_show} h（帧 {idx}）；R: "
          f"{' '.join(f'{R_t[i]:.3f}' for i in idx)} cm；色标 {c_lo:.4f}–{c_hi:.4f} kg/kg"
          f"（=数据 min–max）；抽查 t={t_show[3]:g} h：" + "、".join(chk) +
          f"、表面 C(R)={v_srf:.4f}；t={t_show[1]:g} h 表面 C(R)={v_srf5:.4f}", flush=True)
    save(fig, "fig9_截面演化")


# ====================== F10 力学升级（重标定幂律拟合） ======================
def fig10():
    """修正后重标定的幂律力学模型对附件2 的拟合（三联：拟合 / 残差 / 平台机制）。

    数据源：03-数据/mechRefit.csv（145 点 R_data、R_pred、残差 mm）+
    mechRefit_fields.npz（重标定参数、全程场量 ū/De/g、末端平衡表 terminal）；
    口径见 03-数据/力学升级_修复重标定.md（A04/A05 修复后重标定：幂律保水
    p_c=p*(1−S)/S^β + 松弛蠕变，吸力单驱动；本图取代此前的线性检验本构情景族，
    旧文 力学升级_重标定.md 的数值一律作废）。
    """
    from mech_shrinkage import MechParams
    from solver_q1 import C0, R0

    head, rows = read_csv_rows("mechRefit.csv")
    t_s = np.array([float(r["t_s"]) for r in rows])
    h = t_s / 3600.0
    R_data = np.array([float(r["R_data_cm"]) for r in rows])
    R_pred = np.array([float(r["R_pred_cm"]) for r in rows])
    res = np.array([float(r["residual_mm"]) for r in rows])      # R_pred − R_data（mm）
    d = np.load(DATA_DIR / "mechRefit_fields.npz")
    p_star, beta = float(d["fit_params"][0]), float(d["fit_params"][1])
    t_f = float(d["t_f_h"])

    H_SEG = (3.5, 21.0)                    # 急缩 / 缓缩 / 平台 分界（力学升级_修复重标定.md §4）
    rmse = float(np.sqrt(np.mean(res ** 2)))
    seg = [float(np.sqrt(np.mean(res[(h >= a) & (h < b)] ** 2)))
           for a, b in ((0.0, H_SEG[0]), (H_SEG[0], H_SEG[1]), (H_SEG[1], 1e9))]
    r_max = float(np.max(np.abs(res)))
    k_max = int(np.argmax(np.abs(res)))
    plat_pred, plat_data = float(R_pred[-1]), float(R_data[-1])
    De_lo, De_hi = float(np.min(d["DeS"])), float(np.max(d["DeS"]))
    g_min = float(np.min(d["gS"]))
    # 末端平衡表（terminal，纯力学量）：β 网格 → 末端 $J_{end}=(R/R_0)^2$；
    # 拟合 β 处线性内插即“有限上限”对应的末端半径（力学升级_修复重标定.md §6）
    term_beta, term_J = d["terminal"][:, 0], d["terminal"][:, 1]
    J_end = float(np.interp(beta, term_beta, term_J))
    R_end = R0 * 100.0 * np.sqrt(J_end)
    # 与 力学升级_修复重标定.md §4/§6 逐项核对（总/分段 RMSE、平台、max 偏差、De、g_min、J_end）
    assert abs(rmse - 0.522) < 2e-3 and np.allclose(seg, [1.193, 0.875, 0.163],
                                                    atol=2e-3), (rmse, seg)
    assert abs(plat_pred - 1.1747) < 5e-4 and abs(plat_data - 1.198) < 5e-4
    assert abs(r_max - 1.59) < 0.01 and De_hi < 2e-3 and g_min > 0.0, (r_max, De_hi, g_min)
    assert abs(t_f - 51.1667) < 1e-3 and abs(J_end - 0.3343) < 1e-4, (t_f, J_end)
    # 失水理想基线：同一次重标定求解的 ū(t) 等容反推
    h_ideal = d["t"] / 3600.0
    R_ideal = R0 * 100.0 * np.sqrt((1.0 + d["Um"]) / (1.0 + C0))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.4),
                                        gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.95]))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.17, wspace=0.36)

    # ---- (a) 重标定拟合 ----
    for x0, x1 in ((0.0, H_SEG[0]), (H_SEG[0], H_SEG[1]), (H_SEG[1], 72.0)):
        axA.axvspan(x0, x1, color="0.94", lw=0, zorder=0)
    for xb in H_SEG:
        axA.axvline(xb, color="0.72", ls=":", lw=0.8, zorder=1)
    for x0, x1, lab in ((0.0, H_SEG[0], "急缩"), (H_SEG[0], H_SEG[1], "缓缩"),
                        (H_SEG[1], 72.0, "平台")):
        axA.text(0.5 * (x0 + x1), 2.055, lab, ha="center", va="center",
                 fontsize=7.5, color="0.35")
    axA.plot(h, R_data, color=ROLE["data"], lw=1.6, zorder=5, label="附件2 实测")
    axA.plot(h, R_pred, color=ROLE["model"], ls="--", lw=1.3,
             label=f"重标定力学模型（RMSE={rmse:.3f} mm）")
    axA.plot(h_ideal, R_ideal, color=ROLE["ref"], ls=":", lw=1.2, label="失水理想基线")
    axA.axhline(plat_data, color=ROLE["data"], ls=":", lw=0.9)
    axA.axhline(plat_pred, color=ROLE["model"], ls=":", lw=0.9)
    axA.text(70.0, 1.075, f"平台：实测 {plat_data:.3f} / 模型 {plat_pred:.4f} cm",
             ha="right", va="bottom", fontsize=7.5, color="0.30")
    axA.set_xlim(0, 72)
    axA.set_ylim(1.05, 2.10)
    axA.set_xlabel("烘干时间 $t$ / h")
    axA.set_ylabel("药材半径 $R$ / cm")
    style_axis(axA, grid="y")
    axA.legend(loc="center right", handlelength=1.8, labelspacing=0.26)
    panel(axA, "(a)", dx=-0.135)
    tile(axA, "重标定拟合：力学模型 vs 附件2")

    # ---- (b) 残差 ----
    axB.axvspan(0.0, H_SEG[1], color="0.94", lw=0, zorder=0)
    axB.axvline(H_SEG[1], color="0.72", ls=":", lw=0.8, zorder=1)
    axB.axhline(0.0, color="0.25", lw=0.8)
    axB.plot(h, res, color=ROLE["model"], lw=1.1, zorder=5)
    axB.annotate(f"前 21 h：系统性欠缩\n（峰值 {res[k_max]:+.2f} mm）",
                 xy=(h[k_max], res[k_max]), xytext=(0.30, 0.96), textcoords="axes fraction",
                 fontsize=7.5, ha="left", va="top", color="0.15",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axB.text(1.5, -1.42, f"段 RMSE {seg[0]:.3f}/{seg[1]:.3f}/{seg[2]:.3f} mm",
             ha="left", va="center", fontsize=7.5, color="0.30")
    axB.text(70.0, 0.15, "平台段贴合", ha="right", va="bottom", fontsize=7.5,
             color="0.30")
    axB.set_xlim(0, 72)
    axB.set_ylim(-1.75, 1.75)
    axB.set_xlabel("烘干时间 $t$ / h")
    axB.set_ylabel("残差 $R_{pred}-R_{data}$ / mm")
    style_axis(axB, grid="y")
    panel(axB, "(b)", dx=-0.18)
    tile(axB, "残差：结构性欠缩")

    # ---- (c) 平台机制：吸力饱和（非 De≫1 结皮冻结）----
    S = np.linspace(0.04, 1.0, 240)
    P = MechParams(rho_d0=1.0, rho_s=1468.0, p_star=p_star, ret_model="power",
                   q_pow=beta)
    pbar = np.asarray(P.pbar_c(S), float) / p_star        # 归一化平均吸力
    cap = 1.0 / ((1.0 - beta) * (2.0 - beta))    # 解析上限（有限）；β 取自 npz fit_params
    axC.plot(S, pbar, color=ROLE["model"], lw=1.4)
    axC.axhline(cap, color=ROLE["guide"], ls="--", lw=1.0)
    axC.text(0.03, cap + 0.012, f"有限上限 {cap:.3f} $p^*$", ha="left", va="bottom",
             fontsize=7.5, color="0.30")
    axC.text(0.97, 0.775, "$S\\to0$：$\\bar p_c$ 吸力饱和\n$\\Rightarrow$ 末端 $J$ 有限 → 平台",
             ha="right", va="top", fontsize=7.5, color="0.15")
    axC.text(0.03, 0.03, f"$De\\approx{De_hi:.1e}\\ll1$\n非结皮冻结", ha="left",
             va="bottom", fontsize=7.5, color="0.15",
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axC.set_xlim(0, 1.03)
    axC.set_xticks([0.0, 0.5, 1.0])
    axC.set_ylim(0, 0.78)
    axC.set_xlabel("饱和度 $S$")
    axC.set_ylabel(r"归一化平均吸力 $\bar p_c/p^*$")
    style_axis(axC, grid="y")
    panel(axC, "(c)", dx=-0.20)
    tile(axC, "平台机制：吸力饱和平衡")

    print(f"  [F10] 重标定幂律拟合：RMSE={rmse:.4f} mm（急缩 {seg[0]:.3f} / 缓缩 "
          f"{seg[1]:.3f} / 平台 {seg[2]:.3f}）；平台 pred {plat_pred:.4f} vs data "
          f"{plat_data:.3f} cm；max|残差|={r_max:.3f} mm @ {h[k_max]:.1f} h；"
          f"β={beta:.4f}、p*={p_star:.4g}；"
          f"De={De_lo:.2e}–{De_hi:.2e}（≪1，非结皮冻结）；g_min={g_min:.4f}＞0；"
          f"吸力上限 $1/[(1-β)(2-β)]p^*$={cap:.4f} $p^*$；末端平衡表（npz terminal）"
          f"内插 β={beta:.4f} → $J_{{end}}$={J_end:.4f}（$R_{{end}}\\approx{R_end:.3f}$ cm）；"
          f"t_f={t_f:.4f} h（npz t_f_h）", flush=True)
    save(fig, "fig10_力学升级")


# ====================== F11 扩散系数与特征时间尺度 ======================
def fig11():
    """(a) 三族附录物性的 D(C) 曲线（y 对数轴）＋ D4/D3 比值区间；(b) 特征时间尺度（log 横轴）。

    数据源：D 公式直接取自 02-代码/solver_q1.py（附录2）、solver_q23.py（附录3）、
    solver_q4.py（附录4），并落盘 03-数据/D_of_C.csv 供复现与论文图注引用；
    时间尺度取自 03-数据/v3_bound.csv（问题 3 口径）与 03-数据/q4_bound.csv（问题 4 口径）。
    """
    from solver_q1 import D_of_C as D2, R0, T0_K
    from solver_q23 import D_of as D3
    from solver_q4 import D_of4 as D4

    C = np.linspace(0.15, 2.55, 241)
    T_ref, T_hot = float(T0_K), 323.315
    cols = np.column_stack([C, D2(C), D3(C, T_ref), D4(C, T_ref), D3(C, T_hot), D4(C, T_hot)])
    hdr = ["# D_of_C.csv —— 三族附录物性扩散系数（fig11(a) 的唯一数据源）",
           "# 公式取自 02-代码/solver_q1.py（附录2）、solver_q23.py（附录3）、solver_q4.py（附录4）",
           "# 附录2 D=7e-9*exp(-0.89/C)（只依赖 C）；附录3 D=2.4e-3*exp(-0.45/C)*exp(-3850/T)；",
           "# 附录4 D=4.2e-4*exp(-0.30/C)*exp(-3850/T)；T 开尔文、C 干基含水率 kg/kg",
           "# T_ref=301.15 K（初温）、T_hot=323.315 K（环境温度上界）；生成：python plot_all.py fig11",
           "C_kg_kg,D_apx2_m2_s,D_apx3_Tref_m2_s,D_apx4_Tref_m2_s,D_apx3_Thot_m2_s,D_apx4_Thot_m2_s"]
    with open(DATA_DIR / "D_of_C.csv", "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(hdr) + "\n")
        for r in cols:
            f.write(",".join(f"{v:.8g}" for v in r) + "\n")

    r_at_c0 = float(D4(2.55, T_ref) / D3(2.55, T_ref))
    r_at_cmin = float(D4(0.15, T_ref) / D3(0.15, T_ref))
    _, v3 = read_csv_rows("v3_bound.csv")
    _, q4 = read_csv_rows("q4_bound.csv")
    k3 = {r["quantity"]: float(r["value"]) for r in v3}
    k4 = {r["quantity"]: float(r["value"]) for r in q4}
    t_diff3 = R0 ** 2 / (k3["D_max"] * k3["lam1"] ** 2) / 3600.0      # 现算 R²/(D_max λ₁²) /h

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.865, bottom=0.175, wspace=0.60)

    # ---- (a) D(C) 三族曲线（附录 3/4 的 D 还随 T 变，另有 ×2.4 量级——为避免遮挡不叠虚线，
    #      该倍数在 stdout 与图注里给出）----
    specs = ((D2(C), ROLE["data"], "-", "附录 2（只依赖 $C$）"),
             (D3(C, T_ref), ROLE["model"], "-", "附录 3（$T$=301.15 K）"),
             (D4(C, T_ref), ROLE["ref"], "-", "附录 4（$T$=301.15 K）"))
    for y, col, ls, lab in specs:
        axA.semilogy(C, y, color=col, ls=ls, lw=1.2, label=lab)
    axA.axvline(2.55, color="0.80", lw=0.7, ls=":")
    axA.axvline(0.15, color="0.80", lw=0.7, ls=":")
    axA.set_xlabel("干基含水率 $C$ / (kg/kg)")
    axA.set_ylabel("扩散系数 $D$ / (m$^2$/s)")
    axA.set_xlim(0, 2.75)
    axA.set_ylim(1e-13, 4e-8)
    style_axis(axA, grid="y")
    axA.legend(loc="lower right", handlelength=1.8, labelspacing=0.26)
    note(axA, f"$D_4/D_3=0.175\\,e^{{0.15/C}}$：\n1/{1/r_at_c0:.2f}（$C$=2.55）"
              f" → 1/{1/r_at_cmin:.2f}（$C$=0.15）", xy=(0.985, 0.985), fs=7.5)
    panel(axA, "(a)", dx=-0.115)
    tile(axA, "扩散系数：三族附录物性")

    # ---- (b) 特征时间尺度（log 横轴）----
    items = [("膜界参考（附 3）", k3["t_film_h"], 0),
             ("首模参考（附 3）", k3["t_ref_h"], 0),
             ("$R^2/(D_{\\max}\\lambda_1^2)$", t_diff3, 2),
             ("膜界参考（附 4）", k4["t_film_h"], 1),
             ("$R_{\\min}$ 常值（附 4）", k4["t_rmin_h"], 1),
             ("时间积分版（附 4）", k4["t_int_h"], 1),
             ("固定 $R_0$ 版（附 4）", k4["t_r0_h"], 1)]
    items.sort(key=lambda x: x[1])
    ycol = [ROLE["model"], ROLE["ref"], OI["grey"]]
    ys = np.arange(len(items))[::-1]
    axB.barh(ys, [v for _, v, _ in items], color=[ycol[k] for _, _, k in items],
             height=0.62, zorder=3)
    axB.set_yticks(ys)
    axB.set_yticklabels([n for n, _, _ in items], fontsize=7.5)
    for y, (_, v, _) in zip(ys, items):
        axB.text(v * 1.16, y, f"{v:.2f} h", va="center", ha="left", fontsize=7.5)
    axB.set_xscale("log")
    axB.set_xlim(3.0, 130)
    axB.set_xlabel("特征时间尺度 / h（对数轴）")
    style_axis(axB, grid="x")
    # 颜色语义已在 y 轴标签内（附 3 / 附 4 / 现算）：不再放图例，避免压住条形与数值标注
    panel(axB, "(b)", dx=-0.115)
    tile(axB, "特征时间尺度：口径对照")
    log_minor_off(axA)
    log_minor_off(axB, which="x")

    print(f"  [F11] D 区间：附录2 {D2(C)[0]:.3e}–{D2(C)[-1]:.3e}、附录3 {D3(C, T_ref)[0]:.3e}–"
          f"{D3(C, T_ref)[-1]:.3e}、附录4 {D4(C, T_ref)[0]:.3e}–{D4(C, T_ref)[-1]:.3e} m²/s；"
          f"D4/D3={r_at_c0:.4f}（C=2.55，1/{1/r_at_c0:.2f}）→{r_at_cmin:.4f}（C=0.15，1/{1/r_at_cmin:.2f}）；"
          f"T 由 301.15→323.32 K 的倍数（图内未叠虚线）：附录3 ×{float(D3(C, T_hot)[-1]/D3(C, T_ref)[-1]):.2f}、"
          f"附录4 ×{float(D4(C, T_hot)[-1]/D4(C, T_ref)[-1]):.2f}；"
          f"现算 R²/(D_max·λ₁²)={t_diff3:.4f} h（v3_bound: D_max={k3['D_max']:.4e}、λ₁={k3['lam1']:.4f}）；"
          f"时标 {len(items)} 条（{items[0][1]:.3f}–{items[-1][1]:.3f} h；蓝=附3口径、橙=附4口径、灰=现算）；"
          f"落盘 03-数据/D_of_C.csv", flush=True)
    save(fig, "fig11_扩散系数与时间尺度")


# ====================== F12 温度场时空云图 ======================
def fig12():
    """问题 1 温度场时空云图：x=时间、y=径向位置、色=温度（coolwarm）+ 等值线。

    数据源 03-数据/q1_fields.npz（1800 帧 × 21 径向位置，0–1800 s、0–2 cm）。
    定量读数由本图承担（精度校验见图 2 的偏差面板、收敛见图 15）。
    """
    d = np.load(DATA_DIR / "q1_fields.npz")
    t_s, pos_cm, T_C = d["times"], d["pos_cm"], d["T_C"]
    ti = np.arange(0, len(t_s), 4)                       # 1800 → 450 帧
    Z = T_C[np.ix_(ti, np.arange(len(pos_cm)))].T        # → (径向, 时间)
    lo, hi = float(Z.min()), float(Z.max())

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.4))
    fig.subplots_adjust(left=0.088, right=0.848, top=0.862, bottom=0.178)
    lv = np.linspace(lo - 1e-9, hi + 1e-9, 49)
    cf = ax.contourf(t_s[ti], pos_cm, Z, levels=lv, cmap="coolwarm", extend="both")
    cs = ax.contour(t_s[ti], pos_cm, Z, levels=8, colors="0.20", linewidths=0.45, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.1f")
    cax = fig.add_axes([0.864, 0.178, 0.020, 0.684])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("温度 $T$ / ℃", fontsize=7.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_xlabel("时间 $t$ / s")
    ax.set_ylabel("径向位置 $r$ / cm")
    ax.set_xlim(0, float(t_s[ti][-1]))
    ax.set_ylim(0, float(pos_cm[-1]))
    style_axis(ax, grid="none")
    ax.plot([0, 0], [0, 2], color="0.30", lw=0.9, ls=":")
    ax.text(60, 1.72, "中心 $r$=0（最慢）", fontsize=7.5, color="0.15")
    ax.text(1500, 0.12, "表面 $r$=2 cm（最先升温）", fontsize=7.5, color="0.15", ha="right")
    panel(ax, "(a)", dx=-0.075)
    tile(ax, "温度场：0.5 h 内近准稳态")

    print(f"  [F12] 问题1 温度场云图：$t$ 0–{t_s[ti][-1]/60:.1f} min、$r$ 0–{pos_cm[-1]:.2f} cm、"
          f"{len(ti)}×{len(pos_cm)} 网格；$T$ 全域 {lo:.3f}–{hi:.3f} ℃；"
          f"中心 1800 s={Z[0, -1]:.4f} ℃、表面 1800 s={Z[-1, -1]:.4f} ℃；"
          f"口径：colorbar 温度(℃) + 8 条等温线；定量精度见图 2", flush=True)
    save(fig, "fig12_温度场时空云图")


# ====================== F13 水分场时空云图 ======================
def fig13():
    """问题 2/3 水分场时空云图：x=时间(h)、y=径向位置(cm)、色=干基含水率（viridis）
    + 0.15 达标等值线（壳核分界）+ 壳核结构标注。

    数据源 03-数据/q23_main_steps.npz（t、C；固定 R=2 cm 口径）。
    """
    d = np.load(DATA_DIR / "q23_main_steps.npz")
    t_h = d["t"] / 3600.0
    C = d["C"]
    pos_cm = np.arange(C.shape[1]) * 0.1
    ti = np.arange(0, len(t_h), 60)                      # 26130 → 436 帧
    Z = C[np.ix_(ti, np.arange(len(pos_cm)))].T
    tf_h = float(read_json("q23_summary.json")["tf_h"])

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.4))
    fig.subplots_adjust(left=0.088, right=0.848, top=0.862, bottom=0.178)
    cf = ax.contourf(t_h[ti], pos_cm, Z, levels=np.linspace(0.0, 2.6, 53), cmap=TIME_CMAP)
    cs = ax.contour(t_h[ti], pos_cm, Z, levels=[TH], colors="white", linewidths=1.0)
    ax.clabel(cs, fmt=f"{TH:.2f}", fontsize=7.5, colors="white")
    cax = fig.add_axes([0.864, 0.178, 0.020, 0.684])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("干基含水率 $C$ / (kg/kg)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.axvline(tf_h, color=ROLE["ref"], lw=1.0, ls="--")
    ax.text(tf_h - 1.2, 0.10, f"$t_f$={tf_h:.2f} h", fontsize=7.5, color=ROLE["ref"],
            ha="right", va="bottom")
    ax.text(46.0, 1.60, "干燥壳\n（$C$<0.15）", fontsize=7.5, color="white", ha="center")
    ax.text(46.0, 0.42, "湿核\n（$C\\geq$0.15）", fontsize=7.5, color="0.85", ha="center")
    ax.set_xlabel("时间 $t$ / h")
    ax.set_ylabel("径向位置 $r$ / cm")
    ax.set_xlim(0, float(t_h[ti][-1]))
    ax.set_ylim(0, 2.02)
    style_axis(ax, grid="none")
    panel(ax, "(a)", dx=-0.075)
    tile(ax, "水分场：干燥壳-湿核结构")

    i_tf = int(np.argmin(np.abs(t_h - tf_h)))
    shell = float(pos_cm[np.argmax(C[i_tf] < TH)]) if bool(np.any(C[i_tf] < TH)) else float("nan")
    print(f"  [F13] 问题2/3 水分场云图：$t$ 0–{t_h[ti][-1]:.2f} h、$r$ 0–{pos_cm[-1]:.2f} cm、"
          f"{len(ti)}×{len(pos_cm)} 网格；$C$ 全域 {float(np.nanmin(Z)):.4f}–{float(np.nanmax(Z)):.4f} kg/kg；"
          f"$t_f$={tf_h:.4f} h 时 $C$<0.15 的壳起点 $r$={shell:.2f} cm（壳厚 {pos_cm[-1]-shell:.2f} cm）；"
          f"中心末值 {Z[0, -1]:.4f}、表面末值 {Z[-1, -1]:.4f} kg/kg", flush=True)
    save(fig, "fig13_水分场时空云图")


# ====================== F14 表面通量与表面扩散系数 ======================
def fig14():
    """表面水分通量与表面扩散系数（问题 1，附录 2 物性）。

    (a) 表面水通量 j_w(t)；(b) 表面扩散系数 D_s(t)（与 (a) 共享时间轴，不用双 Y 轴）；
    (c) D_s 随表面含水率 C_s 的变化。
    数据源 03-数据/q1_fields.npz（Cs/Ts/env_*）+ 公式现算，并落盘 03-数据/surface_flux.csv。
    口径：j_w=ρ_d,s·h_m·(U_s−C_env)（正=脱湿）、ρ_d,s=(650+128·U_s)/(1+U_s)、D_s=D_附2(C_s)。
    """
    from solver_q1 import HM, D_of_C as D2, R0
    d = np.load(DATA_DIR / "q1_fields.npz")
    t_s = d["times"]
    Cs, Ts = d["Cs"], d["Ts"]
    env_t, env_C = d["env_t"], d["env_C"]
    C_env = np.interp(t_s, env_t, env_C)
    rho_ds = (650.0 + 128.0 * Cs) / (1.0 + Cs)
    j_w = rho_ds * HM * (Cs - C_env)
    D_s = D2(Cs)
    out = np.column_stack([t_s, Cs, Ts, C_env, rho_ds, j_w, D_s])
    hdr = ["# surface_flux.csv —— 问题 1 表面量分项（fig14 的唯一数据源）",
           "# 口径：j_w=rho_d,s*h_m*(U_s-C_env) [kg 水/(m²·s)]，正=脱湿；h_m=8e-7 m/s（附录2 延用）",
           "# rho_d,s=(650+128*U_s)/(1+U_s) [kg/m³]（附录2 湿密度 ρ=650+128C 除以 1+C，同 latent_scenario.py 口径）",
           "# D_s=D_附2(C_s)=7e-9*exp(-0.89/C_s) [m²/s]；C_env 为附件1 环境水分（线性插值到 1 s 网格）",
           "# 生成：python plot_all.py fig14",
           "t_s,Cs_kg_kg,Ts_K,rho_d_s_kg_m3,j_w_kg_m2_s,D_s_m2_s"]
    with open(DATA_DIR / "surface_flux.csv", "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(hdr[:5]) + "\n" + hdr[5] + ",C_env_kg_kg\n")
        for r in out:
            f.write(f"{r[0]:.0f},{r[1]:.6f},{r[2]:.4f},{r[4]:.4f},{r[5]:.6e},{r[6]:.6e},{r[3]:.6f}\n")

    fig = plt.figure(figsize=(W_DOUBLE, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], wspace=0.34, hspace=0.16,
                          left=0.135, right=0.985, top=0.865, bottom=0.175)
    axJ = fig.add_subplot(gs[0, 0])
    axD = fig.add_subplot(gs[1, 0], sharex=axJ)
    axC = fig.add_subplot(gs[:, 1])
    axJ.plot(t_s / 60.0, j_w, color=ROLE["model"], lw=1.2)
    axJ.set_ylabel("$j_w$ / (kg/(m$^2$·s))")
    axJ.tick_params(labelbottom=False)
    style_axis(axJ, grid="y")
    tile(axJ, "表面水通量：0.5 h 内峰值")
    panel(axJ, "(a)", dx=-0.145)
    axD.plot(t_s / 60.0, D_s / 1e-9, color=ROLE["ref"], lw=1.2)     # 以 1e-9 为单位显示
    axD.set_ylabel("$D_s$ / ($10^{-9}$ m$^2$/s)")
    axD.set_xlabel("时间 $t$ / min")
    style_axis(axD, grid="y")
    axC.plot(Cs, D_s / 1e-9, color=ROLE["ref"], lw=1.2)
    axC.plot(Cs[::120], D_s[::120] / 1e-9, ls="none", marker="o", ms=2.6, mfc="white",
             mec=ROLE["ref"], mew=0.8)
    axC.set_xlabel("表面含水率 $C_s$ / (kg/kg)")
    axC.set_ylabel("$D_s$ / ($10^{-9}$ m$^2$/s)")
    style_axis(axC, grid="y")
    panel(axC, "(b)", dx=-0.135)
    tile(axC, "表面扩散系数随含水率")
    axC.annotate(f"末值 $D_s$={D_s[-1]/1e-9:.2f}x$10^{{-9}}$", xy=(Cs[-1], D_s[-1] / 1e-9),
                 xytext=(0.96, 0.06), textcoords="axes fraction", ha="right", va="bottom",
                 fontsize=7.5, color="0.25",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.45"))

    print(f"  [F14] 问题1 表面量：$j_w$ {float(j_w.min()):.3e}–{float(j_w.max()):.3e} kg/(m²·s)"
          f"（峰值 {t_s[int(np.argmax(j_w))]/60:.2f} min）；$D_s$ {float(D_s.min()):.3e}–"
          f"{float(D_s.max()):.3e} m²/s（$C_s$ {float(Cs.min()):.4f}→{float(Cs.max()):.4f}）；"
          f"$\\rho_{{d,s}}$ {float(rho_ds.min()):.1f}–{float(rho_ds.max()):.1f} kg/m³；"
          f"$C_{{env}}$ {float(C_env.min()):.4f}–{float(C_env.max()):.4f} kg/kg；"
          f"落盘 03-数据/surface_flux.csv（{len(t_s)} 行）", flush=True)
    save(fig, "fig14_表面通量与表面扩散系数")


# ====================== F15 收敛与守恒检验 ======================
def fig15():
    """(a) 问题 1 温度场误差收敛（双对数 + 二阶参考斜率）；(b) t_f 网格收敛（双对数）；
    (c) 守恒相对残差随网格加密（三问题：水量 / 能量）。

    数据源：v1_convergence.csv、tf_convergence.csv、q4_endo_convergence.csv、
    conservation.csv（问题1）、conservation23.csv（问题2·3）、q4_conservation.csv（问题4）。
    """
    head, conv = read_csv_rows("v1_convergence.csv")
    rows = [c for c in conv if c[head[0]] != "order(fit)"]
    Ns = np.array([float(c[head[0]]) for c in rows])
    errs = np.array([[float(c[h]) for h in head[1:]] for c in rows])
    order_row = [c for c in conv if c[head[0]] == "order(fit)"][0]
    orders = [float(order_row[h]) for h in head[1:]]
    _, tf3 = read_csv_rows("tf_convergence.csv")
    _, tf4 = read_csv_rows("q4_endo_convergence.csv")
    d3 = {r["case"]: float(r["t_f_h"]) for r in tf3}
    d4 = {r["case"]: float(r["t_f_h"]) for r in tf4}
    N3 = [40, 80, 160]
    e3 = [abs(d3[f"N={n}"] - d3["richardson_extrap"]) for n in N3]
    e4 = [abs(d4[f"N={n}"] - d4["richardson_extrap"]) for n in N3]
    _, cons1 = read_csv_rows("conservation.csv")
    _, cons23 = read_csv_rows("conservation23.csv")
    _, cons4 = read_csv_rows("q4_conservation.csv")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.4),
                                        gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.06]))
    fig.subplots_adjust(left=0.083, right=0.985, top=0.865, bottom=0.185, wspace=0.34)
    mk = ["o", "s", "^"]
    for k in range(errs.shape[1]):
        col = TIME_CMAP(0.12 + 0.72 * k / max(errs.shape[1] - 1, 1))
        axA.loglog(Ns, errs[:, k], color=col, marker=mk[k], ms=3.0, lw=1.1,
                   label=f"$t$={[100, 600, 1800][k]} s（阶 {orders[k]:.2f}）")
    axA.loglog(Ns, errs[0, 0] * (Ns / Ns[0]) ** -2.0, color=ROLE["guide"], ls="--", lw=1.0,
               label="斜率 −2（二阶参考）")
    axA.set_xlabel("网格数 $N$")
    axA.set_ylabel("最大偏差 / ℃")
    axA.set_xticks([20, 40, 80, 160, 320])
    axA.set_xticklabels(["20", "40", "80", "160", "320"])
    axA.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style_axis(axA, grid="both")
    axA.legend(loc="lower left", handlelength=1.6, labelspacing=0.24)
    panel(axA, "(a)", dx=-0.185)
    tile(axA, "收敛：温度场误差")

    for ns, es, col, lab, m in ((N3, e3, ROLE["model"], "问题 3（固定 $R$）", "o"),
                                (N3, e4, ROLE["ref"], "问题 4（内生 $R(t)$）", "s")):
        axB.loglog(ns, es, color=col, marker=m, ms=3.4, lw=1.2, label=lab)
    axB.loglog(N3, e3[-1] * (np.array(N3) / 160.0) ** -1.0, color=ROLE["guide"], ls="--",
               lw=1.0, label="斜率 −1（一阶参考）")
    axB.set_xlabel("网格数 $N$")
    axB.set_ylabel("$|t_f(N)-t_f^{\\mathrm{Rich}}|$ / h")
    axB.set_xticks(N3)
    axB.set_xticklabels([str(n) for n in N3])
    axB.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style_axis(axB, grid="both")
    axB.legend(loc="lower left", handlelength=1.6, labelspacing=0.24)
    panel(axB, "(b)", dx=-0.185)
    tile(axB, "收敛：$t_f$ 随网格")

    series = [([float(r["N"]) for r in cons1], [float(r["max_rel_residual"]) for r in cons1],
               "水量/能量（问题 1）", ROLE["data"], "o"),
              ([float(r["N"]) for r in cons23],
               [float(r["max_rel_residual_water"]) for r in cons23], "水量（问题 2·3）",
               ROLE["model"], "s"),
              ([float(r["N"]) for r in cons23],
               [float(r["glob_rel_residual_enthalpy"]) for r in cons23], "能量（问题 2·3）",
               ROLE["aux1"], "s"),
              ([float(r["N"]) for r in cons4],
               [float(r["max_rel_residual_water"]) for r in cons4], "水量（问题 4）",
               ROLE["ref"], "^"),
              ([float(r["N"]) for r in cons4],
               [float(r["glob_rel_residual_enthalpy"]) for r in cons4], "能量（问题 4）",
               ROLE["aux2"], "^")]
    for xs, ys_, lab, col, m in series:
        axC.loglog(xs, ys_, color=col, marker=m, ms=3.2, lw=1.1, label=lab)
    axC.axhline(1e-10, color=ROLE["guide"], ls=":", lw=0.9)
    axC.text(0.975, 0.86, "机器精度量级 $10^{-10}$", transform=axC.transAxes, ha="right",
             fontsize=7.5, color="0.35")
    axC.set_xticks([40, 80, 160])
    axC.set_xticklabels(["40", "80", "160"])
    axC.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axC.set_xlabel("网格数 $N$")
    axC.set_ylabel("相对残差（各网格最大值）")
    style_axis(axC, grid="both")
    axC.legend(loc="lower left", handlelength=1.4, labelspacing=0.22, fontsize=7.5)
    panel(axC, "(c)", dx=-0.165)
    tile(axC, "守恒：残差随网格加密")
    for _ax in (axA, axB, axC):
        log_minor_off(_ax)

    print(f"  [F15] 收敛：问题1 阶 {orders}（N=20→320，5 点）；问题3 |Δt_f| "
          f"{e3[0]:.4f}/{e3[1]:.4f}/{e3[2]:.4f} h、问题4 {e4[0]:.5f}/{e4[1]:.5f}/{e4[2]:.6f} h"
          f"（对 Richardson 外推）；守恒：问题1 max {max(float(r['max_rel_residual']) for r in cons1):.2e}、"
          f"问题2·3 水量 max {max(float(r['max_rel_residual_water']) for r in cons23):.2e} / 能量 "
          f"{max(float(r['glob_rel_residual_enthalpy']) for r in cons23):.2e}、问题4 水量 max "
          f"{max(float(r['max_rel_residual_water']) for r in cons4):.2e} / 能量 "
          f"{max(float(r['glob_rel_residual_enthalpy']) for r in cons4):.2e}"
          f"（conservation23.csv 的 enthalpy_window 列为逐时窗口口径，非残差，不入图）", flush=True)
    save(fig, "fig15_收敛与守恒检验")


# ====================== F16 中心表面演化与半径收缩 ======================
def fig16():
    """(a) 中心/表面含水率演化 + 0.15 阈值 + t_f 标注；(b) 半径收缩 + 附件 2 实测对照。

    数据源 03-数据/q4_endo_fields.npz（t_s/Cc/Cs/R_t）、q4_endo_eval.csv（t_f）、
    附件 2（01-题目/原始文件/附件2.xlsx，实测 R）。
    """
    d = np.load(DATA_DIR / "q4_endo_fields.npz")
    h = d["t_s"] / 3600.0
    Cc, Cs, R_t = d["Cc"], d["Cs"], d["R_t"]
    _, ev = read_csv_rows("q4_endo_eval.csv")
    kv = {r["item"]: float(r["value"]) for r in ev}
    tf_h = kv["tf_first_crossing_h"]
    t2, R2 = read_att2()
    h2 = t2 / 3600.0

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(W_SINGLE, 3.30), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.28, 1.0]))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.885, bottom=0.155, hspace=0.22)
    axA.plot(h, Cc, color=ROLE["model"], lw=1.2, label="中心 $s$=0")
    axA.plot(h, Cs, color=ROLE["data"], lw=1.2, ls="--", label="表面 $s$=1")
    axA.axhline(TH, color=ROLE["guide"], ls=":", lw=1.1)
    axA.axvline(tf_h, color=ROLE["ref"], ls="--", lw=1.0)
    axA.text(tf_h + 0.7, 2.42, f"$t_f$={tf_h:.4f} h", fontsize=7.5, color=ROLE["ref"],
             va="top", ha="left")
    axA.text(0.6, TH + 0.10, f"阈值 {TH} kg/kg", fontsize=7.5, color="0.30", ha="left")
    axA.set_ylabel("干基含水率 $C$ / (kg/kg)")
    axA.set_ylim(0.0, 2.75)
    axA.set_xlim(0, 72)
    style_axis(axA, grid="y")
    axA.legend(loc="upper right", handlelength=1.8, labelspacing=0.26)
    panel(axA, "(a)", dx=-0.115)
    tile(axA, "中心/表面含水率演化（内生主解）")

    axB.plot(h2, R2, ls="none", marker="o", ms=2.6, markevery=3, mfc="white",
             mec=ROLE["data"], mew=0.8, label="附件 2 实测（每 3 点标一次）")
    axB.plot(h, R_t, color=ROLE["model2"], lw=1.3, label="内生闭合预测 $R(t)$")
    axB.axvline(tf_h, color=ROLE["ref"], ls="--", lw=1.0)
    axB.set_xlabel("时间 $t$ / h")
    axB.set_ylabel("药材半径 $R$ / cm")
    axB.set_ylim(1.14, 2.06)
    style_axis(axB, grid="y")
    axB.legend(loc="upper right", handlelength=1.8, labelspacing=0.26)
    panel(axB, "(b)", dx=-0.115)
    tile(axB, "半径收缩：预测 vs 实测")

    i_tf = int(np.argmin(np.abs(h - tf_h)))
    ov = (h2 >= h[0]) & (h2 <= h[-1])            # 只在预测覆盖的 0–50.54 h 窗口内比
    rmse = float(np.sqrt(np.mean((np.interp(h2[ov], h, R_t) - R2[ov]) ** 2)))
    print(f"  [F16] $t_f$={tf_h:.6f} h（q4_endo_eval.csv）；$t_f$ 时 中心 {Cc[i_tf]:.4f} / 表面 "
          f"{Cs[i_tf]:.4f} kg/kg；$R$ {R_t[0]:.4f}→$t_f$ {R_t[i_tf]:.4f}→末 {R_t[-1]:.4f} cm"
          f"（{h[-1]:.2f} h）；与附件 2 实测散点（{int(ov.sum())} 点重叠段）RMSE={rmse:.4f} cm；"
          f"表面在 {float(h[np.argmax(Cs < TH)]):.2f} h 首次低于阈值", flush=True)
    save(fig, "fig16_中心表面演化与半径收缩")


# ====================== F17 几何模型与边界条件示意图 ======================
def fig17():
    """几何模型与边界条件示意图（无数据曲线，纯 matplotlib patches 矢量绘制）。

    口径常数取自 02-代码/solver_q1.py：$R_0$=0.02 m、$L$=0.25 m、$h$=25 W/(m²·K)、
    $h_m$=8e-7 m/s、$T_0$=301.15 K（28 ℃）、$U_0$=2.55 kg/kg（题目给定与声明假设）。
    """
    from solver_q1 import R0 as R0_M, H, HM, T0_K, C0
    L_CM = 25.0                     # 题目给定：药材长 25 cm（一维径向模型只用 L/R 比，见论文表 2）

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.3),
                                   gridspec_kw=dict(width_ratios=[1.0, 2.15]))
    fig.subplots_adjust(left=0.035, right=0.975, top=0.905, bottom=0.075, wspace=0.10)

    # ---- (a) 横截面 ----
    axA.set_aspect("equal")
    axA.add_patch(plt.Circle((0, 0), 2.0, fc="#eef3f8", ec=ROLE["model"], lw=1.3))
    axA.plot([0], [0], marker="+", ms=7, mew=1.2, color=ROLE["data"])
    axA.annotate("", xy=(2.0, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ROLE["data"]))
    axA.text(1.05, 0.12, "$R_0$=2 cm", fontsize=8)
    for ang, lab in ((90, "$h$, $h_m$"), (0, "$h$, $h_m$"), (270, "$h$, $h_m$"), (180, "$h$, $h_m$")):
        a = np.deg2rad(ang)
        axA.annotate("", xy=(3.05 * np.cos(a), 3.05 * np.sin(a)),
                     xytext=(2.05 * np.cos(a), 2.05 * np.sin(a)),
                     arrowprops=dict(arrowstyle="-|>", lw=0.9, color=ROLE["ref"]))
    axA.text(0, 3.62, "环境 $T_a(t)$、$C_{\\mathrm{env}}(t)$", fontsize=8, ha="center",
             va="center", color=ROLE["ref"])
    axA.text(0, -3.72, "对称轴 $r$=0（$\\partial_r$=0）", fontsize=7.5, ha="center",
             va="center", color="0.30")
    axA.set_xlim(-3.6, 3.6)
    axA.set_ylim(-4.3, 4.3)
    axA.axis("off")
    panel(axA, "(a)", dx=0.0)
    tile(axA, "横截面：一维径向")
    axA.text(-3.55, 2.75, f"$h$={H:g} W/(m$^2$·K)\n$h_m$={HM:g} m/s", fontsize=7.5,
             va="center", ha="left", color="0.20")

    # ---- (b) 轴向纵剖面（柱体横放：水平向＝柱长 L，竖向＝直径 2R₀）----
    axB.set_aspect("equal")
    axB.add_patch(Rectangle((0, -2), L_CM, 4, fc="#eef3f8", ec=ROLE["model"], lw=1.3))
    # 竖向尺寸线（柱体右侧）：直径 2R₀
    axB.annotate("", xy=(L_CM + 0.9, -2), xytext=(L_CM + 0.9, 2),
                 arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="0.35"))
    axB.text(L_CM + 0.9, 2.45, "$2R_0$=4 cm", fontsize=8, ha="center", va="bottom")
    # 水平尺寸线（柱体下方）：柱长 L，两端画延伸线
    for xe in (0.0, L_CM):
        axB.plot([xe, xe], [-2.0, -3.15], color="0.62", lw=0.6, ls=":", zorder=1)
    axB.annotate("", xy=(0, -3.0), xytext=(L_CM, -3.0),
                 arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="0.35"))
    axB.text(0.5 * L_CM, -3.35, f"$L$={L_CM:.0f} cm", fontsize=8, ha="center", va="top")
    for x in (3.0, 8.0, 13.0, 18.0, 22.0):   # 沿轴向的代表性对流箭头
        for sgn in (1,):
            axB.annotate("", xy=(x, sgn * 3.0), xytext=(x, sgn * 2.05),
                         arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ROLE["ref"]))
    axB.text(12.5, 3.75, "环境 $T_a(t)$、$C_{\\mathrm{env}}(t)$（附件 1）", fontsize=8,
             ha="center", color=ROLE["ref"])
    axB.text(12.5, 0.78, "$\u03c1(C)$、$c_p(C)$、$k(C)$、$D(C,T)$（附录 2/3/4 分问选用）",
             fontsize=7.5, ha="center", color=ROLE["model"])
    axB.text(12.5, -0.32, "干基含水率 $U$，温度 $T$；$r=R$ 处 Robin 边界",
             fontsize=7.5, ha="center", color="0.25")
    axB.text(12.5, -1.45, "初始：$T_0$=28 ℃、$U_0$=2.55 kg/kg（均匀）", fontsize=7.5,
             ha="center", color="0.25")
    axB.set_xlim(-4.2, 31.5)
    axB.set_ylim(-4.6, 4.6)
    axB.axis("off")
    panel(axB, "(b)", dx=0.0)
    tile(axB, "轴向：长柱、表面对流")

    print(f"  [F17] 示意图常数：$R_0$={R0_M*100:.1f} cm、$L$={L_CM:.0f} cm（题目给定）、"
          f"$h$={H:g} W/(m²·K)、$h_m$={HM:g} m/s、$T_0$={T0_K-273.15:.1f} ℃、"
          f"$U_0$={C0} kg/kg（solver_q1.py；无数据曲线）", flush=True)
    save(fig, "fig17_几何模型与边界条件示意图")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig5": fig5, "fig6": fig6,
        "fig7": fig7, "fig8": fig8, "fig9": fig9, "fig10": fig10,
        "fig11": fig11, "fig12": fig12, "fig13": fig13, "fig14": fig14,
        "fig15": fig15, "fig16": fig16, "fig17": fig17}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", default=[], help="只出指定图，如 fig4")
    ap.add_argument("--no-cache", action="store_true", help="fig4 对照曲线强制重算")
    a = ap.parse_args()
    plt.rcParams.update(RC)
    want = a.only or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8",
                      "fig9", "fig10", "fig11", "fig12", "fig13", "fig14", "fig15",
                      "fig16", "fig17"]
    for k in want:
        print(f"[{k}] 作图 …", flush=True)
        if k == "fig4":
            fig4(no_cache=a.no_cache)
        else:
            FIGS[k]()

    print("\n[审计汇总] 图 | 设计尺寸(in) | 纵横比 | 最小字号(pt) | 标题≤28 | 图例项 | 刻度重叠 | "
          "柱基线 | 裁切文字 | 图内文字块 | colorbar(须带label) | 3D面板 | 嵌入位图 | 密集标记 | 结论")
    for r in AUDIT:
        print(f"  {r['stem']:<22} {r['w']:.1f}×{r['h']:.1f} | {r['aspect']:.2f} | "
              f"{r['min_font']:.1f} | {max((len(t) for t in r['titles']), default=0)} | "
              f"{r['legends']} | {len(r['tick_bad'])} | {len(r['bar_base'])} | "
              f"{len(r['clipped'])} | {len(r['figtext'])} | {r['cb']} | "
              f"{r['ax3d']} | {r['embed_img']} | {r['dense_marker']} | "
              f"{'PASS' if r['ok'] else 'FAIL'}", flush=True)
    bad = [r for r in AUDIT if not r["ok"]]
    if GLYPH.msgs:
        print("[字体] 缺字告警如下（应视为失败）：", flush=True)
        for m in GLYPH.msgs:
            print("   ", m, flush=True)
    else:
        print("[字体] 无缺字告警（logging + warnings 双通道）；PDF 嵌入 TrueType 子集"
              "（pdf.fonttype=42）", flush=True)
    print(f"[灰度] 灰度预览见 05-图表/_qa/*_gray.png（逐张目视核对“不靠颜色可区分”）", flush=True)
    if GLYPH.msgs or bad:
        print(f"[门禁] FAIL：缺字 {len(GLYPH.msgs)} 条，审计不合格 {len(bad)} 张 "
              f"({', '.join(r['stem'] for r in bad)})", flush=True)
        sys.exit(1)
    print(f"[门禁] PASS：{len(AUDIT)} 张图审计全部通过", flush=True)


if __name__ == "__main__":
    main()
