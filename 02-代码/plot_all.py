"""M5 图表生成（出版级样式重构版）：从 03-数据/ 的唯一数据源出 6 张论文用矢量图。

用法：  python plot_all.py            # 一次出全部 7 张（PDF + 300dpi PNG + 灰度预览）
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
  fig4  q23_main_steps.npz（um）、q4_main_steps.npz（um）、q4_split.csv、q4_bound.csv、
        v3_bound.csv、sensitivity.csv、结果总账.csv；附4固定R对照曲线由 solver_q4 现算
        （与 q4_split.csv 的 appendix4_fixedR 逐次核对）
  fig5  sensitivity.csv、error_budget.csv
  fig6  endogenous_shrinkage.csv（附件2 时刻网格上的各闭合 R_pred、Ū_model、Ū_inf）；
        缺列时回退 附件2.xlsx、q4_main_steps.npz（见 03-数据/内生收缩分析.md）
  fig7  endogenous_calibrated.csv（附件2 网格上的 R_data、R_pred、残差、失水基线；
        见 03-数据/内生收缩模型.md）

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
    """面板标签 (a)/(b)：粗体小写，左上外侧。"""
    ax.text(dx, 1.03, tag, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color="0.10")


def tile(ax, title):
    """面板标题（≤TITLE_MAX 字符；长论述进图注）。"""
    assert len(title) <= TITLE_MAX, f"标题过长({len(title)}): {title}"
    ax.set_title(title, fontsize=8.5, pad=4)


def note(ax, text, xy=(0.98, 0.97), ha="right", va="top", fs=7.5):
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.80", lw=0.5, alpha=0.95))


def fit_left(fig, axes, pad_px=6.0):
    """按 y 轴刻度文字 + ylabel 的实测宽度加大左边距，防止长中文类别名被裁切
    （bbox 出图为精确 figsize，不裁切，故必须自己留够边距）。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    over = 0.0
    for ax in np.atleast_1d(axes).ravel():
        labs = [t for t in ax.get_yticklabels() if t.get_text().strip()]
        if not labs:
            continue
        left_px = min([t.get_window_extent(ren).x0 for t in labs]
                      + ([ax.yaxis.label.get_window_extent(ren).x0]
                         if ax.get_ylabel() else []))
        over = max(over, ax.get_window_extent(ren).x0 - left_px + pad_px)
    fig.subplots_adjust(left=fig.subplotpars.left + over / (fig.get_size_inches()[0]
                                                            * fig.dpi))
    return fig.subplotpars.left


# ====================== 审计 ======================
AUDIT: list[dict] = []


def tick_texts(axis):
    """仅返回**视图范围内**可见的刻度文字（get_xticklabels 会带上范围外、
    尺寸为 None 的隐藏刻度，拿去判裁切会得到假阳性）。"""
    lo, hi = sorted(axis.get_view_interval())
    out = []
    for tick in axis.get_major_ticks():
        loc = float(tick.get_loc())
        if lo - 1e-9 <= loc <= hi + 1e-9:
            labs = [tick.label1] + ([tick.label2] if tick.label2.get_text().strip() else [])
            out += [t for t in labs if t.get_visible()]
    return out


def drawn_texts(fig):
    """图中**实际绘制**的文字（白名单遍历）——fig.findobj(Text) 会连带返回
    刻度/数学排版留下的孤立 Text（axes=None），不能用来判裁切与字号。"""
    out = list(fig.texts)
    for ax in fig.axes:
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
               clipped=[], cap_hit=[])
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
        for ax in fig.axes:
            lab = ax.xaxis.label
            if lab.get_text().strip() and _overlap(tb, lab.get_window_extent(renderer=ren),
                                                   tol=0.0):
                rec["cap_hit"].append(f"{t.get_text()[:12]}↔{lab.get_text()[:12]}")

    for ax in fig.axes:
        if ax.get_title():
            rec["titles"].append(ax.get_title())
        lg = ax.get_legend()
        if lg is not None:
            rec["legends"].append(len(lg.get_texts()))
        if ax.get_label() == "<colorbar>":
            rec["cb"] += 1
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
    # 文字裁切检查：bbox=None 精确出图，超出画布的文字会被静默切掉
    ren = fig.canvas.get_renderer()
    fw, fh = float(ren.width), float(ren.height)
    rec["clipped"] = []
    for t in texts:
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
    rec["ok"] = (rec["font_ok"] and rec["title_ok"] and rec["legend_ok"] and rec["tick_ok"]
                 and rec["bar_ok"] and rec["clip_ok"] and not rec["figtext"]
                 and rec["cb"] == 0 and rec["ax3d"] == 0
                 and rec["embed_img"] == 0 and rec["dense_marker"] == 0
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
    axR.plot(h2, R, color=ROLE["data"], lw=1.5, label="附件2 实测 $R(t)$")
    axR.plot([h2[-1], x_end], [R[-1]] * 2, color=ROLE["data"], ls="--", lw=1.1)
    axR.axvspan(t_frz, h2[-1], color="0.93", lw=0, zorder=0)
    axR.axvspan(h2[-1], x_end, color="0.96", lw=0, zorder=0)
    axR.axvline(t_frz, color=ROLE["guide"], ls=":", lw=0.9)
    axR.axvline(h2[-1], color=ROLE["guide"], ls=":", lw=0.7)
    axR.axhline(R[-1], color=ROLE["guide"], ls=":", lw=0.9)
    axR.annotate(f"$t$={t_frz:.0f} h：$R$ 首达 {R[i_frz]:.3f} cm\n此后基本冻结",
                 xy=(t_frz, R[i_frz]), xytext=(0.30, 0.72), textcoords="axes fraction",
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
    tile(axR, "附件2：半径收缩与冻结")

    print(f"  [F1] 附件1 {len(t1)} 点 0–{t_cut:.0f} h；附件2 {len(t2)} 点 0–{h2[-1]:.0f} h，"
          f"R {R[0]:.3f}→{R[-1]:.3f} cm，冻结点 {t_frz:.0f} h"
          f"（{t_frz:.0f}–{h2[-1]:.0f} h 仅降 {R[i_frz]-R[-1]:.3f} cm）", flush=True)
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

    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.2))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.875, bottom=0.17)
    mk = ["o", "s", "^"]
    handles = []
    for k, ts in enumerate(t_show):
        col = TIME_CMAP(0.15 + 0.62 * k / max(len(t_show) - 1, 1))
        idx = int(np.where(times_arr == ts)[0][0])
        ax.plot(r_dense * 100, T_ana[k], color=col, lw=1.3, ls="-")
        ax.plot(pos_cm, T_C[idx], ls="none", marker=mk[k], ms=3.6, mfc="white",
                mec=col, mew=0.9)
        handles.append(Line2D([], [], color=col, marker=mk[k], mfc="white", mec=col,
                              lw=1.3, ms=3.6, label=f"$t$={ts} s"))
    ax.set_xlabel("到药材中心的距离 $r$ / cm")
    ax.set_ylabel("药材温度 $T$ / ℃")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(27.4, 45.8)
    style_axis(ax, grid="y")
    ax.legend(handles=handles, loc="upper left", handlelength=1.8,
              title="实线 解析解 / 空心点 数值解", title_fontsize=7.5)
    panel(ax, "(a)", dx=-0.115)
    tile(ax, "问题1 温度场：解析 vs 数值")

    ins = ax.inset_axes([0.415, 0.475, 0.565, 0.455])
    for k in range(errs.shape[1]):
        col = TIME_CMAP(0.15 + 0.62 * k / max(errs.shape[1] - 1, 1))
        ins.loglog(Ns, errs[:, k], color=col, marker=mk[k], ms=3.0, lw=1.1,
                   label=f"$t$={t_show[k]} s，阶 {orders[k]:.3f}")
    ref = errs[0, 0] * (Ns / Ns[0]) ** -2.0
    ins.loglog(Ns, ref, color=ROLE["guide"], ls="--", lw=1.0, label="斜率 −2（二阶参考）")
    ins.set_xlabel("网格数 $N$", fontsize=7.5, labelpad=0.5)
    ins.set_ylabel("最大偏差 / ℃", fontsize=7.5, labelpad=1.0)
    ins.tick_params(labelsize=7.5, pad=1.5)
    ins.grid(True, which="both", **GRID_KW)
    ins.set_axisbelow(True)
    ins.legend(loc="lower left", fontsize=7.5, handlelength=1.6, labelspacing=0.22,
               borderpad=0.25)
    print(f"  [F2] 解析对拍最大偏差 {dev_max:.3e} ℃（四位小数分辨率 5e-05 ℃，低 "
          f"{5e-5/dev_max:.0f} 倍），收敛阶 {orders}", flush=True)
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

    fig, (axT, axC) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4))
    fig.subplots_adjust(left=0.082, right=0.982, top=0.865, bottom=0.155, wspace=0.26)
    legT, legC = [], []
    for k, h in enumerate(t_T):
        col = TIME_CMAP(0.95 - 0.80 * k / (len(t_T) - 1))
        axT.plot(pos_cm, prof(T, h), color=col, lw=1.2)
        legT.append(Line2D([], [], color=col, lw=1.2, label=f"$t$={h:g} h"))
    for k, h in enumerate(t_C):
        col = TIME_CMAP(0.10 + 0.80 * k / (len(t_C) - 1))
        axC.plot(pos_cm, prof(C, h), color=col, lw=1.2)
        legC.append(Line2D([], [], color=col, lw=1.2, label=f"$t$={h:g} h"))
    for k, h in enumerate(t_C):      # 中心（r=0）标记：尾段拖长的直观证据
        col = TIME_CMAP(0.10 + 0.80 * k / (len(t_C) - 1))
        axC.plot([0], [prof(C, h)[0]], marker="o", ms=3.6, color=col, clip_on=False, zorder=5)

    Ta_end = float(T[-1].max())
    axT.set_xlabel("到药材中心的距离 $r$ / cm")
    axT.set_ylabel("药材温度 $T$ / ℃")
    axT.set_xlim(0, 2.02)
    axT.set_ylim(27.0, 58.5)
    style_axis(axT, grid="y")
    axT.legend(handles=legT, loc="upper left", ncol=2, handlelength=1.4,
               columnspacing=1.0, labelspacing=0.28)
    panel(axT, "(a)")
    tile(axT, "温度剖面：3 h 内即准稳态")

    axC.set_xlabel("到药材中心的距离 $r$ / cm")
    axC.set_ylabel("水分浓度 $C$ / (kg/kg)")
    axC.set_xlim(0, 2.02)
    axC.set_ylim(0.0, 3.15)
    style_axis(axC, grid="y")
    axC.axhline(TH, color=ROLE["guide"], ls="--", lw=1.1)
    axC.text(2.0, TH + 0.15, f"阈值 {TH} kg/kg", ha="right", va="bottom",
             fontsize=7.5, color="0.30")
    axC.legend(handles=legC, loc="upper left", ncol=2, handlelength=1.4,
               columnspacing=1.0, labelspacing=0.28)
    c36, c57 = prof(C, 36.0)[0], prof(C, 57.0)[0]
    cs57 = prof(C, 57.0)[-1]
    axC.annotate("中心最慢\n（$r$=0，尾段拖长）", xy=(0.02, 0.5 * (c36 + c57)),
                 xytext=(0.13, 0.36), textcoords="axes fraction", fontsize=7.5,
                 ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"))
    panel(axC, "(b)")
    tile(axC, "水分剖面：中心最慢")

    print(f"  [F3] t_f={tf_h:.3f} h；中心 {c57:.4f} / 表面 {cs57:.4f}；36→57 h 中心降 "
          f"{c36-c57:.4f}；Le={le_txt}；t={t_T[-1]:g} h 全径 {Ta_end-0.4:.1f}–{Ta_end:.1f} ℃",
          flush=True)
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
    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.2))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.875, bottom=0.17)
    ax.plot(h3, um3, color=ROLE["model2"], lw=1.3,
            label=f"问题3 主解（附3 + 固定 $R$）：$t_f$={tf3:.2f} h")
    ax.plot(h4, um4, color=ROLE["model"], lw=1.5,
            label=f"问题4 主解（附4 + $R(t)$）：$t_f$={tf4:.2f} h")
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
    print(f"  [F4] t_f: 问题3 {tf3:.4f} h / 问题4 {tf4:.4f} h / 附4固定R {tfc:.4f} h"
          f"（csv {tf_ctl_csv:.4f}）; 下界 {lb3:.4f} / {lb4:.4f} h", flush=True)
    print(f"  [F4] 效应拆分（源 q4_split.csv）：对照 {tf_ctl_csv:.2f} h → 附3 物性 "
          f"{tf3:.2f} h → 真实收缩 $R(t)$ {tf4:.2f} h（{tf4-tf_ctl_csv:+.2f} h，占对照 "
          f"{(tf_ctl_csv-tf4)/tf_ctl_csv*100:.1f}%）；严格下界余量 {tf3-lb3:.2f} / "
          f"{tf4-lb4:.2f} h（源 v3_bound.csv、q4_bound.csv）", flush=True)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_DOUBLE, 4.15),
                                   gridspec_kw=dict(height_ratios=[1.18, 1.0]))
    fig.subplots_adjust(left=0.30, right=0.985, top=0.905, bottom=0.205, hspace=0.60)
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
    ax1.text(103.0, div, "问题4 效应拆分（T6 归因）", fontsize=7.5,
             color="0.25", ha="right", va="center")
    ce = [b for b in grp1 if b[0].startswith("C_e")][0]
    others = sum(abs(b[1]) for b in grp1 if b is not ce)
    panel(ax1, "(a)")
    tile(ax1, "敏感性：各因素 $\\Delta t_f$")

    mags = np.array([float(r["量级_h"]) for r in budget])
    typs = [r["类型"] for r in budget]
    tcol = {"可收敛": ROLE["model"], "口径依赖": OI["orange"], "固有": ROLE["aux2"],
            "情景假设": ROLE["ref"], "不含C_e口径": OI["grey"]}
    ys2 = np.arange(len(mags))[::-1]
    for y, m, tp in zip(ys2, mags, typs):
        ax2.barh(y, m, color=tcol.get(tp, "0.5"), height=0.60, zorder=3)
        ax2.text(m * 1.45, y, fmt_mag(m), va="center", fontsize=7.5)
    ax2.set_yticks(ys2)
    ax2.set_yticklabels([r["误差源"] for r in budget], fontsize=7.5)
    ax2.set_xscale("log")
    ax2.set_xlim(2e-7, 300)
    ax2.set_xlabel("误差量级 / h（对数轴）")
    style_axis(ax2, grid="x")
    ax2.legend(handles=[Patch(color=c, label=t.replace("不含C_e口径", "不含C_e"))
                        for t, c in tcol.items()],
               loc="upper center", bbox_to_anchor=(0.30, -0.29), ncol=5,
               handlelength=1.2, columnspacing=1.0, labelspacing=0.28)
    panel(ax2, "(b)", dx=-0.10)
    tile(ax2, "误差预算：各类误差量级")
    fit_left(fig, (ax1, ax2))
    print(f"  [F5] 敏感性 {len(shown)} 条（基准 {len(base)} 条未画："
          f"{'；'.join(b[0] for b in base)}），红=偏晚、蓝=偏早；"
          f"主导 {ce[0]} {ce[1]:+.4f} h 是其余敏感性因素之和（{others:.2f} h）的 "
          f"{abs(ce[1])/others:.0f} 倍；误差预算 {len(mags)} 条", flush=True)
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
    if U_mod is None:                       # 回退：问题4 主解步记录
        d4 = np.load(DATA_DIR / "q4_main_steps.npz")
        U_mod = np.interp(t_s, d4["t"], d4["um"])
    R_ideal, R_crust, R_pore = (pick("R1_ideal_mean_cm"), pick("R3b_crust_mean_cm"),
                                pick("R5_pore_cm"))
    miss = [n for n, v in (("R_att2_cm", R_att2), ("U_inferred_att2", U_inf),
                           ("U_mean_model", U_mod), ("R1_ideal_mean_cm", R_ideal),
                           ("R3b_crust_mean_cm", R_crust), ("R5_pore_cm", R_pore))
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
    axL.text(26.0, 2.115, "C 冻结", ha="center", va="center", fontsize=7.5, color="0.30")
    axL.plot(h, R_att2, color=ROLE["data"], lw=1.6, zorder=5,
             label="附件2 实测 $R(t)$")
    axL.plot(h, R_ideal, color=ROLE["ref"], ls="--", lw=1.2,
             label="闭合1 全局理想")
    axL.plot(h, R_crust, color=ROLE["model"], lw=1.5,
             label="闭合3' 理想+结皮")
    axL.plot(h, R_pore, color=ROLE["aux2"], ls="-.", lw=1.2,
             label="闭合5 成孔演化")
    axL.axhline(R_plat, color=ROLE["guide"], ls=":", lw=1.0)
    axL.text(h[-1] - 1.0, R_plat + 0.014, f"平台 {R_plat:.3f} cm", ha="right", va="bottom",
             fontsize=7.5, color="0.30")
    axL.annotate(f"早期塌陷\n峰值超额 {exc[k_exc]:.3f} cm",
                 xy=(h[k_exc], 0.5 * (R_att2[k_exc] + R_ideal[k_exc])),
                 xytext=(0.20, 0.46), textcoords="axes fraction",
                 fontsize=7.5, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axL.annotate(f"后期结皮/成孔\n平台 {R_crust[-1]:.3f} cm",
                 xy=(46.0, R_plat + 0.002), xytext=(0.30, 0.30), textcoords="axes fraction",
                 fontsize=7.5, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35"),
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="0.80", lw=0.5, alpha=0.95))
    axL.set_xlim(0, h[-1] + 2.0)
    axL.set_ylim(1.05, 2.22)
    axL.set_xlabel("烘干时间 $t$ / h")
    axL.set_ylabel("药材半径 $R$ / cm")
    style_axis(axL, grid="y")
    axL.legend(loc="upper right", handlelength=1.4, labelspacing=0.26)
    panel(axL, "(a)")
    tile(axL, "半径对比：三条零拟合闭合")

    # ---- 右：交叉诊断（附件2 等效含水率 vs 模型平均含水率）----
    axR.axvspan(0.0, t_cross, color=ROLE["ref"], alpha=0.05, lw=0)
    axR.axvspan(t_cross, h[-1], color=ROLE["model"], alpha=0.05, lw=0)
    axR.axvline(t_cross, color=ROLE["ref"], ls=":", lw=1.1)
    axR.plot(h, U_inf, color=ROLE["data"], lw=1.4, zorder=5,
             label=r"附件2 $\bar U_{inf}(t)$（由 $R$ 反推）")
    axR.plot(h, U_mod, color=ROLE["model"], lw=1.3,
             label=r"模型 $\bar U_{model}(t)$（问题4）")
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

    print(f"  [F6] 平台 {R_plat:.4f} cm；结皮闭合 {R_crust[-1]:.4f} cm（{dev_crust:+.2f}%）、"
          f"冻结 {h_frz_crust:.1f} h（{dfrz_crust:+.1f}%，口径分界 {H_SEG[1]:g} h）；"
          f"理想闭合 {R_ideal[-1]:.4f} cm"
          f"（{dev_ideal:+.1f}%）、成孔闭合 {R_pore[-1]:.4f} cm（骨架 {RHO_SK:.0f}）；"
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
    axA.text(26.0, 2.125, "C 冻结", ha="center", va="center", fontsize=7.5, color="0.30")
    axA.plot(h, R_data, color=ROLE["data"], lw=1.6, zorder=5, label="附件2 实测 $R(t)$")
    axA.plot(h, R_pred, color=ROLE["model"], ls="--", lw=1.3,
             label=f"内生模型 $R_{{pred}}(t)$（RMSE={rmse:.3f} mm）")
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


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig5": fig5, "fig6": fig6,
        "fig7": fig7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", default=[], help="只出指定图，如 fig4")
    ap.add_argument("--no-cache", action="store_true", help="fig4 对照曲线强制重算")
    a = ap.parse_args()
    plt.rcParams.update(RC)
    want = a.only or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7"]
    for k in want:
        print(f"[{k}] 作图 …", flush=True)
        if k == "fig4":
            fig4(no_cache=a.no_cache)
        else:
            FIGS[k]()

    print("\n[审计汇总] 图 | 设计尺寸(in) | 纵横比 | 最小字号(pt) | 标题≤28 | 图例项 | 刻度重叠 | "
          "柱基线 | 裁切文字 | 图内文字块 | colorbar | 3D | 嵌入位图 | 密集标记 | 结论")
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
