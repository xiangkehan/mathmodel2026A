"""由 03-数据 的 CSV 唯一数据源生成 LaTeX 表格（tables.tex），禁止手工誊数。

用法：python gen_tables.py   （在 06-论文 目录下运行）
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "03-数据"
OUT = Path(__file__).resolve().parent / "tables.tex"
RESULT4 = Path(__file__).resolve().parent.parent / "04-结果" / "result4.xlsx"

POS5 = ["0", "0.5", "1", "1.5", "2"]
COLS6 = ["0", "0.5", "1", "1.5", "药材表面"]


def load(name):
    return list(csv.reader(open(DATA / name, encoding="utf-8")))


def table6_from_result4():
    """表 6（问题 4）直接取自**内生交付工作簿** result4.xlsx，保证论文表 11 与工作簿逐格一致。

    工作簿结构（交付口径.md §4+§5.1）：Sheet1，A1 表头，其后 20 列 r=0.0…1.9 cm、
    末列「药材表面」；常规行是 60 s 行程（6/12/…/48 h 均为其子集），末行是 t_f 数据行。
    位置超出当前表面时工作簿该格为空 → 表中留“—”（留空不外推，与交付口径一致）。
    返回 (行列表, 末行时刻/h)。
    """
    import openpyxl
    wb = openpyxl.load_workbook(RESULT4, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    it = ws.iter_rows(values_only=True)
    head = list(next(it))
    body = [r for r in it if isinstance(r[0], (int, float))]
    wb.close()
    col = {str(head[j]).replace("cm", "").strip(): j for j in range(1, len(head))}
    by_t = {float(r[0]): r for r in body}
    t_last = float(body[-1][0])

    def cell(src, name):
        v = src[-1] if name == "药材表面" else src[col[name]]
        return "—" if v is None else f"{float(v):.4f}"

    rows = []
    for th in range(6, int(t_last // 3600) // 6 * 6 + 1, 6):
        src = by_t.get(th * 3600.0)
        if src is None:
            raise KeyError(f"result4.xlsx 缺 {th} h 行（60 s 行程应含该行）")
        rows.append([f"{th}h"] + [cell(src, c) for c in COLS6])
    rows.append(["烘干结束时间"] + [cell(body[-1], c) for c in COLS6])
    return rows, t_last / 3600.0


def tabular(headers, rows, align=None, fontsize="\\small"):
    align = align or ("l" + "c" * (len(headers[0]) - 1))
    s = [fontsize, "\\begin{tabular}{%s}" % align, "\\toprule",
         " & ".join(headers[0]) + " \\\\", "\\midrule"]
    for r in rows:
        s.append(" & ".join(str(x) for x in r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(s)


def paper_tables():
    rows = [r for r in load("paper_tables.csv")[1:]]
    out = []
    specs = {
        "表1": ("问题~1（预热平衡阶段）药材温度（℃）", "时间/s"),
        "表2": ("问题~1（预热平衡阶段）药材水分浓度（kg/kg）", "时间/s"),
        "表3": ("问题~2（全烘干过程）药材温度（℃）", "时间/h"),
        "表4": ("问题~2（全烘干过程）药材水分浓度（kg/kg）", "时间/h"),
    }
    for tab, (caption, tlabel) in specs.items():
        rs = [r for r in rows if r[0] == tab]
        times = []
        for r in rs:
            if r[1] not in times:
                times.append(r[1])
        body = []
        for t in times:
            vals = [r[3] for r in rs if r[1] == t]
            body.append([t.split("=")[1]] + vals)
        cap = caption + "。数字由 result 工作簿同一数据源抽取，保留四位小数"
        out.append("\\begin{table}[htbp]\n\\centering\n\\caption{%s}\n\\label{tab:%s}\n%s\n\\end{table}"
                   % (cap, tab, tabular([[tlabel] + POS5], body)))
    # 表 5
    rs5 = [r for r in rows if r[0] == "表5"]
    times5 = []
    for r in rs5:
        if r[1] not in times5:
            times5.append(r[1])
    body5 = []
    for t in times5:
        vals = [r[3] for r in rs5 if r[1] == t]
        label = t.split("=")[1] if "=" in t else t
        body5.append([label] + vals)
    out.append("\\begin{table}[htbp]\n\\centering\n\\caption{问题~3 药材烘干过程的水分浓度（kg/kg）。"
               "结束行时刻为未舍入判定值 57.1799~h；表中 0.1500 为舍入显示，判定位用未舍入值}\n"
               "\\label{tab:表5}\n%s\n\\end{table}" % tabular([["时间/h"] + POS5], body5))
    # 表 6（问题 4）：直接从内生交付工作簿 result4.xlsx 取数（论文表 11 与工作簿逐格一致）
    body6, tf6_h = table6_from_result4()
    out.append("\\begin{table}[htbp]\n\\centering\n"
               "\\caption{问题~4 药材烘干过程的水分浓度（kg/kg）。“—”表示该时刻该固定位置"
               "已在药材表面之外（半径收缩，留空不外推）。"
               f"结束行时刻为未舍入判定值 {tf6_h:.4f}~h（内生几何主解）}}"
               "\n\\label{tab:表6}\n%s\n\\end{table}"
               % tabular([["时间/h"] + COLS6], body6))
    return "\n\n".join(out)


def main():
    Path(OUT).write_text(paper_tables(), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
