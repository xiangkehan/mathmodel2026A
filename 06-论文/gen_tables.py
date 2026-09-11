"""由 03-数据 的 CSV 唯一数据源生成 LaTeX 表格（tables.tex），禁止手工誊数。

用法：python gen_tables.py   （在 06-论文 目录下运行）
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "03-数据"
OUT = Path(__file__).resolve().parent / "tables.tex"

POS5 = ["0", "0.5", "1", "1.5", "2"]


def load(name):
    return list(csv.reader(open(DATA / name, encoding="utf-8")))


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
        cap = caption + "。数字由 result 工作簿同一数据源抽取，保留四位小数。"
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
               "结束行时刻为未舍入判定值 57.1799~h；表中 0.1500 为舍入显示，判定位用未舍入值。}\n"
               "\\label{tab:表5}\n%s\n\\end{table}" % tabular([["时间/h"] + POS5], body5))
    # 表 6
    rs6 = [r for r in rows if r[0] == "表6"]
    times6 = []
    for r in rs6:
        if r[1] not in times6:
            times6.append(r[1])
    cols6 = ["0", "0.5", "1", "1.5", "药材表面"]
    body6 = []
    for t in times6:
        rr = [r for r in rs6 if r[1] == t]
        vals = []
        for c in cols6:
            hit = [r[3] for r in rr if r[2] == (f"r={c}cm" if c != "药材表面" else c)]
            vals.append(hit[0] if hit and hit[0] else "—")
        label = t.split("=")[1] if "=" in t else t
        body6.append([label] + vals)
    out.append("\\begin{table}[htbp]\n\\centering\n\\caption{问题~4 药材烘干过程的水分浓度（kg/kg）。"
               "“—”表示该时刻该固定位置已在药材表面之外（半径收缩，留空不外推）。"
               "结束行时刻为未舍入判定值 50.8230~h。}\n\\label{tab:表6}\n%s\n\\end{table}"
               % tabular([["时间/h"] + cols6], body6))
    return "\n\n".join(out)


def main():
    tex = ["% 本文件由 gen_tables.py 从 03-数据/*.csv 自动生成，禁止手工改数。\n",
           paper_tables()]
    Path(OUT).write_text("\n".join(tex), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
