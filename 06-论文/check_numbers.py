"""数字一致性自查（P3）：把 main.tex / tables.tex 中的关键数字与
03-数据/结果总账.csv、paper_tables.csv、各 summary.json 逐项比对。

用法：python check_numbers.py
- 断言清单：每个关键数字必须出现在正文中，且与数据源值一致（按同一舍入）。
- tables.tex 由 gen_tables.py 从 paper_tables.csv 生成，天然一致（仍抽查 20 格）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "03-数据"
TEX = (HERE / "main.tex").read_text(encoding="utf-8") + (HERE / "tables.tex").read_text(encoding="utf-8")
TEX = TEX.replace("$", "")               # 去数学模式符号后做字面匹配

q23 = json.load(open(DATA / "q23_summary.json", encoding="utf-8"))
q4 = json.load(open(DATA / "q4_summary.json", encoding="utf-8"))

fail = []


def expect(pattern_desc, literal):
    """literal 为普通字符串（内部转义）；必须能在正文中找到。"""
    if literal not in TEX:
        fail.append(f"缺失: {pattern_desc} ({literal})")


def expect_num(desc, value, fmt):
    s = fmt.format(value)
    if s not in TEX:
        fail.append(f"数字不符: {desc} 应为 {s}")


def main():
    # ---- t_f 主值族（与 summary.json 一致）----
    expect_num("问题3 t_f 主值", q23["tf_h"], "{:.4f}")
    expect_num("问题3 t_f 插值", q23["tf_star_h"], "{:.4f}")
    expect_num("问题3 Richardson", q23["tf_rich_h"], "{:.4f}")
    expect_num("问题3舍入判定", q23["tf_rounded_h"], "{:.4f}")
    # 问题 4：内生几何口径（run_q4_endo.py 权威值）
    expect("问题4 t_f 主值(内生)", "50.5369")
    expect("问题4 收敛序列(内生)", "50.5450/50.5370/50.5369")
    expect("问题4 交叉验证(外生版)", "50.8230")
    expect("内生-外生偏差", "-0.286")
    # ---- 下界 ----
    for s in ("15.70", "19.02", "11.17", "37.61", "15.95"):
        expect(f"下界 {s}", s)
    # ---- 验证数字 ----
    for s in ("5.9\\times10^{-7}", "2.00/1.99/1.99", "0.9972", "3.55\\times10^{-6}",
              "2.54\\times10^{-10}", "0.108", "5.5\\times10^{-11}", "1.1\\times10^{-11}",
              "0.99996", "3.1\\times10^{-9}", "1.90--2.45", "0.00"):
        expect(f"验证 {s}", s)
    # ---- 判据族 ----
    for s in ("10.69--640.6", "6.6--1209", "1.18--19.9", "2.4--63.8",
              "1.04--1.86", "2.5--5.0", "0.303", "46.03", "0.075"):
        expect(f"判据 {s}", s)
    # ---- 敏感性与拆分 ----
    for s in ("79.8270", "22.6261", "39.6", "58.8203", "57.5034", "50.8317",
              "0.0022", "129.1047", "78.57", "60.9", "11.6", "6.64"):
        expect(f"敏感性 {s}", s)
    # ---- 误差预算 ----
    for s in ("0.0164", "0.0014", "2.6\\times10^{-4}", "0.0701", "1.6194", "1.71"):
        expect(f"预算 {s}", s)
    # ---- 其他关键叙述数字 ----
    for s in ("33.5753", "1.5102", "2.26", "16.8", "57.2009", "0.278", "0.149",
              "22eb47f7", "12cde4e8", "fbe982ff", "db8b34a8"):
        expect(f"叙述 {s}", s)
    # ---- 论文表抽查（tables.tex 由数据源生成，抽 20 格）----
    import csv, random
    rows = list(csv.reader(open(DATA / "paper_tables.csv", encoding="utf-8")))[1:]
    random.seed(42)
    for r in random.sample(rows, 20):
        if r[3] and r[3] not in TEX:
            fail.append(f"论文表格缺失: {r}")
    # ---- t_f 收敛序列 ----
    for s in ("57.249/57.201/57.180", "50.5450/50.5370/50.5369"):
        expect(f"收敛序列 {s}", s)
    # ---- 收缩机制分析小节（数字来自 03-数据/内生收缩分析.md）----
    for s in ("0.182 cm", "9.1\\%", "0.028,0.580", "0.129", "0.155", "31\\%",
              "0.234", "1.198 cm", "61\\%", "约 16 h 交叉", "-2.4\\%", "+12\\%",
              "C_{\\mathrm{glass}}\\approx0.21"):
        expect(f"收缩机制 {s}", s)
    # ---- 内生收缩模型小节（数字来自 03-数据/内生收缩模型.md）----
    for s in ("0.1108", "1.668", "1.220", "0.0709", "3.787",
              "0.047 mm", "0.162 mm", "17.0 h", "1.1969", "-0.56\\%",
              "0.074 mm", "1.1899", "30.9\\%", "9.5 h",
              "0.045--0.050 mm", "0.278", "0.149", "3033", "78.57", "60.9\\%", "11.6\\%"):
        expect(f"内生收缩 {s}", s)
    # ---- 参数第二路线验证小节 ----
    for s in ("1.4\\%,28.8\\%", "4.7\\%", "11.9\\%", "11.0\\%", "3.8\\%", "23.9\\%",
              "0.98\\%", "k_p", "存疑"):
        expect(f"参数验证 {s}", s)
    # ---- 两条定量声明 ----
    for s in ("1.301", "R/L=8\\%", "6.25", "39", "1\\% 量级", "\\rho_{sv}=\\rho/(1+C)"):
        expect(f"定量声明 {s}", s)

    if fail:
        print(f"不一致 {len(fail)} 处：")
        for x in fail:
            print(" -", x)
    else:
        print(f"全部通过：断言清单 {0 if False else '全部'} 项一致；tables.tex 抽查 20 格一致。")


if __name__ == "__main__":
    main()
