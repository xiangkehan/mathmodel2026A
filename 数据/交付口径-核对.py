"""T2 交付口径核对：逐格读取四个结果模板 + 官方 PDF 的四张表结构。

只读 01-题目/ 下的原始文件，输出到 stdout（重定向为 交付口径-核对输出.txt）。
依赖：openpyxl（模板）、pymupdf（PDF 表结构证据）。
用法：PYTHONIOENCODING=utf-8 python 数据/交付口径-核对.py > 数据/交付口径-核对输出.txt
"""

import os

import openpyxl
import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL_DIR = os.path.join(ROOT, "01-题目", "原始文件", "附件3")
PDF = os.path.join(ROOT, "01-题目", "原始文件", "A题.pdf")

TEMPLATES = ["result1.xlsx", "result2.xlsx", "result3.xlsx", "result4.xlsx"]


def dump_templates():
    print("=" * 78)
    print("[1] 结果模板逐格转储（openpyxl，data_only 默认读取公式串；本批模板无公式）")
    print("=" * 78)
    for name in TEMPLATES:
        path = os.path.join(TPL_DIR, name)
        wb = openpyxl.load_workbook(path)
        print("\n" + "-" * 78)
        print(f"FILE {name}")
        print("  sheetnames (顺序):", wb.sheetnames)
        print("  sheet_state      :", [(ws.title, ws.sheet_state) for ws in wb.worksheets])
        for ws in wb.worksheets:
            print(f"  SHEET {ws.title!r}: dims={ws.dimensions} max_row={ws.max_row} max_col={ws.max_column}")
            print("    merged_cells:", sorted(str(r) for r in ws.merged_cells.ranges))
            for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
                cells = []
                for c in row:
                    v = c.value
                    if isinstance(v, str):
                        v = f"'{v}'"
                    cells.append(f"{c.coordinate}={v}")
                print("    R%d: " % row[0].row + " | ".join(cells))
            print("    A1 字符码:", [(ch, hex(ord(ch))) for ch in str(ws["A1"].value)])
            print("    A1/A2 number_format:", ws["A1"].number_format, "/", ws["A2"].number_format)
            print("    comments:", [(c.coordinate, c.comment.text) for r in ws.iter_rows() for c in r if c.comment])
            print("    freeze_panes:", ws.freeze_panes, "| auto_filter:", ws.auto_filter.ref)
    print("\n  结论（模板自证）：A1='时间\\到药材中心的距离'（单反斜杠 0x5c，非换行）；")
    print("    第 1 行为列头、A 列为时间；示例行 A2/A3/A4 与末行 A5='…'。")
    print("    result1/2/3 的末列 F1 = 固定值 2；result4 的末列 F1 = '药材表面'。")


def dump_pdf_tables():
    print("\n" + "=" * 78)
    print("[2] 官方 PDF 表结构（PyMuPDF find_tables 的栅格化结果）")
    print("=" * 78)
    doc = fitz.open(PDF)
    print("pages:", doc.page_count)
    for pno in range(doc.page_count):
        page = doc[pno]
        tabs = page.find_tables()
        if not tabs.tables:
            continue
        print(f"\n--- page {pno + 1}: {len(tabs.tables)} table(s)")
        for ti, t in enumerate(tabs.tables):
            print(f"  table[{ti}] grid rows={t.row_count} cols={t.col_count} bbox={tuple(round(v, 1) for v in t.bbox)}")
            for ri, row in enumerate(t.extract()):
                print(f"    r{ri}: {row}")
            head = t.extract()[0]
            caption = ""
            for blk in page.get_text("blocks"):
                if blk[3] <= t.bbox[1] and t.bbox[1] - blk[3] < 40 and "表" in blk[4]:
                    caption = blk[4].strip().replace("\n", " ")
            print(f"    caption: {caption!r}")
            print(f"    -> 第 2 个表头格 {head[1]!r} 占据栅格第 2..{len(head)} 列，即 colspan={len(head) - 1}")
    doc.close()
    print("\n  结论：表 1–表 5 的表头栅格为 6 列（'到药材中心的距离/cm' colspan=5），")
    print("        表 6 的表头栅格为 5 列（colspan=4），末列为 '药材表面' 而非固定 '2'。")


def dump_source_data():
    print("\n" + "=" * 78)
    print("[3] 附件1/附件2 首末与半径上界（用于判据：固定列是否落在药材之外）")
    print("=" * 78)
    wb1 = openpyxl.load_workbook(os.path.join(ROOT, "01-题目", "原始文件", "附件1.xlsx"))
    ws1 = wb1["Sheet1"]
    print("附件1 sheets:", wb1.sheetnames, ws1.dimensions)
    for r in (1, 2, 32, ws1.max_row):
        print(f"  row {r}:", [ws1.cell(r, c).value for c in range(1, 4)])
    wb2 = openpyxl.load_workbook(os.path.join(ROOT, "01-题目", "原始文件", "附件2.xlsx"))
    ws2 = wb2["Sheet1"]
    print("附件2 sheets:", wb2.sheetnames, ws2.dimensions)
    for r in (1, 2, 3, 4, 5, ws2.max_row):
        print(f"  row {r}:", [ws2.cell(r, c).value for c in range(1, 3)])
    rad = [ws2.cell(r, 2).value for r in range(2, ws2.max_row + 1)]
    print("  半径 max/min:", max(rad), min(rad), "| t=1800 s 半径:", ws2.cell(3, 2).value)


if __name__ == "__main__":
    dump_templates()
    dump_pdf_tables()
    dump_source_data()
