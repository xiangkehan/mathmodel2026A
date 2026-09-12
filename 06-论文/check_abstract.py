"""摘要自查（P3）：中文字数、总字符、四段式标签、关键词数与分隔、无公式图表环境。"""
import re
import sys
from pathlib import Path

tex = (Path(__file__).parent / "main.tex").read_text(encoding="utf-8")
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
body = m.group(1)
plain = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", body)
plain = re.sub(r"[$~\\{}]", "", plain)
zh = sum(1 for c in plain if "一" <= c <= "鿿")
total = len(re.sub(r"\s", "", plain))
print(f"摘要中文字数 {zh}，含数字与符号总字符 {total}")

fail = []
# 四段式标签
for tag in ("问题与挑战", "方法", "关键结果", "验证与创新"):
    if tag not in body:
        fail.append(f"缺四段式标签: {tag}")
# 关键词：4–6 个、空格分隔（\\quad）、不用分号
mk = re.search(r"关键词：?\}?\s*(.+)", body)
kw = mk.group(1).strip() if mk else ""
if ";" in kw or "；" in kw:
    fail.append("关键词含分号（应空格分隔）")
kws = [k for k in re.split(r"\\quad|\s+", kw) if k and "一" <= k[0] <= "鿿"]
print(f"关键词 {len(kws)} 个: {' / '.join(kws)}")
if not (4 <= len(kws) <= 6):
    fail.append(f"关键词数 {len(kws)} 不在 4–6")
# 无公式图表环境
for env in ("equation", "figure", "tabular", "includegraphics"):
    if f"\\begin{{{env}}}" in body or f"\\{env}" in body:
        fail.append(f"摘要含 {env}")
# 字数约束（与一页排版配套）
if not (550 <= zh <= 900):
    fail.append(f"中文字数 {zh} 超出 550–900")
if fail:
    print("摘要自查失败:")
    for x in fail:
        print(" -", x)
    sys.exit(1)
print("摘要自查通过。")
