"""摘要字数统计（P3：600–900 字）。"""
import re
from pathlib import Path

tex = (Path(__file__).parent / "main.tex").read_text(encoding="utf-8")
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
plain = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", m.group(1))
plain = re.sub(r"[$~\\{}]", "", plain)
zh = sum(1 for c in plain if "一" <= c <= "鿿")
total = len(re.sub(r"\s", "", plain))
print(f"摘要中文字数 {zh}，含数字与符号总字符 {total}")
