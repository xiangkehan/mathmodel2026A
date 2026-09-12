"""一次性修复：heredoc 追加时表格行尾 \\ 被吞成 \ 。把行尾的单反斜杠恢复为双反斜杠。"""
from pathlib import Path

p = Path(__file__).resolve().parent / "main.tex"
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
fixed = []
n = 0
for ln in lines:
    body = ln.rstrip("\n")
    if body.endswith(" \\") and not body.endswith(" \\\\"):
        ln = body + "\\" + "\n"
        n += 1
    fixed.append(ln)
p.write_text("".join(fixed), encoding="utf-8")
print(f"fixed {n} lines")
