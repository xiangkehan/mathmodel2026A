"""label/ref/cite 双向核对（P5）。"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
tex = (HERE / "main.tex").read_text(encoding="utf-8") + (HERE / "tables.tex").read_text(encoding="utf-8")
labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex)) | set(re.findall(r"\\eqref\{([^}]*)\}", tex))
cites = set(re.findall(r"\\cite\{([^}]*)\}", tex))
bibs = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
print("labels 未被引用:", sorted(labels - refs))
print("引用无 label:", sorted(refs - labels))
print("bibitem 未被引用:", sorted(bibs - cites))
print("cite 无条目:", sorted(cites - bibs))
