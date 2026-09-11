"""run_all.py：单命令复现全项目（M1–M4）。

用法：  python run_all.py
依次执行 run_q1 → run_q23 → run_q4 → run_m4（各自的一条命令产物与门禁全部包含），
日志落 ../03-数据/run_all.log；结束打印 result1–4 的 md5 供两次运行比对。
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
A_DIR = CODE_DIR.parent
LOG = A_DIR / "03-数据" / "run_all.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    t0 = time.perf_counter()
    logf = open(LOG, "w", encoding="utf-8")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, logf)
    print(f"==== run_all 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    import run_q1
    import run_q23
    import run_q4
    import run_m4
    for name, mod in (("run_q1(问题1+V1)", run_q1),
                      ("run_q23(问题2/3+门禁)", run_q23),
                      ("run_q4(问题4+T6/V5/T7)", run_q4),
                      ("run_m4(判据/敏感性/误差/总账)", run_m4)):
        print(f"\n######## {name} ########", flush=True)
        t1 = time.perf_counter()
        mod.main()
        print(f"######## {name} 完成，耗时 {time.perf_counter() - t1:.1f} s ########", flush=True)
    print("\n==== result 工作簿 md5（两次运行应一致）====")
    for fn in ("result1.xlsx", "result2.xlsx", "result3.xlsx", "result4.xlsx"):
        p = A_DIR / "04-结果" / fn
        print(f"{fn}: {hashlib.md5(p.read_bytes()).hexdigest()}  ({p.stat().st_size} B)")
    print(f"==== run_all 全部完成，总耗时 {time.perf_counter() - t0:.1f} s ====")
    sys.stdout = sys.stderr = sys.__stdout__
    logf.close()


if __name__ == "__main__":
    main()
