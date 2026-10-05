"""向量化 data/chunks.jsonl 并写入 Chroma。先运行 scripts/build_chunks.py。

用法：python scripts/build_index.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index import build_index  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    n = build_index()
    print(f"完成：{n} 块，用时 {time.time() - t:.0f}s")
