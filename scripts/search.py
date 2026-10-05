"""命令行检索演示（不调用大模型）。

用法：python scripts/search.py "帮别人走账会构成什么罪" [-k 8]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import Retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    res = Retriever().search(args.query, top_k=args.k)
    print(f"问题：{res.query}  有依据：{res.has_evidence}")
    if res.expanded_query != res.query:
        print(f"术语扩展：{res.expanded_query}")
    print()
    for i, h in enumerate(res.hits, 1):
        print(f"{i}. [{h.score:.3f}] {h.citation}  (向量#{h.dense_rank} 关键词#{h.sparse_rank})")
        print(f"   {h.text[:100]}\n")
    for c in res.cases:
        print(f"案例 [{c.score:.3f}] {c.citation}  关键词：{'、'.join(c.keywords)}  命中：{'、'.join(c.matched) or '-'}")


if __name__ == "__main__":
    main()
