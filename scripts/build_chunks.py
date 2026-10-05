"""解析 Regulations 下所有文档并按条切分，输出 data/chunks.jsonl。

用法：python scripts/build_chunks.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import chunk_document  # noqa: E402
from rag.config import CHUNKS_PATH, DATA_DIR, DOCS_DIR  # noqa: E402
from rag.parsers import load_documents  # noqa: E402


def main():
    docs, skipped, dups = load_documents(DOCS_DIR)
    for p in skipped:
        print(f"[跳过] {p.relative_to(DOCS_DIR)}")
    for drop, keep, sim in dups:
        print(f"[去重] {Path(drop).name}  与  {Path(keep).name}  相似度 {sim:.0%}，保留后者")

    all_chunks, ids = [], Counter()
    for doc in docs:
        chunks, warnings = chunk_document(doc)
        for w in warnings:
            print(f"[警告] {w}")
        lens = [len(c.text) for c in chunks] or [0]
        kinds = Counter(c.kind for c in chunks)
        flags = "".join(f" [{x}]" for x in (doc.status, "OCR" if doc.ocr else "") if x)
        print(f"{doc.doc_id} [{doc.doc_type}] {doc.title[:40]}{flags}: {len(chunks)} 块 "
              f"{dict(kinds)} 最长 {max(lens)} 字")
        all_chunks += chunks
        ids.update(c.chunk_id for c in chunks)

    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"chunk_id 重复: {dup[:10]}")

    DATA_DIR.mkdir(exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_record(), ensure_ascii=False) + "\n")
    print(f"\n共 {len(all_chunks)} 块 -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
