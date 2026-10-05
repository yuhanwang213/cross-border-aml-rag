"""把 data/chunks.jsonl 写入 Chroma（稠密向量 + 权限布尔字段），稀疏权重另存 json。"""
from __future__ import annotations

import json

from .config import CHROMA_DIR, CHUNKS_PATH, COLLECTION, SPARSE_PATH
from .models import encode

# 写入 Chroma 的标量元数据字段（Chroma 元数据只接受 str/int/float/bool）
_META_FIELDS = [
    "doc_id", "doc_title", "doc_type", "source_file", "version", "effective_date",
    "chapter", "section", "article", "article_no", "page_start", "page_end", "citation",
    "doc_no", "status", "kind", "alias",
]


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def embed_text(c: dict) -> str:
    """向量化时带上文档名和章节，让「公司法里关于……」这类问法也能命中。"""
    head = " ".join(x for x in (f"《{c['doc_title']}》", c["chapter"], c["section"],
                                 f"【{c['alias']}】" if c.get("alias") else "") if x)
    return f"{head}\n{c['text']}"


def to_metadata(c: dict) -> dict:
    meta = {k: c[k] for k in _META_FIELDS if c.get(k) is not None}
    return meta


def get_collection(create: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if create:
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
        return client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    return client.get_collection(COLLECTION)


def build_index(batch_size: int = 32) -> int:
    chunks = load_chunks()
    col = get_collection(create=True)
    sparse_all: dict[str, dict[str, float]] = {}
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        dense, sparse = encode([embed_text(c) for c in batch])
        col.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=dense,
            documents=[c["text"] for c in batch],
            metadatas=[to_metadata(c) for c in batch],
        )
        sparse_all.update({c["chunk_id"]: s for c, s in zip(batch, sparse)})
        print(f"  已写入 {min(i + batch_size, len(chunks))}/{len(chunks)}")
    SPARSE_PATH.write_text(json.dumps(sparse_all, ensure_ascii=False), encoding="utf-8")
    return len(chunks)
