"""模型懒加载：bge-m3（稠密+稀疏）与 bge-reranker-v2-m3。进程内只加载一次。"""
from functools import lru_cache

from .config import EMBED_MODEL, RERANK_MODEL


def _local_or_remote(repo_id: str) -> str:
    """已下载过就直接用本地缓存路径，避免每次启动都联网检查 HuggingFace。"""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return repo_id


def _device() -> str:
    import os

    import torch

    if os.getenv("RAG_DEVICE"):          # 手动指定：cpu / mps / cuda
        return os.environ["RAG_DEVICE"]

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_embedder():
    from FlagEmbedding import BGEM3FlagModel

    dev = _device()
    return BGEM3FlagModel(_local_or_remote(EMBED_MODEL), use_fp16=dev == "cuda", devices=[dev])


@lru_cache(maxsize=1)
def get_reranker():
    from FlagEmbedding import FlagReranker

    dev = _device()
    return FlagReranker(_local_or_remote(RERANK_MODEL), use_fp16=dev in ("cuda", "mps"), devices=[dev])


def encode(texts: list[str], batch_size: int = 16) -> tuple[list[list[float]], list[dict[str, float]]]:
    """返回 (稠密向量, 稀疏权重{token_id: weight})。"""
    out = get_embedder().encode(
        texts, batch_size=batch_size, max_length=1024,
        return_dense=True, return_sparse=True,
    )
    dense = [v.tolist() for v in out["dense_vecs"]]
    sparse = [{k: float(v) for k, v in w.items()} for w in out["lexical_weights"]]
    return dense, sparse


def rerank(query: str, passages: list[str], max_length: int = 512) -> list[float]:
    """max_length 控制截断长度：条文一般几百字，512 token 足够；越长越慢（MPS 上近似线性增长）。"""
    if not passages:
        return []
    scores = get_reranker().compute_score([[query, p] for p in passages], normalize=True,
                                         max_length=max_length, batch_size=8)  # 小批次在 MPS 上更快（少做无效 padding）
    return [float(scores)] if isinstance(scores, float) else [float(s) for s in scores]
