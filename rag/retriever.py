"""混合检索：向量 + 关键词(bge-m3 稀疏) → RRF 融合 → 重排 → 证据阈值。

两个通道分开检索：
- 法规通道：法律、法规、司法解释、规章、规范性文件，作为回答依据；
- 案例通道：典型案例，按案例聚合后打分，达到阈值（或命中案例【关键词】）才展示，作为实务参考。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import (
    CASE_KEYWORD_THRESHOLD, CASE_THRESHOLD, DENSE_TOP_N, EVIDENCE_THRESHOLD, FINAL_TOP_K,
    CASE_POOL, MAX_CASES, MAX_INTERPRETATIONS, RERANK_POOL, RRF_K, SPARSE_PATH, SPARSE_TOP_N,
)
from .domain import expand_query, required_articles
from .index import embed_text, get_collection, load_chunks
from .models import encode, rerank

CASE_TYPE = "典型案例"
INTERPRETATION_TYPE = "权威解读"


@dataclass
class Hit:
    chunk_id: str
    citation: str
    text: str
    score: float          # 重排分数 0~1
    dense_rank: int | None
    sparse_rank: int | None
    meta: dict


@dataclass
class CaseMatch:
    case_id: str          # doc_id#案例序号
    title: str            # 如 "案例1 黄某洗钱案"
    source: str           # 发布文件名
    citation: str         # 《文件名》 案例1 黄某洗钱案 第2-3页
    score: float
    keywords: list[str]   # 案例【关键词】
    matched: list[str]    # 与问题命中的关键词
    sections: dict[str, str]
    full_text: str
    best_chunk: Hit       # 与问题最相关的一段，交给大模型作参考


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]                     # 法规条款
    has_evidence: bool                  # False 时上层应直接回答"知识库中没有相关规定"
    cases: list[CaseMatch] = field(default_factory=list)
    definitions: list[Hit] = field(default_factory=list)  # 命中条款里用到的术语的定义条款
    expanded_query: str = ""


# 定义条款：「本办法所称受益所有人，是指……」
_DEF_SUOCHENG = re.compile(r"所称([^，,。；（(“”\"]{2,15}?)[，,]?(?:是指|包括|指)")
_DEF_LEAD = re.compile(r"^第[零〇一二三四五六七八九十百千]+条\s*([^，,。；：\s]{2,10}?)(?:是指|包括)")
MAX_DEFINITIONS = 2
_CASE_SEC = re.compile(r"^(?:【([^】]{2,12})】|[一二三四五六七八九十]+、\s*(基本案情|诉讼过程|典型意义|裁判结果|检察机关履职过程|处理结果))\s*(.*)$")


class Retriever:
    def __init__(self):
        self.col = get_collection()
        self.chunks = {c["chunk_id"]: c for c in load_chunks()}
        self.sparse = json.loads(SPARSE_PATH.read_text(encoding="utf-8"))
        self.defs = self._build_definitions()
        self.case_groups = self._build_cases()

    # ---------- 定义条款 ----------

    def _build_definitions(self) -> dict[str, list[tuple[str, str]]]:
        """{doc_id: [(术语, 定义条款 chunk_id)]}。过于泛化的术语（如公司法里的「公司」）不收录。"""
        by_doc: dict[str, list[dict]] = {}
        for c in self.chunks.values():
            by_doc.setdefault(c["doc_id"], []).append(c)
        defs: dict[str, list[tuple[str, str]]] = {}
        for doc_id, chunks in by_doc.items():
            for c in chunks:
                terms = set(_DEF_SUOCHENG.findall(c["text"])) | set(_DEF_LEAD.findall(c["text"]))
                for t in terms:
                    t = re.sub(r"^(本\S{0,3}所称|公司的)", "", t)
                    usage = sum(t in x["text"] for x in chunks) / len(chunks)
                    if len(t) >= 2 and usage <= 0.2:
                        defs.setdefault(doc_id, []).append((t, c["chunk_id"]))
        return defs

    def _definitions_for(self, hits: list[Hit]) -> list[Hit]:
        """命中条款用到了同文件中有定义的术语，就把定义条款一并带上，避免模型想当然。"""
        have = {h.chunk_id for h in hits}
        out: list[Hit] = []
        for h in hits:
            for term, cid in self.defs.get(h.meta["doc_id"], []):
                if cid in have or term not in h.text:
                    continue
                c = self.chunks[cid]
                out.append(Hit(cid, c["citation"], c["text"], 0.0, None, None, c))
                have.add(cid)
                if len(out) >= MAX_DEFINITIONS:
                    return out
        return out

    # ---------- 案例聚合 ----------

    def _build_cases(self) -> dict[str, dict]:
        """把同一案例的多个块聚合：{case_id: {title, source, chunk_ids, full_text, keywords, sections, pages}}。"""
        groups: dict[str, dict] = {}
        for cid, c in self.chunks.items():
            if c["kind"] != "case":
                continue
            key = f"{c['doc_id']}#{c['article_no']}"
            title = c["article"].split("·")[0]
            g = groups.setdefault(key, {"title": title, "source": c["doc_title"], "chunk_ids": [],
                                        "parts": [], "pages": []})
            g["chunk_ids"].append(cid)
            g["parts"].append(c["text"].split("\n", 1)[-1])      # 去掉「案例名｜小节」抬头
            g["pages"] += [p for p in (c["page_start"], c["page_end"]) if p]
        for key, g in groups.items():
            full = "\n".join(g.pop("parts"))
            g["full_text"] = full
            g["sections"] = self._case_sections(full)
            kw = g["sections"].get("关键词", "")
            g["keywords"] = [k for k in re.split(r"[\s　、，,；;]+", kw) if len(k) >= 2]
            pages = g.pop("pages")
            p = (f" 第{min(pages)}页" if min(pages) == max(pages) else f" 第{min(pages)}-{max(pages)}页") if pages else ""
            g["citation"] = f"《{g['source']}》 {g['title']}{p}"
        return groups

    @staticmethod
    def _case_sections(full: str) -> dict[str, str]:
        secs: dict[str, list[str]] = {}
        cur = "正文"
        for line in full.split("\n"):
            m = _CASE_SEC.match(line.strip())
            if m:
                cur = (m.group(1) or m.group(2)).strip()
                line = m.group(3)
            if line.strip():
                secs.setdefault(cur, []).append(line.strip())
        return {k: "\n".join(v) for k, v in secs.items()}

    # ---------- 召回 ----------

    def _dense(self, qvec, cases: bool) -> list[str]:
        where = {"doc_type": CASE_TYPE} if cases else {"doc_type": {"$ne": CASE_TYPE}}
        res = self.col.query(query_embeddings=[qvec], n_results=DENSE_TOP_N, where=where, include=[])
        return res["ids"][0]

    def _sparse(self, qw: dict[str, float], cases: bool) -> list[str]:
        scores = []
        for cid, dw in self.sparse.items():
            if (self.chunks[cid]["doc_type"] == CASE_TYPE) != cases:
                continue
            s = sum(w * dw[t] for t, w in qw.items() if t in dw)
            if s > 0:
                scores.append((s, cid))
        scores.sort(reverse=True)
        return [cid for _, cid in scores[:SPARSE_TOP_N]]

    @staticmethod
    def _fuse(*lists: list[str]) -> list[str]:
        fused: dict[str, float] = {}
        for ids in lists:
            for rank, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)

    @staticmethod
    def _short_title(title: str) -> str:
        title = title.removeprefix("中华人民共和国")
        return re.sub(r"（.*?）$", "", title)

    def _pinned(self, query: str) -> list[str]:
        """问题里明确点名「某法第X条」时，直接把该条放进候选并置顶。
        向量/重排对条号不敏感（第十条、第十一条语义几乎一样），需要规则兜底。"""
        arts = set(re.findall(r"第[零〇一二三四五六七八九十百千]+条(?:之[一二三四五六七八九十]+)?", query))
        if not arts:
            return []
        return [cid for cid, c in self.chunks.items()
                if c["article"] in arts and self._short_title(c["doc_title"]) in query]

    def _required(self, query: str) -> list[str]:
        """领域规则要求必须交给模型的条文（见 domain.required_articles）。"""
        want = required_articles(query)
        return [cid for cid, c in self.chunks.items() if (c["doc_title"], c["article"]) in want]

    def _hit(self, cid: str, score: float, dense: list[str], sparse: list[str]) -> Hit:
        c = self.chunks[cid]
        return Hit(cid, c["citation"], c["text"], score,
                   dense.index(cid) + 1 if cid in dense else None,
                   sparse.index(cid) + 1 if cid in sparse else None, c)

    def search(self, query: str, top_k: int = FINAL_TOP_K, with_cases: bool = True) -> SearchResult:
        q = expand_query(query)          # 口语词扩展成规范术语，仅用于召回和重排
        (qvec,), (qw,) = encode([q])

        # 法规通道
        dense, sparse = self._dense(qvec, False), self._sparse(qw, False)
        pinned = self._pinned(query)
        required = self._required(query)
        pool = pinned + [cid for cid in self._fuse(dense, sparse)[:RERANK_POOL] if cid not in pinned]
        pool += [cid for cid in required if cid not in pool]
        # 重排时带上文件名、章节、罪名，短条文（如目录里的一项）才不会因缺上下文被低估
        scores = rerank(q, [embed_text(self.chunks[cid]) for cid in pool])
        ranked = sorted(zip(pool, scores), key=lambda x: (x[0] in pinned, x[1]), reverse=True)
        # 权威解读（答记者问）文字长、术语密，重排分常高于条文；只留分数最高的几块（必带的除外）
        interp = [cid for cid, _ in ranked
                  if self.chunks[cid]["doc_type"] == INTERPRETATION_TYPE and cid not in required]
        ranked = [x for x in ranked if x[0] not in interp[MAX_INTERPRETATIONS:]]
        top, rest = ranked[:top_k], ranked[top_k:]
        # 必带条文（如问"是否洗钱"时的刑法第一百九十一条、第三百一十二条）没进前 top_k 的，追加在后面，
        # 不挤掉原有结果
        ranked = top + [x for x in rest if x[0] in required]
        hits = [self._hit(cid, s, dense, sparse) for cid, s in ranked]

        cases = self._search_cases(q, qvec, qw) if with_cases else []
        top = max([h.score for h in hits[:1]] + [c.score for c in cases[:1]], default=0.0)
        has_evidence = top >= EVIDENCE_THRESHOLD
        return SearchResult(query=query, hits=hits, has_evidence=has_evidence, cases=cases,
                            definitions=self._definitions_for(hits) if has_evidence else [],
                            expanded_query=q)

    def _search_cases(self, q: str, qvec, qw) -> list[CaseMatch]:
        dense, sparse = self._dense(qvec, True), self._sparse(qw, True)
        pool = [cid for cid in self._fuse(dense, sparse) if self.chunks[cid]["kind"] == "case"][:CASE_POOL]
        if not pool:
            return []
        scores = dict(zip(pool, rerank(q, [embed_text(self.chunks[cid]) for cid in pool])))

        best: dict[str, tuple[float, str]] = {}
        for cid, s in scores.items():
            c = self.chunks[cid]
            key = f"{c['doc_id']}#{c['article_no']}"
            if key not in best or s > best[key][0]:
                best[key] = (s, cid)

        out = []
        for key, (s, cid) in best.items():
            g = self.case_groups[key]
            matched = [k for k in g["keywords"] if k in q]
            # 触发条件：相关度足够高；或命中案例关键词且相关度过得去
            if s >= CASE_THRESHOLD or (matched and s >= CASE_KEYWORD_THRESHOLD):
                out.append(CaseMatch(
                    case_id=key, title=g["title"], source=g["source"], citation=g["citation"],
                    score=s, keywords=g["keywords"], matched=matched, sections=g["sections"],
                    full_text=g["full_text"], best_chunk=self._hit(cid, s, dense, sparse),
                ))
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:MAX_CASES]
