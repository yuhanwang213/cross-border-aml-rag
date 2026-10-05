"""引用校验：纯代码，不依赖大模型自觉。

每条引用必须同时满足：
1. source 编号在本次提供给模型的检索结果里（因而也一定是当前角色有权访问的）；
2. quote 是该条款原文的子串（忽略空白、统一引号；允许用"……"省略中间部分）；
3. quote 不能太短。
正文里的 [S*] 标记若指向未通过校验的来源，会被删掉。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import MIN_QUOTE_LEN

_ELLIPSIS = re.compile(r"……|\.\.\.|…")
_MARK = re.compile(r"\[S(\d+)\]")
_PUNCT = ",.;:、!?。！？"


def normalize(s: str) -> str:
    s = re.sub(r"[\s　]+", "", s)
    return s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "（": "(", "）": ")", "，": ",", "：": ":", "；": ";"}))


def quote_in_text(quote: str, text: str) -> bool:
    """quote 的各片段按顺序出现在原文中。"""
    raw = [normalize(p) for p in _ELLIPSIS.split(quote)]
    # 截断处的标点不参与比对：省略号两侧，以及引句首尾（原文在此处是逗号、摘抄时写成句号等）；
    # 字词和顺序仍须与原文一致
    parts = [p.strip(_PUNCT) for p in raw]
    parts = [p.strip("\"'") for p in parts if p.strip("\"'")]
    if not parts or sum(map(len, parts)) < MIN_QUOTE_LEN:
        return False
    body, pos = normalize(text), 0
    for p in parts:
        i = body.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


def _strip_alias(quote: str, chunk: dict) -> str:
    """出处里带的罪名标签（如【集资诈骗罪】）不在条文正文中，模型摘抄时常连带写上，去掉后再比对。"""
    alias = chunk.get("alias")
    return quote.replace(f"【{alias}】", "") if alias else quote


@dataclass
class CheckedCitation:
    source: str           # "S1"
    chunk_id: str | None
    quote: str
    valid: bool
    reason: str = ""


def verify(answer: str, citations: list[dict], sources: dict[str, dict]) -> tuple[str, list[CheckedCitation]]:
    """sources: {"S1": chunk_dict, ...}。返回 (清理后的答案, 逐条校验结果)。"""
    checked = []
    for c in citations:
        sid = str(c.get("source", "")).strip("[] ")
        quote = str(c.get("quote", ""))
        chunk = sources.get(sid)
        if chunk is None:
            checked.append(CheckedCitation(sid, None, quote, False, "来源编号不存在"))
        elif not quote_in_text(_strip_alias(quote, chunk), chunk["text"]):
            checked.append(CheckedCitation(sid, chunk["chunk_id"], quote, False, "引句与原文不符"))
        else:
            checked.append(CheckedCitation(sid, chunk["chunk_id"], quote, True))

    ok = {c.source for c in checked if c.valid}
    cleaned = _MARK.sub(lambda m: m.group(0) if f"S{m.group(1)}" in ok else "", answer)
    return cleaned, checked
