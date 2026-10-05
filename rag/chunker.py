"""切分：按文档版式选择策略，每个 chunk 带出处、页码范围和权限标签。

- 条款式（法律、法规、规章、内部制度）：按「第X条」切分，支持「第二百八十七条之二」，带编/章/节。
  条号必须递增且跳跃不超过 MAX_ARTICLE_GAP，避免把行首的「第十条规定……」误判成新条款。
- 提纲式（外汇局通知、目录）：按「一、二、」切分，过长的再按「（一）（二）」拆开。
- 问答式（答记者问）：按「1、……？」切分，一问一答一块。
- 案例式（典型案例）：一个案例一组，按【基本案情】【典型意义】等小节拆成不超过 MAX_CHARS 的块，
  每块都带上案例标题。
- 兜底：按页切分。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .domain import crime_name
from .parsers import Line, ParsedDoc

CN = "零〇一二三四五六七八九十百千"
_ARTICLE = re.compile(rf"^第([{CN}]+)条(之[一二三四五六七八九十]+)?[\s　]*(.*)$")
_PART = re.compile(rf"^第([{CN}]+)编[\s　]*(.*)$")
_CHAPTER = re.compile(rf"^第([{CN}]+)章[\s　]*(.*)$")
_SECTION = re.compile(rf"^第([{CN}]+)节[\s　]*(.*)$")
_APPENDIX = re.compile(r"^(附\s*\d+|附[:：].{0,40}|附件\s*\d*[:：]?.{0,40})$")
_TOC = re.compile(r"^目[\s　]*录$")
_ITEM = re.compile(r"^[（(][一二三四五六七八九十]+[)）]")
_TOP = re.compile(r"^([一二三四五六七八九十]+)、(.*)$")
_QA = re.compile(r"^(\d+)、(\S.*)$")          # 答记者问的提问行（序号须连续递增），如"5、《解释》就认定非法买卖外汇是如何规定的？"
_CASE = re.compile(r"^案例\s*([一二三四五六七八九十\d]+)\s*[:：]?\s*(.{0,50})$")
_CASE_SUB = re.compile(r"^(【[^】]{2,12}】|[一二三四五六七八九十]+、\s*(基本案情|诉讼过程|典型意义|裁判结果|检察机关履职过程|处理结果|案件办理情况).*)$")
MAX_ARTICLE_GAP = 3
MAX_CHARS = 1200          # 提纲/案例块的目标上限（重排模型最多看约 1024 token）


def cn2int(s: str) -> int:
    if s.isdigit():
        return int(s)
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")} | {"〇": 0}
    units = {"十": 10, "百": 100, "千": 1000}
    total, cur = 0, 0
    for ch in s:
        if ch in digits:
            cur = digits[ch]
        elif ch in units:
            total += (cur or 1) * units[ch]
            cur = 0
    return total + cur


def _squash(s: str) -> str:
    """去掉标题里的排版空格：'总　　则' -> '总则'。"""
    return re.sub(r"[\s　]+", "", s)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    source_file: str
    version: str
    effective_date: str
    doc_no: str
    status: str
    kind: str             # article / outline / case / page / preamble
    chapter: str
    section: str
    article: str          # 如 "第十二条"、"第三项"、"案例二 李某洗钱案·典型意义"
    article_no: int       # 排序用；题注/前言为 0
    alias: str            # 补充的检索标签，如刑法条文的罪名「洗钱罪」
    page_start: int | None
    page_end: int | None
    text: str
    citation: str         # 给用户看的出处，由代码生成，不让大模型编

    def to_record(self) -> dict:
        return asdict(self)


def _join(buf: list[Line]) -> str:
    """PDF 的硬换行要拼回去；列举项（一）（二）、编号行保留换行。"""
    out = ""
    for ln in buf:
        if not out:
            out = ln.text
        elif not ln.page or _ITEM.match(ln.text) or re.match(r"^(\d+[.．、]|【)", ln.text):
            out += "\n" + ln.text
        else:
            out += ln.text
    return out


class _Emitter:
    def __init__(self, doc: ParsedDoc):
        self.doc, self.chunks, self.ids = doc, [], set()

    def emit(self, kind: str, label: str, no: int, buf: list[Line], chapter: str = "", section: str = "",
             text: str | None = None, min_len: int = 1):
        if not buf:
            return
        text = text if text is not None else _join(buf)
        if len(text) < min_len:
            return
        p0, p1 = buf[0].page, buf[-1].page
        parts = [f"《{self.doc.title}》", chapter, section, label]
        if p0 is not None:
            parts.append(f"第{p0}页" if p0 == p1 else f"第{p0}-{p1}页")
        alias = crime_name(self.doc.title, label)
        if alias:
            parts[3] = f"{label}【{alias}】"
        cid, k = f"{self.doc.doc_id}#{label}", 2
        while cid in self.ids:
            cid, k = f"{self.doc.doc_id}#{label}({k})", k + 1
        self.ids.add(cid)
        d = self.doc
        self.chunks.append(Chunk(
            chunk_id=cid, doc_id=d.doc_id, doc_title=d.title, doc_type=d.doc_type,
            source_file=d.source_file, version=d.version, effective_date=d.effective_date,
            doc_no=d.doc_no, status=d.status, kind=kind, chapter=chapter, section=section,
            article=label, article_no=no, alias=alias, page_start=p0, page_end=p1, text=text,
            citation=" ".join(x for x in parts if x),
        ))


# ---------- 条款式 ----------

def _chunk_articles(doc: ParsedDoc, em: _Emitter, warnings: list[str]):
    part = chapter = section = ""
    last_key, last_no = (0, 0), 0
    cur_label, cur_no, cur_key, buf = "题注", 0, (0, 0), []
    pending = None          # "part"/"chapter"/"section"：标题文字在下一行
    in_toc, toc_first = False, None

    def chap():
        return " ".join(x for x in (part, chapter) if x)

    def flush():
        if cur_no == -1 or not buf:
            return
        if cur_no == 0:
            em.emit("preamble", "题注", 0, buf, chap(), section, min_len=40)
        else:
            em.emit("article", cur_label, cur_no, buf, chap(), section, text=f"{cur_label} {_join(buf)}".strip())

    for ln in doc.lines:
        t = ln.text
        if pending:
            if not (_ARTICLE.match(t) or _SECTION.match(t) or _CHAPTER.match(t) or _PART.match(t)) and len(t) <= 25:
                if pending == "part":
                    part += " " + _squash(t)
                elif pending == "chapter":
                    chapter += " " + _squash(t)
                else:
                    section += " " + _squash(t)
                pending = None
                continue
            pending = None

        if _TOC.match(t):
            in_toc = True
            continue
        if in_toc:
            # 目录只含编/章/节标题：记住第一项，正文里它再次出现时目录结束
            if _PART.match(t) or _CHAPTER.match(t):
                if toc_first is None:
                    toc_first = _squash(t)
                    continue
                if _squash(t) != toc_first:
                    continue
                in_toc = False
            else:
                continue

        if last_no > 0 and _APPENDIX.match(t):
            break

        heading = None
        for kind, rx in (("part", _PART), ("chapter", _CHAPTER), ("section", _SECTION)):
            m = rx.match(t)
            if m and len(t) <= 30:
                heading = (kind, m)
                break
        if heading:
            kind, m = heading
            flush(); buf = []
            name = f"第{m.group(1)}{ {'part': '编', 'chapter': '章', 'section': '节'}[kind] } {_squash(m.group(2))}".strip()
            if kind == "part":
                part, chapter, section = name, "", ""
            elif kind == "chapter":
                chapter, section = name, ""
            else:
                section = name
            pending = None if m.group(2) else kind
            cur_no = -1
            continue

        m = _ARTICLE.match(t)
        if m:
            no = cn2int(m.group(1))
            sub = cn2int(m.group(2)[1:]) if m.group(2) else 0
            key = (no, sub)
            if key == cur_key and cur_no > 0:
                # 网页 PDF 常见：「第七条」单独成行，下一行又以「第七条　……」开头
                if m.group(3):
                    buf.append(Line(m.group(3), ln.page))
                continue
            if key > last_key and no <= last_no + MAX_ARTICLE_GAP:
                if sub == 0 and no != last_no + 1:
                    warnings.append(f"{doc.doc_id}: 条号跳跃 {last_no} -> {no}")
                flush()
                cur_label = f"第{m.group(1)}条{m.group(2) or ''}"
                cur_no, cur_key, last_key, last_no = no, key, key, no
                buf = [Line(m.group(3), ln.page)] if m.group(3) else []
                continue
        if cur_no == -1:
            continue  # 编/章/节标题后、第一条前的零碎文字忽略
        buf.append(ln)
    flush()
    if in_toc:
        warnings.append(f"{doc.doc_id}: 目录未正常结束，请检查")


# ---------- 按长度打包 ----------

def _pack(groups: list[list[Line]], limit: int = MAX_CHARS) -> list[list[Line]]:
    """把若干行组依次装箱，每箱总字数不超过 limit（单组超长则单独成箱）。"""
    # 单组超长（如整段案情、表格）先按行切成若干组
    split: list[list[Line]] = []
    for g in groups:
        if sum(len(l.text) for l in g) <= limit:
            split.append(g)
            continue
        part, n = [], 0
        for ln in g:
            if part and n + len(ln.text) > limit:
                split.append(part)
                part, n = [], 0
            part.append(ln)
            n += len(ln.text)
        if part:
            split.append(part)
    out, cur, size = [], [], 0
    for g in split:
        n = sum(len(l.text) for l in g)
        if cur and size + n > limit:
            out.append(cur)
            cur, size = [], 0
        cur, size = cur + g, size + n
    if cur:
        out.append(cur)
    return out


def _split_by(lines: list[Line], rx: re.Pattern) -> list[list[Line]]:
    groups: list[list[Line]] = []
    for ln in lines:
        if rx.match(ln.text) or not groups:
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


# ---------- 提纲式 ----------

def _chunk_outline(doc: ParsedDoc, em: _Emitter, rx: re.Pattern = _TOP, unit: str = "项"):
    sections: list[tuple[str, int, list[Line]]] = []
    pre: list[Line] = []
    expect = 1
    for ln in doc.lines:
        m = rx.match(ln.text)
        if m and cn2int(m.group(1)) == expect:
            sections.append((m.group(1), expect, [ln]))
            expect += 1
        elif sections:
            sections[-1][2].append(ln)
        else:
            pre.append(ln)
    em.emit("preamble", "前言", 0, pre, min_len=40)
    for cn, no, lines in sections:
        label = f"第{cn}{unit}"
        heading = lines[0].text[:30]
        packs = _pack(_split_by(lines, _ITEM))
        for k, pack in enumerate(packs):
            text = _join(pack)
            if k > 0:
                text = f"（接{label}：{heading}……）\n{text}"
            em.emit("outline", label if len(packs) == 1 else f"{label}-{k + 1}", no, pack, text=text)


# ---------- 案例式 ----------

def _chunk_cases(doc: ParsedDoc, em: _Emitter):
    lines = doc.lines
    starts: list[tuple[int, int, str]] = []        # (行号, 案例序号, 标题)
    seq = 0
    for i, ln in enumerate(lines):
        m = _CASE.match(ln.text)
        nxt = lines[i + 1].text if i + 1 < len(lines) else ""
        if m:
            starts.append((i, cn2int(m.group(1)), m.group(2).strip()))
        elif nxt.startswith("【关键词】") and len(ln.text) <= 40 and not ln.text.startswith("【"):
            seq += 1
            starts.append((i, seq, ln.text))
    # 目录里的「案例1: 标题」与正文「案例一」同号：用目录补标题，丢掉只有标题的目录项
    titles = {n: t for _, n, t in starts if t}
    bodies = []
    for k, (i, n, t) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        body = lines[i + 1:end]
        if sum(len(l.text) for l in body) >= 80:
            bodies.append((n, t or titles.get(n, ""), lines[i], body))
    em.emit("preamble", "说明", 0, lines[:starts[0][0]] if starts else lines, min_len=80)

    for n, title, head, body in bodies:
        name = f"案例{n} {title}".strip()
        groups = _split_by(body, _CASE_SUB)
        for k, pack in enumerate(_pack(groups)):
            subs = [l.text.strip("【】") for l in pack if _CASE_SUB.match(l.text)]
            sub = "、".join(dict.fromkeys(s.split("、")[-1][:8] for s in subs)) or "正文"
            text = f"{name}｜{sub}\n{_join(pack)}"
            em.emit("case", f"{name}·{sub}", n, [head] + pack if k == 0 else pack, text=text)


# ---------- 兜底：按页 ----------

def _chunk_pages(doc: ParsedDoc, em: _Emitter):
    pages: dict = {}
    for ln in doc.lines:
        pages.setdefault(ln.page, []).append(ln)
    for k, pack in enumerate(_pack(list(pages.values()), limit=1000), 1):
        label = f"第{pack[0].page}页" if pack[0].page else f"第{k}段"
        em.emit("page", label, k, pack)


# ---------- 入口 ----------

def chunk_document(doc: ParsedDoc) -> tuple[list[Chunk], list[str]]:
    warnings: list[str] = []
    em = _Emitter(doc)
    n_articles = sum(bool(_ARTICLE.match(l.text)) for l in doc.lines)
    n_top = sum(bool(_TOP.match(l.text)) for l in doc.lines)
    total = sum(len(l.text) for l in doc.lines)
    if doc.doc_type != "典型案例" and n_articles < 5 and total <= MAX_CHARS:
        em.emit("page", "全文", 1, doc.lines)          # 短文件（如敏感行业目录）整篇一块，保留上下文
    elif doc.doc_type == "典型案例":
        _chunk_cases(doc, em)
    elif n_articles >= 5:
        _chunk_articles(doc, em, warnings)
    elif n_top >= 2:
        _chunk_outline(doc, em)
    elif sum(bool(_QA.match(l.text)) for l in doc.lines) >= 3:
        _chunk_outline(doc, em, _QA, "问")             # 答记者问：一问一答一块
    else:
        _chunk_pages(doc, em)
    if not em.chunks:
        warnings.append(f"{doc.doc_id}: 没有切出任何块")
    return em.chunks, warnings
