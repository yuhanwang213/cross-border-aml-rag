"""文档解析：把 md / docx / doc / pdf 统一成「元数据 + 行列表(文本, 页码)」。

- md：读取 front-matter（doc_id、access 等），无页码。
- docx：python-docx 逐段落读取；doc：macOS 自带 textutil 转文本。二者都没有固定页码。
- pdf：PyMuPDF 逐页读取，保留页码；文字层几乎为空的扫描件用 Apple Vision OCR（本地），结果缓存。
- 网页另存的 PDF：去掉导航栏、网址、页码等杂质，并提取发布日期、文号、效力状态。
- 重复文件（同一法规的 docx 与网页 PDF 等）按内容相似度自动去重。
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import DATA_DIR, DOC_META_OVERRIDES, EXCLUDE_DIRS, EXCLUDE_FILES, SUPPORTED_SUFFIXES

OCR_CACHE_DIR = DATA_DIR / "ocr"
REGISTRY_PATH = DATA_DIR / "doc_registry.json"
MIN_CHARS_PER_PAGE = 50     # 低于此值视为扫描件，走 OCR
DUP_THRESHOLD = 0.8         # 内容相似度（字符 6-gram 重合度）超过此值视为重复


@dataclass
class Line:
    text: str
    page: int | None = None


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    source_file: str
    doc_type: str            # 效力层级：法律 / 司法解释 / 行政法规 / 部门规章 / 规范性文件 / 典型案例 / 内部制度
    access: list[str]
    lines: list[Line]
    version: str = ""
    effective_date: str = ""  # 发布或施行日期（尽力提取）
    doc_no: str = ""          # 文号，如 汇发〔2026〕23号
    status: str = ""          # 已修改 / 已废止 等（网页标题里有标注时）
    has_pages: bool = False
    ocr: bool = False
    extra: dict = field(default_factory=dict)


# ---------- 元数据 ----------

_NAME_DATE = re.compile(r"^(?P<title>.+?)_(?P<date>\d{8})$")


def _title_from_name(path: Path) -> tuple[str, str]:
    stem = re.sub(r"^\d+\.", "", path.stem)            # 去掉 "1." 这类序号前缀
    stem = re.sub(r"_(国务院文件|行政规范性文件|跨境债权债务|基本法规)?_?(中国政府网|国家外汇管理局门户网站|中华人民共和国最高人民检察院)$", "", stem)
    stem = re.sub(r"-陕西法院网$", "", stem)
    m = _NAME_DATE.match(stem)
    if m:
        d = m.group("date")
        return m.group("title"), f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return stem, ""


def doc_level(title: str) -> str:
    t = re.sub(r"（[^）]*(修正|修订|年版|最新版)[^）]*）$", "", re.sub(r"^【.*?】", "", title))
    if "案例" in t:
        return "典型案例"
    if t.startswith(("最高人民法院", "最高人民检察院")) and "解释" in t:
        return "司法解释"
    if t.startswith("中华人民共和国") and t.endswith("法"):
        return "法律"
    if t.startswith("全国人民代表大会常务委员会") and t.endswith("决定"):   # 如惩治骗购外汇犯罪的决定（单行刑法）
        return "法律"
    if t.endswith("条例") or t.startswith("国务院关于"):
        return "行政法规"
    if t.endswith(("办法", "规定", "细则")) and "通知" not in t:
        return "部门规章"
    return "规范性文件"


# ---------- 网页 PDF 清洗 ----------

_JUNK = re.compile(
    r"^(首\s*页|主页\s*>.*|当前位置.*|简体中文|繁体中文|English|微信|微博|无障碍.*|政策法规|打印|关闭|"
    r"字号.*|【字号.*|.*本站已支持IPv6.*|请输入关键字|网上接待|新闻发布会|走进皖检|检察要闻|检务公开|检察业务|"
    r"队伍建设|科技强检|检察专题|[｜|]|提交|手机版|简|繁|网站无.*|欢迎来到.*|本院概况|雁检快讯|通知公告|刑事检察|"
    r"最高检新闻|地方动态|直播访谈|图片|专题|法律规章|权威发布|陕西法院|新闻中心|案件快报|法院建设|法官论坛|法院文化|"
    r"裁判文书|审务公开|司法为民|网站公告|专题报道|网上直播|法苑微视界|财务管理|地方联播|索\s*引\s*号.*|分\s*类：.*|"
    r"来\s*源：.*|名\s*称：.*)$"
)
_URL = re.compile(r"^(https?://)?[\w.-]+\.(gov|com|org|net)\.cn/\S*$|^https?://\S+$")
_PRINT_TIME = re.compile(r"^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}$")   # 浏览器打印页眉里的时间
_PAGE_NO = re.compile(r"^([-—–一]\s*\d+\s*[-—–一]|\d+\s*/\s*\d+|第\s*\d+\s*页(\s*共\s*\d+\s*页)?)$")
_DATE = re.compile(r"(?:发布日期|发布时间|时间)[：:]\s*(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})")
_DOC_NO = re.compile(r"文\s*号：\s*(\S+〔\d{4}〕\d+号)")
_STATUS = re.compile(r"（(已修改|已废止|已失效|部分失效)）")


def _clean_pages(pages: list[list[str]]) -> tuple[list[list[str]], dict]:
    meta: dict = {}
    head = "\n".join(l for p in pages[:1] for l in p)
    if m := _DATE.search(head):
        meta["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if m := _DOC_NO.search(head):
        meta["doc_no"] = m.group(1)
    if m := _STATUS.search("\n".join(l for p in pages for l in p[-4:])):
        meta["status"] = m.group(1)

    # 每页都出现的页眉/页脚（如网页标题）
    repeated = set()
    if len(pages) >= 3:
        cnt = Counter(l for p in pages for l in set(p) if len(l) >= 6 and not l.isdigit())
        repeated = {l for l, n in cnt.items() if n >= max(3, len(pages) * 0.5)}

    out = []
    for i, p in enumerate(pages, 1):
        keep = []
        for l in p:
            if (_JUNK.match(l) or _URL.match(l) or _PAGE_NO.match(l) or _PRINT_TIME.match(l) or l in repeated
                    or (l == str(i) and not keep)          # 页首的纯数字页码
                    or _DATE.match(l)):
                continue
            keep.append(l)
        if keep and keep[-1] == str(i):                     # 页尾的纯数字页码
            keep.pop()
        out.append(keep)
    return out, meta


# ---------- 各格式读取 ----------

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def parse_md(path: Path) -> ParsedDoc:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT_MATTER.match(raw)
    meta = yaml.safe_load(m.group(1)) if m else {}
    body = raw[m.end():] if m else raw
    lines = [Line(l.lstrip("#").strip()) for l in body.splitlines() if l.strip()]
    return ParsedDoc(
        doc_id=meta.get("doc_id", path.stem),
        title=meta.get("title", path.stem),
        source_file=str(path),
        doc_type=meta.get("doc_type", "内部制度"),
        access=list(meta.get("access", [])),
        version=str(meta.get("version", "")),
        effective_date=str(meta.get("effective_date", "")),
        lines=lines,
    )


def _read_docx(path: Path) -> list[str]:
    import docx  # python-docx

    lines = [p.text.strip() for p in docx.Document(str(path)).paragraphs if p.text.strip()]
    return [_strip_db_junk(l) for l in lines if _strip_db_junk(l)]


# 法规数据库（如北大法宝）导出文件里夹带的推荐栏，如"法宝联想：案例与裁判文书 40 篇……修订沿革"
_DB_JUNK = re.compile(r"(法宝联想：.*|修订沿革|相关法规.*篇.*)$")


def _strip_db_junk(line: str) -> str:
    return _DB_JUNK.sub("", line).strip()


def _read_doc(path: Path) -> list[str]:
    """旧版 .doc：用 macOS 自带 textutil 转纯文本。"""
    txt = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return [l.strip() for l in txt.splitlines() if l.strip() and not re.fullmatch(r"PAGE\s*\d*", l.strip())]


def _ocr_pdf(path: Path) -> list[list[str]]:
    """Apple Vision OCR（本地）。结果按文件内容哈希缓存到 data/ocr/。"""
    import pymupdf
    from ocrmac import ocrmac
    from PIL import Image

    digest = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    cache = OCR_CACHE_DIR / f"{path.stem}_{digest}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    pages = []
    with pymupdf.open(str(path)) as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            res = ocrmac.OCR(img, language_preference=["zh-Hans"], recognition_level="accurate").recognize()
            res.sort(key=lambda r: (-round(r[2][1], 3), r[2][0]))   # bbox 原点在左下：从上到下、从左到右
            pages.append([r[0].strip() for r in res if r[0].strip()])
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


def _read_pdf(path: Path) -> tuple[list[list[str]], bool]:
    import pymupdf

    with pymupdf.open(str(path)) as pdf:
        pages = [[l.strip() for l in p.get_text().splitlines() if l.strip()] for p in pdf]
    chars = sum(len(l) for p in pages for l in p)
    if chars < MIN_CHARS_PER_PAGE * len(pages):
        return _ocr_pdf(path), True
    return pages, False


def parse_file(path: Path, doc_id: str) -> ParsedDoc:
    title, date = _title_from_name(path)
    suffix = path.suffix.lower()
    meta, ocr = {}, False
    if suffix == ".pdf":
        pages, ocr = _read_pdf(path)
        pages, meta = _clean_pages(pages)
        lines = [Line(l, i) for i, p in enumerate(pages, 1) for l in p]
    else:
        raw = _read_docx(path) if suffix == ".docx" else _read_doc(path)
        lines = [Line(l) for l in raw]
    over = DOC_META_OVERRIDES.get(path.name, {})
    return ParsedDoc(
        doc_id=doc_id, title=title, source_file=str(path),
        doc_type=over.get("doc_type") or doc_level(title), access=["全员"],
        effective_date=date or meta.get("date", "") or over.get("effective_date", ""),
        doc_no=meta.get("doc_no", ""),
        status=meta.get("status", "") or over.get("status", ""), lines=lines, has_pages=suffix == ".pdf", ocr=ocr,
    )


# ---------- 稳定的 doc_id ----------

def _registry() -> dict[str, str]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {}


def _assign_id(reg: dict[str, str], key: str) -> str:
    """同一文件（按相对路径）永远拿到同一个 PUB-xxx，新增文件不会打乱已有编号。"""
    if key not in reg:
        used = {int(v.split("-")[1]) for v in reg.values() if v.startswith("PUB-")}
        reg[key] = f"PUB-{max(used, default=0) + 1:03d}"
    return reg[key]


# ---------- 去重 ----------

def _shingles(doc: ParsedDoc, n: int = 6) -> set[str]:
    s = re.sub(r"[\s　，。、；：（）()“”\"'【】《》]", "", "".join(l.text for l in doc.lines))
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 0))}


def _prefer(a: ParsedDoc, b: ParsedDoc) -> ParsedDoc:
    """保留哪一份：docx/doc 优于网页 PDF（无重复标题行等杂质）；同类型取文件名更短的。"""
    rank = {".docx": 0, ".doc": 1, ".md": 1, ".pdf": 2}
    ka = (rank[Path(a.source_file).suffix.lower()], len(Path(a.source_file).name))
    kb = (rank[Path(b.source_file).suffix.lower()], len(Path(b.source_file).name))
    return a if ka <= kb else b


def dedupe(docs: list[ParsedDoc]) -> tuple[list[ParsedDoc], list[tuple[str, str, float]]]:
    sh = [_shingles(d) for d in docs]
    dropped: set[int] = set()
    report = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if i in dropped or j in dropped or not sh[i] or not sh[j]:
                continue
            sim = len(sh[i] & sh[j]) / min(len(sh[i]), len(sh[j]))
            if sim >= DUP_THRESHOLD:
                keep = _prefer(docs[i], docs[j])
                drop = j if keep is docs[i] else i
                dropped.add(drop)
                report.append((docs[drop].source_file, keep.source_file, sim))
    return [d for k, d in enumerate(docs) if k not in dropped], report


# ---------- 入口 ----------

def load_documents(docs_dir: Path) -> tuple[list[ParsedDoc], list[Path], list[tuple[str, str, float]]]:
    """解析目录下所有支持的文档。返回 (文档列表, 跳过的文件, 去重记录)。"""
    docs, skipped = [], []
    reg = _registry()
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if (".ipynb_checkpoints" in path.parts or EXCLUDE_DIRS & set(path.relative_to(docs_dir).parts)
                or path.name in EXCLUDE_FILES
                or path.suffix.lower() not in SUPPORTED_SUFFIXES):
            skipped.append(path)
            continue
        if path.suffix.lower() == ".md":
            docs.append(parse_md(path))
        else:
            docs.append(parse_file(path, _assign_id(reg, str(path.relative_to(docs_dir)))))
    DATA_DIR.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    docs, dup = dedupe(docs)
    return docs, skipped, dup
