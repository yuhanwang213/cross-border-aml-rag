"""全局配置：路径、检索与生成参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "Regulations"
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# 支持入库的文件类型；其余（如 Regulations/Untitled.ipynb 这类副本）一律忽略
SUPPORTED_SUFFIXES = {".md", ".docx", ".doc", ".pdf"}
# 不入库的子目录（Internal 是早期的示范企业制度，与资金出境/反洗钱主题无关）
EXCLUDE_DIRS = {"Internal"}
# 暂不入库的文件（按文件名）：用于原文有缺字、版本存疑等情况，入库会引出错误引句时先放在这里。
# 2026-09-30 曾排除缺字的《关于惩治骗购外汇……的决定》旧文件，已换成完整原文，名单清空。
EXCLUDE_FILES: set[str] = {
    # 2026-10-01 新加的网页版 PDF，与已入库的 Case/ 下同名 docx（法释〔2019〕1号）内容相同，重复入库会出现两份同样的出处
    "最高人民法院    最高人民检察院      关于办理非法从事资金支付结算业务、      非法买卖外汇刑事案件适用法律      若干问题的解释.pdf",
}
# 文件本身读不出日期/效力状态时手工补充（按文件名）。
# 《海关行政处罚实施条例》现有文件是 2004 年原版（国务院令第420号），2022 年经国务院令第752号修订；
# 经核对第十九条、第二十条两版文字相同，先标"已修改"，回答时提醒核实现行版本。
DOC_META_OVERRIDES: dict[str, dict] = {
    "中华人民共和国海关行政处罚实施条例.pdf": {"effective_date": "2004-11-01", "status": "已修改"},
    # 两高负责人就法释〔2019〕1号答记者问（最高人民法院网），讲解"对敲型"地下钱庄等认定问题；
    # 不是法律规范，单列为"权威解读"，按问答切分
    "依法惩治涉“地下钱庄”犯罪 维护金融市场秩序 ——最高人民法院刑三庭、最高人民检察院法律政策研究室负责人就涉地下钱庄刑事案件司法解释答记者问.pdf":
        {"doc_type": "权威解读", "effective_date": "2019-02-01"},
}

# ---------- 向量化 / 检索 ----------
CHROMA_DIR = DATA_DIR / "chroma"
SPARSE_PATH = DATA_DIR / "sparse.json"   # bge-m3 稀疏权重（关键词通道）
COLLECTION = "regulations"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

DENSE_TOP_N = 30      # 向量通道召回数
SPARSE_TOP_N = 30     # 关键词通道召回数
RERANK_POOL = 24      # 融合后送入重排的候选数（重排是主要耗时）
MAX_INTERPRETATIONS = 2   # 每次检索最多保留的"权威解读"块，避免讲解文字挤掉条文本身
FINAL_TOP_K = 8       # 最终返回的法规条款数（资金出境类问题常需综合多份文件）
RRF_K = 60
# 重排分数(0~1)低于该值视为"知识库中没有相关规定"；需用评测集标定
EVIDENCE_THRESHOLD = 0.3

# 案例通道：相关度达到 CASE_THRESHOLD，或命中案例【关键词】且达到 CASE_KEYWORD_THRESHOLD 时展示
CASE_THRESHOLD = 0.5
SCENARIO_CASE_THRESHOLD = 0.7   # 场景评估里展示案例的门槛（更严格）
CASE_KEYWORD_THRESHOLD = 0.2
MAX_CASES = 3
CASE_POOL = 12        # 案例通道送入重排的候选块数

# ---------- 生成 ----------
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.0
REFUSAL_TEXT = "知识库中没有相关规定。"
MIN_QUOTE_LEN = 6        # 引句太短（如"应当"）无法证明出处，视为无效
