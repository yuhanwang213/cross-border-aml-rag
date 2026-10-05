"""问答主流程。

两种模式：
- 问答（qa）：检索 → DeepSeek 生成（JSON，必须引用）→ 代码校验引用 → 输出。
- 场景评估（scenario）：用户描述想做的事 → 大模型拆解成若干合规问题 → 逐个检索法规并匹配案例
  → 大模型给出结构化评估（结论、适用规定、手续与限额、风险与责任、建议）→ 代码校验引用 → 输出。
两种模式下，命中的典型案例都会单独展示（CaseMatch），并把最相关的片段交给模型作参考。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

from .config import LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, REFUSAL_TEXT, ROOT, SCENARIO_CASE_THRESHOLD
from .retriever import CaseMatch, Hit, Retriever, SearchResult
from .verify import CheckedCitation, verify

_COMMON_RULES = f"""只能依据用户消息中【资料】里的内容回答，禁止使用资料以外的知识。
- 每个结论后用 [S编号] 标注依据；并在 citations 中为每个用到的来源给出一句从原文中**逐字摘抄**的引句
  （不得改写，可用"……"省略中间部分）。
- 所引条款中与问题相关的金额、期限、额度、审批/登记要求、例外情形都要写出，不要只摘一半。
- 遇到"原则上不得……，如因特殊情况确需……""除……外"这类"原则 + 例外"条款：先写原则；
  再对照条款列举的例外条件判断用户情形是否符合，不符合或无法判断时，结论按原则下，
  不得把例外途径当成常规可行的办法（例如旅游、购物不属于携带大额现钞出境的"特殊情况"）。
- 判断某人/某事项是否落在条款适用范围内，必须对照资料中的定义条款；定义未列入的要明确指出。
- 结论优先依据法律、行政法规、司法解释、部门规章和规范性文件；【典型案例】只能作为"实务参考"，
  不能单独作为规定依据，引用时要写明"参考案例"。
  【权威解读】（如两高负责人答记者问）可用来解释概念、作案手法和条文含义，引用时写明"权威解读"；
  定罪、处罚等结论仍要引用其所解读的条文本身。
- 认定洗钱罪（刑法第一百九十一条）时，上游犯罪必须属于资料列明的七类：毒品犯罪、黑社会性质的组织犯罪、
  恐怖活动犯罪、走私犯罪、贪污贿赂犯罪、破坏金融管理秩序犯罪、金融诈骗犯罪。按刑法分则的章节归类：
  · 破坏金融管理秩序犯罪 = 刑法第三章第四节"破坏金融管理秩序罪"各罪（如非法吸收公众存款罪、逃汇罪、妨害信用卡管理罪）；
  · 金融诈骗犯罪 = 第三章第五节"金融诈骗罪"各罪（如集资诈骗罪、贷款诈骗罪、信用卡诈骗罪、保险诈骗罪）；
  · 贪污贿赂犯罪 = 第八章"贪污贿赂罪"各罪；走私犯罪 = 第三章第二节"走私罪"各罪；
  · 不属于七类的常见犯罪：第三章第六节"危害税收征管罪"（逃税、骗取出口退税、虚开发票）、诈骗罪（第二百六十六条）、赌博罪等。
  非法集资（非法吸收公众存款、集资诈骗）属于七类，可以构成洗钱罪。资料中条文出处写有所在章节时，以出处为准。
  不属于七类的，应说明一般不构成洗钱罪，并指出可能适用的掩饰、隐瞒犯罪所得、犯罪所得收益罪（刑法第三百一十二条）；
  无法确定上游犯罪具体罪名时，分别说明两种情形。
- 用户问是否构成某罪、是否违法时，结论句要写明"构成/可能构成/一般不构成××罪"或"违法/不违法"，
  不要只写"属于""是"；罪名要写法定全称（以资料中司法解释、案例的写法为准，如"帮助信息网络犯罪活动罪"），
  不得自行缩写或改字；回答"知识库中没有相关规定"之前，先确认资料中确实没有对应条文（如罚则、处罚条款）。
- 回答认定或描述的行为可能违法、犯罪时（如分拆购汇、出租出借账户、跑分、对敲），即使用户只问手法、概念或运作方式，
  也要用一两句写明法律后果（行政处罚、可能触犯的罪名），并引用资料中对应的条文；资料中没有对应条文的，不得补写条号。
  只写回答中认定的那种行为的后果，不要顺带列出其他违法行为的罚则。用户描述的是正常合规行为
  （如为留学购汇汇款）时，不要附加违法后果；最多提醒与其做法直接相关的禁止情形（如超额度时不得分拆、借用他人额度），
  不要写非法买卖外汇、洗钱等与其情形无关的处罚。
- 认定某行为属于非法买卖外汇（含倒买倒卖、变相买卖外汇，如"对敲"、资金跨境兑付）或非法从事资金支付结算业务时，
  要同时写出法律后果：情节严重的按非法经营罪处罚，引用刑法第二百二十五条和两高《关于办理非法从事资金支付结算业务、
  非法买卖外汇刑事案件适用法律若干问题的解释》第二条（支付结算为第一条）；资料中没有这些条文时不得自行补写条号。
- 标注"已修改/已废止"的文件，要提示用户该文件可能已被修改或废止、需核实现行版本。
- 不同文件规定不一致时，如实列出并说明各自的适用范围。
- 不要编造条款号、金额、期限；出处以资料标题为准，不要自己写页码。
- 资料不能回答（包括只沾边、需要推测、资料之外的知识）时：found 设为 false，citations 为空。"""

QA_PROMPT = f"""你是资金出境与反洗钱合规问答助手。{_COMMON_RULES}
- 篇幅：第一句直接给结论，全文一般不超过 400 字；要点多时用编号分条，只保留回答问题所必需的内容，
  不要逐字照抄长条文，也不要罗列与问题无关的相邻条款。
found 为 false 时 answer 固定为"{REFUSAL_TEXT}"。

只输出 JSON：
{{"found": true/false, "answer": "……[S1]……", "citations": [{{"source": "S1", "quote": "原文引句"}}]}}"""

ANALYZE_PROMPT = """你是资金出境与反洗钱合规分析师。用户会描述自己想做的事。
请提炼关键事实，并拆解出需要查询法规的合规问题（2～5 个），每个问题写成一句适合检索法规的规范表述
（使用法律术语，如"个人购汇年度便利化额度""携带外币现钞出境申报""境外直接投资外汇登记""分拆购汇"
"可疑交易报告""洗钱罪""非法买卖外汇"等）。只做拆解，不要给结论。

只输出 JSON：
{"summary": "一句话概括用户想做的事", "facts": ["主体：个人/企业…", "金额：…", "币种/方向：…", "目的：…"],
 "issues": [{"topic": "简短主题", "query": "检索用的问题"}]}"""

SCENARIO_PROMPT = f"""你是资金出境与反洗钱合规分析师，要对用户想做的事给出合规评估。{_COMMON_RULES}

评估要求：
- verdict 从以下选一个："合规可行" / "有条件可行（需办理手续或受额度限制）" / "存在较高合规风险" / "违法或被禁止" / "依据不足，无法判断"。
- summary：两三句话说明结论和最关键的理由（带 [S编号]）。
- sections 固定四节：适用的规定、需要办理的手续与额度限制、风险与法律责任、合规建议。
  只写与用户情形直接相关的内容；与本情形无关的资料（哪怕检索到了）不要引用、不要复述，
  也不要列举只在其他情形下才适用的罚则（例如用户并未买卖外汇，就不要写非法买卖外汇的处罚）。
  verdict 为"合规可行"或"有条件可行"时，"风险与法律责任"只写用户若违反所述额度、真实性要求（如分拆、借用他人额度、
  提供虚假材料）会怎样，不写其他违法行为的罚则，也不要提及并说明"某资料与本情形不同、不适用"。
  每节 2～4 个要点，每个要点一两句话；全文（summary + 四节）不得超过 800 字，按 600～700 字写。
  summary 不超过 120 字；各节不要重复 summary 或其他节已写过的内容（如同一条规定的复述）。
  "合规建议"要从所引规定推导出可操作的做法（例如走哪种合法渠道、准备什么材料、哪些做法必须避免），同样要标注依据；
  其他各节资料中确实没有依据时，content 写"知识库中没有相关规定"，不得编造。
- 用户的做法如果符合资料中明确禁止或重点监管的情形（如分拆、借用他人额度、虚构交易背景、出租出借账户、
  帮助转移犯罪所得），必须在 summary 和"风险与法律责任"中点名该情形并引用对应条款。
- 相关典型案例要在"风险与法律责任"里点明是"参考案例"，说明它与用户情形的相似点。
found 为 false 时 summary 固定为"{REFUSAL_TEXT}"，sections 为空。

只输出 JSON：
{{"found": true/false, "verdict": "…", "summary": "……[S1]……",
  "sections": [{{"title": "适用的规定", "content": "……[S2]……"}}, {{"title": "需要办理的手续与额度限制", "content": "……"}},
               {{"title": "风险与法律责任", "content": "……"}}, {{"title": "合规建议", "content": "……"}}],
  "citations": [{{"source": "S1", "quote": "原文引句"}}]}}"""

# 「我想/打算/准备……」「能不能把……」这类描述自身计划的输入，自动走场景评估
_SCENARIO_RX = re.compile(
    r"(我|我们|本人|我司|我公司|公司|客户|朋友|家人|老板).{0,6}(想|打算|准备|计划|要|需要|希望)|"
    r"(让|叫|请|托)我.{0,12}(帮|把|借|租|卖|转|换|汇|代|收)|"
    r"(能不能|能否|可不可以|可以|是否可以|允许).{0,20}(汇|转|带|寄|投资|买|购|换|收|付|借|注册|开户|提现|存)|"
    r"(我|我们).{0,4}(能|可以).{0,12}吗|会不会.{0,6}(违法|犯罪|被查|有风险)|(合法|违法|犯法)吗")
MAX_SCENARIO_SOURCES = 14
PER_SEARCH_QUOTA = (4, 3)   # 合并时原始描述保底 4 条、每个拆解问题保底 3 条，其余按分数补足
CASES_TO_LLM = 2


# 「我想问/想知道/请问」是在提问，不是描述自己的计划，匹配前先去掉，避免把咨询误判为场景
_ASKING = re.compile(r"(我|就|还)?(想|要)(问问?|知道|了解|请教|咨询)|请问|问一下|问问|"
                     r"需要(办理|提交|准备|满足|具备)?(什么|哪些)")


def is_scenario(query: str) -> bool:
    q = _ASKING.sub("", query)
    return len(q) >= 10 and bool(_SCENARIO_RX.search(q))


@dataclass
class Answer:
    query: str
    mode: str                            # qa / scenario
    text: str                            # 问答：最终回答；场景：summary（[S1] 已换成 [1]）
    refused: bool
    refuse_reason: str = ""              # no_evidence / llm_not_found / citation_failed
    verdict: str = ""                    # 场景评估结论
    sections: list[dict] = field(default_factory=list)      # 场景评估各节 {title, content}
    analysis: dict = field(default_factory=dict)            # 场景拆解结果
    references: list[dict] = field(default_factory=list)    # 通过校验的出处
    cases: list[CaseMatch] = field(default_factory=list)    # 匹配到的典型案例（单独展示）
    checks: list[CheckedCitation] = field(default_factory=list)
    search: SearchResult | None = None
    sub_searches: list[SearchResult] = field(default_factory=list)
    raw_llm: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请在项目根目录 .env 中配置")
    return OpenAI(api_key=key, base_url=LLM_BASE_URL, timeout=90, max_retries=2)  # 网络不通时尽快报错，不要卡住10分钟


def _call_llm(messages: list[dict]) -> dict:
    resp = _client().chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content) | {"_raw": content}
    except json.JSONDecodeError:
        return {"found": False, "_raw": content}


def _build_sources(hits: list[Hit], definitions: list[Hit], cases: list[CaseMatch]) -> tuple[str, dict[str, dict]]:
    items = [(h, []) for h in hits] + [(h, ["定义条款"]) for h in definitions]
    items += [(c.best_chunk, ["实务参考"]) for c in cases[:CASES_TO_LLM]]
    sources, blocks = {}, []
    for i, (h, extra) in enumerate(items, 1):
        sid, m = f"S{i}", h.meta
        sources[sid] = m
        tags = [m.get("doc_type", ""), m.get("status", ""), *extra]
        blocks.append(f"[{sid}] 【{'｜'.join(t for t in tags if t)}】{h.citation}\n{h.text}")
    return "【资料】\n" + "\n\n".join(blocks), sources


def _merge_searches(main: SearchResult, subs: list[SearchResult]) -> list[Hit]:
    """合并多次检索的结果。不同问题的重排分数不可直接比较（子问题用规范术语检索，分数普遍更高），
    只按分数截断会把原始描述里最关键的条款挤掉，所以每次检索先保底几条，再按分数补足。"""
    chosen: dict[str, Hit] = {}
    for res, quota in [(main, PER_SEARCH_QUOTA[0])] + [(s, PER_SEARCH_QUOTA[1]) for s in subs]:
        for h in [h for h in res.hits if h.chunk_id not in chosen][:quota]:
            chosen[h.chunk_id] = h
    rest = sorted((h for r in [main, *subs] for h in r.hits if h.chunk_id not in chosen),
                  key=lambda h: h.score, reverse=True)
    for h in rest:
        if len(chosen) >= MAX_SCENARIO_SOURCES:
            break
        chosen.setdefault(h.chunk_id, h)
    return list(chosen.values())[:MAX_SCENARIO_SOURCES]


def _refuse(ans: Answer, reason: str) -> Answer:
    ans.text, ans.refused, ans.refuse_reason = REFUSAL_TEXT, True, reason
    ans.references, ans.sections, ans.verdict = [], [], ""
    return ans


def _renumber(texts: list[str], valid: list[CheckedCitation], sources: dict) -> tuple[list[str], list[dict]]:
    """按出现顺序把 S 编号重排成 [1][2]…，同一来源多条引句合并。"""
    order: list[str] = []
    for t in texts:
        for sid in re.findall(r"\[S(\d+)\]", t):
            if f"S{sid}" not in order:
                order.append(f"S{sid}")
    for c in valid:
        if c.source not in order:
            order.append(c.source)
    num = {sid: i for i, sid in enumerate(order, 1)}
    out = [re.sub(r"\[S(\d+)\]", lambda m: f"[{num[f'S{m.group(1)}']}]", t).strip() for t in texts]
    refs = [{
        "n": num[sid],
        "chunk_id": sources[sid]["chunk_id"],
        "citation": sources[sid]["citation"],            # 由代码生成的出处
        "doc_type": sources[sid].get("doc_type", ""),
        "quotes": [c.quote for c in valid if c.source == sid],
        "text": sources[sid]["text"],
    } for sid in order]
    return out, refs


class QA:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    def ask(self, query: str, mode: str = "auto", max_retries: int = 1) -> Answer:
        if mode == "auto":
            mode = "scenario" if is_scenario(query) else "qa"
        return self._scenario(query, max_retries) if mode == "scenario" else self._qa(query, max_retries)

    # ---------- 问答 ----------

    def _qa(self, query: str, max_retries: int) -> Answer:
        res = self.retriever.search(query)
        ans = Answer(query=query, mode="qa", text="", refused=False, search=res, cases=res.cases)
        if not res.has_evidence:
            return _refuse(ans, "no_evidence")
        material, sources = _build_sources(res.hits, res.definitions, res.cases)
        messages = [{"role": "system", "content": QA_PROMPT},
                    {"role": "user", "content": f"{material}\n\n【问题】\n{query}"}]
        out = self._generate(ans, messages, sources, lambda o: [o.get("answer", "")], max_retries)
        if out is None:
            return ans
        (ans.text,), ans.references = out
        return ans

    # ---------- 场景评估 ----------

    def _scenario(self, query: str, max_retries: int) -> Answer:
        ans = Answer(query=query, mode="scenario", text="", refused=False)

        plan = _call_llm([{"role": "system", "content": ANALYZE_PROMPT},
                          {"role": "user", "content": query}])
        ans.raw_llm.append(plan.pop("_raw"))
        issues = [i for i in plan.get("issues", []) if isinstance(i, dict) and i.get("query")][:5]
        ans.analysis = {"summary": plan.get("summary", ""), "facts": plan.get("facts", []), "issues": issues}

        # 原始描述检索一次（含案例），每个拆解问题再各检索一次法规
        main = self.retriever.search(query)
        subs = [self.retriever.search(i["query"], top_k=5, with_cases=False) for i in issues]
        ans.search, ans.sub_searches = main, subs
        # 场景评估面向具体做法，门槛更高，避免正当业务（如给孩子汇学费）弹出不相干的犯罪案例
        ans.cases = [c for c in main.cases if c.score >= SCENARIO_CASE_THRESHOLD]

        hits = _merge_searches(main, [s for s in subs if s.has_evidence])
        if not any(r.has_evidence for r in [main, *subs]):
            return _refuse(ans, "no_evidence")

        definitions = self.retriever._definitions_for(hits)
        material, sources = _build_sources(hits, definitions, ans.cases)
        facts = "\n".join(f"- {f}" for f in ans.analysis["facts"])
        messages = [{"role": "system", "content": SCENARIO_PROMPT},
                    {"role": "user", "content": f"{material}\n\n【用户想做的事】\n{query}\n\n【提炼的事实】\n{facts}"}]

        def texts(o):
            secs = [s for s in o.get("sections", []) if isinstance(s, dict)]
            return [o.get("summary", "")] + [s.get("content", "") for s in secs]

        out = self._generate(ans, messages, sources, texts, max_retries)
        if out is None:
            return ans
        cleaned, ans.references = out
        ans.text = cleaned[0]
        secs = [s for s in json.loads(ans.raw_llm[-1]).get("sections", []) if isinstance(s, dict)]
        ans.sections = [{"title": s.get("title", ""), "content": c} for s, c in zip(secs, cleaned[1:])]
        ans.verdict = json.loads(ans.raw_llm[-1]).get("verdict", "")
        return ans

    # ---------- 生成 + 校验（两种模式共用） ----------

    def _generate(self, ans: Answer, messages: list[dict], sources: dict, get_texts, max_retries: int):
        for _ in range(max_retries + 1):
            out = _call_llm(messages)
            ans.raw_llm.append(out["_raw"])
            if not out.get("found"):
                _refuse(ans, "llm_not_found")
                return None
            citations = out.get("citations") or []
            cleaned = [verify(t, citations, sources)[0] for t in get_texts(out)]
            _, checks = verify("", citations, sources)
            ans.checks = checks
            valid = [c for c in checks if c.valid]
            if valid:
                return _renumber(cleaned, valid, sources)
            # 引用全部无效：把校验结果反馈给模型重试一次
            bad = "; ".join(f"{c.source}: {c.reason}" for c in checks) or "没有给出引用"
            messages = messages + [
                {"role": "assistant", "content": out["_raw"]},
                {"role": "user", "content": f"引用校验未通过（{bad}）。请只使用资料中的原文逐字摘抄引句后重新输出 JSON；若资料无法支持，found 设为 false。"},
            ]
        _refuse(ans, "citation_failed")
        return None
