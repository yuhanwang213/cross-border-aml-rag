"""资金出境与反洗钱合规问答助手 —— Streamlit 界面。

启动：streamlit run app.py
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")

import streamlit as st  # noqa: E402

from rag.config import EVIDENCE_THRESHOLD  # noqa: E402
from rag.generator import QA  # noqa: E402

REASONS = {
    "no_evidence": "检索分数低于阈值，未找到依据",
    "llm_not_found": "模型判断资料不足以回答",
    "citation_failed": "引用校验未通过，已拒绝输出",
}
VERDICT_ICON = {"合规可行": "🟢", "有条件可行": "🟡", "存在较高合规风险": "🟠", "违法或被禁止": "🔴"}
MODES = {"自动识别": "auto", "法规问答": "qa", "场景评估（描述你想做的事）": "scenario"}
CASE_SECTIONS = ["关键词", "基本案情", "裁判结果", "诉讼过程", "检察机关履职过程", "处理结果", "典型意义"]

st.set_page_config(page_title="资金出境与反洗钱合规问答", page_icon="📘", layout="wide")


@st.cache_resource(show_spinner="正在加载检索模型（首次约需 10–20 秒）…")
def get_qa() -> QA:
    qa = QA()
    qa.retriever.search("预热")  # 启动时加载好两个模型，避免第一次提问时等待
    return qa


@st.cache_data
def doc_stats() -> dict[str, list[str]]:
    by_type: dict[str, set] = {}
    for c in get_qa().retriever.chunks.values():
        by_type.setdefault(c["doc_type"], set()).add(c["doc_title"])
    return {k: sorted(v) for k, v in sorted(by_type.items())}


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("模式")
    mode = MODES[st.radio("模式", list(MODES), label_visibility="collapsed")]
    st.caption("场景评估：直接描述你打算做的事，例如“我想把 80 万人民币换成美元汇给在美国读书的孩子”，"
               "系统会拆解合规要点、匹配法规和相关案例。")
    st.divider()
    st.caption("知识库")
    for t, titles in doc_stats().items():
        with st.expander(f"{t}（{len(titles)}）"):
            for x in titles:
                st.markdown(f"- {x}")
    st.divider()
    show_debug = st.toggle("显示检索与校验明细", value=False)
    st.caption(f"拒答阈值：重排分数 < {EVIDENCE_THRESHOLD}")
    if st.button("清空对话"):
        st.session_state.history = []

st.title("📘 资金出境与反洗钱合规问答助手")
st.caption("回答仅依据知识库中的法律法规和监管文件，并附出处；找不到依据时回答“知识库中没有相关规定”。"
           "典型案例仅作实务参考。本工具不构成法律意见。")


def render_refs(refs):
    if not refs:
        return
    st.markdown("**出处**")
    for r in refs:
        tag = "（参考案例）" if r["doc_type"] == "典型案例" else ""
        with st.expander(f"[{r['n']}] {r['citation']}{tag}"):
            for q in r["quotes"]:
                st.markdown(f"> {q}")
            st.caption("原文")
            st.text(r["text"])


def render_cases(cases):
    if not cases:
        return
    st.markdown("**⚖️ 相关案例**")
    for c in cases:
        hit = f"｜命中关键词：{'、'.join(c.matched)}" if c.matched else ""
        with st.expander(f"{c.title} —— {c.source}（相关度 {c.score:.2f}{hit}）"):
            st.caption(c.citation)
            shown = False
            for name in CASE_SECTIONS:
                if c.sections.get(name):
                    st.markdown(f"**{name}**")
                    st.write(c.sections[name])
                    shown = True
            if not shown:
                st.write(c.full_text)


def render(ans):
    if ans.mode == "scenario" and not ans.refused:
        icon = next((v for k, v in VERDICT_ICON.items() if ans.verdict.startswith(k)), "⚪")
        st.markdown(f"#### {icon} 评估结论：{ans.verdict}")
        if ans.analysis.get("summary"):
            st.caption(f"理解的情形：{ans.analysis['summary']}")
        st.markdown(ans.text)
        for s in ans.sections:
            st.markdown(f"**{s['title']}**")
            st.markdown(s["content"])
    else:
        st.markdown(ans.text)
    if ans.refused:
        st.caption(f"拒答原因：{REASONS.get(ans.refuse_reason, ans.refuse_reason)}")
    render_refs(ans.references)
    render_cases(ans.cases)

    if show_debug and ans.search:
        with st.expander("🔍 检索与校验明细"):
            st.write(f"模式={ans.mode}，has_evidence={ans.search.has_evidence}")
            if ans.search.expanded_query != ans.query:
                st.write(f"术语扩展：{ans.search.expanded_query}")
            if ans.analysis.get("issues"):
                st.write("拆解的合规问题：")
                for i, sub in zip(ans.analysis["issues"], ans.sub_searches):
                    best = f"{sub.hits[0].score:.2f} {sub.hits[0].citation}" if sub.hits else "-"
                    st.write(f"- {i.get('topic', '')}：{i['query']}（最佳 {best}）")
            st.dataframe(
                [{"重排分": round(h.score, 3), "出处": h.citation,
                  "向量排名": h.dense_rank, "关键词排名": h.sparse_rank} for h in ans.search.hits],
                hide_index=True, width="stretch",
            )
            if ans.search.definitions:
                st.write("补充的定义条款：" + "；".join(h.citation for h in ans.search.definitions))
            for c in ans.checks:
                st.write(("✅ " if c.valid else f"❌ {c.reason} ") + f"{c.source}：{c.quote}")


for ans in st.session_state.get("history", []):
    with st.chat_message("user"):
        st.markdown(ans.query)
    with st.chat_message("assistant"):
        render(ans)

placeholder = "提问或描述你想做的事，例如：个人每年能换多少美元？/ 我想带 2 万美元现金出境旅游"
if query := st.chat_input(placeholder):
    st.session_state.setdefault("history", [])
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("检索并分析中…（场景评估需调用两次大模型，约 20–40 秒）"):
            try:
                ans = get_qa().ask(query, mode=mode)
            except Exception as e:  # 网络/API 错误直接展示，便于演示时排查
                st.error(f"出错了：{e}")
                st.stop()
        render(ans)
    st.session_state.history.append(ans)
