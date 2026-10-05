"""把评测明细 CSV 生成可视化网页报告（单文件 HTML，可直接用浏览器打开或转发）。

用法：
  python scripts/report_html.py                       # 取 eval/results 下最新的 detail_full_*.csv
  python scripts/report_html.py eval/results/detail_full_20260930_1237.csv
  python scripts/report_html.py --fragment ...        # 输出不带 <!doctype> 外壳的片段（发布为 Artifact 时用）
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"

TYPE_NAME = {"normal": "法规问答", "scenario": "场景评估", "out_of_kb": "知识库外"}
VERDICT_TONE = {"合规可行": "ok", "有条件可行": "warn", "存在较高合规风险": "risk", "违法或被禁止": "bad"}


def _b(v: str) -> bool | None:
    return {"True": True, "False": False}.get(v)


def _list(v: str) -> list:
    try:
        return json.loads(v) if v else []
    except json.JSONDecodeError:
        return []


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for k in ("refused", "hit@1", "hit@k", "cited_gold", "points_ok", "case_ok", "expect_case"):
                r[k] = _b(r.get(k, ""))
            for k in ("cited", "missing_points", "cases_shown", "retrieved", "gold"):
                r[k] = _list(r.get(k, ""))
            # 旧版明细没有 has_gold 列：知识库外题没有标准条款，其余都有
            r["has_gold"] = _b(r["has_gold"]) if "has_gold" in r else r["type"] != "out_of_kb"
            r["seconds"] = float(r.get("seconds") or 0)
            rows.append(r)
    return rows


def metrics(rows: list[dict]) -> list[tuple[str, int, int, str]]:
    ans = [r for r in rows if r["expect"] == "answer"]
    gold = [r for r in ans if r["has_gold"]]
    ook = [r for r in rows if r["type"] == "out_of_kb"]
    case = [r for r in rows if r["case_ok"] is not None]
    return [
        ("检索命中率", sum(r["hit@k"] for r in gold), len(gold), "标准条款出现在检索结果中"),
        ("引用准确率", sum((not r["refused"]) and r["cited_gold"] for r in gold), len(gold), "作答且引用了标准条款"),
        ("要点覆盖率", sum(bool(r["points_ok"]) for r in ans), len(ans), "回答包含全部必备要点"),
        ("拒答正确率", sum(bool(r["refused"]) for r in ook), len(ook), "知识库外问题正确拒答"),
        ("误拒率", sum(bool(r["refused"]) for r in ans), len(ans), "应回答却拒答，越低越好"),
        ("案例触发准确率", sum(bool(r["case_ok"]) for r in case), len(case), "该展示案例时展示、不该时不展示"),
    ]


def issues(r: dict) -> list[str]:
    out = []
    if r["expect"] == "answer":
        if r["has_gold"] and not r["hit@k"]:
            out.append("标准条款未检索到")
        if r["refused"]:
            out.append("被拒答")
        else:
            if r["has_gold"] and not r["cited_gold"]:
                out.append("引用未含标准条款")
            if r["missing_points"]:
                out.append("缺少要点：" + "、".join(r["missing_points"]))
    elif not r["refused"]:
        out.append("应拒答却作答")
    if r["case_ok"] is False:
        out.append("案例展示不符合预期")
    return out


def esc(s) -> str:
    return html.escape(str(s))


def fmt_answer(text: str) -> str:
    t = esc(text)
    return re.sub(r"\[(\d+)\]", r'<sup class="ref">\1</sup>', t)


CSS = """
/* 布局：顶部摘要 → 指标条 → 可筛选的逐题清单（每题可展开看回答、引用、案例） */
:root{
  --bg:#f5f6f8; --surface:#ffffff; --ink:#1b2330; --muted:#5c6776; --line:#dde1e7;
  --accent:#b3261e; --accent-soft:#f8e6e4;
  --ok:#1f7a4d; --ok-soft:#e3f2ea; --warn:#9a6700; --warn-soft:#fbf1dc; --bad:#b3261e; --bad-soft:#f8e6e4;
  --risk:#b4541a; --risk-soft:#fbebe0;
  --display:"Noto Serif SC","Songti SC","STSong",serif;
  --body:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  --mono:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#12161c; --surface:#1a2029; --ink:#e6e9ee; --muted:#9aa4b2; --line:#2c3440;
  --accent:#ef7b6f; --accent-soft:#3a2220;
  --ok:#5fc592; --ok-soft:#173226; --warn:#e3b04b; --warn-soft:#3a2f15; --bad:#ef7b6f; --bad-soft:#3a2220;
  --risk:#f09a5e; --risk-soft:#3a2818; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#12161c; --surface:#1a2029; --ink:#e6e9ee; --muted:#9aa4b2; --line:#2c3440;
  --accent:#ef7b6f; --accent-soft:#3a2220;
  --ok:#5fc592; --ok-soft:#173226; --warn:#e3b04b; --warn-soft:#3a2f15; --bad:#ef7b6f; --bad-soft:#3a2220;
  --risk:#f09a5e; --risk-soft:#3a2818; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.65 var(--body);margin:0}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px;display:grid;gap:28px}
header{display:grid;gap:6px;border-bottom:2px solid var(--ink);padding-bottom:16px}
.eyebrow{font:600 12px/1 var(--body);letter-spacing:.12em;color:var(--accent)}
h1{font:700 30px/1.25 var(--display);margin:0;text-wrap:balance}
.meta{color:var(--muted);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}
.kpi{background:var(--surface);padding:14px 16px;display:grid;gap:4px;min-width:0}
.kpi .name{font-size:13px;color:var(--muted)}
.kpi .val{font:700 28px/1.1 var(--display);font-variant-numeric:tabular-nums}
.kpi .frac{font:12px var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
.kpi .bar{height:4px;background:var(--line)}
.kpi .bar i{display:block;height:100%;background:var(--ok)}
.kpi.inverse .bar i{background:var(--accent)}
.kpi .hint{font-size:12px;color:var(--muted)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.toolbar button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--surface);color:var(--ink);
  padding:5px 12px;border-radius:999px;cursor:pointer}
.toolbar button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.toolbar button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.toolbar .count{margin-left:auto;color:var(--muted);font-size:13px}
.list{display:grid;gap:8px}
details.q{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--ok)}
details.q.flag{border-left-color:var(--accent)}
summary{list-style:none;cursor:pointer;padding:12px 16px;display:grid;grid-template-columns:52px 1fr auto;gap:12px;align-items:start}
summary::-webkit-details-marker{display:none}
.qid{font:600 13px var(--mono);color:var(--muted);padding-top:2px}
.qtext{min-width:0;display:grid;gap:6px}
.qtext .text{font-weight:600}
.tags{display:flex;flex-wrap:wrap;gap:6px}
.pill{font-size:12px;line-height:1;padding:4px 8px;border-radius:4px;background:var(--bg);color:var(--muted);border:1px solid var(--line)}
.pill.ok{background:var(--ok-soft);color:var(--ok);border-color:transparent}
.pill.warn{background:var(--warn-soft);color:var(--warn);border-color:transparent}
.pill.risk{background:var(--risk-soft);color:var(--risk);border-color:transparent}
.pill.bad{background:var(--bad-soft);color:var(--bad);border-color:transparent}
.side{text-align:right;font:12px var(--mono);color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.body{padding:4px 16px 18px 80px;display:grid;gap:14px;min-width:0}
.body h3{font:600 12px/1 var(--body);letter-spacing:.1em;color:var(--muted);margin:0 0 6px}
.answer{white-space:pre-wrap;max-width:72ch;overflow-wrap:anywhere}
.ref{color:var(--accent);font-weight:700;font-size:11px;padding:0 1px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:12px var(--mono);padding:3px 7px;border:1px solid var(--line);border-radius:3px;overflow-wrap:anywhere}
.chip.gold{border-color:var(--ok);color:var(--ok)}
.issues{color:var(--accent);font-weight:600}
.key{color:var(--muted);font-size:14px}
.empty{color:var(--muted)}
@media (max-width:640px){
  summary{grid-template-columns:1fr;gap:6px}
  .side{text-align:left}
  .body{padding:4px 16px 18px}
  h1{font-size:24px}
}
@media (prefers-reduced-motion:no-preference){details.q{transition:border-color .2s}}
"""

JS = """
const btns=[...document.querySelectorAll('.toolbar button')];
const items=[...document.querySelectorAll('details.q')];
const count=document.querySelector('.toolbar .count');
function apply(f){
  let n=0;
  items.forEach(el=>{const show=f==='all'||(f==='flag'?el.classList.contains('flag'):el.dataset.type===f);
    el.hidden=!show; if(show)n++;});
  btns.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.f===f)));
  count.textContent=n+' 题';
}
btns.forEach(b=>b.addEventListener('click',()=>apply(b.dataset.f)));
apply(document.querySelector('details.q.flag')?'flag':'all');
"""


def render(rows: list[dict], src: Path, fragment: bool) -> str:
    stamp = re.search(r"(\d{8})_(\d{4})", src.name)
    when = f"{stamp.group(1)[:4]}-{stamp.group(1)[4:6]}-{stamp.group(1)[6:]} {stamp.group(2)[:2]}:{stamp.group(2)[2:]}" if stamp else ""
    counts = {t: sum(r["type"] == t for r in rows) for t in TYPE_NAME}
    flagged = sum(bool(issues(r)) for r in rows)

    kpis = []
    for name, n, d, hint in metrics(rows):
        ratio = n / d if d else 0
        val = f"{ratio:.0%}" if d else "—"
        cls = "kpi inverse" if name == "误拒率" else "kpi"
        kpis.append(f'<div class="{cls}"><span class="name">{name}</span><span class="val">{val}</span>'
                    f'<span class="frac">{n}/{d}</span><span class="bar"><i style="width:{ratio*100:.0f}%"></i></span>'
                    f'<span class="hint">{hint}</span></div>')

    items = []
    for r in rows:
        prob = issues(r)
        tags = [f'<span class="pill">{TYPE_NAME.get(r["type"], r["type"])}</span>']
        if r["refused"]:
            tags.append(f'<span class="pill {"ok" if r["expect"] == "refuse" else "bad"}">已拒答</span>')
        if r.get("verdict"):
            tone = next((v for k, v in VERDICT_TONE.items() if r["verdict"].startswith(k)), "")
            tags.append(f'<span class="pill {tone}">{esc(r["verdict"])}</span>')
        tags.append(f'<span class="pill {"bad" if prob else "ok"}">{"需关注" if prob else "通过"}</span>')
        gold = set(r.get("gold") or [])          # 旧版明细没有 gold 列，则不做高亮
        cited = "".join(f'<span class="chip{" gold" if c in gold else ""}">{esc(c)}</span>' for c in r["cited"]) \
            or '<span class="empty">无</span>'
        cases = "".join(f'<span class="chip">{esc(c)}</span>' for c in r["cases_shown"]) or '<span class="empty">未展示</span>'
        body = [f'<div><h3>回答</h3><div class="answer">{fmt_answer(r["answer"]) or "（无）"}</div></div>',
                f'<div><h3>引用的条款</h3><div class="chips">{cited}</div></div>',
                f'<div><h3>展示的案例</h3><div class="chips">{cases}</div></div>']
        if prob:
            body.insert(0, f'<div class="issues">{"；".join(esc(p) for p in prob)}</div>')
        if r.get("key"):
            body.append(f'<div><h3>参考答案</h3><div class="key">{esc(r["key"])}</div></div>')
        items.append(
            f'<details class="q{" flag" if prob else ""}" data-type="{r["type"]}">'
            f'<summary><span class="qid">{esc(r["id"])}</span>'
            f'<span class="qtext"><span class="text">{esc(r["question"])}</span><span class="tags">{"".join(tags)}</span></span>'
            f'<span class="side">top {float(r["top_score"] or 0):.3f}<br>{r["seconds"]:.1f}s</span></summary>'
            f'<div class="body">{"".join(body)}</div></details>')

    avg = [r["seconds"] for r in rows]
    page = f"""<title>RAG 评测报告</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=JetBrains+Mono&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <span class="eyebrow">资金出境与反洗钱合规问答 · 完整问答评测</span>
  <h1>评测报告 {when}</h1>
  <span class="meta">共 {len(rows)} 题：法规问答 {counts['normal']}、场景评估 {counts['scenario']}、知识库外 {counts['out_of_kb']}；
  需关注 {flagged} 题；平均每题 {sum(avg)/max(len(avg),1):.1f} 秒。数据来源：{esc(src.name)}</span>
</header>
<section class="kpis">{"".join(kpis)}</section>
<section class="toolbar" aria-label="筛选">
  <button type="button" data-f="all">全部</button>
  <button type="button" data-f="flag">需关注</button>
  <button type="button" data-f="normal">法规问答</button>
  <button type="button" data-f="scenario">场景评估</button>
  <button type="button" data-f="out_of_kb">知识库外</button>
  <span class="count"></span>
</section>
<section class="list">{"".join(items)}</section>
</div>
<script>{JS}</script>"""
    return page if fragment else f'<!doctype html>\n<html lang="zh-CN"><meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n{page}\n</html>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--fragment", action="store_true")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    src = Path(args.csv) if args.csv else max(RESULTS.glob("detail_full_*.csv"), key=lambda p: p.stat().st_mtime)
    out = Path(args.out) if args.out else src.with_name(src.stem.replace("detail_", "report_") + ".html")
    out.write_text(render(load(src), src, args.fragment), encoding="utf-8")
    print(f"已生成：{out}")


if __name__ == "__main__":
    main()
