"""评测：检索命中率、引用准确率、要点覆盖率、拒答正确率、案例触发准确率。

两种模式
  python scripts/eval.py --retrieval-only   只跑检索（不调用 DeepSeek），附拒答阈值扫描
  python scripts/eval.py                    完整问答（问答题每题 1 次 DeepSeek，场景题每题 2 次）

指标定义
  检索命中率    法规问答题 + 场景题中，标准条款出现在检索结果里的比例（另报 Hit@1）
                场景题在完整模式下按"原始描述 + 拆解后各子问题"的合并结果计；检索模式下只用原始描述
  引用准确率    法规问答题 + 场景题中，系统作答且引用里包含标准条款的比例
  要点覆盖率    同上范围内，回答包含 must 所列全部要点的比例（衡量"答得对不对"）
  拒答正确率    知识库外题中，系统拒答的比例；另报误拒率 = 应回答的题被拒答的比例
  案例触发准确率  标注了 expect_case 的题中，"是否展示案例"与预期一致的比例
  检索模式下"拒答"按检索层判断（分数低于阈值），不含大模型自身的拒答。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from rag.config import EVIDENCE_THRESHOLD  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

EVAL_DIR = ROOT / "eval"


def load_questions(path: Path) -> list[dict]:
    qs = yaml.safe_load(path.read_text(encoding="utf-8"))
    for q in qs:
        q.setdefault("gold", [])
        q.setdefault("must", [])
        q.setdefault("expect_case", None)
    return qs


def missing_points(answer: str, must: list[str]) -> list[str]:
    """返回回答中缺失的要点；每项可用 | 分隔同义写法。"""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", re.sub(r"\s+", "", answer))
    return [m for m in must if not any(re.sub(r"\s+", "", alt) in text for alt in m.split("|"))]


def avg_len(rows, qtype: str) -> str:
    xs = [len(r["answer"]) for r in rows if r["type"] == qtype and not r["refused"] and r["answer"]]
    return f"{sum(xs) // len(xs)}" if xs else "-"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.0%}" if d else "-"


def run(questions, retrieval_only: bool):
    retriever = Retriever()
    qa = None
    if not retrieval_only:
        from rag.generator import QA
        qa = QA(retriever)

    rows = []
    for i, q in enumerate(questions, 1):
        t = time.time()
        # mode 可在题目里指定（auto = 与界面一致的自动识别）；默认按 type
        mode = q.get("mode") or ("scenario" if q["type"] == "scenario" else "qa")
        if qa:
            ans = qa.ask(q["question"], mode=mode)
            searches = [ans.search, *ans.sub_searches]
            refused, reason = ans.refused, ans.refuse_reason
            text = "\n".join([ans.text] + [s["content"] for s in ans.sections])
            cited = [r["chunk_id"] for r in ans.references]
            dropped = sum(not c.valid for c in ans.checks)
            dropped_detail = [f"{c.source}({c.chunk_id}) {c.reason}：{c.quote}" for c in ans.checks if not c.valid]
            cases, verdict = ans.cases, ans.verdict
        else:
            res = retriever.search(q["question"])
            searches = [res]
            refused = not res.has_evidence
            reason = "no_evidence" if refused else ""
            text, cited, dropped, cases, verdict = "", [], 0, res.cases, ""
            dropped_detail = []

        main = searches[0]
        hit_ids = [h.chunk_id for h in main.hits]
        all_ids = list(dict.fromkeys(h.chunk_id for s in searches if s for h in s.hits))
        gold = set(q["gold"])
        missing = missing_points(text, q["must"]) if (qa and not refused) else q["must"]
        shown = bool(cases)
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "expect": q["expect"],
            "refused": refused, "reason": reason, "verdict": verdict,
            "top_score": round(main.hits[0].score, 3) if main.hits else 0.0,
            "hit@1": bool(hit_ids) and hit_ids[0] in gold,
            "hit@k": bool(gold & set(all_ids)),
            "gold_rank": next((r for r, cid in enumerate(hit_ids, 1) if cid in gold), None),
            "has_gold": bool(gold), "gold": sorted(gold),
            "cited": cited, "cited_gold": bool(gold & set(cited)),
            "dropped_citations": dropped, "dropped_detail": dropped_detail,
            "missing_points": missing, "points_ok": not missing,
            "expect_case": q["expect_case"], "cases_shown": [f"{c.title}({c.score:.2f})" for c in cases],
            "case_ok": None if q["expect_case"] is None else (shown == q["expect_case"]),
            "retrieved": all_ids, "answer": text, "key": q.get("key", ""),
            "seconds": round(time.time() - t, 1),
        })
        r = rows[-1]
        print(f"[{i:02d}/{len(questions)}] {q['id']} {'拒答' if refused else '作答'} top={r['top_score']:.3f} "
              f"gold_rank={r['gold_rank']} 案例={len(cases)} {r['seconds']}s", flush=True)
    return rows


def summarize(rows, retrieval_only: bool) -> list[str]:
    answerable = [r for r in rows if r["expect"] == "answer"]
    with_gold = [r for r in answerable if r["has_gold"]]      # 自拟检验题可以不标 gold，不计入命中/引用指标
    ook = [r for r in rows if r["type"] == "out_of_kb"]
    case_rows = [r for r in rows if r["case_ok"] is not None]
    lines = [
        f"# 评测结果（{'仅检索' if retrieval_only else '完整问答'}，{datetime.now():%Y-%m-%d %H:%M}）",
        "",
        f"题目：法规问答 {sum(r['type'] == 'normal' for r in rows)}，场景 {sum(r['type'] == 'scenario' for r in rows)}，"
        f"知识库外 {len(ook)}；拒答阈值 {EVIDENCE_THRESHOLD}",
        "",
        "| 指标 | 结果 |",
        "|---|---|",
        f"| **检索命中率** | {pct(sum(r['hit@k'] for r in with_gold), len(with_gold))} |",
        f"| 检索命中率（Top-1） | {pct(sum(r['hit@1'] for r in with_gold), len(with_gold))} |",
    ]
    if not retrieval_only:
        lines += [
            f"| **引用准确率** | {pct(sum((not r['refused']) and r['cited_gold'] for r in with_gold), len(with_gold))} |",
            f"| **要点覆盖率** | {pct(sum(r['points_ok'] for r in answerable), len(answerable))} |",
            f"| 被校验剔除的引用数 | {sum(r['dropped_citations'] for r in rows)} |",
            f"| 平均回答长度（问答 / 场景，字） | {avg_len(rows, 'normal')} / {avg_len(rows, 'scenario')} |",
        ]
    lines += [
        f"| **拒答正确率** | {pct(sum(r['refused'] for r in ook), len(ook))} |",
        f"| 误拒率 | {pct(sum(r['refused'] for r in answerable), len(answerable))} |",
        f"| **案例触发准确率** | {pct(sum(r['case_ok'] for r in case_rows), len(case_rows))} |",
        "",
    ]
    bad = []
    for r in rows:
        why = []
        if r["expect"] == "answer":
            if r["has_gold"] and not r["hit@k"]:
                why.append("标准条款未检索到")
            if r["refused"]:
                why.append(f"被拒答({r['reason']}, top={r['top_score']})")
            elif not retrieval_only:
                if r["has_gold"] and not r["cited_gold"]:
                    why.append(f"引用未含标准条款：{','.join(r['cited'])}")
                if r["dropped_detail"]:
                    why.append(f"引用被剔除：{'；'.join(r['dropped_detail'])}")
                if r["missing_points"]:
                    why.append(f"回答缺少要点：{'、'.join(r['missing_points'])}")
        elif not r["refused"]:
            why.append(f"应拒答却作答(top={r['top_score']})")
        if r["case_ok"] is False:
            why.append("应展示案例却没有" if r["expect_case"] else f"不应展示案例却展示了：{'、'.join(r['cases_shown'])}")
        if why:
            bad.append(f"| {r['id']} | {r['question']} | {'；'.join(why)} |")
    if bad:
        lines += ["## 需要关注的题", "", "| 题号 | 问题 | 情况 |", "|---|---|---|", *bad, ""]
    return lines


def threshold_sweep(rows) -> list[str]:
    """用检索分数模拟不同阈值下的表现。"""
    ans = [r for r in rows if r["expect"] == "answer"]
    ook = [r for r in rows if r["type"] == "out_of_kb"]
    lines = ["## 拒答阈值扫描", "", "| 阈值 | 拒答正确率 | 误拒率 |", "|---|---|---|"]
    for th in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        lines.append(f"| {th} | {pct(sum(r['top_score'] < th for r in ook), len(ook))} | "
                     f"{pct(sum(r['top_score'] < th for r in ans), len(ans))} |")
    lines += ["", f"分数分布：应回答题最低 top 分 {min(r['top_score'] for r in ans):.3f}；"
              f"知识库外题最高 top 分 {max((r['top_score'] for r in ook), default=0):.3f}", ""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="只评检索，不调用 DeepSeek")
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions.yaml"))
    ap.add_argument("--ids", help="只跑指定题号，逗号分隔，如 N01,S03")
    args = ap.parse_args()

    qs = load_questions(Path(args.questions))
    if args.ids:
        keep = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in keep]
    rows = run(qs, args.retrieval_only)

    report = summarize(rows, args.retrieval_only)
    if args.retrieval_only:
        report += threshold_sweep(rows)
    print("\n" + "\n".join(report))

    tag = "retrieval" if args.retrieval_only else "full"
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"report_{tag}_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    with (out_dir / f"detail_{tag}_{stamp}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in r.items()})
    print(f"已保存：eval/results/report_{tag}_{stamp}.md 与 detail_{tag}_{stamp}.csv")


if __name__ == "__main__":
    main()
