"""命令行问答演示。

用法：
  python scripts/ask.py "个人每年购汇额度是多少"            # 自动识别模式
  python scripts/ask.py "我想带3万美元现金出境" --mode scenario -v
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generator import QA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="auto", choices=["auto", "qa", "scenario"])
    ap.add_argument("-v", "--verbose", action="store_true", help="显示检索与引用校验明细")
    args = ap.parse_args()

    ans = QA().ask(args.query, mode=args.mode)
    print(f"【{ans.mode}】{ans.query}\n")
    if ans.verdict:
        print(f"评估结论：{ans.verdict}\n")
    print(ans.text + "\n")
    for s in ans.sections:
        print(f"■ {s['title']}\n{s['content']}\n")
    for r in ans.references:
        print(f"[{r['n']}] {r['citation']}")
        for q in r["quotes"]:
            print(f"    “{q}”")
    if ans.cases:
        print("\n相关案例：")
        for c in ans.cases:
            print(f"  - {c.title}（{c.source}，相关度 {c.score:.2f}）{' 命中关键词:' + '、'.join(c.matched) if c.matched else ''}")
    if args.verbose:
        print(f"\n拒答原因: {ans.refuse_reason or '-'}")
        if ans.analysis:
            print("拆解：", ans.analysis)
        for h in ans.search.hits:
            print(f"  检索 {h.score:.3f} {h.citation}")
        for c in ans.checks:
            print(f"  校验 {c.source} {'通过' if c.valid else '失败:' + c.reason} “{c.quote[:40]}”")


if __name__ == "__main__":
    main()
