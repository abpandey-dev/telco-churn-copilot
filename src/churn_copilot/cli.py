"""`churn` command line.

  churn train                       train both models, print metrics and the true vs learned drivers
  churn customer C-04124            churn risk and explanation for one customer (no API call)
  churn search "discount approval"  search the policy knowledge base (no API call)
  churn eval-rag                    retrieval hit@1 / hit@3 on eval/rag_questions.json (no API call)
  churn ask "Customer C-04124 is about to leave. What can we offer?"   LLM answer with citations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from churn_copilot import model as churn_model
from churn_copilot.data import TRUE_EFFECTS


def _ws():
    from churn_copilot.agent import Workspace

    return Workspace()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="churn", description="Telco Churn Copilot: ML + RAG + LLM")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="Train models and show metrics")
    c = sub.add_parser("customer", help="Explain one customer's churn risk")
    c.add_argument("customer_id")
    s = sub.add_parser("search", help="Search the policy knowledge base")
    s.add_argument("query")
    e = sub.add_parser("eval-rag", help="Evaluate retrieval")
    e.add_argument("--file", default="eval/rag_questions.json")
    a = sub.add_parser("ask", help="Ask the LLM copilot (uses the Claude API)")
    a.add_argument("question")
    a.add_argument("--model", default="claude-opus-5-5")
    a.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    a.add_argument("--trace", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "train":
        ws = _ws()
        print(json.dumps(ws.models.metrics, indent=2))
        print("\nLearned importance (drop in ROC-AUC when shuffled):")
        for f, v in churn_model.global_importance(ws.models).items():
            print(f"  {f:<24} {v:+.4f}")
        print("\nTrue effects used to generate the data (log-odds):")
        for k, v in TRUE_EFFECTS.items():
            print(f"  {k:<44} {v:+.3f}")
        return 0

    if args.cmd == "customer":
        from churn_copilot.agent import ToolError, Tools

        try:
            print(Tools(_ws()).call("get_customer_risk", {"customer_id": args.customer_id}))
        except ToolError as err:
            print(err)
            return 1
        return 0

    if args.cmd == "search":
        from churn_copilot.rag import Retriever

        for h in Retriever().search(args.query):
            print(f"[{h['id']}] (score {h['score']})\n  {h['text'][:200]}...\n")
        return 0

    if args.cmd == "eval-rag":
        from churn_copilot.rag import Retriever

        r, qs = Retriever(), json.loads(Path(args.file).read_text(encoding="utf-8"))
        h1 = h3 = 0
        for q in qs:
            ids = [x["id"] for x in r.search(q["question"], 3)]
            h1 += ids[:1] == [q["expected"]]
            h3 += q["expected"] in ids
            mark = "ok " if ids[:1] == [q["expected"]] else ("top3" if q["expected"] in ids else "MISS")
            print(f"  [{mark}] {q['question']}")
        print(f"\nhit@1 {h1}/{len(qs)} ({h1 / len(qs):.0%})   hit@3 {h3}/{len(qs)} ({h3 / len(qs):.0%})")
        return 0

    from churn_copilot.agent import ChurnCopilot

    r = ChurnCopilot(model=args.model, effort=args.effort).ask(args.question)
    if args.trace:
        for t in r.trace:
            print(f"  turn {t['turn']}: {t['tool']}({json.dumps(t['input'])}){'  [error]' if t['is_error'] else ''}")
    cost = f", ${r.cost_usd:.4f}" if r.cost_usd is not None else ""
    print(f"\n{r.input_tokens}+{r.output_tokens} tokens, {r.latency_s}s{cost}")
    if r.error:
        print(f"Error: {r.error}")
    if r.answer:
        print(f"\n{r.answer.answer}\n")
        for act in r.answer.recommended_actions:
            print(f"  • {act.action}   [{act.policy_chunk_id}]")
        print(f"\nCitations: {', '.join(r.answer.citations)}")
        if r.unsupported_citations:
            print(f"WARNING: cited but never retrieved (possible hallucination): {r.unsupported_citations}")
        else:
            print("Citation check: every citation was actually retrieved ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
