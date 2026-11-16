#!/usr/bin/env python
"""Ask one question against the indexed corpus (no server, no conversation).

Usage:
    python scripts/run_query.py "What problem does the VDF-RLS algorithm solve?"
    python scripts/run_query.py --json "Which baseline did it outperform?"
"""
import argparse
import json
import logging

from _bootstrap import build_rag_system

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args()

    question = " ".join(args.question)
    result = build_rag_system().ask_full(question)

    if args.json:
        print(json.dumps({
            "answer": result.answer, "answer_kind": result.answer_kind, "intent": result.intent,
            "standalone_query": result.standalone_query, "grounded": result.grounded,
            "seconds": round(result.total_seconds, 2),
            "sources": [s.to_dict() for s in result.sources],
            "events": result.events,
        }, indent=2, ensure_ascii=False))
        return

    print(f"\n[{result.answer_kind} | {result.total_seconds:.1f}s]")
    if result.standalone_query and result.standalone_query != question:
        print(f"(searched as: {result.standalone_query})")
    print("\n" + result.answer)
    if result.cited_sources:
        print("\nSources:")
        for source in result.cited_sources:
            print(f"  [{source.label}] {source.format()}")
            if source.paper_title:
                print(f"        {source.paper_title}")
    for e in result.events:
        print(f"  · guardrail {e['guardrail']}: {e['action']} — {e['detail']}")


if __name__ == "__main__":
    main()
