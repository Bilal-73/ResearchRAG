#!/usr/bin/env python
"""Multi-turn chat in the terminal — exercises the full LangGraph agent
(intent routing, follow-up handling, memory, guardrails) without the HTTP API.

Usage:
    python scripts/run_chat.py

Type /new for a fresh conversation, /memory to see what the agent remembers,
/quit to exit.
"""
import logging

from _bootstrap import build_agent  # also initializes logging as a side effect

logger = logging.getLogger(__name__)


def main() -> None:
    agent, _registry, _limiter, _llm, corpus, _cache = build_agent()
    thread_id = agent.new_thread_id()
    print(f"\nCorpus:\n{corpus.describe()}\n")
    print("Ask about the papers. /new = new conversation, /memory = show memory, /quit = exit.\n")

    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/new":
            thread_id = agent.new_thread_id()
            print("(new conversation)\n")
            continue
        if message == "/memory":
            snap = agent.memory_snapshot(thread_id)
            print(f"turns={snap['turns']} focus={snap['focus']}\nsummary={snap['summary'] or '(none yet)'}\n")
            continue

        result = agent.chat(message, thread_id)
        print(f"\n[{result.intent or result.answer_kind} | {result.total_seconds:.1f}s]")
        if result.standalone_query and result.standalone_query != result.clean_message:
            print(f"(searched as: {result.standalone_query})")
        print(result.answer)
        cited = result.cited_sources
        if cited:
            print("\nSources:")
            for s in cited:
                print(f"  [{s.label}] {s.format()}")
        for e in result.events:
            print(f"  · guardrail {e['guardrail']}: {e['action']} — {e['detail']}")
        print()


if __name__ == "__main__":
    main()
