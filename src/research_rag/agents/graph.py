"""Wires AgentNodes into a LangGraph StateGraph and exposes ResearchRAGAgent,
the one object the UI / CLI / tests talk to.

Memory: the graph is compiled with a checkpointer, and every call passes a
`thread_id`. LangGraph stores the persisted state keys (history / summary /
focus) per thread, so a follow-up on turn 5 sees turns 1-4 with no code here
managing it. Threads are isolated from each other.

  checkpoint_backend="memory"  -> lives as long as the process (default)
  checkpoint_backend="sqlite"  -> survives restarts (needs langgraph-checkpoint-sqlite)
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Iterator

from langgraph.graph import END, START, StateGraph

from research_rag.models import Chunk, SourceCitation
from .nodes import AgentNodes
from .result import AgentResult
from .routing import build_route_dot, route_path
from .state import AgentState, fresh_turn

logger = logging.getLogger(__name__)

# Human-readable labels for live progress in the UI
NODE_LABELS = {
    "guard_input": "Checking input (rate limit, PII, injection, safety)",
    "classify": "Classifying the message",
    "respond_chitchat": "Replying",
    "respond_scope": "Redirecting off-topic request",
    "respond_not_found": "Nothing relevant found",
    "rewrite_query": "Rewriting the query",
    "plan": "Planning: checking if the question has several parts",
    "retrieve_plan": "Searching for each part of the question",
    "hyde": "Drafting a hypothetical answer passage (HyDE)",
    "retrieve": "Searching, fusing and reranking",
    "grade": "Grading whether the evidence is enough",
    "generate": "Writing a cited answer",
    "output_guard": "Verifying citations & groundedness",
    "finalize": "Updating memory",
}


def build_checkpointer(backend: str = "memory", path: str = "data/state/checkpoints.sqlite"):
    if backend == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False)   # API workers serve requests on different threads
            logger.info(f"Conversation memory: SQLite checkpointer at {path}")
            return SqliteSaver(conn)
        except ImportError:
            logger.warning("CHECKPOINT_BACKEND=sqlite but langgraph-checkpoint-sqlite is not installed — "
                           "falling back to in-memory conversation state.")
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Conversation memory: in-process (lost on restart)")
    return MemorySaver()


def build_graph(nodes: AgentNodes, checkpointer):
    g = StateGraph(AgentState)

    g.add_node("guard_input", nodes.guard_input)
    g.add_node("classify", nodes.classify)
    g.add_node("respond_chitchat", nodes.respond_chitchat)
    g.add_node("respond_scope", nodes.respond_scope)
    g.add_node("respond_not_found", nodes.respond_not_found)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("plan", nodes.plan)
    g.add_node("hyde", nodes.hyde_node)
    g.add_node("retrieve_plan", nodes.retrieve_plan)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("generate", nodes.generate)
    g.add_node("output_guard", nodes.output_guard_node)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", nodes.route_after_guard, ["classify", "finalize"])
    g.add_conditional_edges("classify", nodes.route_after_classify,
                            ["respond_chitchat", "respond_scope", "rewrite_query", "finalize"])
    for terminal in ("respond_chitchat", "respond_scope", "respond_not_found"):
        g.add_edge(terminal, "finalize")

    g.add_edge("rewrite_query", "plan")
    g.add_conditional_edges("plan", nodes.route_after_plan, ["hyde", "retrieve_plan"])
    g.add_edge("hyde", "retrieve")
    for search_node in ("retrieve", "retrieve_plan"):
        g.add_conditional_edges(search_node, nodes.route_after_retrieve, ["grade", "respond_not_found", "finalize"])
    g.add_conditional_edges("grade", nodes.route_after_grade, ["rewrite_query", "generate"])
    g.add_edge("generate", "output_guard")
    g.add_edge("output_guard", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


class ResearchRAGAgent:
    def __init__(self, nodes: AgentNodes, checkpointer, defaults: dict):
        self.nodes = nodes
        self.defaults = defaults
        self.checkpointer = checkpointer
        self.graph = build_graph(nodes, checkpointer)

    # ---------------------------------------------------------------- public API
    @staticmethod
    def new_thread_id() -> str:
        return f"thread-{uuid.uuid4().hex[:12]}"

    def chat_stream(self, message: str, thread_id: str, options: dict | None = None,
                    client_id: str | None = None) -> Iterator[tuple[str, object]]:
        """Yields ("node", node_name) and ("step", {node, detail, seconds}) as each agent finishes,
        then ("result", AgentResult). The step events let the UI show which agent just ran."""
        config = {"configurable": {"thread_id": thread_id},
                  "recursion_limit": 40}      # hard backstop on top of MAX_RETRIEVAL_RETRIES
        # Rate limiting is keyed to the caller, not the conversation — otherwise a client
        # could reset its own quota just by starting a new conversation each request.
        turn_input = fresh_turn(message, client_id or thread_id, options or {})
        try:
            seen_steps = 0
            for update in self.graph.stream(turn_input, config, stream_mode="updates"):
                for node_name, delta in update.items():
                    yield "node", node_name
                    trace = (delta or {}).get("trace") or []
                    for step in trace[seen_steps:]:
                        yield "step", step
                    seen_steps = max(seen_steps, len(trace))
            values = self.graph.get_state(config).values
        except Exception as error:  # noqa: BLE001 — never surface a raw traceback to the chat window
            logger.exception("Agent run failed")
            yield "result", AgentResult(
                answer="Something went wrong while processing that message. Please try again.",
                answer_kind="error", error=str(error), thread_id=thread_id)
            return
        yield "result", self._to_result(values, thread_id)

    def chat(self, message: str, thread_id: str, options: dict | None = None,
             client_id: str | None = None) -> AgentResult:
        result = None
        for kind, payload in self.chat_stream(message, thread_id, options, client_id):
            if kind == "result":
                result = payload
        return result   # type: ignore[return-value]

    def route_dot(self, result: AgentResult) -> str:
        """Graphviz DOT of the real compiled graph with this turn's path highlighted."""
        return build_route_dot(self.graph.get_graph(), route_path(result))

    def memory_snapshot(self, thread_id: str) -> dict:
        """What the agent currently remembers for this thread (for the sidebar)."""
        try:
            values = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values or {}
        except Exception:  # noqa: BLE001
            values = {}
        history = values.get("history", [])
        return {"turns": len(history) // 2, "summary": values.get("summary", ""), "focus": values.get("focus", {}),
                "history": history}

    def reset_thread(self, thread_id: str) -> None:
        delete = getattr(self.checkpointer, "delete_thread", None)
        if callable(delete):
            delete(thread_id)

    # ---------------------------------------------------------------- internals
    def _to_result(self, v: dict, thread_id: str) -> AgentResult:
        return AgentResult(
            answer=v.get("answer", ""), answer_kind=v.get("answer_kind", ""),
            intent=v.get("intent", ""), intent_confidence=v.get("intent_confidence", 0.0),
            intent_reason=v.get("intent_reason", ""), intent_source=v.get("intent_source", ""),
            sources=[SourceCitation.from_dict(s) for s in v.get("sources", [])],
            chunks=[Chunk.from_dict(c) for c in v.get("top_chunks", [])],
            standalone_query=v.get("standalone_query", ""), alternatives=v.get("alternatives", []),
            sub_questions=v.get("sub_questions", []), plan_reason=v.get("plan_reason", ""),
            plan_coverage=v.get("plan_coverage", []),
            hyde_passage=v.get("hyde_passage", ""), variants_used=v.get("variants_used", []),
            n_candidates=v.get("n_candidates", 0), retry_count=v.get("retry_count", 0),
            context_ok=v.get("context_ok", True), context_reason=v.get("context_reason", ""),
            citations_ok=v.get("citations_ok", True), grounded=v.get("grounded"),
            unsupported_claims=v.get("unsupported_claims", []),
            events=v.get("events", []), trace=v.get("trace", []),
            blocked=v.get("blocked", False), block_kind=v.get("block_kind", ""),
            clean_message=v.get("clean_message", ""), thread_id=thread_id, error=v.get("error", ""),
        )
