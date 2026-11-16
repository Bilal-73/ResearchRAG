"""Explains a finished turn: WHICH agents were called, in what order, and WHY it was routed that way.

Everything here is derived from data the graph already produced (the trace and the decision fields),
and the diagram is drawn from the compiled LangGraph itself, so it cannot drift from the real wiring.
"""
from __future__ import annotations

import re

from .result import AgentResult

# kind: llm = calls the language model · rule = deterministic code · search = vector search / reranker
#       static = fixed pre-written reply · mixed = deterministic checks plus an optional LLM pass
NODE_INFO: dict[str, dict] = {
    "guard_input":       {"role": "Input Guard",                    "kind": "rule",   "does": "rate limit, length, PII masking, crisis, dual-use & injection checks"},
    "classify":          {"role": "Router (intent classifier)",     "kind": "llm",    "does": "labels the message and chooses the route"},
    "respond_chitchat":  {"role": "Chit-chat responder",            "kind": "llm",    "does": "friendly reply, no retrieval"},
    "respond_scope":     {"role": "Out-of-scope redirect",          "kind": "static", "does": "fixed polite decline"},
    "respond_not_found": {"role": "Not-found reply",                "kind": "static", "does": "nothing usable retrieved"},
    "rewrite_query":     {"role": "Query Rewriter",                 "kind": "llm",    "does": "standalone question / follow-up resolution"},
    "plan":              {"role": "Planner",                        "kind": "llm",    "does": "splits multi-part questions (rule check first)"},
    "hyde":              {"role": "HyDE generator",                 "kind": "llm",    "does": "hypothetical answer passage, used only for search"},
    "retrieve":          {"role": "Retriever + Reranker",           "kind": "search", "does": "multi-query hybrid search, RRF fusion, rerank"},
    "retrieve_plan":     {"role": "Per-part Retriever",             "kind": "search", "does": "retrieve + rerank for each sub-question"},
    "grade":             {"role": "Context Grader",                 "kind": "llm",    "does": "is the evidence enough? triggers one retry"},
    "generate":          {"role": "Answer Generator",               "kind": "llm",    "does": "cited answer from the excerpts"},
    "output_guard":      {"role": "Output Guard + Groundedness",    "kind": "mixed",  "does": "citations, leak canary, PII, groundedness check"},
    "finalize":          {"role": "Memory manager",                 "kind": "rule",   "does": "updates conversation memory"},
}
KIND_ICON = {"llm": "🤖", "rule": "🛡️", "search": "🔎", "static": "📄", "mixed": "🛡️🤖"}
KIND_COLOR = {"llm": "#2563eb", "rule": "#059669", "search": "#7c3aed", "static": "#d97706", "mixed": "#0891b2"}


def normalize_node(name: str) -> str:
    """Trace names -> graph node names ('retrieve_plan[2]' -> 'retrieve_plan'; the 'rerank' step lives inside 'retrieve')."""
    name = re.sub(r"\[\d+\]$", "", name)
    return "retrieve" if name == "rerank" else name


def route_path(result: AgentResult) -> list[str]:
    path: list[str] = []
    for step in result.trace:
        node = normalize_node(step["node"])
        if not path or path[-1] != node:
            path.append(node)
    return path


def route_decisions(result: AgentResult) -> list[str]:
    """Plain-language 'why it went there' lines."""
    lines: list[str] = []
    if result.blocked:
        lines.append(f"🛑 **Input Guard** stopped the message ({result.block_kind}); no other agent ran.")
        if result.answer_kind == "safety":
            lines[-1] = ("🛟 **Input Guard** detected crisis language and returned a fixed support message; "
                         "no other agent ran.")
        elif result.block_kind == "dual_use":
            lines[-1] = "🚫 **Input Guard** declined a procedural weapons/mass-harm request; no other agent ran."
        return lines
    if result.intent:
        how = {"rule": "a simple rule (no LLM call)", "llm": f"the LLM, {result.intent_confidence:.0%} confident",
               "heuristic": "a fallback heuristic (the LLM output was unusable)"}.get(result.intent_source, result.intent_source)
        why = f" — {result.intent_reason}" if result.intent_reason else ""
        lines.append(f"🧭 **Router** chose **{result.intent}** using {how}{why}.")
    destination = {"chit_chat": "Chit-chat responder (no retrieval)",
                   "out_of_scope": "Out-of-scope redirect (no retrieval)"}.get(result.intent)
    if destination:
        lines.append(f"➡️ Routed to **{destination}**.")
    elif result.answer_kind in ("rag", "not_found", "error"):
        if result.intent == "follow_up":
            lines.append(f"🔁 Follow-up: **Query Rewriter** used memory to rewrite it as “{result.standalone_query}”.")
        if result.sub_questions:
            lines.append(f"🧩 **Planner** split it into {len(result.sub_questions)} parts → **Per-part Retriever**.")
        elif result.plan_reason:
            lines.append(f"🧩 **Planner**: {result.plan_reason}. → single-query **Retriever**.")
        if result.retry_count:
            lines.append(f"♻️ **Context Grader** judged the evidence weak ({result.context_reason or 'insufficient'}) → "
                         f"looped back to the Query Rewriter for {result.retry_count} corrective retr{'y' if result.retry_count == 1 else 'ies'}.")
        elif not result.context_ok:
            lines.append(f"⚠️ **Context Grader**: evidence still weak ({result.context_reason}); the generator was told to answer cautiously.")
        if result.answer_kind == "not_found":
            lines.append("📭 Retrieval found nothing usable → **Not-found reply**.")
    return lines


def agents_called(result: AgentResult) -> list[dict]:
    """One row per trace step with the agent role, kind, what it actually did, and time."""
    rows = []
    for step in result.trace:
        node = normalize_node(step["node"])
        info = NODE_INFO.get(node, {"role": node, "kind": "rule", "does": ""})
        rows.append({"node": step["node"], "role": info["role"], "kind": info["kind"],
                     "detail": step["detail"], "seconds": step["seconds"]})
    return rows


def build_route_dot(drawable_graph, visited: list[str]) -> str:
    """Graphviz DOT of the REAL compiled graph with the taken path highlighted."""
    visited_set = set(visited)
    taken = {(a, b) for a, b in zip(["__start__"] + visited, visited + ["__end__"])}
    lines = ['digraph G {', 'rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.06"];',
             'edge [color="#cbd5e1", arrowsize=0.6];']
    for node in drawable_graph.nodes:
        if node in ("__start__", "__end__"):
            lines.append(f'"{node}" [shape=circle, label="{"start" if node == "__start__" else "end"}", '
                         f'fillcolor="#111827", fontcolor="white", width=0.4, fixedsize=true];')
            continue
        info = NODE_INFO.get(node, {"role": node, "kind": "rule"})
        label = f'{info["role"]}'
        if node in visited_set:
            lines.append(f'"{node}" [label="{label}", fillcolor="{KIND_COLOR[info["kind"]]}", fontcolor="white", color="{KIND_COLOR[info["kind"]]}"];')
        else:
            lines.append(f'"{node}" [label="{label}", fillcolor="#f3f4f6", fontcolor="#9ca3af", color="#e5e7eb"];')
    for edge in drawable_graph.edges:
        if (edge.source, edge.target) in taken:
            lines.append(f'"{edge.source}" -> "{edge.target}" [color="#2563eb", penwidth=2.4];')
        else:
            style = "dashed" if edge.conditional else "solid"
            lines.append(f'"{edge.source}" -> "{edge.target}" [style={style}];')
    lines.append("}")
    return "\n".join(lines)
