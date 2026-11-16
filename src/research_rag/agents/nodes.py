"""The agent's steps. Each public method is one LangGraph node: it receives the
state dict and returns ONLY the keys it changes.

Flow (see graph.py for the wiring):

  guard_input ─┬─(blocked)──────────────────────────────────────────► finalize
               └► classify ─┬─ chit_chat ────────► respond_chitchat ─► finalize
                            ├─ out of scope ─────► respond_scope ────► finalize
                            ├─ injection flag ───────────────────────► finalize
                            └─ follow_up | new_question
                                  ▼
                       rewrite_query ► hyde ► retrieve ─┬─(no hits)► respond_not_found
                             ▲                          └► grade ─┬─(weak & retries left)─┐
                             └────────────────────────────────────┘                       │
                                                                    └─(ok / out of retries)► generate
                                                                                    ▼
                                                                               output_guard ► finalize
"""
from __future__ import annotations

import logging
import time

from research_rag.corpus import CorpusInfo
from research_rag.generation import AnswerGenerator
from research_rag.guardrails import (
    InputGuardrails, OutputGuardrails, PromptInjectionGuard, event, is_refusal,
)
from langchain_core.documents import Document

from research_rag.llm import LLMClient, LLMError
from research_rag.models import Chunk
from research_rag.retrieval import AdvancedRetriever, BaseReranker, HyDEGenerator, NoOpReranker, QueryRewriter, QueryVariant
from . import responses
from .intent import IntentClassifier
from .planner import QuestionPlanner
from .memory import append_turn, fold_old_turns, format_recent_turns
from .prompts import CHITCHAT_PROMPT, GRADER_PROMPT, GROUNDEDNESS_PROMPT

logger = logging.getLogger(__name__)

INJECTION_REPLY = (
    "I can't act on that message because it looks like an attempt to change my instructions. "
    "I'm here to answer questions about the indexed research papers — feel free to ask one."
)


class AgentNodes:
    def __init__(
        self,
        *,
        llm: LLMClient,
        input_guard: InputGuardrails,
        output_guard: OutputGuardrails,
        injection: PromptInjectionGuard,
        classifier: IntentClassifier,
        planner: QuestionPlanner,
        rewriter: QueryRewriter,
        hyde: HyDEGenerator,
        retriever: AdvancedRetriever,
        reranker: BaseReranker,
        generator: AnswerGenerator,
        corpus: CorpusInfo,
        defaults: dict,
    ):
        self.llm = llm
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.injection = injection
        self.classifier = classifier
        self.planner = planner
        self.rewriter = rewriter
        self.hyde = hyde
        self.retriever = retriever
        self.reranker = reranker
        self.noop_reranker = NoOpReranker()
        self.generator = generator
        self.corpus = corpus
        self.defaults = defaults     # feature flags + numeric knobs from Settings

        # LCEL chains for the steps that call the LLM directly (the others own theirs)
        self.chitchat_chain = llm.text_chain(CHITCHAT_PROMPT, purpose="chitchat", temperature=0.3, max_tokens=140)
        self.grader_chain = llm.json_chain(
            GRADER_PROMPT, purpose="grade", max_tokens=100,
            default={"sufficient": True, "reason": "grader unavailable — assuming sufficient"})
        self.groundedness_chain = llm.json_chain(GROUNDEDNESS_PROMPT, purpose="groundedness", default={}, max_tokens=200)

    # ------------------------------------------------------------------ helpers
    def _opt(self, state: dict, key: str):
        return (state.get("options") or {}).get(key, self.defaults[key])

    @staticmethod
    def _step(state: dict, node: str, detail: str, t0: float) -> list[dict]:
        return state.get("trace", []) + [{"node": node, "detail": detail, "seconds": time.monotonic() - t0}]

    @staticmethod
    def _events(state: dict, new: list[dict]) -> list[dict]:
        return state.get("events", []) + new

    # ------------------------------------------------------------------ nodes
    def guard_input(self, state: dict) -> dict:
        t0 = time.monotonic()
        verdict = self.input_guard.check(state["user_message"], state["session_id"])
        update = {
            "clean_message": verdict.clean_text,
            "suspicious": verdict.suspicious,
            "events": self._events(state, verdict.events),
        }
        if verdict.blocked:
            kind = "safety" if verdict.block_kind == "self_harm" else "blocked"
            update.update(blocked=True, block_kind=verdict.block_kind, answer=verdict.message,
                          answer_kind=kind, store_turn=verdict.store_in_memory)
            detail = f"blocked: {verdict.block_kind}"
        else:
            detail = "passed" + (" (flagged)" if verdict.suspicious else "")
        update["trace"] = self._step(state, "guard_input", detail, t0)
        return update

    def classify(self, state: dict) -> dict:
        t0 = time.monotonic()
        result = self.classifier.classify(
            state["clean_message"],
            recent_turns=format_recent_turns(state.get("history", [])),
            summary=state.get("summary", ""),
        )
        update = {
            "intent": result.intent, "intent_confidence": result.confidence,
            "intent_reason": result.reason, "intent_source": result.source,
        }
        if result.injection_flag:
            update.update(
                blocked=True, block_kind="injection", answer=INJECTION_REPLY, answer_kind="blocked",
                store_turn=False,
                events=self._events(state, [event("prompt_injection", "blocked", "flagged by the intent classifier")]),
            )
        update["trace"] = self._step(
            state, "classify", f"{result.intent} ({result.confidence:.2f}, via {result.source})", t0)
        return update

    def respond_chitchat(self, state: dict) -> dict:
        t0 = time.monotonic()
        try:
            answer = self.chitchat_chain.invoke(
                {"corpus": self.corpus.describe(), "message": state["clean_message"]}).strip()
        except LLMError:
            answer = ""
        if not answer:
            answer = responses.CHITCHAT_FALLBACK.format(scope=self.corpus.short_scope())
        return {"answer": answer, "answer_kind": "chitchat",
                "trace": self._step(state, "respond_chitchat", "no retrieval performed", t0)}

    def respond_scope(self, state: dict) -> dict:
        t0 = time.monotonic()
        return {
            "answer": responses.OUT_OF_SCOPE.format(scope=self.corpus.short_scope()),
            "answer_kind": "out_of_scope",
            "events": self._events(state, [event("scope_redirect", "redirected", "out-of-scope request")]),
            "trace": self._step(state, "respond_scope", "no retrieval performed", t0),
        }

    def respond_not_found(self, state: dict) -> dict:
        t0 = time.monotonic()
        return {
            "answer": responses.NOT_FOUND.format(scope=self.corpus.short_scope()),
            "answer_kind": "not_found",
            "trace": self._step(state, "respond_not_found", "retrieval returned nothing usable", t0),
        }

    # -- retrieval agent ----------------------------------------------------------
    def rewrite_query(self, state: dict) -> dict:
        t0 = time.monotonic()
        retry = state.get("retry_count", 0) > 0
        message = state["clean_message"]

        if not self._opt(state, "enable_query_rewrite"):
            return {"standalone_query": message, "alternatives": [],
                    "trace": self._step(state, "rewrite_query", "disabled — using the message as written", t0)}

        if retry:   # corrective pass: broaden the previous standalone question, no history needed
            result = self.rewriter.rewrite(state.get("standalone_query") or message, mode="broaden")
        elif state["intent"] == "follow_up":
            result = self.rewriter.rewrite(
                message, summary=state.get("summary", ""),
                recent_turns=format_recent_turns(state.get("history", []), n_turns=3))
        else:   # new question: deliberately ignore history so an old topic can't leak in
            result = self.rewriter.rewrite(message)

        mode = "broaden (retry)" if retry else ("follow-up → standalone" if state["intent"] == "follow_up" else "standalone")
        return {"standalone_query": result.standalone_question, "alternatives": result.alternatives,
                "trace": self._step(state, "rewrite_query", f"{mode}: {result.standalone_question[:90]}", t0)}

    def plan(self, state: dict) -> dict:
        """Planner agent: split a multi-part question into self-contained sub-questions.
        Not re-run on a corrective retry (the retry falls back to one broadened query)."""
        t0 = time.monotonic()
        question = state["standalone_query"] or state["clean_message"]
        if not self._opt(state, "enable_planner") or state.get("retry_count", 0) > 0:
            why = "disabled" if not self._opt(state, "enable_planner") else "skipped on corrective retry"
            return {"sub_questions": [], "plan_reason": why, "plan_coverage": [],
                    "trace": self._step(state, "plan", why, t0)}
        self.planner.max_sub_questions = self._opt(state, "max_sub_questions")
        result = self.planner.plan(question)
        detail = (f"split into {len(result.sub_questions)} sub-questions" if result.is_compound
                  else f"single question — {result.reason}")
        return {"sub_questions": result.sub_questions if result.is_compound else [],
                "plan_reason": result.reason, "plan_coverage": [],
                "events": self._events(state, [event("planner", "split", f"{len(result.sub_questions)} parts")]
                                      if result.is_compound else []),
                "trace": self._step(state, "plan", detail, t0)}

    def retrieve_plan(self, state: dict) -> dict:
        """Planned path: retrieve + rerank PER sub-question, then merge round-robin so every part keeps evidence."""
        t0 = time.monotonic()
        subs = state["sub_questions"]
        per_sub = max(2, -(-self.defaults["rerank_top_k"] // len(subs)))      # ceil
        reranker = self.reranker if self._opt(state, "enable_reranker") else self.noop_reranker
        events, trace, per_part, used, n_cands = [], list(state.get("trace", [])), [], [], 0

        for i, sub in enumerate(subs, start=1):
            t1 = time.monotonic()
            variants = [QueryVariant("original", sub)]
            if self._opt(state, "enable_hyde"):
                passage = self.hyde.generate(sub)
                if passage:
                    variants.append(QueryVariant("hyde", passage, self.retriever.hyde_weight))
            try:
                result = self.retriever.search_variants(variants, top_k=self.defaults["retrieval_top_k"])
            except Exception as error:  # noqa: BLE001 — fail closed, same as the single path
                logger.error(f"Planned retrieval failed on part {i}: {error}")
                trace.append({"node": "retrieve_plan", "detail": f"FAILED on part {i}: {error}", "seconds": time.monotonic() - t0})
                return {"answer": responses.RETRIEVAL_ERROR, "answer_kind": "error", "store_turn": False,
                        "error": str(error), "trace": trace}
            candidates, ev = self._drop_poisoned(result.documents)
            events += ev
            n_cands += len(candidates)
            used += [f"q{i}:{v}" for v in result.variants_used]
            per_part.append(reranker.rerank(sub, candidates, top_k=per_sub))
            trace.append({"node": f"retrieve_plan[{i}]", "detail": f"“{sub[:70]}” → {len(candidates)} candidates → top {len(per_part[-1])}",
                          "seconds": time.monotonic() - t1})

        merged, seen, cap = [], set(), self.defaults["rerank_top_k"] + len(subs)
        for rank in range(per_sub):                       # round-robin: part 1's best, part 2's best, ...
            for part in per_part:
                if rank < len(part) and part[rank].id not in seen and len(merged) < cap:
                    seen.add(part[rank].id)
                    merged.append(part[rank])
        position = {doc.id: i for i, doc in enumerate(merged, start=1)}
        coverage = [{"question": sub, "sources": [f"S{position[d.id]}" for d in part if d.id in position]}
                    for sub, part in zip(subs, per_part)]
        return {"n_candidates": n_cands, "variants_used": used, "top_chunks": _to_state(merged),
                "plan_coverage": coverage, "events": self._events(state, events), "trace": trace}

    def _drop_poisoned(self, docs: list[Document]) -> tuple[list[Document], list[dict]]:
        events, safe = [], []
        for doc in docs:
            scan = self.injection.scan_chunk_text(doc.page_content) if self.injection.enabled else None
            if scan and scan.blocked:
                events.append(event("retrieved_content_scan", "dropped",
                                    f"{doc.metadata.get('paper_id')} p.{doc.metadata.get('page_no')}: "
                                    f"looks like an instruction ({', '.join(scan.families)})"))
            else:
                safe.append(doc)
        return safe, events

    def hyde_node(self, state: dict) -> dict:
        t0 = time.monotonic()
        retry = state.get("retry_count", 0) > 0
        if not self._opt(state, "enable_hyde"):
            return {"hyde_passage": "", "trace": self._step(state, "hyde", "disabled", t0)}
        passage = self.hyde.generate(state["standalone_query"] or state["clean_message"])
        return {"hyde_passage": passage,
                "trace": self._step(state, "hyde", f"{len(passage)} chars" + (" (retry)" if retry else ""), t0)}

    def retrieve(self, state: dict) -> dict:
        """Multi-variant hybrid search -> RRF fusion -> injection scan -> rerank."""
        t0 = time.monotonic()
        query = state["standalone_query"] or state["clean_message"]
        variants = [QueryVariant("original", state["clean_message"]), QueryVariant("rewritten", query)]
        variants += [QueryVariant(f"alt{i}", alt) for i, alt in enumerate(state.get("alternatives", []), start=1)]
        if state.get("hyde_passage"):
            variants.append(QueryVariant("hyde", state["hyde_passage"], self.retriever.hyde_weight))

        try:
            result = self.retriever.search_variants(variants, top_k=self.defaults["retrieval_top_k"])
        except Exception as error:  # noqa: BLE001 — index down: fail closed with a clear message
            logger.error(f"Retrieval failed: {error}")
            return {"answer": responses.RETRIEVAL_ERROR, "answer_kind": "error", "store_turn": False,
                    "error": str(error), "trace": self._step(state, "retrieve", f"FAILED: {error}", t0)}
        t_search = time.monotonic() - t0

        candidates, events = self._drop_poisoned(result.documents)

        t1 = time.monotonic()
        reranker = self.reranker if self._opt(state, "enable_reranker") else self.noop_reranker
        top = reranker.rerank(query, candidates, top_k=self.defaults["rerank_top_k"])
        rerank_label = "cross-encoder" if reranker is self.reranker else "off"

        trace = self._step(state, "retrieve", f"{len(candidates)} fused candidates from {len(result.variants_used)} query variant(s) "
                                              f"[{', '.join(result.variants_used)}]", t0)
        trace[-1]["seconds"] = t_search
        trace.append({"node": "rerank", "detail": f"{len(candidates)} → top {len(top)} ({rerank_label})",
                      "seconds": time.monotonic() - t1})
        return {
            "n_candidates": len(candidates), "variants_used": result.variants_used,
            "top_chunks": _to_state(top),
            "events": self._events(state, events), "trace": trace,
        }

    def grade(self, state: dict) -> dict:
        """Agentic check: can these excerpts answer the question? If not, and a
        retry is allowed, loop back to rewrite_query in 'broaden' mode."""
        t0 = time.monotonic()
        if not self._opt(state, "enable_context_grading"):
            return {"context_ok": True, "next_action": "generate",
                    "trace": self._step(state, "grade", "disabled", t0)}

        query = state["standalone_query"] or state["clean_message"]
        excerpts = "\n\n".join(f"[{i}] {c['text'][:600]}" for i, c in enumerate(state["top_chunks"], start=1))
        verdict = self.grader_chain.invoke({"question": query, "excerpts": excerpts})
        ok = bool(verdict.get("sufficient", True))
        reason = str(verdict.get("reason", ""))[:160]
        retries_left = state.get("retry_count", 0) < self.defaults["max_retrieval_retries"]

        if ok:
            return {"context_ok": True, "context_reason": reason, "next_action": "generate",
                    "trace": self._step(state, "grade", f"sufficient — {reason}", t0)}
        if retries_left:
            return {"context_ok": False, "context_reason": reason, "next_action": "retry",
                    "retry_count": state.get("retry_count", 0) + 1,
                    "events": self._events(state, [event("context_grading", "retry", f"weak evidence: {reason}")]),
                    "trace": self._step(state, "grade", f"insufficient → corrective retry — {reason}", t0)}
        return {"context_ok": False, "context_reason": reason, "next_action": "generate",
                "events": self._events(state, [event("context_grading", "flagged", f"still weak after retry: {reason}")]),
                "trace": self._step(state, "grade", f"insufficient (no retries left) — {reason}", t0)}

    def generate(self, state: dict) -> dict:
        t0 = time.monotonic()
        chunks = [Chunk.from_dict(d) for d in state["top_chunks"]]
        question = state["standalone_query"] or state["clean_message"]
        low_context = not state.get("context_ok", True)
        hardened = state.get("suspicious", False)
        events: list[dict] = []

        try:
            subs = state.get("sub_questions") or None
            response = self.generator.generate(question, chunks, low_context=low_context, hardened=hardened,
                                               sub_questions=subs)
            if not response.answer.strip() or not (any(s.cited for s in response.sources) or is_refusal(response.answer)):
                logger.info("Answer had no valid citations — regenerating once with a citation reminder")
                events.append(event("citation_enforcement", "regenerated", "first draft carried no valid [S#] citation"))
                response = self.generator.generate(question, chunks, low_context=low_context,
                                                   hardened=hardened, citation_reminder=True,
                                                   sub_questions=subs)
        except LLMError as error:
            return {"answer": responses.LLM_ERROR, "answer_kind": "error", "store_turn": False, "error": str(error),
                    "trace": self._step(state, "generate", f"FAILED: {error}", t0)}

        return {
            "answer": response.answer, "answer_kind": "rag",
            "sources": [s.to_dict() for s in response.sources],
            "events": self._events(state, events),
            "trace": self._step(state, "generate",
                                f"{len(response.answer)} chars, {sum(s.cited for s in response.sources)}/{len(response.sources)} sources cited", t0),
        }

    def output_guard_node(self, state: dict) -> dict:
        t0 = time.monotonic()
        answer, events = state["answer"], []

        answer, ev = self.output_guard.check_leak(answer)
        events += ev
        leaked = bool(ev)
        sources = [] if leaked else state.get("sources", [])

        answer, ev = self.output_guard.scan_pii(answer)
        events += ev

        if not leaked:
            answer, ev = self.output_guard.check_references(
                answer,
                paper_ids=[c["paper_id"] for c in state.get("top_chunks", [])],
                excerpts=[c["text"] for c in state.get("top_chunks", [])],
            )
            events += ev

        cited_any = any(s["cited"] for s in sources)
        refusal = is_refusal(answer)
        citations_ok = leaked or cited_any or refusal
        if not citations_ok:
            answer += "\n\n⚠️ *This answer did not include source citations. Please verify it against the papers before relying on it.*"
            events.append(event("citation_enforcement", "flagged", "no valid citations after regeneration — warning added"))

        grounded, unsupported = None, []
        if (self._opt(state, "enable_groundedness_check") and not leaked and not refusal and sources):
            grounded, unsupported = self._check_groundedness(answer, state["top_chunks"])
            if grounded is False:
                claims = "; ".join(unsupported) if unsupported else "one or more statements"
                answer += (f"\n\n⚠️ *An automatic check found statements that may not be fully supported by the retrieved "
                           f"excerpts ({claims}). Please verify against the cited sources.*")
                events.append(event("groundedness", "flagged", claims))

        return {"answer": answer, "sources": sources, "citations_ok": citations_ok,
                "grounded": grounded, "unsupported_claims": unsupported,
                "events": self._events(state, events),
                "trace": self._step(state, "output_guard",
                                    f"citations {'ok' if citations_ok else 'MISSING'}; groundedness "
                                    f"{'skipped' if grounded is None else 'ok' if grounded else 'FLAGGED'}", t0)}

    def _check_groundedness(self, answer: str, top_chunks: list[dict]) -> tuple[bool | None, list[str]]:
        excerpts = "\n\n".join(f"[S{i}] {c['text'][:900]}" for i, c in enumerate(top_chunks, start=1))
        verdict = self.groundedness_chain.invoke({"excerpts": excerpts, "answer": answer})
        if "grounded" not in verdict:
            return None, []       # checker unavailable: report "not checked" rather than pretending it passed
        claims = [str(c)[:140] for c in (verdict.get("unsupported_claims") or [])][:3]
        return bool(verdict["grounded"]), claims

    def finalize(self, state: dict) -> dict:
        """Memory update + source bookkeeping. Runs on EVERY path."""
        t0 = time.monotonic()
        update: dict = {}
        history, summary = state.get("history", []), state.get("summary", "")

        if state.get("store_turn", True) and state.get("clean_message") and state.get("answer"):
            history = append_turn(history, state["clean_message"], state["answer"], state.get("intent") or state.get("answer_kind", ""))
            history, summary = fold_old_turns(history, summary, self.defaults["memory_max_turns"], self.llm)
            if state.get("answer_kind") == "rag":
                cited = [s["paper_id"] for s in state.get("sources", []) if s.get("cited")]
                update["focus"] = {"question": state.get("standalone_query", ""), "papers": list(dict.fromkeys(cited))}
        update.update(history=history, summary=summary)
        update["trace"] = self._step(state, "finalize", f"memory: {len(history) // 2} turn(s) kept" + (", summary present" if summary else ""), t0)
        return update

    # ------------------------------------------------------------------ routing
    @staticmethod
    def route_after_guard(state: dict) -> str:
        return "finalize" if state.get("blocked") else "classify"

    @staticmethod
    def route_after_classify(state: dict) -> str:
        if state.get("blocked"):
            return "finalize"
        return {
            "chit_chat": "respond_chitchat",
            "out_of_scope": "respond_scope",
        }.get(state["intent"], "rewrite_query")     # follow_up | new_question

    @staticmethod
    def route_after_plan(state: dict) -> str:
        return "retrieve_plan" if state.get("sub_questions") else "hyde"

    @staticmethod
    def route_after_retrieve(state: dict) -> str:
        if state.get("answer_kind") == "error":
            return "finalize"
        return "grade" if state.get("top_chunks") else "respond_not_found"

    @staticmethod
    def route_after_grade(state: dict) -> str:
        return "rewrite_query" if state.get("next_action") == "retry" else "generate"


def _to_state(docs: list[Document]) -> list[dict]:
    """Documents -> JSON-safe chunk dicts for the LangGraph state (keeps checkpoints serializable)."""
    return [Chunk.from_document(doc).to_dict() for doc in docs]
