import json

from fakes import Harness
from research_rag.agents import QuestionPlanner, looks_compound
from research_rag.llm import LLMClient
from fakes import FakeChatModel

COMPOUND = "What RMSPE did the proposed algorithm reach and how does the Kalman filter behave without excitation?"
SPLIT = lambda m: json.dumps({"sub_questions": ["What RMSPE did the proposed algorithm reach under excitation?",
                                                "How does the Kalman filter estimation behave without excitation?"],
                              "reason": "two topics"})


# ---------------- cheap rule gate ----------------
def test_rule_gate_flags_compound_and_spares_simple_questions():
    assert looks_compound(COMPOUND)
    assert looks_compound("Compare the proposed method and the Kalman filter on RMSPE")
    assert looks_compound("What was the RMSPE? How many simulations were run?")
    assert not looks_compound("What RMSPE did the proposed algorithm reach?")
    assert not looks_compound("Did the proposed method outperform the Kalman filter baseline?")


# ---------------- planner unit ----------------
def _planner(handler, n=3):
    fake = FakeChatModel(); fake.handlers["plan"] = handler
    return QuestionPlanner(LLMClient(fake), n), fake


def test_simple_question_makes_no_llm_call():
    planner, fake = _planner(SPLIT)
    assert not planner.plan("What RMSPE did the proposed algorithm reach?").is_compound and fake.count("plan") == 0


def test_splits_dedupes_and_caps():
    many = lambda m: json.dumps({"sub_questions": ["a?", "A?", "b?", "c?", "d?", "  "], "reason": "x"})
    result = _planner(many, n=3)[0].plan(COMPOUND)
    assert result.sub_questions == ["a?", "b?", "c?"]


def test_llm_keeping_it_whole_or_failing_means_no_plan():
    assert not _planner(lambda m: json.dumps({"sub_questions": ["only one?"]}))[0].plan(COMPOUND).is_compound
    assert not _planner(lambda m: "garbage")[0].plan(COMPOUND).is_compound
    assert not _planner(lambda m: RuntimeError("down"))[0].plan(COMPOUND).is_compound


# ---------------- in the graph ----------------
def test_compound_question_is_planned_retrieved_per_part_and_answered_with_parts():
    h = Harness()
    h.model.handlers["plan"] = SPLIT
    seen = []
    orig = h.model.handlers["generate"]
    h.model.handlers["generate"] = lambda m: (seen.append(m[-1]["content"]), orig(m))[1]
    r = h.ask(COMPOUND)
    assert len(r.sub_questions) == 2 and r.answer_kind == "rag"
    assert "several parts" in seen[0] and "1. What RMSPE did the proposed algorithm reach under excitation?" in seen[0]
    section_ids = {c.section_id for c in r.chunks}
    assert {8, 9} <= section_ids                                          # each part kept its own evidence
    assert len(r.plan_coverage) == 2 and all(c["sources"] for c in r.plan_coverage)
    nodes = [t["node"] for t in r.trace]
    assert "plan" in nodes and "retrieve_plan[1]" in nodes and "retrieve_plan[2]" in nodes and "retrieve" not in nodes
    assert h.model.count("hyde") == 2                                    # one HyDE passage per part
    assert any(e["guardrail"] == "planner" for e in r.events)


def test_simple_question_skips_planner_entirely():
    h = Harness()
    r = h.ask("What RMSPE did the proposed algorithm reach?")
    assert r.sub_questions == [] and h.model.count("plan") == 0
    assert "retrieve" in [t["node"] for t in r.trace]


def test_planner_toggle_off():
    h = Harness(); h.model.handlers["plan"] = SPLIT
    r = h.ask(COMPOUND, enable_planner=False)
    assert r.sub_questions == [] and h.model.count("plan") == 0


def test_follow_up_compound_is_planned_after_being_made_standalone():
    h = Harness(); h.model.handlers["plan"] = SPLIT
    h.ask("What RMSPE did the proposed algorithm reach?")
    orig = h.model.handlers["rewrite"]
    h.model.handlers["rewrite"] = lambda m: json.dumps({"standalone_question": COMPOUND, "alternatives": []})
    r = h.ask("what about without excitation, and the RMSPE too?")
    assert r.intent == "follow_up" and len(r.sub_questions) == 2


def test_weak_evidence_retry_falls_back_to_single_broadened_query():
    h = Harness(); h.model.handlers["plan"] = SPLIT
    verdicts = iter([{"sufficient": False, "reason": "weak"}, {"sufficient": True, "reason": "ok"}])
    h.model.handlers["grade"] = lambda m: json.dumps(next(verdicts))
    r = h.ask(COMPOUND)
    assert r.retry_count == 1 and r.sub_questions == [] and "retrieve" in [t["node"] for t in r.trace]
    assert h.model.count("plan") == 1                                    # not re-planned on the retry


def test_planned_path_drops_poisoned_chunks_and_handles_empty_index():
    from fakes import CORPUS, POISONED
    h = Harness(chunks=CORPUS + [POISONED]); h.model.handlers["plan"] = SPLIT
    r = h.ask(COMPOUND)
    assert all(c.id != "cx" for c in r.chunks)
    empty = Harness(chunks=[]); empty.model.handlers["plan"] = SPLIT
    assert empty.ask(COMPOUND).answer_kind == "not_found"


def test_planned_retrieval_outage_fails_closed():
    h = Harness(); h.model.handlers["plan"] = SPLIT
    h.store.hybrid_search_by_vector = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("qdrant down"))
    r = h.ask(COMPOUND)
    assert r.answer_kind == "error" and h.agent.memory_snapshot("t1")["turns"] == 0
