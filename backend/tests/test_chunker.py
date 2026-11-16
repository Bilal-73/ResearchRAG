"""Data-model tests: Chunk/SourceCitation serialisation and provenance formatting.

(Chunking itself is covered end-to-end in test_ingestion.py against a
miniature Open RAG Benchmark corpus.)
"""
from research_rag.models import Chunk, ChunkType, SourceCitation
from research_rag.models.response import arxiv_url


def _chunk(**overrides):
    base = dict(id="2410.14077v2::s8::t0", text="RMSPE of 2.3426 percent.", chunk_type=ChunkType.TEXT,
                paper_id="2410.14077v2", section="A. Performance comparison", section_id=8,
                paper_title="Inverter Output Impedance Estimation")
    return Chunk(**{**base, **overrides})


def test_chunk_type_enum_values():
    assert ChunkType.TEXT.value == "text"
    assert ChunkType.TABLE.value == "table"
    assert ChunkType.FIGURE.value == "figure"


def test_round_trip_through_dict_preserves_benchmark_fields():
    """LangGraph checkpoints store chunks as plain dicts — section_id must survive,
    since it is what retrieval is scored against."""
    original = _chunk(score=0.42)
    restored = Chunk.from_dict(original.to_dict())
    assert restored.section_id == 8 and restored.paper_id == "2410.14077v2"
    assert restored.paper_title == original.paper_title and restored.score == 0.42


def test_langchain_document_view_carries_the_payload_as_metadata():
    chunk = _chunk()
    doc = chunk.to_document()
    assert doc.id == chunk.id and doc.page_content == chunk.text
    assert doc.metadata == {k: v for k, v in chunk.to_payload().items() if k != "text"}
    assert Chunk.from_document(doc).to_dict() == chunk.to_dict()


def test_payload_round_trip_tolerates_missing_optional_fields():
    """Chunks indexed before section_id existed must still load."""
    restored = Chunk.from_payload("id-1", {"text": "x", "chunk_type": "text", "paper_id": "2410.14077v2"})
    assert restored.section == "unknown" and restored.section_id is None and restored.page_no is None


def test_location_reports_section_and_figure_provenance():
    assert _chunk().location() == "2410.14077v2 › §8 A. Performance comparison"
    table = _chunk(chunk_type=ChunkType.TABLE, figure_no="I")
    assert table.location().endswith("Table I")
    figure = _chunk(chunk_type=ChunkType.FIGURE, figure_no="1", section_id=1, section="I. Introduction")
    assert figure.location() == "2410.14077v2 › §1 I. Introduction, Figure 1"


def test_location_includes_a_page_when_one_is_known():
    assert "p.4" in _chunk(page_no=4).location()


def test_arxiv_url_only_for_real_arxiv_ids():
    assert arxiv_url("2410.14077v2") == "https://arxiv.org/abs/2410.14077v2"
    assert arxiv_url("2410.14077") == "https://arxiv.org/abs/2410.14077"
    assert arxiv_url("some-local-pdf") is None


def test_source_citation_round_trip_and_format():
    citation = SourceCitation(paper_id="2410.14077v2", section="IV. Results", page_no=None,
                              snippet="RMSPE ...", label="S1", section_id=8, cited=True,
                              url="https://arxiv.org/abs/2410.14077v2")
    assert citation.format() == "2410.14077v2 › §8 IV. Results"
    assert SourceCitation.from_dict(citation.to_dict()) == citation
