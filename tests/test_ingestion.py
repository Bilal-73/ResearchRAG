"""Ingestion tests against a miniature Open RAG Benchmark corpus.

The fixture mirrors the real dataset's quirks exactly — markdown placeholders
for tables and images, captions on the line before or after, a base64 data
URI, a references section — because those quirks are what the parser exists to
handle.
"""
from __future__ import annotations

import base64
import json

import pytest

from research_rag.ingestion import (
    IngestionPipeline, RAGBenchCorpus, SectionChunker, parse_document, parse_section,
)
from research_rag.models import ChunkType

PNG_1PX = base64.b64encode(
    bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a"
                  "49444154789c6300010000050001")
).decode()

RAW_DOC = {
    "id": "2410.14077v2",
    "title": "Inverter Output Impedance Estimation\n  in Power Networks",
    "authors": ["Jaesang Park", "Alireza Askarian"],
    "categories": ["cs.SY"],
    "abstract": "We present a non-invasive method for estimating output-line impedance.",
    "published": "2024-10-17T22:49:53Z",
    "updated": "2025-04-09T17:59:27Z",
    "sections": [
        {"section_id": 0, "text": "#### Abstract\n\nWe present a non-invasive method.", "tables": {}, "images": {}},
        {"section_id": 1,
         "text": "## I. INTRODUCTION\n\nDistributed energy resources are expanding rapidly.\n\n"
                 "[^0]![img-0.jpeg](img-0.jpeg)\n\nFig. 1: Grid model showing the Thevenin equivalent.\n\n"
                 "In a complex power system, we model the grid as seen by the inverter.",
         "tables": {}, "images": {"img-0.jpeg": f"data:image/png;base64,{PNG_1PX}"}},
        {"section_id": 8,
         "text": "## A. Performance comparison\n\nThese algorithms become susceptible to noise.\n\n"
                 "TABLE I: RMSPE of R and L estimation under different algorithms\n\n"
                 "![table_0](table_0)\n\nThe proposed method achieves the lowest error.",
         "tables": {"table_0": "| Algorithm | RMSPE R(%) |\n| :--: | :--: |\n| VDF-RLS | 2.3426 |"},
         "images": {}},
        {"section_id": 11, "text": "## REFERENCES\n\n[1] T. Hossen, Grid-forming inverters, 2021.",
         "tables": {}, "images": {}},
    ],
}


@pytest.fixture
def corpus_dir(tmp_path):
    root = tmp_path / "open_ragbench"
    (root / "corpus").mkdir(parents=True)
    (root / "corpus" / "2410.14077v2.json").write_text(json.dumps(RAW_DOC), encoding="utf-8")
    (root / "queries.json").write_text(json.dumps({
        "q1": {"query": "What RMSPE did VDF-RLS reach?", "type": "extractive", "source": "text-table"},
        "q2": {"query": "What does Figure 1 show?", "type": "abstractive", "source": "text-image"},
        "q3": {"query": "A question about an unindexed paper", "type": "abstractive", "source": "text"},
    }), encoding="utf-8")
    (root / "qrels.json").write_text(json.dumps({
        "q1": {"doc_id": "2410.14077v2", "section_id": 8},
        "q2": {"doc_id": "2410.14077v2", "section_id": 1},
        "q3": {"doc_id": "2401.07294v4", "section_id": 3},
    }), encoding="utf-8")
    (root / "answers.json").write_text(json.dumps({
        "q1": "VDF-RLS reached an RMSPE of 2.3426%.", "q2": "A Thevenin equivalent grid model.",
        "q3": "Something else.",
    }), encoding="utf-8")
    (root / "pdf_urls.json").write_text(json.dumps({
        "2410.14077v2": "https://arxiv.org/pdf/2410.14077v2",
        "2401.07294v4": "https://arxiv.org/pdf/2401.07294v4",
    }), encoding="utf-8")
    return root


# ---------------------------------------------------------------- parsing ------
def test_section_parsing_extracts_heading_prose_and_assets():
    section = parse_section(RAW_DOC["sections"][2])
    assert section.section_id == 8
    assert section.heading == "A. Performance comparison"
    assert "![table_0]" not in section.text and "placeholder" not in section.text
    assert "The proposed method achieves the lowest error." in section.text
    assert len(section.assets) == 1
    asset = section.assets[0]
    assert asset.kind == "table" and asset.number == "I"
    assert "RMSPE of R and L estimation" in asset.caption
    assert "VDF-RLS | 2.3426" in asset.content


def test_figure_caption_is_found_after_the_placeholder_and_footnotes_dropped():
    section = parse_section(RAW_DOC["sections"][1])
    assert "[^0]" not in section.text
    figure = section.assets[0]
    assert figure.kind == "figure" and figure.number == "1"
    assert "Thevenin equivalent" in figure.caption


def test_images_are_only_decoded_when_requested():
    assert parse_section(RAW_DOC["sections"][1]).assets[0].data_uri == ""
    loaded = parse_section(RAW_DOC["sections"][1], load_images=True).assets[0]
    assert loaded.image_bytes.startswith(b"\x89PNG")


def test_document_metadata_is_normalised():
    doc = parse_document(RAW_DOC)
    assert doc.title == "Inverter Output Impedance Estimation in Power Networks"   # newline collapsed
    assert doc.categories == ["cs.SY"] and len(doc.sections) == 4
    assert doc.sections[3].is_reference_section


# ---------------------------------------------------------------- chunking -----
def test_chunks_carry_section_id_title_and_never_span_sections():
    chunks = SectionChunker(max_tokens=400).chunk_document(parse_document(RAW_DOC))
    assert all(c.paper_id == "2410.14077v2" for c in chunks)
    assert all(c.section_id is not None for c in chunks)
    assert all(c.paper_title.startswith("Inverter Output") for c in chunks)
    assert {c.section_id for c in chunks} == {0, 1, 8}         # references skipped


def test_tables_and_figures_become_their_own_chunks():
    chunks = SectionChunker().chunk_document(parse_document(RAW_DOC))
    by_type = {c.chunk_type: c for c in chunks}
    assert by_type[ChunkType.TABLE].section_id == 8
    assert "2.3426" in by_type[ChunkType.TABLE].text
    assert by_type[ChunkType.FIGURE].section_id == 1
    assert "Thevenin" in by_type[ChunkType.FIGURE].text


def test_every_chunk_is_contextualised_with_title_and_heading():
    chunks = SectionChunker().chunk_document(parse_document(RAW_DOC))
    body = next(c for c in chunks if c.section_id == 8 and c.chunk_type == ChunkType.TEXT)
    assert body.text.startswith("Inverter Output Impedance Estimation in Power Networks › A. Performance comparison")


def test_reference_sections_can_be_kept():
    chunks = SectionChunker(skip_reference_sections=False).chunk_document(parse_document(RAW_DOC))
    assert 11 in {c.section_id for c in chunks}


def test_long_sections_are_packed_with_overlap():
    paragraphs = [f"Paragraph {i} describes the estimation procedure in detail. " * 6 for i in range(8)]
    doc = parse_document({**RAW_DOC, "sections": [
        {"section_id": 2, "text": "## II. METHOD\n\n" + "\n\n".join(paragraphs), "tables": {}, "images": {}}]})
    chunks = SectionChunker(max_tokens=120, overlap_paragraphs=1).chunk_document(doc)
    text_chunks = [c for c in chunks if c.section_id == 2]
    assert len(text_chunks) > 1
    # consecutive chunks share the overlapping paragraph
    assert any(p in text_chunks[1].text for p in text_chunks[0].text.split("\n\n")[-1:])


def test_figure_without_caption_or_vlm_is_skipped():
    raw = {**RAW_DOC, "sections": [{"section_id": 3, "text": "## III. X\n\nBody.\n\n![img-9.jpeg](img-9.jpeg)",
                                    "tables": {}, "images": {"img-9.jpeg": "data:image/png;base64,AAAA"}}]}
    chunks = SectionChunker().chunk_document(parse_document(raw))
    assert not any(c.chunk_type == ChunkType.FIGURE for c in chunks)


def test_captioner_output_is_folded_into_the_figure_chunk():
    class FakeCaptioner:
        def caption(self, image_bytes, prompt):
            return "A line plot of impedance against time with two traces."

    doc = parse_document(RAW_DOC, load_images=True)
    chunks = SectionChunker(captioner=FakeCaptioner()).chunk_document(doc)
    figure = next(c for c in chunks if c.chunk_type == ChunkType.FIGURE)
    assert "line plot of impedance" in figure.text


def test_captioner_failure_falls_back_to_the_paper_caption():
    class BrokenCaptioner:
        def caption(self, image_bytes, prompt):
            raise RuntimeError("VLM down")

    doc = parse_document(RAW_DOC, load_images=True)
    figure = next(c for c in SectionChunker(captioner=BrokenCaptioner()).chunk_document(doc)
                  if c.chunk_type == ChunkType.FIGURE)
    assert "Thevenin" in figure.text


# ---------------------------------------------------------------- corpus/pipeline
def test_corpus_reports_available_and_judged_ids(corpus_dir):
    corpus = RAGBenchCorpus(corpus_dir)
    assert corpus.available_ids() == ["2410.14077v2"]
    assert corpus.judged_ids() == ["2401.07294v4", "2410.14077v2"]      # judged ≠ downloaded
    assert len(corpus.queries()) == 3


def test_missing_metadata_file_explains_how_to_fix_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_corpus.py"):
        RAGBenchCorpus(tmp_path).queries()


def test_pipeline_embeds_upserts_and_writes_a_manifest(corpus_dir, tmp_path):
    from fakes import FakeEmbedder, FakeVectorStore

    store = FakeVectorStore(chunks=[])
    manifest = tmp_path / "manifest.json"
    pipeline = IngestionPipeline(RAGBenchCorpus(corpus_dir), SectionChunker(),
                                 FakeEmbedder(), store, manifest_path=manifest)

    report = pipeline.ingest(["2410.14077v2"])
    assert report.papers_ingested == 1 and report.chunks_upserted > 0
    assert report.by_type["table"] == 1 and report.by_type["figure"] == 1
    assert all(c.dense_vector and c.sparse_vector for c in store.upserted)

    entry = json.loads(manifest.read_text(encoding="utf-8"))["2410.14077v2"]
    assert entry["categories"] == ["cs.SY"] and entry["n_chunks"] == report.chunks_upserted


def test_rerunning_skips_indexed_papers_unless_forced(corpus_dir, tmp_path):
    from fakes import FakeEmbedder, FakeVectorStore

    manifest = tmp_path / "manifest.json"
    pipeline = IngestionPipeline(RAGBenchCorpus(corpus_dir), SectionChunker(),
                                 FakeEmbedder(), FakeVectorStore(chunks=[]), manifest_path=manifest)
    pipeline.ingest(["2410.14077v2"])
    again = pipeline.ingest(["2410.14077v2"])
    assert again.papers_skipped == 1 and again.papers_ingested == 0
    forced = pipeline.ingest(["2410.14077v2"], force=True)
    assert forced.papers_ingested == 1


def test_one_bad_paper_does_not_abort_the_run(corpus_dir, tmp_path):
    from fakes import FakeEmbedder, FakeVectorStore

    pipeline = IngestionPipeline(RAGBenchCorpus(corpus_dir), SectionChunker(),
                                 FakeEmbedder(), FakeVectorStore(chunks=[]),
                                 manifest_path=tmp_path / "m.json")
    report = pipeline.ingest(["missing-paper", "2410.14077v2"])
    assert report.papers_failed == 1 and report.papers_ingested == 1
    assert "missing-paper" in report.failures
