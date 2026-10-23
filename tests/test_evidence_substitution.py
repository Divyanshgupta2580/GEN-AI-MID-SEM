"""
TARGETED VERIFICATION: Evidence Substitution and Empty Retrieval

These tests prove that the answer generator strictly responds to whatever
evidence is provided to it — not to question text alone. They replace real
retrieved chunks with controlled contradictory passages and verify behavior.

No production index is modified. These are isolated unit tests.
"""
import pytest
from unittest.mock import patch

from src.rag_app.retriever import RetrievedChunk, RetrievalResult, format_evidence_context
from src.rag_app.answer_generator import AnswerGenerator, GroundedAnswerResponse, validate_grounded_response
from src.rag_app.rag_pipeline import RAGPipeline
from src.rag_app.config import config


# ---------------------------------------------------------------------------
# 1. validate_grounded_response: fabricated evidence IDs are always rejected
# ---------------------------------------------------------------------------

def test_fabricated_evidence_id_rejected():
    """Generator cannot use evidence IDs not supplied in the retrieval set."""
    allowed = {"D1", "D2"}
    fabricated = GroundedAnswerResponse(
        answerable=True,
        answer="Some invented answer.",
        evidence_ids=["D99", "D100"]
    )
    result = validate_grounded_response(fabricated, allowed)
    assert result.answerable is False
    assert result.answer == "Not found"
    assert result.evidence_ids == []


def test_partially_fabricated_evidence_ids_filtered():
    """Only the valid subset of evidence IDs survives validation."""
    allowed = {"D1", "D3"}
    mixed = GroundedAnswerResponse(
        answerable=True,
        answer="A valid answer.",
        evidence_ids=["D1", "D99"]
    )
    result = validate_grounded_response(mixed, allowed)
    assert result.answerable is True
    assert result.evidence_ids == ["D1"]
    assert "D99" not in result.evidence_ids


# ---------------------------------------------------------------------------
# 2. Mock LLM evidence substitution: contradictory passage -> contradictory answer
# ---------------------------------------------------------------------------

def make_retrieval_result_with_custom_chunks(chunks):
    """Helper: build a RetrievalResult from controlled chunks."""
    context = format_evidence_context(chunks)
    return RetrievalResult(
        query="test query",
        chunks=chunks,
        passed_quality_gate=True,
        gate_reason="Passed quality gate.",
        evidence_context=context
    )


def test_contradictory_evidence_yields_contradictory_answer():
    """
    Substitution test: replace real evidence with a contradictory passage asserting
    copyright lasts 25 years. Verify the mock LLM responds with this fake duration
    (because the mock reads from evidence_context), NOT the true 10 years.

    This proves the pipeline is NOT hardcoded — it processes whatever evidence
    is provided.
    """
    fake_chunk = RetrievedChunk(
        evidence_id="D1",
        content="The proprietor shall have copyright in the design for twenty-five years from the date of registration.",
        pdf_page=99,
        section="FAKE_SECTION",
        chunk_id="fake_chunk_1",
        distance=0.3,
        similarity=0.95,
        start_line=1,
        end_line=3,
        line_range="Lines 1-3"
    )
    retrieval_result = make_retrieval_result_with_custom_chunks([fake_chunk])

    def mock_llm_reads_evidence(question: str, evidence_context: str) -> GroundedAnswerResponse:
        """Mock that extracts its answer strictly from the supplied evidence."""
        if "twenty-five years" in evidence_context:
            return GroundedAnswerResponse(
                answerable=True,
                answer="According to the supplied evidence, copyright lasts twenty-five years.",
                evidence_ids=["D1"]
            )
        return GroundedAnswerResponse(answerable=False, answer="Not found", evidence_ids=[])

    generator = AnswerGenerator(cfg=config)
    result = generator.generate_answer(retrieval_result, mock_llm=mock_llm_reads_evidence)

    assert result.answerable is True
    assert "twenty-five" in result.answer.lower()
    assert "D1" in result.evidence_ids


def test_evidence_not_containing_keyword_returns_not_found():
    """
    Substitution test: evidence does NOT contain expected keywords.
    The mock LLM, reading only from evidence_context, returns Not found.
    """
    irrelevant_chunk = RetrievedChunk(
        evidence_id="D1",
        content="The fee schedule for trademark registration is outlined in the First Schedule.",
        pdf_page=50,
        section="UNRELATED",
        chunk_id="unrelated_chunk_1",
        distance=0.9,
        similarity=0.6,
        start_line=1,
        end_line=2,
        line_range="Lines 1-2"
    )
    retrieval_result = make_retrieval_result_with_custom_chunks([irrelevant_chunk])

    def strict_evidence_mock(question: str, evidence_context: str) -> GroundedAnswerResponse:
        if "copyright" in evidence_context and "ten years" in evidence_context:
            return GroundedAnswerResponse(
                answerable=True,
                answer="Ten years.",
                evidence_ids=["D1"]
            )
        return GroundedAnswerResponse(answerable=False, answer="Not found", evidence_ids=[])

    generator = AnswerGenerator(cfg=config)
    result = generator.generate_answer(retrieval_result, mock_llm=strict_evidence_mock)

    assert result.answerable is False
    assert result.answer == "Not found"


# ---------------------------------------------------------------------------
# 3. Empty retrieval -> exact "Not found" output
# ---------------------------------------------------------------------------

def test_empty_retrieval_returns_exact_not_found():
    """
    When retrieval produces zero chunks (failed quality gate),
    the pipeline must return exactly 'Not found'.
    """
    empty_retrieval = RetrievalResult(
        query="What is the weather in Delhi?",
        chunks=[],
        passed_quality_gate=False,
        gate_reason="Best distance (1.72) exceeds threshold (1.45).",
        evidence_context=""
    )
    generator = AnswerGenerator(cfg=config)
    result = generator.generate_answer(empty_retrieval)

    assert result.answerable is False
    assert result.answer == "Not found"
    assert result.evidence_ids == []


def test_pipeline_empty_retrieval_end_to_end():
    """
    End-to-end: patch retrieve_evidence to return empty result.
    PipelineResult must have answer='Not found', is_answered=False, citations=[].
    Production index is not modified.
    """
    empty = RetrievalResult(
        query="irrelevant query",
        chunks=[],
        passed_quality_gate=False,
        gate_reason="Distance exceeds threshold.",
        evidence_context=""
    )

    pipeline = RAGPipeline(config)
    with patch("src.rag_app.rag_pipeline.retrieve_evidence", return_value=empty):
        result = pipeline.run("irrelevant query")

    assert result.is_answered is False
    assert result.answer == "Not found"
    assert result.citations == []


def test_pipeline_gate_passed_llm_returns_not_found():
    """
    Chunks retrieved (quality gate passes), but mock LLM cannot find answer in evidence.
    Pipeline must still return exactly 'Not found'.
    """
    chunk = RetrievedChunk(
        evidence_id="D1",
        content="The fee for a trademark registration is not covered in this manual.",
        pdf_page=5,
        section="General",
        chunk_id="chunk_5_1",
        distance=0.6,
        similarity=0.82,
        start_line=1,
        end_line=2,
        line_range="Lines 1-2"
    )
    retrieval = RetrievalResult(
        query="What is the trademark fee?",
        chunks=[chunk],
        passed_quality_gate=True,
        gate_reason="Passed quality gate.",
        evidence_context=format_evidence_context([chunk])
    )

    def always_not_found(question: str, evidence_context: str) -> GroundedAnswerResponse:
        return GroundedAnswerResponse(answerable=False, answer="Not found", evidence_ids=[])

    pipeline = RAGPipeline(config)
    with patch("src.rag_app.rag_pipeline.retrieve_evidence", return_value=retrieval):
        result = pipeline.run("What is the trademark fee?", mock_llm=always_not_found)

    assert result.is_answered is False
    assert result.answer == "Not found"
    assert result.citations == []
