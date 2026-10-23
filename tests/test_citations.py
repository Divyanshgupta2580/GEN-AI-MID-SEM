import pytest
from src.rag_app.config import config
from src.rag_app.retriever import RetrievalResult, RetrievedChunk
from src.rag_app.rag_pipeline import build_citations
from src.rag_app.answer_generator import GroundedAnswerResponse, validate_grounded_response


@pytest.fixture
def sample_retrieval_result():
    chunks = [
        RetrievedChunk(
            evidence_id="D1",
            content="Section 03.03 Type of Applications: Ordinary application does not claim priority.",
            pdf_page=11,
            section="03.03 Type of Applications",
            chunk_id="p11_c1",
            distance=0.4,
            similarity=0.92
        ),
        RetrievedChunk(
            evidence_id="D2",
            content="Where to apply: An application shall be addressed to Controller of Designs, Kolkata 700091.",
            pdf_page=11,
            section="03.02 Where to apply",
            chunk_id="p11_c2",
            distance=0.45,
            similarity=0.898
        ),
        RetrievedChunk(
            evidence_id="D3",
            content="Section 02.02.04 Design means features of shape, configuration...",
            pdf_page=8,
            section="02.02.04 Design",
            chunk_id="p8_c2",
            distance=0.5,
            similarity=0.875
        )
    ]
    return RetrievalResult(
        query="test query",
        chunks=chunks,
        passed_quality_gate=True,
        evidence_context=""
    )


def test_build_citations_from_valid_evidence(sample_retrieval_result):
    """Verify that citations resolve properly from real chunk metadata."""
    citations = build_citations(["D1", "D3"], sample_retrieval_result, config)
    assert len(citations) == 2

    pages = [c.pdf_page for c in citations]
    assert 8 in pages
    assert 11 in pages

    c_p11 = next(c for c in citations if c.pdf_page == 11)
    assert c_p11.section == "03.03 Type of Applications"
    assert "PDF page 11" in c_p11.formatted
    assert "Section 03.03 Type of Applications" in c_p11.formatted
    assert "Extracted lines:" in c_p11.formatted


def test_reject_hallucinated_evidence_identifiers(sample_retrieval_result):
    """Verify that invented evidence IDs (e.g., D99, D4) are discarded."""
    citations = build_citations(["D1", "D99", "INVALID_ID"], sample_retrieval_result, config)
    assert len(citations) == 1
    assert citations[0].evidence_id == "D1"


def test_deduplicate_identical_page_and_section(sample_retrieval_result):
    """Verify that multiple chunks from the same page and section are deduplicated."""
    # Add a duplicate chunk for D1's page and section
    sample_retrieval_result.chunks.append(
        RetrievedChunk(
            evidence_id="D4",
            content="Duplicate section content",
            pdf_page=11,
            section="03.03 Type of Applications",
            chunk_id="p11_c3",
            distance=0.42,
            similarity=0.91
        )
    )
    citations = build_citations(["D1", "D4"], sample_retrieval_result, config)
    assert len(citations) == 1
    assert citations[0].pdf_page == 11


def test_validate_grounded_response_clears_citations_for_not_found():
    """Verify that when answer is 'Not found', evidence IDs are unconditionally cleared."""
    resp = GroundedAnswerResponse(
        answerable=False,
        answer="Not found",
        evidence_ids=["D1", "D2"]  # Incorrectly populated by LLM
    )
    validated = validate_grounded_response(resp, allowed_ids={"D1", "D2"})
    assert validated.answerable is False
    assert validated.answer == "Not found"
    assert len(validated.evidence_ids) == 0


def test_validate_grounded_response_rejects_unsupported_claim():
    """Verify that claiming answerable=True without valid evidence IDs fails closed to Not found."""
    resp = GroundedAnswerResponse(
        answerable=True,
        answer="Fabricated answer",
        evidence_ids=["D999"]  # Hallucinated ID
    )
    validated = validate_grounded_response(resp, allowed_ids={"D1", "D2"})
    assert validated.answerable is False
    assert validated.answer == "Not found"
    assert len(validated.evidence_ids) == 0
