import pytest
from src.rag_app.config import config
from src.rag_app.vector_store import get_vector_store
from src.rag_app.retriever import retrieve_evidence, format_evidence_context, RetrievedChunk


@pytest.fixture(scope="module")
def vector_store():
    return get_vector_store(config)


def test_empty_query_retrieval(vector_store):
    """Verify that empty or whitespace-only queries are safely rejected."""
    result = retrieve_evidence("", vector_store, config)
    assert result.passed_quality_gate is False
    assert len(result.chunks) == 0
    assert "empty" in result.gate_reason.lower()


def test_relevant_query_retrieval_contains_metadata(vector_store):
    """Verify that a relevant question retrieves chunks containing accurate page metadata."""
    query = "What does the manual mean by the term design?"
    result = retrieve_evidence(query, vector_store, config, top_k=6)

    assert result.passed_quality_gate is True
    assert len(result.chunks) > 0

    pages = [c.pdf_page for c in result.chunks]
    assert 8 in pages or 25 in pages, f"Expected Page 8 or 25 in retrieved pages {pages}"

    # Verify metadata fields on chunks
    for chunk in result.chunks:
        assert chunk.evidence_id.startswith("D")
        assert 1 <= chunk.pdf_page <= 87
        assert chunk.chunk_id != ""
        assert chunk.distance >= 0.0


def test_evidence_context_formatting():
    """Verify that evidence context is formatted with exact required citation labels."""
    mock_chunks = [
        RetrievedChunk(
            evidence_id="D1",
            content="An ordinary application does not claim priority.",
            pdf_page=11,
            section="03.03 Type of Applications",
            chunk_id="p11_c1",
            distance=0.4,
            similarity=0.92
        ),
        RetrievedChunk(
            evidence_id="D2",
            content="In India, the Patent Office is located at four locations viz. Delhi, Mumbai, Kolkata & Chennai.",
            pdf_page=9,
            section="02.02.07 Patent Office",
            chunk_id="p9_c1",
            distance=0.5,
            similarity=0.875
        )
    ]

    context = format_evidence_context(mock_chunks)
    assert "[D1 | PDF page 11 | Line 1 | Section 03.03 Type of Applications]" in context
    assert "[D2 | PDF page 9 | Line 1 | Section 02.02.07 Patent Office]" in context
    assert "does not claim priority" in context


def test_quality_gate_rejects_unrelated_query(vector_store):
    """Verify that queries far outside the document domain are rejected by the distance threshold."""
    irrelevant_query = "What was the weather in Delhi yesterday?"
    result = retrieve_evidence(irrelevant_query, vector_store, config)

    # Retrieval distance threshold (1.45) must flag the irrelevant query
    assert result.passed_quality_gate is False
    assert "exceeds threshold" in result.gate_reason
