import pytest
from src.rag_app.config import config
from src.rag_app.pdf_loader import acquire_pdf, extract_pages
from src.rag_app.text_processing import chunk_extracted_pages, extract_section_title
from src.rag_app.vector_store import ingest_documents, get_vector_store


@pytest.fixture(scope="module")
def extracted_pages():
    pdf_path = acquire_pdf(config)
    pages, _ = extract_pages(pdf_path)
    return pages


def test_chunking_generates_valid_documents(extracted_pages):
    """Verify that chunking divides text into sensible, non-empty chunks with metadata."""
    chunks = chunk_extracted_pages(extracted_pages, chunk_size=1000, chunk_overlap=150)
    assert len(chunks) > 0, "No chunks generated from pages"

    for chunk in chunks:
        # Validate non-empty content
        assert len(chunk.page_content.strip()) > 0
        # Validate metadata fields
        meta = chunk.metadata
        assert "pdf_page" in meta
        assert "source" in meta
        assert "section" in meta
        assert "chunk_id" in meta
        assert 1 <= meta["pdf_page"] <= 87
        assert meta["chunk_id"].startswith(f"p{meta['pdf_page']}_c")


def test_section_title_extraction():
    """Verify that multi-line and single-line manual sections are correctly identified."""
    sample_text_1 = "02.02.04\nDesign\n\n'design' means features of shape..."
    sec1 = extract_section_title(sample_text_1)
    assert sec1 == "02.02.04 Design"

    sample_text_2 = "CHAPTER-3\nFILING OF DESIGN APPLICATION\n\n03.01 General"
    sec2 = extract_section_title(sample_text_2)
    assert "CHAPTER-3" in sec2

    sample_text_3 = "03.02 Where to apply\nAn application shall be addressed..."
    sec3 = extract_section_title(sample_text_3)
    assert "03.02 Where to apply" in sec3


def test_idempotent_ingestion_avoids_duplicate_records(extracted_pages):
    """Verify that re-ingesting the same document reuses existing records without duplicates."""
    chunks = chunk_extracted_pages(extracted_pages, chunk_size=1000, chunk_overlap=150)
    store = get_vector_store(config)
    initial_count = store._collection.count()

    # Re-run ingestion without force
    from src.rag_app.pdf_loader import IngestionSummary
    summary = IngestionSummary(
        source_filename="designs_manual.pdf",
        physical_pages_detected=87,
        pages_processed=87,
        pages_with_text=87,
        pdf_sha256=config.expected_sha256
    )

    _, updated_summary = ingest_documents(chunks, summary, config, force=False)
    assert updated_summary.db_reused is True
    post_count = store._collection.count()
    assert initial_count == post_count, f"Duplicate records created: {initial_count} vs {post_count}"
