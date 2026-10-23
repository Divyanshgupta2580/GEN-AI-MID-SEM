import pytest
from pathlib import Path
import tempfile

from src.rag_app.config import config
from src.rag_app.pdf_loader import (
    acquire_pdf,
    extract_pages,
    verify_pdf_integrity,
    normalize_page_text,
    calculate_sha256,
)


def test_valid_pdf_loading():
    """Verify that the PDF loader processes the official PDF and detects 87 physical pages."""
    pdf_path = acquire_pdf(config)
    assert pdf_path.exists(), f"PDF does not exist at {pdf_path}"

    pages, summary = extract_pages(pdf_path)
    assert len(pages) == 87, f"Expected 87 pages, got {len(pages)}"
    assert summary.physical_pages_detected == 87
    assert summary.pages_processed == 87
    assert summary.pages_with_text == 87
    assert len(summary.empty_or_problematic_pages) == 0


def test_checksum_verification():
    """Verify that the official document matches the expected SHA-256 hash."""
    pdf_path = acquire_pdf(config)
    sha256 = calculate_sha256(pdf_path)
    assert sha256 == config.expected_sha256, f"Checksum mismatch: expected {config.expected_sha256}, got {sha256}"


def test_page_identity_and_content_preservation():
    """Verify 1-based page indexing and ensure content is not assigned to adjacent pages."""
    pdf_path = acquire_pdf(config)
    pages, _ = extract_pages(pdf_path)

    # Page 1 must have physical index 1
    assert pages[0].pdf_page == 1
    assert "Manual Of" in pages[0].text or "Designs" in pages[0].text

    # Page 8 must contain statutory definition of Design
    page_8 = pages[7]
    assert page_8.pdf_page == 8
    assert "02.02.04" in page_8.text
    assert "Design" in page_8.text

    # Page 9 must contain Patent Office locations
    page_9 = pages[8]
    assert page_9.pdf_page == 9
    assert "Patent Office" in page_9.text
    assert "Kolkata" in page_9.text

    # Page 11 must contain Where to apply and Section 03.02
    page_11 = pages[10]
    assert page_11.pdf_page == 11
    assert "03.02" in page_11.text
    assert "Where to apply" in page_11.text

    # Page 87 (final page) must have index 87
    page_87 = pages[86]
    assert page_87.pdf_page == 87
    assert page_87.char_count > 0


def test_invalid_pdf_handling():
    """Verify that invalid or corrupted PDF files produce clear failures."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"NOT_A_VALID_PDF_HEADER_DATA")
        tmp_path = Path(tmp.name)

    try:
        valid, msg = verify_pdf_integrity(tmp_path)
        assert valid is False
        assert "Invalid PDF header" in msg

        with pytest.raises(ValueError):
            extract_pages(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_text_normalization():
    """Verify that excessive whitespace and unicode characters are properly normalized."""
    raw = "Section \u00a0 02.01\n\n\n\nGeneral   \n\n―design‖  means   ‘pattern’"
    normalized = normalize_page_text(raw)
    assert "\u00a0" not in normalized
    assert "―" not in normalized
    assert "‘" not in normalized
    assert "General" in normalized
    assert "Section" in normalized
