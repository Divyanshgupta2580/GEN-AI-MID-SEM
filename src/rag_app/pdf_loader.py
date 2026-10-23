import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import requests
import pymupdf

from src.rag_app.config import AppConfig, config

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Represents a single extracted physical page from the PDF."""
    pdf_page: int  # 1-based physical page number
    text: str
    source: str
    char_count: int
    is_empty: bool = False


@dataclass
class IngestionSummary:
    """Detailed summary of the PDF ingestion and extraction stage."""
    source_filename: str
    physical_pages_detected: int
    pages_processed: int
    pages_with_text: int
    empty_or_problematic_pages: List[int] = field(default_factory=list)
    total_extracted_chars: int = 0
    total_chunks_generated: int = 0
    embedding_model: str = ""
    db_reused: bool = False
    pdf_sha256: str = ""

    def to_dict(self) -> dict:
        return {
            "source_filename": self.source_filename,
            "physical_pages_detected": self.physical_pages_detected,
            "pages_processed": self.pages_processed,
            "pages_with_text": self.pages_with_text,
            "empty_or_problematic_pages": self.empty_or_problematic_pages,
            "total_extracted_chars": self.total_extracted_chars,
            "total_chunks_generated": self.total_chunks_generated,
            "embedding_model": self.embedding_model,
            "db_reused": self.db_reused,
            "pdf_sha256": self.pdf_sha256,
        }


def calculate_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def verify_pdf_integrity(filepath: Path, expected_sha256: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate that the file exists, has PDF magic bytes (%PDF-),
    can be opened by PyMuPDF, and optionally matches expected SHA-256.
    """
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False, f"File does not exist or is empty: {filepath}"

    # Check magic bytes
    with open(filepath, "rb") as f:
        header = f.read(5)
        if header != b"%PDF-":
            return False, f"Invalid PDF header: found {header!r}, expected b'%PDF-'"

    actual_hash = calculate_sha256(filepath)
    if expected_sha256 and actual_hash != expected_sha256:
        logger.warning(
            f"Checksum divergence for {filepath.name}. Expected: {expected_sha256}, Actual: {actual_hash}"
        )

    # Validate with pymupdf
    try:
        doc = pymupdf.open(str(filepath))
        page_count = len(doc)
        doc.close()
        if page_count == 0:
            return False, f"PDF contains 0 pages: {filepath}"
    except Exception as e:
        return False, f"Failed to open PDF with PyMuPDF: {e}"

    return True, actual_hash


def acquire_pdf(cfg: Optional[AppConfig] = None) -> Path:
    """
    Acquire the Designs Manual PDF reliably.
    Follows priority:
    1. Check if configured target file exists and is valid.
    2. Check if local fallback copy exists (e.g. root workspace file) and copy it.
    3. If neither is available, download from the official URL with timeout and integrity checks.
    """
    cfg = cfg or config
    target_path = cfg.pdf_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Check existing target
    if target_path.exists():
        valid, msg = verify_pdf_integrity(target_path)
        if valid:
            logger.info(f"Using verified existing local PDF at {target_path} (SHA-256: {msg})")
            return target_path
        else:
            logger.warning(f"Existing target PDF invalid: {msg}. Attempting fallback...")

    # 2. Check fallback local path
    fallback = cfg.fallback_pdf_path
    if fallback.exists():
        valid, msg = verify_pdf_integrity(fallback)
        if valid:
            logger.info(f"Copying verified fallback PDF from {fallback} to {target_path}")
            shutil.copy2(fallback, target_path)
            return target_path

    # 3. Download from official URL
    logger.info(f"Downloading official PDF from {cfg.pdf_url}...")
    temp_target = target_path.with_suffix(".tmp")
    try:
        response = requests.get(
            cfg.pdf_url,
            timeout=cfg.download_timeout_seconds,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DesignsManualRAG/1.0"}
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download PDF. HTTP status {response.status_code} from {cfg.pdf_url}"
            )

        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise RuntimeError(
                f"Official URL returned HTML error page instead of PDF (Content-Type: {content_type})"
            )

        with open(temp_target, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Verify downloaded temp file
        valid, msg = verify_pdf_integrity(temp_target)
        if not valid:
            if temp_target.exists():
                temp_target.unlink()
            raise RuntimeError(f"Downloaded file failed validation: {msg}")

        # Atomically replace target
        temp_target.replace(target_path)
        logger.info(f"PDF acquired successfully and saved to {target_path}")
        return target_path

    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        raise RuntimeError(
            f"PDF acquisition error: {e}. Provide a valid PDF manually at {target_path}"
        ) from e


def normalize_page_text(raw_text: str) -> str:
    """
    Normalize extracted page text:
    - Replace non-breaking spaces and unusual unicode quotes/dashes
    - Normalize repeated spaces without destroying section numbers or line breaks
    - Preserve meaningful headings and paragraph structure
    """
    if not raw_text:
        return ""

    # Replace special unicode spaces and typographical quotes
    text = raw_text.replace("\u00a0", " ")
    text = text.replace("―", "-").replace("―", "-").replace("―", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Split into lines and strip trailing whitespace
    lines = [line.strip() for line in text.splitlines()]

    # Reassemble paragraphs while preserving line breaks for distinct items/sections
    cleaned_lines = []
    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def extract_pages(pdf_path: Path) -> Tuple[List[ExtractedPage], IngestionSummary]:
    """
    Extract text page-by-page from the PDF using PyMuPDF.
    Ensures 1-based physical page indexing and identifies any empty/problematic pages.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    valid, sha256_or_err = verify_pdf_integrity(pdf_path)
    if not valid:
        raise ValueError(f"Cannot process invalid PDF: {sha256_or_err}")

    doc = pymupdf.open(str(pdf_path))
    total_physical_pages = len(doc)
    source_name = pdf_path.name

    extracted_pages: List[ExtractedPage] = []
    empty_pages: List[int] = []
    total_chars = 0

    for idx in range(total_physical_pages):
        page_num = idx + 1  # Physical 1-based page index
        page = doc[idx]
        raw_text = page.get_text()
        clean_text = normalize_page_text(raw_text)
        char_count = len(clean_text)
        is_empty = (char_count == 0)

        if is_empty:
            empty_pages.append(page_num)
            logger.warning(f"Page {page_num} yielded no extractable text. Flagged as empty.")
        else:
            total_chars += char_count

        extracted_pages.append(
            ExtractedPage(
                pdf_page=page_num,
                text=clean_text,
                source=source_name,
                char_count=char_count,
                is_empty=is_empty
            )
        )

    doc.close()

    summary = IngestionSummary(
        source_filename=source_name,
        physical_pages_detected=total_physical_pages,
        pages_processed=len(extracted_pages),
        pages_with_text=len(extracted_pages) - len(empty_pages),
        empty_or_problematic_pages=empty_pages,
        total_extracted_chars=total_chars,
        pdf_sha256=sha256_or_err
    )

    return extracted_pages, summary
