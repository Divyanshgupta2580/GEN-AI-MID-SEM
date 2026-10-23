import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag_app.pdf_loader import ExtractedPage

# Pattern 1: Multi-line numbered sections e.g. "02.02.04 \n Design"
MULTILINE_SECTION_REGEX = re.compile(
    r"^(\d{2}\.\d{2}(?:\.\d{2})?)\s*\n\s*([A-Za-z][A-Za-z0-9\s&,/-]{2,50})",
    re.MULTILINE
)

# Pattern 2: Multi-line chapter titles e.g. "CHAPTER-3 \n FILING OF DESIGN APPLICATION"
MULTILINE_CHAPTER_REGEX = re.compile(
    r"^(CHAPTER\s*[-–]\s*\d+)\s*\n\s*([A-Za-z][A-Za-z0-9\s&,/-]{2,50})",
    re.MULTILINE
)

# Pattern 3: Single-line headings e.g. "03.02 Where to apply" or "CHAPTER 2 - DEFINITIONS"
SINGLELINE_SECTION_REGEX = re.compile(
    r"^(?:"
    r"(\d{2}\.\d{2}(?:\.\d{2})?\s+[A-Za-z0-9\s&,/-]{2,50})|"
    r"(CHAPTER\s*[-–]\s*\d+\s*[-–:]?\s*[A-Za-z0-9\s&,/-]{2,50})|"
    r"([a-z]\.\s+[A-Za-z0-9\s&,/-]{2,50})"
    r")$",
    re.MULTILINE
)


def extract_section_title(text: str, default: str = "") -> str:
    """
    Extract the most specific section title or chapter identifier from text.
    Handles both multi-line and single-line heading structures.
    """
    if not text:
        return default

    # First check multi-line section (e.g., 02.02.04 \n Design)
    m_sec = MULTILINE_SECTION_REGEX.search(text)
    if m_sec:
        num = m_sec.group(1).strip()
        title = m_sec.group(2).strip()
        # Clean title of any trailing newline/numbers
        title = title.splitlines()[0].strip()
        return f"{num} {title}"

    # Check multi-line chapter
    m_chap = MULTILINE_CHAPTER_REGEX.search(text)
    if m_chap:
        chap = m_chap.group(1).strip()
        title = m_chap.group(2).strip().splitlines()[0].strip()
        return f"{chap}: {title}"

    # Check single-line headings
    matches = SINGLELINE_SECTION_REGEX.findall(text)
    if matches:
        for group in matches[0]:
            if group.strip():
                return group.strip().splitlines()[0].strip()

    return default


def chunk_extracted_pages(
    pages: List[ExtractedPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Chunk extracted pages using page-bounded splitting to maintain 1-to-1 fidelity
    between chunks and physical PDF pages.
    Preserves exact physical page numbers (1-indexed), section titles, and stable chunk IDs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        strip_whitespace=True
    )

    documents: List[Document] = []
    current_active_section = "General"

    for page in pages:
        if page.is_empty or not page.text.strip():
            continue

        # Check if page updates section
        page_section = extract_section_title(page.text, default="")
        if page_section:
            current_active_section = page_section

        page_chunks = splitter.split_text(page.text)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_specific_section = extract_section_title(chunk_text, default=current_active_section)
            chunk_id = f"p{page.pdf_page}_c{chunk_idx + 1}"

            # Calculate deterministic physical line range for this chunk
            start_line = 1
            end_line = 1
            if page.lines:
                chunk_start_char = page.text.find(chunk_text)
                chunk_end_char = chunk_start_char + len(chunk_text) if chunk_start_char != -1 else 0

                for pline in page.lines:
                    if chunk_start_char != -1 and pline.end_char >= chunk_start_char:
                        start_line = pline.line_number
                        break

                for pline in reversed(page.lines):
                    if chunk_end_char != -1 and pline.start_char <= chunk_end_char:
                        end_line = pline.line_number
                        break

            line_range = f"Lines {start_line}-{end_line}" if start_line != end_line else f"Line {start_line}"

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": page.source,
                    "pdf_page": page.pdf_page,
                    "start_line": start_line,
                    "end_line": end_line,
                    "line_range": line_range,
                    "section": chunk_specific_section,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx + 1,
                    "total_page_chunks": len(page_chunks)
                }
            )
            documents.append(doc)

    return documents
