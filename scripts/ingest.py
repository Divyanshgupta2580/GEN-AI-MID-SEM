import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_app.config import config, AppConfig
from src.rag_app.pdf_loader import acquire_pdf, extract_pages
from src.rag_app.text_processing import chunk_extracted_pages
from src.rag_app.vector_store import ingest_documents, is_index_initialized, load_manifest


def run_ingestion(force: bool = False, custom_pdf_path: str = None) -> int:
    """Execute the full PDF ingestion pipeline."""
    start_time = time.time()
    print("=" * 65)
    print("      DESIGNS MANUAL RAG ASSISTANT - DOCUMENT INGESTION")
    print("=" * 65)

    cfg = AppConfig()
    if custom_pdf_path:
        cfg.pdf_path = Path(custom_pdf_path)

    # 1. Acquire PDF
    print(f"[*] Acquiring source PDF...")
    try:
        pdf_path = acquire_pdf(cfg)
        print(f"[+] PDF located at: {pdf_path}")
    except Exception as e:
        print(f"[!] FAILED to acquire PDF: {e}", file=sys.stderr)
        return 1

    # 2. Extract Pages
    print(f"[*] Extracting pages using PyMuPDF...")
    try:
        pages, summary = extract_pages(pdf_path)
    except Exception as e:
        print(f"[!] FAILED during page extraction: {e}", file=sys.stderr)
        return 1

    # Validate physical page count
    if summary.physical_pages_detected != cfg.expected_pages:
        print(
            f"[!] WARNING: Detected {summary.physical_pages_detected} pages; expected {cfg.expected_pages} pages.",
            file=sys.stderr
        )
    else:
        print(f"[+] Verified {summary.physical_pages_detected} physical pages matching specification.")

    if summary.pages_with_text == 0:
        print("[!] ERROR: No extractable text found in PDF. Ingestion aborted.", file=sys.stderr)
        return 1

    # 3. Chunk Pages
    print(f"[*] Chunking document text (chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap})...")
    documents = chunk_extracted_pages(
        pages=pages,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap
    )
    print(f"[+] Generated {len(documents)} page-bounded chunks.")

    # 4. Embeddings & ChromaDB Storage
    print(f"[*] Initializing persistent ChromaDB store at {cfg.chroma_dir}...")
    try:
        store, summary = ingest_documents(
            documents=documents,
            summary=summary,
            cfg=cfg,
            force=force
        )
    except Exception as e:
        print(f"[!] FAILED to store vectors in ChromaDB: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - start_time

    # 5. Output Ingestion Summary
    print("\n" + "=" * 65)
    print("                    INGESTION SUMMARY")
    print("=" * 65)
    print(f"Source PDF filename:            {summary.source_filename}")
    print(f"Source SHA-256 Checksum:        {summary.pdf_sha256}")
    print(f"Physical pages detected:        {summary.physical_pages_detected}")
    print(f"Pages successfully processed:   {summary.pages_processed}")
    print(f"Pages containing text:          {summary.pages_with_text}")
    print(f"Empty or problematic pages:     {summary.empty_or_problematic_pages or 'None (0)'}")
    print(f"Total extracted characters:     {summary.total_extracted_chars:,}")
    print(f"Total chunks generated:         {summary.total_chunks_generated}")
    print(f"Embedding model used:           {summary.embedding_model}")
    print(f"Vector DB status:               {'Reused existing index' if summary.db_reused else 'Newly created'}")
    print(f"Execution time:                 {elapsed:.2f} seconds")
    print("=" * 65)
    print("[+] Ingestion pipeline completed successfully.\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Indian Designs Manual into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Force rebuild vector store from scratch.")
    parser.add_argument("--pdf-path", type=str, default=None, help="Custom path to designs manual PDF.")
    args = parser.parse_args()

    sys.exit(run_ingestion(force=args.force, custom_pdf_path=args.pdf_path))
