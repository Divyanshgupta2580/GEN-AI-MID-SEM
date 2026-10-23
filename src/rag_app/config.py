import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()

# Project Root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

@dataclass
class AppConfig:
    """Central configuration for the Designs Manual RAG Assistant."""
    
    # Official PDF Acquisition & Verification
    pdf_url: str = "https://ipindia.gov.in/storage/uploads/docs-operator/1ebf65d0-3b78-499a-a9a8-ba28863962f2.pdf"
    pdf_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "raw" / "designs_manual.pdf")
    fallback_pdf_path: Path = field(default_factory=lambda: BASE_DIR / "1ebf65d0-3b78-499a-a9a8-ba28863962f2.pdf")
    expected_sha256: str = "2a126035cfed4951c61c22c96f43091a964cc562f287e77c393bf2b53fcf2852"
    expected_pages: int = 87
    download_timeout_seconds: int = 45

    # Vector Database & Embeddings (Local Persistent ChromaDB only - No cloud/remote credentials)
    chroma_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_PERSIST_DIRECTORY", os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
        )
    )
    collection_name: str = "designs_manual"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    manifest_filename: str = "ingestion_manifest.json"

    # Text Splitting / Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Information Retrieval
    top_k: int = 6
    # Quality gate: similarity threshold (in [0, 1] cosine similarity scale)
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.45"))

    # LLM Answer Generation
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))

    def resolve_pdf_path(self) -> Path:
        """Resolve the path to the PDF, checking custom configured path, default path, or root fallback."""
        env_pdf = os.getenv("PDF_PATH")
        if env_pdf:
            custom_path = Path(env_pdf)
            if not custom_path.is_absolute():
                custom_path = BASE_DIR / custom_path
            if custom_path.exists():
                return custom_path

        if self.pdf_path.exists():
            return self.pdf_path

        if self.fallback_pdf_path.exists():
            return self.fallback_pdf_path

        return self.pdf_path


# Global default configuration instance
config = AppConfig()
