import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_app.config import AppConfig, config
from src.rag_app.pdf_loader import IngestionSummary

logger = logging.getLogger(__name__)

# Cache embedding function in memory across calls
_EMBEDDING_CACHE = {}


def get_embedding_model(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Load or retrieve the cached HuggingFace sentence transformer embedding model."""
    name = model_name or config.embedding_model
    if name not in _EMBEDDING_CACHE:
        logger.info(f"Loading local embedding model: {name}")
        _EMBEDDING_CACHE[name] = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDING_CACHE[name]


def get_manifest_path(cfg: Optional[AppConfig] = None) -> Path:
    """Get path to the ingestion manifest file."""
    cfg = cfg or config
    return cfg.chroma_dir / cfg.manifest_filename


def load_manifest(cfg: Optional[AppConfig] = None) -> Optional[dict]:
    """Load the ingestion manifest if it exists."""
    path = get_manifest_path(cfg)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing manifest at {path}: {e}")
    return None


def write_manifest(
    cfg: AppConfig,
    pdf_sha256: str,
    total_chunks: int,
    total_pages: int
) -> dict:
    """Persist ingestion manifest recording the document fingerprint and configuration."""
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pdf_sha256": pdf_sha256,
        "embedding_model": cfg.embedding_model,
        "collection_name": cfg.collection_name,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "total_chunks": total_chunks,
        "total_pages": total_pages,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    manifest_path = get_manifest_path(cfg)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote ingestion manifest to {manifest_path}")
    return manifest


def is_index_initialized(cfg: Optional[AppConfig] = None) -> bool:
    """
    Check if persistent ChromaDB index is already initialized and valid.
    Verifies that the directory exists, manifest exists, and vectors can be loaded.
    """
    cfg = cfg or config
    manifest = load_manifest(cfg)
    if not manifest:
        return False

    # Check directory
    if not cfg.chroma_dir.exists():
        return False

    try:
        store = get_vector_store(cfg)
        collection = store._collection
        count = collection.count()
        return count > 0 and count == manifest.get("total_chunks", count)
    except Exception as e:
        logger.warning(f"Index validation check failed: {e}")
        return False


def get_vector_store(cfg: Optional[AppConfig] = None) -> Chroma:
    """Get persistent Chroma vector store instance."""
    cfg = cfg or config
    embeddings = get_embedding_model(cfg.embedding_model)
    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
        persist_directory=str(cfg.chroma_dir)
    )


def ingest_documents(
    documents: List[Document],
    summary: IngestionSummary,
    cfg: Optional[AppConfig] = None,
    force: bool = False
) -> Tuple[Chroma, IngestionSummary]:
    """
    Idempotent vector ingestion into ChromaDB.
    Reuses existing store if fingerprint matches unless force=True.
    """
    cfg = cfg or config
    manifest = load_manifest(cfg)

    # Check for identical fingerprint
    can_reuse = (
        not force
        and manifest is not None
        and manifest.get("pdf_sha256") == summary.pdf_sha256
        and manifest.get("embedding_model") == cfg.embedding_model
        and manifest.get("chunk_size") == cfg.chunk_size
        and manifest.get("chunk_overlap") == cfg.chunk_overlap
        and is_index_initialized(cfg)
    )

    if can_reuse:
        logger.info("ChromaDB index matches source fingerprint and configuration. Reusing existing store.")
        summary.db_reused = True
        summary.total_chunks_generated = manifest.get("total_chunks", len(documents))
        summary.embedding_model = cfg.embedding_model
        store = get_vector_store(cfg)
        return store, summary

    logger.info(f"Building persistent ChromaDB vector store in {cfg.chroma_dir} (force={force})...")

    # If force re-ingestion, remove existing directory safely
    if force and cfg.chroma_dir.exists():
        logger.info(f"Purging existing index directory {cfg.chroma_dir} for fresh ingestion.")
        shutil.rmtree(cfg.chroma_dir)

    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_model(cfg.embedding_model)

    # Use stable unique chunk_id as Chroma document IDs to avoid duplicates
    ids = [doc.metadata["chunk_id"] for doc in documents]

    # Ingest into Chroma
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(cfg.chroma_dir),
        collection_name=cfg.collection_name,
        ids=ids
    )

    summary.db_reused = False
    summary.total_chunks_generated = len(documents)
    summary.embedding_model = cfg.embedding_model

    # Save manifest
    write_manifest(
        cfg=cfg,
        pdf_sha256=summary.pdf_sha256,
        total_chunks=len(documents),
        total_pages=summary.physical_pages_detected
    )

    logger.info(f"Successfully ingested {len(documents)} chunks into ChromaDB.")
    return store, summary
