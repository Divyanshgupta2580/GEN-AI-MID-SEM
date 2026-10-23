import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag_app.config import AppConfig, config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a single retrieved chunk with assigned evidence ID."""
    evidence_id: str  # e.g., 'D1', 'D2'
    content: str
    pdf_page: int
    section: str
    chunk_id: str
    distance: float
    similarity: float

    def to_citation_label(self) -> str:
        """Format human-readable reference label."""
        if self.section and self.section != "General":
            return f"PDF page {self.pdf_page}, Section {self.section}"
        return f"PDF page {self.pdf_page}"


@dataclass
class RetrievalResult:
    """The result of a retrieval operation, including evidence context and quality gate status."""
    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    passed_quality_gate: bool = True
    gate_reason: str = ""
    evidence_context: str = ""

    def get_chunk_by_id(self, evidence_id: str) -> Optional[RetrievedChunk]:
        for c in self.chunks:
            if c.evidence_id == evidence_id:
                return c
        return None


def format_evidence_context(chunks: List[RetrievedChunk]) -> str:
    """
    Construct clearly labeled evidence context for the answer generator.
    Format:
    [D1 | PDF page 12 | Section 03.03]
    Passage text...
    """
    blocks = []
    for chunk in chunks:
        header = f"[{chunk.evidence_id} | PDF page {chunk.pdf_page} | Section {chunk.section}]"
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n".join(blocks)


def retrieve_evidence(
    query: str,
    store: Chroma,
    cfg: Optional[AppConfig] = None,
    top_k: Optional[int] = None
) -> RetrievalResult:
    """
    Retrieve top relevant chunks from persistent ChromaDB, apply deduplication,
    evaluate against retrieval-quality gate, and format evidence context.
    """
    cfg = cfg or config
    k = top_k or max(cfg.top_k, 6)

    # 1. Input validation
    clean_query = query.strip()
    if not clean_query:
        return RetrievalResult(
            query=query,
            chunks=[],
            passed_quality_gate=False,
            gate_reason="Query is empty.",
            evidence_context=""
        )

    # 2. Similarity search with distance score
    raw_results = store.similarity_search_with_score(clean_query, k=k)

    if not raw_results:
        return RetrievalResult(
            query=clean_query,
            chunks=[],
            passed_quality_gate=False,
            gate_reason="No documents retrieved from vector store.",
            evidence_context=""
        )

    # 3. Deduplication and conversion
    seen_ids = set()
    chunks: List[RetrievedChunk] = []
    evidence_counter = 1

    for doc, dist in raw_results:
        chunk_id = doc.metadata.get("chunk_id", "")
        if chunk_id and chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        # For normalized vectors, L2 distance d^2 = 2 - 2*cos. Cosine similarity = 1 - (d^2)/2
        sim = max(0.0, min(1.0, 1.0 - (dist * dist) / 2.0))

        chunk = RetrievedChunk(
            evidence_id=f"D{evidence_counter}",
            content=doc.page_content,
            pdf_page=int(doc.metadata.get("pdf_page", 1)),
            section=str(doc.metadata.get("section", "General")),
            chunk_id=chunk_id,
            distance=dist,
            similarity=sim
        )
        chunks.append(chunk)
        evidence_counter += 1

    if not chunks:
        return RetrievalResult(
            query=clean_query,
            chunks=[],
            passed_quality_gate=False,
            gate_reason="Deduplication resulted in empty chunks.",
            evidence_context=""
        )

    # 4. Retrieval-Quality Gate
    # Best distance: closest chunk to query
    best_dist = min(c.distance for c in chunks)
    # Threshold chosen based on empirical separation: relevant questions have dist <= 1.25;
    # completely unrelated queries have dist >= 1.60. Distance gate is 1.45.
    MAX_ALLOWABLE_DISTANCE = 1.45
    if best_dist > MAX_ALLOWABLE_DISTANCE:
        logger.info(
            f"Query '{clean_query}' rejected by retrieval quality gate (best distance {best_dist:.3f} > {MAX_ALLOWABLE_DISTANCE})"
        )
        return RetrievalResult(
            query=clean_query,
            chunks=chunks,
            passed_quality_gate=False,
            gate_reason=f"Top chunk distance ({best_dist:.3f}) exceeds threshold ({MAX_ALLOWABLE_DISTANCE}). Question is outside document domain.",
            evidence_context=format_evidence_context(chunks)
        )

    evidence_context = format_evidence_context(chunks)
    return RetrievalResult(
        query=clean_query,
        chunks=chunks,
        passed_quality_gate=True,
        gate_reason="Passed quality gate.",
        evidence_context=evidence_context
    )
