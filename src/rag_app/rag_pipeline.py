import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.rag_app.config import AppConfig, config
from src.rag_app.vector_store import get_vector_store, is_index_initialized
from src.rag_app.retriever import retrieve_evidence, RetrievalResult, RetrievedChunk
from src.rag_app.answer_generator import AnswerGenerator, GroundedAnswerResponse

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A verified citation programmatically resolved from real chunk metadata."""
    pdf_page: int
    section: str
    evidence_id: str
    snippet: str
    pdf_link: str
    start_line: int = 1
    end_line: int = 1
    line_range: str = "Line 1"
    supporting_passage: str = ""

    @property
    def formatted(self) -> str:
        lines_str = f"{self.start_line}-{self.end_line}" if self.start_line != self.end_line else str(self.start_line)
        sec_str = f", Section {self.section}" if self.section and self.section != "General" else ""
        return f"PDF page {self.pdf_page}, Extracted lines: {lines_str}{sec_str}"

    def detailed_display(self) -> str:
        lines_str = f"{self.start_line}-{self.end_line}" if self.start_line != self.end_line else str(self.start_line)
        sec = self.section if self.section and self.section != "General" else "General"
        quotation = (self.supporting_passage or self.snippet).strip().replace("\n", " ")
        if len(quotation) > 280:
            quotation = quotation[:277] + "..."
        return (
            f"Source: Manual of Designs Practice & Procedure\n"
            f"PDF page: {self.pdf_page}\n"
            f"Extracted lines: {lines_str}\n"
            f"Section: {sec}\n"
            f'Supporting passage: "{quotation}"'
        )


@dataclass
class PipelineResult:
    """Complete output of the RAG pipeline."""
    query: str
    answer: str
    is_answered: bool
    citations: List[Citation] = field(default_factory=list)
    references_text: str = ""
    retrieval_result: Optional[RetrievalResult] = None
    is_error: bool = False
    error_message: str = ""

    def formatted_display(self) -> str:
        """Format the exact required user-facing response."""
        if not self.is_answered:
            return "Not found"

        output = f"Answer:\n{self.answer}"
        if self.citations:
            detailed_refs = "\n\n".join([c.detailed_display() for c in self.citations])
            output += f"\n\nSupporting PDF References:\n{detailed_refs}"
        elif self.references_text:
            output += f"\n\nReference:\n{self.references_text}"
        return output


def build_citations(
    evidence_ids: List[str],
    retrieval_result: RetrievalResult,
    cfg: AppConfig
) -> List[Citation]:
    """
    Programmatically construct verified citations from real chunk metadata.
    Deduplicates page and section references, strictly rejecting fabricated IDs.
    """
    citations: List[Citation] = []
    seen_page_sections = set()

    for eid in evidence_ids:
        chunk = retrieval_result.get_chunk_by_id(eid)
        if not chunk:
            logger.warning(f"Rejected unknown evidence identifier {eid}")
            continue

        key = (chunk.pdf_page, chunk.start_line, chunk.end_line, chunk.section)
        if key in seen_page_sections:
            continue
        seen_page_sections.add(key)

        pdf_rel_path = cfg.pdf_path.name
        pdf_link = f"data/raw/{pdf_rel_path}#page={chunk.pdf_page}"

        passage = chunk.content.strip()

        citations.append(
            Citation(
                pdf_page=chunk.pdf_page,
                section=chunk.section,
                evidence_id=chunk.evidence_id,
                snippet=chunk.content[:200],
                pdf_link=pdf_link,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                line_range=chunk.line_range,
                supporting_passage=passage
            )
        )

    # Sort citations in ascending physical page order, then start_line
    citations.sort(key=lambda c: (c.pdf_page, c.start_line))
    return citations


class RAGPipeline:
    """The central RAG execution pipeline."""

    def __init__(self, cfg: Optional[AppConfig] = None, api_key: Optional[str] = None):
        self.cfg = cfg or config
        self.generator = AnswerGenerator(self.cfg, api_key=api_key)
        self._store = None

    @property
    def store(self):
        if self._store is None:
            self._store = get_vector_store(self.cfg)
        return self._store

    def run(self, query: str, mock_llm=None) -> PipelineResult:
        """
        Execute the full RAG workflow:
        1. Validate index readiness.
        2. Validate question input.
        3. Retrieve passages and evaluate retrieval quality gate.
        4. Generate strictly grounded answer.
        5. Validate evidence IDs and programmatically construct citations.
        6. Return strictly grounded result or exactly 'Not found'.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return PipelineResult(
                query=query,
                answer="Not found",
                is_answered=False,
                error_message="Question is empty."
            )

        # 1. Index availability check
        if not is_index_initialized(self.cfg):
            return PipelineResult(
                query=clean_query,
                answer="",
                is_answered=False,
                is_error=True,
                error_message="Vector index is not initialized. Please run ingestion first."
            )

        # 2. Retrieve evidence
        retrieval_result = retrieve_evidence(clean_query, self.store, self.cfg)

        # 3. Quality gate check
        if not retrieval_result.passed_quality_gate or not retrieval_result.chunks:
            logger.info(f"Query '{clean_query}' did not pass retrieval quality gate: {retrieval_result.gate_reason}")
            return PipelineResult(
                query=clean_query,
                answer="Not found",
                is_answered=False,
                citations=[],
                references_text="",
                retrieval_result=retrieval_result
            )

        # 4. Generate document-grounded answer
        try:
            gen_resp: GroundedAnswerResponse = self.generator.generate_answer(
                retrieval_result=retrieval_result,
                mock_llm=mock_llm
            )
        except Exception as e:
            logger.error(f"Operational error in LLM generation: {e}")
            return PipelineResult(
                query=clean_query,
                answer="",
                is_answered=False,
                is_error=True,
                error_message=f"Answer generation error: {e}",
                retrieval_result=retrieval_result
            )

        # 5. Handle Not found result
        if not gen_resp.answerable or gen_resp.answer.strip() == "Not found":
            return PipelineResult(
                query=clean_query,
                answer="Not found",
                is_answered=False,
                citations=[],
                references_text="",
                retrieval_result=retrieval_result
            )

        # 6. Build citations programmatically from real metadata
        citations = build_citations(gen_resp.evidence_ids, retrieval_result, self.cfg)

        # If evidence resolution produces no valid citations, fail closed
        if not citations:
            logger.warning("No valid citations resolved from evidence IDs. Failing closed to Not found.")
            return PipelineResult(
                query=clean_query,
                answer="Not found",
                is_answered=False,
                citations=[],
                references_text="",
                retrieval_result=retrieval_result
            )

        references_text = "\n".join([c.formatted for c in citations])

        return PipelineResult(
            query=clean_query,
            answer=gen_resp.answer,
            is_answered=True,
            citations=citations,
            references_text=references_text,
            retrieval_result=retrieval_result
        )
