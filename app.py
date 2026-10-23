import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag_app.config import config, AppConfig
from src.rag_app.vector_store import is_index_initialized, load_manifest, get_vector_store
from src.rag_app.rag_pipeline import RAGPipeline, PipelineResult
from scripts.ingest import run_ingestion

# Streamlit Page Configuration
st.set_page_config(
    page_title="Designs Manual RAG Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom Styling for polished, responsive, professional UI (strictly no emojis)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 28px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .main-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.5;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .badge-green {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .badge-blue {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-amber {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .answer-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        padding: 20px;
        margin-top: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .answer-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #2563eb;
        margin-bottom: 8px;
    }
    .answer-content {
        font-size: 15px;
        line-height: 1.65;
        color: #0f172a;
    }
    
    .not-found-card {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #e11d48;
        border-radius: 6px;
        padding: 20px;
        margin-top: 16px;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .not-found-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #e11d48;
        margin-bottom: 6px;
    }
    .not-found-text {
        font-size: 17px;
        font-weight: 700;
        color: #9f1239;
    }
    
    .citation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 16px;
        margin-top: 12px;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .citation-badge {
        background: #e0e7ff;
        color: #3730a3;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 11px;
        margin-right: 6px;
    }
    
    /* Responsive media queries */
    @media (max-width: 768px) {
        .main-header {
            padding: 16px 18px;
        }
        .main-header h1 {
            font-size: 20px;
        }
        .main-header p {
            font-size: 13px;
        }
        .answer-card, .not-found-card, .citation-card {
            padding: 14px 16px;
        }
    }
</style>
""", unsafe_allow_html=True)


def get_pipeline(api_key: str = None) -> RAGPipeline:
    """Retrieve pipeline instance with optional dynamic API key override."""
    return RAGPipeline(config, api_key=api_key)


# Main App Layout
st.markdown("""
<div class="main-header">
    <h1>Designs Manual RAG Assistant</h1>
    <p>Document-grounded question answering for the <em>Manual of Designs Practice & Procedure</em> published by the Office of the Controller General of Patents, Designs & Trade Marks, Government of India.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: System Configuration & Status
with st.sidebar:
    st.header("System Status & Controls")

    # Document & Vector Index Status
    index_ready = is_index_initialized(config)
    manifest = load_manifest(config)

    if index_ready and manifest:
        st.markdown(f'<span class="status-badge badge-green">[READY] Local Vector DB Active</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-badge badge-blue">Document: {manifest.get("total_pages", 87)} Pages</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-badge badge-blue">Index: {manifest.get("total_chunks", 186)} Chunks</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge badge-amber">[NOT READY] Vector Index Missing</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Document Metadata")
    st.markdown("""
    - **Document:** Manual of Designs Practice & Procedure
    - **Authority:** Office of the CGPDTM, Govt of India
    - **Physical Pages:** 87 verified pages
    - **Vector Store:** Local Persistent ChromaDB (`./chroma_db`)
    - **Embedding Model:** `all-MiniLM-L6-v2` (Local Sentence Transformers)
    """)

    st.markdown("---")
    st.subheader("Model Configuration")
    user_api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Reads by default from GOOGLE_API_KEY in local .env"
    )
    
    current_model = config.gemini_model
    st.caption(f"Active Model: `{current_model}`")

    # Force Ingestion Control
    if st.button("Rebuild Vector Index", help="Re-runs local ingestion from the PDF"):
        with st.spinner("Re-indexing 87 pages into ChromaDB..."):
            ret_code = run_ingestion(force=True)
            if ret_code == 0:
                st.success("Index rebuilt successfully!")
                st.rerun()
            else:
                st.error("Failed to re-index document.")

    st.markdown("---")
    st.subheader("Example Questions")
    st.caption("Click to load tested demonstration queries:")

    example_questions = [
        "How long does a proprietor have copyright in a registered design, and can it be extended?",
        "Which classification system is used to register designs?",
        "What is the government fee to register a trademark?",
        "Who is the current Controller General of Patents?",
        "What is the fee to register a design?",
        "What was the weather in Delhi yesterday?"
    ]

    for eq in example_questions:
        label = f"Q: {eq[:42]}..."
        if st.button(label, key=f"btn_{eq}"):
            st.session_state["question_input"] = eq


# Primary Interaction Interface
if not index_ready:
    st.warning("The local ChromaDB vector store is not yet initialized. Please click below to index the PDF.")
    if st.button("Initialize Document Index Now", type="primary"):
        with st.spinner("Processing official PDF and generating local embeddings..."):
            code = run_ingestion(force=False)
            if code == 0:
                st.success("Vector database successfully initialized!")
                st.rerun()
            else:
                st.error("Document ingestion encountered an error. Check application logs.")
    st.stop()

# Query Input
default_q = st.session_state.get("question_input", "")
question = st.text_input(
    "Enter your question about the Designs Manual:",
    value=default_q,
    placeholder="e.g., How long does a proprietor have copyright in a registered design, and can it be extended?",
    key="query_text_input"
)

col1, col2 = st.columns([1, 6])
with col1:
    ask_button = st.button("Ask Question", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=False):
        st.session_state["question_input"] = ""
        st.rerun()

# Execution & Grounded Generation
if ask_button and question:
    pipeline = get_pipeline(api_key=user_api_key)

    with st.spinner("Searching local ChromaDB & generating document-grounded answer..."):
        result: PipelineResult = pipeline.run(question)

    if result.is_error:
        st.error(f"Service Error: {result.error_message}")
        if "GOOGLE_API_KEY" in result.error_message:
            st.info("Please set your Google Gemini API key in `.env` or in the sidebar above.")
    elif not result.is_answered:
        # Strict Not Found display requirement
        st.markdown("""
        <div class="not-found-card">
            <div class="not-found-title">Query Evaluation Result</div>
            <div class="not-found-text">Not found</div>
            <p style="margin: 8px 0 0 0; color: #881337; font-size: 13px;">
                The supplied Manual of Designs Practice & Procedure does not contain sufficient factual evidence to support an answer to this question.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Grounded Answer Display
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-title">Document Grounded Answer</div>
            <div class="answer-content">{result.answer}</div>
        </div>
        """, unsafe_allow_html=True)

        # Citations Display
        if result.citations:
            st.markdown("### Supporting PDF References")
            for c in result.citations:
                lines_display = f"{c.start_line}-{c.end_line}" if c.start_line != c.end_line else str(c.start_line)
                sec_display = c.section if c.section and c.section != "General" else "General"
                quotation_display = (c.supporting_passage or c.snippet).strip().replace("\n", " ")
                if len(quotation_display) > 300:
                    quotation_display = quotation_display[:297] + "..."

                st.markdown(f"""
                <div class="citation-card">
                    <div style="margin-bottom: 6px;">
                        <span class="citation-badge">{c.evidence_id}</span>
                        <strong>Source:</strong> Manual of Designs Practice & Procedure
                    </div>
                    <div style="font-size: 14px; color: #334155; line-height: 1.6; margin-left: 4px;">
                        <strong>PDF page:</strong> {c.pdf_page}<br>
                        <strong>Extracted lines:</strong> {lines_display}<br>
                        <strong>Section:</strong> {sec_display}<br>
                        <strong>Supporting passage:</strong> <em>"{quotation_display}"</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Inspection Expander for Transparency
        if result.retrieval_result and result.retrieval_result.chunks:
            with st.expander("Inspect Retrieved ChromaDB Chunks"):
                for c in result.retrieval_result.chunks:
                    st.markdown(f"**[{c.evidence_id}] PDF Page {c.pdf_page}** | *{c.section}* | Distance: `{c.distance:.4f}`")
                    st.text(c.content)
                    st.divider()

