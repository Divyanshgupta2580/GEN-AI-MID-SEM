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
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, professional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
    }
    .main-header p {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: 15px;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .badge-blue {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .answer-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .answer-title {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #3b82f6;
        margin-bottom: 8px;
    }
    .answer-content {
        font-size: 16px;
        line-height: 1.6;
        color: #1e293b;
    }
    
    .not-found-card {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 4px solid #e11d48;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .not-found-title {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        color: #e11d48;
        margin-bottom: 4px;
    }
    .not-found-text {
        font-size: 18px;
        font-weight: 600;
        color: #9f1239;
    }
    
    .citation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 12px;
    }
    .citation-item {
        margin-bottom: 8px;
        font-size: 14px;
        color: #334155;
    }
    .citation-badge {
        background: #e0e7ff;
        color: #3730a3;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
        margin-right: 8px;
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
        st.markdown(f'<span class="status-badge badge-green">✓ Local Vector DB Ready</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-badge badge-blue">📄 {manifest.get("total_pages", 87)} Pages</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-badge badge-blue">🧩 {manifest.get("total_chunks", 186)} Chunks</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge badge-amber">⚠️ Index Not Ready</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Document Metadata")
    st.markdown("""
    - **Document:** Manual of Designs Practice & Procedure
    - **Authority:** Office of the CGPDTM, Govt of India
    - **Total Pages:** 87 physical pages
    - **Vector Store:** Local Persistent ChromaDB (`./chroma_db`)
    - **Embeddings:** `all-MiniLM-L6-v2` (Local Sentence Transformers)
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
        "What does the manual mean by the term 'design'?",
        "What is an ordinary application?",
        "What is a reciprocity application, and what filing period does the manual describe?",
        "To whom and at what address should a design registration application be addressed according to the manual?",
        "Which four Patent Office locations are mentioned in the manual?",
        "What was the weather in Delhi yesterday?",
        "What is the price of a MacBook Air?"
    ]

    for eq in example_questions:
        label = f"Q: {eq[:40]}..."
        if st.button(label, key=f"btn_{eq}"):
            st.session_state["question_input"] = eq


# Primary Interaction Interface
if not index_ready:
    st.warning("⚠️ The local ChromaDB vector store is not yet initialized. Please click below to index the PDF.")
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
    placeholder="e.g., What is an ordinary application?",
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
        st.error(f"⚠️ Service Error: {result.error_message}")
        if "GOOGLE_API_KEY" in result.error_message:
            st.info("💡 Please set your Google Gemini API key in `.env` or in the sidebar above.")
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
            st.markdown("### Supporting References")
            for c in result.citations:
                link_text = f"📄 Physical PDF Page {c.pdf_page}"
                sec_text = f"Section: {c.section}" if c.section and c.section != "General" else ""
                st.markdown(f"""
                <div class="citation-card">
                    <span class="citation-badge">{c.evidence_id}</span>
                    <strong>{link_text}</strong> &nbsp;|&nbsp; <em>{sec_text}</em>
                    <div style="margin-top: 6px; font-size: 13px; color: #475569; font-style: italic;">
                        "{c.snippet}..."
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Inspection Expander for Transparency
        if result.retrieval_result and result.retrieval_result.chunks:
            with st.expander("🔍 Inspect Retrieved ChromaDB Chunks"):
                for c in result.retrieval_result.chunks:
                    st.markdown(f"**[{c.evidence_id}] PDF Page {c.pdf_page}** | *{c.section}* | Distance: `{c.distance:.4f}`")
                    st.text(c.content)
                    st.divider()
