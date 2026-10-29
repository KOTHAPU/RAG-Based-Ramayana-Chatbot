"""
app.py — Ramayana AI Chatbot — Streamlit Frontend.

A professional, Ramayana-themed RAG chatbot with:
- Chat interface with message history
- Source citations panel
- Sidebar controls (model, top_k, threshold)
- Index status and ingestion trigger
- Full error handling for all failure modes
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for src imports
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# Sync Streamlit Community Cloud secrets into os.environ
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)) and k not in os.environ:
                os.environ[k] = str(v)
except Exception:
    pass

from src.config import settings
from src.utils import setup_logging

setup_logging(logging.WARNING)
logger = logging.getLogger("app")

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ramayana AI — Ask the Epic",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/",
        "About": "Ramayana RAG Chatbot — answers grounded in indexed Ramayana texts.",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --saffron:       #FF9933;
    --saffron-light: #FFBB66;
    --saffron-dark:  #CC6600;
    --maroon:        #800020;
    --maroon-light:  #A0253F;
    --gold:          #D4AF37;
    --gold-light:    #F0D060;
    --cream:         #FFF8EE;
    --cream-dark:    #F5ECD7;
    --text-primary:  #1A0A00;
    --text-secondary:#5A3D1A;
    --bg-dark:       #1A0E05;
    --bg-card:       #2A1A0A;
    --border:        rgba(212,175,55,0.3);
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #1A0E05 0%, #2D1507 40%, #1A0E05 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit decoration ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #250F02 0%, #1A0900 100%) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: var(--cream) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Chat container ── */
.chat-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* ── Message bubbles ── */
.user-message {
    background: linear-gradient(135deg, var(--maroon) 0%, var(--maroon-light) 100%);
    color: var(--cream);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0 8px 15%;
    font-family: 'Lora', serif;
    font-size: 0.97rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(128,0,32,0.4);
    border: 1px solid rgba(212,175,55,0.2);
    animation: slideInRight 0.3s ease;
}

.assistant-message {
    background: linear-gradient(135deg, #2A1A0A 0%, #3A2010 100%);
    color: var(--cream);
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    margin: 8px 15% 8px 0;
    font-family: 'Lora', serif;
    font-size: 0.97rem;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    border: 1px solid var(--border);
    animation: slideInLeft 0.3s ease;
}

.assistant-message p {
    margin: 0.4rem 0;
}
.assistant-message ul, .assistant-message ol {
    margin: 0.4rem 0 0.4rem 1.2rem;
    padding-left: 0.5rem;
}
.assistant-message li {
    margin-bottom: 0.3rem;
}
.assistant-message strong {
    color: var(--gold-light);
}

.message-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.75;
}

.user-message .message-label { color: var(--gold-light); }
.assistant-message .message-label { color: var(--saffron-light); }

/* ── Animations ── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── Title area ── */
.app-title {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
    animation: fadeIn 0.8s ease;
}
.app-title h1 {
    font-family: 'Cinzel', serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--gold) 0%, var(--saffron) 50%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem !important;
    text-shadow: none;
}
.app-title .subtitle {
    font-family: 'Lora', serif;
    color: var(--cream-dark);
    font-size: 1rem;
    font-style: italic;
    opacity: 0.85;
}
.app-title .divider {
    width: 200px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 0.8rem auto;
}

/* ── Sources panel ── */
.source-card {
    background: rgba(212,175,55,0.06);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--cream-dark);
    transition: all 0.2s ease;
}
.source-card:hover {
    background: rgba(212,175,55,0.12);
    border-color: rgba(212,175,55,0.5);
}
.source-title {
    font-weight: 600;
    color: var(--gold-light);
    margin-bottom: 4px;
    font-size: 0.88rem;
}
.source-excerpt {
    color: rgba(255,248,238,0.75);
    font-style: italic;
    font-family: 'Lora', serif;
    font-size: 0.83rem;
    line-height: 1.5;
    border-left: 2px solid var(--saffron-dark);
    padding-left: 10px;
    margin-top: 6px;
}
.source-score {
    font-size: 0.75rem;
    color: var(--saffron-light);
    opacity: 0.8;
    margin-top: 6px;
}

/* ── Example questions ── */
.example-chip {
    display: inline-block;
    background: rgba(255,153,51,0.12);
    border: 1px solid rgba(255,153,51,0.35);
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px;
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem;
    color: var(--saffron-light);
    cursor: pointer;
    transition: all 0.2s ease;
}
.example-chip:hover {
    background: rgba(255,153,51,0.25);
    border-color: var(--saffron);
}

/* ── Error/warning boxes ── */
.error-box {
    background: rgba(128,0,32,0.2);
    border: 1px solid rgba(128,0,32,0.5);
    border-radius: 10px;
    padding: 16px;
    color: #FFB3C1;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    animation: fadeIn 0.4s ease;
}
.info-box {
    background: rgba(212,175,55,0.08);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    color: var(--cream);
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
}

/* ── Streamlit widgets styling ── */
.stTextInput > div > div > input,
.stChatInput > div > div > input,
.stChatInput textarea {
    background: rgba(42,26,10,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--cream) !important;
    font-family: 'Lora', serif !important;
}
.stChatInput textarea:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.2) !important;
}

/* Streamlit buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--maroon), var(--maroon-light)) !important;
    color: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--maroon-light), var(--saffron-dark)) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(128,0,32,0.4) !important;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}
.status-ready {
    background: rgba(0,180,100,0.2);
    color: #66FFB0;
    border: 1px solid rgba(0,180,100,0.4);
}
.status-empty {
    background: rgba(255,153,51,0.15);
    color: var(--saffron-light);
    border: 1px solid rgba(255,153,51,0.35);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb {
    background: var(--saffron-dark);
    border-radius: 3px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(212,175,55,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--gold-light) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}

/* Insufficient evidence */
.insufficient-box {
    background: rgba(255,153,51,0.08);
    border: 1px solid rgba(255,153,51,0.3);
    border-left: 3px solid var(--saffron);
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'Lora', serif;
    font-style: italic;
    color: var(--cream-dark);
    margin-top: 8px;
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 12px 16px;
    animation: fadeIn 0.3s ease;
}
.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--saffron);
    animation: pulse 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* Sticker / ornament */
.ornament {
    text-align: center;
    color: var(--gold);
    font-size: 1.4rem;
    opacity: 0.6;
    margin: 4px 0;
    letter-spacing: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "messages": [],          # [{"role": "user"|"assistant", "content": str, "sources": list}]
        "pipeline": None,
        "pipeline_error": "",
        "top_k": settings.retriever_top_k,
        "similarity_threshold": settings.retriever_similarity_threshold,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.effective_llm_model,
        "use_streaming": True,
        "example_clicked": "",
        "custom_api_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def get_effective_api_key() -> str:
    """Get Groq API key from session state, os.environ, st.secrets, or settings."""
    if st.session_state.get("custom_api_key"):
        return st.session_state["custom_api_key"].strip()
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"].strip()
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return settings.groq_api_key.strip()


# ── Pipeline loader ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline(provider: str, model: str, api_key: str):
    """Load the RAG pipeline (cached — only rebuilt when config changes)."""
    from src.rag_pipeline import RAGPipeline
    from src.config import Settings

    import os
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    cfg = Settings(llm_model=model, groq_api_key=api_key)
    return RAGPipeline.from_settings(cfg)


def get_pipeline():
    """Return the cached pipeline, initialising if needed."""
    if st.session_state.pipeline is not None:
        return st.session_state.pipeline, ""

    provider = st.session_state.llm_provider
    model = st.session_state.llm_model
    api_key = get_effective_api_key()

    try:
        with st.spinner("🪔 Initialising Ramayana AI…"):
            pipeline = load_pipeline(provider, model, api_key)
        st.session_state.pipeline = pipeline
        st.session_state.pipeline_error = ""
        return pipeline, ""
    except Exception as exc:
        error = f"Failed to initialise pipeline: {exc}"
        st.session_state.pipeline_error = error
        return None, error


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        # Logo / title
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0;">
                <div style="font-family:'Cinzel',serif; font-size:1.3rem; color:#D4AF37; letter-spacing:0.1em;">🪔 RAMAYANA AI</div>
                <div style="font-family:'Inter',sans-serif; font-size:0.72rem; color:#FF9933; letter-spacing:0.15em; text-transform:uppercase; margin-top:4px;">Retrieval-Augmented Epic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Index status ──────────────────────────────────────────────────────
        st.markdown("**📚 Index Status**")
        pipeline, err = get_pipeline()
        if pipeline:
            count = pipeline.index_count()
            sources = pipeline.vector_store.list_sources()
            if count > 0:
                st.markdown(
                    f'<span class="status-badge status-ready">✓ {count:,} chunks indexed</span>',
                    unsafe_allow_html=True,
                )
                if sources:
                    with st.expander(f"📄 {len(sources)} source(s)", expanded=False):
                        for src in sources:
                            st.caption(f"• {src}")
                st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
                if st.button("🔄 Ingest / Sync New Documents", use_container_width=True):
                    _run_ingestion_in_sidebar(pipeline)
            else:
                st.markdown(
                    '<span class="status-badge status-empty">⚠ Empty index</span>',
                    unsafe_allow_html=True,
                )
                st.caption("Run ingestion to index documents.")

                # Quick ingest button
                if st.button("⚡ Ingest Documents", use_container_width=True):
                    _run_ingestion_in_sidebar(pipeline)
        else:
            st.warning(f"Pipeline error: {err}")

        st.markdown("---")

        # ── LLM Settings ──────────────────────────────────────────────────────
        st.markdown("**🤖 LLM Settings**")
        provider = st.selectbox(
            "Provider",
            options=["groq", "ollama"],
            index=0 if st.session_state.llm_provider == "groq" else 1,
            key="provider_select",
        )
        if provider != st.session_state.llm_provider:
            st.session_state.llm_provider = provider
            st.session_state.pipeline = None

        if provider == "groq":
            groq_models = [
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ]
            current_m = st.session_state.llm_model
            if current_m not in groq_models:
                groq_models.insert(0, current_m)
            model_idx = groq_models.index(current_m) if current_m in groq_models else 0
            model = st.selectbox(
                "Model",
                options=groq_models,
                index=model_idx,
            )
        else:
            model = st.text_input("Ollama Model", value=settings.ollama_model)

        if model != st.session_state.llm_model:
            st.session_state.llm_model = model
            st.session_state.pipeline = None

        # API key status & direct input
        if provider == "groq":
            api_key = get_effective_api_key()
            if api_key:
                st.success("✓ GROQ_API_KEY active", icon="🔑")
            else:
                st.warning("🔑 GROQ_API_KEY required", icon="⚠")
                custom_key = st.text_input(
                    "Enter Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    key="direct_api_key_input",
                    help="Paste your Groq API key here to activate the chatbot immediately.",
                )
                if custom_key and custom_key.strip():
                    st.session_state["custom_api_key"] = custom_key.strip()
                    os.environ["GROQ_API_KEY"] = custom_key.strip()
                    st.session_state.pipeline = None
                    st.rerun()

        st.markdown("---")

        # ── Retrieval Settings ────────────────────────────────────────────────
        st.markdown("**🔍 Retrieval Settings**")
        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=15,
            value=st.session_state.top_k,
            step=1,
            help="Number of document chunks to retrieve per query.",
        )
        st.session_state.top_k = top_k

        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.similarity_threshold),
            step=0.05,
            format="%.2f",
            help="Minimum similarity score for a chunk to be included (0 = no filter).",
        )
        st.session_state.similarity_threshold = threshold

        streaming = st.toggle("⚡ Streaming Responses", value=st.session_state.use_streaming)
        st.session_state.use_streaming = streaming

        st.markdown("---")

        # ── Clear conversation ────────────────────────────────────────────────
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # ── Notes ─────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:rgba(255,248,238,0.55); line-height:1.6;">
            <strong style="color:rgba(255,248,238,0.8);">ℹ About Answers</strong><br>
            Answers are grounded in the complete <em>Valmiki Ramayana</em> (Gita Press, Gorakhpur — English translation),
            covering all seven Kandas: Balakanda, Ayodhyakanda, Aranyakanda, Kishkindhakanda,
            Sundarakanda, Yuddhakanda, and Uttarakanda.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _run_ingestion_in_sidebar(pipeline=None) -> None:
    """Run document ingestion and refresh the pipeline."""
    with st.spinner("Ingesting documents…"):
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "scripts/ingest_documents.py"],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent),
                timeout=300,
            )
            if result.returncode == 0:
                # Invalidate cache to pick up new chunks
                load_pipeline.clear()
                st.session_state.pipeline = None
                st.success("✓ Ingestion complete. Reloading pipeline…")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Ingestion failed:\n{result.stderr[-500:]}")
        except Exception as exc:
            st.error(f"Ingestion error: {exc}")


# ── Header ────────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        """
        <div class="app-title">
            <div class="ornament">✦ ✦ ✦</div>
            <h1>Ramayana AI</h1>
            <div class="subtitle">Ask the Epic — Answers Grounded in Indexed Ramayana Texts</div>
            <div class="divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Example questions ─────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "Who is Hanuman and what makes him extraordinary?",
    "Why was Rama exiled from Ayodhya for 14 years?",
    "How did Hanuman find Sita in Lanka? (Sundarakanda)",
    "Who is Ravana and why did he abduct Sita?",
    "What is the significance of the bridge (Rama Setu) to Lanka?",
    "Describe the relationship between Rama and Lakshmana.",
    "Who is Vibhishana and what role did he play in the war?",
    "What is the Agni Pariksha and why did Sita undergo it?",
    "How did Rama kill Ravana in the Yuddhakanda?",
    "What boon did Kaikeyi demand from King Dasharatha?",
    "Describe the Kishkindhakanda — Rama's alliance with Sugriva.",
    "Who are the Vanaras and what was their role in Lanka war?",
]


def render_examples() -> str | None:
    """Render clickable example question chips. Returns clicked question or None."""
    st.markdown(
        '<div style="font-family:\'Inter\',sans-serif; font-size:0.8rem; color:rgba(255,248,238,0.6); '
        'text-align:center; margin-bottom:8px;">Try asking about:</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        col = cols[i % 4]
        with col:
            if st.button(q[:45] + ("…" if len(q) > 45 else ""), key=f"ex_{i}", use_container_width=True):
                return q
    return None


# ── Message rendering ─────────────────────────────────────────────────────────

def render_message(role: str, content: str, sources: list | None = None) -> None:
    """Render a chat message with optional source citations."""
    if role == "user":
        st.markdown(
            f'<div class="user-message"><div class="message-label">You</div>{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        is_insufficient = any(
            phrase in content
            for phrase in ["could not find", "insufficient information", "not yet indexed"]
        )
        content_html = content.replace("\n", "<br>")

        if is_insufficient:
            st.markdown(
                f'<div class="assistant-message">'
                f'<div class="message-label">🪔 Ramayana AI</div>'
                f'<div class="insufficient-box">⚠️\n\n{content}\n\n</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-message">'
                f'<div class="message-label">🪔 Ramayana AI</div>\n\n'
                f"{content}\n\n"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Sources panel
        if sources:
            with st.expander(f"📜 Sources used ({len(sources)})", expanded=False):
                for src in sources:
                    label = src.get("label", "Unknown Source")
                    excerpt = src.get("excerpt", "")
                    score = src.get("score", "—")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-title">📄 {label}</div>'
                        f'<div class="source-excerpt">"{excerpt}"</div>'
                        f'<div class="source-score">Relevance score: {score}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ── Main chat logic ───────────────────────────────────────────────────────────

def run_rag_query(question: str, pipeline) -> tuple[str, list]:
    """Run a RAG query and return (answer_text, sources_list)."""
    from src.citations import format_citations_for_display

    try:
        prior_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
        if st.session_state.use_streaming:
            # Streaming path
            retrieved, token_stream = pipeline.stream_answer(
                question=question,
                history=prior_history,
                top_k=st.session_state.top_k,
                similarity_threshold=st.session_state.similarity_threshold,
            )

            # Stream tokens into a placeholder
            answer_parts: list[str] = []
            placeholder = st.empty()
            full_text = ""
            for token in token_stream:
                answer_parts.append(token)
                full_text = "".join(answer_parts)
                # Show streaming text
                placeholder.markdown(
                    f'<div class="assistant-message">'
                    f'<div class="message-label">🪔 Ramayana AI</div>\n\n'
                    f"{full_text}\n\n"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            placeholder.empty()
            answer = full_text

            # Build sources from retrieved chunks
            from src.citations import extract_and_validate_citations
            citation_result = extract_and_validate_citations(answer, retrieved)
            sources = format_citations_for_display(citation_result)

        else:
            # Non-streaming path
            response = pipeline.answer(
                question=question,
                history=prior_history,
                top_k=st.session_state.top_k,
                similarity_threshold=st.session_state.similarity_threshold,
            )
            answer = response.answer
            sources = format_citations_for_display(response.citation_result)

        return answer, sources

    except Exception as exc:
        logger.error("RAG query failed: %s", exc, exc_info=True)
        error_msg = (
            f"An error occurred while generating the answer.\n\n"
            f"**Error**: {type(exc).__name__}: {str(exc)[:200]}\n\n"
            "Please check your API key and try again."
        )
        return error_msg, []


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    # Main content area
    render_header()

    pipeline, pipeline_error = get_pipeline()

    # Show critical errors prominently
    if pipeline_error and not pipeline:
        st.markdown(
            f'<div class="error-box">⚠️ <strong>Initialisation Error</strong><br>{pipeline_error}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # API key warning
    effective_api_key = get_effective_api_key()
    if settings.llm_provider == "groq" and not effective_api_key:
        st.markdown(
            """
            <div class="error-box">
            🔑 <strong>Groq API Key Required</strong><br>
            Please enter your Groq API key in the sidebar, or configure it in Streamlit Secrets / <code>.env</code> file.<br>
            Get a free key at <a href="https://console.groq.com" style="color:#FF9933;">console.groq.com</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Empty index warning
    if pipeline and pipeline.vector_store.is_empty():
        st.markdown(
            """
            <div class="info-box">
            📚 <strong>No Documents Indexed Yet</strong><br>
            The vector store is empty. Click <em>⚡ Ingest Demo Dataset</em> in the sidebar, or run:<br>
            <code>python scripts/ingest_documents.py</code><br>
            from the project directory. See <code>data/README.md</code> for source document guidance.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Render existing messages ──────────────────────────────────────────────
    if st.session_state.messages:
        for msg in st.session_state.messages:
            render_message(
                msg["role"],
                msg["content"],
                msg.get("sources"),
            )
    else:
        # Welcome screen with examples
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        render_examples_col, _ = st.columns([3, 1])
        with render_examples_col:
            clicked = render_examples()
            if clicked:
                st.session_state.example_clicked = clicked
                st.rerun()

    # Handle example click
    if st.session_state.example_clicked:
        question = st.session_state.example_clicked
        st.session_state.example_clicked = ""
        # Add to messages and process
        st.session_state.messages.append({"role": "user", "content": question})
        render_message("user", question)
        _process_question(question, pipeline)
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input(
        "Ask about characters, events, teachings, or themes in the Ramayana…",
        key="chat_input",
    )

    if question and question.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        render_message("user", question)
        _process_question(question, pipeline)
        st.rerun()


def _process_question(question: str, pipeline) -> None:
    """Process a user question and add the assistant's reply to the session."""
    if pipeline is None:
        st.error("Pipeline not initialised. Check sidebar for errors.")
        return

    if pipeline.vector_store.is_empty():
        answer = (
            "The document index is empty. Please ingest documents first by clicking "
            "'⚡ Ingest Demo Dataset' in the sidebar, or running the ingestion script."
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [],
        })
        return

    effective_api_key = get_effective_api_key()
    if not effective_api_key and settings.llm_provider == "groq":
        answer = (
            "A Groq API key is required to generate answers. "
            "Please enter your Groq API key in the sidebar, or configure it in Streamlit Secrets."
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [],
        })
        return

    # Run the RAG pipeline
    with st.spinner("🔍 Searching the Ramayana…"):
        answer, sources = run_rag_query(question, pipeline)

    # Store in session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    # Render the answer (will be properly re-rendered on rerun)
    render_message("assistant", answer, sources)


if __name__ == "__main__":
    main()
