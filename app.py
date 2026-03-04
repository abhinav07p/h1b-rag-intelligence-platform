"""
H-1B RAG Application — Entry Point
====================================
AI-powered H-1B visa sponsorship intelligence platform.

INFO 7390 Final Project
Author: Abhinav Kumar Piyush

This file is the Streamlit entry point.  All page logic lives in pages/*.py.
This file only handles global config, shared CSS, and session state / sidebar.
"""

import streamlit as st
from src.data.loader import load_h1b_data

# =============================================================================
# PAGE CONFIG (must be first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="H-1B Sponsorship Intelligence",
    page_icon="🎯",
    layout="wide",
)

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #14b8a6, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.prediction-high   { color: #10b981; font-size: 2rem; font-weight: bold; }
.prediction-medium { color: #f59e0b; font-size: 2rem; font-weight: bold; }
.prediction-low    { color: #ef4444; font-size: 2rem; font-weight: bold; }
.factor-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 3px solid #14b8a6;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "df" not in st.session_state:
    st.session_state.df = load_h1b_data()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🎯 H-1B Intelligence")
    st.markdown("*Real 2024 USCIS Data*")
    st.markdown("---")

    st.markdown("### 🔑 API Keys")
    try:
        default_openai   = st.secrets.get("OPENAI_API_KEY", "")
        default_pinecone = st.secrets.get("PINECONE_API_KEY", "")
    except Exception:
        default_openai   = ""
        default_pinecone = ""

    openai_key   = st.text_input("OpenAI API Key",   value=default_openai,   type="password")
    pinecone_key = st.text_input("Pinecone API Key", value=default_pinecone, type="password")
    use_demo     = st.checkbox("🎮 Demo Mode", value=(not openai_key or not pinecone_key))

    # Store in session so pages can read them
    st.session_state["openai_key"]   = openai_key
    st.session_state["pinecone_key"] = pinecone_key
    st.session_state["use_demo"]     = use_demo

    st.markdown("---")
    st.markdown("### 📈 Data Stats")
    df = st.session_state.df
    st.metric("Companies",     len(df))
    st.metric("Total Filings", f"{df['total_filings'].sum():,}")

# =============================================================================
# HOME / LANDING
# =============================================================================
st.markdown('<h1 class="main-header">H-1B Sponsorship Intelligence</h1>', unsafe_allow_html=True)
st.markdown(
    "Use the **sidebar navigation** to explore the dashboard, compare companies, "
    "predict sponsorship likelihood, or chat with the AI advisor."
)

col1, col2, col3 = st.columns(3)
col1.metric("Companies",     len(df))
col2.metric("Total Filings", f"{df['total_filings'].sum():,}")
col3.metric("Avg Salary",    f"${df['avg_salary'].mean():,.0f}")
