"""
Shared session state initializer for all pages.
Import this at the top of every page file to ensure session state is always ready,
even if the user navigates directly to a page without going through app.py first.
"""
import streamlit as st
from src.data.loader import load_h1b_data

def init_session_state():
    """Initialize all required session state keys if not already set."""
    if "df" not in st.session_state:
        st.session_state.df = load_h1b_data()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "pinecone_key" not in st.session_state:
        st.session_state.pinecone_key = ""
    if "use_demo" not in st.session_state:
        st.session_state.use_demo = True
