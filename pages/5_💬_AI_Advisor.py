"""
AI Advisor (RAG) Page
"""
import streamlit as st
from src.rag.vector_store import H1BVectorStore, MockVectorStore
from src.rag.agent import H1BRAGAgent, MockRAGAgent
from pages._init_session import init_session_state

init_session_state()

st.markdown('<h1 class="main-header">AI Advisor (RAG)</h1>', unsafe_allow_html=True)
st.markdown("Ask questions about H-1B sponsorship using GPT-4o + Pinecone")

df        = st.session_state.df
use_demo  = st.session_state.get("use_demo", True)
openai_key   = st.session_state.get("openai_key", "")
pinecone_key = st.session_state.get("pinecone_key", "")

if use_demo:
    st.info("🎮 **Demo Mode** - Using mock responses. Add API keys in the sidebar for real GPT-4o.")
    vector_store = MockVectorStore(df)
    agent        = MockRAGAgent(vector_store, df)
else:
    if not st.session_state.indexed:
        st.warning("⚠️ Index companies to Pinecone first for semantic search.")
        if st.button("📥 Index to Pinecone", type="primary"):
            with st.spinner("Indexing..."):
                try:
                    vs = H1BVectorStore(openai_key, pinecone_key)
                    vs.initialize()
                    vs.index_companies(df)
                    st.session_state.vector_store = vs
                    st.session_state.indexed = True
                    st.success("✅ Indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.indexed:
        vector_store = st.session_state.vector_store
        agent        = H1BRAGAgent(openai_key, vector_store)
    else:
        vector_store = MockVectorStore(df)
        agent        = MockRAGAgent(vector_store, df)

st.markdown("### 💡 Sample Questions")
samples = ["Top H-1B sponsors?", "Highest paying companies?", "H-1B lottery odds?", "Job search tips?"]
cols = st.columns(4)
for i, q in enumerate(samples):
    if cols[i].button(q, key=f"s{i}"):
        st.session_state.chat_history.append({"role": "user", "content": q})
        response = agent.chat(q)
        st.session_state.chat_history.append({
            "role": "assistant", "content": response.response, "sources": response.sources
        })
        st.rerun()

st.markdown("---")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about H-1B..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
            st.write(response.response)
    st.session_state.chat_history.append({
        "role": "assistant", "content": response.response, "sources": response.sources
    })

if st.button("🗑️ Clear"):
    st.session_state.chat_history = []
    st.rerun()
