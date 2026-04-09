import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import VECTORSTORE_DIR
from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()


st.set_page_config(page_title="Debales AI Assistant", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #0f1117; color: #e6e6e6; }
    .stChatMessage { border-radius: 12px; padding: 12px; }
    .stChatMessage.user { background: #1b1f2a; }
    .stChatMessage.assistant { background: #151a24; }
    .stTextInput > div > div { background-color: #1b1f2a; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Debales AI Assistant")
st.caption("RAG for Debales AI + SERP for external queries, powered by LangGraph.")


@st.cache_resource
def _graph():
    return build_graph()


vectorstore_ready = Path(VECTORSTORE_DIR).exists()
if not vectorstore_ready:
    st.warning("Vector store not found. Run `python scraper.py` then `python ingest.py` before chatting.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about Debales AI or anything else...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = _graph()
                result = graph.invoke(
                    {"question": prompt, "route": "unknown", "rag_context": None, "serp_context": None}
                )
                answer = result.get("answer", "I don't know.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = (
                    "⚠️ I hit a runtime issue while processing that query. "
                    "Please try again. If this continues, rebuild data with `python ingest.py`."
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                logging.error(f"Graph invocation error: {e}", exc_info=True)
