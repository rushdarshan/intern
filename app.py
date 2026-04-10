import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import OPENROUTER_API_KEY, VECTORSTORE_DIR
from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()


st.set_page_config(page_title="Debales AI Assistant", page_icon="D", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #111827 45%, #0b1220 100%);
        color: #e5e7eb;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
    }
    .stChatMessage.user {
        background: rgba(255, 255, 255, 0.04);
    }
    .stChatMessage.assistant {
        background: rgba(255, 255, 255, 0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Debales AI Assistant")
st.caption("LangGraph routing with Debales RAG and external SERP search.")


@st.cache_resource
def _graph():
    return build_graph()


def _render_sources(sources: list[dict[str, str]], warnings: list[str], route: str, route_reason: str) -> None:
    with st.expander("Routing and sources", expanded=False):
        st.write(f"Route: `{route}`")
        if route_reason:
            st.write(f"Reason: {route_reason}")

        if warnings:
            st.write("Warnings:")
            for warning in warnings:
                st.write(f"- {warning}")

        if sources:
            st.write("Sources:")
            for source in sources:
                title = source.get("title") or source.get("url") or "Untitled source"
                url = source.get("url", "")
                label = f"{title} - {url}" if url else title
                st.write(f"- {label}")
        else:
            st.write("Sources: none")


vectorstore_ready = (Path(VECTORSTORE_DIR) / "chroma.sqlite3").exists()
llm_ready = bool(OPENROUTER_API_KEY)

with st.sidebar:
    st.subheader("Runtime status")
    st.write(f"Knowledge base: {'ready' if vectorstore_ready else 'missing'}")
    st.write(f"LLM key: {'configured' if llm_ready else 'missing'}")
    st.caption("SERP availability is checked at query time from `SERPAPI_API_KEY`.")

if not vectorstore_ready:
    st.warning(
        "Debales-specific questions need a built vector store. Run `python scraper.py` and `python ingest.py` first."
    )
if not llm_ready:
    st.warning("OPENROUTER_API_KEY is missing. Classification and answer generation will fail until it is set.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            _render_sources(
                meta.get("sources", []),
                meta.get("warnings", []),
                meta.get("route", "unknown"),
                meta.get("route_reason", ""),
            )

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
                    {
                        "question": prompt,
                        "route": "unknown",
                        "route_reason": "",
                        "rag_context": "",
                        "serp_context": "",
                        "sources": [],
                        "warnings": [],
                    }
                )
                answer = result.get("answer", "I don't know based on the available context.")
                route = result.get("route", "unknown")
                route_reason = result.get("route_reason", "")
                sources = result.get("sources", [])
                warnings = result.get("warnings", [])

                st.markdown(answer)
                st.caption(f"Route: {route} • Sources: {len(sources)}")
                _render_sources(sources, warnings, route, route_reason)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "route": route,
                            "route_reason": route_reason,
                            "sources": sources,
                            "warnings": warnings,
                        },
                    }
                )
            except Exception as exc:
                error_msg = (
                    "I hit a runtime issue while processing that query. "
                    "Check your API keys and rebuild the knowledge base if the issue persists."
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                logging.error("Graph invocation error: %s", exc, exc_info=True)
