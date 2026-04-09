import json
import logging
from typing import Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from rag import get_retriever
from serp_tool import serp_search

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    question: str
    route: Literal["debales", "serp", "both", "unknown"]
    rag_context: Optional[str]
    serp_context: Optional[str]
    answer: Optional[str]


def _llm() -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")
    return ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def classify_question(state: GraphState) -> GraphState:
    question = state["question"]
    lowered = question.lower()
    if "debales" in lowered or "debales ai" in lowered or "debales.ai" in lowered:
        logger.info(f"🤖 Classified as 'debales': {question[:50]}...")
        return {**state, "route": "debales"}

    prompt = (
        "Classify the user question into one of these routes: debales, serp, both, unknown.\n"
        "debales: about Debales AI company/products/blog/integrations.\n"
        "serp: not about Debales AI.\n"
        "both: mixed question containing Debales AI and external topics.\n"
        "unknown: cannot determine intent.\n\n"
        "Return JSON only, e.g. {\"route\": \"serp\"}.\n"
        f"Question: {question}"
    )
    response = _llm().invoke(prompt).content
    try:
        data = json.loads(response)
        route = data.get("route", "unknown")
    except json.JSONDecodeError:
        logger.warning(f"⚠️  Failed to parse LLM route response: {response[:50]}")
        route = "unknown"

    if route not in {"debales", "serp", "both", "unknown"}:
        logger.warning(f"⚠️  Invalid route '{route}', defaulting to 'unknown'")
        route = "unknown"

    logger.info(f"🤖 Classified as '{route}': {question[:50]}...")
    return {**state, "route": route}


def retrieve_rag(state: GraphState) -> GraphState:
    logger.info("📚 Starting RAG retrieval...")
    try:
        retriever = get_retriever()
        docs = retriever.invoke(state["question"])
        if not docs:
            logger.info("📚 No RAG documents found")
            return {**state, "rag_context": None}
        context = "\n\n".join(
            f"[{idx + 1}] {doc.page_content}\nSource: {doc.metadata.get('source', '')}"
            for idx, doc in enumerate(docs)
        )
        logger.info(f"✓ Retrieved {len(docs)} RAG documents")
        return {**state, "rag_context": context}
    except Exception as e:
        logger.error(f"❌ RAG retrieval error: {e}")
        raise


def search_serp(state: GraphState) -> GraphState:
    logger.info("🔍 Starting SerpAPI search...")
    try:
        results = serp_search(state["question"])
        if not results:
            logger.info("🔍 No SerpAPI results found")
            return {**state, "serp_context": None}
        context = "\n\n".join(
            f"[{idx + 1}] {item['title']}\n{item['snippet']}\nSource: {item['link']}"
            for idx, item in enumerate(results)
        )
        logger.info(f"✓ Retrieved {len(results)} SERP results")
        return {**state, "serp_context": context}
    except Exception as e:
        logger.error(f"❌ SERP search error: {e}")
        raise


def generate_answer(state: GraphState) -> GraphState:
    logger.info("💬 Generating answer...")
    rag_context = state.get("rag_context")
    serp_context = state.get("serp_context")
    if not rag_context and not serp_context:
        logger.warning("⚠️  No context available for answer generation")
        return {
            **state,
            "answer": "I don't have enough information to answer that yet. Please rephrase or provide more detail.",
        }

    prompt = (
        "You are a helpful AI assistant. Answer the user's question based on the provided context.\n"
        "Use information from either Debales context OR web search context, whichever is relevant.\n"
        "If the answer is present in the context, provide it with citations like [1].\n"
        "If the answer is NOT in the context, say 'I don't know'.\n\n"
        f"Question: {state['question']}\n\n"
        f"Debales context:\n{rag_context or 'N/A'}\n\n"
        f"Web search context:\n{serp_context or 'N/A'}\n"
    )
    response = _llm().invoke(prompt).content
    logger.info("✓ Answer generated")
    return {**state, "answer": response}


def route_decision(state: GraphState) -> str:
    route = state.get("route", "unknown")
    logger.info(f"🚦 Routing decision: {route}")
    if route == "debales":
        return "rag"
    if route == "serp":
        return "serp"
    if route == "both":
        return "both"
    return "unknown"


def after_rag_decision(state: GraphState) -> str:
    route = state.get("route")
    next_step = "serp" if route == "both" else "answer"
    logger.info(f"🚦 After RAG: {next_step}")
    return next_step


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify", classify_question)
    graph.add_node("rag", retrieve_rag)
    graph.add_node("serp", search_serp)
    graph.add_node("answer", generate_answer)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_decision,
        {
            "rag": "rag",
            "serp": "serp",
            "both": "rag",
            "unknown": "answer",
        },
    )
    graph.add_conditional_edges(
        "rag",
        after_rag_decision,
        {
            "serp": "serp",
            "answer": "answer",
        },
    )
    graph.add_edge("serp", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_question(question: str) -> str:
    graph = build_graph()
    result = graph.invoke({"question": question, "route": "unknown", "rag_context": None, "serp_context": None})
    return result["answer"]
