import json
import logging
import re
from functools import lru_cache
from typing import Literal, TypedDict, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from rag import search_relevant_chunks
from serp_tool import SerpResponse, serp_search

logger = logging.getLogger(__name__)

Route = Literal["debales", "serp", "both", "unknown"]


class SourceItem(TypedDict, total=False):
    type: str
    title: str
    url: str
    snippet: str


class GraphState(TypedDict, total=False):
    question: str
    route: Route
    route_reason: str
    rag_context: str
    serp_context: str
    sources: list[SourceItem]
    warnings: list[str]
    answer: str


_ROUTE_PATTERN = re.compile(r"\b(debales|serp|both|unknown)\b", re.IGNORECASE)
_DEBALES_TERMS = (
    "debales",
    "debales ai",
    "debales.ai",
)
_DEBALES_DOMAIN_TERMS = (
    "ai agent",
    "ai agents",
    "integration",
    "integrations",
    "case study",
    "case studies",
    "freight",
    "3pl",
    "carrier",
    "shipment",
    "logistics",
    "broker",
    "quote",
    "quoting",
    "load building",
    "customer service",
    "messometer",
)
_MIXED_TERMS = (
    "compare",
    "vs",
    "versus",
    "difference",
    "compared",
    "along with",
    "latest",
    "current",
    "pricing",
    "competitor",
)
_GENERAL_QUERY_PREFIXES = (
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "tell me",
    "explain",
    "list",
    "show me",
    "give me",
)
_GENERAL_QUERY_TERMS = (
    "capital",
    "weather",
    "latest",
    "current",
    "price",
    "definition",
    "meaning",
    "news",
    "compare",
)
_EMPTY_QUESTION = {"", "hi", "hello", "hey", "test"}


def _llm() -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question or "").strip().lower()


def _route_from_text(text: str) -> Route | None:
    if not text:
        return None
    match = _ROUTE_PATTERN.search(text)
    if not match:
        return None
    return cast(Route, match.group(1).lower())


def _has_debales_signal(question: str) -> bool:
    return any(term in question for term in _DEBALES_TERMS)


def _has_debales_context_signal(question: str) -> bool:
    if _has_debales_signal(question):
        return True

    domain_matches = sum(1 for term in _DEBALES_DOMAIN_TERMS if term in question)
    if domain_matches >= 2:
        return True

    padded_question = f" {question} "
    if domain_matches >= 1 and (
        " your " in padded_question or " you " in padded_question or question.startswith("do you")
    ):
        return True

    return False


def _has_mixed_signal(question: str) -> bool:
    return any(term in question for term in _MIXED_TERMS)


def _looks_like_general_question(question: str) -> bool:
    return (
        question.endswith("?")
        or any(question.startswith(prefix) for prefix in _GENERAL_QUERY_PREFIXES)
        or any(term in question for term in _GENERAL_QUERY_TERMS)
    )


def _is_trivial_question(question: str) -> bool:
    return question in _EMPTY_QUESTION or len(question.split()) <= 1


def _unique_sources(items: list[SourceItem]) -> list[SourceItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[SourceItem] = []
    for item in items:
        key = (
            item.get("type", ""),
            item.get("url", ""),
            item.get("title", ""),
            item.get("snippet", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_source_block(sources: list[SourceItem]) -> str:
    if not sources:
        return "No sources were retrieved."
    lines = []
    for index, source in enumerate(sources, start=1):
        title = source.get("title") or source.get("url") or "Untitled source"
        url = source.get("url", "")
        snippet = source.get("snippet", "").strip()
        suffix = f" - {snippet}" if snippet else ""
        lines.append(f"[{index}] {title} ({url}){suffix}")
    return "\n".join(lines)


def _append_warning(state: GraphState, warning: str) -> GraphState:
    warnings = list(state.get("warnings", []))
    warnings.append(warning)
    return {**state, "warnings": warnings}


def classify_question(state: GraphState) -> GraphState:
    question = state["question"]
    normalized = _normalize_question(question)

    if _is_trivial_question(normalized):
        logger.info("Classified as unknown because the question is too short: %s", question)
        return {**state, "route": "unknown", "route_reason": "Question is too vague to route confidently."}

    if _has_debales_context_signal(normalized):
        route = "both" if _has_mixed_signal(normalized) else "debales"
        reason = "Debales AI domain signal detected."
        if route == "both":
            reason = "Debales AI signal plus an external comparison/current-information cue."
        logger.info("Classified as %s: %s", route, question[:80])
        return {**state, "route": route, "route_reason": reason}

    if _looks_like_general_question(normalized):
        logger.info("Classified as serp from general question heuristic: %s", question[:80])
        return {**state, "route": "serp", "route_reason": "General external question detected."}

    prompt = (
        "Classify the user question into one of these routes: debales, serp, both, unknown.\n"
        "debales: about Debales AI company/products/blog/integrations.\n"
        "serp: not about Debales AI.\n"
        "both: mixed question containing Debales AI and external topics.\n"
        "unknown: too vague or impossible to determine safely.\n\n"
        "Treat product, integration, AI agent, logistics automation, quoting, load-building, or case-study "
        "questions as debales even if the user does not say the company name explicitly.\n"
        "Return JSON only, for example {\"route\": \"serp\"}.\n"
        f"Question: {question}"
    )

    try:
        response = _llm().invoke(prompt).content
    except Exception as exc:
        logger.warning("Route classification fell back to unknown after LLM failure: %s", exc)
        return {
            **state,
            "route": "unknown",
            "route_reason": "Unable to classify confidently without the language model.",
        }

    route = _route_from_text(response)
    if route is None:
        try:
            data = json.loads(response)
            route = data.get("route")
        except Exception:
            route = None

    if route not in {"debales", "serp", "both", "unknown"}:
        logger.warning("Invalid route response '%s'; using unknown", response[:120])
        route = "unknown"

    logger.info("Classified as %s: %s", route, question[:80])
    return {**state, "route": route, "route_reason": "LLM classification fallback."}


def retrieve_rag(state: GraphState) -> GraphState:
    logger.info("Starting RAG retrieval")
    try:
        matches = search_relevant_chunks(state["question"])
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return _append_warning(state, f"RAG retrieval failed: {exc}")

    if not matches:
        logger.info("No RAG documents found")
        updated_state: GraphState = {
            **state,
            "rag_context": "",
            "sources": list(state.get("sources", [])),
        }
        return _append_warning(updated_state, "No relevant Debales knowledge base matches were found.")

    sources = list(state.get("sources", []))
    context_blocks = []
    for idx, (doc, score) in enumerate(matches, start=1):
        title = str(doc.metadata.get("title", "")).strip() or "Debales AI source"
        url = str(doc.metadata.get("source", "")).strip()
        chunk = doc.page_content.strip()
        context_blocks.append(f"[{idx}] {title}\n{chunk}\nSource: {url}\nRelevance: {score:.2f}")
        sources.append(
            {
                "type": "debales",
                "title": title,
                "url": url,
                "snippet": chunk[:220],
            }
        )

    deduped_sources = _unique_sources(sources)
    logger.info("Retrieved %s RAG documents", len(matches))
    return {
        **state,
        "rag_context": "\n\n".join(context_blocks),
        "sources": deduped_sources,
    }


def search_serp(state: GraphState) -> GraphState:
    logger.info("Starting SerpAPI search")
    response: SerpResponse = serp_search(state["question"])
    results = response["results"]
    sources = list(state.get("sources", []))

    if response["error"]:
        logger.warning("SERP search returned a warning: %s", response["error"])
        state = _append_warning(state, response["error"])

    if not results:
        logger.info("No SERP results found")
        return {
            **state,
            "serp_context": "",
            "sources": _unique_sources(sources),
        }

    result_lines = []
    for item in results:
        result_lines.append(
            f"[{item['position']}] {item['title']}\n{item['snippet']}\nSource: {item['link']}"
        )
        sources.append(
            {
                "type": "serp",
                "title": item["title"],
                "url": item["link"],
                "snippet": item["snippet"],
            }
        )

    logger.info("Retrieved %s SERP results", len(results))
    return {
        **state,
        "serp_context": "\n\n".join(result_lines),
        "sources": _unique_sources(sources),
    }


def generate_answer(state: GraphState) -> GraphState:
    logger.info("Generating answer")
    rag_context = state.get("rag_context", "").strip()
    serp_context = state.get("serp_context", "").strip()
    sources = state.get("sources", [])
    warnings = state.get("warnings", [])

    if not rag_context and not serp_context:
        return {
            **state,
            "answer": "I don't know based on the available context.",
        }

    prompt = (
        "You are a careful assistant for the Debales AI assignment.\n"
        "Use only the supplied context. Do not invent facts.\n"
        "If the context does not support the answer, reply exactly: I don't know based on the available context.\n"
        "For supported claims, cite sources inline using bracket numbers like [1].\n"
        "Keep the answer concise and factual.\n\n"
        f"Question:\n{state['question']}\n\n"
        f"Debales context:\n{rag_context or 'None'}\n\n"
        f"Web context:\n{serp_context or 'None'}\n\n"
        f"Available sources:\n{_format_source_block(sources)}\n\n"
        f"Warnings:\n{'; '.join(warnings) if warnings else 'None'}"
    )

    try:
        response = _llm().invoke(prompt).content.strip()
    except Exception as exc:
        logger.warning("Answer generation failed: %s", exc)
        response = "I don't know based on the available context."

    if not response:
        response = "I don't know based on the available context."

    return {**state, "answer": response}


def route_decision(state: GraphState) -> str:
    route = state.get("route", "unknown")
    logger.info("Routing decision: %s", route)
    if route == "debales":
        return "rag"
    if route == "serp":
        return "serp"
    if route == "both":
        return "rag"
    return "answer"


def after_rag_decision(state: GraphState) -> str:
    route = state.get("route", "unknown")
    next_step = "serp" if route == "both" else "answer"
    logger.info("After RAG routing: %s", next_step)
    return next_step


@lru_cache(maxsize=1)
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
            "answer": "answer",
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


def run_question(question: str) -> GraphState:
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "route": "unknown",
        "route_reason": "",
        "rag_context": "",
        "serp_context": "",
        "sources": [],
        "warnings": [],
    }
    return graph.invoke(initial_state)
