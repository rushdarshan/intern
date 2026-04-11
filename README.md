# Debales AI Assistant

A LangGraph-based assistant that answers Debales AI questions with RAG and external questions with SerpAPI. It includes a Streamlit UI, a CLI, focused Debales scraping, relevance-filtered retrieval, and grounded answer generation that refuses unsupported claims.

## What It Does

- Debales website, blog, AI-agent, logistics, docs, and integration questions go to the local Chroma knowledge base.
- External questions go to SerpAPI.
- Mixed questions use both paths.
- If the retrieved evidence is weak or missing, the assistant replies with `I don't know based on the available context.`

## Architecture

```text
User question
  -> classify
  -> debales -> retrieve from Chroma
  -> serp    -> search with SerpAPI
  -> both    -> retrieve from Chroma, then search with SerpAPI
  -> grounded answer generation with citations
```

The mixed route is sequential by design so the graph stays easy to explain in an internship submission.

## Project Files

- `app.py`: Streamlit chat UI with route/source inspection.
- `cli.py`: terminal interface for quick demos and testing.
- `graph.py`: LangGraph workflow and answer generation.
- `scraper.py`: focused Debales crawler with sitemap support, retries, normalization, and metadata extraction.
- `ingest.py`: chunking and clean Chroma rebuilds.
- `rag.py`: vector store loading plus relevance-filtered retrieval.
- `serp_tool.py`: SerpAPI wrapper with cleaned results and graceful error handling.
- `config.py`: environment-driven configuration.
- `tests/`: unit tests for routing, scraper behavior, and SERP result handling.

## Requirements

- Python 3.10+
- `OPENROUTER_API_KEY`
- `SERPAPI_API_KEY`

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your real API keys.

## Build The Knowledge Base

```bash
python scraper.py
python ingest.py
```

`python ingest.py` resets and rebuilds the local Chroma store each time, so repeated ingests do not accumulate duplicate chunks.

## Run The Assistant

Streamlit UI:

```bash
streamlit run app.py
```

CLI:

```bash
python cli.py
python cli.py "What integrations does Debales AI support?" --show-metadata
```

## Configuration

Key environment variables:

```env
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini

SERPAPI_API_KEY=
SERP_TIMEOUT=20
SERP_MAX_RESULTS=5

DEBALES_BASE_URL=https://debales.ai
DEBALES_SITEMAP_URL=https://debales.ai/sitemap.xml
DEBALES_MAX_PAGES=80
ALLOWED_PATH_PREFIXES=/blog,/docs,/faq,/logistics,/ai-agent,/ai-agents,/case-study,/case-studies,/integration,/integrations,/product,/products,/solution,/solutions,/use-cases
EXCLUDED_PATH_PREFIXES=/sign-in,/sign-up,/login,/register,/privacy,/terms,/contact
REQUEST_TIMEOUT=20

RAW_DOCS_PATH=data/raw/docs.jsonl
VECTORSTORE_DIR=data/vectorstore
EMBEDDING_PROVIDER=auto
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=1024
EMBEDDING_DEVICE=cpu
CHUNK_SIZE=900
CHUNK_OVERLAP=150
RAG_TOP_K=4
RAG_RELEVANCE_THRESHOLD=0.18
```

`EMBEDDING_PROVIDER=auto` tries Hugging Face embeddings first and falls back to local hashing embeddings if the model is unavailable.

## Example Prompts

Debales:

```text
What does Debales AI do for freight brokers?
```

External:

```text
What is the capital of France?
```

Mixed:

```text
How does Debales AI compare with Salesforce Service Cloud for logistics support?
```

**Demo Video:** https://youtu.be/zA21HQRzRvE

Unsupported:

```text
Tell me about Debales AI's unreleased secret roadmap.
```

## Reliability Notes

- Crawl scope is focused to relevant Debales sections instead of all same-domain pages.
- URLs are normalized before deduping, so fragments and query variants do not create duplicate documents.
- Retrieval filters low-relevance chunks before answer generation.
- SERP failures degrade gracefully and do not crash Debales-only answers.
- The UI and CLI both expose route, warnings, and sources, which makes the graph easy to explain in a demo.

## Verification

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Useful manual checks:

```bash
python cli.py "What does Debales AI do?" --show-metadata
python cli.py "What is the capital of France?" --show-metadata
python cli.py "What integrations do you support?" --show-metadata
```

## Demo Video Outline

1. Show `.env.example` and the main config knobs.
2. Run `python scraper.py` and `python ingest.py`.
3. Launch `streamlit run app.py` or `python cli.py`.
4. Ask one Debales question, one external question, and one mixed question.
5. Open the route/source metadata for at least one response.
