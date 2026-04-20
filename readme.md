# Autonomous Financial AI Agent 

An asynchronous, multi-agent LLM pipeline designed to automate quantitative financial research, fundamental document synthesis, and real-time news sentiment analysis. Built entirely with open-source models, this system leverages graph-based agentic workflows to reason through market data, corporate filings, and daily news cycles.

## Architecture & Tech Stack

* **Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph) / LangChain
* **Frontend:** Streamlit
* **Local Inference (Apple MPS):** Ollama (Llama-3.1-8B-Instruct)
* **Cloud Inference:** Groq API (Llama-3.1-8B-Instruct)
* **Deployment System:** Docker + Google Cloud Run
* **Data Pipelines:** Pandas, NumPy
* **Market Data APIs:** `yfinance` (Quantitative), SEC EDGAR API (Fundamental), Yahoo Finance RSS (News/Event-Driven)
* **Vector Database:** ChromaDB (for RAG on SEC Filings)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **HTTP API:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) (Docker-ready; container not yet added)
* **Observability:** [LangSmith](https://smith.langchain.com/) tracing (optional, via `.env`)

---

##  The Planner-Executor Model

This project abandons fragile "chat loops" in favor of a robust, deterministic **State Machine** architecture:

1. **The Planner:** Analyzes the user's query and generates a strict JSON array of required tasks.
2. **The Supervisor:** A Python-controlled traffic cop that reads the task queue and routes work to the appropriate specialist agents without LLM cognitive overload.
3. **The Workers:** Four highly specialized LangChain React Agents:
   * **Quant_Agent:** Fetches live pricing, volume, and volatility metrics.
   * **Fundamental_Agent:** Fetches live SEC XBRL accounting metrics and performs RAG (Retrieval-Augmented Generation) on 10-K filings for qualitative insights.
   * **Sentiment_Agent:** Analyzes recent news headlines to compute market sentiment scores.
   * **Earnings_Agent:** Performs earnings-call analysis via pre-ingested transcripts — sentiment divergence (Prepared Remarks vs Q&A), keyword/entity trend tracking, and targeted retrieval of management commentary.

---

## Key Features

* **Zero-Hallucination Guardrails:** Worker nodes programmatically verify that a `ToolMessage` was successfully executed before allowing the LLM to output an answer. "Creative guesses" are blocked.
* **Advanced SEC XBRL Parsing:** Handles the messy reality of SEC EDGAR data with Pandas-driven datetime sorting, deduplication, and recursive GAAP tag fallbacks (e.g., gracefully handling `RevenueFromContract...` vs. `Revenues`).
* **On-the-Fly Vector Databases:** Automatically downloads, cleans (HTML Regex isolation), chunks, and embeds corporate 10-K filings into a local ChromaDB instance the moment a user asks about corporate risks or supply chains.
* **Earnings-Call Analysis:** Two-pipeline architecture — an offline ingest pipeline fetches and segments transcripts (Financial Modeling Prep / SEC 8-K), while the runtime inference pipeline uses RAG retrieval with section (Prepared Remarks / Q&A) and quarter filters. Handles SEC-8 (8-K) fallbacks gracefully even when Q&A is missing.

---

## Development Milestones

### Phase 1: Foundation & Hardware Agnostic Setup ✅
*Objective: Establish a local development environment that seamlessly ports to cloud GPUs.*
- [x] Set up a virtual environment and define dependencies.
- [x] Install and configure Ollama for local Mac (MPS) acceleration.
- [x] Build the "Quant Tool": A robust Python function using `yfinance` and `pandas` to fetch historical data.
- [x] Connect the LLM to the tool and verify reliable function-calling with defensive prompt engineering.

### Phase 2: Fundamental Analysis & RAG Pipeline ✅
*Objective: Enable the agent to read and summarize massive, unstructured corporate documents.*
- [x] Integrate the SEC EDGAR API to programmatically download 10-K filings, handling strict rate limits and headers.
- [x] Integrate SEC XBRL endpoints for highly structured GAAP accounting metrics.
- [x] Build a text-chunking pipeline using BeautifulSoup to clean HTML/base64 from SEC filings.
- [x] Set up a local Vector Database (ChromaDB) and use a local HuggingFace embedding model.
- [x] Create a "Fundamental Tool": A RAG-based retriever allowing the LLM to query specific business risks.

### Phase 3: Event-Driven Sentiment Pipeline ✅
*Objective: Enable the agent to react to breaking news and score market sentiment.*
- [x] Integrate a real-time XML RSS feed to fetch the latest headlines for a given ticker.
- [x] Build a "Sentiment Tool": A function that passes recent headlines to the LLM to gauge market catalysts.
- [x] Update the system prompt to enforce strict LLM behavior (Sentiment Scoring from -1.0 to 1.0).

### Phase 4: Multi-Agent Graph Orchestration ✅
*Objective: Move from a single zero-shot agent to a deterministic, cyclic multi-agent system.*
- [x] Implement **LangGraph** to define the system's State (shared memory).
- [x] Build the **Planner & Supervisor Agents**: Parse user queries and route tasks sequentially.
- [x] Build the **Quant, Fundamental, and Sentiment Agents** as isolated execution nodes.

### Phase 5: UI & Cloud Deployment ⏳ *(In Progress)*
*Objective: Synthesize outputs, add a frontend, and deploy for production.*
- [x] Build the **Summary Node** to compile agent outputs into a unified Investment Memo.
- [x] Expose the graph via **FastAPI** (`GET /health`, `POST /chat`) with env-driven LLM settings (OpenAI-compatible endpoint, e.g. Ollama).
- [x] Refactor: shared **graph** + **runner** modules so CLI and API use the same execution path (no `input()` on the server).
- [x] Build a **Streamlit UI** for an interactive web chatbot experience.
- [x] Containerize both FastAPI and Streamlit using **Docker** (`docker-compose`).
- [x] Migrate LLM inference to **Groq API** (Llama-3.1-8B) for fast, robust serverless inference.
- [ ] Deploy the Docker application to **Google Cloud Run** for a scalable portfolio showcase.

### Phase 6: Earnings Call Analysis ✅
*Objective: Add LLM-driven earnings-call analysis with a free-first data pipeline.*
- [x] Build two-pipeline architecture: offline ingest + runtime inference.
- [x] Implement transcript fetching (Financial Modeling Prep free tier / SEC 8-K fallback).
- [x] Add transcript normalization and section segmentation (Prepared Remarks vs Q&A).
- [x] Build keyword/entity frequency extraction with quarter-over-quarter tracking.
- [x] Create three inference `@tool` functions: search, sentiment divergence, keyword trends.
- [x] Wire `Earnings_Agent` into the LangGraph planner/supervisor/summarizer.
- [x] Update Streamlit UI with earnings-call example and data indicators.
- [x] Add CLI ingest script (`scripts/ingest_earnings_calls.py`).

### Phase 7: Benchmarking & Quantitative Evaluation ✅
*Objective: Prove system efficacy through rigorous comparison against zero-shot LLM baselines using anchored ground truth.*
- [x] Generate a 43-item "Golden Dataset" (Numeric, Risk, News, Earnings).
- [x] Build an Evaluation Runner with **Baseline vs. Agent** modes and resume support.
- [x] Implement a **Cross-Model Judge** (Llama-3.3-70B) and check against anchored 10-K quotes.
- [x] Achieve a **+40% Accuracy Lift** over vanilla LLM performance on verifiable facts.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/devg24/financial-analysis-agent
   cd FinAgent
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv agent_env
   source agent_env/bin/activate  # On Windows: agent_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   # Edit .env: OPENAI_* for LLM endpoint; LangSmith vars if you want traces.
   ```

5. **Ensure the LLM server is reachable** (default assumes local Ollama):
   ```bash
   ollama run llama3.1
   ```

6. **Run the Application:**

   **Option A: Full Stack via Docker (Recommended)**
   ```bash
   docker compose up --build
   ```
   *The interactive Streamlit UI will be available at `http://localhost:8501`.*

   **Option B: Local Streamlit Testing (No Docker)**
   Open two terminal windows. In the first, start the API backend:
   ```bash
   uvicorn backend.api:app --host 127.0.0.1 --port 8000
   ```
   In the second terminal, start the Streamlit frontend:
   ```bash
   streamlit run frontend/streamlit_app.py --server.port 8501
   ```
   *Navigate to `http://localhost:8501` to use the chatbot.*

## Project Layout

| Directory/File | Role |
|------|------|
| `backend/api.py` | FastAPI app: exposes LangGraph via `/chat` and `/chat/stream`. |
| `frontend/streamlit_app.py` | Streamlit interactive web frontend with real-time SSE streaming. |
| `core/graph_builder.py` | LangGraph state, nodes, agents, `build_financial_graph()`. |
| `core/runner.py` | `create_llm()`, `run_financial_query()`, and the SSE generator. |
| `core/config.py` | `pydantic-settings` configuring the LLM endpoints via `.env`. |
| `core/*_tools.py` | Underlying specialist worker tools given to the graph agents. |
| `core/earnings_tools.py` | Earnings-call ingest + inference tools (fetch, segment, embed, RAG, trends). |
| `scripts/ingest.py` | CLI tool to scrape, chunk, and embed SEC 10-K filings into vector DB. |
| `scripts/ingest_earnings_calls.py` | CLI tool to ingest earnings-call transcripts into vector DB. |
| `scripts/main.py` | Minimal CLI alternative to the web UI. |
| `docker-compose.yml`, `Dockerfile.*` | Container orchestration for deployment and local testing. |

## Usage Examples

The agent can handle single metrics, multi-ticker comparisons, and qualitative deep dives:

* *"How is Apple's stock doing?"*
* *"What is the market sentiment on Microsoft and Tesla?"*
* *"What are the manufacturing risks mentioned in Tesla's latest 10-K?"*
* *"Give me general information about the stock of Apple and Google."*
* *"Analyze the latest earnings call for Apple — compare management tone in prepared remarks vs Q&A."*
* *"Show keyword trends across Apple's recent earnings calls."*

> ```bash
> python scripts/ingest_earnings_calls.py --tickers AAPL --quarters Q4-2024 Q1-2025
> ```
> *Note: SEC 8-K (sec-8) filings are supported as a fallback when FMP results are unavailable, focusing on Prepared Remarks if Q&A is missing.*

## 📊 Benchmarks & Evaluation

We evaluated FinAgent using a **43-item Golden Dataset** (Benchmark V2) comparing its performance to a Vanilla LLM (Llama-4-Scout) without access to tools. Unlike standard benchmarks, our qualitative questions are **anchored to specific, verifiable quotes** from the source documents to prevent hallucination-rewarding.

| Category | Metric | Baseline (Zero-Shot) | FinAgent | Lift |
| :--- | :--- | :--- | :--- | :--- |
| **Numeric** | Exact XBRL Extraction | 0% | **100%** | **+100%** |
| **Fundamental** | Risk Retrieval (Anchored RAG) | 60% | **77%** | **+17%** |
| **News** | Nuanced Sentiment Analysis | 80% | **60%** | -20%* |
| **Earnings** | transcript Q&A Analysis | 0% | **67%** | **+67%** |
| **Overall** | **Weighted Accuracy** | **40%** | **80%** | **+40%** |

*\*Note: The agent is more conservative on news sentiment as it fetches live data via RSS, occasionally failing to match a specific outdated headline, whereas the baseline uses parametric memory.*

### Key Insights
*   **Zero-Hallucination Numeric**: FinAgent achieved **100% accuracy** on numeric retrieval across 15 complex XBRL tags. The baseline failed every item, proving that tool-use is mandatory for financial grounding.
*   **Verifiable Grounding**: By anchoring risk questions to specific 10-K sentences, we eliminated "vague pass" scores. The agent's **+17% lift** represents real, documented facts that the baseline could not "hallucinate" successfully.
*   **Multi-Agent Synergies**: The agent correctly synthesized earnings call Q&A segments (67% accuracy) that the baseline had no visibility into.

### Running the Evaluation
For a deep-dive into our scientific approach, Anchored Ground Truth, and cross-model judging, see our **[Technical Evaluation Methodology](evaluation/evaluation_methodology.md)**.

```bash
# 1. Run Baseline
python3 evaluation/run_eval.py --mode baseline

# 2. Run FinAgent Execution
python3 evaluation/run_eval.py --mode agent --resume
```
