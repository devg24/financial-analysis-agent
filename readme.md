# Autonomous Financial AI Agent 📈🤖

An asynchronous, multi-agent LLM pipeline designed to automate quantitative financial research, fundamental document synthesis, and real-time news sentiment analysis. Built entirely with open-source models, this system leverages graph-based agentic workflows to reason through market data, corporate filings, and daily news cycles.

## 🏗 Architecture & Tech Stack

* **Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph) / LangChain
* **Local Inference (Apple MPS):** Ollama (Llama-3.1-8B-Instruct)
* **Cloud Inference (GCP L4):** vLLM 
* **Data Pipelines:** Pandas, NumPy
* **Market Data APIs:** `yfinance` (Quantitative), SEC EDGAR API (Fundamental), Yahoo Finance RSS (News/Event-Driven)
* **Vector Database:** ChromaDB (for RAG on SEC Filings)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Deployment:** Docker, FastAPI

---

## 🗺 Development Milestones

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

### Phase 4: Multi-Agent Graph Orchestration ⏳ *(Up Next)*
*Objective: Move from a single zero-shot agent to a deterministic, cyclic multi-agent system.*
- [ ] Implement **LangGraph** to define the system's State (shared memory).
- [ ] Build the **Supervisor Agent**: Parses user queries and routes tasks.
- [ ] Build the **Quant, Fundamental, and Sentiment Agents** as isolated nodes.

### Phase 5: Cloud Migration & MLOps (GCP L4)
- [ ] Containerize the application using Docker.
- [ ] Provision a GCP Compute Engine instance with an Nvidia L4 GPU.
- [ ] Deploy vLLM and expose the multi-agent system via FastAPI.