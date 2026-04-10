# Autonomous Financial AI Agent 

An asynchronous, multi-agent LLM pipeline designed to automate quantitative financial research, fundamental document synthesis, and real-time news sentiment analysis. Built entirely with open-source models, this system leverages graph-based agentic workflows to reason through market data, corporate filings, and daily news cycles.

## Architecture & Tech Stack

* **Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph) / LangChain
* **Local Inference (Apple MPS):** Ollama (Llama-3.1-8B-Instruct)
* **Cloud Inference (GCP L4):** vLLM 
* **Data Pipelines:** Pandas, NumPy
* **Market Data APIs:** `yfinance` (Quantitative), SEC EDGAR API (Fundamental), Yahoo Finance RSS (News/Event-Driven)
* **Vector Database:** ChromaDB (for RAG on SEC Filings)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Deployment:** Docker, FastAPI

---

##  The Planner-Executor Model

This project abandons fragile "chat loops" in favor of a robust, deterministic **State Machine** architecture:

1. **The Planner:** Analyzes the user's query and generates a strict JSON array of required tasks.
2. **The Supervisor:** A Python-controlled traffic cop that reads the task queue and routes work to the appropriate specialist agents without LLM cognitive overload.
3. **The Workers:** Three highly specialized LangChain React Agents:
   * **Quant_Agent:** Fetches live pricing, volume, and volatility metrics.
   * **Fundamental_Agent:** Fetches live SEC XBRL accounting metrics and performs RAG (Retrieval-Augmented Generation) on 10-K filings for qualitative insights.
   * **Sentiment_Agent:** Analyzes recent news headlines to compute market sentiment scores.

---

## Key Features

* **Zero-Hallucination Guardrails:** Worker nodes programmatically verify that a `ToolMessage` was successfully executed before allowing the LLM to output an answer. "Creative guesses" are blocked.
* **Advanced SEC XBRL Parsing:** Handles the messy reality of SEC EDGAR data with Pandas-driven datetime sorting, deduplication, and recursive GAAP tag fallbacks (e.g., gracefully handling `RevenueFromContract...` vs. `Revenues`).
* **On-the-Fly Vector Databases:** Automatically downloads, cleans (HTML Regex isolation), chunks, and embeds corporate 10-K filings into a local ChromaDB instance the moment a user asks about corporate risks or supply chains.

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

### Phase 5: Synthesis, Cloud Migration & MLOps ⏳ *(In Progress)*
*Objective: Synthesize outputs and deploy for production.*
- [x] Build the **Summary Node** to compile agent outputs into a unified Investment Memo.
- [ ] Containerize the application using Docker.
- [ ] Provision a GCP Compute Engine instance with an Nvidia L4 GPU.
- [ ] Deploy vLLM and expose the multi-agent system via FastAPI.

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

4. **Ensure Ollama is running locally:**
   ```bash
   ollama run llama3.1
   ```

5. **Run the Orchestrator:**
   ```bash
   python main.py
   ```

## Usage Examples

The agent can handle single metrics, multi-ticker comparisons, and qualitative deep dives:

* *"How is Apple's stock doing?"*
* *"What is the market sentiment on Microsoft and Tesla?"*
* *"What are the manufacturing risks mentioned in Tesla's latest 10-K?"*
* *"Give me general information about the stock of Apple and Google."*
