# Autonomous Financial AI Agent 📈🤖

An asynchronous, multi-agent LLM pipeline designed to automate quantitative financial research, fundamental document synthesis, and portfolio analysis. Built entirely with open-source models, this system leverages graph-based agentic workflows to reason through market data and corporate filings.

## 🏗 Architecture & Tech Stack

* **Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph) / LangChain
* **Local Inference (Apple MPS):** Ollama (Llama-3.1-8B-Instruct)
* **Cloud Inference (GCP L4):** vLLM 
* **Data Pipelines:** Pandas, NumPy
* **Market Data APIs:** `yfinance` (Quantitative), SEC EDGAR API (Fundamental)
* **Vector Database:** ChromaDB / Qdrant (for RAG on SEC Filings)
* **Deployment:** Docker, FastAPI

---

## 🗺 Development Milestones

This project is structured into progressive phases, moving from a local single-agent proof-of-concept to a cloud-deployed, multi-agent trading research system.

### Phase 1: Foundation & Hardware Agnostic Setup 
*Objective: Establish a local development environment that seamlessly ports to cloud GPUs.*
- [ ] Set up a virtual environment and define dependencies (`requirements.txt` or `poetry`).
- [ ] Install and configure Ollama for local Mac (MPS) acceleration.
- [ ] Implement a base `ChatOpenAI` client pointing to the local Ollama server to ensure future compatibility with vLLM.
- [ ] Build the first "Quant Tool": A robust Python function using `yfinance` and `pandas` to fetch historical data and calculate basic indicators (e.g., 50-day SMA, Volatility).
- [ ] Connect the LLM to the tool and verify reliable function-calling.

### Phase 2: Fundamental Analysis & RAG Pipeline
*Objective: Enable the agent to read and summarize massive, unstructured corporate documents.*
- [ ] Integrate the SEC EDGAR API to programmatically download 10-K and 10-Q filings for a given ticker.
- [ ] Build a text-chunking pipeline to clean HTML/text from SEC filings.
- [ ] Set up a local Vector Database (ChromaDB) and use an open-source embedding model (e.g., HuggingFace `all-MiniLM-L6-v2`) to store document embeddings.
- [ ] Create a "Fundamental Tool": A RAG-based retriever that allows the LLM to query specific financial metrics (e.g., "What were the primary risk factors mentioned in Apple's latest 10-K?").

### Phase 3: Multi-Agent Graph Orchestration
*Objective: Move from a single zero-shot agent to a deterministic, cyclic multi-agent system.*
- [ ] Implement **LangGraph** to define the system's State (shared memory).
- [ ] Build the **Supervisor Agent**: Parses user queries and routes tasks.
- [ ] Build the **Quant Agent**: Specialized in writing/executing Pandas code for market data.
- [ ] Build the **Fundamental Agent**: Specialized in querying the Vector DB for SEC insights.
- [ ] Build the **Synthesizer Agent**: Combines quantitative data and fundamental context into a cohesive markdown investment brief.

### Phase 4: Cloud Migration & MLOps (GCP L4)
*Objective: Deploy the system to a production-grade cloud environment.*
- [ ] Containerize the application using Docker (`Dockerfile` and `docker-compose.yml`).
- [ ] Provision a GCP Compute Engine instance with an Nvidia L4 GPU.
- [ ] Install CUDA drivers and deploy **vLLM** on the GCP instance to serve the chosen open-source model.
- [ ] Point the application's `base_url` to the vLLM endpoint.
- [ ] Expose the agent via a FastAPI backend endpoint.

### Phase 5: Advanced Features & Backtesting (Stretch Goals)
*Objective: Add Quant-specific features to demonstrate domain expertise.*
- [ ] Give the Quant Agent the ability to utilize `Backtrader` or `vectorbt` to simulate simple trading strategies based on its analysis.
- [ ] Implement memory persistence in LangGraph (using SQLite or Postgres) so the agent remembers previous conversations and portfolio contexts.
- [ ] Add an evaluation framework (e.g., Ragas or LangSmith) to benchmark the accuracy of the agent's RAG retrieval.

---

## 🚀 Getting Started (Local Development)

*(Instructions for cloning the repo, starting Ollama, installing dependencies, and running `main.py` will go here)*