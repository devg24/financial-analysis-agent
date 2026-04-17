# Changelog

All notable changes to this project will be documented in this file.

## [Phase 4] - Multi-Agent LangGraph Integration & SEC Pipeline Hardening

### Architecture Overhaul
* **Transitioned to LangGraph:** Replaced the legacy `while` loop with a deterministic StateGraph "Planner-Executor" architecture.
* **Dual-Node Governance:** Separated routing logic into a stateless `Planner` (generates JSON task arrays) and a stateful `Supervisor` (manages task queues), eliminating LLM cognitive overload and infinite routing loops.
* **Separation of Concerns:** Split worker capabilities into three strict React Agents: `Quant_Agent`, `Fundamental_Agent`, and `Sentiment_Agent`.

### Added
* **The "Honesty Guardrail":** Implemented programmatic checks in `make_worker_node` to verify `ToolMessage` execution. If an agent attempts to answer without triggering a tool, the output is blocked to prevent hallucination.
* **Strict Capability Matrix:** Updated the Planner prompt to explicitly map query types (e.g., "Risks", "Supply Chain") to specific RAG tool workflows.
* **Broad Query Protocol:** Added fallback logic for the Planner to execute standard Quant/Sentiment tasks when users ask for "general info" on a ticker.
* **Pydantic Enum Enforcement:** Added strict `args_schema` to SEC tools using `Literal` types to prevent the LLM from hallucinating invalid XBRL tags.

### Fixed
* **The SEC "2010 Bug":** The SEC API returns unordered historical data. Added Pandas-based datetime sorting, filing deduplication, and a 2-year lookback filter to ensure only modern data is served.
* **The SEC "Missing Revenue" Bug:** Implemented recursive fallback logic to try `RevenueFromContractWithCustomerIncludingAssessedTax` if the standard `Revenues` GAAP tag returns 404 (fixing data retrieval for MSFT, AAPL, etc.).
* **ChromaDB Deprecation:** Updated imports to `langchain_chroma` and improved SEC HTML `<DOCUMENT>` regex parsing for cleaner 10-K embeddings.

## [Phase 5] - API Streaming, Web UI & Containerization

### UI & Architecture
* **FastAPI Backend:** Exposed the LangGraph state machine via an asynchronous `GET /chat/stream` utilizing Server-Sent Events (SSE).
* **Streamlit Frontend:** Built a responsive agentic UI (`streamlit_app.py`) that visually streams the intermediate ReAct reasoning blocks into dynamically expanding dropdown menus.
* **State Persistence:** Rewrote the UI memory loop to preserve "Agent Thoughts" sequentially so historical messages contain dropdown logs linking exactly to how the agents generated their specific conclusions.
* **Docker Migration:** Shipped `Dockerfile.api` and `Dockerfile.ui` bridged via a custom network inside `docker-compose.yml`, successfully moving the application payload off MacOS and onto standardized, immutable infrastructure.
* **LLM Engine Swap:** Migrated the brain architecture permanently to Groq's `llama-3.1-8b-instant` and `llama3-70b-versatile` endpoints for incredibly fast serverless inference.

### Performance & Token Thrashing Fixes
* **LRU Caching:** Prevented the `SentenceTransformers` model from re-instantiating on disk during every RAG pipeline call, silencing repetitive console logs and saving heavy JVM/Python CPU overhead.
* **Groq 'Token Ghosting' Fix:** Squashed a massive token quota error ("Requested 17k tokens") by explicitly declaring `max_tokens=800`. This prevented Groq's load balancer from assuming maximum context window limits and throttling free-tier TPM budgets.
* **ReAct Infinite Loop Resolution:** The Fundamental agent historically trapped itself looping `search_10k_filings` tools when ordered to strictly "only output data". Bridged the behavior by injecting a hard "Stop once data is fetched" logic trap inside the system prompt.
* **Docker Hot-Reloading:** Injected `- ./:/app` mapping masks and explicitly overrode the Uvicorn execution loop with `--reload` to support real-time Python development without tearing down the containers.

### Next Steps 🚀
* Final push of the `finagent` backend/frontend images to **Google Cloud Run**.