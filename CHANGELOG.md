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

### Known Limitations / Next Steps
* Add a `Summary_Node` (Phase 5) to synthesize the sequential agent logs into a single, cohesive Investment Memo.