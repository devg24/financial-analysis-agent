---
title: FinAgent - Autonomous Financial AI
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 📈 FinAgent: Autonomous Financial AI

An asynchronous, multi-agent LLM pipeline that automates quantitative financial research, fundamental document synthesis, earnings-call analysis, and real-time news sentiment scoring — built entirely with open-source models.

## 🏗️ Architecture

This system uses a **deterministic state-machine** architecture powered by [LangGraph](https://python.langchain.com/docs/langgraph):

1. **Planner Agent** — Parses the user query and generates a strict JSON task queue.
2. **Supervisor** — A Python-controlled router that dispatches tasks to specialist agents.
3. **Specialist Agents:**
   - 🔢 **Quant Agent** — Live pricing, volume, and volatility metrics via `yfinance`.
   - 📊 **Fundamental Agent** — SEC XBRL accounting data + RAG on 10-K filings.
   - 📰 **Sentiment Agent** — Real-time news headline analysis and scoring.
   - 🎙️ **Earnings Agent** — Sentiment divergence (Prepared Remarks vs Q&A) and keyword trend tracking from earnings-call transcripts.
4. **Summarizer** — Compiles all agent outputs into a unified Investment Memo.

## 🚀 Try It

Type a query in the chat box — here are some examples:

| Query | What It Does |
|-------|-------------|
| *"How is Apple's stock doing?"* | Quant analysis (price, volume, RSI) |
| *"What are the manufacturing risks in Tesla's latest 10-K?"* | RAG retrieval on SEC filings |
| *"What is the market sentiment on Microsoft?"* | Real-time news sentiment scoring |
| *"Analyze the latest earnings call for AAPL — compare management tone in prepared remarks vs Q&A"* | Earnings-call divergence analysis |
| *"Compare the current stock performance of Microsoft and Google"* | Multi-ticker parallel analysis |

## 📚 Pre-Loaded Data

This demo comes with pre-ingested data for immediate use:

- **SEC 10-K Filings:** AAPL, MSFT, TSLA, GOOGL, NVDA
- **Earnings Call Transcripts:** AAPL, MSFT (Q4-2024, Q1-2025)

> Quantitative data (prices, volume) and sentiment (news) are fetched **live** — no pre-loading needed.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph / LangChain |
| LLM Inference | Groq API (Llama-3.1-8B-Instruct) |
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Vector DB | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Market Data | yfinance, SEC EDGAR API |

## ⚡ Performance Optimizations

This system was deliberately engineered for low-latency response times:

- **Parallel Agent Dispatch** — The Supervisor routes independent tasks to multiple specialist agents simultaneously (e.g., Quant + Sentiment + Fundamental in one batch) rather than sequentially, cutting multi-agent latency by up to 3×.
- **Server-Sent Event (SSE) Streaming** — Results stream live to the UI as each agent completes, so users see intermediate progress immediately instead of waiting for the full pipeline.
- **Groq Cloud Inference** — LLM calls use the Groq API (~200 tok/s on Llama-3.1-8B), eliminating local GPU bottlenecks and delivering sub-second per-agent response times.
- **Singleton Embedding Cache** — The HuggingFace embedding model is loaded once via `@lru_cache` and shared across all RAG queries (10-K, earnings, etc.), avoiding repeated 500MB+ model re-initialization.
- **Token Budget Tuning** — `max_tokens` is capped at 800 per LLM call to prevent Groq from reserving excessive context window, reducing queue wait times by ~40%.
- **Pre-Seeded Vector DB** — ChromaDB collections are embedded at Docker build time, so the app starts with zero cold-start ingestion delay.
- **Per-Step Latency Tracking** — Every agent step reports wall-clock latency in the UI, making performance bottlenecks immediately visible.

## 📂 Source Code

[GitHub Repository](https://github.com/devg24/financial-analysis-agent)
