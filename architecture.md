# 🏗️ FinAgent System Architecture

This diagram visualizes the flow of data and control through the multi-agent system.

```mermaid
graph TD
    subgraph UI ["User Interface"]
        User((User)) --> Streamlit[Streamlit App]
    end

    subgraph API ["Orchestration Layer (FastAPI/LangGraph)"]
        Streamlit --> Router[FastAPI /chat/stream]
        Router --> Planner[Planner Agent]
        
        subgraph Graph ["LangGraph State Machine"]
            Planner -- "Generates Task List" --> Supervisor{Supervisor Agent}
            
            Supervisor --> Quant[Quant Agent]
            Supervisor --> Fund[Fundamental Agent]
            Supervisor --> Sent[Sentiment Agent]
            Supervisor --> Earn[Earnings Agent]
            
            Quant --> Summary
            Fund --> Summary
            Sent --> Summary
            Earn --> Summary
            
            Summary[Summarizer / Investment Memo]
        end
    end

    subgraph Tools ["Specialist Tools / Data Sources"]
        Quant --> yf[yfinance API]
        Fund --> SEC[SEC EDGAR / XBRL]
        Fund --> Chroma1[(ChromaDB: 10-K RAG)]
        Sent --> RSS[Yahoo Finance RSS]
        Earn --> FMP[Earnings Ingest: 8-K/FMP]
        Earn --> Chroma2[(ChromaDB: Transcript RAG)]
    end

    subgraph Eval ["Validation Pipeline"]
        Golden[(Golden Dataset)] --> Scorer[Evaluation Runner]
        Graph -. "Audit Output" .-> Scorer
        Scorer -- "Cross-Model Critique" --> Judge[[Llama-3.3-70B Judge]]
    end

    classDef agent fill:#f9f,stroke:#333,stroke-width:2px;
    classDef tool fill:#bbf,stroke:#333,stroke-width:1px;
    classDef ui fill:#dfd,stroke:#333,stroke-width:1px;
    
    class Planner,Supervisor,Quant,Fund,Sent,Earn,Summary agent;
    class yf,SEC,Chroma1,RSS,FMP,Chroma2 tool;
    class Streamlit ui;
```

---

### Component Breakdown

1.  **The Planner (LLM)**: Deconstructs raw natural language into a structured JSON task plan.
2.  **The Supervisor (Python)**: Decouples the LLM from routing logic to ensure deterministic execution and prevent "agent infinite loops."
3.  **Worker Agents (ReAct)**: Specialized nodes with specific tool-access:
    *   **Quant**: Financial metrics and time-series data.
    *   **Fundamental**: RAG over 10-K filings + XBRL tag extraction.
    *   **Sentiment**: Real-time RSS news analysis.
    *   **Earnings**: Q&A vs. Prepared Remarks divergence analysis.
4.  **ChromaDB**: Local vector store providing context for RAG operations.
5.  **Validation Pipeline**: An independent audit layer that measures agent precision against "Anchored Ground Truth."
