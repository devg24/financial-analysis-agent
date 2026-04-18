# ──────────────────────────────────────────────────────────────────────────────
# FinAgent — Hugging Face Spaces Dockerfile (Docker SDK)
# Runs FastAPI backend + Streamlit frontend in a single container via supervisord
# Pre-seeds ChromaDB with demo tickers at build time
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ curl supervisor && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ───────────────────────────────────────────────────────────────
COPY . .

# ── Pre-seed ChromaDB at build time ─────────────────────────────────────────
# Ingest SEC 10-K filings for demo tickers
RUN python scripts/ingest.py --tickers AAPL MSFT TSLA GOOGL NVDA

# Ingest SEC 8-K / earnings call data for demo tickers
RUN python scripts/ingest_earnings_calls.py --tickers AAPL MSFT --quarters Q4-2024 Q1-2025

# ── Supervisord config (runs both services) ─────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── HF Spaces expects port 7860 ─────────────────────────────────────────────
EXPOSE 7860

# Streamlit health-check endpoint for HF Spaces
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
