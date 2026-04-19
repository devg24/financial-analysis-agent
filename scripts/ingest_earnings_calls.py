#!/usr/bin/env python3
"""
CLI script to ingest earnings-call transcripts into ChromaDB.

Usage:
    python scripts/ingest_earnings_calls.py --tickers AAPL MSFT --quarters Q4-2024 Q1-2025
    python scripts/ingest_earnings_calls.py --tickers TSLA --quarters Q1-2025

Data sources (tried in order):
    1. Financial Modeling Prep (FMP) (free tier, 250 req/day)
    2. SEC EDGAR 8-K filings (free, always available)
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on sys.path so `core.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import Settings
from core.earnings_tools import ingest_earnings_call, parse_quarter


def main():
    parser = argparse.ArgumentParser(
        description="Ingest earnings-call transcripts into ChromaDB."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Stock tickers to ingest (e.g. --tickers AAPL MSFT)",
    )
    parser.add_argument(
        "--quarters",
        nargs="+",
        required=True,
        help="Quarters to ingest, format Q<N>-<YYYY> (e.g. --quarters Q4-2024 Q1-2025)",
    )
    args = parser.parse_args()

    settings = Settings()
    api_key = settings.fmp_api_key or os.getenv("FMP_API_KEY", "")
    chroma_path = settings.earnings_chroma_path

    os.makedirs(chroma_path, exist_ok=True)

    # Parse quarters upfront to fail fast on bad formats
    parsed_quarters: list[tuple[int, int]] = []
    for q_str in args.quarters:
        try:
            q, y = parse_quarter(q_str)
            parsed_quarters.append((q, y))
        except ValueError as e:
            print(f"[Error] {e}")
            sys.exit(1)

    results: list[dict] = []

    for ticker in args.tickers:
        ticker = ticker.upper()
        for quarter, year in parsed_quarters:
            print(f"\n{'=' * 50}")
            print(f"Ingesting {ticker} Q{quarter}-{year}")
            print(f"{'=' * 50}")
            try:
                status = ingest_earnings_call(
                    ticker=ticker,
                    quarter=quarter,
                    year=year,
                    api_key=api_key,
                    chroma_path=chroma_path,
                )
            except Exception as e:
                print(f"[Error] Failed to ingest {ticker} Q{quarter}-{year}: {e}")
                status = "error"

            results.append(
                {"ticker": ticker, "quarter": f"Q{quarter}-{year}", "status": status}
            )

    # Summary
    print(f"\n{'=' * 50}")
    print("INGEST SUMMARY")
    print(f"{'=' * 50}")
    for r in results:
        icon = {
            "success": "✅",
            "partial": "🟡",
            "failed": "❌",
            "exists": "⏭️",
            "error": "💥",
        }.get(r["status"], "❓")
        print(f"  {icon}  {r['ticker']} {r['quarter']}: {r['status']}")

    errors = [r for r in results if r["status"] == "error"]
    failed = [r for r in results if r["status"] == "failed"]

    if errors:
        print(f"\n[CRITICAL] {len(errors)} ingest(s) hit technical errors. Check logs.")
        sys.exit(1)

    if failed:
        print(f"\n[INFO] {len(failed)} transcript(s) could not be found (likely not yet reported).")
        print("This is not treated as a build failure.")
    
    print("\nIngestion process completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
