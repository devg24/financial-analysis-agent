import os
import json
import random
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from core.sec_tools import get_company_concept_xbrl, get_latest_10k_url
from core.sentiment_tools import get_recent_news
from core.earnings_tools import _load_metadata, get_earnings_sentiment_divergence

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "NFLX", "AMD", "INTC"]
CONCEPTS = [
    "Revenues", "NetIncomeLoss", "Assets", "Liabilities", "GrossProfit", 
    "OperatingIncomeLoss", "AssetsCurrent", "LiabilitiesCurrent", 
    "NetCashProvidedByUsedInOperatingActivities", 
    "PaymentsToAcquirePropertyPlantAndEquipment", 
    "EntityCommonStockSharesOutstanding"
]

def generate_numeric_candidates() -> List[Dict]:
    candidates = []
    print("Generating Numeric Candidates...")
    for ticker in TICKERS:
        for concept in CONCEPTS:
            try:
                # Get real data to formulate the expected answer accurately
                res = get_company_concept_xbrl.invoke({"ticker": ticker, "concept": concept})
                if "Latest official" in res:
                    lines = res.strip().split("\n")
                    # Line 1 is the header: "- Period End: 2024-09-28 (Filed: 2024-10-31): $94,930,000,000"
                    if len(lines) > 1:
                        # Grab the most recent one
                        data_line = lines[1]
                        period = data_line.split("Period End: ")[1].split(" (")[0]
                        value = data_line.split(": ")[-1]
                        
                        candidates.append({
                            "id": f"num_{ticker}_{concept}_{period}",
                            "type": "numeric",
                            "ticker": ticker,
                            "question": f"What was {ticker}'s official {concept} for the period ending {period}?",
                            "expected_answer": value,
                            "evidence_source": "SEC XBRL"
                        })
            except Exception as e:
                print(f"Error generating numeric for {ticker} {concept}: {e}")
    return candidates

def generate_risk_candidates() -> List[Dict]:
    candidates = []
    print("Generating Risk Candidates...")
    for ticker in TICKERS:
        try:
            url = get_latest_10k_url(ticker)
            # We'll create generic risk questions that force the agent to Actually Read the 10-K
            topics = ["competition", "regulation", "supply chain", "currency fluctuations"]
            for topic in topics:
                candidates.append({
                    "id": f"risk_{ticker}_{topic.replace(' ', '_')}",
                    "type": "risk",
                    "ticker": ticker,
                    "question": f"Based on the latest 10-K, what specific details does {ticker} provide regarding the risk of {topic}?",
                    "expected_answer": "Short summary of the specific risk segment in 10-K",
                    "evidence_source": url
                })
        except Exception as e:
            print(f"Error generating risk for {ticker}: {e}")
    return candidates

def generate_news_sentiment_candidates() -> List[Dict]:
    candidates = []
    print("Generating News Sentiment Candidates...")
    for ticker in TICKERS:
        try:
            res = get_recent_news.invoke({"ticker": ticker})
            if "Recent News Headlines" in res:
                lines = [l for l in res.split("\n") if l and l[0].isdigit()]
                for i, line in enumerate(lines[:5]):
                    # Line format: "1. [Date] Headline"
                    headline = line.split("] ")[1] if "] " in line else line
                    candidates.append({
                        "id": f"news_{ticker}_{i}",
                        "type": "news_sentiment",
                        "ticker": ticker,
                        "question": f"Analyze the sentiment of this recent headline for {ticker}: '{headline}'. Is it Positive, Negative, or Neutral, and why?",
                        "expected_answer": "Sentiment label + key driver from headline",
                        "evidence_source": "Yahoo Finance News"
                    })
        except Exception as e:
            print(f"Error generating news for {ticker}: {e}")
    return candidates

def generate_earnings_candidates() -> List[Dict]:
    candidates = []
    print("Generating Earnings Candidates...")
    for ticker in TICKERS:
        try:
            # Check for ingested data
            meta = _load_metadata("./chroma_db", ticker)
            if not meta:
                # Add a dummy or placeholder for now if no data
                candidates.append({
                    "id": f"earn_{ticker}_placeholder",
                    "type": "earnings_sentiment",
                    "ticker": ticker,
                    "question": f"What was the key tone of {ticker}'s most recent earnings call?",
                    "expected_answer": "Requires ingestion of Q1-2025/Q4-2024",
                    "evidence_source": "ChromaDB Transcript"
                })
                continue
            
            for m in meta:
                q, y = m['quarter'], m['year']
                candidates.append({
                    "id": f"earn_{ticker}_Q{q}_{y}",
                    "type": "earnings_sentiment",
                    "ticker": ticker,
                    "question": f"In the Q{q}-{y} earnings call for {ticker}, what was the main focus of the management Q&A session?",
                    "expected_answer": "Summary of Q&A topics",
                    "evidence_source": f"Earnings Call Q{q}-{y}"
                })
        except Exception as e:
            print(f"Error generating earnings for {ticker}: {e}")
    return candidates

def main():
    all_candidates = []
    all_candidates.extend(generate_numeric_candidates())
    all_candidates.extend(generate_risk_candidates())
    all_candidates.extend(generate_news_sentiment_candidates())
    all_candidates.extend(generate_earnings_candidates())
    
    # Pad to 150 if needed with some variations or just more tickers
    # For now let's see how many we have
    print(f"Total candidates generated: {len(all_candidates)}")
    
    with open("evaluation/candidates.jsonl", "w") as f:
        for c in all_candidates:
            f.write(json.dumps(c) + "\n")
    print("Saved to evaluation/candidates.jsonl")

if __name__ == "__main__":
    main()
