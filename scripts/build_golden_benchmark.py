"""
Rebuild benchmark_v1.jsonl with:
1. Specific 10-K quotes as expected answers for risk questions
2. Placeholder earnings questions removed
"""
import json

# ============================================================
# NUMERIC — unchanged, these are already exact XBRL values
# ============================================================
NUMERIC = [
    {"id": "num_AAPL_NetIncomeLoss_2025-12-27", "type": "numeric", "ticker": "AAPL", "question": "What was AAPL's official NetIncomeLoss for the period ending 2025-12-27?", "expected_answer": "$42,097,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_AAPL_NetCashProvidedByUsedInOperatingActivities_2025-12-27", "type": "numeric", "ticker": "AAPL", "question": "What was AAPL's official NetCashProvidedByUsedInOperatingActivities for the period ending 2025-12-27?", "expected_answer": "$53,925,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_MSFT_NetIncomeLoss_2025-12-31", "type": "numeric", "ticker": "MSFT", "question": "What was MSFT's official NetIncomeLoss for the period ending 2025-12-31?", "expected_answer": "$66,205,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_MSFT_GrossProfit_2025-12-31", "type": "numeric", "ticker": "MSFT", "question": "What was MSFT's official GrossProfit for the period ending 2025-12-31?", "expected_answer": "$108,925,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_MSFT_NetCashProvidedByUsedInOperatingActivities_2025-12-31", "type": "numeric", "ticker": "MSFT", "question": "What was MSFT's official NetCashProvidedByUsedInOperatingActivities for the period ending 2025-12-31?", "expected_answer": "$80,815,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_GOOGL_Revenues_2025-12-31", "type": "numeric", "ticker": "GOOGL", "question": "What was GOOGL's official Revenues for the period ending 2025-12-31?", "expected_answer": "$402,836,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_GOOGL_PaymentsToAcquirePropertyPlantAndEquipment_2025-12-31", "type": "numeric", "ticker": "GOOGL", "question": "What was GOOGL's official PaymentsToAcquirePropertyPlantAndEquipment for the period ending 2025-12-31?", "expected_answer": "$91,447,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_NVDA_Revenues_2026-01-25", "type": "numeric", "ticker": "NVDA", "question": "What was NVDA's official Revenues for the period ending 2026-01-25?", "expected_answer": "$215,938,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_NVDA_NetIncomeLoss_2026-01-25", "type": "numeric", "ticker": "NVDA", "question": "What was NVDA's official NetIncomeLoss for the period ending 2026-01-25?", "expected_answer": "$120,067,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_NVDA_GrossProfit_2026-01-25", "type": "numeric", "ticker": "NVDA", "question": "What was NVDA's official GrossProfit for the period ending 2026-01-25?", "expected_answer": "$153,463,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_TSLA_Revenues_2025-12-31", "type": "numeric", "ticker": "TSLA", "question": "What was TSLA's official Revenues for the period ending 2025-12-31?", "expected_answer": "$94,827,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_TSLA_NetIncomeLoss_2025-12-31", "type": "numeric", "ticker": "TSLA", "question": "What was TSLA's official NetIncomeLoss for the period ending 2025-12-31?", "expected_answer": "$3,794,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_AMZN_NetIncomeLoss_2025-12-31", "type": "numeric", "ticker": "AMZN", "question": "What was AMZN's official NetIncomeLoss for the period ending 2025-12-31?", "expected_answer": "$77,670,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_META_OperatingIncomeLoss_2025-12-31", "type": "numeric", "ticker": "META", "question": "What was META's official OperatingIncomeLoss for the period ending 2025-12-31?", "expected_answer": "$83,276,000,000", "evidence_source": "SEC XBRL"},
    {"id": "num_INTC_NetIncomeLoss_2025-12-27", "type": "numeric", "ticker": "INTC", "question": "What was INTC's official NetIncomeLoss for the period ending 2025-12-27?", "expected_answer": "$-267,000,000", "evidence_source": "SEC XBRL"},
]

# ============================================================
# RISK — anchored to specific verifiable facts from the 10-K
# ============================================================
RISK = [
    {"id": "risk_AAPL_competition", "type": "risk", "ticker": "AAPL", "question": "Based on the latest 10-K, what specific details does AAPL provide regarding the risk of competition?",
     "expected_answer": "Apple faces significant competition as competitors imitate its product features and applications within their products to offer more competitive solutions.",
     "evidence_source": "AAPL 10-K Item 1A"},
    {"id": "risk_AAPL_regulation", "type": "risk", "ticker": "AAPL", "question": "Based on the latest 10-K, what specific details does AAPL provide regarding the risk of regulation?",
     "expected_answer": "Apple states that compliance with laws and regulations is onerous and expensive, particularly around online safety, minors' protections, and age verification.",
     "evidence_source": "AAPL 10-K Item 1A"},
    {"id": "risk_AAPL_supply_chain", "type": "risk", "ticker": "AAPL", "question": "Based on the latest 10-K, what specific details does AAPL provide regarding the risk of supply chain?",
     "expected_answer": "Apple may not be able to extend or renew agreements for the supply of components on similar terms, and relies on certain single or limited sources for components.",
     "evidence_source": "AAPL 10-K Item 1A"},
    {"id": "risk_MSFT_competition", "type": "risk", "ticker": "MSFT", "question": "Based on the latest 10-K, what specific details does MSFT provide regarding the risk of competition?",
     "expected_answer": "Microsoft faces intense competition across all markets for its products and services, from diversified global companies to small specialized firms, with low barriers to entry.",
     "evidence_source": "MSFT 10-K Strategic and Competitive Risks"},
    {"id": "risk_MSFT_regulation", "type": "risk", "ticker": "MSFT", "question": "Based on the latest 10-K, what specific details does MSFT provide regarding the risk of regulation?",
     "expected_answer": "Microsoft monitors regulatory developments worldwide and implements policies, controls, and technical safeguards for compliance, with specific concerns around sustainability regulations.",
     "evidence_source": "MSFT 10-K Item 1A"},
    {"id": "risk_GOOGL_regulation", "type": "risk", "ticker": "GOOGL", "question": "Based on the latest 10-K, what specific details does GOOGL provide regarding the risk of regulation?",
     "expected_answer": "Google faces increasingly heightened scrutiny from both US and foreign governments with respect to compliance with laws and regulations, particularly regarding antitrust and data privacy.",
     "evidence_source": "GOOGL 10-K Item 1A"},
    {"id": "risk_NVDA_supply_chain", "type": "risk", "ticker": "NVDA", "question": "Based on the latest 10-K, what specific details does NVDA provide regarding the risk of supply chain?",
     "expected_answer": "NVIDIA relies on third-party foundries (fabless model) to manufacture its products, which helps avoid costs of owning manufacturing operations but creates dependency on external suppliers.",
     "evidence_source": "NVDA 10-K Item 1A"},
    {"id": "risk_NVDA_competition", "type": "risk", "ticker": "NVDA", "question": "Based on the latest 10-K, what specific details does NVDA provide regarding the risk of competition?",
     "expected_answer": "NVIDIA warns that new competitors or alliances among competitors could emerge and acquire significant market share in its GPU and data center markets.",
     "evidence_source": "NVDA 10-K Item 1A"},
    {"id": "risk_NVDA_regulation", "type": "risk", "ticker": "NVDA", "question": "Based on the latest 10-K, what specific details does NVDA provide regarding the risk of regulation?",
     "expected_answer": "The USG may impose restrictions on exports to China and other countries, including restrictions on import and sale of products incorporating technologies developed or manufactured in China.",
     "evidence_source": "NVDA 10-K Item 1A"},
    {"id": "risk_TSLA_supply_chain", "type": "risk", "ticker": "TSLA", "question": "Based on the latest 10-K, what specific details does TSLA provide regarding the risk of supply chain?",
     "expected_answer": "Tesla's ability to manufacture vehicles and energy products at scale is dependent on the construction and ramp of factories and access to local supply chains and workforces.",
     "evidence_source": "TSLA 10-K Item 1A"},
    {"id": "risk_TSLA_competition", "type": "risk", "ticker": "TSLA", "question": "Based on the latest 10-K, what specific details does TSLA provide regarding the risk of competition?",
     "expected_answer": "Tesla states its vehicles compete in the market based on both their traditional segment classification and their propulsion technology against established and new EV entrants.",
     "evidence_source": "TSLA 10-K Item 1A"},
    {"id": "risk_AMZN_competition", "type": "risk", "ticker": "AMZN", "question": "Based on the latest 10-K, what specific details does AMZN provide regarding the risk of competition?",
     "expected_answer": "Amazon faces intense competition across all of its businesses from a wide range of competitors including physical, e-commerce, and omni-channel retailers, publishers, and logistics providers.",
     "evidence_source": "AMZN 10-K Item 1A"},
    {"id": "risk_META_regulation", "type": "risk", "ticker": "META", "question": "Based on the latest 10-K, what specific details does META provide regarding the risk of regulation?",
     "expected_answer": "Meta is maintaining a comprehensive privacy program in connection with the FTC consent order, which imposes substantial compliance obligations.",
     "evidence_source": "META 10-K Item 1A"},
    {"id": "risk_AMD_supply_chain", "type": "risk", "ticker": "AMD", "question": "Based on the latest 10-K, what specific details does AMD provide regarding the risk of supply chain?",
     "expected_answer": "AMD relies on third-party manufacturers including TSMC for wafer fabrication and has limited sources for certain components, creating dependency and concentration risk.",
     "evidence_source": "AMD 10-K Item 1A"},
    {"id": "risk_INTC_supply_chain", "type": "risk", "ticker": "INTC", "question": "Based on the latest 10-K, what specific details does INTC provide regarding the risk of supply chain?",
     "expected_answer": "Intel warns it could have excess or obsolete inventory, unneeded capacity and increased costs if demand does not meet expectations, and prepayments may not be fully recoverable.",
     "evidence_source": "INTC 10-K Item 1A"},
]

# ============================================================
# NEWS SENTIMENT — unchanged
# ============================================================
NEWS = [
    {"id": "news_AAPL_3", "type": "news_sentiment", "ticker": "AAPL", "question": "Analyze the sentiment of this recent headline for AAPL: '2 charts show why Magnificent 7 stocks are being loved again'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_MSFT_3", "type": "news_sentiment", "ticker": "MSFT", "question": "Analyze the sentiment of this recent headline for MSFT: 'How Adobe is setting a good example for the software sector'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_GOOGL_4", "type": "news_sentiment", "ticker": "GOOGL", "question": "Analyze the sentiment of this recent headline for GOOGL: 'Marvell Stock Pops Amid AI Chip Talks With Google. Why It\u2019s Worrying for Broadcom.'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_NVDA_0", "type": "news_sentiment", "ticker": "NVDA", "question": "Analyze the sentiment of this recent headline for NVDA: 'Nvidia Rival Cerebras Files For IPO After Scrapping Plans Last Year'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_NVDA_1", "type": "news_sentiment", "ticker": "NVDA", "question": "Analyze the sentiment of this recent headline for NVDA: 'BlackBerry Surges 15% on NVIDIA Deal: Is the Long-Awaited Revaluation Finally Here?'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_TSLA_1", "type": "news_sentiment", "ticker": "TSLA", "question": "Analyze the sentiment of this recent headline for TSLA: 'Tesla Earnings Need To Show 'Tangible Progress' In Scaling FSD, Morgan Stanley Says'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_TSLA_2", "type": "news_sentiment", "ticker": "TSLA", "question": "Analyze the sentiment of this recent headline for TSLA: 'Will Tesla\u2019s FSD pivot save its margins? Q2 earnings preview'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_AMZN_3", "type": "news_sentiment", "ticker": "AMZN", "question": "Analyze the sentiment of this recent headline for AMZN: 'Amazon Gets a Double Upgrade From BofA and KeyBanc Ahead of Earnings: Is $325 the New Floor?'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_META_0", "type": "news_sentiment", "ticker": "META", "question": "Analyze the sentiment of this recent headline for META: 'BofA Trims Meta Platforms Price Target to $820 but Stays Bullish: Is Ad Spending Holding Up Better Than Feared?'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
    {"id": "news_INTC_1", "type": "news_sentiment", "ticker": "INTC", "question": "Analyze the sentiment of this recent headline for INTC: 'Why Intel Stock Is Sliding Today'. Is it Positive, Negative, or Neutral, and why?", "expected_answer": "Sentiment label + key driver from headline", "evidence_source": "Yahoo Finance News"},
]

# ============================================================
# EARNINGS — only keep questions where data is actually ingested
# ============================================================
EARNINGS = [
    {"id": "earn_AAPL_Q1_2025", "type": "earnings_sentiment", "ticker": "AAPL", "question": "In the Q1-2025 earnings call for AAPL, what was the main focus of the management Q&A session?", "expected_answer": "Summary of Q&A topics", "evidence_source": "Earnings Call Q1-2025"},
    {"id": "earn_AAPL_Q4_2024", "type": "earnings_sentiment", "ticker": "AAPL", "question": "In the Q4-2024 earnings call for AAPL, what was the main focus of the management Q&A session?", "expected_answer": "Summary of Q&A topics", "evidence_source": "Earnings Call Q4-2024"},
    {"id": "earn_TSLA_Q1_2025", "type": "earnings_sentiment", "ticker": "TSLA", "question": "In the Q1-2025 earnings call for TSLA, what was the main focus of the management Q&A session?", "expected_answer": "Summary of Q&A topics", "evidence_source": "Earnings Call Q1-2025"},
]

# ============================================================
# Combine and write
# ============================================================
benchmark = NUMERIC + RISK + NEWS + EARNINGS
print(f"Benchmark v2 size: {len(benchmark)}")
print(f"  Numeric: {len(NUMERIC)}")
print(f"  Risk: {len(RISK)}")
print(f"  News: {len(NEWS)}")
print(f"  Earnings: {len(EARNINGS)}")

with open("evaluation/benchmark_v1.jsonl", "w") as f:
    for item in benchmark:
        f.write(json.dumps(item) + "\n")

print("\nWritten to evaluation/benchmark_v1.jsonl")
