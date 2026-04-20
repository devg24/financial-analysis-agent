"""
Extract specific risk sentences from the actual 10-K ChromaDB vector stores
to serve as ground-truth expected answers for the benchmark.
"""
import os, sys
sys.path.insert(0, os.getcwd())

from core.rag_tools import get_10k_vector_db

RISK_QUERIES = [
    ("AAPL", "competition", "competitive pressure rivals market share"),
    ("AAPL", "regulation", "regulatory laws compliance government"),
    ("AAPL", "supply_chain", "supply chain manufacturing components suppliers"),
    ("MSFT", "competition", "competitive pressure rivals market share"),
    ("MSFT", "regulation", "regulatory laws compliance government"),
    ("GOOGL", "regulation", "regulatory antitrust laws compliance government"),
    ("NVDA", "supply_chain", "supply chain manufacturing components foundry"),
    ("NVDA", "competition", "competitive pressure rivals market share"),
    ("NVDA", "regulation", "regulatory export controls government compliance"),
    ("TSLA", "supply_chain", "supply chain battery manufacturing components"),
    ("TSLA", "competition", "competitive pressure rivals EV market share"),
    ("AMZN", "competition", "competitive pressure rivals market share"),
    ("META", "regulation", "regulatory privacy laws compliance government"),
    ("AMD", "supply_chain", "supply chain manufacturing foundry TSMC"),
    ("INTC", "supply_chain", "supply chain manufacturing foundry components"),
]

def extract_key_sentence(text: str, max_len: int = 200) -> str:
    """Extract the most informative sentence from a chunk."""
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 30]
    if not sentences:
        return text[:max_len]
    # Pick the longest sentence that's still under max_len (usually the most specific)
    candidates = [s for s in sentences if len(s) < max_len]
    if candidates:
        return max(candidates, key=len) + "."
    return sentences[0][:max_len] + "."

for ticker, risk_type, query in RISK_QUERIES:
    try:
        db = get_10k_vector_db(ticker)
        results = db.similarity_search(query, k=3)
        if results:
            # Get the most relevant chunk and extract a key sentence
            best_chunk = results[0].page_content
            key_sentence = extract_key_sentence(best_chunk)
            print(f"RISK|{ticker}|{risk_type}|{key_sentence}")
        else:
            print(f"RISK|{ticker}|{risk_type}|NO_RESULTS")
    except Exception as e:
        print(f"RISK|{ticker}|{risk_type}|ERROR: {e}")
