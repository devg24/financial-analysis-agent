"""
Earnings-call ingest + inference tools.

Ingest layer  – fetch transcript (Alpha Vantage → SEC 8-K fallback),
               normalize into Prepared Remarks / Q&A segments,
               extract keyword counts, and embed into ChromaDB.

Inference layer – LangGraph @tool functions for retrieval,
                 sentiment divergence, and keyword trend analysis.
"""

import json
import os
import re
from collections import Counter
from typing import Optional

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .rag_tools import get_cached_embeddings
from .sec_tools import HEADERS, get_cik_from_ticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACKED_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning",
    "headwinds", "tailwinds", "guidance", "margin", "growth",
    "inflation", "recession", "tariff", "supply chain",
    "cloud", "capex", "capital expenditure", "free cash flow",
    "buyback", "dividend", "restructuring", "layoff",
    "regulation", "competition", "demand", "inventory",
]

# Markers used to split transcripts into sections
QA_MARKERS = [
    "question-and-answer session",
    "question-and-answer",
    "q&a session",
    "q & a session",
    "operator instructions",
    "and our first question",
    "we will now begin the question",
    "we'll now begin the question",
]

METADATA_DIR_NAME = "_earnings_meta"

# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def parse_quarter(quarter_str: str) -> tuple[int, int]:
    """Parse 'Q1-2025' → (1, 2025). Also accepts 'Q1 2025' or 'q1-2025'."""
    m = re.match(r"[Qq](\d)\s*[-_ ]?\s*(\d{4})", quarter_str.strip())
    if not m:
        raise ValueError(
            f"Invalid quarter format '{quarter_str}'. Expected e.g. 'Q1-2025'."
        )
    q, y = int(m.group(1)), int(m.group(2))
    if q < 1 or q > 4:
        raise ValueError(f"Quarter must be 1-4, got {q}.")
    return q, y


def _quarter_to_month(q: int) -> str:
    """Map fiscal quarter to approximate month for Alpha Vantage API."""
    return {1: "03", 2: "06", 3: "09", 4: "12"}[q]


# ---------------------------------------------------------------------------
# Transcript fetchers
# ---------------------------------------------------------------------------

def fetch_transcript_alpha_vantage(
    ticker: str, quarter: int, year: int, api_key: str
) -> Optional[str]:
    """
    Try the Alpha Vantage EARNINGS_CALL_TRANSCRIPT endpoint.
    Returns raw transcript text or None on failure (premium-only).
    """
    if not api_key:
        return None
    url = (
        "https://www.alphavantage.co/query"
        f"?function=EARNINGS_CALL_TRANSCRIPT"
        f"&symbol={ticker}"
        f"&quarter={year}Q{quarter}"
        f"&apikey={api_key}"
    )
    try:
        print(f"[Earnings Ingest] Trying Alpha Vantage for {ticker} Q{quarter}-{year}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage returns a list of transcript segments on success
        if isinstance(data, dict) and "transcript" in data:
            segments = data["transcript"]
            lines = []
            for seg in segments:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("content", "")
                lines.append(f"{speaker}: {text}")
            full = "\n".join(lines)
            if len(full) > 200:
                print(f"[Earnings Ingest] Alpha Vantage returned transcript ({len(full)} chars).")
                return full
        # Premium-required or empty response
        info = data.get("Information") or data.get("Note") or ""
        if info:
            print(f"[Earnings Ingest] Alpha Vantage: {info[:120]}")
        return None
    except Exception as e:
        print(f"[Earnings Ingest] Alpha Vantage failed: {e}")
        return None


def fetch_transcript_sec_8k(ticker: str, quarter: int, year: int) -> Optional[str]:
    """
    Fallback: search SEC EDGAR for 8-K filings around the quarter-end date
    that mention 'earnings' or 'results of operations'.
    Returns extracted text or None.
    """
    try:
        cik = get_cik_from_ticker(ticker)
    except ValueError:
        print(f"[Earnings Ingest] Ticker {ticker} not found in SEC database.")
        return None

    try:
        print(f"[Earnings Ingest] Trying SEC 8-K fallback for {ticker} Q{quarter}-{year}...")
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        filings = resp.json()["filings"]["recent"]

        target_month = int(_quarter_to_month(quarter))
        best_doc_url = None

        for i, form in enumerate(filings["form"]):
            if form != "8-K":
                continue
            filed = filings["filingDate"][i]  # "2025-01-30"
            filed_year, filed_month = int(filed[:4]), int(filed[5:7])

            # Build a set of acceptable (year, month) pairs:
            # Accept filings from the quarter-end month through 3 months after,
            # handling year rollover (e.g., Q4 target_month=12 → Dec, Jan, Feb, Mar)
            acceptable = set()
            for offset in range(4):  # 0, 1, 2, 3 months after quarter end
                m = target_month + offset
                y = year
                if m > 12:
                    m -= 12
                    y += 1
                acceptable.add((y, m))

            if (filed_year, filed_month) in acceptable:
                accession = filings["accessionNumber"][i]
                acc_clean = accession.replace("-", "")
                primary_doc = filings["primaryDocument"][i]
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik.lstrip('0')}/{acc_clean}/{primary_doc}"
                )
                best_doc_url = doc_url
                break  # Take the first matching 8-K

        if not best_doc_url:
            print(f"[Earnings Ingest] No matching SEC 8-K found for {ticker} Q{quarter}-{year}.")
            return None

        print(f"[Earnings Ingest] Downloading 8-K from {best_doc_url}...")
        doc_resp = requests.get(best_doc_url, headers=HEADERS, timeout=30)
        doc_resp.raise_for_status()

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(doc_resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        if len(text) > 500:
            print(f"[Earnings Ingest] SEC 8-K text extracted ({len(text)} chars).")
            return text
        print("[Earnings Ingest] SEC 8-K text too short, likely not a transcript.")
        return None

    except Exception as e:
        print(f"[Earnings Ingest] SEC 8-K fallback failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Transcript normalization & segmentation
# ---------------------------------------------------------------------------

def normalize_transcript(
    raw_text: str, ticker: str, quarter: int, year: int
) -> dict:
    """
    Split a raw transcript into Prepared Remarks and Q&A Session.
    Returns:
        {
            "ticker": ..., "quarter": ..., "year": ...,
            "prepared_remarks": str,
            "qa_session": str,
            "source": "alpha_vantage" | "sec_8k",
        }
    """
    text_lower = raw_text.lower()
    split_pos = -1
    for marker in QA_MARKERS:
        idx = text_lower.find(marker)
        if idx != -1:
            split_pos = idx
            break

    if split_pos > 0:
        prepared = raw_text[:split_pos].strip()
        qa = raw_text[split_pos:].strip()
    else:
        # Could not find Q&A boundary — treat entire text as prepared remarks
        prepared = raw_text.strip()
        qa = ""

    return {
        "ticker": ticker.upper(),
        "quarter": quarter,
        "year": year,
        "prepared_remarks": prepared,
        "qa_session": qa,
    }


# ---------------------------------------------------------------------------
# Keyword / entity extraction
# ---------------------------------------------------------------------------

def extract_keywords(text: str) -> dict[str, int]:
    """
    Count occurrences of tracked financial keywords in the text.
    Returns a dict of keyword → count (only keywords with count > 0).
    """
    text_lower = text.lower()
    counts: dict[str, int] = {}
    for kw in TRACKED_KEYWORDS:
        c = len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
        if c > 0:
            counts[kw] = c
    return counts


# ---------------------------------------------------------------------------
# ChromaDB ingest
# ---------------------------------------------------------------------------

def _meta_path(chroma_path: str, ticker: str) -> str:
    d = os.path.join(chroma_path, f"{ticker.upper()}{METADATA_DIR_NAME}")
    os.makedirs(d, exist_ok=True)
    return d


def _save_metadata(
    chroma_path: str,
    ticker: str,
    quarter: int,
    year: int,
    keywords: dict[str, int],
    status: str,
) -> None:
    meta_dir = _meta_path(chroma_path, ticker)
    fname = os.path.join(meta_dir, f"Q{quarter}_{year}.json")
    payload = {
        "ticker": ticker.upper(),
        "quarter": quarter,
        "year": year,
        "status": status,
        "keywords": keywords,
    }
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Earnings Ingest] Metadata saved → {fname}")


def _load_metadata(chroma_path: str, ticker: str) -> list[dict]:
    """Load all quarter metadata files for a ticker."""
    meta_dir = _meta_path(chroma_path, ticker)
    results = []
    if not os.path.isdir(meta_dir):
        return results
    for fname in sorted(os.listdir(meta_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(meta_dir, fname)) as f:
                results.append(json.load(f))
    return results


def ingest_earnings_call(
    ticker: str,
    quarter: int,
    year: int,
    api_key: str = "",
    chroma_path: str = "./chroma_db",
) -> str:
    """
    Full ingest pipeline for one ticker/quarter pair.
    Returns a status string: 'success', 'partial', or 'failed'.
    """
    ticker = ticker.upper()
    collection_dir = os.path.join(chroma_path, f"{ticker}_earnings")

    # Check if already ingested
    meta_dir = _meta_path(chroma_path, ticker)
    meta_file = os.path.join(meta_dir, f"Q{quarter}_{year}.json")
    if os.path.exists(meta_file):
        print(f"[Earnings Ingest] Q{quarter}-{year} for {ticker} already ingested. Skipping.")
        return "exists"

    # 1. Fetch transcript
    raw_text = fetch_transcript_alpha_vantage(ticker, quarter, year, api_key)
    source = "alpha_vantage" if raw_text else None

    if not raw_text:
        raw_text = fetch_transcript_sec_8k(ticker, quarter, year)
        source = "sec_8k" if raw_text else None

    if not raw_text:
        _save_metadata(chroma_path, ticker, quarter, year, {}, "failed")
        return "failed"

    # 2. Normalize & segment
    segments = normalize_transcript(raw_text, ticker, quarter, year)

    # 3. Extract keywords from both sections
    all_text = segments["prepared_remarks"] + " " + segments["qa_session"]
    keywords = extract_keywords(all_text)

    # 4. Chunk & embed into ChromaDB
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []

    if segments["prepared_remarks"]:
        pr_doc = Document(
            page_content=segments["prepared_remarks"],
            metadata={
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "section": "Prepared Remarks",
                "source": source,
            },
        )
        docs.extend(splitter.split_documents([pr_doc]))

    if segments["qa_session"]:
        qa_doc = Document(
            page_content=segments["qa_session"],
            metadata={
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "section": "Q&A Session",
                "source": source,
            },
        )
        docs.extend(splitter.split_documents([qa_doc]))

    if not docs:
        _save_metadata(chroma_path, ticker, quarter, year, keywords, "partial")
        return "partial"

    print(f"[Earnings Ingest] Embedding {len(docs)} chunks into {collection_dir}...")
    embeddings = get_cached_embeddings()
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=collection_dir,
    )

    status = "success" if segments["qa_session"] else "partial"
    _save_metadata(chroma_path, ticker, quarter, year, keywords, status)
    print(f"[Earnings Ingest] {ticker} Q{quarter}-{year} ingested ({status}).")
    return status


# ---------------------------------------------------------------------------
# Inference tools (LangGraph runtime)
# ---------------------------------------------------------------------------

def _get_earnings_db(ticker: str, chroma_path: str = "./chroma_db") -> Chroma:
    """Load the earnings-call Chroma collection for a ticker."""
    ticker = ticker.upper()
    persist_directory = os.path.join(chroma_path, f"{ticker}_earnings")

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Earnings data for {ticker} not ingested. "
            f"Run: python scripts/ingest_earnings_calls.py --tickers {ticker} --quarters Q<N>-<YYYY>"
        )
    embeddings = get_cached_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


@tool
def search_earnings_call(ticker: str, query: str) -> str:
    """
    Searches pre-ingested earnings-call transcripts for a given ticker.
    Use this to find specific management commentary, guidance, or discussion topics.
    CRITICAL: The ticker's earnings data must already be ingested.
    Pass the stock ticker (e.g. 'AAPL') and a natural-language query.
    """
    try:
        db = _get_earnings_db(ticker.upper())
        results = db.similarity_search(query, k=3)

        if not results:
            return f"No earnings-call matches found for '{query}' on {ticker}."

        output_parts = [f"EARNINGS CALL SEARCH RESULTS FOR {ticker.upper()} — '{query}':\n"]
        total_chars = 0
        for doc in results:
            meta = doc.metadata
            label = f"[{meta.get('section', 'Unknown')} | Q{meta.get('quarter', '?')}-{meta.get('year', '?')}]"
            snippet = doc.page_content[:700]
            total_chars += len(snippet)
            output_parts.append(f"{label}\n{snippet}\n")
            if total_chars > 2000:
                break

        return "\n".join(output_parts)
    except Exception as e:
        return f"Error searching earnings data: {e}"


@tool
def get_earnings_sentiment_divergence(ticker: str) -> str:
    """
    Retrieves evidence from both Prepared Remarks and Q&A sections of the
    most recent earnings call for a ticker. Use this to analyze whether
    management tone differs between the scripted portion and live Q&A.
    CRITICAL: The ticker's earnings data must already be ingested.
    """
    try:
        db = _get_earnings_db(ticker.upper())

        # Retrieve top chunks from each section
        pr_results = db.similarity_search(
            "management outlook guidance performance",
            k=3,
            filter={"section": "Prepared Remarks"},
        )
        qa_results = db.similarity_search(
            "analyst question concern risk challenge",
            k=3,
            filter={"section": "Q&A Session"},
        )

        output = f"SENTIMENT DIVERGENCE EVIDENCE FOR {ticker.upper()}:\n\n"

        output += "=== PREPARED REMARKS (scripted management commentary) ===\n"
        if pr_results:
            for doc in pr_results:
                output += doc.page_content[:600] + "\n---\n"
        else:
            output += "(No Prepared Remarks data found.)\n"

        output += "\n=== Q&A SESSION (live analyst questions & management responses) ===\n"
        if qa_results:
            for doc in qa_results:
                output += doc.page_content[:600] + "\n---\n"
        else:
            output += "(No Q&A Session data found — transcript may not have contained a Q&A segment.)\n"

        output += (
            "\nINSTRUCTION: Compare the tone, confidence, and specificity between "
            "Prepared Remarks and Q&A. Note any divergence where management was more "
            "cautious, evasive, or forthcoming in one section vs the other."
        )
        return output

    except Exception as e:
        return f"Error retrieving divergence data: {e}"


@tool
def get_earnings_keyword_trends(ticker: str) -> str:
    """
    Returns quarter-over-quarter keyword frequency trends from ingested
    earnings calls for a given ticker. Shows how often key terms (AI, headwinds,
    growth, guidance, etc.) were mentioned across available quarters.
    CRITICAL: Multiple quarters must be ingested for trend comparison.
    """
    try:
        ticker = ticker.upper()
        all_meta = _load_metadata("./chroma_db", ticker)

        if not all_meta:
            return (
                f"No earnings metadata found for {ticker}. "
                f"Run: python scripts/ingest_earnings_calls.py --tickers {ticker} --quarters Q<N>-<YYYY>"
            )

        # Sort by year, quarter
        all_meta.sort(key=lambda m: (m["year"], m["quarter"]))

        # Build output table
        quarters = [f"Q{m['quarter']}-{m['year']}" for m in all_meta]
        header = f"KEYWORD TRENDS FOR {ticker} ({', '.join(quarters)}):\n\n"

        # Collect all keywords across quarters
        all_kws = set()
        for m in all_meta:
            all_kws.update(m.get("keywords", {}).keys())

        if not all_kws:
            return header + "No tracked keywords found in any ingested quarter."

        rows = []
        rows.append(f"{'Keyword':<30} " + " ".join(f"{q:>10}" for q in quarters))
        rows.append("-" * (30 + 11 * len(quarters)))

        for kw in sorted(all_kws):
            vals = []
            for m in all_meta:
                c = m.get("keywords", {}).get(kw, 0)
                vals.append(f"{c:>10}")
            rows.append(f"{kw:<30} " + " ".join(vals))

        # Add trend commentary for the last two quarters
        if len(all_meta) >= 2:
            rows.append("")
            rows.append("NOTABLE CHANGES (latest vs prior quarter):")
            prev_kw = all_meta[-2].get("keywords", {})
            curr_kw = all_meta[-1].get("keywords", {})
            for kw in sorted(all_kws):
                p, c = prev_kw.get(kw, 0), curr_kw.get(kw, 0)
                if p != c:
                    direction = "↑" if c > p else "↓"
                    rows.append(f"  {kw}: {p} → {c} ({direction})")

        return header + "\n".join(rows)

    except Exception as e:
        return f"Error loading keyword trends: {e}"
