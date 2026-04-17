import os
import argparse
import requests
import re
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.sec_tools import get_latest_10k_url, HEADERS

def ingest_10k(ticker: str):
    """Downloads, cleans, and embeds a 10-K into a local Chroma Vector Database."""
    ticker = ticker.upper()
    persist_directory = f"./chroma_db/{ticker}_10k"
    
    if os.path.exists(persist_directory):
        print(f"[Ingest: Vector DB for {ticker} 10-K already exists at {persist_directory}. Skipping...]")
        return
    
    print(f"\n==============================================")
    print(f"Starting Ingestion Pipeline for {ticker}")
    print(f"==============================================")

    url = get_latest_10k_url(ticker)
    if url.startswith("Error") or url.startswith("No 10-K"):
        print(f"[Error: SEC URL Fetch failed: {url}]")
        return
        
    print(f"[1/4] Downloading raw 10-K from SEC: {url}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    raw_text = response.text
    
    print(f"[2/4] Parsing HTML and isolating text payload...")
    doc_match = re.search(r'<DOCUMENT>(.*?)</DOCUMENT>', raw_text, re.DOTALL | re.IGNORECASE)
    if doc_match:
        raw_text = doc_match.group(1)
        
    soup = BeautifulSoup(raw_text, "html.parser")
    clean_text = soup.get_text(separator=" ", strip=True)
    
    print(f"[3/4] Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = [Document(page_content=clean_text, metadata={"source": url, "ticker": ticker})]
    chunks = text_splitter.split_documents(docs)
    
    print(f"[4/4] Embedding {len(chunks)} chunks into Chroma DB. (This may take a minute) ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print(f"[Success] {ticker} 10-K successfully ingested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K filings into Chroma DB.")
    parser.add_argument(
        "--tickers", 
        nargs="+", 
        required=True, 
        help="List of stock tickers to ingest (e.g., --tickers AAPL MSFT TSLA)"
    )
    args = parser.parse_args()
    
    os.makedirs("./chroma_db", exist_ok=True)
    
    for t in args.tickers:
        try:
            ingest_10k(t)
        except Exception as e:
            print(f"[Error] Failed to ingest {t}: {str(e)}")
