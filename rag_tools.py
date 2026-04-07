import os
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# Updated import to avoid the deprecation warning in your logs
from langchain_chroma import Chroma 
from sec_tools import get_latest_10k_url, HEADERS 
import re

# Initialize embeddings once
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_10k_vector_db(ticker: str) -> Chroma:
    """Downloads, cleans, and embeds a 10-K into a local Chroma Vector Database."""
    
    ticker = ticker.upper()
    persist_directory = f"./chroma_db/{ticker}_10k"
    
    # Check if we already built the DB for this ticker
    if os.path.exists(persist_directory):
        print(f"[System: Loading existing Vector DB for {ticker} 10-K...]")
        return Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
    
    # 1. Get the URL using updated sec_tools logic
    url = get_latest_10k_url(ticker)
    if url.startswith("Error") or url.startswith("No 10-K"):
        raise ValueError(f"SEC URL Fetch failed: {url}")
        
    print(f"[System: Downloading raw 10-K for {ticker} from SEC...]")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    raw_text = response.text
    
    # 2. Clean SEC text
    # We use a greedy regex or find to isolate the main document body
    print(f"[System: Isolating document content...]")
    doc_match = re.search(r'<DOCUMENT>(.*?)</DOCUMENT>', raw_text, re.DOTALL | re.IGNORECASE)
    if doc_match:
        raw_text = doc_match.group(1)
        
    soup = BeautifulSoup(raw_text, "html.parser")
    # separator=" " prevents words from sticking together when HTML tags are removed
    clean_text = soup.get_text(separator=" ", strip=True)
    
    # 3. Chunking logic
    print(f"[System: Chunking and Embedding...]")
    # We'll use a string-based splitter directly instead of saving to a temp file
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    # Create document objects directly
    from langchain_core.documents import Document
    docs = [Document(page_content=clean_text, metadata={"source": url, "ticker": ticker})]
    chunks = text_splitter.split_documents(docs)
    
    # 4. Store in Chroma
    print(f"[System: Embedding {len(chunks)} chunks. This may take a minute...]")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    return vector_db

@tool
def search_10k_filings(ticker: str, query: str) -> str:
    """Searches 10-K and returns a CONCISE summary of findings."""
    try:
        db = build_10k_vector_db(ticker)
        results = db.similarity_search(query, k=5) 
        
        if not results:
            return f"No info found for {query}."
            
        # Combine the text
        context = "\n".join([doc.page_content for doc in results])
        
        # We can use the LLM to 'pre-process' the data so the Supervisor stays clean
        # Note: You'll need to pass the 'llm' object into this tool or initialize a local one
        return f"SUMMARY OF 10-K FINDINGS FOR {ticker} ({query}):\n\n{context}"
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Test using .invoke() as we discussed for sec_tools
    print(search_10k_filings.invoke({"ticker": "TSLA", "query": "marketing risks"}))