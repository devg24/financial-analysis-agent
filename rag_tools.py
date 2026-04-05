import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sec_tools import get_latest_10k_url, HEADERS 

# 1. Initialize the open-source embedding model (Downloads automatically the first time)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_10k_vector_db(ticker: str) -> Chroma:
    """Downloads, cleans, and embeds a 10-K into a local Chroma Vector Database."""
    
    # Check if we already built the DB for this ticker so we don't rebuild it every time
    persist_directory = f"./chroma_db/{ticker}_10k"
    
    import os
    if os.path.exists(persist_directory):
        print(f"[System: Loading existing Vector DB for {ticker} 10-K...]")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # 1. Get the URL
    url = get_latest_10k_url(ticker)
    if url.startswith("Error") or url.startswith("No 10-K"):
        raise ValueError(url)
        
    print(f"[System: Downloading raw 10-K text for {ticker} from SEC...]")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    raw_text = response.text
    
    # 2. Clean the messy SEC HTML into pure text
    print(f"[System: Isolating 10-K document from raw submission...]")
    
    # The SEC .txt file contains multiple documents (including encoded images/PDFs).
    # We ONLY want the first document block, which contains the 10-K text.
    doc_start = raw_text.find("<DOCUMENT>")
    doc_end = raw_text.find("</DOCUMENT>")
    
    if doc_start != -1 and doc_end != -1:
        raw_text = raw_text[doc_start:doc_end]
        
    print(f"[System: Cleaning HTML and parsing text...]")
    soup = BeautifulSoup(raw_text, "html.parser")
    clean_text = soup.get_text(separator=" ", strip=True)
    
    # Save temporarily to use LangChain's loader
    temp_file = f"{ticker}_temp.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(clean_text)
        
    # 3. Chunk the document
    print(f"[System: Chunking text for embeddings...]")
    loader = TextLoader(temp_file)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    
    # 4. Embed and store in ChromaDB
    print(f"[System: Embedding {len(chunks)} chunks into ChromaDB. This may take a minute...]")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    # Clean up the temp file
    os.remove(temp_file)
    
    return vector_db

@tool
def search_10k_filings(ticker: str, query: str) -> str:
    """
    Searches a company's most recent 10-K annual report to answer qualitative questions.
    CRITICAL INSTRUCTIONS:
    1. 'ticker': The official uppercase ticker symbol (e.g., AAPL).
    2. 'query': A highly specific search query to find in the text (e.g., "What are the supply chain risk factors?", "How does the company generate revenue?").
    Do NOT use this tool for quantitative numbers (like net income). Use it for qualitative business insights, risks, and management commentary.
    """
    try:
        # Get or build the database
        db = build_10k_vector_db(ticker)
        
        # Perform a semantic similarity search
        print(f"\n[System: Searching {ticker} 10-K for: '{query}'...]")
        results = db.similarity_search(query, k=3) # Retrieve the top 3 most relevant paragraphs
        
        if not results:
            return f"No relevant information found in the 10-K for query: {query}"
            
        # Compile the retrieved paragraphs into a single string for the LLM to read
        synthesis = f"Excerpts from {ticker}'s 10-K regarding '{query}':\n\n"
        for i, doc in enumerate(results):
            synthesis += f"--- Excerpt {i+1} ---\n{doc.page_content}\n\n"
            
        return synthesis
        
    except Exception as e:
        return f"Error reading 10-K: {str(e)}"

# Quick test block
if __name__ == "__main__":
    test_ticker = "AAPL"
    test_query = "What are the primary risk factors related to manufacturing?"
    
    print("Testing RAG Pipeline...")
    result = search_10k_filings.invoke({"ticker": test_ticker, "query": test_query})
    print("\n--- RETRIEVED TEXT ---")
    print(result)