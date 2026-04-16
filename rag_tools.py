import os
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_core.messages import SystemMessage, HumanMessage

def get_10k_vector_db(ticker: str) -> Chroma:
    """Loads a pre-computed 10-K Chroma Vector Database from disk."""
    ticker = ticker.upper()
    persist_directory = f"./chroma_db/{ticker}_10k"
    
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"10-K data for {ticker} has not been ingested. "
            f"Please run the ingestion pipeline: `python ingest.py --tickers {ticker}`"
        )
        
    print(f"[System: Loading Vector DB for {ticker} 10-K...]")
    # Lazily initialize embeddings so we don't pay the startup cost unless accessed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

@tool
def search_10k_filings(ticker: str, query: str, llm=None) -> str:
    """Searches 10-K and returns a CONCISE summary of findings."""
    try:
        db = get_10k_vector_db(ticker)
        results = db.similarity_search(query, k=5) 
        
        if not results:
            return f"No info found for {query}."
            
        # Combine the text
        context = "\n".join([doc.page_content for doc in results])
        
        # We can use the LLM to 'pre-process' the data so the Supervisor stays clean
        # Note: You'll need to pass the 'llm' object into this tool or initialize a local one
        if llm:
            response = llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=f"Summarize the following 10-K findings for {ticker} regarding {query}:\n\n{context}")
            ])
            return response.content
        else:
            return f"SUMMARY OF 10-K FINDINGS FOR {ticker} ({query}):\n\n{context}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Test using .invoke() as we discussed for sec_tools
    print(search_10k_filings.invoke({"ticker": "TSLA", "query": "marketing risks"}))