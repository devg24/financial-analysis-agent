import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def get_recent_news(ticker: str) -> str:
    """
    Fetches the most recent news headlines for a given stock ticker.
    CRITICAL INSTRUCTIONS:
    1. 'ticker': Must be the official uppercase ticker symbol (e.g., AAPL). DO NOT pass the full company name.
    2. Use this tool to gauge current market sentiment, breaking news, and short-term catalysts.
    """
    try:
        ticker = ticker.upper()
        print(f"\n[System: Fetching latest news for {ticker} via Yahoo Finance RSS...]")
        
        # Hit the official Yahoo Finance RSS endpoint
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        
        # A standard web browser User-Agent so Yahoo doesn't block us
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the XML feed using BeautifulSoup
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.find_all("item")
        
        if not items:
            return f"No recent news found for {ticker}."
            
        summary = f"Recent News Headlines for {ticker}:\n\n"
        
        # Grab the top 5 most recent articles
        for i, item in enumerate(items[:10]): 
            title = item.title.text if item.title else "No Title"
            # RSS provides nicely formatted publication dates
            pub_date = item.pubDate.text if item.pubDate else "Recent" 
            
            summary += f"{i+1}. [{pub_date}] {title}\n"
            
        return summary
        
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"

# Quick test block
if __name__ == "__main__":
    test_ticker = "NVDA"
    print("Testing News Pipeline...")
    print(get_recent_news.invoke({"ticker": test_ticker}))