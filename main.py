import os
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

langchain.debug = True


load_dotenv()

@tool
def get_stock_metrics(ticker: str) -> str:
    """
    Fetches historical market data and calculates basic metrics for a stock.
    CRITICAL: You must pass the official stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft, 'GOOGL' for Google). 
    DO NOT pass the full company name. If the user provides a company name, convert it to the ticker symbol first.
    """
    try:
        ticker = ticker.upper()
        print(f"\n[System: Fetching yfinance data for {ticker}...]")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo") 
        
        if hist.empty:
            return f"Could not find price data for ticker: {ticker}. Tell the user the data fetch failed."
            
        current_price = hist['Close'].iloc[-1]
        monthly_high = hist['High'].max()
        monthly_low = hist['Low'].min()
        avg_volume = hist['Volume'].mean()
        
        summary = (
            f"Data for {ticker}:\n"
            f"- Current Price: ${current_price:.2f}\n"
            f"- 1-Month High: ${monthly_high:.2f}\n"
            f"- 1-Month Low: ${monthly_low:.2f}\n"
            f"- Average Daily Volume: {int(avg_volume):,}"
        )
        return summary
    except Exception as e:
        return f"Error fetching data: {str(e)}"

def main():
    # Hardware Agnostic LLM Setup
    llm = ChatOpenAI(
        model="llama3.1", # Must match the model name in Ollama
        api_key="ollama", # API key is required by the client, but ignored by Ollama
        base_url="http://localhost:11434/v1", 
        temperature=0
    )
    
    tools = [get_stock_metrics]
    llm_with_tools = llm.bind_tools(tools)
    available_tools = {t.name: t for t in tools}
    
    print("--- Phase 1: Hardware-Agnostic Agent Initialized ---")
    print("Type 'exit' to quit.\n")
    
# 4. The Tool-Calling Loop
    while True:
        user_query = input("\nAsk about a stock: ")
        if user_query.lower() == 'exit':
            break
            
        messages = [HumanMessage(content=user_query)]
        
        # Step A: The LLM processes the query and decides if it needs a tool
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # ==========================================
        # DEBUGGING: THE AGENT'S THOUGHT PROCESS
        # ==========================================
        print("\n[--- AGENT INTERNAL REASONING ---]")
        if response.tool_calls:
            print("Decision: I need to fetch external data to answer this.")
            for call in response.tool_calls:
                print(f"Tool Selected: '{call['name']}'")
                print(f"Arguments Generated: {call['args']}")
        else:
            print("Decision: I have enough internal knowledge to answer this directly.")
        print("[--------------------------------]\n")
        # ==========================================
        
        # Step B: Check if the LLM requested a tool execution
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Dynamically find the right tool 
                selected_tool = available_tools[tool_call["name"]]
                
                # Execute the tool
                tool_msg = selected_tool.invoke(tool_call)
                messages.append(tool_msg)
            
            # Step C: Send the raw tool output back to the LLM to synthesize an answer
            final_response = llm_with_tools.invoke(messages)
            print(f"\nAgent: {final_response.content}")
        else:
            # If no tool was needed, just print the standard conversational response
            print(f"\nAgent: {response.content}")

if __name__ == "__main__":
    main()