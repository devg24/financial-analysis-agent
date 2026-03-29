import os
import yfinance as yf
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

@tool
def get_stock_price(ticker: str) -> str:
    """Fetches the current closing stock price for a given ticker symbol."""
    try:
        stock = yf.Ticker(ticker)
        # Fetch the last day's data
        hist = stock.history(period="1d")
        if hist.empty:
            return f"Could not find price data for ticker: {ticker}."
            
        price = hist['Close'].iloc[-1]
        return f"The current price of {ticker} is ${price:.2f}."
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"

def main():
    # 1. Initialize the reasoning engine (LLM)
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    # 2. Provide the agent with its toolkit
    tools = [get_stock_price]

    # 3. Initialize the agent with function-calling capabilities
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True # Set to True so you can see the agent's thought process in the terminal
    )

    print("--- Financial Analysis Agent Initialized ---")
    print("Type 'exit' to quit.\n")
    
    # 4. Create a simple loop to chat with the agent
    while True:
        user_query = input("Ask the agent a financial question: ")
        if user_query.lower() == 'exit':
            break
            
        response = agent.run(user_query)
        print(f"\nAgent: {response}\n")

if __name__ == "__main__":
    main()