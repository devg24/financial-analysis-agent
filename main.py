import os
import operator
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel, Field

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

# Import your custom tools
from sec_tools import get_company_concept_xbrl
from rag_tools import search_10k_filings
from sentiment_tools import get_recent_news

from typing import Set

langchain.debug = True
load_dotenv()


@tool
def get_stock_metrics(ticker: str) -> str:
    """
    Fetches historical market data and calculates basic metrics for a stock.
    CRITICAL: You must pass the official stock ticker symbol (e.g., 'AAPL' for Apple). 
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


def merge_sets(a: Set, b: Set) -> Set:
    return a | b

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    steps: Annotated[int, operator.add]
    completed_tasks: Annotated[Set[str], merge_sets]  
    pending_tasks: list  

# Define the routing options for the Supervisor
members = ["Quant_Agent", "Fundamental_Agent", "Sentiment_Agent"]
options = ["FINISH"] + members


def make_worker_node(agent, name: str):
    def node(state: AgentState):
        pending = state.get("pending_tasks", [])
        completed = state.get("completed_tasks", set())
        
        # Find this agent's next uncompleted task
        my_task = next(
            (t for t in pending 
             if t["agent"] == name and t["task_id"] not in completed),
            None
        )
        
        if not my_task:
            return {"completed_tasks": set()}
        
        # Give the agent a precise, unambiguous instruction
        task_message = HumanMessage(
            content=f"Ticker: {my_task['ticker']}. Task: {my_task['description']}"
        )
        
        result = agent.invoke({"messages": [task_message]})
        has_tool_call = any(isinstance(m, AIMessage) and m.tool_calls for m in result["messages"])
        if not has_tool_call:
            content = f"ERROR: The {name} attempted to answer without using a data tool. This analysis is unauthorized."
        else:
            content = result["messages"][-1].content.strip() or f"[{name}: No data retrieved]"
        return {
            "messages": [AIMessage(content=f"[{my_task['task_id']}] {content}", name=name)],
            "completed_tasks": {my_task["task_id"]}
        }
        
    return node



def create_planner_node(llm):


    planner_prompt = """You are a task planner for a financial AI system.
GOLDEN RULE: Never assume or guess a number. 

CRITICAL AGENT CAPABILITY MAPPING:
1. Quant_Agent: ONLY use for current stock price, trading volume, and 52-week high/lows.
   => GROUPING RULE: If the user asks for multiple price/volume metrics for the SAME ticker, group them into EXACTLY ONE Quant_Agent task. Do NOT make separate tasks for price and volume.
2. Sentiment_Agent: ONLY use for recent news headlines and market sentiment scores.
3. Fundamental_Agent: Use for TWO things:
   - SEC Financial Metrics (Revenue, Net Income, Margins, Cash Flow).
   - SEC 10-K RAG Searches: Use this for ANY qualitative questions about business strategy, supply chain, manufacturing, competition, and corporate RISKS.

BROAD QUERY PROTOCOL:
If the user asks for "general information", "an overview", or just gives a ticker name without specific metrics, you MUST default to creating exactly TWO tasks per ticker:
1. A Quant_Agent task for the price.
2. A Sentiment_Agent task for recent news.
Do NOT use the Fundamental_Agent unless the user explicitly asks for SEC data, revenue, earnings, or risks.

Read the user's request and output a JSON list of tasks needed to answer it.
Each task must have:
- "agent": "Quant_Agent", "Fundamental_Agent", or "Sentiment_Agent"
- "ticker": the stock ticker symbol (e.g. "AAPL")
- "task_id": a unique string
- "description": specific instructions on what to fetch or search

Output ONLY valid JSON. No explanation.
example output:
[
  {"agent": "Quant_Agent", "ticker": "AAPL", "task_id": "Quant_AAPL", "description": "Get price and volume for AAPL"},
  {"agent": "Sentiment_Agent", "ticker": "MSFT", "task_id": "Sentiment_MSFT", "description": "Get sentiment analysis for MSFT"}
]"""
    def planner_function(state: AgentState):
        if state.get("pending_tasks"):
            return {}

        user_message = next(m for m in state["messages"] if isinstance(m, HumanMessage))
        response = llm.invoke([
            SystemMessage(content=planner_prompt),
            HumanMessage(content=user_message.content)
        ])

        import json
        raw = response.content.strip().replace("```json", "").replace("```", "")
        start = raw.find('[')
        end = raw.rfind(']')
        try:
            tasks = json.loads(raw[start:end+1]) if start != -1 and end != -1 else []
        except Exception:
            tasks = []

        if not tasks:
            print("[Planner]: No valid financial tasks found.")
            return {
                "pending_tasks": [],
                "completed_tasks": set(),
                "messages": [AIMessage(
                    content="I can only answer questions about stock prices, SEC filings, and market sentiment.",
                    name="Supervisor"
                )]
            }

        print(f"\n[Planner]: Created {len(tasks)} tasks: {[t['task_id'] for t in tasks]}")
        return {"pending_tasks": tasks, "completed_tasks": set()}

    return planner_function


# --- THE PYTHON-CONTROLLED SUPERVISOR NODE ---
def create_supervisor_node(llm):    
    def supervisor_function(state: AgentState):
        steps = state.get("steps", 0)
        if steps >= 10:
            return {"next": "FINISH", "steps": 1}
        
        pending = state.get("pending_tasks", [])
        completed = state.get("completed_tasks", set())
        
        # Filter out already-completed tasks
        remaining = [t for t in pending if t["task_id"] not in completed]
        
        if not remaining:
            print("-> All tasks complete. Routing to Summarizer.")
            return {"next": "FINISH", "steps": 1}
        
        next_task = remaining[0]
        print(f"\n[Supervisor]: Next task → {next_task['task_id']} ({next_task['description']})")
        return {
            "next": next_task["agent"],
            "steps": 1
        }
    
    return supervisor_function


def create_summarizer_node(llm):
    """
    Compiles specialist worker outputs (and any early Supervisor refusal) into one
    unified Investment Memo. Read-only synthesis: no tools.
    """
    summarizer_system = """You are a senior investment analyst drafting an internal **Investment Memo** for colleagues.

You will receive the user's original question and verbatim outputs from specialist agents (Quant_Agent, Fundamental_Agent, Sentiment_Agent), or a single clarification/refusal if no research ran.

Write the memo using this structure and markdown headings:

# Investment Memo
## Executive Summary
2–4 sentences answering the user in plain language.

## Key Facts & Data
Bullet points. Use ONLY numbers, metrics, and quotes that appear in the specialist outputs. If a section had no data, say "No quantitative/fundamental/sentiment data provided" as appropriate.

## Risks, Sentiment, and Context
Integrate fundamental and sentiment findings when present. If missing, state that briefly.

## Caveats
Note missing specialists, tool errors, or "unauthorized" / ERROR lines exactly as reported—do not soften them.

Rules:
- Do NOT invent tickers, prices, filings, or sentiment scores not present in the inputs.
- Do NOT cite tool names; write for a portfolio manager reader.
- Keep the tone professional and concise."""

    def summarizer_function(state: AgentState):
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_query = user_messages[0].content if user_messages else ""

        blocks = []
        for m in state["messages"]:
            if not isinstance(m, AIMessage):
                continue
            label = m.name or "Assistant"
            blocks.append(f"### {label}\n{m.content}")

        specialist_blob = "\n\n".join(blocks) if blocks else "(No specialist outputs.)"

        response = llm.invoke(
            [
                SystemMessage(content=summarizer_system),
                HumanMessage(
                    content=(
                        f"User request:\n{user_query}\n\n"
                        f"Specialist outputs (verbatim):\n{specialist_blob}"
                    )
                ),
            ]
        )
        memo = (response.content or "").strip()
        return {"messages": [AIMessage(content=memo, name="Summarizer")]}

    return summarizer_function


def build_financial_graph(llm):
    workflow = StateGraph(AgentState)
    
    quant_agent = create_agent(
        model=llm, 
        tools=[get_stock_metrics], 
        system_prompt=(
            "You are a Quantitative Analyst. "
            "GOLDEN RULE: Output ONLY the requested data values. "
            "NEVER use introductory phrases like 'Here is the data' or 'Alternatively'. "
            "NEVER describe the tool you used. Just print the numbers."
        ),
        name="Quant_Agent" 
    )
    
    fundamental_agent = create_agent(
        model=llm, 
        tools=[search_10k_filings, get_company_concept_xbrl], 
        system_prompt=(
            "You are a Fundamental Analyst. "
            "GOLDEN RULE: You must output the EXACT DATA or TEXT returned by your tools. "
            "Do NOT explain how the tools work or what they do. "
            "Just answer the user's question using the fetched data."
        ),
        name="Fundamental_Agent"
    )
    sentiment_agent = create_agent(
            model=llm, 
            tools=[get_recent_news], 
            system_prompt=(
                "You are a Sentiment Analyst. Fetch recent news using your tool. "
                "CRITICAL RULES: Your final response MUST be exactly two lines. "
                "Line 1: The sentiment score (a single number between -1.0 and 1.0). "
                "Line 2: A strict ONE-SENTENCE justification. "
                "Do not add conversational filler. Do not ask the user follow-up questions."
            ),
            name="Sentiment_Agent"
    )
    
    # 2. Add Nodes
    workflow.add_node("Planner", create_planner_node(llm))
    workflow.add_node("Supervisor", create_supervisor_node(llm))
    
    workflow.add_node("Quant_Agent", make_worker_node(quant_agent, "Quant_Agent"))
    workflow.add_node("Fundamental_Agent", make_worker_node(fundamental_agent, "Fundamental_Agent"))
    workflow.add_node("Sentiment_Agent", make_worker_node(sentiment_agent, "Sentiment_Agent"))
    workflow.add_node("Summarizer", create_summarizer_node(llm))

    # 3. Add Edges (Workers always report back to Supervisor)
    for member in members:
        workflow.add_edge(member, "Supervisor")
        
    workflow.add_edge(START, "Planner")
    workflow.add_edge("Planner", "Supervisor")
    
    # 4. Add Conditional Routing from Supervisor
    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "Quant_Agent": "Quant_Agent",
            "Fundamental_Agent": "Fundamental_Agent",
            "Sentiment_Agent": "Sentiment_Agent",
            "FINISH": "Summarizer",
        }
    )

    workflow.add_edge("Summarizer", END)

    return workflow.compile()


def main():
    # Hardware Agnostic LLM Setup
    llm = ChatOpenAI(
        model="llama3.1", 
        api_key="ollama", 
        base_url="http://localhost:11434/v1", 
        temperature=0
    )
    
    print("--- Multi-Agent System + Summarizer Initialized ---")
    print("Type 'exit' to quit.\n")
    
    # Compile the graph
    app = build_financial_graph(llm)
    
    while True:
        user_query = input("\nAsk about a stock: ")
        if user_query.lower() == 'exit':
            break
            
        # Initial state for the graph
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "steps": 0,
            "completed_tasks": set(),
            "pending_tasks": []
        }
                
        # Invoke the graph and stream outputs
        print("\n--- Agent Workflow Started ---")
        for output in app.stream(initial_state):
            for node_name, state_update in output.items():
                if node_name in ("Supervisor", "Planner"):
                    continue
                messages = state_update.get("messages", [])
                if not messages:
                    continue
                content = messages[-1].content
                if node_name == "Summarizer":
                    print("\n--- Investment Memo ---")
                    print(content)
                else:
                    print(f"\n[{node_name}]: {content}")
        
        print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    main()