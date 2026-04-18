import operator
from typing import Annotated, Sequence, TypedDict, Literal, Set

import yfinance as yf
import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

from .sec_tools import get_company_concept_xbrl
from .rag_tools import search_10k_filings
from .sentiment_tools import get_recent_news
from .earnings_tools import (
    search_earnings_call,
    get_earnings_sentiment_divergence,
    get_earnings_keyword_trends,
)


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

        current_price = hist["Close"].iloc[-1]
        monthly_high = hist["High"].max()
        monthly_low = hist["Low"].min()
        avg_volume = hist["Volume"].mean()

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
    next: str | list[str]
    steps: Annotated[int, operator.add]
    completed_tasks: Annotated[Set[str], merge_sets]
    pending_tasks: list


members = ["Quant_Agent", "Fundamental_Agent", "Sentiment_Agent", "Earnings_Agent"]


def make_worker_node(agent, name: str):
    def node(state: AgentState):
        pending = state.get("pending_tasks", [])
        completed = state.get("completed_tasks", set())

        my_task = next(
            (
                t
                for t in pending
                if t["agent"] == name and t["task_id"] not in completed
            ),
            None,
        )

        if not my_task:
            return {"completed_tasks": set()}

        task_message = HumanMessage(
            content=f"Ticker: {my_task['ticker']}. Task: {my_task['description']}"
        )

        result = agent.invoke({"messages": [task_message]})
        has_tool_call = any(
            isinstance(m, AIMessage) and m.tool_calls for m in result["messages"]
        )
        if not has_tool_call:
            content = f"ERROR: The {name} attempted to answer without using a data tool. This analysis is unauthorized."
        else:
            content = result["messages"][-1].content.strip() or f"[{name}: No data retrieved]"
        return {
            "messages": [AIMessage(content=f"[{my_task['task_id']}] {content}", name=name)],
            "completed_tasks": {my_task["task_id"]},
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
4. Earnings_Agent: Use for earnings-call analysis. This includes:
   - Management commentary and guidance from earnings calls.
   - Sentiment divergence between Prepared Remarks and Q&A sessions.
   - Keyword and entity tracking across quarters (e.g., mentions of "AI", "headwinds", "growth").
   => Use this agent when the user asks about earnings calls, management tone, guidance language, or quarter-over-quarter keyword trends.

Read the user's request and output a JSON list of tasks needed to answer it.
Each task must have:
- "agent": "Quant_Agent", "Fundamental_Agent", "Sentiment_Agent", or "Earnings_Agent"
- "ticker": the stock ticker symbol (e.g. "AAPL")
- "task_id": a unique string
- "description": specific instructions on what to fetch or search

Output ONLY valid JSON. No explanation.
example output:
[
  {"agent": "Quant_Agent", "ticker": "AAPL", "task_id": "Quant_AAPL", "description": "Get price and volume for AAPL"},
  {"agent": "Sentiment_Agent", "ticker": "MSFT", "task_id": "Sentiment_MSFT", "description": "Get sentiment analysis for MSFT"},
  {"agent": "Earnings_Agent", "ticker": "AAPL", "task_id": "Earnings_AAPL", "description": "Analyze sentiment divergence between prepared remarks and Q&A in the latest earnings call"}
]"""

    def planner_function(state: AgentState):
        if state.get("pending_tasks"):
            return {}

        user_message = next(m for m in state["messages"] if isinstance(m, HumanMessage))
        response = llm.invoke(
            [
                SystemMessage(content=planner_prompt),
                HumanMessage(content=user_message.content),
            ]
        )

        import json

        raw = response.content.strip().replace("```json", "").replace("```", "")
        start = raw.find("[")
        end = raw.rfind("]")
        try:
            tasks = json.loads(raw[start : end + 1]) if start != -1 and end != -1 else []
        except Exception:
            tasks = []

        if not tasks:
            print("[Planner]: No valid financial tasks found.")
            return {
                "pending_tasks": [],
                "completed_tasks": set(),
                "messages": [
                    AIMessage(
                        content="I can only answer questions about stock prices, SEC filings, and market sentiment.",
                        name="Supervisor",
                    )
                ],
            }

        print(f"\n[Planner]: Created {len(tasks)} tasks: {[t['task_id'] for t in tasks]}")
        return {"pending_tasks": tasks, "completed_tasks": set()}

    return planner_function


def create_supervisor_node(llm):
    def supervisor_function(state: AgentState):
        steps = state.get("steps", 0)
        if steps >= 10:
            return {"next": "FINISH", "steps": 1}

        pending = state.get("pending_tasks", [])
        completed = state.get("completed_tasks", set())

        remaining = [t for t in pending if t["task_id"] not in completed]

        if not remaining:
            print("-> All tasks complete. Routing to Summarizer.")
            return {"next": "FINISH", "steps": 1}

        # Dispatch one task per unique agent in parallel
        agents_to_dispatch = []
        dispatched_tasks = []
        for task in remaining:
            if task["agent"] not in agents_to_dispatch:
                agents_to_dispatch.append(task["agent"])
                dispatched_tasks.append(task["task_id"])

        print(f"\n[Supervisor]: Dispatching tasks in parallel → {dispatched_tasks}")
        return {
            "next": agents_to_dispatch,
            "steps": 1,
        }

    return supervisor_function


def create_summarizer_node(llm):
    summarizer_system = """You are a senior investment analyst drafting an internal **Investment Memo** for colleagues.

You will receive the user's original question and verbatim outputs from specialist agents (Quant_Agent, Fundamental_Agent, Sentiment_Agent, Earnings_Agent), or a single clarification/refusal if no research ran.

Write the memo using this structure and markdown headings:

# Investment Memo
## Executive Summary
2-4 sentences answering the user in plain language.

## Key Facts & Data
Bullet points. Use ONLY numbers, metrics, and quotes that appear in the specialist outputs. If a section had no data, say "No quantitative/fundamental/sentiment data provided" as appropriate.

## Earnings Call Insights
If Earnings_Agent data is present, summarize:
- Sentiment divergence between Prepared Remarks and Q&A (was management more cautious or bullish in live Q&A vs. scripted remarks?).
- Notable keyword/entity trends across quarters (e.g., increasing mentions of "AI", declining mentions of "headwinds").
If no earnings data was provided, omit this section entirely.

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
            "You have exactly ONE tool: get_stock_metrics(ticker). "
            "For any price, volume, or trading-range question you MUST call get_stock_metrics—do not answer from memory. "
            "NEVER invent other tool names, NEVER output JSON blocks suggesting tools that do not exist. "
            "GOLDEN RULE: After the tool returns, you must format the output gracefully so it is easy to read. "
            "Bold the labels (like **Current Price:** or **Average Volume:**) before injecting the numbers. "
            "NEVER use introductory conversational filler like 'Here is the data'. Just print the labeled metrics directly."
        ),
        name="Quant_Agent",
    )

    fundamental_agent = create_agent(
        model=llm,
        tools=[search_10k_filings, get_company_concept_xbrl],
        system_prompt=(
            "You are a Fundamental Analyst. "
            "GOLDEN RULE: You must output the EXACT DATA or TEXT returned by your tools. "
            "Do NOT explain how the tools work or what they do. "
            "CRITICAL: ONCE YOU HAVE CALLED TO THE TOOL ONCE AND RECEIVED THE DATA, YOU MUST WRITE YOUR FINAL ANSWER IMMEDIATELY. DO NOT CALL THE TOOL A SECOND TIME. "
            "Just answer the user's question using the fetched data and stop."
        ),
        name="Fundamental_Agent",
    )
    sentiment_agent = create_agent(
        model=llm,
        tools=[get_recent_news],
        system_prompt=(
            "You are a Sentiment Analyst. Fetch recent news using your tool. "
            "CRITICAL RULES: Your final response MUST be exactly five lines. "
            "Line 1: The sentiment score (a single number between -1.0 and 1.0). "
            "Line 2-5: Justify the sentiment score based on the news articles."
            "Include important keywords from the news articles in your response."
            "Do not add conversational filler. Do not ask the user follow-up questions."
        ),
        name="Sentiment_Agent",
    )

    earnings_agent = create_agent(
        model=llm,
        tools=[
            search_earnings_call,
            get_earnings_sentiment_divergence,
            get_earnings_keyword_trends,
        ],
        system_prompt=(
            "You are an Earnings Call Analyst specializing in management commentary analysis. "
            "You have THREE tools for analyzing pre-ingested earnings-call transcripts:\n"
            "1. search_earnings_call: Search transcripts for specific topics (guidance, margins, strategy, etc.).\n"
            "2. get_earnings_sentiment_divergence: Compare management tone in scripted Prepared Remarks vs live Q&A.\n"
            "3. get_earnings_keyword_trends: Track keyword frequency changes across quarters.\n\n"
            "CRITICAL RULES:\n"
            "- You MUST call at least one tool. Do NOT answer from memory.\n"
            "- If a tool returns an error about missing data, report that the earnings data for that "
            "ticker/quarter has not been ingested and suggest running the ingest script.\n"
            "- After the tool returns, write a clear, evidence-backed analysis. Bold key findings.\n"
            "- Do NOT add conversational filler. Do NOT ask follow-up questions."
        ),
        name="Earnings_Agent",
    )

    workflow.add_node("Planner", create_planner_node(llm))
    workflow.add_node("Supervisor", create_supervisor_node(llm))

    workflow.add_node("Quant_Agent", make_worker_node(quant_agent, "Quant_Agent"))
    workflow.add_node(
        "Fundamental_Agent", make_worker_node(fundamental_agent, "Fundamental_Agent")
    )
    workflow.add_node("Sentiment_Agent", make_worker_node(sentiment_agent, "Sentiment_Agent"))
    workflow.add_node("Earnings_Agent", make_worker_node(earnings_agent, "Earnings_Agent"))
    workflow.add_node("Summarizer", create_summarizer_node(llm))

    for member in members:
        workflow.add_edge(member, "Supervisor")

    workflow.add_edge(START, "Planner")
    workflow.add_edge("Planner", "Supervisor")

    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "Quant_Agent": "Quant_Agent",
            "Fundamental_Agent": "Fundamental_Agent",
            "Sentiment_Agent": "Sentiment_Agent",
            "Earnings_Agent": "Earnings_Agent",
            "FINISH": "Summarizer",
        },
    )

    workflow.add_edge("Summarizer", END)

    return workflow.compile()
