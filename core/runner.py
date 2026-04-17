from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

from .config import Settings


def create_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.openai_temperature,
        max_tokens=800, # CRITICAL FIX: Stops Groq from 'reserving' 8000+ tokens per API call
    )


def run_financial_query(compiled_graph, user_query: str) -> dict:
    """
    Run one LangGraph turn. Returns memo (if Summarizer ran) and per-node step contents.
    """
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "steps": 0,
        "completed_tasks": set(),
        "pending_tasks": [],
    }
    run_label = user_query if len(user_query) <= 80 else user_query[:77] + "..."
    stream_config = {
        "run_name": run_label,
        "tags": ["fin-agent", "langgraph"],
        "metadata": {"app": "FinAgent"},
    }
    steps: list[dict] = []
    memo: str | None = None
    for output in compiled_graph.stream(initial_state, stream_config):
        for node_name, state_update in output.items():
            if node_name in ("Supervisor", "Planner"):
                continue
            messages = state_update.get("messages", [])
            if not messages:
                continue
            content = messages[-1].content
            if node_name == "Summarizer":
                memo = content
            else:
                steps.append({"node": node_name, "content": content})
    return {"memo": memo, "steps": steps}


async def astream_financial_query(compiled_graph, user_query: str):
    """
    Async generator yielding Server-Sent Events (SSE) for each graph step.
    Useful for streaming over HTTP.
    """
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "steps": 0,
        "completed_tasks": set(),
        "pending_tasks": [],
    }
    run_label = user_query if len(user_query) <= 80 else user_query[:77] + "..."
    stream_config = {
        "run_name": run_label,
        "tags": ["fin-agent", "langgraph"],
        "metadata": {"app": "FinAgent"},
    }
    
    async for output in compiled_graph.astream(initial_state, stream_config):
        for node_name, state_update in output.items():
            if node_name in ("Supervisor", "Planner"):
                continue
            messages = state_update.get("messages", [])
            if not messages:
                continue
            content = messages[-1].content
            
            data = {"node": node_name, "content": content}
            yield f"data: {json.dumps(data)}\n\n"
