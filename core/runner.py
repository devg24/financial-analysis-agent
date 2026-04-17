from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import time

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
    
    start_time = time.time()
    last_time = start_time
    total_latency = 0.0
    
    for output in compiled_graph.stream(initial_state, stream_config):
        for node_name, state_update in output.items():
            current_time = time.time()
            step_latency = current_time - last_time
            total_latency = current_time - start_time
            last_time = current_time
            
            if node_name == "Planner":
                tasks = state_update.get("pending_tasks", [])
                content = f"Generated {len(tasks)} parallel task(s): {[t['task_id'] for t in tasks]}"
            elif node_name == "Supervisor":
                next_agents = state_update.get("next", [])
                if next_agents == "FINISH":
                    content = "All tasks complete. Routing to Summarizer."
                else:
                    content = f"Dispatching tasks to: {next_agents}"
            else:
                messages = state_update.get("messages", [])
                if not messages:
                    continue
                content = messages[-1].content
                
            if node_name == "Summarizer":
                memo = content
            else:
                steps.append({
                    "node": node_name, 
                    "content": content,
                    "step_latency": round(step_latency, 2),
                    "total_latency": round(total_latency, 2)
                })
    return {"memo": memo, "steps": steps, "total_latency": round(total_latency, 2)}


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
    
    start_time = time.time()
    last_time = start_time
    
    async for output in compiled_graph.astream(initial_state, stream_config):
        for node_name, state_update in output.items():
            current_time = time.time()
            step_latency = current_time - last_time
            total_latency = current_time - start_time
            last_time = current_time
            
            if node_name == "Planner":
                tasks = state_update.get("pending_tasks", [])
                content = f"Generated {len(tasks)} parallel task(s): {[t['task_id'] for t in tasks]}"
            elif node_name == "Supervisor":
                next_agents = state_update.get("next", [])
                if next_agents == "FINISH":
                    content = "All tasks complete. Routing to Summarizer."
                else:
                    content = f"Dispatching tasks to: {next_agents}"
            else:
                messages = state_update.get("messages", [])
                if not messages:
                    continue
                content = messages[-1].content
            
            data = {
                "node": node_name, 
                "content": content,
                "step_latency": round(step_latency, 2),
                "total_latency": round(total_latency, 2)
            }
            yield f"data: {json.dumps(data)}\n\n"
