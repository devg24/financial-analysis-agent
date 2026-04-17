import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import langchain

from core.config import Settings
from core.graph_builder import build_financial_graph
from core.runner import create_llm, run_financial_query, astream_financial_query


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    langchain.debug = os.getenv("LANGCHAIN_DEBUG", "").lower() in ("1", "true", "yes")
    settings = Settings()
    llm = create_llm(settings)
    app.state.settings = settings
    app.state.graph = build_financial_graph(llm)
    yield


app = FastAPI(title="FinAgent", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=16000)


class StepOut(BaseModel):
    node: str
    content: str
    step_latency: float | None = None
    total_latency: float | None = None


class ChatResponse(BaseModel):
    memo: str | None = None
    steps: list[StepOut] = Field(default_factory=list)
    total_latency: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    graph = request.app.state.graph
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        result = await run_in_threadpool(run_financial_query, graph, q)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ChatResponse(**result)


@app.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    graph = request.app.state.graph
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")
    
    return StreamingResponse(
        astream_financial_query(graph, q), 
        media_type="text/event-stream"
    )
