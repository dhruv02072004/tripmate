from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tripmate.bootstrap import build_agent
from tripmate.config import get_settings
from tripmate.logging_setup import collector, setup_logging


class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(default="web", max_length=64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level, s.log_file)
    app.state.agent = build_agent(s)      # built once: loads the embedding model once
    yield


app = FastAPI(title="TripMate API", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    # plain `def` (not async): FastAPI runs it in a thread pool,
    # so the blocking LLM call doesn't freeze the server
    r = app.state.agent.ask(req.message, thread_id=req.thread_id)
    return {
        "answer": r.answer,
        "status": r.status,          # ok | rejected | error
        "trace_id": r.trace_id,
        "tool_calls": r.tool_calls,
        "events": collector.pop(r.trace_id) 
    }