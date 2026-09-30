"""REST-API (FastAPI) + ausgelieferte Single-Page-Web-UI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import ReportAgent
from .config import get_settings
from .models import Answer

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Annual Report RAG Agent",
    version=__version__,
    description="Analysiert Geschäftsberichte mit einem Tool-Calling-Agenten (Ollama oder Azure OpenAI).",
)


@lru_cache
def get_agent() -> ReportAgent:
    return ReportAgent()


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=10)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm": s.llm_provider,
        "embeddings": s.embedding_provider,
        "vector_store": s.vector_store,
    }


@app.get("/api/reports")
def reports() -> list[dict]:
    return [{"company": c, "year": y, "chunks": n} for c, y, n in get_agent().store.companies()]


@app.post("/api/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    try:
        return get_agent().ask(req.question, [t.model_dump() for t in req.history])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM/Backend-Fehler: {e}") from e
