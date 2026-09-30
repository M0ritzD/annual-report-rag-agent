from __future__ import annotations

from pydantic import BaseModel, Field


class ReportMeta(BaseModel):
    """Metadaten eines Geschäftsberichts (aus `data/reports.json`)."""

    file: str
    company: str
    year: int
    title: str
    language: str = "de"
    url: str | None = None


class Chunk(BaseModel):
    id: str
    company: str
    year: int
    page: int
    section: str | None = None
    text: str
    source_file: str

    def citation(self) -> str:
        return f"{self.company} {self.year}, S. {self.page}"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float


class Source(BaseModel):
    ref: int
    company: str
    year: int
    page: int
    section: str | None = None
    snippet: str


class AgentStep(BaseModel):
    tool: str
    arguments: dict
    result_preview: str


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    model: str
    latency_s: float
