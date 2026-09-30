"""Evaluation des Agenten gegen einen kleinen, manuell verifizierten Fragenkatalog.

Pro Frage (`eval/questions.jsonl`) werden gemessen:

* **retrieval_hit** – wurde mindestens eine der erwarteten Seiten gefunden?
* **answer_correct** – enthält die Antwort alle erwarteten Kennzahlen/Begriffe?
  (Zahlenformate wie "34.176", "34,2 Mrd." oder "34176" werden normalisiert.)
* **cited** – verweist die Antwort auf mindestens eine Quelle?
* **latency_s**
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .agent import ReportAgent


def _normalize(text: str) -> str:
    text = text.lower().replace("\u202f", " ").replace("\u00a0", " ")
    # Tausendertrennzeichen entfernen: 34.176 / 34,176 -> 34176
    text = re.sub(r"(?<=\d)[.,'](?=\d{3}\b)", "", text)
    return text.replace(",", ".")


def contains_all(answer: str, expected: list[str]) -> bool:
    norm = _normalize(answer)
    return all(any(_normalize(alt) in norm for alt in exp.split("|")) for exp in expected)


def run_eval(dataset: Path, output: Path, console: Console | None = None) -> dict:
    console = console or Console()
    agent = ReportAgent()
    items = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for i, item in enumerate(items, 1):
        console.print(f"[dim]({i}/{len(items)})[/] {item['question']}")
        # Retrieval isoliert messen (ohne LLM-Varianz)
        hits = agent.store.search(
            item.get("search_query", item["question"]),
            agent.embedder.embed_one(item.get("search_query", item["question"])),
            agent.s.top_k,
            item.get("company"),
        )
        found_pages = {(h.chunk.company, h.chunk.page) for h in hits}
        retrieval_hit = any((item["company"], p) in found_pages for p in item["pages"])

        ans = agent.ask(item["question"])
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieval_hit": retrieval_hit,
                "answer_correct": contains_all(ans.answer, item["expected"]),
                "cited": bool(ans.sources),
                "latency_s": ans.latency_s,
                "tool_calls": len(ans.steps),
                "answer": ans.answer,
                "sources": [s.model_dump() for s in ans.sources],
            }
        )

    n = len(rows)
    summary = {
        "model": agent.chat.label,
        "embeddings": agent.embedder.label,
        "questions": n,
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows) / n, 3),
        "answer_accuracy": round(sum(r["answer_correct"] for r in rows) / n, 3),
        "citation_rate": round(sum(r["cited"] for r in rows) / n, 3),
        "median_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    t = Table("ID", "Retrieval", "Antwort", "Zitiert", "Latenz")
    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    for r in rows:
        t.add_row(r["id"], ok(r["retrieval_hit"]), ok(r["answer_correct"]), ok(r["cited"]), f"{r['latency_s']}s")
    console.print(t)
    return {"summary": summary, "results": rows}
