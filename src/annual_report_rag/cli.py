"""Kommandozeile: `arr index`, `arr ask`, `arr chat`, `arr serve`, `arr eval`, `arr upload`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from .config import get_settings

app = typer.Typer(help="RAG-Agent für Geschäftsberichte (Ollama lokal / Azure)", no_args_is_help=True)
console = Console()


@app.command()
def index(company: list[str] = typer.Option(None, "--company", "-c", help="Nur diese Unternehmen")):
    """PDFs einlesen, chunken, einbetten und in den Vektorspeicher schreiben."""
    from .indexer import build_index

    s = get_settings()
    console.print(
        f"[bold]Embeddings:[/] {s.embedding_provider} · [bold]Vektorspeicher:[/] {s.vector_store} · [bold]Quelle:[/] {s.document_source}"
    )
    with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}")) as prog:
        tasks: dict[str, int] = {}

        def on_progress(name: str, done: int, total: int):
            if name not in tasks:
                tasks[name] = prog.add_task(name, total=total)
            prog.update(tasks[name], completed=done)

        stats = build_index(s, company or None, on_progress)
    table = Table("Bericht", "Chunks")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)


def _print_answer(ans, show_steps: bool = True):
    if show_steps:
        for st in ans.steps:
            console.print(f"[dim]🔧 {st.tool}({json.dumps(st.arguments, ensure_ascii=False)})[/]")
    console.print(Panel(Markdown(ans.answer), title=f"Antwort · {ans.model} · {ans.latency_s}s", border_style="green"))
    if ans.sources:
        t = Table("#", "Quelle", "Kapitel", show_header=True, header_style="bold")
        for src in ans.sources:
            t.add_row(str(src.ref), f"{src.company} {src.year}, S. {src.page}", src.section or "–")
        console.print(t)


@app.command()
def ask(question: str, quiet: bool = typer.Option(False, "--quiet", "-q")):
    """Eine einzelne Frage stellen."""
    from .agent import ReportAgent

    with console.status("Agent recherchiert…"):
        ans = ReportAgent().ask(question)
    _print_answer(ans, show_steps=not quiet)


@app.command()
def chat():
    """Interaktiver Chat mit Gesprächsverlauf."""
    from .agent import ReportAgent

    agent = ReportAgent()
    history: list[dict] = []
    console.print("[bold]Geschäftsbericht-Agent[/] – 'exit' zum Beenden")
    while True:
        q = console.input("[bold cyan]Frage> [/]").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        with console.status("Agent recherchiert…"):
            ans = agent.ask(q, history)
        _print_answer(ans)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.answer}]
        history = history[-8:]


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Web-UI + REST-API starten."""
    import uvicorn

    uvicorn.run("annual_report_rag.api:app", host=host, port=port, reload=reload)


@app.command()
def upload():
    """PDFs + Manifest in Azure Blob Storage hochladen."""
    from .storage import upload_reports

    for name in upload_reports():
        console.print(f"✔ {name}")


@app.command(name="eval")
def evaluate(
    dataset: Path = typer.Option(Path("eval/questions.jsonl"), "--dataset", "-d"),
    output: Path = typer.Option(Path("eval/results.json"), "--output", "-o"),
):
    """Evaluation: Retrieval-Trefferquote, Faktentreue und Latenz."""
    from .evaluation import run_eval

    report = run_eval(dataset, output, console)
    console.print_json(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    app()
