"""Werkzeuge, die der Agent per Function Calling nutzen kann."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field

from .config import Settings
from .llm import Embedder
from .models import Chunk, SearchHit, Source
from .query import expand_query
from .vectorstore import VectorStore, rrf

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": (
                "Durchsucht die indexierten Geschäftsberichte (hybride Vektor- + Stichwortsuche). "
                "Liefert nummerierte Textstellen mit Seitenangabe. Für Vergleiche mehrerer "
                "Unternehmen: pro Unternehmen einzeln suchen und `company` setzen. "
                "Formuliere die Suchanfrage in der Sprache des Berichts (SAP/BMW: Deutsch, Siemens: Englisch)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Präzise Suchanfrage, z. B. 'Umsatzerlöse Konzern 2024'",
                    },
                    "company": {"type": "string", "description": "Optional: Unternehmen filtern, z. B. 'SAP'"},
                    "year": {"type": "integer", "description": "Optional: Geschäftsjahr filtern"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reports",
            "description": "Listet alle verfügbaren Unternehmen und Geschäftsjahre im Index.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Rechnet einen arithmetischen Ausdruck exakt aus (+ - * / ** und Klammern). "
                "Für Wachstumsraten, Margen, Differenzen. Beispiel: '(34176 - 31207) / 31207 * 100'"
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Sicherer Taschenrechner (kein `eval`): erlaubt nur Zahlen und Grundrechenarten."""
    expr = expression.replace(",", ".").replace("%", "")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
                raise ValueError("Exponent zu groß")
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Nicht erlaubter Ausdruck: {ast.dump(node)[:60]}")

    return _eval(ast.parse(expr, mode="eval"))


@dataclass
class ToolBox:
    """Führt Tool-Aufrufe aus und verwaltet die fortlaufend nummerierten Quellen."""

    store: VectorStore
    embedder: Embedder
    settings: Settings
    question: str | None = None  # Originalfrage des Nutzers (für Multi-Query-Fusion)
    sources: dict[int, Chunk] = field(default_factory=dict)
    _by_id: dict[str, int] = field(default_factory=dict)

    def _ref(self, chunk: Chunk) -> int:
        if chunk.id not in self._by_id:
            ref = len(self._by_id) + 1
            self._by_id[chunk.id] = ref
            self.sources[ref] = chunk
        return self._by_id[chunk.id]

    def search_reports(self, query: str, company: str | None = None, year: int | None = None) -> str:
        # Leere Strings / Platzhalter vom LLM tolerieren
        company = company or None
        if company and company.lower() in {"all", "alle", "none", "null"}:
            company = None
        hits = self._multi_query_search(query, company, year or None)
        if not hits:
            return "Keine Treffer. Andere Formulierung, andere Sprache oder ohne Filter versuchen."
        blocks = []
        for h in hits:
            c = h.chunk
            section = f" – {c.section}" if c.section else ""
            blocks.append(f"[{self._ref(c)}] {c.company} {c.year}, Seite {c.page}{section}\n{c.text}")
        return "\n\n---\n\n".join(blocks)

    def _multi_query_search(self, query: str, company: str | None, year: int | None) -> list[SearchHit]:
        """Sucht mit der Anfrage des LLM *und* der Originalfrage und fusioniert per RRF.

        Kleine Modelle formulieren oft zu knappe Suchanfragen ("Automobile 2024").
        Die Originalfrage enthält mehr Kontext; die Fusion ist robust gegenüber beidem.
        """
        queries = [query]
        if self.question and self.question.strip().lower() != query.strip().lower():
            queries.append(self.question)
        k = self.settings.top_k
        rankings: list[list[str]] = []
        by_id: dict[str, SearchHit] = {}
        for q in queries:
            expanded = expand_query(q)
            hits = self.store.search(expanded, self.embedder.embed_one(expanded), k * 2, company, year)
            rankings.append([h.chunk.id for h in hits])
            for h in hits:
                by_id.setdefault(h.chunk.id, h)
        fused = rrf(rankings)
        best = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [SearchHit(chunk=by_id[cid].chunk, score=fused[cid]) for cid in best]

    def list_reports(self) -> str:
        rows = self.store.companies()
        return "\n".join(f"- {co} (Geschäftsjahr {yr}): {n} Textabschnitte" for co, yr, n in rows) or "Index ist leer."

    def calculate(self, expression: str) -> str:
        try:
            value = safe_eval(expression)
        except Exception as e:  # noqa: BLE001 – Fehler geht als Feedback an das LLM
            return f"Fehler: {e}"
        return f"{expression} = {value:,.4f}".rstrip("0").rstrip(".")

    def call(self, name: str, arguments: dict) -> str:
        fn = {
            "search_reports": self.search_reports,
            "list_reports": self.list_reports,
            "calculate": self.calculate,
        }.get(name)
        if fn is None:
            return f"Unbekanntes Werkzeug: {name}"
        try:
            return fn(**arguments)
        except TypeError as e:
            return f"Ungültige Argumente für {name}: {e}"

    def as_sources(self, refs: list[int]) -> list[Source]:
        out = []
        for ref in refs:
            c = self.sources.get(ref)
            if c:
                out.append(
                    Source(
                        ref=ref, company=c.company, year=c.year, page=c.page, section=c.section, snippet=c.text[:300]
                    )
                )
        return out
