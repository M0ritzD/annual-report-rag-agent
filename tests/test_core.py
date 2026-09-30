"""Unit-Tests ohne Netzwerk/LLM – laufen in der CI (GitHub Actions)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from annual_report_rag.agent import ReportAgent, extract_citations
from annual_report_rag.config import Settings
from annual_report_rag.evaluation import contains_all
from annual_report_rag.ingest import clean_page, find_boilerplate, split_text
from annual_report_rag.models import Chunk
from annual_report_rag.tools import ToolBox, safe_eval
from annual_report_rag.vectorstore import LocalVectorStore, rrf, tokenize

# ---------------------------------------------------------------- Ingestion


def test_boilerplate_detection_removes_repeated_headers():
    pages = [["BMW Group Bericht 2024", "Konzernabschluss", f"Inhalt Seite {i}"] for i in range(10)]
    bp = find_boilerplate(pages)
    assert "BMW Group Bericht 2024" in bp
    assert "Inhalt Seite 3" not in bp
    assert clean_page(pages[3] + ["42"], bp) == "Inhalt Seite 3"


def test_clean_page_joins_hyphenation():
    assert clean_page(["Umsatz-", "erlöse stiegen"], set()) == "Umsatzerlöse stiegen"


def test_split_text_respects_size_and_overlap():
    text = "\n".join(f"Zeile {i} " + "x" * 50 for i in range(100))
    chunks = split_text(text, size=500, overlap=100)
    assert len(chunks) > 5
    assert all(len(c) <= 600 for c in chunks)
    # Überlappung: Ende von Chunk n taucht am Anfang von Chunk n+1 auf
    assert chunks[0].splitlines()[-1][-20:] in chunks[1]


def test_split_text_short_text_single_chunk():
    assert split_text("kurz", 100, 10) == ["kurz"]


# ---------------------------------------------------------------- Retrieval


def test_tokenize_keeps_umlauts_and_numbers():
    assert tokenize("Umsatzerlöse: 34.176 Mio. €") == ["umsatzerlöse", "34.176", "mio"]


def test_rrf_prefers_items_ranked_high_in_both_lists():
    scores = rrf([[1, 2, 3], [2, 1, 4]])
    assert max(scores, key=lambda k: scores[k]) in {1, 2}
    assert scores[4] < scores[3] or scores[4] == pytest.approx(scores[3], rel=0.1)


def _chunk(i: int, company: str, text: str) -> Chunk:
    return Chunk(id=f"c{i}", company=company, year=2024, page=i, text=text, source_file="x.pdf")


@pytest.fixture
def store(tmp_path):
    chunks = [
        _chunk(1, "SAP", "Die Umsatzerlöse stiegen um 10 % auf 34,2 Mrd. €."),
        _chunk(2, "SAP", "Der Free Cashflow sank auf 4,1 Mrd. €."),
        _chunk(3, "BMW", "Die Konzernumsatzerlöse betrugen 142.380 Mio. €."),
        _chunk(4, "BMW", "Mitarbeitende am Jahresende: 159.104"),
    ]
    vecs = np.eye(4, dtype=np.float32)
    s = LocalVectorStore(tmp_path / "idx")
    s.add(chunks, vecs)
    return s


def test_local_store_hybrid_search_and_filter(store):
    hits = store.search("Free Cashflow", np.array([0, 1, 0, 0], dtype=np.float32), k=2)
    assert hits[0].chunk.id == "c2"
    hits = store.search("Umsatzerlöse", np.array([1, 0, 0, 0], dtype=np.float32), k=5, company="bmw")
    assert {h.chunk.company for h in hits} == {"BMW"}


def test_local_store_persists(store, tmp_path):
    reloaded = LocalVectorStore(tmp_path / "idx")
    assert len(reloaded.chunks) == 4
    assert reloaded.companies() == [("BMW", 2024, 2), ("SAP", 2024, 2)]


# ---------------------------------------------------------------- Tools


@pytest.mark.parametrize(
    ("expr", "expected"),
    [("(34176 - 31207) / 31207 * 100", 9.514), ("2 ** 3", 8), ("-4,5 + 1", -3.5)],
)
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "a + 1", "9 ** 9999"])
def test_safe_eval_rejects_code(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_extract_citations():
    assert extract_citations("Umsatz [1], Marge [2, 3] und [1]; FCF [4;5]") == [1, 2, 3, 4, 5]


def test_contains_all_normalizes_number_formats():
    assert contains_all("Der Umsatz betrug 34.176 Mio. € (+10 %).", ["34176|34,2", "10 %"])
    assert contains_all("Umsatz: 34,2 Mrd. €", ["34.176|34,2"])
    assert not contains_all("Umsatz: 31,2 Mrd. €", ["34,2"])


# ---------------------------------------------------------------- Agent-Loop


class FakeEmbedder:
    label = "fake"

    def embed_one(self, text):
        return (
            np.array([0, 1, 0, 0], dtype=np.float32)
            if "cash" in text.lower()
            else np.array([1, 0, 0, 0], dtype=np.float32)
        )


class FakeChat:
    """Simuliert ein LLM: 1. Schritt Tool-Call, 2. Schritt Antwort mit Zitat."""

    label = "fake:model"

    def __init__(self, first_tool_call: bool = True):
        self.calls = 0
        self.first_tool_call = first_tool_call

    def complete(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1 and self.first_tool_call:
            call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="search_reports", arguments=json.dumps({"query": "Free Cashflow", "company": "SAP"})
                ),
            )
            return SimpleNamespace(content="", tool_calls=[call])
        return SimpleNamespace(content="Der Free Cashflow sank auf 4,1 Mrd. € [1]. Unbelegt [99].", tool_calls=None)


def test_agent_tool_loop_and_citations(store):
    agent = ReportAgent(settings=Settings(), chat=FakeChat(), embedder=FakeEmbedder(), store=store)
    ans = agent.ask("Wie hoch war der Free Cashflow der SAP?")
    assert [s.tool for s in ans.steps] == ["search_reports"]
    assert [s.ref for s in ans.sources] == [1]  # [99] existiert nicht und wird verworfen
    assert ans.sources[0].page == 2


def test_agent_forces_retrieval_when_model_skips_tools(store):
    agent = ReportAgent(settings=Settings(), chat=FakeChat(first_tool_call=False), embedder=FakeEmbedder(), store=store)
    ans = agent.ask("Free Cashflow SAP?")
    assert ans.steps[0].arguments.get("forced") is True
    assert ans.sources


def test_toolbox_numbers_sources_stably(store):
    tb = ToolBox(store, FakeEmbedder(), Settings())
    tb.search_reports("Free Cashflow")
    first = dict(tb.sources)
    tb.search_reports("Free Cashflow")
    assert tb.sources == first


# ---------------------------------------------------------------- Query-Expansion


def test_expand_query_adds_english_terms_for_german_question():
    from annual_report_rag.query import expand_query

    q = expand_query("Wie viele Mitarbeitende hatte Siemens?")
    assert "employees" in q
    assert expand_query("Wie ist das Wetter?") == "Wie ist das Wetter?"


def test_expand_query_adds_german_terms_for_english_question():
    from annual_report_rag.query import expand_query

    assert "umsatz" in expand_query("revenue growth SAP").lower()
