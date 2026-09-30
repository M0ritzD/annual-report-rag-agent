"""Query-Expansion mit einem kleinen DE↔EN-Finanzglossar.

Das Korpus ist gemischtsprachig (SAP/BMW deutsch, Siemens englisch). Dense-
Embeddings (bge-m3) sind mehrsprachig, BM25 aber nicht: "Mitarbeitende"
findet im englischen Bericht nichts. Statt einer zusätzlichen LLM-Übersetzung
(langsam, nicht deterministisch) ergänzt ein Glossar die Anfrage um die
Fachbegriffe der jeweils anderen Sprache.
"""

from __future__ import annotations

import re

# deutscher Begriff (Regex, lowercase) -> englische Entsprechung(en)
GLOSSARY: dict[str, str] = {
    r"umsatz(erlöse)?|erlöse": "revenue",
    r"mitarbeit(er|ende)|beschäftigte|belegschaft": "employees headcount",
    r"free cash ?flow": "free cash flow",
    r"dividende": "dividend per share",
    r"auslieferungen|ausgeliefert": "deliveries",
    r"auftragseingang": "orders",
    r"auftragsbestand": "backlog",
    r"ergebnis vor steuern|ebt": "income before income taxes",
    r"konzernergebnis|jahresüberschuss|konzernüberschuss": "net income",
    r"ergebnis je aktie": "earnings per share basic",
    r"marge": "margin",
    r"betriebsergebnis": "operating profit",
    r"forschung|entwicklung": "research and development",
    r"investitionen": "investments capital expenditures",
    r"nettoverschuldung|verschuldung": "net debt",
    r"risiken|risiko": "risks",
    r"prognose|ausblick": "outlook guidance",
    r"geschäftsjahr": "fiscal year",
    r"eigenkapital": "equity",
}


def expand_query(query: str) -> str:
    """Hängt englische (bzw. deutsche) Synonyme an die Anfrage an."""
    q = query.lower()
    extra: list[str] = []
    for de_pattern, en in GLOSSARY.items():
        if re.search(de_pattern, q):
            extra.append(en)
        elif en.split()[0] in q:  # englische Anfrage -> deutschen Kernbegriff ergänzen
            extra.append(de_pattern.split("|")[0].replace("(", "").replace(")", "").replace("?", ""))
    extra = [e for e in dict.fromkeys(extra) if e not in q]
    return f"{query} {' '.join(extra)}" if extra else query
