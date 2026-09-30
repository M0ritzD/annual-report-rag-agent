"""PDF-Ingestion: Text extrahieren, Kopf-/Fußzeilen entfernen, in Chunks zerlegen.

Geschäftsberichte haben Eigenheiten, die ein naives Chunking verschlechtern:

* Auf jeder Seite wiederholte Navigationsleisten ("An unsere Stakeholder |
  Konzernabschluss | ...") und Seitenzahlen – diese werden über eine
  Häufigkeitsanalyse erkannt und entfernt.
* Die Kapitelstruktur steckt im PDF-Inhaltsverzeichnis (Outline). Sie wird
  jeder Seite zugeordnet, damit Antworten "Konzernabschluss → Segmente"
  zitieren können und die Suche den Kontext kennt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pymupdf

from .models import Chunk, ReportMeta

_WS = re.compile(r"[ \t\u00a0]+")
_PAGE_NO = re.compile(r"^\s*\d{1,3}(\s*/\s*\d{1,3})?\s*$")


def load_manifest(path: Path) -> list[ReportMeta]:
    return [ReportMeta(**r) for r in json.loads(path.read_text(encoding="utf-8"))]


def _normalize_line(line: str) -> str:
    return _WS.sub(" ", line).strip()


def _page_lines(doc: pymupdf.Document) -> list[list[str]]:
    pages = []
    for page in doc:
        # Bewusst ohne sort=True: die Content-Stream-Reihenfolge respektiert bei
        # mehrspaltigen Layouts die Lesereihenfolge, sort=True mischt Spalten.
        text = str(page.get_text("text"))
        pages.append([_normalize_line(line) for line in text.splitlines()])
    return pages


def find_boilerplate(pages: list[list[str]], min_share: float = 0.3) -> set[str]:
    """Zeilen, die auf mindestens `min_share` aller Seiten vorkommen (Header/Footer)."""
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({line for line in lines if line})
    threshold = max(3, int(len(pages) * min_share))
    return {line for line, n in counts.items() if n >= threshold}


def section_map(doc: pymupdf.Document) -> dict[int, str]:
    """Ordnet jeder Seite (1-basiert) den Pfad im Inhaltsverzeichnis zu."""
    toc = [(lvl, title.strip(), page) for lvl, title, page in doc.get_toc(simple=True) if page > 0]
    result: dict[int, str] = {}
    if not toc:
        return result
    # Viele Berichte haben einen einzigen Wurzelknoten ("BMW GROUP BERICHT 2024") -> überspringen
    if sum(1 for lvl, _, _ in toc if lvl == 1) == 1:
        toc = [(lvl - 1, title, page) for lvl, title, page in toc if lvl > 1]
    stack: list[str] = []
    entries_by_page: dict[int, list[tuple[int, str]]] = {}
    for lvl, title, page in toc:
        entries_by_page.setdefault(page, []).append((lvl, title))
    for page_no in range(1, doc.page_count + 1):
        for lvl, title in entries_by_page.get(page_no, []):
            if lvl > 2:  # nur die zwei obersten Ebenen – tiefer wird es unübersichtlich
                continue
            stack = stack[: lvl - 1] + [title]
        if stack:
            result[page_no] = " → ".join(stack)
    return result


def clean_page(lines: list[str], boilerplate: set[str]) -> str:
    kept = [line for line in lines if line and line not in boilerplate and not _PAGE_NO.match(line)]
    text = "\n".join(kept)
    # Silbentrennung am Zeilenende zusammenführen ("Umsatz-\nerlöse" -> "Umsatzerlöse")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Absatz-/zeilenbewusstes Chunking mit Überlappung."""
    if len(text) <= size:
        return [text] if text else []
    units = [u for u in re.split(r"\n{2,}|\n", text) if u.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > size:  # sehr lange Zeilen hart teilen
            head, unit = unit[:size], unit[size - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(unit) + 1 > size and current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # Überlappung an Wortgrenze beginnen lassen
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{tail}\n{unit}" if tail else unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_report(pdf_path: Path, meta: ReportMeta, size: int, overlap: int) -> Iterator[Chunk]:
    with pymupdf.open(pdf_path) as doc:
        pages = _page_lines(doc)
        boilerplate = find_boilerplate(pages)
        sections = section_map(doc)
        for page_no, lines in enumerate(pages, start=1):
            text = clean_page(lines, boilerplate)
            if len(text) < 80:  # Deckblätter, Trennseiten, reine Grafiken
                continue
            for i, piece in enumerate(split_text(text, size, overlap)):
                digest = hashlib.sha1(f"{meta.file}:{page_no}:{i}".encode()).hexdigest()[:16]
                yield Chunk(
                    id=digest,
                    company=meta.company,
                    year=meta.year,
                    page=page_no,
                    section=sections.get(page_no),
                    text=piece,
                    source_file=meta.file,
                )
