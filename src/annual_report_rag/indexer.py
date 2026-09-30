"""Indexierungs-Pipeline: Dokumentquelle → Chunks → Embeddings → Vektorspeicher."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import Settings, get_settings
from .ingest import chunk_report, load_manifest
from .llm import Embedder
from .storage import sync_reports
from .vectorstore import get_vector_store


def build_index(
    settings: Settings | None = None,
    companies: list[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, int]:
    s = settings or get_settings()
    reports_dir, manifest_path = sync_reports(s)
    manifest = load_manifest(manifest_path)
    if companies:
        wanted = {c.lower() for c in companies}
        manifest = [m for m in manifest if m.company.lower() in wanted]

    store = get_vector_store(s)
    store.reset()
    embedder = Embedder(s)

    stats: dict[str, int] = {}
    for meta in manifest:
        chunks = list(chunk_report(reports_dir / meta.file, meta, s.chunk_size, s.chunk_overlap))
        # Kontext-Präfix verbessert die Embeddings (Unternehmen + Kapitel sind im Chunk selbst oft nicht genannt)
        texts = [f"{c.company} {c.year} | {c.section or ''}\n{c.text}" for c in chunks]
        vectors = []
        for i in range(0, len(texts), 64):
            vectors.append(embedder.embed(texts[i : i + 64]))
            if progress:
                progress(meta.company, min(i + 64, len(texts)), len(texts))
        store.add(chunks, np.vstack(vectors))
        stats[f"{meta.company} {meta.year}"] = len(chunks)
    return stats
