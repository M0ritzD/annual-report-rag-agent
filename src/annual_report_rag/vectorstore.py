"""Vektorspeicher: lokal (NumPy + BM25) oder Azure AI Search.

Beide Implementierungen machen **hybride Suche** (semantische Vektoren +
lexikalisches BM25). Für Geschäftsberichte ist das wichtig: Fachbegriffe
("EBIT-Marge", "Free Cashflow Automobile") und exakte Kennzahlen werden
lexikalisch oft besser gefunden, Umschreibungen ("Wie profitabel war...")
semantisch. Die Ergebnislisten werden per Reciprocal Rank Fusion kombiniert.
"""

from __future__ import annotations

import contextlib
import re
from abc import ABC, abstractmethod
from collections.abc import Hashable
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings, get_settings
from .models import Chunk, SearchHit

_TOKEN = re.compile(r"[\wÄÖÜäöüß%€$.,-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.strip(".,-")
        if len(tok) > 1:
            tokens.append(tok)
    return tokens


def rrf(rankings: list[list[Hashable]], k: int = 60) -> dict[Hashable, float]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank)."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class VectorStore(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        query: str,
        vector: np.ndarray,
        k: int,
        company: str | None = None,
        year: int | None = None,
    ) -> list[SearchHit]: ...

    @abstractmethod
    def companies(self) -> list[tuple[str, int, int]]:
        """(Unternehmen, Jahr, Anzahl Chunks)"""


class LocalVectorStore(VectorStore):
    """Datei-basierter Index: `chunks.jsonl` + `vectors.npy`. Ideal für Demo & Tests."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._bm25: BM25Okapi | None = None
        self._load()

    # --- Persistenz ---
    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def _vectors_path(self) -> Path:
        return self.index_dir / "vectors.npy"

    def _load(self) -> None:
        if self._chunks_path.exists() and self._vectors_path.exists():
            with self._chunks_path.open(encoding="utf-8") as f:
                self.chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
            self.vectors = np.load(self._vectors_path)
            self._build_bm25()

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with self._chunks_path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        np.save(self._vectors_path, self.vectors)

    def _build_bm25(self) -> None:
        corpus = [tokenize(f"{c.section or ''} {c.text}") for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # --- API ---
    def reset(self) -> None:
        self.chunks, self.vectors, self._bm25 = [], np.zeros((0, 0), dtype=np.float32), None
        for p in (self._chunks_path, self._vectors_path):
            p.unlink(missing_ok=True)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
        self._build_bm25()
        self._save()

    def _mask(self, company: str | None, year: int | None) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if company:
            mask &= np.array([c.company.lower() == company.lower() for c in self.chunks])
        if year:
            mask &= np.array([c.year == year for c in self.chunks])
        return mask

    def search(self, query, vector, k, company=None, year=None):
        if not self.chunks:
            return []
        mask = self._mask(company, year)
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        pool = max(k * 5, 30)

        dense = self.vectors[candidates] @ vector
        dense_rank = candidates[np.argsort(-dense)[:pool]].tolist()

        rankings = [dense_rank]
        if self._bm25 is not None:
            bm25 = self._bm25.get_scores(tokenize(query))[candidates]
            rankings.append(candidates[np.argsort(-bm25)[:pool]].tolist())

        fused = sorted(rrf(rankings).items(), key=lambda kv: -kv[1])[:k]
        return [SearchHit(chunk=self.chunks[i], score=s) for i, s in fused]

    def companies(self):
        counts: dict[tuple[str, int], int] = {}
        for c in self.chunks:
            counts[(c.company, c.year)] = counts.get((c.company, c.year), 0) + 1
        return [(co, yr, n) for (co, yr), n in sorted(counts.items())]


class AzureSearchVectorStore(VectorStore):
    """Azure AI Search mit HNSW-Vektorfeld, BM25 und Filtern auf Unternehmen/Jahr.

    Die Hybrid-Suche (Text + Vektor, RRF) läuft serverseitig in Azure.
    """

    def __init__(self, s: Settings):
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        if not s.azure_search_endpoint:
            raise RuntimeError("ARR_AZURE_SEARCH_ENDPOINT ist nicht gesetzt.")
        cred = AzureKeyCredential(s.azure_search_api_key) if s.azure_search_api_key else DefaultAzureCredential()
        self.s = s
        self.index_client = SearchIndexClient(s.azure_search_endpoint, cred)
        self.client = SearchClient(s.azure_search_endpoint, s.azure_search_index, cred)

    def _index_definition(self):
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="company", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="year", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
            SimpleField(name="page", type=SearchFieldDataType.Int32),
            SimpleField(name="source_file", type=SearchFieldDataType.String),
            SearchableField(name="section", type=SearchFieldDataType.String, analyzer_name="de.microsoft"),
            SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="de.microsoft"),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.s.embedding_dimensions,
                vector_search_profile_name="hnsw-profile",
            ),
        ]
        return SearchIndex(
            name=self.s.azure_search_index,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
            ),
        )

    def reset(self) -> None:
        with contextlib.suppress(Exception):  # Index existiert evtl. noch nicht
            self.index_client.delete_index(self.s.azure_search_index)
        self.index_client.create_index(self._index_definition())

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        docs = [{**c.model_dump(), "vector": v.tolist()} for c, v in zip(chunks, vectors, strict=True)]
        for i in range(0, len(docs), 500):
            self.client.upload_documents(docs[i : i + 500])

    def search(self, query, vector, k, company=None, year=None):
        from azure.search.documents.models import VectorizedQuery

        filters = []
        if company:
            filters.append(f"company eq '{company.replace(chr(39), chr(39) * 2)}'")
        if year:
            filters.append(f"year eq {int(year)}")
        results = self.client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=vector.tolist(), k_nearest_neighbors=k * 5, fields="vector")],
            filter=" and ".join(filters) or None,
            top=k,
            select=["id", "company", "year", "page", "section", "text", "source_file"],
        )
        return [
            SearchHit(chunk=Chunk(**{f: r[f] for f in Chunk.model_fields}), score=r["@search.score"]) for r in results
        ]

    def companies(self):
        results = self.client.search(search_text="*", facets=["company,count:100", "year"], top=0)
        facets = results.get_facets() or {}
        years = [f["value"] for f in facets.get("year", [])]
        year = years[0] if len(years) == 1 else 0
        return [(f["value"], year, f["count"]) for f in facets.get("company", [])]


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    s = settings or get_settings()
    if s.vector_store == "azure_search":
        return AzureSearchVectorStore(s)
    return LocalVectorStore(s.index_dir)
