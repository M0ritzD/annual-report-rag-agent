"""LLM- und Embedding-Clients.

Ollama und Azure OpenAI sprechen beide das OpenAI-Protokoll. Deshalb genügt
ein einziger Code-Pfad (`openai`-SDK); nur die Client-Erzeugung unterscheidet
sich. Für Azure wird ohne API-Key automatisch Entra ID (z. B. die Managed
Identity der Container App) verwendet – keine Secrets im Container.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from openai import AzureOpenAI, OpenAI

from .config import Settings, get_settings


def _azure_client(s: Settings) -> AzureOpenAI:
    if not s.azure_openai_endpoint:
        raise RuntimeError("ARR_AZURE_OPENAI_ENDPOINT ist nicht gesetzt.")
    if s.azure_openai_api_key:
        return AzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version=s.azure_openai_api_version,
        )
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=s.azure_openai_api_version,
    )


def _ollama_client(s: Settings) -> OpenAI:
    return OpenAI(base_url=s.ollama_base_url, api_key="ollama")  # Key wird ignoriert


class ChatModel:
    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        if s.llm_provider == "azure_openai":
            self.client: OpenAI = _azure_client(s)
            self.model = s.azure_openai_chat_deployment
        else:
            self.client = _ollama_client(s)
            self.model = s.ollama_chat_model
        self.provider = s.llm_provider
        self.temperature = s.temperature

    def complete(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs).choices[0].message

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class Embedder:
    batch_size = 32

    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        if s.embedding_provider == "azure_openai":
            self.client: OpenAI = _azure_client(s)
            self.model = s.azure_openai_embedding_deployment
            self._dims: int | None = s.embedding_dimensions  # text-embedding-3-* kann kürzen
        else:
            self.client = _ollama_client(s)
            self.model = s.ollama_embedding_model
            self._dims = None
        self.label = f"{s.embedding_provider}:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs: dict = {"model": self.model, "input": batch}
            if self._dims:
                kwargs["dimensions"] = self._dims
            resp = self.client.embeddings.create(**kwargs)
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        arr = np.asarray(vectors, dtype=np.float32)
        # L2-normalisieren => Skalarprodukt == Kosinus-Ähnlichkeit
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache
def get_chat_model() -> ChatModel:
    return ChatModel()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
