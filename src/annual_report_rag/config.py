"""Zentrale Konfiguration über Umgebungsvariablen bzw. `.env`.

Jede Komponente (LLM, Embeddings, Vektorspeicher, Dokumentablage) ist
austauschbar, sodass dasselbe Programm vollständig lokal (Ollama + lokaler
Index) oder in Azure (Azure OpenAI + Azure AI Search + Blob Storage) läuft –
oder in einer hybriden Kombination ("Daten in Azure, Inferenz lokal").
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ARR_", extra="ignore")

    # --- Provider-Auswahl -------------------------------------------------
    llm_provider: Literal["ollama", "azure_openai"] = "ollama"
    embedding_provider: Literal["ollama", "azure_openai"] = "ollama"
    vector_store: Literal["local", "azure_search"] = "local"
    document_source: Literal["local", "azure_blob"] = "local"

    # --- Ollama (lokal) ---------------------------------------------------
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model: str = "qwen2.5:7b-instruct-q4_k_m"
    ollama_embedding_model: str = "bge-m3"

    # --- Azure OpenAI -----------------------------------------------------
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None  # leer => Entra ID (Managed Identity)
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # --- Azure AI Search --------------------------------------------------
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None  # leer => Entra ID
    azure_search_index: str = "annual-reports"

    # --- Azure Blob Storage -----------------------------------------------
    azure_storage_account_url: str | None = None
    azure_storage_connection_string: str | None = None
    azure_storage_container: str = "reports"

    # --- Pfade ------------------------------------------------------------
    reports_dir: Path = Path("data/reports")
    manifest_path: Path = Path("data/reports.json")
    index_dir: Path = Path("data/index")

    # --- RAG-Parameter ----------------------------------------------------
    embedding_dimensions: int = 1024  # bge-m3 = 1024, text-embedding-3-small = 1536
    chunk_size: int = Field(default=1200, description="Zielgröße eines Chunks in Zeichen")
    chunk_overlap: int = 200
    top_k: int = 6
    max_agent_steps: int = 5
    temperature: float = 0.1


@lru_cache
def get_settings() -> Settings:
    return Settings()
