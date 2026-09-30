"""Dokumentquelle: lokaler Ordner oder Azure Blob Storage.

In Azure liegen die PDFs plus `reports.json` im Blob-Container. Beim
Indexieren werden sie in ein lokales Cache-Verzeichnis geladen.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings, get_settings


def _container(s: Settings):
    from azure.storage.blob import BlobServiceClient

    if s.azure_storage_connection_string:
        svc = BlobServiceClient.from_connection_string(s.azure_storage_connection_string)
    elif s.azure_storage_account_url:
        from azure.identity import DefaultAzureCredential

        svc = BlobServiceClient(s.azure_storage_account_url, credential=DefaultAzureCredential())
    else:
        raise RuntimeError("ARR_AZURE_STORAGE_ACCOUNT_URL oder ..._CONNECTION_STRING setzen.")
    return svc.get_container_client(s.azure_storage_container)


def upload_reports(settings: Settings | None = None) -> list[str]:
    """Lädt `data/reports/*.pdf` und das Manifest in den Blob-Container hoch."""
    s = settings or get_settings()
    container = _container(s)
    if not container.exists():
        container.create_container()
    uploaded = []
    files = [*sorted(s.reports_dir.glob("*.pdf")), s.manifest_path]
    for path in files:
        with path.open("rb") as f:
            container.upload_blob(path.name, f, overwrite=True)
        uploaded.append(path.name)
    return uploaded


def sync_reports(settings: Settings | None = None) -> tuple[Path, Path]:
    """Gibt (reports_dir, manifest_path) zurück – bei Azure nach dem Download."""
    s = settings or get_settings()
    if s.document_source == "local":
        return s.reports_dir, s.manifest_path
    cache = Path(".cache/blob")
    cache.mkdir(parents=True, exist_ok=True)
    container = _container(s)
    for blob in container.list_blobs():
        target = cache / blob.name
        if target.exists() and target.stat().st_size == blob.size:
            continue
        with target.open("wb") as f:
            container.download_blob(blob.name).readinto(f)
    return cache, cache / s.manifest_path.name
