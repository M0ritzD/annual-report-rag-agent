# 📊 Annual Report RAG Agent

**Ein KI-Agent, der Geschäftsberichte analysiert – mit Quellenangaben auf Seitenebene.**
Läuft **vollständig lokal** mit [Ollama](https://ollama.com) (keine Daten verlassen den Rechner) oder **cloud-nativ in Azure**
(Azure OpenAI · Azure AI Search · Blob Storage · Container Apps) – umschaltbar per Konfiguration, ohne Code-Änderung.

[![CI](https://github.com/M0ritzD/annual-report-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/M0ritzD/annual-report-rag-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Azure%20OpenAI-black)
![Azure](https://img.shields.io/badge/IaC-Bicep-0078D4)

![Screenshot der Web-UI](docs/screenshot.png)

```text
$ arr ask "Wie hoch war der Free Cashflow der SAP 2024 und wie hat er sich zum Vorjahr verändert?"
🔧 search_reports({"company": "SAP", "query": "Free Cash Flow 2024", "year": 2024})
╭─────────────── Antwort · ollama:qwen2.5:7b-instruct-q4_k_m ───────────────╮
│ Der Free Cashflow der SAP im Geschäftsjahr 2024 betrug 4.113 Mio. €, was  │
│ um 980 Mio. € (oder um 19%) weniger als im Vorjahr lag [3].               │
│  • 2023: 5.093 Mio. €                                                     │
│  • 2024: 4.113 Mio. €                                                     │
╰───────────────────────────────────────────────────────────────────────────╯
 [3]  SAP 2024, S. 67 – Zusammengefasster Konzernlagebericht → Steuerungssystem
```

---

## Features

| | |
|---|---|
| 🤖 **Agent statt Einzel-Prompt** | Tool-Calling-Loop: Das LLM entscheidet selbst, ob es sucht, pro Unternehmen filtert oder rechnet (`search_reports`, `calculate`, `list_reports`). Unternehmensvergleiche werden in Teilrecherchen zerlegt. |
| 🔎 **Hybride Suche** | Dense-Vektoren (`bge-m3`, mehrsprachig) + BM25, kombiniert per *Reciprocal Rank Fusion*. Kennzahlen wie „EBT-Marge“ oder „4.113 Mio. €“ werden lexikalisch gefunden, Umschreibungen semantisch. |
| 📑 **Berichtsspezifische Ingestion** | Wiederkehrende Kopf-/Navigationszeilen werden statistisch erkannt und entfernt, Silbentrennung zusammengeführt, jede Seite bekommt ihr Kapitel aus dem PDF-Inhaltsverzeichnis. |
| 🧾 **Nachprüfbare Antworten** | Jede Aussage wird mit `[n]` belegt; nur tatsächlich gefundene Quellen werden ausgegeben (halluzinierte Referenzen werden verworfen). |
| 🧮 **Exaktes Rechnen** | Wachstumsraten & Margen über einen sicheren AST-Taschenrechner statt „Kopfrechnen“ des LLM. |
| 🔁 **Provider-agnostisch** | Ollama und Azure OpenAI über dieselbe OpenAI-kompatible Schnittstelle; lokaler Index oder Azure AI Search; lokale Dateien oder Blob Storage. |
| ☁️ **Infrastructure as Code** | Komplette Azure-Umgebung als Bicep inkl. Managed Identity & RBAC – **keine API-Keys im Container**. |
| ✅ **Evaluation & CI** | Fragenkatalog mit manuell verifizierten Kennzahlen, Metriken für Retrieval, Faktentreue, Zitate und Latenz; Unit-Tests ohne LLM in GitHub Actions. |

## Architektur

```mermaid
flowchart LR
    subgraph Ingestion
        PDF[Geschäftsberichte<br/>PDF] --> P[PyMuPDF<br/>Header-Filter · Kapitel · Chunking]
        P --> E[Embeddings<br/>bge-m3 / text-embedding-3-small]
        E --> IDX[(Vektorindex<br/>lokal: NumPy + BM25<br/>Azure: AI Search HNSW + BM25)]
    end
    subgraph Agent
        U[Frage] --> L{LLM<br/>Ollama / Azure OpenAI}
        L -- search_reports --> IDX
        L -- calculate --> C[AST-Rechner]
        IDX -- nummerierte Quellen --> L
        C --> L
        L --> A[Antwort mit Zitaten<br/>Unternehmen · Jahr · Seite]
    end
    A --> UI[Web-UI / REST-API / CLI]
```

**Deployment-Modi** (nur Umgebungsvariablen, siehe [`.env.example`](.env.example)):

| Modus | LLM | Embeddings | Index | Dokumente | Anwendungsfall |
|---|---|---|---|---|---|
| **Lokal** | Ollama | Ollama | lokal | lokal | Entwicklung, vertrauliche Berichte, 0 € |
| **Azure** | Azure OpenAI | Azure OpenAI | AI Search | Blob Storage | Produktiv, skalierbar, Managed Identity |
| **Hybrid** | Ollama | Ollama | AI Search | Blob Storage | Zentrale Wissensbasis, Inferenz on-prem (DSGVO) |

## Schnellstart (lokal)

Voraussetzungen: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), ~6 GB RAM frei.

```bash
git clone https://github.com/M0ritzD/annual-report-rag-agent && cd annual-report-rag-agent
uv sync

ollama pull qwen2.5:7b-instruct-q4_k_m   # Chat-Modell mit Tool-Calling
ollama pull bge-m3                       # mehrsprachige Embeddings (DE/EN)

./scripts/download_reports.sh            # SAP, BMW, Siemens 2024 (öffentliche PDFs)
uv run arr index                         # ~10 Min. auf einem MacBook M2 (3.771 Chunks)

uv run arr serve                         # → http://127.0.0.1:8000
uv run arr ask "Vergleiche den Free Cashflow von SAP und BMW 2024."
uv run arr chat                          # interaktiv mit Gesprächsverlauf
```

Eigene Berichte: PDF nach `data/reports/` legen, Eintrag in [`data/reports.json`](data/reports.json) ergänzen, `arr index`.

## Deployment in Azure

```bash
az login
./infra/deploy.sh rg-annual-report-rag swedencentral
```

Das Skript
1. deployt [`infra/main.bicep`](infra/main.bicep): Storage Account, Azure AI Search (Free-Tier), Azure OpenAI (`gpt-4o-mini`, `text-embedding-3-small`), Log Analytics, Container Apps Environment und die Container App mit **System-Managed-Identity**,
2. vergibt RBAC-Rollen (Blob Data Reader, Search Index Data Contributor, Cognitive Services OpenAI User) – der Container authentifiziert sich per Entra ID, ohne Secrets,
3. lädt die PDFs in Blob Storage und baut den Index in Azure AI Search auf,
4. gibt die öffentliche URL der Web-UI aus.

Das Container-Image wird von GitHub Actions nach `ghcr.io` gebaut. Die App skaliert auf 0 Replikas, im Leerlauf fallen praktisch nur Speicherkosten an.

**Kosten-Schätzung Demo:** AI Search Free 0 € · Container Apps im Free-Kontingent · Azure OpenAI pay-per-token (Indexierung aller drei Berichte mit `text-embedding-3-small` ≈ 0,05 €).

**Aufräumen:** `az group delete -n rg-annual-report-rag`

## Evaluation

`uv run arr eval` fährt 16 Fragen aus [`eval/questions.jsonl`](eval/questions.jsonl), deren Soll-Antworten manuell in den PDFs verifiziert wurden (Umsatz, Free Cashflow, Mitarbeitende, Dividende, … je Unternehmen).

<!-- EVAL_RESULTS -->

Gemessen wird:
- **Retrieval-Trefferquote** – ist eine Seite mit der gesuchten Kennzahl unter den Top-6-Treffern?
- **Antwort-Genauigkeit** – enthält die Antwort alle erwarteten Werte? (Zahlenformate wie `34.176`, `34,2 Mrd.` werden normalisiert)
- **Zitierquote** – belegt die Antwort ihre Aussagen mit Quellen?

## Projektstruktur

```
src/annual_report_rag/
├── ingest.py        PDF → bereinigte, kapitelbewusste Chunks
├── llm.py           Chat- & Embedding-Clients (Ollama | Azure OpenAI, Entra ID)
├── vectorstore.py   Hybride Suche: lokal (NumPy + BM25 + RRF) | Azure AI Search
├── storage.py       Dokumentquelle: lokal | Azure Blob Storage
├── indexer.py       Ingestion-Pipeline
├── tools.py         Agenten-Werkzeuge + sicherer Rechner
├── agent.py         Tool-Calling-Loop, Forced Retrieval, Zitat-Validierung
├── evaluation.py    Metriken
├── api.py           FastAPI (REST + Web-UI)
├── cli.py           Typer-CLI: index · ask · chat · serve · eval · upload
└── static/index.html
infra/               Bicep + Deploy-Skript
tests/               Unit-Tests (ohne LLM/Netzwerk)
eval/                Fragenkatalog
```

## Designentscheidungen

- **Warum ein Agent und nicht „klassisches“ RAG?** Bei Fragen wie „Vergleiche SAP und BMW“ liefert eine einzelne Vektorsuche meist nur Treffer eines Unternehmens. Der Agent sucht gezielt pro Unternehmen, in der Sprache des jeweiligen Berichts, und rechnet Differenzen exakt nach.
- **Forced Retrieval:** Kleine lokale Modelle (7B) überspringen manchmal den Tool-Call und antworten aus dem Gedächtnis. Der Agent erkennt das und führt die Suche selbst aus.
- **`bge-m3` statt `nomic-embed-text`:** Die Berichte sind deutsch *und* englisch; `bge-m3` ist mehrsprachig und findet deutsche Fragen auch im englischen Siemens-Bericht.
- **Hybrid statt nur Vektoren:** Finanzkennzahlen sind häufig Tabellenzeilen mit wenig Kontext – dort ist BM25 der Vektorsuche klar überlegen.
- **Managed Identity:** Keine Keys in Umgebungsvariablen oder Images; Rechte sind auf Ressourcenebene minimal vergeben.

## Tech-Stack

Python 3.11 · uv · PyMuPDF · NumPy · rank-bm25 · OpenAI SDK · Ollama · FastAPI · Typer · Rich · Pydantic Settings ·
Azure OpenAI · Azure AI Search · Azure Blob Storage · Azure Container Apps · Entra ID / Managed Identity · Bicep · Docker · GitHub Actions · pytest · Ruff

## Hinweis zu den Daten

Die Geschäftsberichte sind öffentliche Dokumente der jeweiligen Unternehmen und werden **nicht** im Repository mitgeliefert, sondern über `scripts/download_reports.sh` von den offiziellen Investor-Relations-Seiten geladen. Die Antworten des Agenten sind keine Anlageberatung.

## Lizenz

MIT
