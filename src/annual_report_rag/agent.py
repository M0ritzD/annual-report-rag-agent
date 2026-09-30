"""Der Analyse-Agent: ein ReAct-artiger Tool-Calling-Loop.

Ablauf pro Frage:
1. Das LLM entscheidet, welche Werkzeuge es braucht (Suche je Unternehmen,
   Rechnen, Übersicht) – auch mehrfach hintereinander.
2. Werkzeug-Ergebnisse werden als nummerierte Quellen zurückgegeben.
3. Das LLM schreibt die Antwort und zitiert jede Aussage mit [n].
4. Nur tatsächlich zitierte Quellen werden ausgegeben.

Kleine lokale Modelle rufen Werkzeuge nicht immer zuverlässig auf. Deshalb
gibt es zwei Absicherungen: Wird beim ersten Schritt nicht gesucht, führt der
Agent selbst eine Suche mit der Originalfrage aus ("forced retrieval"). Und
wird das Schrittlimit erreicht, wird eine Antwort ohne weitere Tools erzwungen.
"""

from __future__ import annotations

import json
import re
import time

from .config import Settings, get_settings
from .llm import ChatModel, Embedder
from .models import AgentStep, Answer
from .tools import TOOL_SCHEMAS, ToolBox
from .vectorstore import VectorStore, get_vector_store

SYSTEM_PROMPT = """Du bist ein sorgfältiger Finanzanalyst. Du beantwortest Fragen zu Geschäftsberichten \
ausschließlich auf Basis der Textstellen, die du über das Werkzeug `search_reports` findest.

Regeln:
- Suche IMMER zuerst. Bei Vergleichen suche für jedes Unternehmen separat (Parameter `company`).
- Siemens-Bericht ist englisch: suche dort mit englischen Begriffen (z. B. "revenue", "free cash flow").
- Nenne konkrete Zahlen mit Einheit und Geschäftsjahr. Rechne mit `calculate`, nie im Kopf.
- Bevorzuge Konzernwerte aus Kennzahlen-Übersichten ("in Zahlen", "Kennzahlen", "Key figures") gegenüber \
Werten einzelner Marken, Segmente oder Tochtergesellschaften. Gibt es mehrere Definitionen \
(z. B. IFRS vs. Non-IFRS, Personenzahl vs. Vollzeitäquivalente), nenne den Konzern-/IFRS-Wert und erwähne die Abweichung.
- Belege jede Aussage direkt mit der Quellennummer in eckigen Klammern, z. B. "Der Umsatz stieg auf 34,2 Mrd. € [3]."
- Steht etwas nicht in den Quellen, sag das offen. Erfinde keine Zahlen.
- Antworte auf Deutsch, strukturiert und prägnant (Stichpunkte oder kurze Tabelle sind willkommen)."""

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def extract_citations(text: str) -> list[int]:
    refs: list[int] = []
    for group in _CITATION.findall(text):
        for n in re.split(r"[,;]", group):
            n_int = int(n.strip())
            if n_int not in refs:
                refs.append(n_int)
    return refs


def _parse_args(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        args = json.loads(raw)
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        return {}


class ReportAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        chat: ChatModel | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ):
        self.s = settings or get_settings()
        self.chat = chat or ChatModel(self.s)
        self.embedder = embedder or Embedder(self.s)
        self.store = store or get_vector_store(self.s)

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        t0 = time.perf_counter()
        tools = ToolBox(self.store, self.embedder, self.s, question=question)
        steps: list[AgentStep] = []
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history or []
        messages.append({"role": "user", "content": question})

        final = ""
        for step in range(self.s.max_agent_steps):
            last_step = step == self.s.max_agent_steps - 1
            msg = self.chat.complete(messages, tools=None if last_step else TOOL_SCHEMAS)
            calls = msg.tool_calls or []

            if not calls and not tools.sources and step == 0:
                # Forced retrieval: Modell wollte ohne Recherche antworten
                result = tools.search_reports(question)
                steps.append(
                    AgentStep(
                        tool="search_reports",
                        arguments={"query": question, "forced": True},
                        result_preview=result[:200],
                    )
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Ergebnisse von search_reports:\n\n{result}\n\nBeantworte jetzt die Frage mit Quellenangaben [n].",
                    }
                )
                continue

            if not calls:
                final = msg.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in calls
                    ],
                }
            )
            for c in calls:
                args = _parse_args(c.function.arguments)
                result = tools.call(c.function.name, args)
                steps.append(AgentStep(tool=c.function.name, arguments=args, result_preview=result[:200]))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        else:
            final = final or "Ich konnte die Frage im vorgegebenen Schrittlimit nicht beantworten."

        cited = [r for r in extract_citations(final) if r in tools.sources]
        return Answer(
            question=question,
            answer=final.strip(),
            sources=tools.as_sources(cited),
            steps=steps,
            model=self.chat.label,
            latency_s=round(time.perf_counter() - t0, 2),
        )
