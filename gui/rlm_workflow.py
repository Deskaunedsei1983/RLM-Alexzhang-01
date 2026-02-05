"""
RLM-Zentrierter Dokumentations-Workflow

WICHTIG: Dieser Workflow nutzt das ECHTE RLM-Paradigma!

Statt Python-Schleifen ueber Dateien macht das LLM die Arbeit selbst:
1. LLM bekommt Dateipfade
2. LLM nutzt REPL um Dateien zu lesen
3. LLM chunked grosse Dateien SELBST
4. LLM ruft sich SELBST rekursiv auf (llm_query/llm_query_batched)
5. LLM aggregiert die Ergebnisse

Das ist der Kern des RLM-Paradigmas: Das Sprachmodell steuert den
gesamten Prozess durch Code-Ausfuehrung in der REPL.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Callable, List

from gui.workflow_engine import (
    WorkflowConfig,
    FileInfo,
    FileDiscovery,
)
from gui.rlm_backend import RLMBackend, RLMResult
from gui.memory_system import MemorySystem


@dataclass
class RLMWorkflowProgress:
    """Fortschrittsmeldung."""
    stage: str
    message: str
    progress: float
    detail: str = ""
    is_error: bool = False


@dataclass
class RLMWorkflowResult:
    """Ergebnis des RLM-Workflows."""
    success: bool
    documentation: str
    files_analyzed: int
    execution_time: float
    iterations: int = 0
    error: Optional[str] = None


# =============================================================================
# RLM Prompts - Das LLM steuert den Prozess
# =============================================================================

RLM_ANALYSIS_PROMPT = '''Du bist ein Code-Analyse-Experte. Du hast Zugriff auf eine Python REPL.

DEINE AUFGABE: Analysiere das Projekt im Verzeichnis "{source_path}"

DATEIEN IM PROJEKT ({file_count} Dateien):
{file_list}

WICHTIG - NUTZE DIE REPL:
Du MUSST Python-Code in der REPL ausfuehren um:
1. Dateien zu lesen
2. Grosse Dateien zu chunken
3. Dich selbst mit llm_query() rekursiv aufzurufen fuer Teilanalysen

BEISPIEL fuer grosse Dateien:
```repl
# Datei lesen
with open("/pfad/zur/datei.py", "r") as f:
    content = f.read()

# Falls gross, chunken und rekursiv analysieren
if len(content) > 5000:
    chunks = [content[i:i+4000] for i in range(0, len(content), 3500)]
    analyses = []
    for i, chunk in enumerate(chunks):
        result = llm_query(f"Analysiere Teil {{i+1}} dieser Python-Datei:\\n{{chunk}}")
        analyses.append(result)
    combined = "\\n".join(analyses)
    final = llm_query(f"Fasse diese Teilanalysen zusammen:\\n{{combined}}")
    print(final)
else:
    result = llm_query(f"Analysiere diese Datei:\\n{{content}}")
    print(result)
```

BEISPIEL fuer viele Dateien (batched):
```repl
files_to_analyze = {file_paths}

# Batch-Analyse fuer Effizienz
prompts = []
for f in files_to_analyze[:10]:  # In Gruppen verarbeiten
    with open(f, "r", errors="ignore") as file:
        content = file.read()[:3000]  # Erste 3000 Zeichen
    prompts.append(f"Kurze Analyse von {{f}}:\\n{{content}}")

results = llm_query_batched(prompts)
for f, r in zip(files_to_analyze[:10], results):
    print(f"=== {{f}} ===\\n{{r}}\\n")
```

DEIN ZIEL:
1. Lies die wichtigsten Dateien (priorisiere: README, main.py, __init__.py, etc.)
2. Analysiere sie mit llm_query() oder llm_query_batched()
3. Erstelle eine VOLLSTAENDIGE Projektdokumentation

AUSGABEFORMAT (am Ende):
```
# Projektdokumentation

## Uebersicht
[Was macht das Projekt?]

## Struktur
[Hauptverzeichnisse und deren Zweck]

## Kernkomponenten
[Die wichtigsten Module/Dateien]

## Abhaengigkeiten
[Externe Bibliotheken]

## Verwendung
[Wie benutzt man das Projekt?]
```

Beginne JETZT mit der Analyse. Fuehre Code in der REPL aus!
'''

RLM_DEEP_ANALYSIS_PROMPT = '''Analysiere diese Datei DETAILLIERT:

DATEI: {filepath}
GROESSE: {size} Bytes

ANWEISUNGEN:
1. Lies die Datei mit Python in der REPL
2. Falls > 10000 Zeichen: Teile in Chunks und analysiere jeden mit llm_query()
3. Falls <= 10000 Zeichen: Analysiere direkt

```repl
with open("{filepath}", "r", errors="ignore") as f:
    content = f.read()

print(f"Dateigroesse: {{len(content)}} Zeichen")

if len(content) > 10000:
    # Chunking
    chunk_size = 8000
    overlap = 500
    chunks = []
    for i in range(0, len(content), chunk_size - overlap):
        chunks.append(content[i:i+chunk_size])

    print(f"Teile in {{len(chunks)}} Chunks auf")

    analyses = []
    for i, chunk in enumerate(chunks):
        prompt = f"Analysiere Teil {{i+1}}/{{len(chunks)}} der Datei {filepath}:\\n{{chunk}}"
        result = llm_query(prompt)
        analyses.append(f"Teil {{i+1}}:\\n{{result}}")
        print(f"Chunk {{i+1}} analysiert")

    # Zusammenfuehren
    combined = "\\n\\n".join(analyses)
    final = llm_query(f"Fasse diese {{len(chunks)}} Teilanalysen zu einer Gesamtanalyse zusammen:\\n{{combined}}")
    print("=== FINALE ANALYSE ===")
    print(final)
else:
    # Direkte Analyse
    result = llm_query(f"Analysiere diese Datei detailliert:\\n{{content}}")
    print(result)
```

Fuehre den Code aus und gib die Analyse zurueck.
'''


class RLMCentricWorkflow:
    """
    RLM-zentrierter Workflow.

    Das LLM steuert den gesamten Analyseprozess durch:
    - REPL Code-Ausfuehrung
    - Rekursive Selbst-Aufrufe (llm_query)
    - Batch-Verarbeitung (llm_query_batched)

    Python macht nur:
    - Dateien auflisten
    - RLM starten
    - Ergebnis speichern
    """

    def __init__(
        self,
        backend: RLMBackend,
        config: WorkflowConfig,
        memory: Optional[MemorySystem] = None,
    ):
        self.backend = backend
        self.config = config
        self.memory = memory
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self) -> Iterator[RLMWorkflowProgress]:
        """
        Fuehrt den RLM-zentrierten Workflow aus.

        WICHTIG: Nur WENIGE RLM-Aufrufe!
        Das LLM macht die Iteration intern.
        """
        start_time = datetime.now()

        # Phase 1: Dateien sammeln (Python-seitig, schnell)
        yield RLMWorkflowProgress(
            stage="discover",
            message="Sammle Dateipfade...",
            progress=0.05,
        )

        files = FileDiscovery.discover(
            self.config.source_path,
            self.config,
        )

        if not files:
            yield RLMWorkflowProgress(
                stage="error",
                message="Keine Dateien gefunden",
                progress=0.1,
                is_error=True,
            )
            return

        yield RLMWorkflowProgress(
            stage="discover",
            message=f"{len(files)} Dateien gefunden",
            progress=0.1,
            detail=f"Uebergebe an RLM zur Analyse",
        )

        # Phase 2: Dateipfade fuer RLM vorbereiten
        text_files = [f for f in files if not f.is_binary]

        # Sortiere nach Wichtigkeit
        priority_files = self._prioritize_files(text_files)

        # Erstelle Dateiliste fuer RLM
        file_list = self._format_file_list(priority_files[:200])  # Max 200 fuer Prompt
        file_paths = [f.path for f in priority_files[:50]]  # Top 50 Pfade

        yield RLMWorkflowProgress(
            stage="prepare",
            message="Bereite RLM-Prompt vor...",
            progress=0.15,
            detail=f"{len(priority_files)} Dateien priorisiert",
        )

        # Phase 3: EINEN RLM-Aufruf - das LLM macht den Rest!
        yield RLMWorkflowProgress(
            stage="analyze",
            message="RLM analysiert Projekt (LLM steuert)...",
            progress=0.2,
            detail="Das LLM nutzt REPL und llm_query intern",
        )

        prompt = RLM_ANALYSIS_PROMPT.format(
            source_path=self.config.source_path,
            file_count=len(text_files),
            file_list=file_list,
            file_paths=str(file_paths),
        )

        # DER EINE GROSSE RLM-AUFRUF
        # Das LLM iteriert intern ueber Dateien!
        result = self.backend.run_completion(
            context=f"Projektpfad: {self.config.source_path}",
            aufgabe=prompt,
        )

        if not result.success:
            yield RLMWorkflowProgress(
                stage="error",
                message=f"RLM-Fehler: {result.error}",
                progress=0.5,
                is_error=True,
            )
            return

        yield RLMWorkflowProgress(
            stage="analyze",
            message="RLM-Analyse abgeschlossen",
            progress=0.9,
            detail=f"{result.iterations} Iterationen",
        )

        # Phase 4: Ergebnis speichern
        documentation = result.response

        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        doc_file = output_dir / "DOCUMENTATION.md"
        doc_file.write_text(documentation, encoding='utf-8')

        # Im Memory speichern
        if self.memory:
            self.memory.add_long_term(
                key=f"rlm_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content=documentation[:2000],
                source=self.config.source_path,
                entry_type="rlm_documentation",
                relevance=1.0,
            )

        execution_time = (datetime.now() - start_time).total_seconds()

        yield RLMWorkflowProgress(
            stage="complete",
            message="Dokumentation erstellt!",
            progress=1.0,
            detail=f"Gespeichert: {doc_file}",
        )

        # Speichere Ergebnis
        self.result = RLMWorkflowResult(
            success=True,
            documentation=documentation,
            files_analyzed=len(text_files),
            execution_time=execution_time,
            iterations=result.iterations,
        )

    def analyze_single_file(self, filepath: str) -> RLMResult:
        """
        Analysiert eine einzelne Datei mit RLM.

        Das LLM entscheidet selbst ueber Chunking.
        """
        size = Path(filepath).stat().st_size

        prompt = RLM_DEEP_ANALYSIS_PROMPT.format(
            filepath=filepath,
            size=size,
        )

        return self.backend.run_completion(
            context=f"Datei zur Analyse: {filepath}",
            aufgabe=prompt,
        )

    def _prioritize_files(self, files: List[FileInfo]) -> List[FileInfo]:
        """Sortiert Dateien nach Wichtigkeit."""
        priority_names = [
            'README', 'readme', 'main', 'app', 'index', '__init__',
            'setup', 'config', 'settings', 'models', 'views', 'routes',
        ]

        def priority_score(f: FileInfo) -> int:
            score = 0
            name_lower = f.name.lower()

            # Hohe Prioritaet fuer wichtige Dateien
            for i, pn in enumerate(priority_names):
                if pn in name_lower:
                    score += (len(priority_names) - i) * 10

            # Python/JS/TS Dateien bevorzugen
            if f.extension in ['.py', '.js', '.ts', '.tsx', '.jsx']:
                score += 5

            # Groessere Dateien sind oft wichtiger
            score += min(f.size // 1000, 10)

            return score

        return sorted(files, key=priority_score, reverse=True)

    def _format_file_list(self, files: List[FileInfo]) -> str:
        """Formatiert Dateiliste fuer LLM."""
        lines = []
        for f in files:
            try:
                rel_path = str(Path(f.path).relative_to(self.config.source_path))
            except ValueError:
                rel_path = f.name
            lines.append(f"- {rel_path} ({f.size} bytes)")
        return "\n".join(lines)


def run_rlm_workflow(
    source_path: str,
    backend: RLMBackend,
    output_path: str = "./documentation",
) -> RLMWorkflowResult:
    """
    Convenience-Funktion fuer RLM-Workflow.

    Args:
        source_path: Zu analysierendes Verzeichnis
        backend: RLM Backend
        output_path: Ausgabeverzeichnis

    Returns:
        RLMWorkflowResult
    """
    config = WorkflowConfig(
        source_path=source_path,
        output_path=output_path,
    )

    workflow = RLMCentricWorkflow(backend, config)

    for progress in workflow.run():
        print(f"[{progress.stage}] {progress.message}")

    return workflow.result
