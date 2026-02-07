"""
RLM-Zentrierter Dokumentations-Workflow

WICHTIG: Dieser Workflow nutzt das ECHTE RLM-Paradigma!

Statt Python-Schleifen ueber Dateien macht das LLM die Arbeit selbst:
1. Projektverzeichnis wird in Docker-Container gemountet
2. LLM bekommt Dateipfade
3. LLM nutzt REPL um Dateien zu lesen
4. LLM chunked grosse Dateien SELBST
5. LLM ruft sich SELBST rekursiv auf (llm_query/llm_query_batched)
6. LLM aggregiert die Ergebnisse

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

DEINE AUFGABE: Analysiere das Projekt "{project_name}"

WICHTIG: Das Projektverzeichnis ist gemountet unter: /project
Du kannst ALLE Dateien direkt lesen mit Python!

PROJEKT-STATISTIK:
- Gesamtzahl Dateien: {total_files}
- Kategorien: {categories}

WICHTIGSTE DATEIEN (Top 50):
{file_list}

=== KRITISCHE REGELN ZUR KONTEXTGROESSE ===
1. Sende NIEMALS mehr als 3000 Zeichen an llm_query()!
2. Bei grossen Dateien: Lies nur die ersten 2000 Zeichen
3. Fasse Ergebnisse KURZ zusammen (max 500 Zeichen)
4. Analysiere max 10-20 wichtige Dateien, nicht alle!

BEISPIEL - Dateien auflisten (im Container):
```repl
import os
files = []
for root, dirs, files_list in os.walk("/project"):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", "node_modules", ".git", ".venv"]]
    for f in files_list[:5]:
        files.append(os.path.join(root, f))
print(f"Gefunden: {{len(files)}} Dateien")
print(files[:20])
```

BEISPIEL - Datei lesen (MIT LIMIT!):
```repl
with open("/project/README.md", "r", errors="ignore") as f:
    content = f.read()[:2000]  # MAX 2000 Zeichen!
print(f"Inhalt ({{len(content)}} Zeichen):")
print(content)
```

BEISPIEL - Rekursive Analyse (KURZ halten!):
```repl
# WICHTIG: Nur kurze Zusammenfassungen senden!
filepath = "/project/src/main.py"
with open(filepath, "r", errors="ignore") as f:
    content = f.read()[:2000]

# Kurze Analyse anfordern
result = llm_query(f"Kurze Zusammenfassung (max 200 Woerter) von {{filepath}}:\\n{{content}}")
print(f"Analyse: {{result[:500]}}")  # Nur ersten 500 Zeichen ausgeben
```

BEISPIEL - Mehrere Dateien effizient:
```repl
import os
summaries = []

# Nur die 10 wichtigsten Dateien
important = ["README.md", "main.py", "app.py", "index.js", "package.json"]
for root, dirs, files in os.walk("/project"):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", "node_modules", ".git"]]
    for f in files:
        if any(imp in f for imp in important):
            path = os.path.join(root, f)
            with open(path, "r", errors="ignore") as file:
                content = file.read()[:1500]
            # Kurze Zusammenfassung
            summary = llm_query(f"1 Satz Zusammenfassung von {{f}}: {{content}}")
            summaries.append(f"{{f}}: {{summary[:100]}}")
            if len(summaries) >= 10:
                break
    if len(summaries) >= 10:
        break

for s in summaries:
    print(s)
```

DEIN ZIEL:
1. Lies README und Hauptdateien (max 10-15 Dateien)
2. Halte alle llm_query() Aufrufe KURZ (< 3000 Zeichen)
3. Erstelle eine KOMPAKTE Projektdokumentation

AUSGABEFORMAT (FINAL ANSWER):
```
# Projektdokumentation: {project_name}

## Uebersicht
[2-3 Saetze]

## Struktur
[Hauptverzeichnisse]

## Kernkomponenten
[Wichtigste 5-10 Dateien]

## Verwendung
[Wie startet man das Projekt?]
```

Beginne JETZT. Halte den Kontext KLEIN!
'''

RLM_DEEP_ANALYSIS_PROMPT = '''Analysiere diese Datei:

DATEI: {filepath}
GROESSE: {size} Bytes
Gemountet unter: /project/{relative_path}

WICHTIG: Halte alle llm_query() Aufrufe unter 2000 Zeichen!

```repl
filepath = "/project/{relative_path}"
with open(filepath, "r", errors="ignore") as f:
    content = f.read()[:3000]  # Max 3000 Zeichen lesen!

print(f"Gelesen: {{len(content)}} Zeichen")

# Kurze Analyse anfordern
result = llm_query(f"Kurze Analyse (max 200 Woerter) von {filepath}:\\n{{content}}")
print(result[:500])
```

Gib eine kurze Zusammenfassung zurueck.
'''


class RLMCentricWorkflow:
    """
    RLM-zentrierter Workflow.

    Das LLM steuert den gesamten Analyseprozess durch:
    - REPL Code-Ausfuehrung
    - Rekursive Selbst-Aufrufe (llm_query)
    - Batch-Verarbeitung (llm_query_batched)

    Python macht nur:
    - Dateien auflisten (fuer Statistik)
    - Projektverzeichnis in Docker mounten
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
        self.result = None

        # Kompatibilitaet mit DocumentationWorkflow GUI
        # state-Objekt das die GUI erwartet
        from gui.workflow_engine import WorkflowState
        self.state = WorkflowState()

    def stop(self):
        self._stop_requested = True

    def run(self) -> Iterator[RLMWorkflowProgress]:
        """
        Fuehrt den RLM-zentrierten Workflow aus.

        WICHTIG: Das Projektverzeichnis wird in Docker gemountet!
        Das LLM liest ALLE Dateien selbst.
        """
        start_time = datetime.now()

        # Phase 1: Dateien sammeln (nur fuer Statistik)
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
            detail=f"Projektverzeichnis wird in Docker gemountet",
        )

        # Phase 2: Statistik vorbereiten
        text_files = [f for f in files if not f.is_binary]

        # Kategorien zaehlen
        categories = {}
        for f in files:
            categories[f.category] = categories.get(f.category, 0) + 1
        cat_str = ", ".join([f"{k}: {v}" for k, v in categories.items()])

        # Dateiliste erstellen - NUR die 50 wichtigsten Dateien!
        # Priorisiere wichtige Dateien
        priority_files = self._prioritize_files(text_files)[:50]

        file_list_lines = []
        for f in priority_files:
            try:
                rel_path = str(Path(f.path).relative_to(self.config.source_path))
            except ValueError:
                rel_path = f.name
            # Kurze Zeilen - nur Name und Groesse
            file_list_lines.append(f"- {rel_path}")

        file_list = "\n".join(file_list_lines)

        yield RLMWorkflowProgress(
            stage="prepare",
            message="Bereite RLM-Aufruf vor...",
            progress=0.15,
            detail=f"Projektverzeichnis: {self.config.source_path}",
        )

        # Phase 3: EINEN RLM-Aufruf mit gemountentem Verzeichnis
        yield RLMWorkflowProgress(
            stage="analyze",
            message="RLM analysiert Projekt (LLM steuert)...",
            progress=0.2,
            detail="Projektverzeichnis gemountet unter /project",
        )

        prompt = RLM_ANALYSIS_PROMPT.format(
            project_name=Path(self.config.source_path).name,
            total_files=len(text_files),
            categories=cat_str,
            file_list=file_list,
        )

        # Kontext-String mit Dateiliste
        context = f"""Projektanalyse fuer: {self.config.source_path}
Dateien: {len(text_files)}
Kategorien: {cat_str}

Das Projekt ist gemountet unter /project - lies Dateien direkt!"""

        # DER EINE GROSSE RLM-AUFRUF MIT GEMOUNTENTEM VERZEICHNIS
        # Das LLM hat Zugriff auf /project im Container
        result = self.backend.run_completion_with_source_mount(
            source_path=self.config.source_path,
            context=context,
            aufgabe=prompt,
            container_mount_path="/project",
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

        # GUI-Kompatibilitaet: state-Objekt befuellen
        self.state.files = files
        self.state.final_documentation = documentation
        self.state.summaries = {"rlm_analysis": documentation[:500]}
        # Kategorien aus den Dateien erstellen
        for f in files:
            if f.category not in self.state.categories:
                self.state.categories[f.category] = []
            self.state.categories[f.category].append(f.path)

    def analyze_single_file(self, filepath: str) -> RLMResult:
        """
        Analysiert eine einzelne Datei mit RLM.

        Das LLM entscheidet selbst ueber Chunking.
        """
        size = Path(filepath).stat().st_size

        try:
            rel_path = str(Path(filepath).relative_to(self.config.source_path))
        except ValueError:
            rel_path = Path(filepath).name

        prompt = RLM_DEEP_ANALYSIS_PROMPT.format(
            filepath=filepath,
            relative_path=rel_path,
            size=size,
        )

        return self.backend.run_completion_with_source_mount(
            source_path=self.config.source_path,
            context=f"Datei zur Analyse: {filepath}",
            aufgabe=prompt,
            container_mount_path="/project",
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
            lines.append(f"- /project/{rel_path} ({f.size} bytes)")
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
