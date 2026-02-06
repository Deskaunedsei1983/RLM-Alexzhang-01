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

DEINE AUFGABE: Analysiere das Projekt "{project_name}"

WICHTIG: Die Dateien sind bereits fuer dich geladen!
Die Variable `context` ist ein Dictionary mit:
- context["files_content"]: Dict mit Dateiname -> Inhalt
- context["project_path"]: Pfad zum Projekt
- context["total_files"]: Gesamtzahl der Dateien

PROJEKT-STATISTIK:
- Gesamtzahl Dateien: {total_files}
- Geladene Dateien: {loaded_files}
- Kategorien: {categories}

GELADENE DATEIEN:
{file_list}

BEISPIEL - So greifst du auf Dateien zu:
```repl
# Dateien aus context extrahieren
files_content = context["files_content"]
print(f"{{len(files_content)}} Dateien geladen")
print("Verfuegbare Dateien:", list(files_content.keys())[:10])

# Eine Datei lesen
for name, content in list(files_content.items())[:3]:
    print(f"\\n=== {{name}} ({{len(content)}} Zeichen) ===")
    print(content[:500])
```

BEISPIEL - Grosse Dateien mit llm_query analysieren:
```repl
files_content = context["files_content"]

for filename, content in files_content.items():
    if len(content) > 5000:
        # Chunking fuer grosse Dateien
        chunks = [content[i:i+4000] for i in range(0, len(content), 3500)]
        print(f"{{filename}}: {{len(chunks)}} Chunks")

        analyses = []
        for i, chunk in enumerate(chunks[:3]):  # Max 3 Chunks
            result = llm_query(f"Analysiere Teil {{i+1}} von {{filename}}:\\n{{chunk}}")
            analyses.append(result)
            print(f"Chunk {{i+1}} analysiert")

        # Zusammenfassen
        if analyses:
            summary = llm_query(f"Fasse zusammen: {{analyses}}")
            print(f"Summary fuer {{filename}}: {{summary[:200]}}")
```

DEIN ZIEL:
1. Extrahiere files_content aus context
2. Analysiere die wichtigsten Dateien
3. Nutze llm_query() fuer tiefe Analysen grosser Dateien
4. Nutze llm_query_batched() fuer parallele Verarbeitung
5. Erstelle eine VOLLSTAENDIGE Projektdokumentation

AUSGABEFORMAT (am Ende als FINAL ANSWER):
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

Beginne JETZT mit der Analyse. Fuehre REPL Code aus!
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

        # Phase 2: Dateien laden und fuer RLM vorbereiten
        text_files = [f for f in files if not f.is_binary]

        # Sortiere nach Wichtigkeit
        priority_files = self._prioritize_files(text_files)

        yield RLMWorkflowProgress(
            stage="prepare",
            message="Lade Dateiinhalte...",
            progress=0.15,
            detail=f"{len(priority_files)} Dateien priorisiert",
        )

        # Lade die wichtigsten Dateien (max 100 oder max 2MB gesamt)
        files_content = {}
        total_size = 0
        max_total_size = 2 * 1024 * 1024  # 2MB max
        max_files = 100

        for f in priority_files:
            if len(files_content) >= max_files:
                break
            if total_size >= max_total_size:
                break

            content, success = FileDiscovery.read_file_safe(
                f.path,
                max_size=min(50000, max_total_size - total_size),  # Max 50KB pro Datei
            )

            if success and content:
                files_content[f.name] = content
                total_size += len(content)

        yield RLMWorkflowProgress(
            stage="prepare",
            message=f"{len(files_content)} Dateien geladen ({total_size // 1024} KB)",
            progress=0.2,
        )

        # Erstelle Dateiliste fuer Prompt
        file_list = "\n".join([
            f"- {name} ({len(content)} Zeichen)"
            for name, content in files_content.items()
        ])

        # Kategorien zaehlen
        categories = {}
        for f in files:
            categories[f.category] = categories.get(f.category, 0) + 1
        cat_str = ", ".join([f"{k}: {v}" for k, v in categories.items()])

        # Phase 3: EINEN RLM-Aufruf - das LLM macht den Rest!
        yield RLMWorkflowProgress(
            stage="analyze",
            message="RLM analysiert Projekt (LLM steuert)...",
            progress=0.25,
            detail="Dateien sind im REPL als files_content verfuegbar",
        )

        prompt = RLM_ANALYSIS_PROMPT.format(
            project_name=Path(self.config.source_path).name,
            total_files=len(text_files),
            loaded_files=len(files_content),
            categories=cat_str,
            file_list=file_list,
        )

        # Kontext mit Dateiinhalten als JSON
        # Dies wird ins Docker /workspace als context.json geschrieben
        # und ist als Variable `context` verfuegbar
        context_with_files = {
            "files_content": files_content,
            "project_path": self.config.source_path,
            "total_files": len(text_files),
        }

        # Setup-Code um files_content aus context zu extrahieren
        setup_code = "files_content = context.get('files_content', {})"

        # DER EINE GROSSE RLM-AUFRUF
        # Das LLM hat Zugriff auf files_content im REPL
        result = self.backend.run_completion_with_context(
            context_payload=context_with_files,
            setup_code=setup_code,
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
