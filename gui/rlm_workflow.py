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

DEINE AUFGABE: Analysiere das GESAMTE Projekt "{project_name}" - ALLE {total_files} Dateien!

Das Projektverzeichnis ist gemountet unter: /project
Du MUSST alle Dateien systematisch durchgehen!

PROJEKT-STATISTIK:
- Gesamtzahl Dateien: {total_files}
- Kategorien: {categories}

WICHTIGSTE DATEIEN (Startpunkt):
{file_list}

=== WICHTIG: SO GIBST DU DEINE FINALE ANTWORT ===
Am Ende MUSST du FINAL_VAR("variable_name") verwenden!
Speichere deine Dokumentation in einer Variable und rufe dann FINAL_VAR auf.

Beispiel:
```repl
documentation = "# Meine Dokumentation\\n\\n## Inhalt..."
```
Dann schreibe ausserhalb des Code-Blocks:
FINAL_VAR("documentation")

=== STRATEGIE FUER VOLLSTAENDIGE ANALYSE ===

SCHRITT 1: Alle Dateien sammeln
```repl
import os

all_files = []
for root, dirs, files in os.walk("/project"):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", "node_modules", ".git", ".venv", "venv", ".next", "dist", "build", ".idea", ".vscode"]]
    for f in files:
        if f.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".html", ".css", ".vue", ".svelte")):
            all_files.append(os.path.join(root, f))

print(f"Zu analysieren: {{len(all_files)}} Dateien")
```

SCHRITT 2: Dateien in Batches analysieren
```repl
batch_size = 10
all_summaries = []

for batch_start in range(0, min(len(all_files), 500), batch_size):  # Max 500 Dateien
    batch = all_files[batch_start:batch_start + batch_size]
    prompts = []

    for filepath in batch:
        try:
            with open(filepath, "r", errors="ignore") as f:
                content = f.read()
            if len(content) > 3000:
                content = content[:1500] + "\\n...\\n" + content[-500:]
            prompts.append(f"Kurze Beschreibung (1-2 Saetze) von {{filepath}}:\\n{{content[:2000]}}")
        except:
            pass

    if prompts:
        results = llm_query_batched(prompts)
        for fp, result in zip(batch, results):
            all_summaries.append(f"{{fp}}: {{result[:200]}}")

    print(f"Fortschritt: {{batch_start + len(batch)}}/{{min(len(all_files), 500)}}")

print(f"{{len(all_summaries)}} Dateien analysiert")
```

SCHRITT 3: Nach Verzeichnis gruppieren und Modul-Docs erstellen
```repl
from collections import defaultdict
by_dir = defaultdict(list)

for summary in all_summaries:
    parts = summary.split(": ", 1)
    if len(parts) == 2:
        path, desc = parts
        dir_name = "/".join(path.replace("/project/", "").split("/")[:-1]) or "root"
        by_dir[dir_name].append(desc[:150])

module_docs = []
for dir_name in sorted(by_dir.keys())[:30]:  # Max 30 Module
    files_desc = by_dir[dir_name][:10]
    module_text = f"### {{dir_name}}\\n" + "\\n".join([f"- {{d}}" for d in files_desc])
    module_docs.append(module_text)

print(f"{{len(module_docs)}} Module dokumentiert")
```

SCHRITT 4: Finale Dokumentation erstellen und in Variable speichern
```repl
modules_text = "\\n\\n".join(module_docs)

final_prompt = f"""Erstelle eine Projektdokumentation fuer {project_name}:

Module:
{{modules_text}}

Erstelle eine strukturierte Dokumentation mit:
- Uebersicht (was macht das Projekt)
- Architektur (Hauptkomponenten)
- Module (kurze Beschreibung jedes Moduls)
- Verwendung (wie startet man das Projekt)
"""

documentation = llm_query(final_prompt)
print("Dokumentation erstellt!")
print(documentation[:500])
```

SCHRITT 5: FINALE ANTWORT GEBEN
Nachdem du die documentation Variable erstellt hast, schreibe:
FINAL_VAR("documentation")

Beginne JETZT mit Schritt 1!
'''

RLM_DEEP_ANALYSIS_PROMPT = '''Analysiere diese Datei:

DATEI: {filepath}
GROESSE: {size} Bytes
Gemountet unter: /project/{relative_path}

WICHTIG: Speichere das Ergebnis in der Variable "analysis" und beende mit FINAL_VAR("analysis")

```repl
filepath = "/project/{relative_path}"
with open(filepath, "r", errors="ignore") as f:
    content = f.read()

print(f"Dateigroesse: {{len(content)}} Zeichen")

if len(content) > 6000:
    # Grosse Datei: Nur Anfang und Ende analysieren
    content_short = content[:3000] + "\\n...\\n" + content[-1500:]
    analysis = llm_query(f"Analysiere diese Datei (gekuerzt):\\n{{content_short}}")
else:
    analysis = llm_query(f"Analysiere diese Datei:\\n{{content}}")

print("Analyse erstellt")
print(analysis[:300])
```

Jetzt gib die finale Antwort:
FINAL_VAR("analysis")
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

        # Dateiliste erstellen - Top 100 priorisierte Dateien als Startpunkt
        # Das LLM findet den Rest selbst via os.walk()
        priority_files = self._prioritize_files(text_files)[:100]

        file_list_lines = []
        for f in priority_files:
            try:
                rel_path = str(Path(f.path).relative_to(self.config.source_path))
            except ValueError:
                rel_path = f.name
            file_list_lines.append(f"- /project/{rel_path}")

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
