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

DEINE AUFGABE: Analysiere das Projekt "{project_name}" mit {total_files} Dateien.

Das Projektverzeichnis ist gemountet unter: /project

PROJEKT-STATISTIK:
- Gesamtzahl Dateien: {total_files}
- Kategorien: {categories}

WICHTIGSTE DATEIEN:
{file_list}

=== DEINE ANALYSE-STRATEGIE ===

Fuehre diese Schritte aus:

SCHRITT 1 - Dateien erkunden:
```repl
import os
files = []
for root, dirs, fs in os.walk("/project"):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", "node_modules", ".git", ".venv"]]
    for f in fs:
        if f.endswith((".py", ".js", ".ts", ".md", ".json")):
            files.append(os.path.join(root, f))
print(f"Gefunden: {{len(files)}} relevante Dateien")
print("Erste 20:", files[:20])
```

SCHRITT 2 - Wichtige Dateien lesen und analysieren:
```repl
# Lies und analysiere die wichtigsten Dateien
wichtig = ["README", "main", "app", "index", "setup", "config"]
summaries = []

for filepath in files[:50]:
    name = os.path.basename(filepath).lower()
    if any(w in name for w in wichtig) or filepath in files[:10]:
        try:
            with open(filepath, "r", errors="ignore") as f:
                content = f.read()[:2000]
            # Kurze Analyse
            desc = llm_query(f"Beschreibe in 1-2 Saetzen was diese Datei macht:\\n{{content}}")
            summaries.append(f"{{filepath}}: {{desc[:150]}}")
            print(f"Analysiert: {{filepath}}")
        except Exception as e:
            print(f"Fehler: {{e}}")

print(f"\\n{{len(summaries)}} Dateien analysiert")
```

SCHRITT 3 - Dokumentation erstellen:
```repl
# Erstelle die finale Dokumentation
summary_text = "\\n".join(summaries[:30])

doc_prompt = f"""Erstelle eine Projektdokumentation basierend auf diesen Datei-Analysen:

{{summary_text}}

Format:
# {project_name} - Projektdokumentation

## Uebersicht
[Was macht das Projekt?]

## Hauptkomponenten
[Wichtigste Dateien/Module]

## Verwendung
[Wie nutzt man das Projekt?]
"""

final_doc = llm_query(doc_prompt)
print("===DOKUMENTATION_START===")
print(final_doc)
print("===DOKUMENTATION_ENDE===")
```

Nachdem du Schritt 3 ausgefuehrt hast, schreibe diese Zeile alleine:
FINAL_VAR("final_doc")
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
        print(f"[RLMWorkflow] RLM-Response Laenge: {len(documentation)} Zeichen")

        # FALLBACK: Wenn Antwort zu kurz, versuche aus Logs zu extrahieren
        if len(documentation) < 200:
            yield RLMWorkflowProgress(
                stage="analyze",
                message=f"Kurze Antwort ({len(documentation)} Zeichen), extrahiere aus Logs...",
                progress=0.92,
            )
            documentation = self._extract_documentation_from_logs(result.log_file, documentation)
            print(f"[RLMWorkflow] Nach Log-Extraktion: {len(documentation)} Zeichen")

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

    def _extract_documentation_from_logs(self, log_file: Optional[str], fallback: str) -> str:
        """
        Extrahiert Dokumentation aus den RLM-Logs wenn FINAL_VAR fehlgeschlagen ist.

        Sucht nach:
        1. ===DOKUMENTATION_START=== ... ===DOKUMENTATION_ENDE=== Markern
        2. LLM-Responses die wie Dokumentation aussehen
        3. Laengster stdout-Abschnitt der wie Dokumentation aussieht
        """
        import json
        import re

        if not log_file or not Path(log_file).exists():
            print(f"[RLMWorkflow] Kein Log-File gefunden: {log_file}")
            return fallback

        print(f"[RLMWorkflow] Extrahiere Dokumentation aus: {log_file}")

        try:
            with open(log_file, 'r') as f:
                log_content = f.read()

            # Suche nach Dokumentations-Markern im gesamten Log
            marker_pattern = r'===DOKUMENTATION_START===(.*?)===DOKUMENTATION_ENDE==='
            marker_match = re.search(marker_pattern, log_content, re.DOTALL)
            if marker_match:
                doc = marker_match.group(1).strip()
                if len(doc) > 100:
                    print(f"[RLMWorkflow] Dokumentation via Marker gefunden: {len(doc)} Zeichen")
                    return doc

            # Parse JSON lines und sammle alle relevanten Inhalte
            all_stdout = []
            all_responses = []
            final_answers = []

            for line in log_content.split('\n'):
                if line.strip():
                    try:
                        entry = json.loads(line)

                        # Sammle final_answer wenn vorhanden
                        if 'final_answer' in entry and entry['final_answer']:
                            fa = entry['final_answer']
                            if isinstance(fa, str) and len(fa) > 50:
                                final_answers.append(fa)

                        # Sammle LLM-Responses
                        if 'response' in entry:
                            resp = entry['response']
                            if isinstance(resp, str) and len(resp) > 100:
                                all_responses.append(resp)

                        # Suche nach stdout in code_blocks
                        if 'code_blocks' in entry:
                            for block in entry.get('code_blocks', []):
                                if 'result' in block and 'stdout' in block['result']:
                                    stdout = block['result']['stdout']
                                    if len(stdout) > 50:
                                        all_stdout.append(stdout)

                    except json.JSONDecodeError:
                        pass

            # Wenn es final_answers gibt, nimm die laengste
            if final_answers:
                longest_fa = max(final_answers, key=len)
                if len(longest_fa) > 100:
                    print(f"[RLMWorkflow] Final Answer aus Logs: {len(longest_fa)} Zeichen")
                    return longest_fa

            print(f"[RLMWorkflow] Gefunden: {len(all_responses)} Responses, {len(all_stdout)} Stdout-Bloecke")

            # Finde die beste Dokumentation
            best_doc = fallback
            best_source = "fallback"

            # 1. Suche in stdout nach Markern oder Markdown
            for stdout in all_stdout:
                # Suche nach Dokumentations-Markern im stdout
                if '===DOKUMENTATION_START===' in stdout:
                    match = re.search(marker_pattern, stdout, re.DOTALL)
                    if match:
                        doc = match.group(1).strip()
                        if len(doc) > len(best_doc):
                            best_doc = doc
                            best_source = "stdout_marker"
                # Oder nimm den laengsten stdout der Markdown-artig ist
                elif len(stdout) > len(best_doc) and ('# ' in stdout or '## ' in stdout):
                    best_doc = stdout
                    best_source = "stdout_markdown"

            # 2. Suche in LLM-Responses nach Dokumentation
            for resp in all_responses:
                # Suche nach Markern
                if '===DOKUMENTATION_START===' in resp:
                    match = re.search(marker_pattern, resp, re.DOTALL)
                    if match:
                        doc = match.group(1).strip()
                        if len(doc) > len(best_doc):
                            best_doc = doc
                            best_source = "response_marker"
                # Oder Response die wie Markdown-Doku aussieht
                elif len(resp) > len(best_doc) and resp.strip().startswith('#'):
                    # Pruefe ob es wirklich Dokumentation ist (nicht Code)
                    if '```' not in resp[:100]:
                        best_doc = resp
                        best_source = "response_markdown"

            # 3. Falls immer noch zu kurz, nimm die laengste Response
            if len(best_doc) < 200:
                for resp in sorted(all_responses, key=len, reverse=True):
                    if len(resp) > len(best_doc):
                        # Filtere Code-lastige Responses
                        code_ratio = resp.count('```') / max(1, len(resp) / 1000)
                        if code_ratio < 5:  # Nicht zu viel Code
                            best_doc = resp
                            best_source = "longest_response"
                            break

            print(f"[RLMWorkflow] Beste Dokumentation: {len(best_doc)} Zeichen (Quelle: {best_source})")
            return best_doc

        except Exception as e:
            print(f"[RLMWorkflow] Fehler beim Log-Parsing: {e}")
            import traceback
            traceback.print_exc()
            return fallback

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
