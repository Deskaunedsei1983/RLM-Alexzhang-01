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

RLM_ANALYSIS_PROMPT = '''Du bist ein Code-Analyse-Experte mit Python REPL Zugriff.

AUFGABE: Analysiere das GESAMTE Projekt "{project_name}" mit {total_files} Dateien.
Das Projekt ist gemountet unter: /project

Kategorien: {categories}

=== WICHTIG: NUTZE llm_query_batched() FUER EFFIZIENZ ===

Du hast Zugriff auf:
- llm_query(prompt) - einzelne LLM-Anfrage
- llm_query_batched(prompts_list) - VIELE Anfragen parallel (NUTZE DAS!)

SCHRITT 1 - Alle Dateien sammeln:
```repl
import os
all_files = []
for root, dirs, fs in os.walk("/project"):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", "node_modules", ".git", ".venv", "dist", "build", ".next"]]
    for f in fs:
        if f.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml", ".vue", ".svelte", ".go", ".rs", ".java")):
            all_files.append(os.path.join(root, f))
print(f"Gefunden: {{len(all_files)}} Dateien")
```

SCHRITT 2 - Dateien in Batches analysieren - BEHALTE VOLLSTAENDIGE BESCHREIBUNGEN:
```repl
batch_size = 20
all_summaries = []

for batch_start in range(0, len(all_files), batch_size):
    batch = all_files[batch_start:batch_start + batch_size]
    prompts = []

    for filepath in batch:
        try:
            with open(filepath, "r", errors="ignore") as f:
                content = f.read()
            if len(content) > 4000:
                content = content[:2000] + "\\n[...]\\n" + content[-1000:]
            prompts.append(f"Beschreibe detailliert (2-3 Saetze) was {{filepath}} macht, welche Funktionen/Klassen es enthaelt:\\n{{content[:3000]}}")
        except:
            pass

    if prompts:
        results = llm_query_batched(prompts)
        for fp, result in zip(batch, results):
            # WICHTIG: Behalte 500 Zeichen pro Datei fuer Details!
            all_summaries.append(f"{{fp}}:\\n{{str(result)[:500]}}")

    if batch_start % 200 == 0:
        print(f"Fortschritt: {{batch_start}}/{{len(all_files)}}")

print(f"\\nGesamt: {{len(all_summaries)}} Dateien analysiert")
```

SCHRITT 3 - Nach Verzeichnis gruppieren und ALLE Beschreibungen behalten:
```repl
from collections import defaultdict
by_module = defaultdict(list)

for summary in all_summaries:
    parts = summary.split(":\\n", 1)
    if len(parts) == 2:
        path, desc = parts
        rel_path = path.replace("/project/", "")
        parts2 = rel_path.split("/")
        # Tiefere Hierarchie: bis zu 2 Ebenen
        if len(parts2) > 1:
            module = "/".join(parts2[:2])
        else:
            module = parts2[0] if parts2 else "root"
        # BEHALTE VOLLE BESCHREIBUNG
        by_module[module].append(f"- {{parts2[-1]}}: {{desc}}")

print(f"{{len(by_module)}} Module gefunden")
```

SCHRITT 4 - DETAILLIERTE Modul-Dokumentation erstellen:
```repl
module_docs = []
for module in sorted(by_module.keys()):
    files_in_module = by_module[module]
    # Erstelle ausfuehrliche Modul-Doku mit ALLEN Dateien (max 100 pro Modul)
    files_text = "\\n".join(files_in_module[:100])

    if len(files_in_module) > 5:
        # Nur bei groesseren Modulen LLM-Zusammenfassung
        module_summary = llm_query(f"""Erstelle eine DETAILLIERTE Modul-Dokumentation (mindestens 200 Woerter) fuer das Modul '{{module}}' mit {{len(files_in_module)}} Dateien.

Dateien und ihre Funktionen:
{{files_text}}

Beschreibe:
1. Hauptzweck des Moduls
2. Wichtigste Komponenten/Klassen/Funktionen
3. Wie die Dateien zusammenarbeiten
4. Oeffentliche API/Schnittstellen
""")
        module_docs.append(f"### {{module}} ({{len(files_in_module)}} Dateien)\\n\\n{{module_summary}}\\n\\n**Enthaltene Dateien:**\\n{{files_text[:2000]}}")
    else:
        # Kleine Module: Direkt auflisten
        module_docs.append(f"### {{module}} ({{len(files_in_module)}} Dateien)\\n\\n{{files_text}}")

print(f"{{len(module_docs)}} Module dokumentiert")
```

SCHRITT 5 - VOLLSTAENDIGE finale Dokumentation zusammenstellen:
```repl
# Erstelle Einleitung
intro = llm_query(f"""Schreibe eine ausfuehrliche Einleitung (300-500 Woerter) fuer die Projektdokumentation von '{project_name}'.

Das Projekt hat {{len(all_files)}} Dateien in {{len(by_module)}} Modulen.

Beschreibe:
- Was ist der Hauptzweck des Projekts?
- Welche Technologien werden verwendet?
- Wie ist die Architektur aufgebaut?
- Wie installiert/verwendet man es?
""")

# Baue VOLLSTAENDIGE Dokumentation zusammen - NICHT nochmal zusammenfassen!
modules_text = "\\n\\n".join(module_docs)

final_doc = f"""# {project_name} - Projektdokumentation

## Uebersicht

{{intro}}

## Projektstatistik

- **Gesamtzahl Dateien:** {{len(all_files)}}
- **Module/Verzeichnisse:** {{len(by_module)}}
- **Analysierte Beschreibungen:** {{len(all_summaries)}}

## Module im Detail

{{modules_text}}

## Dateiuebersicht

Die vollstaendige Liste aller analysierten Dateien:

{{chr(10).join([s.split(chr(10))[0] for s in all_summaries[:500]])}}
"""

print("===DOKUMENTATION_START===")
print(final_doc)
print("===DOKUMENTATION_ENDE===")
print(f"\\nDokumentation Laenge: {{len(final_doc)}} Zeichen")
```

Nachdem du alle 5 Schritte ausgefuehrt hast, schreibe:
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

        # Kategorien zaehlen (kurz halten)
        categories = {}
        for f in files:
            categories[f.category] = categories.get(f.category, 0) + 1
        # Nur Top 5 Kategorien
        top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        cat_str = ", ".join([f"{k}: {v}" for k, v in top_cats])

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
        )

        # Kurzer Kontext
        context = f"Projekt: {Path(self.config.source_path).name}, {len(text_files)} Dateien, gemountet unter /project"

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
