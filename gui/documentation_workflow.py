"""
Dokumentations-Workflow

Mehrstufiger Workflow zur automatischen Dokumentationserstellung
mit intelligentem Kontextmanagement.

WICHTIG: Iterative Implementierung fuer grosse Dateimengen (100.000+)
Keine rekursiven Generator-Chains mehr!

Workflow-Phasen:
1. DISCOVER: Dateien im Verzeichnis finden
2. CATEGORIZE: Dateien nach Typ gruppieren
3. ANALYZE: Jede Datei analysieren (mit Chunking)
4. SUMMARIZE: Modul-Zusammenfassungen erstellen
5. DOCUMENT: Finale Dokumentation generieren

Analysetiefe (1-4):
1 = Nur Dateiliste + Struktur
2 = + Einfache Dateianalyse (Standard)
3 = + Modul-Zusammenfassungen
4 = + Vollstaendige Dokumentation
"""

import os
import sys
import time
import gc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Callable, List
from enum import Enum

# WICHTIG: Rekursionslimit fuer grosse Dateimengen erhoehen
# Das RLM-Framework nutzt intern Rekursion
sys.setrecursionlimit(100000)

from gui.workflow_engine import (
    WorkflowStage,
    WorkflowState,
    WorkflowConfig,
    FileInfo,
    FileDiscovery,
    ContextManager,
    ChunkingConfig,
    PROMPTS,
)
from gui.rlm_backend import RLMBackend, RLMConfig, RLMResult
from gui.memory_system import MemorySystem


@dataclass
class WorkflowProgress:
    """Fortschrittsmeldung fuer die GUI."""
    stage: str
    message: str
    progress: float  # 0.0 - 1.0
    detail: str = ""
    is_error: bool = False


class DocumentationWorkflow:
    """
    Orchestriert den mehrstufigen Dokumentations-Workflow.

    ITERATIVE Implementierung - keine Rekursion!
    Unterstuetzt bis zu 100.000+ Dateien durch Batch-Verarbeitung.

    Analysetiefe-Stufen:
    1 = Nur Dateiliste (schnell, fuer sehr grosse Projekte)
    2 = Dateianalyse (Standard)
    3 = Mit Modul-Zusammenfassungen
    4 = Vollstaendig mit finaler Dokumentation
    """

    # Batch-Groesse fuer Dateiverarbeitung
    BATCH_SIZE = 100

    def __init__(
        self,
        backend: RLMBackend,
        workflow_config: WorkflowConfig,
        progress_callback: Optional[Callable[[WorkflowProgress], None]] = None,
        memory_system: Optional[MemorySystem] = None,
        analysis_depth: int = 2,  # 1-4, Standard ist 2
    ):
        self.backend = backend
        self.config = workflow_config
        self.progress_callback = progress_callback
        self.state = WorkflowState()
        self.context_manager = ContextManager(
            max_context=workflow_config.available_context
        )
        self.memory = memory_system
        self.analysis_depth = max(1, min(4, analysis_depth))  # Clamp 1-4
        self._stop_requested = False
        self._workflow_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._progress_queue: List[WorkflowProgress] = []

    def stop(self):
        """Stoppt den Workflow."""
        self._stop_requested = True

    def _emit_progress(self, stage: str, message: str, progress: float,
                       detail: str = "", is_error: bool = False):
        """Fuegt Progress zur Queue hinzu (nicht-rekursiv)."""
        self._progress_queue.append(WorkflowProgress(
            stage=stage,
            message=message,
            progress=progress,
            detail=detail,
            is_error=is_error,
        ))

    def run(self) -> Iterator[WorkflowProgress]:
        """
        Fuehrt den kompletten Workflow ITERATIV aus.

        Yields:
            WorkflowProgress Objekte fuer GUI-Updates

        Diese Implementierung vermeidet rekursive Generator-Chains
        und kann daher beliebig viele Dateien verarbeiten.
        """
        try:
            # Phase 1: Dateien entdecken (immer)
            self._run_phase_discover()
            while self._progress_queue:
                yield self._progress_queue.pop(0)
            if self._stop_requested:
                return

            # Phase 2: Kategorisieren (immer)
            self._run_phase_categorize()
            while self._progress_queue:
                yield self._progress_queue.pop(0)
            if self._stop_requested:
                return

            # Phase 3: Analysieren (ab Tiefe 2)
            if self.analysis_depth >= 2:
                # Iterativ ueber Batches - KEIN yield from!
                for progress in self._run_phase_analyze_iterative():
                    yield progress
                    if self._stop_requested:
                        return

            # Phase 4: Zusammenfassen (ab Tiefe 3)
            if self.analysis_depth >= 3:
                self._run_phase_summarize()
                while self._progress_queue:
                    yield self._progress_queue.pop(0)
                if self._stop_requested:
                    return

            # Phase 5: Dokumentation erstellen (ab Tiefe 4)
            if self.analysis_depth >= 4:
                self._run_phase_document()
                while self._progress_queue:
                    yield self._progress_queue.pop(0)
            else:
                # Bei niedrigerer Tiefe: Einfache Zusammenfassung
                self._run_phase_simple_summary()
                while self._progress_queue:
                    yield self._progress_queue.pop(0)

        except Exception as e:
            self.state.errors.append(str(e))
            yield WorkflowProgress(
                stage="error",
                message=f"Workflow-Fehler: {e}",
                progress=self.state.progress,
                is_error=True,
            )

    def _run_phase_discover(self):
        """Phase 1: Dateien finden (nicht-rekursiv)."""
        self.state.stage = WorkflowStage.DISCOVER

        self._emit_progress(
            "discover",
            "Suche Dateien...",
            0.05,
            f"Durchsuche {self.config.source_path}",
        )

        # Dateien finden
        self.state.files = FileDiscovery.discover(
            self.config.source_path,
            self.config,
        )

        if not self.state.files:
            self._emit_progress(
                "discover",
                "Keine Dateien gefunden",
                0.1,
                is_error=True,
            )
            return

        # Dateiliste fuer LLM vorbereiten (max 100 fuer grosse Projekte)
        sample_size = min(100, len(self.state.files))
        file_list = self._format_file_list(self.state.files[:sample_size])

        self._emit_progress(
            "discover",
            f"{len(self.state.files)} Dateien gefunden",
            0.1,
            file_list[:500],
        )

        # LLM fuer Uebersicht fragen
        result = self.backend.run_completion(
            context=file_list,
            aufgabe="Erstelle eine kurze Uebersicht der Projektstruktur.",
        )

        if result.success:
            self.context_manager.add_knowledge("projekt_struktur", result.response)
            if self.memory:
                self.memory.add_long_term(
                    key=f"{self._workflow_id}_struktur",
                    content=result.response,
                    source=self.config.source_path,
                    entry_type="project_structure",
                    relevance=1.0,
                )
            self._emit_progress(
                "discover",
                "Projektstruktur analysiert",
                0.15,
                result.response[:300],
            )

    def _run_phase_categorize(self):
        """Phase 2: Dateien kategorisieren (nicht-rekursiv)."""
        self.state.stage = WorkflowStage.CATEGORIZE

        self._emit_progress(
            "categorize",
            "Kategorisiere Dateien...",
            0.2,
        )

        # Nach Kategorie gruppieren
        for file_info in self.state.files:
            category = file_info.category
            if category not in self.state.categories:
                self.state.categories[category] = []
            self.state.categories[category].append(file_info.path)

        cat_summary = "\n".join([
            f"- {cat}: {len(files)} Dateien"
            for cat, files in self.state.categories.items()
        ])

        self.context_manager.add_knowledge("kategorien", cat_summary)
        if self.memory:
            self.memory.add_short_term(
                key=f"{self._workflow_id}_kategorien",
                content=cat_summary,
                source=self.config.source_path,
                entry_type="categories",
                relevance=0.8,
            )

        self._emit_progress(
            "categorize",
            f"{len(self.state.categories)} Kategorien erstellt",
            0.25,
            cat_summary,
        )

    def _run_phase_analyze_iterative(self) -> Iterator[WorkflowProgress]:
        """
        Phase 3: Dateien analysieren - ITERATIV ohne Rekursion!

        Verarbeitet Dateien in Batches um Stack-Overflow zu vermeiden.
        Nach jedem Batch wird Garbage Collection ausgefuehrt.
        """
        self.state.stage = WorkflowStage.ANALYZE

        # Nur Text-Dateien analysieren
        text_files = [f for f in self.state.files if not f.is_binary]
        total = len(text_files)

        if total == 0:
            yield WorkflowProgress(
                stage="analyze",
                message="Keine Text-Dateien zum Analysieren",
                progress=0.5,
            )
            return

        yield WorkflowProgress(
            stage="analyze",
            message=f"Analysiere {total} Dateien in Batches a {self.BATCH_SIZE}...",
            progress=0.3,
        )

        # Verarbeite in Batches
        processed = 0
        batch_num = 0
        errors_in_row = 0  # Zaehle aufeinanderfolgende Fehler

        while processed < total:
            if self._stop_requested:
                return

            # Aktueller Batch
            batch_start = processed
            batch_end = min(processed + self.BATCH_SIZE, total)
            batch = text_files[batch_start:batch_end]
            batch_num += 1

            yield WorkflowProgress(
                stage="analyze",
                message=f"Batch {batch_num}: Dateien {batch_start+1}-{batch_end} von {total}",
                progress=0.3 + (0.4 * processed / total),
            )

            # Verarbeite jeden File im Batch (flach, keine Rekursion)
            for file_info in batch:
                if self._stop_requested:
                    return

                self.state.current_file = file_info.name
                processed += 1

                progress = 0.3 + (0.4 * processed / total)

                # Datei lesen
                content, success = FileDiscovery.read_file_safe(
                    file_info.path,
                    max_size=self.config.max_file_size,
                )

                if not success:
                    self.state.errors.append(f"Konnte {file_info.path} nicht lesen")
                    continue

                # Analysiere Datei - DIREKT, ohne Generator
                # Mit Fehlerbehandlung fuer Stack-Probleme
                try:
                    summary = self._analyze_file_direct(file_info, content)
                    errors_in_row = 0  # Reset bei Erfolg
                except RecursionError:
                    # Bei RecursionError: Garbage Collection und Retry
                    gc.collect()
                    try:
                        summary = self._analyze_file_direct(file_info, content)
                        errors_in_row = 0
                    except RecursionError:
                        summary = f"[Datei zu komplex fuer Analyse: {file_info.name}]"
                        errors_in_row += 1
                        self.state.errors.append(f"RecursionError bei {file_info.name}")
                        if errors_in_row > 5:
                            yield WorkflowProgress(
                                stage="analyze",
                                message="Zu viele Fehler - ueberspringe Rest",
                                progress=progress,
                                is_error=True,
                            )
                            # Nicht abbrechen, aber Rest ueberspringen
                            break

                if summary:
                    file_info.summary = summary
                    self.state.summaries[file_info.path] = summary
                    self.context_manager.add_knowledge(
                        f"datei:{file_info.name}",
                        summary[:500],
                    )
                    if self.memory:
                        self.memory.add_short_term(
                            key=f"{self._workflow_id}_file_{file_info.name}",
                            content=summary,
                            source=f"{self.config.source_path}:{file_info.path}",
                            entry_type="file_summary",
                            relevance=0.7 + (0.3 * processed / total),
                        )
                        if file_info.category == "source_code" and len(summary) > 200:
                            self.memory.add_long_term(
                                key=f"file_{file_info.name}",
                                content=summary[:400],
                                source=file_info.path,
                                entry_type="important_file",
                                relevance=0.9,
                            )

                # Alle 10 Dateien Progress melden
                if processed % 10 == 0 or processed == total:
                    yield WorkflowProgress(
                        stage="analyze",
                        message=f"Analysiert: {processed}/{total} Dateien",
                        progress=progress,
                        detail=f"Aktuell: {file_info.name}",
                    )

            # WICHTIG: Nach jedem Batch Garbage Collection
            # Dies hilft, den Python-Stack zu bereinigen
            gc.collect()

    def _analyze_file_direct(self, file_info: FileInfo, content: str) -> Optional[str]:
        """
        Analysiert eine einzelne Datei DIREKT ohne Generator.

        Dies ist die nicht-rekursive Version von _analyze_file.
        Fuer sehr grosse Dateien wird eine vereinfachte Analyse durchgefuehrt.
        """
        # Cross-File Kontext aus Memory
        cross_file_context = ""
        if self.memory:
            cross_file_context = self.memory.get_relevant_context(
                max_chars=800,
                include_long_term=True,
                include_short_term=True,
            )

        # Kleine Dateien direkt analysieren
        if len(content) <= self.config.chunking.max_chunk_size:
            combined_knowledge = self.context_manager.knowledge_buffer[:800]
            if cross_file_context:
                combined_knowledge = f"Projektwissen:\n{cross_file_context}\n\n{combined_knowledge}"

            result = self.backend.run_completion(
                context=f"{combined_knowledge}\n\nDatei {file_info.name}:\n{content[:2000]}",
                aufgabe=f"Analysiere diese {file_info.extension} Datei kurz.",
            )

            return result.response if result.success else None

        # SEHR grosse Dateien (>50KB): Vereinfachte Analyse
        # Nur Anfang, Mitte und Ende analysieren
        if len(content) > 50000:
            file_info.chunks_analyzed = 3  # Nur 3 Teile

            # Anfang (erste 2000 Zeichen)
            start_part = content[:2000]
            # Mitte
            mid_start = len(content) // 2 - 1000
            mid_part = content[mid_start:mid_start + 2000]
            # Ende (letzte 2000 Zeichen)
            end_part = content[-2000:]

            context = f"""Sehr grosse Datei {file_info.name} ({len(content)} Zeichen):

ANFANG:
{start_part}

[... {len(content) - 6000} Zeichen ausgelassen ...]

MITTE:
{mid_part}

[...]

ENDE:
{end_part}"""

            result = self.backend.run_completion(
                context=context,
                aufgabe=f"Analysiere diese grosse {file_info.extension} Datei anhand von Anfang, Mitte und Ende. Was ist der Zweck?",
            )

            return result.response if result.success else f"[Grosse Datei: {file_info.name}, {len(content)} Zeichen]"

        # Mittelgrosse Dateien: Chunking (iterativ, nicht rekursiv)
        # Aber maximal 5 Chunks um Stack-Probleme zu vermeiden
        chunks = self.context_manager.chunk_content(content, self.config.chunking)

        # Limitiere auf maximal 5 Chunks
        if len(chunks) > 5:
            # Nimm nur ersten, mittleren und letzten Chunk + 2 dazwischen
            indices = [0, len(chunks)//4, len(chunks)//2, 3*len(chunks)//4, len(chunks)-1]
            chunks = [chunks[i] for i in indices if i < len(chunks)]

        file_info.chunks_analyzed = len(chunks)

        accumulated_summary = ""

        for j, chunk in enumerate(chunks):
            if self._stop_requested:
                return accumulated_summary or None

            if j == 0:
                context = chunk
                if cross_file_context:
                    context = f"Projektwissen:\n{cross_file_context[:500]}\n\nDatei {file_info.name}:\n{chunk}"
                result = self.backend.run_completion(
                    context=context,
                    aufgabe=f"Analysiere den Anfang der Datei {file_info.name}.",
                )
            else:
                result = self.backend.run_completion(
                    context=f"Bisherige Analyse:\n{accumulated_summary}\n\nNeuer Teil:\n{chunk}",
                    aufgabe="Ergaenze die Analyse mit neuen Erkenntnissen.",
                )

            if result.success:
                accumulated_summary = result.response

        return accumulated_summary

    def _run_phase_summarize(self):
        """Phase 4: Modul-Zusammenfassungen (nicht-rekursiv)."""
        self.state.stage = WorkflowStage.SUMMARIZE

        self._emit_progress(
            "summarize",
            "Erstelle Modul-Zusammenfassungen...",
            0.75,
        )

        # Gruppiere nach Verzeichnis
        modules = {}
        for file_info in self.state.files:
            try:
                module = str(Path(file_info.path).parent.relative_to(self.config.source_path))
            except ValueError:
                module = "root"
            if module == ".":
                module = "root"
            if module not in modules:
                modules[module] = []
            modules[module].append(file_info)

        # Nur Top-Module verarbeiten bei sehr vielen
        module_list = list(modules.items())
        if len(module_list) > 50:
            # Sortiere nach Anzahl Dateien, nimm Top 50
            module_list.sort(key=lambda x: len(x[1]), reverse=True)
            module_list = module_list[:50]

        for module_name, files in module_list:
            if self._stop_requested:
                return

            summaries = [f.summary for f in files if f.summary]
            if not summaries:
                continue

            combined = "\n\n".join([
                f"### {f.name}\n{f.summary}"
                for f in files if f.summary
            ])[:3000]

            result = self.backend.run_completion(
                context=combined,
                aufgabe=f"Fasse Modul '{module_name}' zusammen.",
            )

            if result.success:
                self.context_manager.add_knowledge(f"modul:{module_name}", result.response)
                if self.memory:
                    self.memory.add_long_term(
                        key=f"module_{module_name}",
                        content=result.response,
                        source=f"{self.config.source_path}/{module_name}",
                        entry_type="module_summary",
                        relevance=0.95,
                    )

        self._emit_progress(
            "summarize",
            f"{len(module_list)} Module zusammengefasst",
            0.85,
        )

    def _run_phase_document(self):
        """Phase 5: Finale Dokumentation (nicht-rekursiv)."""
        self.state.stage = WorkflowStage.DOCUMENT

        self._emit_progress(
            "document",
            "Erstelle finale Dokumentation...",
            0.9,
        )

        knowledge = self.context_manager.knowledge_buffer
        if self.memory:
            memory_context = self.memory.get_relevant_context(
                max_chars=1500,
                include_long_term=True,
                include_short_term=False,
            )
            if memory_context:
                knowledge = f"{knowledge}\n\nPersistentes Wissen:\n{memory_context}"

        result = self.backend.run_completion(
            context=knowledge[:3500],
            aufgabe="""Erstelle Projektdokumentation im Markdown:
1. Projektuebersicht
2. Verzeichnisstruktur
3. Hauptkomponenten
4. Wichtige Dateien
5. Abhaengigkeiten
6. Verwendungshinweise""",
        )

        if result.success:
            self.state.final_documentation = result.response

            output_dir = Path(self.config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            doc_file = output_dir / "DOCUMENTATION.md"
            doc_file.write_text(result.response, encoding='utf-8')

            if self.memory:
                self.memory.add_long_term(
                    key=f"{self._workflow_id}_documentation",
                    content=result.response[:1000],
                    source=str(doc_file),
                    entry_type="final_documentation",
                    relevance=1.0,
                )

            self._emit_progress(
                "document",
                "Dokumentation erstellt!",
                1.0,
                f"Gespeichert: {doc_file}",
            )
        else:
            self._emit_progress(
                "document",
                "Fehler bei Dokumentation",
                0.95,
                result.error or "Unbekannter Fehler",
                is_error=True,
            )

    def _run_phase_simple_summary(self):
        """Einfache Zusammenfassung bei niedriger Analysetiefe."""
        self._emit_progress(
            "summary",
            "Erstelle Zusammenfassung...",
            0.9,
        )

        # Einfache Zusammenfassung basierend auf gesammeltem Wissen
        knowledge = self.context_manager.knowledge_buffer

        summary_parts = [
            f"# Projekt-Analyse",
            f"",
            f"## Statistiken",
            f"- Dateien gesamt: {len(self.state.files)}",
            f"- Kategorien: {len(self.state.categories)}",
            f"- Analysierte Dateien: {len(self.state.summaries)}",
            f"- Analysetiefe: {self.analysis_depth}",
            f"",
            f"## Kategorien",
        ]

        for cat, files in self.state.categories.items():
            summary_parts.append(f"- {cat}: {len(files)} Dateien")

        if self.state.summaries:
            summary_parts.append(f"")
            summary_parts.append(f"## Analysierte Dateien (Auswahl)")
            for path, summary in list(self.state.summaries.items())[:20]:
                filename = Path(path).name
                summary_parts.append(f"")
                summary_parts.append(f"### {filename}")
                summary_parts.append(summary[:300] + "..." if len(summary) > 300 else summary)

        self.state.final_documentation = "\n".join(summary_parts)

        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_file = output_dir / "DOCUMENTATION.md"
        doc_file.write_text(self.state.final_documentation, encoding='utf-8')

        self._emit_progress(
            "summary",
            "Zusammenfassung erstellt!",
            1.0,
            f"Gespeichert: {doc_file}",
        )

    def _format_file_list(self, files: list[FileInfo]) -> str:
        """Formatiert eine Dateiliste fuer LLM-Kontext."""
        lines = []
        for f in files:
            try:
                rel_path = str(Path(f.path).relative_to(self.config.source_path))
            except ValueError:
                rel_path = f.name
            lines.append(f"- {rel_path} ({f.category}, {f.size} bytes)")
        return "\n".join(lines)


def run_documentation_workflow(
    source_path: str,
    backend: RLMBackend,
    progress_callback: Optional[Callable[[WorkflowProgress], None]] = None,
    analysis_depth: int = 2,
) -> WorkflowState:
    """
    Convenience-Funktion zum Starten eines Dokumentations-Workflows.

    Args:
        source_path: Pfad zum zu dokumentierenden Verzeichnis
        backend: RLM Backend Instanz
        progress_callback: Callback fuer Fortschrittsmeldungen
        analysis_depth: Analysetiefe 1-4 (Standard: 2)

    Returns:
        WorkflowState mit Ergebnissen
    """
    config = WorkflowConfig(source_path=source_path)
    workflow = DocumentationWorkflow(
        backend,
        config,
        progress_callback,
        analysis_depth=analysis_depth,
    )

    for progress in workflow.run():
        if progress_callback:
            progress_callback(progress)

    return workflow.state
