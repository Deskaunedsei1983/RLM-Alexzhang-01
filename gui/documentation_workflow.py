"""
Dokumentations-Workflow

Mehrstufiger Workflow zur automatischen Dokumentationserstellung
mit intelligentem Kontextmanagement.

Workflow-Phasen:
1. DISCOVER: Dateien im Verzeichnis finden
2. CATEGORIZE: Dateien nach Typ gruppieren
3. ANALYZE: Jede Datei analysieren (mit Chunking)
4. SUMMARIZE: Modul-Zusammenfassungen erstellen
5. DOCUMENT: Finale Dokumentation generieren
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator, Callable
from enum import Enum

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

    Loest das Kontextproblem durch:
    1. Hierarchische Analyse: Erst Struktur, dann Details
    2. Chunk-weise Verarbeitung: Grosse Dateien aufteilen
    3. Akkumulierende Zusammenfassungen: Wissen aufbauen
    4. Komprimierung: Alte Infos zusammenfassen
    """

    def __init__(
        self,
        backend: RLMBackend,
        workflow_config: WorkflowConfig,
        progress_callback: Optional[Callable[[WorkflowProgress], None]] = None,
        memory_system: Optional[MemorySystem] = None,
    ):
        self.backend = backend
        self.config = workflow_config
        self.progress_callback = progress_callback
        self.state = WorkflowState()
        self.context_manager = ContextManager(
            max_context=workflow_config.available_context
        )
        # Memory-System fuer persistente dateiuebergreifende Zusammenhaenge
        self.memory = memory_system
        self._stop_requested = False
        self._workflow_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def stop(self):
        """Stoppt den Workflow."""
        self._stop_requested = True

    def _report_progress(self, stage: str, message: str, progress: float, detail: str = "", is_error: bool = False):
        """Meldet Fortschritt an die GUI."""
        if self.progress_callback:
            self.progress_callback(WorkflowProgress(
                stage=stage,
                message=message,
                progress=progress,
                detail=detail,
                is_error=is_error,
            ))

    def run(self) -> Generator[WorkflowProgress, None, WorkflowState]:
        """
        Fuehrt den kompletten Workflow aus.

        Yields:
            WorkflowProgress Objekte fuer GUI-Updates

        Returns:
            Finaler WorkflowState
        """
        try:
            # Phase 1: Dateien entdecken
            yield from self._phase_discover()
            if self._stop_requested:
                return self.state

            # Phase 2: Kategorisieren
            yield from self._phase_categorize()
            if self._stop_requested:
                return self.state

            # Phase 3: Analysieren
            yield from self._phase_analyze()
            if self._stop_requested:
                return self.state

            # Phase 4: Zusammenfassen
            yield from self._phase_summarize()
            if self._stop_requested:
                return self.state

            # Phase 5: Dokumentation erstellen
            yield from self._phase_document()

        except Exception as e:
            self.state.errors.append(str(e))
            yield WorkflowProgress(
                stage="error",
                message=f"Workflow-Fehler: {e}",
                progress=self.state.progress,
                is_error=True,
            )

        return self.state

    def _phase_discover(self) -> Generator[WorkflowProgress, None, None]:
        """Phase 1: Dateien finden."""
        self.state.stage = WorkflowStage.DISCOVER

        yield WorkflowProgress(
            stage="discover",
            message="Suche Dateien...",
            progress=0.05,
            detail=f"Durchsuche {self.config.source_path}",
        )

        # Dateien finden
        self.state.files = FileDiscovery.discover(
            self.config.source_path,
            self.config,
        )

        if not self.state.files:
            yield WorkflowProgress(
                stage="discover",
                message="Keine Dateien gefunden",
                progress=0.1,
                is_error=True,
            )
            return

        # Dateiliste fuer LLM vorbereiten
        file_list = self._format_file_list(self.state.files[:50])  # Max 50 fuer Uebersicht

        yield WorkflowProgress(
            stage="discover",
            message=f"{len(self.state.files)} Dateien gefunden",
            progress=0.1,
            detail=file_list[:500],
        )

        # LLM fuer Uebersicht fragen
        prompt = PROMPTS["discover_summary"].format(file_list=file_list)
        result = self.backend.run_completion(
            context=file_list,
            aufgabe="Erstelle eine kurze Uebersicht der Projektstruktur.",
        )

        if result.success:
            self.context_manager.add_knowledge("projekt_struktur", result.response)
            # Auch im Memory-System speichern (Long-Term fuer Wiederverwendung)
            if self.memory:
                self.memory.add_long_term(
                    key=f"{self._workflow_id}_struktur",
                    content=result.response,
                    source=self.config.source_path,
                    entry_type="project_structure",
                    relevance=1.0,
                )
            yield WorkflowProgress(
                stage="discover",
                message="Projektstruktur analysiert",
                progress=0.15,
                detail=result.response[:300],
            )

    def _phase_categorize(self) -> Generator[WorkflowProgress, None, None]:
        """Phase 2: Dateien kategorisieren."""
        self.state.stage = WorkflowStage.CATEGORIZE

        yield WorkflowProgress(
            stage="categorize",
            message="Kategorisiere Dateien...",
            progress=0.2,
        )

        # Nach Kategorie gruppieren
        for file_info in self.state.files:
            category = file_info.category
            if category not in self.state.categories:
                self.state.categories[category] = []
            self.state.categories[category].append(file_info.path)

        # Zusammenfassung der Kategorien
        cat_summary = "\n".join([
            f"- {cat}: {len(files)} Dateien"
            for cat, files in self.state.categories.items()
        ])

        self.context_manager.add_knowledge("kategorien", cat_summary)
        # Im Memory speichern
        if self.memory:
            self.memory.add_short_term(
                key=f"{self._workflow_id}_kategorien",
                content=cat_summary,
                source=self.config.source_path,
                entry_type="categories",
                relevance=0.8,
            )

        yield WorkflowProgress(
            stage="categorize",
            message=f"{len(self.state.categories)} Kategorien erstellt",
            progress=0.25,
            detail=cat_summary,
        )

    def _phase_analyze(self) -> Generator[WorkflowProgress, None, None]:
        """Phase 3: Dateien analysieren."""
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
            message=f"Analysiere {total} Dateien...",
            progress=0.3,
        )

        for i, file_info in enumerate(text_files):
            if self._stop_requested:
                return

            self.state.current_file = file_info.name
            progress = 0.3 + (0.4 * (i / total))

            yield WorkflowProgress(
                stage="analyze",
                message=f"Analysiere {file_info.name}",
                progress=progress,
                detail=f"Datei {i+1}/{total}",
            )

            # Datei lesen
            content, success = FileDiscovery.read_file_safe(
                file_info.path,
                max_size=self.config.max_file_size,
            )

            if not success:
                self.state.errors.append(f"Konnte {file_info.path} nicht lesen")
                continue

            # Analysieren (mit Chunking falls noetig)
            summary = yield from self._analyze_file(file_info, content)

            if summary:
                file_info.summary = summary
                self.state.summaries[file_info.path] = summary
                self.context_manager.add_knowledge(
                    f"datei:{file_info.name}",
                    summary[:500],  # Komprimiert speichern
                )
                # Im Memory-System speichern fuer dateiuebergreifende Zusammenhaenge
                if self.memory:
                    # Short-Term fuer aktuelle Session
                    self.memory.add_short_term(
                        key=f"{self._workflow_id}_file_{file_info.name}",
                        content=summary,
                        source=f"{self.config.source_path}:{file_info.path}",
                        entry_type="file_summary",
                        relevance=0.7 + (0.3 * (i / total)),  # Spaetere Dateien haben mehr Kontext
                    )
                    # Wichtige Dateien (Hauptmodule) ins Long-Term
                    if file_info.category == "source_code" and len(summary) > 200:
                        self.memory.add_long_term(
                            key=f"file_{file_info.name}",
                            content=summary[:400],
                            source=file_info.path,
                            entry_type="important_file",
                            relevance=0.9,
                        )

    def _analyze_file(
        self,
        file_info: FileInfo,
        content: str,
    ) -> Generator[WorkflowProgress, None, Optional[str]]:
        """Analysiert eine einzelne Datei, ggf. in Chunks."""

        # Hole vorhandenes Wissen aus Memory fuer dateiuebergreifenden Kontext
        cross_file_context = ""
        if self.memory:
            cross_file_context = self.memory.get_relevant_context(
                max_chars=800,
                include_long_term=True,
                include_short_term=True,
            )

        # Pruefen ob Chunking noetig
        if len(content) <= self.config.chunking.max_chunk_size:
            # Direkt analysieren
            context, task = self.context_manager.get_context_for_prompt(
                content,
                f"Analysiere die Datei {file_info.name}",
            )

            # Kombiniere lokales Wissen mit Memory-Kontext
            combined_knowledge = self.context_manager.knowledge_buffer[:800]
            if cross_file_context:
                combined_knowledge = f"Bisheriges Projektwissen:\n{cross_file_context}\n\nAktuelle Analyse:\n{combined_knowledge}"

            result = self.backend.run_completion(
                context=f"{combined_knowledge}\n\nDatei {file_info.name}:\n{content[:2000]}",
                aufgabe=f"Analysiere diese {file_info.extension} Datei. Beruecksichtige das bisherige Projektwissen fuer Zusammenhaenge zu anderen Dateien.",
            )

            if result.success:
                return result.response
            return None

        # Chunking noetig
        chunks = self.context_manager.chunk_content(content, self.config.chunking)
        file_info.chunks_analyzed = len(chunks)

        accumulated_summary = ""

        for j, chunk in enumerate(chunks):
            if self._stop_requested:
                return accumulated_summary or None

            yield WorkflowProgress(
                stage="analyze",
                message=f"Analysiere {file_info.name} (Teil {j+1}/{len(chunks)})",
                progress=self.state.progress,
                detail=f"Chunk {j+1} von {len(chunks)}",
            )

            if j == 0:
                # Erster Chunk - mit Cross-File Kontext
                context_with_memory = chunk
                if cross_file_context:
                    context_with_memory = f"Projektwissen:\n{cross_file_context[:500]}\n\nDatei {file_info.name} (Teil 1):\n{chunk}"
                result = self.backend.run_completion(
                    context=context_with_memory,
                    aufgabe=f"Analysiere den Anfang der Datei {file_info.name}. Beruecksichtige Zusammenhaenge zum Projektwissen.",
                )
            else:
                # Folge-Chunks
                result = self.backend.run_completion(
                    context=f"Bisherige Analyse:\n{accumulated_summary}\n\nNeuer Teil:\n{chunk}",
                    aufgabe=f"Ergaenze die Analyse mit Erkenntnissen aus diesem Teil.",
                )

            if result.success:
                accumulated_summary = result.response

        return accumulated_summary

    def _phase_summarize(self) -> Generator[WorkflowProgress, None, None]:
        """Phase 4: Modul-Zusammenfassungen erstellen."""
        self.state.stage = WorkflowStage.SUMMARIZE

        yield WorkflowProgress(
            stage="summarize",
            message="Erstelle Modul-Zusammenfassungen...",
            progress=0.75,
        )

        # Gruppiere nach Verzeichnis (Module)
        modules = {}
        for file_info in self.state.files:
            module = str(Path(file_info.path).parent.relative_to(self.config.source_path))
            if module == ".":
                module = "root"
            if module not in modules:
                modules[module] = []
            modules[module].append(file_info)

        # Fuer jedes Modul eine Zusammenfassung
        for module_name, files in modules.items():
            if self._stop_requested:
                return

            summaries = [f.summary for f in files if f.summary]
            if not summaries:
                continue

            combined = "\n\n".join([
                f"### {f.name}\n{f.summary}"
                for f in files if f.summary
            ])

            result = self.backend.run_completion(
                context=combined[:3000],
                aufgabe=f"Fasse die Dateien im Modul '{module_name}' zu einer Modulbeschreibung zusammen.",
            )

            if result.success:
                self.context_manager.add_knowledge(f"modul:{module_name}", result.response)
                # Module ins Long-Term Memory (wichtig fuer Wiederverwendung)
                if self.memory:
                    self.memory.add_long_term(
                        key=f"module_{module_name}",
                        content=result.response,
                        source=f"{self.config.source_path}/{module_name}",
                        entry_type="module_summary",
                        relevance=0.95,
                    )

        yield WorkflowProgress(
            stage="summarize",
            message=f"{len(modules)} Module zusammengefasst",
            progress=0.85,
        )

    def _phase_document(self) -> Generator[WorkflowProgress, None, None]:
        """Phase 5: Finale Dokumentation erstellen."""
        self.state.stage = WorkflowStage.DOCUMENT

        yield WorkflowProgress(
            stage="document",
            message="Erstelle finale Dokumentation...",
            progress=0.9,
        )

        # Gesammeltes Wissen fuer finale Doku - kombiniere ContextManager und Memory
        knowledge = self.context_manager.knowledge_buffer
        if self.memory:
            # Hole zusaetzlich Long-Term Wissen (Module, wichtige Dateien)
            memory_context = self.memory.get_relevant_context(
                max_chars=1500,
                include_long_term=True,
                include_short_term=False,  # Nur persistentes Wissen
            )
            if memory_context:
                knowledge = f"{knowledge}\n\nPersistentes Projektwissen:\n{memory_context}"

        result = self.backend.run_completion(
            context=knowledge[:3500],
            aufgabe="""Erstelle eine vollstaendige Projektdokumentation im Markdown-Format mit:
1. Projektuebersicht
2. Verzeichnisstruktur
3. Hauptkomponenten
4. Wichtige Dateien und deren Zweck
5. Abhaengigkeiten (falls erkennbar)
6. Hinweise zur Verwendung""",
        )

        if result.success:
            self.state.final_documentation = result.response

            # Dokumentation speichern
            output_dir = Path(self.config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            doc_file = output_dir / "DOCUMENTATION.md"
            doc_file.write_text(result.response, encoding='utf-8')

            # Finale Doku im Long-Term Memory speichern
            if self.memory:
                self.memory.add_long_term(
                    key=f"{self._workflow_id}_documentation",
                    content=result.response[:1000],  # Kurzversion
                    source=str(doc_file),
                    entry_type="final_documentation",
                    relevance=1.0,
                )

            yield WorkflowProgress(
                stage="document",
                message="Dokumentation erstellt!",
                progress=1.0,
                detail=f"Gespeichert: {doc_file}",
            )
        else:
            yield WorkflowProgress(
                stage="document",
                message="Fehler bei Dokumentationserstellung",
                progress=0.95,
                is_error=True,
                detail=result.error or "Unbekannter Fehler",
            )

    def _format_file_list(self, files: list[FileInfo]) -> str:
        """Formatiert eine Dateiliste fuer LLM-Kontext."""
        lines = []
        for f in files:
            rel_path = str(Path(f.path).relative_to(self.config.source_path))
            lines.append(f"- {rel_path} ({f.category}, {f.size} bytes)")
        return "\n".join(lines)


def run_documentation_workflow(
    source_path: str,
    backend: RLMBackend,
    progress_callback: Optional[Callable[[WorkflowProgress], None]] = None,
) -> WorkflowState:
    """
    Convenience-Funktion zum Starten eines Dokumentations-Workflows.

    Args:
        source_path: Pfad zum zu dokumentierenden Verzeichnis
        backend: RLM Backend Instanz
        progress_callback: Callback fuer Fortschrittsmeldungen

    Returns:
        WorkflowState mit Ergebnissen
    """
    config = WorkflowConfig(source_path=source_path)
    workflow = DocumentationWorkflow(backend, config, progress_callback)

    # Workflow ausfuehren
    for progress in workflow.run():
        pass  # Progress wird via Callback gemeldet

    return workflow.state
