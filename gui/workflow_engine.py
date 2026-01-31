"""
RLM Workflow Engine

Mehrstufige Workflows fuer komplexe Analysen mit Kontextmanagement.
Loest das Problem der begrenzten Kontextfenster durch:
- Hierarchische Analyse (grob -> fein)
- Chunking (grosse Dateien aufteilen)
- Summarization (Zwischenzusammenfassungen)
- Akkumulierendes Wissen (Buffer aufbauen)
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from enum import Enum


class WorkflowStage(Enum):
    """Phasen eines Dokumentations-Workflows."""
    DISCOVER = "discover"       # Dateien finden
    CATEGORIZE = "categorize"   # Dateien kategorisieren
    ANALYZE = "analyze"         # Dateien analysieren (chunk-weise)
    SUMMARIZE = "summarize"     # Zusammenfassungen erstellen
    DOCUMENT = "document"       # Finale Dokumentation


@dataclass
class FileInfo:
    """Informationen ueber eine Datei."""
    path: str
    name: str
    extension: str
    size: int
    is_binary: bool = False
    category: str = "unknown"
    summary: str = ""
    chunks_analyzed: int = 0


@dataclass
class WorkflowState:
    """Zustand eines laufenden Workflows."""
    stage: WorkflowStage = WorkflowStage.DISCOVER
    files: list[FileInfo] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)
    summaries: dict[str, str] = field(default_factory=dict)
    knowledge_buffer: str = ""
    final_documentation: str = ""
    progress: float = 0.0
    current_file: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Konvertiert State zu Dictionary fuer JSON-Serialisierung."""
        return {
            "stage": self.stage.value,
            "files_count": len(self.files),
            "categories": self.categories,
            "summaries_count": len(self.summaries),
            "knowledge_buffer_size": len(self.knowledge_buffer),
            "progress": self.progress,
            "current_file": self.current_file,
            "errors": self.errors,
        }


@dataclass
class ChunkingConfig:
    """Konfiguration fuer das Chunking grosser Dateien."""
    max_chunk_size: int = 2000      # Max Zeichen pro Chunk
    overlap: int = 200              # Ueberlappung zwischen Chunks
    max_chunks_per_file: int = 10   # Max Chunks pro Datei


@dataclass
class WorkflowConfig:
    """Konfiguration fuer den Dokumentations-Workflow."""
    source_path: str                          # Quellverzeichnis
    output_path: str = "./documentation"      # Ausgabeverzeichnis
    file_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h",
        ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".xml",
        ".html", ".css", ".scss", ".sql", ".sh", ".bash"
    ])
    ignore_patterns: list[str] = field(default_factory=lambda: [
        "__pycache__", "node_modules", ".git", ".venv", "venv",
        "dist", "build", ".egg-info", ".pytest_cache"
    ])
    max_file_size: int = 100000               # Max Dateigroesse in Bytes
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Kontext-Limits (basierend auf eurem LLM)
    max_context_tokens: int = 4096
    max_output_tokens: int = 1024
    reserved_for_prompt: int = 500

    @property
    def available_context(self) -> int:
        """Verfuegbarer Kontext fuer Daten."""
        return self.max_context_tokens - self.max_output_tokens - self.reserved_for_prompt


class ContextManager:
    """
    Verwaltet den Kontext ueber mehrere Prompts hinweg.

    Strategien:
    1. Rolling Summary: Alte Infos werden zusammengefasst
    2. Hierarchical: Erst Ueberblick, dann Details
    3. Chunked Analysis: Grosse Dateien in Teilen
    """

    def __init__(self, max_context: int = 2500):
        self.max_context = max_context
        self.knowledge_buffer = ""
        self.summaries = {}

    def add_knowledge(self, key: str, content: str):
        """Fuegt Wissen zum Buffer hinzu."""
        self.summaries[key] = content
        self._update_buffer()

    def _update_buffer(self):
        """Aktualisiert den komprimierten Wissensbuffer."""
        # Kombiniere alle Zusammenfassungen
        parts = []
        for key, summary in self.summaries.items():
            parts.append(f"[{key}]: {summary}")

        self.knowledge_buffer = "\n".join(parts)

        # Wenn zu gross, kuerze aeltere Eintraege
        while len(self.knowledge_buffer) > self.max_context and len(self.summaries) > 1:
            oldest_key = list(self.summaries.keys())[0]
            # Komprimiere statt loeschen
            old_content = self.summaries[oldest_key]
            self.summaries[oldest_key] = old_content[:200] + "..."
            self._update_buffer()

    def get_context_for_prompt(self, new_content: str, task: str) -> tuple[str, str]:
        """
        Bereitet Kontext und Prompt fuer naechste Anfrage vor.

        Returns:
            Tuple aus (kontext, aufgabe)
        """
        available = self.max_context - len(task) - 100  # Buffer

        # Prioritaet: Neuer Content > Wissensbuffer
        if len(new_content) > available:
            # Chunk den neuen Content
            new_content = new_content[:available-len(self.knowledge_buffer)-100]

        context = f"""Bisheriges Wissen:
{self.knowledge_buffer}

Aktueller Inhalt:
{new_content}"""

        return context, task

    def chunk_content(self, content: str, config: ChunkingConfig) -> list[str]:
        """Teilt grossen Content in Chunks auf."""
        if len(content) <= config.max_chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content) and len(chunks) < config.max_chunks_per_file:
            end = start + config.max_chunk_size

            # Versuche an Zeilenumbruch zu trennen
            if end < len(content):
                newline_pos = content.rfind('\n', start, end)
                if newline_pos > start + config.max_chunk_size // 2:
                    end = newline_pos + 1

            chunks.append(content[start:end])
            start = end - config.overlap

        return chunks


class FileDiscovery:
    """Findet und kategorisiert Dateien."""

    CATEGORY_PATTERNS = {
        "source_code": [".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp"],
        "config": [".json", ".yaml", ".yml", ".toml", ".xml", ".env"],
        "documentation": [".md", ".txt", ".rst", ".adoc"],
        "web": [".html", ".css", ".scss", ".vue", ".jsx", ".tsx"],
        "data": [".csv", ".sql", ".db"],
        "scripts": [".sh", ".bash", ".ps1", ".bat"],
    }

    BINARY_EXTENSIONS = [
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".zip", ".tar", ".gz", ".rar",
        ".exe", ".dll", ".so", ".dylib",
        ".pyc", ".pyo", ".class",
    ]

    @classmethod
    def discover(cls, path: str, config: WorkflowConfig) -> list[FileInfo]:
        """Findet alle relevanten Dateien im Verzeichnis."""
        files = []
        root_path = Path(path)

        if not root_path.exists():
            return files

        for file_path in root_path.rglob("*"):
            # Ignoriere Verzeichnisse
            if file_path.is_dir():
                continue

            # Ignoriere Patterns
            if any(pattern in str(file_path) for pattern in config.ignore_patterns):
                continue

            # Pruefe Extension
            ext = file_path.suffix.lower()
            if ext not in config.file_extensions and ext not in cls.BINARY_EXTENSIONS:
                continue

            # Pruefe Groesse
            try:
                size = file_path.stat().st_size
                if size > config.max_file_size:
                    continue
            except OSError:
                continue

            # Erstelle FileInfo
            is_binary = ext in cls.BINARY_EXTENSIONS
            category = cls._categorize(ext)

            files.append(FileInfo(
                path=str(file_path),
                name=file_path.name,
                extension=ext,
                size=size,
                is_binary=is_binary,
                category=category,
            ))

        return files

    @classmethod
    def _categorize(cls, extension: str) -> str:
        """Kategorisiert eine Datei nach Extension."""
        for category, extensions in cls.CATEGORY_PATTERNS.items():
            if extension in extensions:
                return category
        return "other"

    @classmethod
    def read_file_safe(cls, path: str, max_size: int = 50000) -> tuple[str, bool]:
        """
        Liest eine Datei sicher ein.

        Returns:
            Tuple aus (inhalt, erfolg)
        """
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_size)
                if len(content) == max_size:
                    content += "\n... [Datei gekuerzt]"
                return content, True
        except Exception as e:
            return f"Fehler beim Lesen: {e}", False


# =============================================================================
# Prompt Templates fuer die verschiedenen Workflow-Phasen
# =============================================================================

PROMPTS = {
    "discover_summary": """
Analysiere diese Dateiliste und erstelle eine kurze Uebersicht:

{file_list}

Erstelle eine Zusammenfassung mit:
1. Anzahl Dateien pro Kategorie
2. Hauptverzeichnisse/Module
3. Vermuteter Projekttyp
""",

    "categorize_files": """
Basierend auf den Dateinamen und Pfaden, kategorisiere diese Dateien:

{file_list}

Gruppiere nach:
- Kernfunktionalitaet
- Tests
- Konfiguration
- Dokumentation
- Utilities
""",

    "analyze_file": """
Bisheriges Projektwissen:
{knowledge}

Analysiere diese Datei und extrahiere:
1. Zweck/Funktion
2. Wichtige Klassen/Funktionen
3. Abhaengigkeiten
4. Besonderheiten

Datei: {filename}
```
{content}
```

Erstelle eine praegnante Zusammenfassung (max 200 Worte).
""",

    "analyze_chunk": """
Bisheriges Wissen ueber {filename}:
{previous_summary}

Analysiere diesen Teil der Datei (Chunk {chunk_num}/{total_chunks}):
```
{content}
```

Ergaenze die Zusammenfassung mit neuen Erkenntnissen.
""",

    "create_module_doc": """
Projektwissen:
{knowledge}

Dateien in diesem Modul:
{files}

Zusammenfassungen:
{summaries}

Erstelle eine Moduldokumentation mit:
1. Moduluebersicht
2. Hauptkomponenten
3. Abhaengigkeiten
4. Verwendung
""",

    "create_final_doc": """
Gesammeltes Projektwissen:
{knowledge}

Modulzusammenfassungen:
{module_docs}

Erstelle eine vollstaendige Projektdokumentation mit:
1. Projektuebersicht
2. Architektur
3. Modulbeschreibungen
4. Installation/Verwendung
5. Wichtige Hinweise

Format: Markdown
""",
}
