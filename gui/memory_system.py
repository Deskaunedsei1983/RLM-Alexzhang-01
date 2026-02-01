"""
Memory System fuer RLM

Implementiert Short-Term und Long-Term Memory fuer die Verarbeitung
grosser Dokumente mit Kontexterhaltung.

Short-Term Memory: Aktuelle Session, wird bei Neustart geloescht
Long-Term Memory: Persistentes Wissen, bleibt erhalten
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class MemoryEntry:
    """Ein Eintrag im Memory-System."""
    key: str                      # Eindeutiger Schluessel
    content: str                  # Der Inhalt/Zusammenfassung
    source: str                   # Woher kommt das Wissen (Dateiname, Chunk-ID)
    timestamp: str                # Wann erstellt
    entry_type: str = "chunk"     # chunk, summary, final
    relevance: float = 1.0        # Relevanz-Score (0-1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


@dataclass
class MemoryStats:
    """Statistiken ueber das Memory."""
    short_term_entries: int = 0
    long_term_entries: int = 0
    total_chars: int = 0
    oldest_entry: str = ""
    newest_entry: str = ""


class MemorySystem:
    """
    Zweistufiges Memory-System:

    Short-Term: Schneller Zugriff, Session-basiert
    Long-Term: Persistente Speicherung in JSON-Dateien

    Verwendung:
        memory = MemorySystem(memory_dir="./memory")
        memory.add_short_term("chunk_1", "Zusammenfassung...", source="datei.py")
        memory.promote_to_long_term("chunk_1")  # Wichtiges Wissen behalten
        context = memory.get_relevant_context(max_chars=2000)
    """

    def __init__(self, memory_dir: str = "./memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.short_term_file = self.memory_dir / "short_term.json"
        self.long_term_file = self.memory_dir / "long_term.json"

        # In-Memory Caches
        self._short_term: dict[str, MemoryEntry] = {}
        self._long_term: dict[str, MemoryEntry] = {}

        # Long-Term aus Datei laden
        self._load_long_term()

    def _load_long_term(self):
        """Laedt Long-Term Memory aus Datei."""
        if self.long_term_file.exists():
            try:
                with open(self.long_term_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self._long_term[key] = MemoryEntry.from_dict(entry_data)
            except Exception:
                self._long_term = {}

    def _save_long_term(self):
        """Speichert Long-Term Memory in Datei."""
        data = {key: entry.to_dict() for key, entry in self._long_term.items()}
        with open(self.long_term_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_short_term(self):
        """Speichert Short-Term Memory in Datei (fuer Session-Recovery)."""
        data = {key: entry.to_dict() for key, entry in self._short_term.items()}
        with open(self.short_term_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_short_term(
        self,
        key: str,
        content: str,
        source: str = "",
        entry_type: str = "chunk",
        relevance: float = 1.0,
    ):
        """Fuegt einen Eintrag zum Short-Term Memory hinzu."""
        entry = MemoryEntry(
            key=key,
            content=content,
            source=source,
            timestamp=datetime.now().isoformat(),
            entry_type=entry_type,
            relevance=relevance,
        )
        self._short_term[key] = entry
        self._save_short_term()

    def add_long_term(
        self,
        key: str,
        content: str,
        source: str = "",
        entry_type: str = "knowledge",
        relevance: float = 1.0,
    ):
        """Fuegt einen Eintrag direkt zum Long-Term Memory hinzu."""
        entry = MemoryEntry(
            key=key,
            content=content,
            source=source,
            timestamp=datetime.now().isoformat(),
            entry_type=entry_type,
            relevance=relevance,
        )
        self._long_term[key] = entry
        self._save_long_term()

    def promote_to_long_term(self, key: str):
        """Befördert einen Short-Term Eintrag zu Long-Term."""
        if key in self._short_term:
            entry = self._short_term[key]
            self._long_term[key] = entry
            self._save_long_term()

    def get_short_term(self, key: str) -> Optional[MemoryEntry]:
        """Holt einen Short-Term Eintrag."""
        return self._short_term.get(key)

    def get_long_term(self, key: str) -> Optional[MemoryEntry]:
        """Holt einen Long-Term Eintrag."""
        return self._long_term.get(key)

    def get_all_short_term(self) -> list[MemoryEntry]:
        """Gibt alle Short-Term Eintraege zurueck."""
        return list(self._short_term.values())

    def get_all_long_term(self) -> list[MemoryEntry]:
        """Gibt alle Long-Term Eintraege zurueck."""
        return list(self._long_term.values())

    def get_relevant_context(
        self,
        max_chars: int = 2000,
        include_long_term: bool = True,
        include_short_term: bool = True,
    ) -> str:
        """
        Erstellt einen relevanten Kontext aus dem Memory.
        Priorisiert nach Relevanz und Aktualitaet.
        """
        entries = []

        if include_long_term:
            entries.extend(self._long_term.values())
        if include_short_term:
            entries.extend(self._short_term.values())

        # Sortiere nach Relevanz (hoch -> niedrig), dann nach Zeit (neu -> alt)
        entries.sort(key=lambda e: (-e.relevance, e.timestamp), reverse=True)

        # Baue Kontext auf
        context_parts = []
        current_chars = 0

        for entry in entries:
            entry_text = f"[{entry.source}]: {entry.content}"
            if current_chars + len(entry_text) > max_chars:
                # Kuerze wenn noetig
                remaining = max_chars - current_chars - 50
                if remaining > 100:
                    entry_text = entry_text[:remaining] + "..."
                    context_parts.append(entry_text)
                break

            context_parts.append(entry_text)
            current_chars += len(entry_text) + 1

        return "\n".join(context_parts)

    def clear_short_term(self):
        """Loescht das Short-Term Memory."""
        self._short_term = {}
        if self.short_term_file.exists():
            self.short_term_file.unlink()

    def clear_long_term(self):
        """Loescht das Long-Term Memory."""
        self._long_term = {}
        if self.long_term_file.exists():
            self.long_term_file.unlink()

    def clear_all(self):
        """Loescht alles."""
        self.clear_short_term()
        self.clear_long_term()

    def get_stats(self) -> MemoryStats:
        """Gibt Statistiken zurueck."""
        all_entries = list(self._short_term.values()) + list(self._long_term.values())

        total_chars = sum(len(e.content) for e in all_entries)

        timestamps = [e.timestamp for e in all_entries]
        oldest = min(timestamps) if timestamps else ""
        newest = max(timestamps) if timestamps else ""

        return MemoryStats(
            short_term_entries=len(self._short_term),
            long_term_entries=len(self._long_term),
            total_chars=total_chars,
            oldest_entry=oldest,
            newest_entry=newest,
        )

    def create_document_hash(self, content: str) -> str:
        """Erstellt einen Hash fuer ein Dokument (fuer Deduplizierung)."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
