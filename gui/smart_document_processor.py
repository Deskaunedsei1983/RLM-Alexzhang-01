"""
Smart Document Processor fuer RLM

Verarbeitet grosse Dokumente intelligent durch:
1. Chunking - Aufteilen in verarbeitbare Teile
2. Progressive Summarization - Jeder Chunk wird zusammengefasst
3. Memory Integration - Short/Long-Term Speicherung
4. Context Accumulation - Wissen wird ueber Chunks aufgebaut
5. Final Synthesis - Alle Erkenntnisse werden zusammengefuehrt

Verwendung:
    processor = SmartDocumentProcessor(backend, memory)
    result = processor.process_document(content, filename, aufgabe)
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Generator, Callable
from pathlib import Path

from memory_system import MemorySystem, MemoryEntry


@dataclass
class ChunkResult:
    """Ergebnis der Verarbeitung eines Chunks."""
    chunk_id: str
    chunk_num: int
    total_chunks: int
    summary: str
    key_findings: list[str] = field(default_factory=list)
    relevance: float = 1.0


@dataclass
class ProcessingConfig:
    """Konfiguration fuer die Dokumentverarbeitung."""
    # Chunking
    max_chunk_size: int = 3000          # Zeichen pro Chunk
    chunk_overlap: int = 300            # Ueberlappung zwischen Chunks
    max_chunks: int = 50                # Maximale Anzahl Chunks

    # Context Management
    max_context_size: int = 2500        # Max Zeichen fuer Kontext
    summary_length: int = 200           # Ziel-Laenge fuer Zusammenfassungen

    # Memory
    auto_promote_threshold: float = 0.8  # Ab dieser Relevanz -> Long-Term

    # Processing
    progress_callback: Optional[Callable[[float, str], None]] = None


@dataclass
class DocumentResult:
    """Gesamtergebnis der Dokumentverarbeitung."""
    success: bool
    document_id: str
    filename: str
    total_chunks: int
    processed_chunks: int
    final_summary: str
    answer: str
    chunk_results: list[ChunkResult] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None


class SmartDocumentProcessor:
    """
    Intelligenter Dokumentenprozessor mit Memory-Integration.

    Verarbeitet grosse Dokumente chunk-weise und baut dabei
    ein akkumulierendes Wissensmodell auf.
    """

    def __init__(
        self,
        backend,  # RLMBackend instance
        memory: Optional[MemorySystem] = None,
        config: Optional[ProcessingConfig] = None,
    ):
        self.backend = backend
        self.memory = memory or MemorySystem()
        self.config = config or ProcessingConfig()

    def _create_document_id(self, content: str, filename: str) -> str:
        """Erstellt eine eindeutige ID fuer das Dokument."""
        hash_input = f"{filename}:{len(content)}:{content[:500]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _chunk_content(self, content: str) -> list[tuple[int, str]]:
        """
        Teilt den Inhalt in Chunks auf.

        Returns:
            Liste von (chunk_nummer, chunk_inhalt) Tupeln
        """
        if len(content) <= self.config.max_chunk_size:
            return [(1, content)]

        chunks = []
        start = 0
        chunk_num = 1

        while start < len(content) and chunk_num <= self.config.max_chunks:
            end = start + self.config.max_chunk_size

            # Versuche an sinnvoller Stelle zu trennen
            if end < len(content):
                # Prioritaet: Leerzeile > Zeilenumbruch > Leerzeichen
                best_break = -1

                # Suche Leerzeile
                double_newline = content.rfind('\n\n', start + self.config.max_chunk_size // 2, end)
                if double_newline > start:
                    best_break = double_newline + 2

                # Suche Zeilenumbruch
                if best_break == -1:
                    newline = content.rfind('\n', start + self.config.max_chunk_size // 2, end)
                    if newline > start:
                        best_break = newline + 1

                # Suche Leerzeichen
                if best_break == -1:
                    space = content.rfind(' ', start + self.config.max_chunk_size // 2, end)
                    if space > start:
                        best_break = space + 1

                if best_break > start:
                    end = best_break

            chunks.append((chunk_num, content[start:end]))
            chunk_num += 1

            # Naechster Start mit Ueberlappung
            start = end - self.config.chunk_overlap
            if start < 0:
                start = end

        return chunks

    def _get_accumulated_context(self, doc_id: str) -> str:
        """Holt den akkumulierten Kontext fuer dieses Dokument."""
        # Hole alle Short-Term Eintraege fuer dieses Dokument
        entries = []
        for entry in self.memory.get_all_short_term():
            if entry.source.startswith(doc_id):
                entries.append(entry)

        # Sortiere nach Relevanz
        entries.sort(key=lambda e: -e.relevance)

        # Baue Kontext auf
        context_parts = []
        current_size = 0

        for entry in entries:
            if current_size + len(entry.content) > self.config.max_context_size:
                break
            context_parts.append(f"[{entry.key}]: {entry.content}")
            current_size += len(entry.content)

        return "\n".join(context_parts)

    def _report_progress(self, progress: float, message: str):
        """Meldet Fortschritt zurueck."""
        if self.config.progress_callback:
            self.config.progress_callback(progress, message)

    def process_chunk(
        self,
        chunk_content: str,
        chunk_num: int,
        total_chunks: int,
        doc_id: str,
        filename: str,
        aufgabe: str,
    ) -> ChunkResult:
        """
        Verarbeitet einen einzelnen Chunk.

        Args:
            chunk_content: Der Inhalt des Chunks
            chunk_num: Nummer des Chunks (1-basiert)
            total_chunks: Gesamtzahl der Chunks
            doc_id: Dokument-ID
            filename: Dateiname
            aufgabe: Die zu bearbeitende Aufgabe

        Returns:
            ChunkResult mit Zusammenfassung
        """
        chunk_id = f"{doc_id}_chunk_{chunk_num}"

        # Hole bisheriges Wissen
        accumulated = self._get_accumulated_context(doc_id)

        # Erstelle Prompt
        if accumulated:
            context = f"""Bisheriges Wissen ueber {filename}:
{accumulated}

Aktueller Abschnitt ({chunk_num}/{total_chunks}):
{chunk_content}"""
        else:
            context = f"""Analysiere diesen Abschnitt der Datei {filename} ({chunk_num}/{total_chunks}):
{chunk_content}"""

        chunk_aufgabe = f"""Aufgabe: {aufgabe}

Analysiere diesen Abschnitt und:
1. Fasse die wichtigsten Erkenntnisse zusammen (max {self.config.summary_length} Zeichen)
2. Notiere relevante Details fuer die Gesamtaufgabe
3. Baue auf dem bisherigen Wissen auf

Antworte strukturiert und praegnant."""

        # Fuehre RLM aus
        result = self.backend.run_completion(context, chunk_aufgabe)

        if result.success:
            summary = result.response

            # Berechne Relevanz basierend auf Antwortlaenge und Position
            relevance = min(1.0, len(summary) / 500)
            # Spaetere Chunks oft relevanter (haben mehr Kontext)
            relevance *= 0.7 + 0.3 * (chunk_num / total_chunks)
        else:
            summary = f"Fehler bei Chunk {chunk_num}: {result.error}"
            relevance = 0.3

        # Speichere im Short-Term Memory
        self.memory.add_short_term(
            key=chunk_id,
            content=summary[:self.config.summary_length * 2],  # Etwas mehr behalten
            source=f"{doc_id}:{filename}",
            entry_type="chunk_summary",
            relevance=relevance,
        )

        # Bei hoher Relevanz zum Long-Term promoten
        if relevance >= self.config.auto_promote_threshold:
            self.memory.promote_to_long_term(chunk_id)

        return ChunkResult(
            chunk_id=chunk_id,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
            summary=summary,
            relevance=relevance,
        )

    def synthesize_final_answer(
        self,
        doc_id: str,
        filename: str,
        aufgabe: str,
        chunk_results: list[ChunkResult],
    ) -> str:
        """
        Synthetisiert die finale Antwort aus allen Chunk-Ergebnissen.
        """
        # Sammle alle Zusammenfassungen
        summaries = []
        for cr in chunk_results:
            summaries.append(f"[Teil {cr.chunk_num}]: {cr.summary[:300]}")

        all_summaries = "\n\n".join(summaries)

        # Hole auch Long-Term Wissen (falls relevant)
        long_term_context = ""
        for entry in self.memory.get_all_long_term():
            if doc_id in entry.source:
                long_term_context += f"\n[{entry.key}]: {entry.content}"

        # Erstelle Synthese-Prompt
        context = f"""Gesammelte Erkenntnisse aus {filename}:

{all_summaries}
{long_term_context}"""

        synthese_aufgabe = f"""Urspruengliche Aufgabe: {aufgabe}

Basierend auf allen gesammelten Erkenntnissen:
1. Beantworte die urspruengliche Aufgabe vollstaendig
2. Fasse die wichtigsten Punkte zusammen
3. Gib eine klare, strukturierte Antwort

Die Antwort sollte alle relevanten Informationen aus dem gesamten Dokument beruecksichtigen."""

        result = self.backend.run_completion(context, synthese_aufgabe)

        if result.success:
            return result.response
        else:
            return f"Fehler bei der Synthese: {result.error}"

    def process_document(
        self,
        content: str,
        filename: str,
        aufgabe: str,
    ) -> DocumentResult:
        """
        Verarbeitet ein komplettes Dokument.

        Args:
            content: Der gesamte Dokumentinhalt
            filename: Name der Datei
            aufgabe: Die zu bearbeitende Aufgabe

        Returns:
            DocumentResult mit allen Ergebnissen
        """
        start_time = datetime.now()
        doc_id = self._create_document_id(content, filename)

        self._report_progress(0.0, f"Starte Verarbeitung von {filename}")

        # Chunking
        chunks = self._chunk_content(content)
        total_chunks = len(chunks)

        self._report_progress(0.05, f"Dokument in {total_chunks} Teile aufgeteilt")

        # Speichere Dokument-Metadaten im Memory
        self.memory.add_short_term(
            key=f"{doc_id}_meta",
            content=f"Dokument: {filename}, Groesse: {len(content)} Zeichen, Chunks: {total_chunks}",
            source=doc_id,
            entry_type="metadata",
            relevance=0.5,
        )

        # Verarbeite jeden Chunk
        chunk_results = []
        for i, (chunk_num, chunk_content) in enumerate(chunks):
            progress = 0.1 + (0.7 * i / total_chunks)
            self._report_progress(progress, f"Verarbeite Teil {chunk_num}/{total_chunks}")

            try:
                result = self.process_chunk(
                    chunk_content=chunk_content,
                    chunk_num=chunk_num,
                    total_chunks=total_chunks,
                    doc_id=doc_id,
                    filename=filename,
                    aufgabe=aufgabe,
                )
                chunk_results.append(result)
            except Exception as e:
                # Bei Fehler trotzdem weitermachen
                chunk_results.append(ChunkResult(
                    chunk_id=f"{doc_id}_chunk_{chunk_num}",
                    chunk_num=chunk_num,
                    total_chunks=total_chunks,
                    summary=f"Fehler: {str(e)}",
                    relevance=0.0,
                ))

        self._report_progress(0.85, "Erstelle Gesamtzusammenfassung")

        # Erstelle Gesamtzusammenfassung
        summaries = [cr.summary for cr in chunk_results if cr.relevance > 0.3]
        final_summary = "\n".join(summaries[:5])  # Top 5 Zusammenfassungen

        # Synthetisiere finale Antwort
        self._report_progress(0.9, "Synthetisiere Antwort")
        answer = self.synthesize_final_answer(doc_id, filename, aufgabe, chunk_results)

        # Speichere Endergebnis im Long-Term Memory
        self.memory.add_long_term(
            key=f"{doc_id}_final",
            content=answer[:500],  # Gekuerztes Endergebnis
            source=f"{doc_id}:{filename}",
            entry_type="final_answer",
            relevance=1.0,
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        self._report_progress(1.0, "Verarbeitung abgeschlossen")

        return DocumentResult(
            success=True,
            document_id=doc_id,
            filename=filename,
            total_chunks=total_chunks,
            processed_chunks=len(chunk_results),
            final_summary=final_summary,
            answer=answer,
            chunk_results=chunk_results,
            processing_time=processing_time,
        )

    def process_multiple_documents(
        self,
        documents: list[tuple[str, str]],  # [(content, filename), ...]
        aufgabe: str,
    ) -> list[DocumentResult]:
        """
        Verarbeitet mehrere Dokumente mit geteiltem Wissen.

        Das Wissen aus frueheren Dokumenten fliesst in die
        Verarbeitung spaeterer Dokumente ein.
        """
        results = []
        total_docs = len(documents)

        for i, (content, filename) in enumerate(documents):
            self._report_progress(
                i / total_docs,
                f"Verarbeite Dokument {i+1}/{total_docs}: {filename}"
            )

            result = self.process_document(content, filename, aufgabe)
            results.append(result)

        return results

    def get_memory_stats(self) -> dict:
        """Gibt Statistiken ueber das Memory zurueck."""
        stats = self.memory.get_stats()
        return {
            "short_term_entries": stats.short_term_entries,
            "long_term_entries": stats.long_term_entries,
            "total_chars": stats.total_chars,
            "oldest_entry": stats.oldest_entry,
            "newest_entry": stats.newest_entry,
        }

    def clear_session(self):
        """Loescht das Short-Term Memory (neue Session)."""
        self.memory.clear_short_term()

    def clear_all_memory(self):
        """Loescht das gesamte Memory."""
        self.memory.clear_all()
