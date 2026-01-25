"""
Lokale LLM Konfiguration

Einstellungen fuer das lokale Sprachmodell (z.B. llama.cpp Server).
Diese Datei konfiguriert die Verbindung zum lokalen LLM-Server.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """
    Konfigurationsklasse fuer das lokale LLM.

    Attributes:
        base_url: URL zum lokalen LLM-Server
        api_key: API-Schluessel (bei lokalem Server oft "dummy")
        model_name: Name des geladenen Modells
        max_ctx_window: Maximale Kontextfenstergroesse
        max_output_tokens: Maximale Anzahl der Ausgabe-Tokens
        rag_limit: Maximale Zeichen fuer RAG-Kontext
        error_log_limit: Maximale Zeichen fuer Fehlermeldungen
    """

    # Server-Verbindung
    base_url: str = "http://0.0.0.0:5567/v1"
    api_key: str = "dummy"
    model_name: str = "GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf"

    # Token-Limits
    max_ctx_window: int = 4096
    max_output_tokens: int = 1024
    rag_limit: int = 1000
    error_log_limit: int = 500

    # Abgeleitete Werte
    @property
    def available_input_tokens(self) -> int:
        """Berechnet verfuegbare Input-Tokens nach Abzug der Reservierungen."""
        return self.max_ctx_window - self.max_output_tokens - self.error_log_limit

    @property
    def effective_rag_tokens(self) -> int:
        """Effektiv nutzbare RAG-Tokens."""
        return min(self.rag_limit, self.available_input_tokens)

    def to_backend_kwargs(self) -> dict:
        """
        Konvertiert die Konfiguration in backend_kwargs fuer RLM.

        Returns:
            Dictionary mit den Parametern fuer den OpenAI-kompatiblen Client
        """
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_name": self.model_name,
        }

    def validate(self) -> list[str]:
        """
        Validiert die Konfiguration.

        Returns:
            Liste mit Warnungen/Fehlern (leer wenn alles OK)
        """
        warnings = []

        if self.max_output_tokens > self.max_ctx_window:
            warnings.append(
                f"max_output_tokens ({self.max_output_tokens}) > "
                f"max_ctx_window ({self.max_ctx_window})"
            )

        if self.available_input_tokens < 500:
            warnings.append(
                f"Wenig verfuegbare Input-Tokens: {self.available_input_tokens}"
            )

        if self.rag_limit > self.available_input_tokens:
            warnings.append(
                f"RAG-Limit ({self.rag_limit}) uebersteigt verfuegbare "
                f"Input-Tokens ({self.available_input_tokens})"
            )

        return warnings


# Standard-Instanz fuer einfachen Import
DEFAULT_CONFIG = LLMConfig()


def get_config(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> LLMConfig:
    """
    Erstellt eine LLM-Konfiguration mit optionalen Ueberschreibungen.

    Args:
        base_url: Optionale Ueberschreibung der Server-URL
        api_key: Optionaler API-Schluessel
        model_name: Optionaler Modellname
        **kwargs: Weitere Konfigurationsoptionen

    Returns:
        Konfigurierte LLMConfig-Instanz
    """
    config_kwargs = {}

    if base_url is not None:
        config_kwargs["base_url"] = base_url
    if api_key is not None:
        config_kwargs["api_key"] = api_key
    if model_name is not None:
        config_kwargs["model_name"] = model_name

    config_kwargs.update(kwargs)

    return LLMConfig(**config_kwargs)
