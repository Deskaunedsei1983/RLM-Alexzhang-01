#!/usr/bin/env python3
"""
RLM Quickstart - Lokaler LLM-Server

Dieses Beispiel zeigt, wie RLM mit einem lokalen LLM-Server
(z.B. llama.cpp, vLLM, Ollama) verwendet wird.

Voraussetzungen:
1. Ein laufender OpenAI-kompatibler LLM-Server auf http://0.0.0.0:5567/v1
2. RLM installiert: pip install -e .

Verwendung:
    python examples/quickstart_local.py
"""

import sys
from pathlib import Path

# Fuege Projektroot zum Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlm import RLM
from config.local_llm_config import LLMConfig, DEFAULT_CONFIG


def main():
    """Hauptfunktion fuer das Quickstart-Beispiel."""

    print("=" * 60)
    print("RLM Quickstart - Lokaler LLM-Server")
    print("=" * 60)

    # Konfiguration laden und validieren
    config = DEFAULT_CONFIG
    warnings = config.validate()

    if warnings:
        print("\nKonfigurations-Warnungen:")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nLLM-Server: {config.base_url}")
    print(f"Modell: {config.model_name}")
    print(f"Max Context: {config.max_ctx_window} tokens")
    print(f"Max Output: {config.max_output_tokens} tokens")
    print()

    # RLM initialisieren
    print("Initialisiere RLM...")

    rlm = RLM(
        backend="openai",
        backend_kwargs=config.to_backend_kwargs(),
        environment="local",
        max_iterations=10,
        verbose=True,
    )

    # Beispiel-Kontext
    context = """
    Die Fibonacci-Folge ist eine mathematische Folge,
    bei der jede Zahl die Summe der beiden vorhergehenden ist.
    Die Folge beginnt mit: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

    Interessante Fakten:
    - Das Verhaeltnis aufeinanderfolgender Fibonacci-Zahlen
      naehert sich dem Goldenen Schnitt (phi ≈ 1.618).
    - Die Summe der ersten n Fibonacci-Zahlen ist F(n+2) - 1.
    - Fibonacci-Zahlen erscheinen in der Natur, z.B. in
      Blattanordnungen und Spiralen von Sonnenblumenkernen.
    """

    query = "Berechne die 10. Fibonacci-Zahl und erklaere den Algorithmus."

    print(f"Query: {query}")
    print("-" * 60)

    # RLM-Completion ausfuehren
    try:
        result = rlm.completion(
            prompt=context,
            root_prompt=query,
        )

        print("\n" + "=" * 60)
        print("ERGEBNIS:")
        print("=" * 60)
        print(result.response)
        print()
        print(f"Ausfuehrungszeit: {result.execution_time:.2f}s")
        print(f"Token-Verbrauch: {result.usage_summary}")

    except Exception as e:
        print(f"\nFehler: {e}")
        print("\nBitte stelle sicher, dass:")
        print(f"  1. Der LLM-Server unter {config.base_url} erreichbar ist")
        print(f"  2. Das Modell '{config.model_name}' geladen ist")
        sys.exit(1)


if __name__ == "__main__":
    main()
