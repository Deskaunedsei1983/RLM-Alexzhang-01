#!/usr/bin/env python3
"""
RLM Test - Lokaler LLM Server mit Docker REPL

Voraussetzungen:
1. .env Datei mit LLM-Konfiguration (cp .env.example .env)
2. LLM-Server laeuft auf http://0.0.0.0:5567/v1
3. Docker installiert und laeuft
4. Docker-Image gebaut: docker build -t rlm-sandbox -f Dockerfile.sandbox .

Ausfuehrung:
    python examples/test_local_llm.py
"""

import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

# .env Datei laden
load_dotenv()

# Konfiguration aus Umgebungsvariablen
BASE_URL = os.getenv("RLM_BASE_URL", "http://0.0.0.0:5567/v1")
API_KEY = os.getenv("RLM_API_KEY", "dummy")
MODEL_NAME = os.getenv("RLM_MODEL_NAME", "GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf")

print("=" * 60)
print("RLM Test - Lokaler LLM mit Docker REPL")
print("=" * 60)
print(f"LLM Server: {BASE_URL}")
print(f"Modell: {MODEL_NAME}")
print("=" * 60)

# Logger initialisieren (speichert Logs in ./logs/)
logger = RLMLogger(log_dir="./logs")

# RLM initialisieren
rlm = RLM(
    backend="openai",  # OpenAI-kompatible API (llama.cpp, vLLM, etc.)
    backend_kwargs={
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "model_name": MODEL_NAME,
    },
    environment="docker",  # Docker-REPL fuer sichere Ausfuehrung
    environment_kwargs={
        "image": "python:3.11-slim",  # Standard-Image oder "rlm-sandbox"
    },
    max_depth=1,
    max_iterations=15,
    logger=logger,
    verbose=True,
)

# =============================================================================
# Beispiel: prompt + root_prompt
# =============================================================================
# - prompt: Der Kontext/Daten die das LLM verarbeiten soll
# - root_prompt: Die eigentliche Frage/Aufgabe
# =============================================================================

# Kontext (wird als 'context' Variable in der REPL verfuegbar)
context = """
Verkaufsdaten Q1-Q4 2024:

Q1 (Januar-Maerz):
  - Januar: 45.000 EUR
  - Februar: 52.000 EUR
  - Maerz: 48.000 EUR

Q2 (April-Juni):
  - April: 61.000 EUR
  - Mai: 58.000 EUR
  - Juni: 72.000 EUR

Q3 (Juli-September):
  - Juli: 68.000 EUR
  - August: 55.000 EUR
  - September: 63.000 EUR

Q4 (Oktober-Dezember):
  - Oktober: 71.000 EUR
  - November: 89.000 EUR
  - Dezember: 125.000 EUR
"""

# Die Aufgabe/Frage
aufgabe = """
Analysiere die Verkaufsdaten und beantworte:
1. Gesamtumsatz des Jahres
2. Bester und schlechtester Monat
3. Durchschnittlicher Monatsumsatz
4. Quartal mit hoechstem Umsatz
"""

print("\n--- Starte RLM Completion ---\n")

try:
    result = rlm.completion(
        prompt=context,       # Kontext/Daten
        root_prompt=aufgabe,  # Aufgabe/Frage
    )

    print("\n" + "=" * 60)
    print("ERGEBNIS:")
    print("=" * 60)
    print(result.response)
    print("\n" + "=" * 60)
    print(f"Ausfuehrungszeit: {result.execution_time:.2f} Sekunden")
    print(f"Iterationen: {len(result.iterations) if hasattr(result, 'iterations') else 'N/A'}")
    print("=" * 60)
    print("\nLogs gespeichert in: ./logs/")

except Exception as e:
    print(f"\nFEHLER: {e}")
    print("\nCheckliste:")
    print(f"  [ ] LLM-Server laeuft auf {BASE_URL}?")
    print("  [ ] Docker-Daemon gestartet?")
    print("  [ ] .env Datei vorhanden?")
    print("\nServer testen:")
    print(f"  curl {BASE_URL}/models")
