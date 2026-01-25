#!/usr/bin/env python3
"""
RLM Docker REPL - Lokaler LLM-Server

Dieses Beispiel zeigt, wie RLM mit Docker-Isolation
und einem lokalen LLM-Server verwendet wird.

Die Docker-REPL bietet:
- Vollstaendige Isolation vom Host-System
- Ressourcenlimits (CPU, Memory)
- Sicherer Sandboxed-Ausfuehrung

Voraussetzungen:
1. Docker installiert und laufend
2. Docker-Image gebaut: docker build -t rlm-sandbox -f Dockerfile.sandbox .
3. LLM-Server unter http://0.0.0.0:5567/v1

Verwendung:
    python examples/docker_repl_local.py
"""

import sys
from pathlib import Path

# Fuege Projektroot zum Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlm import RLM
from config.local_llm_config import LLMConfig, DEFAULT_CONFIG
from config.docker_config import DockerConfig, DockerProfiles


def check_docker_available() -> bool:
    """Prueft ob Docker verfuegbar ist."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    """Hauptfunktion fuer das Docker-REPL-Beispiel."""

    print("=" * 60)
    print("RLM Docker REPL - Lokaler LLM-Server")
    print("=" * 60)

    # Docker-Verfuegbarkeit pruefen
    if not check_docker_available():
        print("\nFEHLER: Docker ist nicht verfuegbar!")
        print("Bitte stelle sicher, dass:")
        print("  1. Docker installiert ist")
        print("  2. Der Docker-Daemon laeuft")
        print("  3. Du Berechtigungen hast, Docker zu nutzen")
        sys.exit(1)

    print("\nDocker verfuegbar!")

    # Konfigurationen laden
    llm_config = DEFAULT_CONFIG
    docker_config = DockerProfiles.development()

    print(f"\nLLM-Server: {llm_config.base_url}")
    print(f"Modell: {llm_config.model_name}")
    print(f"Docker-Image: {docker_config.get_image()}")
    print(f"Sicherheitsprofil: {docker_config.get_security_profile()}")
    print()

    # RLM mit Docker-REPL initialisieren
    print("Initialisiere RLM mit Docker-REPL...")

    rlm = RLM(
        backend="openai",
        backend_kwargs=llm_config.to_backend_kwargs(),
        environment="docker",
        environment_kwargs={
            "image": docker_config.get_image(),
        },
        max_iterations=10,
        verbose=True,
    )

    # Beispiel-Kontext mit Datenverarbeitung
    context = """
    Verkaufsdaten Q1-Q4:

    Q1: {"jan": 15000, "feb": 18000, "mar": 22000}
    Q2: {"apr": 19500, "may": 21000, "jun": 24500}
    Q3: {"jul": 23000, "aug": 20000, "sep": 26000}
    Q4: {"okt": 28000, "nov": 32000, "dez": 45000}

    Aufgabe: Analysiere die Verkaufsdaten.
    """

    query = "Berechne den Gesamtumsatz und identifiziere den besten Monat."

    print(f"Query: {query}")
    print("-" * 60)

    # RLM-Completion in Docker-Sandbox ausfuehren
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

    except Exception as e:
        print(f"\nFehler: {e}")
        print("\nBitte pruefe:")
        print("  1. Ist der LLM-Server erreichbar?")
        print("  2. Laeuft Docker korrekt?")
        print("  3. Wurde das Docker-Image gebaut?")

        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
