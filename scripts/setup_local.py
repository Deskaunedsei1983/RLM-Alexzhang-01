#!/usr/bin/env python3
"""
RLM Lokales Setup-Skript

Dieses Skript automatisiert die Einrichtung von RLM fuer die lokale Nutzung.
Es prueft Voraussetzungen, erstellt Konfigurationen und validiert die Installation.

Verwendung:
    python scripts/setup_local.py
    python scripts/setup_local.py --check-only
    python scripts/setup_local.py --with-docker
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Farben fuer Terminal-Ausgabe
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Druckt eine formatierte Ueberschrift."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(text: str):
    """Druckt eine Erfolgsmeldung."""
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_warning(text: str):
    """Druckt eine Warnung."""
    print(f"{Colors.YELLOW}[WARNUNG]{Colors.RESET} {text}")


def print_error(text: str):
    """Druckt eine Fehlermeldung."""
    print(f"{Colors.RED}[FEHLER]{Colors.RESET} {text}")


def print_info(text: str):
    """Druckt eine Information."""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {text}")


def check_python_version() -> bool:
    """Prueft die Python-Version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.11+ erforderlich (gefunden: {version.major}.{version.minor})")
        return False


def check_uv_installed() -> bool:
    """Prueft ob uv installiert ist."""
    if shutil.which("uv"):
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        print_success(f"uv {result.stdout.strip()}")
        return True
    else:
        print_warning("uv nicht gefunden - empfohlen fuer schnelles Dependency-Management")
        return False


def check_docker_installed() -> bool:
    """Prueft ob Docker installiert ist."""
    if shutil.which("docker"):
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        print_success(f"Docker: {result.stdout.strip()}")
        return True
    else:
        print_warning("Docker nicht gefunden - erforderlich fuer Docker-REPL")
        return False


def check_docker_running() -> bool:
    """Prueft ob Docker laeuft."""
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print_success("Docker Daemon laeuft")
        return True
    else:
        print_warning("Docker Daemon nicht erreichbar")
        return False


def check_llm_server(base_url: str = "http://0.0.0.0:5567/v1") -> bool:
    """Prueft ob der LLM-Server erreichbar ist."""
    try:
        import requests
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            print_success(f"LLM-Server erreichbar unter {base_url}")
            return True
    except Exception:
        pass

    print_warning(f"LLM-Server nicht erreichbar unter {base_url}")
    return False


def install_dependencies() -> bool:
    """Installiert die Abhaengigkeiten."""
    print_info("Installiere Abhaengigkeiten...")

    if shutil.which("uv"):
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
    else:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True
        )

    if result.returncode == 0:
        print_success("Abhaengigkeiten installiert")
        return True
    else:
        print_error(f"Installation fehlgeschlagen: {result.stderr}")
        return False


def build_docker_image() -> bool:
    """Baut das Docker-Sandbox-Image."""
    print_info("Baue Docker-Sandbox-Image...")

    result = subprocess.run(
        ["docker", "build", "-t", "rlm-sandbox", "-f", "Dockerfile.sandbox", "."],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print_success("Docker-Image 'rlm-sandbox' erfolgreich gebaut")
        return True
    else:
        print_error(f"Docker-Build fehlgeschlagen: {result.stderr}")
        return False


def create_env_file() -> bool:
    """Erstellt eine .env-Datei falls nicht vorhanden."""
    env_path = Path(".env")

    if env_path.exists():
        print_info(".env Datei existiert bereits")
        return True

    env_content = """# RLM Umgebungsvariablen
# ======================

# Lokaler LLM-Server (z.B. llama.cpp)
RLM_LLM_BASE_URL=http://0.0.0.0:5567/v1
RLM_LLM_API_KEY=dummy
RLM_LLM_MODEL_NAME=GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf

# Optional: OpenAI API (falls gewuenscht)
# OPENAI_API_KEY=sk-...

# Optional: Anthropic API (falls gewuenscht)
# ANTHROPIC_API_KEY=sk-ant-...

# Docker-Einstellungen
RLM_DOCKER_WORKSPACE_DIR=./.rlm_workspace

# Logging
RLM_LOG_LEVEL=INFO
"""

    env_path.write_text(env_content)
    print_success(".env Datei erstellt")
    return True


def run_tests() -> bool:
    """Fuehrt die Tests aus."""
    print_info("Fuehre Tests aus...")

    if shutil.which("uv"):
        result = subprocess.run(["uv", "run", "pytest", "-v", "--tb=short"], capture_output=True, text=True)
    else:
        result = subprocess.run([sys.executable, "-m", "pytest", "-v", "--tb=short"], capture_output=True, text=True)

    if result.returncode == 0:
        print_success("Alle Tests bestanden")
        return True
    else:
        print_warning(f"Einige Tests fehlgeschlagen:\n{result.stdout}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RLM Lokales Setup-Skript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
    python scripts/setup_local.py              # Vollstaendiges Setup
    python scripts/setup_local.py --check-only # Nur Pruefungen
    python scripts/setup_local.py --with-docker # Mit Docker-Image-Build
        """
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Nur Voraussetzungen pruefen, nichts installieren"
    )
    parser.add_argument(
        "--with-docker",
        action="store_true",
        help="Docker-Sandbox-Image bauen"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Tests ueberspringen"
    )

    args = parser.parse_args()

    print_header("RLM Setup")

    # Wechsle ins Projektverzeichnis
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print_info(f"Projektverzeichnis: {project_root}")

    # Pruefungen
    print_header("Voraussetzungen")

    checks_passed = True
    checks_passed &= check_python_version()
    has_uv = check_uv_installed()
    has_docker = check_docker_installed()

    if has_docker:
        check_docker_running()

    check_llm_server()

    if args.check_only:
        print_header("Pruefung abgeschlossen")
        sys.exit(0 if checks_passed else 1)

    # Installation
    print_header("Installation")

    if not install_dependencies():
        print_error("Abbruch wegen fehlgeschlagener Installation")
        sys.exit(1)

    create_env_file()

    # Docker-Image bauen
    if args.with_docker and has_docker:
        print_header("Docker Setup")
        build_docker_image()

    # Tests ausfuehren
    if not args.skip_tests:
        print_header("Tests")
        run_tests()

    # Zusammenfassung
    print_header("Setup abgeschlossen")

    print(f"""
{Colors.GREEN}RLM wurde erfolgreich eingerichtet!{Colors.RESET}

Naechste Schritte:
1. Starte deinen LLM-Server (llama.cpp, vLLM, etc.)
2. Passe die .env Datei an deine Umgebung an
3. Fuehre ein Beispiel aus:

   {Colors.BOLD}python examples/quickstart_local.py{Colors.RESET}

Dokumentation:
   docs/installation-de.md

Bei Fragen oder Problemen:
   https://github.com/alexzhang/rlm/issues
    """)


if __name__ == "__main__":
    main()
