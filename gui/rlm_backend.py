"""
RLM Backend Modul

Dieses Modul stellt die Backend-Funktionalitaet fuer die Streamlit GUI bereit.
Es kapselt die RLM-Logik und macht sie ueber einfache Funktionen zugaenglich.
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator

from dotenv import load_dotenv


@dataclass
class RLMConfig:
    """Konfiguration fuer den RLM-Client."""
    base_url: str = "http://0.0.0.0:5567/v1"
    api_key: str = "dummy"
    model_name: str = "GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf"
    max_iterations: int = 15
    max_depth: int = 1
    environment: str = "docker"
    docker_image: str = "python:3.11-slim"
    log_dir: str = "./logs"
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "RLMConfig":
        """Laedt Konfiguration aus Umgebungsvariablen."""
        load_dotenv()
        return cls(
            base_url=os.getenv("RLM_BASE_URL", cls.base_url),
            api_key=os.getenv("RLM_API_KEY", cls.api_key),
            model_name=os.getenv("RLM_MODEL_NAME", cls.model_name),
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", cls.max_iterations)),
            environment=os.getenv("RLM_ENVIRONMENT", cls.environment),
            log_dir=os.getenv("RLM_LOG_DIR", cls.log_dir),
        )


@dataclass
class RLMResult:
    """Ergebnis einer RLM-Completion."""
    success: bool
    response: str
    execution_time: float = 0.0
    iterations: int = 0
    error: Optional[str] = None
    log_file: Optional[str] = None


class RLMBackend:
    """
    Backend-Klasse fuer RLM-Operationen.

    Verwendung:
        backend = RLMBackend(config)
        result = backend.run_completion(context, aufgabe)
    """

    def __init__(self, config: Optional[RLMConfig] = None):
        """Initialisiert das Backend mit der gegebenen Konfiguration."""
        self.config = config or RLMConfig.from_env()
        self._rlm = None
        self._logger = None

    def _init_rlm(self):
        """Initialisiert RLM lazy beim ersten Aufruf."""
        if self._rlm is not None:
            return

        from rlm import RLM
        from rlm.logger import RLMLogger

        # Logger erstellen
        os.makedirs(self.config.log_dir, exist_ok=True)
        self._logger = RLMLogger(log_dir=self.config.log_dir)

        # Environment kwargs
        env_kwargs = {}
        if self.config.environment == "docker":
            env_kwargs["image"] = self.config.docker_image

        # RLM initialisieren
        self._rlm = RLM(
            backend="openai",
            backend_kwargs={
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "model_name": self.config.model_name,
            },
            environment=self.config.environment,
            environment_kwargs=env_kwargs,
            max_depth=self.config.max_depth,
            max_iterations=self.config.max_iterations,
            logger=self._logger,
            verbose=self.config.verbose,
        )

    def check_llm_server(self) -> tuple[bool, str]:
        """
        Prueft ob der LLM-Server erreichbar ist.

        Returns:
            Tuple aus (erreichbar: bool, nachricht: str)
        """
        import requests
        try:
            response = requests.get(
                f"{self.config.base_url}/models",
                timeout=5
            )
            if response.status_code == 200:
                return True, "LLM-Server erreichbar"
            else:
                return False, f"Server antwortet mit Status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Keine Verbindung zu {self.config.base_url}"
        except requests.exceptions.Timeout:
            return False, "Timeout beim Verbinden"
        except Exception as e:
            return False, f"Fehler: {str(e)}"

    def check_docker(self) -> tuple[bool, str]:
        """
        Prueft ob Docker verfuegbar ist.

        Returns:
            Tuple aus (verfuegbar: bool, nachricht: str)
        """
        import subprocess
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "Docker laeuft"
            else:
                return False, "Docker-Daemon nicht erreichbar"
        except FileNotFoundError:
            return False, "Docker nicht installiert"
        except subprocess.TimeoutExpired:
            return False, "Docker-Timeout"
        except Exception as e:
            return False, f"Fehler: {str(e)}"

    def run_completion(
        self,
        context: str,
        aufgabe: str,
    ) -> RLMResult:
        """
        Fuehrt eine RLM-Completion aus.

        Args:
            context: Der Kontext/Daten
            aufgabe: Die Aufgabe/Frage

        Returns:
            RLMResult mit dem Ergebnis
        """
        try:
            self._init_rlm()

            result = self._rlm.completion(
                prompt=context,
                root_prompt=aufgabe,
            )

            # Log-Datei finden
            log_file = None
            if self._logger and hasattr(self._logger, 'current_log_file'):
                log_file = self._logger.current_log_file

            return RLMResult(
                success=True,
                response=result.response,
                execution_time=result.execution_time,
                iterations=len(result.iterations) if hasattr(result, 'iterations') else 0,
                log_file=log_file,
            )

        except Exception as e:
            return RLMResult(
                success=False,
                response="",
                error=str(e),
            )

    def run_completion_with_source_mount(
        self,
        source_path: str,
        context: str,
        aufgabe: str,
        container_mount_path: str = "/project",
    ) -> RLMResult:
        """
        Fuehrt eine RLM-Completion mit gemountentem Quellverzeichnis aus.

        ECHTES RLM-PARADIGMA:
        - Das Quellverzeichnis wird in den Docker-Container gemountet
        - Das LLM kann ALLE Dateien selbst lesen via REPL
        - Das LLM entscheidet selbst ueber Chunking und Rekursion

        Args:
            source_path: Pfad zum Quellverzeichnis auf dem Host
            context: Kontext-String (z.B. Dateiliste)
            aufgabe: Die Aufgabe/Frage (root_prompt)
            container_mount_path: Pfad im Container (default: /project)

        Returns:
            RLMResult mit dem Ergebnis
        """
        try:
            from rlm import RLM
            from rlm.logger import RLMLogger
            from gui.docker_repl_extended import DockerREPLExtended

            # Logger erstellen
            os.makedirs(self.config.log_dir, exist_ok=True)
            logger = RLMLogger(log_dir=self.config.log_dir)

            # Environment kwargs MIT extra_mounts
            env_kwargs = {
                "image": self.config.docker_image,
                "extra_mounts": [
                    (source_path, container_mount_path),  # Projektverzeichnis mounten
                ],
            }

            # Eigene RLM-Instanz mit erweitertem Docker REPL
            # Wir muessen den environment_type auf einen custom handler setzen
            # Da RLM nur built-in environments kennt, erstellen wir RLM manuell

            # WICHTIG: Wir muessen rlm.core.rlm.get_environment patchen,
            # nicht rlm.environments.get_environment!
            # Der Import in rlm.core.rlm ist: from rlm.environments import get_environment
            # Das bedeutet die Funktion ist direkt in rlm.core.rlm gespeichert
            import rlm.core.rlm as rlm_module
            original_get_env = rlm_module.get_environment

            def patched_get_environment(env_type, kwargs):
                if env_type == "docker" and "extra_mounts" in kwargs:
                    return DockerREPLExtended(**kwargs)
                return original_get_env(env_type, kwargs)

            # Temporaer patchen - an der richtigen Stelle!
            rlm_module.get_environment = patched_get_environment

            try:
                rlm_instance = RLM(
                    backend="openai",
                    backend_kwargs={
                        "base_url": self.config.base_url,
                        "api_key": self.config.api_key,
                        "model_name": self.config.model_name,
                    },
                    environment="docker",
                    environment_kwargs=env_kwargs,
                    max_depth=self.config.max_depth,
                    max_iterations=self.config.max_iterations,
                    logger=logger,
                    verbose=self.config.verbose,
                )

                result = rlm_instance.completion(
                    prompt=context,
                    root_prompt=aufgabe,
                )
            finally:
                # Patch zuruecksetzen
                rlm_module.get_environment = original_get_env

            # Log-Datei finden
            log_file = None
            if logger and hasattr(logger, 'current_log_file'):
                log_file = logger.current_log_file

            return RLMResult(
                success=True,
                response=result.response,
                execution_time=result.execution_time,
                iterations=len(result.iterations) if hasattr(result, 'iterations') else 0,
                log_file=log_file,
            )

        except Exception as e:
            import traceback
            return RLMResult(
                success=False,
                response="",
                error=f"{str(e)}\n{traceback.format_exc()}",
            )

    def get_log_files(self) -> list[Path]:
        """Gibt eine Liste aller Log-Dateien zurueck."""
        log_dir = Path(self.config.log_dir)
        if not log_dir.exists():
            return []
        return sorted(log_dir.glob("*.jsonl"), reverse=True)

    def read_log_file(self, filepath: str) -> list[dict]:
        """Liest eine Log-Datei und gibt die Eintraege zurueck."""
        entries = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception:
            pass
        return entries


# Beispiel-Kontexte fuer die GUI
EXAMPLE_CONTEXTS = {
    "Verkaufsdaten": """
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
""",
    "Fibonacci": """
Die Fibonacci-Folge ist eine mathematische Zahlenfolge:
0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...

Jede Zahl ist die Summe der beiden vorhergehenden.
Formel: F(n) = F(n-1) + F(n-2)
""",
    "Python Code": """
def calculate_stats(numbers):
    total = sum(numbers)
    avg = total / len(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    return {
        'total': total,
        'average': avg,
        'min': minimum,
        'max': maximum
    }

data = [23, 45, 67, 12, 89, 34, 56, 78, 90, 11]
""",
}

EXAMPLE_TASKS = {
    "Verkaufsdaten": "Analysiere die Verkaufsdaten: Berechne Gesamtumsatz, besten/schlechtesten Monat und Durchschnitt.",
    "Fibonacci": "Berechne die 20. Fibonacci-Zahl und erklaere den Algorithmus.",
    "Python Code": "Analysiere die Funktion und fuehre sie mit den Testdaten aus. Was ist das Ergebnis?",
}
