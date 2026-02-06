"""
Erweiterter Docker REPL mit Unterstuetzung fuer zusaetzliche Volume-Mounts.

Dies ermoeglicht das Mounten von Projektverzeichnissen in den Container,
damit das LLM ALLE Dateien selbst lesen kann - das echte RLM-Paradigma!
"""

import os
import subprocess
import tempfile
import json
import time
import threading
from http.server import HTTPServer

from rlm.environments.docker_repl import DockerREPL, LLMProxyHandler, _build_exec_script
from rlm.core.types import REPLResult


class DockerREPLExtended(DockerREPL):
    """
    Erweiterter DockerREPL mit Unterstuetzung fuer zusaetzliche Volume-Mounts.

    Verwendung:
        env_kwargs = {
            "extra_mounts": [
                ("/host/path/to/project", "/project"),  # Quellprojekt
            ]
        }
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        extra_mounts: list[tuple[str, str]] | None = None,
        **kwargs,
    ):
        """
        Args:
            extra_mounts: Liste von (host_path, container_path) Tupeln
                          z.B. [("/home/user/project", "/project")]
        """
        self.extra_mounts = extra_mounts or []

        # Rufe Parent-Konstruktor NICHT auf, da wir setup() ueberschreiben muessen
        # Stattdessen manuell initialisieren
        if persistent:
            raise NotImplementedError(
                "Persistent REPLs are currently not supported for environment: DockerREPLExtended"
            )

        self.image = image
        self.lm_handler_address = lm_handler_address
        self.container_id: str | None = None
        self.proxy_server: HTTPServer | None = None
        self.proxy_thread: threading.Thread | None = None
        self.proxy_port: int = 0
        self.depth = depth
        self.persistent = persistent

        base_dir = os.environ.get(
            "RLM_DOCKER_WORKSPACE_DIR", os.path.join(os.getcwd(), ".rlm_workspace")
        )
        os.makedirs(base_dir, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="docker_repl_", dir=base_dir)
        self.pending_calls = []
        self._calls_lock = threading.Lock()

        # Setup mit extra_mounts
        self.setup_with_mounts()

        if context_payload:
            self.load_context(context_payload)
        if setup_code:
            self.execute_code(setup_code)

    def setup_with_mounts(self):
        """Start proxy server und Docker container MIT zusaetzlichen Mounts."""
        # Start LLM proxy server
        handler = type(
            "Handler",
            (LLMProxyHandler,),
            {
                "lm_handler_address": self.lm_handler_address,
                "pending_calls": self.pending_calls,
                "lock": self._calls_lock,
                "depth": self.depth,
            },
        )
        self.proxy_server = HTTPServer(("0.0.0.0", 0), handler)
        self.proxy_port = self.proxy_server.server_address[1]
        self.proxy_thread = threading.Thread(target=self.proxy_server.serve_forever, daemon=True)
        self.proxy_thread.start()

        # Docker run command mit extra mounts
        docker_cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-v",
            f"{self.temp_dir}:/workspace",
        ]

        # Fuege extra mounts hinzu
        for host_path, container_path in self.extra_mounts:
            # Stelle sicher dass Host-Pfad existiert
            if os.path.exists(host_path):
                # Read-only mount fuer Sicherheit
                docker_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])

        docker_cmd.extend([
            "--add-host",
            "host.docker.internal:host-gateway",
            self.image,
            "tail",
            "-f",
            "/dev/null",
        ])

        result = subprocess.run(docker_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self.container_id = result.stdout.strip()

        # Install dependencies
        subprocess.run(
            ["docker", "exec", self.container_id, "pip", "install", "-q", "dill", "requests"],
            capture_output=True,
        )


def get_extended_docker_environment(env_kwargs: dict):
    """
    Factory-Funktion um DockerREPLExtended zu erstellen.

    Kann als custom environment verwendet werden.
    """
    return DockerREPLExtended(**env_kwargs)
