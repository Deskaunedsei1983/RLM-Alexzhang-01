"""
Docker-Konfiguration fuer RLM REPL

Einstellungen fuer die Docker-basierte REPL-Umgebung.
Diese Datei definiert Container-Einstellungen und Sicherheitsparameter.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DockerConfig:
    """
    Konfigurationsklasse fuer die Docker-REPL-Umgebung.

    Attributes:
        image: Docker-Image fuer die REPL
        workspace_dir: Arbeitsverzeichnis im Container
        memory_limit: Speicherlimit (z.B. "512m")
        cpu_limit: CPU-Limit (z.B. "1.0" fuer einen Kern)
        timeout: Maximale Ausfuehrungszeit in Sekunden
        network_mode: Netzwerkmodus (none, bridge, host)
        read_only: Nur-Lese-Dateisystem
        no_new_privileges: Verhindert Privilege Escalation
    """

    # Container-Image
    image: str = "python:3.11-slim"
    custom_image: Optional[str] = None

    # Verzeichnisse
    workspace_dir: str = "/workspace"
    host_workspace: Optional[str] = None

    # Ressourcenlimits
    memory_limit: str = "512m"
    memory_swap: str = "512m"
    cpu_limit: float = 1.0
    pids_limit: int = 100

    # Zeitlimits
    timeout: int = 60
    execution_timeout: int = 30

    # Sicherheitseinstellungen
    network_mode: str = "none"
    read_only: bool = False
    no_new_privileges: bool = True
    cap_drop: list = field(default_factory=lambda: ["ALL"])
    cap_add: list = field(default_factory=list)

    # Zusaetzliche Optionen
    remove_after_run: bool = True
    user: str = "nobody"
    tmpfs_size: str = "64m"

    def get_image(self) -> str:
        """Gibt das zu verwendende Docker-Image zurueck."""
        return self.custom_image if self.custom_image else self.image

    def to_docker_run_args(self) -> list[str]:
        """
        Generiert die Docker-Run-Argumente.

        Returns:
            Liste der Docker-CLI-Argumente
        """
        args = []

        # Grundlegende Optionen
        args.extend(["-d"])  # Detached mode

        if self.remove_after_run:
            args.extend(["--rm"])

        # Ressourcenlimits
        args.extend(["-m", self.memory_limit])
        args.extend(["--memory-swap", self.memory_swap])
        args.extend(["--cpus", str(self.cpu_limit)])
        args.extend(["--pids-limit", str(self.pids_limit)])

        # Sicherheitsoptionen
        args.extend(["--network", self.network_mode])

        if self.read_only:
            args.extend(["--read-only"])

        if self.no_new_privileges:
            args.extend(["--security-opt", "no-new-privileges"])

        # Capabilities
        for cap in self.cap_drop:
            args.extend(["--cap-drop", cap])

        for cap in self.cap_add:
            args.extend(["--cap-add", cap])

        # Benutzer
        if self.user:
            args.extend(["--user", self.user])

        # tmpfs fuer temporaere Dateien
        args.extend(["--tmpfs", f"/tmp:rw,noexec,nosuid,size={self.tmpfs_size}"])

        # Host-Gateway fuer LLM-Proxy
        args.extend(["--add-host", "host.docker.internal:host-gateway"])

        return args

    def get_security_profile(self) -> dict:
        """
        Gibt das Sicherheitsprofil als Dictionary zurueck.

        Returns:
            Dictionary mit Sicherheitseinstellungen
        """
        return {
            "network_isolated": self.network_mode == "none",
            "read_only_fs": self.read_only,
            "no_privileges": self.no_new_privileges,
            "memory_limited": self.memory_limit,
            "cpu_limited": self.cpu_limit,
            "process_limited": self.pids_limit,
        }

    def validate(self) -> list[str]:
        """
        Validiert die Docker-Konfiguration.

        Returns:
            Liste mit Warnungen (leer wenn alles OK)
        """
        warnings = []

        # Pruefe Netzwerk-Sicherheit
        if self.network_mode not in ("none", "bridge"):
            warnings.append(
                f"Netzwerkmodus '{self.network_mode}' kann ein Sicherheitsrisiko sein"
            )

        # Pruefe Ressourcenlimits
        if self.cpu_limit > 2.0:
            warnings.append(f"CPU-Limit {self.cpu_limit} ist sehr hoch")

        # Pruefe Benutzer
        if self.user in ("root", "0"):
            warnings.append("Container laeuft als root - Sicherheitsrisiko!")

        return warnings


# Vordefinierte Konfigurationsprofile
class DockerProfiles:
    """Vordefinierte Docker-Konfigurationsprofile."""

    @staticmethod
    def secure() -> DockerConfig:
        """Maximale Sicherheit - empfohlen fuer Produktion."""
        return DockerConfig(
            network_mode="none",
            read_only=True,
            no_new_privileges=True,
            memory_limit="256m",
            cpu_limit=0.5,
            timeout=30,
            user="nobody",
        )

    @staticmethod
    def development() -> DockerConfig:
        """Entwicklungsmodus - mehr Freiheit fuer Tests."""
        return DockerConfig(
            network_mode="none",
            read_only=False,
            no_new_privileges=True,
            memory_limit="1g",
            cpu_limit=2.0,
            timeout=120,
            user="nobody",
        )

    @staticmethod
    def minimal() -> DockerConfig:
        """Minimale Ressourcen - fuer einfache Aufgaben."""
        return DockerConfig(
            network_mode="none",
            read_only=True,
            memory_limit="128m",
            cpu_limit=0.25,
            timeout=15,
        )


# Standard-Instanz
DEFAULT_DOCKER_CONFIG = DockerConfig()
