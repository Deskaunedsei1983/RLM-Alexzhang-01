"""
Sichere REPL-Umgebung mit erweiterten Sicherheitspruefungen.

Diese Klasse erweitert die LocalREPL und DockerREPL um zusaetzliche
Sicherheitspruefungen, die verhindern, dass gefaehrlicher Code
ausgefuehrt wird.

Verwendung:
    from rlm.environments.secure_repl import SecureLocalREPL

    repl = SecureLocalREPL(...)
    result = repl.execute_code(code)  # Prueft Code vor Ausfuehrung
"""

import re
import sys
from typing import Any

# Relativer Import innerhalb des Pakets
sys.path.insert(0, str(__file__).rsplit("/rlm/", 1)[0])

from rlm.core.types import REPLResult
from rlm.environments.local_repl import LocalREPL


class SecurityViolationError(Exception):
    """Wird ausgeloest, wenn Code Sicherheitsregeln verletzt."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        message = "Sicherheitsverletzung erkannt:\n" + "\n".join(
            f"  - {v}" for v in violations
        )
        super().__init__(message)


class SecureREPLMixin:
    """
    Mixin-Klasse fuer sichere Code-Ausfuehrung.

    Kann mit LocalREPL oder DockerREPL kombiniert werden.
    """

    # Verbotene Module (Komplett-Liste)
    BLOCKED_MODULES = frozenset({
        # Systemzugriff
        "os", "sys", "subprocess", "shutil", "pathlib",
        # Netzwerk
        "socket", "http", "urllib", "requests", "httpx", "aiohttp",
        "ftplib", "smtplib", "telnetlib", "ssl",
        # Code-Ausfuehrung
        "code", "codeop", "ast", "dis", "inspect", "importlib",
        "pkgutil", "modulefinder",
        # Prozesse und Threads
        "multiprocessing", "threading", "concurrent", "asyncio", "signal",
        # Dateisystem
        "io", "tempfile", "glob", "fnmatch", "fileinput", "stat", "filecmp",
        # Serialisierung (Code-Injection-Risiko)
        "pickle", "cPickle", "shelve", "marshal", "dill",
        # Gefaehrliche Stdlib
        "ctypes", "pty", "tty", "termios", "resource", "gc",
        "traceback", "linecache",
        # Weitere
        "builtins", "__builtin__", "runpy", "pdb", "profile",
        "cProfile", "timeit",
    })

    # Verbotene Builtins
    BLOCKED_BUILTINS = frozenset({
        "eval", "exec", "compile", "open", "__import__",
        "input", "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr", "hasattr",
        "memoryview", "bytearray", "breakpoint",
        "help", "exit", "quit",
    })

    # Verbotene Muster (Regex)
    BLOCKED_PATTERNS = [
        # Shell-Escapes
        (re.compile(r'os\.system\s*\('), "os.system() Aufruf"),
        (re.compile(r'os\.popen\s*\('), "os.popen() Aufruf"),
        (re.compile(r'os\.spawn\w*\s*\('), "os.spawn*() Aufruf"),
        (re.compile(r'os\.exec\w*\s*\('), "os.exec*() Aufruf"),
        (re.compile(r'subprocess\.\w+\s*\('), "subprocess Aufruf"),

        # Netzwerkzugriff
        (re.compile(r'socket\.\w+\s*\('), "socket Zugriff"),
        (re.compile(r'urllib\.\w+'), "urllib Zugriff"),
        (re.compile(r'requests\.\w+\s*\('), "requests Aufruf"),

        # Dateisystem
        (re.compile(r'open\s*\([^)]*["\'][waxb]'), "Dateischreibzugriff"),
        (re.compile(r'shutil\.\w+\s*\('), "shutil Aufruf"),

        # Reflection-Angriffe
        (re.compile(r'\.__class__'), "__class__ Zugriff"),
        (re.compile(r'\.__bases__'), "__bases__ Zugriff"),
        (re.compile(r'\.__subclasses__\s*\('), "__subclasses__() Aufruf"),
        (re.compile(r'\.__mro__'), "__mro__ Zugriff"),
        (re.compile(r'\.__globals__'), "__globals__ Zugriff"),
        (re.compile(r'\.__code__'), "__code__ Zugriff"),
        (re.compile(r'\.__builtins__'), "__builtins__ Zugriff"),

        # Pickle-Angriffe
        (re.compile(r'pickle\.loads?\s*\('), "pickle Deserialisierung"),
        (re.compile(r'dill\.loads?\s*\('), "dill Deserialisierung"),

        # Environment-Manipulation
        (re.compile(r'os\.environ'), "os.environ Zugriff"),
        (re.compile(r'os\.getenv\s*\('), "os.getenv() Aufruf"),

        # Base64-verschleierte Befehle
        (re.compile(r'base64\.(b64)?decode\s*\('), "base64 Dekodierung"),
    ]

    # Erlaubte Module (Whitelist)
    ALLOWED_MODULES = frozenset({
        "math", "statistics", "decimal", "fractions",
        "collections", "heapq", "bisect", "array",
        "string", "re", "difflib", "textwrap", "unicodedata",
        "datetime", "calendar", "time",
        "json", "csv",
        "itertools", "functools", "operator",
        "typing", "dataclasses", "enum", "copy",
        "random",
    })

    def check_code_safety(self, code: str) -> tuple[bool, list[str]]:
        """
        Prueft Code auf Sicherheitsverletzungen.

        Args:
            code: Der zu pruefende Python-Code

        Returns:
            Tuple aus (ist_sicher: bool, liste_der_verletzungen: list[str])
        """
        violations = []

        # Pruefe auf verbotene Imports
        import_pattern = re.compile(r'(?:^|\n)\s*(?:import|from)\s+(\w+)')
        for match in import_pattern.finditer(code):
            module = match.group(1)
            if module in self.BLOCKED_MODULES:
                violations.append(f"Verbotener Import: {module}")
            elif module not in self.ALLOWED_MODULES:
                violations.append(f"Nicht erlaubter Import: {module}")

        # Pruefe auf verbotene Muster
        for pattern, description in self.BLOCKED_PATTERNS:
            if pattern.search(code):
                violations.append(f"Verbotenes Muster: {description}")

        # Pruefe auf verbotene Builtins als Funktionsaufrufe
        for builtin in self.BLOCKED_BUILTINS:
            builtin_pattern = re.compile(rf'\b{builtin}\s*\(')
            if builtin_pattern.search(code):
                violations.append(f"Verbotener Builtin-Aufruf: {builtin}()")

        return len(violations) == 0, violations

    def sanitize_output(self, output: str, max_length: int = 10000) -> str:
        """
        Bereinigt REPL-Ausgabe von sensiblen Informationen.

        Args:
            output: Die Ausgabe der REPL
            max_length: Maximale Laenge der Ausgabe

        Returns:
            Bereinigte Ausgabe
        """
        # Kuerze zu lange Ausgaben
        if len(output) > max_length:
            output = output[:max_length] + "\n... [Ausgabe gekuerzt]"

        # Entferne potentiell sensitive Informationen
        sensitive_patterns = [
            (re.compile(r'/home/\w+'), '/home/***'),
            (re.compile(r'/root'), '/***'),
            (re.compile(r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+'), 'password=***'),
            (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[^"\'\s]+', re.I), 'api_key=***'),
            (re.compile(r'secret["\']?\s*[:=]\s*["\']?[^"\'\s]+', re.I), 'secret=***'),
            (re.compile(r'token["\']?\s*[:=]\s*["\']?[^"\'\s]+', re.I), 'token=***'),
        ]

        for pattern, replacement in sensitive_patterns:
            output = pattern.sub(replacement, output)

        return output


class SecureLocalREPL(SecureREPLMixin, LocalREPL):
    """
    Sichere lokale REPL-Umgebung.

    Erweitert LocalREPL um Sicherheitspruefungen vor der Ausfuehrung.

    Beispiel:
        repl = SecureLocalREPL(lm_handler_address=("127.0.0.1", 8080))
        result = repl.execute_code("print('Hello')")  # OK
        result = repl.execute_code("import os")  # Wird blockiert
    """

    def __init__(
        self,
        strict_mode: bool = True,
        **kwargs,
    ):
        """
        Initialisiert die sichere REPL.

        Args:
            strict_mode: Wenn True, wird bei Sicherheitsverletzungen eine
                        Exception ausgeloest. Wenn False, wird nur gewarnt.
            **kwargs: Argumente fuer LocalREPL
        """
        super().__init__(**kwargs)
        self.strict_mode = strict_mode

    def execute_code(self, code: str) -> REPLResult:
        """
        Fuehrt Code nach Sicherheitspruefung aus.

        Args:
            code: Der auszufuehrende Python-Code

        Returns:
            REPLResult mit dem Ergebnis der Ausfuehrung

        Raises:
            SecurityViolationError: Wenn der Code Sicherheitsregeln verletzt
                                   (nur bei strict_mode=True)
        """
        # Sicherheitspruefung
        is_safe, violations = self.check_code_safety(code)

        if not is_safe:
            if self.strict_mode:
                raise SecurityViolationError(violations)
            else:
                # Im nicht-strikten Modus: Fehler als stderr zurueckgeben
                return REPLResult(
                    stdout="",
                    stderr="SICHERHEITSWARNUNG:\n" + "\n".join(
                        f"  - {v}" for v in violations
                    ),
                    locals={},
                    execution_time=0.0,
                    rlm_calls=[],
                )

        # Code ausfuehren
        result = super().execute_code(code)

        # Ausgabe bereinigen
        result = REPLResult(
            stdout=self.sanitize_output(result.stdout),
            stderr=self.sanitize_output(result.stderr),
            locals=result.locals,
            execution_time=result.execution_time,
            rlm_calls=result.rlm_calls,
        )

        return result


def create_secure_repl(
    environment_type: str = "local",
    strict_mode: bool = True,
    **kwargs,
) -> SecureLocalREPL:
    """
    Factory-Funktion fuer sichere REPL-Umgebungen.

    Args:
        environment_type: Typ der REPL ("local" oder "docker")
        strict_mode: Strikter Modus fuer Sicherheitspruefungen
        **kwargs: Zusaetzliche Argumente fuer die REPL

    Returns:
        Konfigurierte sichere REPL-Instanz
    """
    if environment_type == "local":
        return SecureLocalREPL(strict_mode=strict_mode, **kwargs)
    elif environment_type == "docker":
        # Docker-REPL ist bereits isoliert, aber wir fuegen
        # dennoch die Sicherheitspruefungen hinzu
        # (Dies waere eine Erweiterung von DockerREPL)
        raise NotImplementedError(
            "SecureDockerREPL noch nicht implementiert. "
            "Verwende environment='docker' mit der Standard-RLM-Klasse."
        )
    else:
        raise ValueError(f"Unbekannter environment_type: {environment_type}")
