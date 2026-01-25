"""
Sicherheitskonfiguration fuer RLM REPL

WICHTIG: Diese Datei definiert Befehle und Muster, die NIEMALS
in der REPL-Umgebung ausgefuehrt werden duerfen.

Die Sicherheitsrichtlinien schuetzen vor:
- Systemmanipulation
- Datenverlust
- Netzwerkangriffen
- Privilege Escalation
- Container-Ausbruch
"""

import re
from dataclasses import dataclass, field
from typing import Pattern


@dataclass
class SecurityConfig:
    """
    Sicherheitskonfiguration fuer die REPL-Umgebung.

    Diese Klasse definiert alle verbotenen Befehle, Module und Muster,
    die bei der Code-Ausfuehrung blockiert werden muessen.
    """

    # =========================================================================
    # VERBOTENE PYTHON-MODULE
    # =========================================================================
    # Diese Module duerfen NIEMALS importiert werden

    blocked_modules: frozenset = field(default_factory=lambda: frozenset({
        # Systemzugriff
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",

        # Netzwerk
        "socket",
        "http",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "telnetlib",
        "ssl",

        # Code-Ausfuehrung
        "code",
        "codeop",
        "compile",
        "ast",
        "dis",
        "inspect",
        "importlib",
        "pkgutil",
        "modulefinder",

        # Prozesse und Threads
        "multiprocessing",
        "threading",
        "concurrent",
        "asyncio",
        "signal",

        # Dateisystem
        "io",
        "tempfile",
        "glob",
        "fnmatch",
        "fileinput",
        "stat",
        "filecmp",

        # Pickle und Serialisierung (Code-Injection-Risiko)
        "pickle",
        "cPickle",
        "shelve",
        "marshal",
        "dill",

        # Gefaehrliche Stdlib-Module
        "ctypes",
        "pty",
        "tty",
        "termios",
        "resource",
        "gc",
        "traceback",
        "linecache",

        # Kryptographie (kann fuer Angriffe missbraucht werden)
        "hashlib",
        "hmac",
        "secrets",
        "cryptography",

        # Weitere gefaehrliche Module
        "builtins",
        "__builtin__",
        "runpy",
        "pdb",
        "profile",
        "cProfile",
        "timeit",
    }))

    # =========================================================================
    # VERBOTENE PYTHON-FUNKTIONEN UND BUILTINS
    # =========================================================================
    # Diese Funktionen/Builtins sind STRIKT VERBOTEN

    blocked_builtins: frozenset = field(default_factory=lambda: frozenset({
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "isinstance",  # kann fuer type confusion verwendet werden
        "issubclass",
        "super",
        "object",
        "memoryview",
        "bytearray",
        "breakpoint",
        "help",  # kann interaktive Shells starten
        "license",
        "credits",
        "copyright",
        "exit",
        "quit",
    }))

    # =========================================================================
    # VERBOTENE SHELL-BEFEHLE (fuer Docker-REPL)
    # =========================================================================
    # Diese Befehle duerfen NIEMALS in Shell-Escapes ausgefuehrt werden

    blocked_shell_commands: frozenset = field(default_factory=lambda: frozenset({
        # Systemzerstoerung
        "rm",
        "rmdir",
        "dd",
        "mkfs",
        "fdisk",
        "parted",
        "wipefs",
        "shred",

        # Netzwerk-Angriffe
        "nc",
        "netcat",
        "ncat",
        "curl",
        "wget",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "ftp",
        "telnet",
        "nmap",
        "ping",
        "traceroute",
        "dig",
        "nslookup",
        "host",

        # Privilege Escalation
        "sudo",
        "su",
        "doas",
        "pkexec",
        "chmod",
        "chown",
        "chgrp",
        "chattr",
        "setcap",
        "getcap",

        # Container-Ausbruch
        "docker",
        "podman",
        "kubectl",
        "crictl",
        "containerd",
        "runc",
        "mount",
        "umount",
        "chroot",
        "pivot_root",
        "unshare",
        "nsenter",

        # Prozessmanipulation
        "kill",
        "killall",
        "pkill",
        "renice",
        "nohup",
        "disown",
        "screen",
        "tmux",
        "at",
        "batch",
        "cron",
        "crontab",

        # Paketmanager (koennen Malware installieren)
        "apt",
        "apt-get",
        "dpkg",
        "yum",
        "dnf",
        "rpm",
        "pacman",
        "pip",
        "pip3",
        "easy_install",
        "npm",
        "yarn",
        "gem",

        # Compiler und Interpreter
        "python",
        "python3",
        "python2",
        "perl",
        "ruby",
        "node",
        "php",
        "bash",
        "sh",
        "zsh",
        "fish",
        "dash",
        "csh",
        "tcsh",
        "ksh",

        # Weitere gefaehrliche Befehle
        "make",
        "gcc",
        "g++",
        "clang",
        "ld",
        "as",
        "gdb",
        "strace",
        "ltrace",
        "env",
        "export",
        "source",
        "alias",
        "history",
        "xargs",
        "find",
        "locate",
        "updatedb",
    }))

    # =========================================================================
    # VERBOTENE MUSTER (Regex)
    # =========================================================================
    # Regex-Muster fuer gefaehrliche Code-Konstrukte

    @property
    def blocked_patterns(self) -> list[tuple[Pattern, str]]:
        """Gibt Liste von (Pattern, Beschreibung) zurueck."""
        return [
            # Shell-Escapes in Python
            (re.compile(r'os\.system\s*\('), "os.system() Aufruf"),
            (re.compile(r'os\.popen\s*\('), "os.popen() Aufruf"),
            (re.compile(r'os\.spawn\w*\s*\('), "os.spawn*() Aufruf"),
            (re.compile(r'os\.exec\w*\s*\('), "os.exec*() Aufruf"),
            (re.compile(r'subprocess\.\w+\s*\('), "subprocess Aufruf"),

            # Netzwerkzugriff
            (re.compile(r'socket\.\w+\s*\('), "socket Zugriff"),
            (re.compile(r'urllib\.\w+'), "urllib Zugriff"),
            (re.compile(r'requests\.\w+\s*\('), "requests Aufruf"),
            (re.compile(r'http\.client'), "http.client Zugriff"),

            # Dateisystemmanipulation
            (re.compile(r'open\s*\([^)]*["\'][wax]'), "Dateischreibzugriff"),
            (re.compile(r'shutil\.\w+\s*\('), "shutil Aufruf"),
            (re.compile(r'pathlib\.Path.*\.(write|unlink|rmdir|mkdir)'), "pathlib Schreibzugriff"),

            # Code-Injection
            (re.compile(r'eval\s*\('), "eval() Aufruf"),
            (re.compile(r'exec\s*\('), "exec() Aufruf"),
            (re.compile(r'compile\s*\('), "compile() Aufruf"),
            (re.compile(r'__import__\s*\('), "__import__() Aufruf"),

            # Reflection-Angriffe
            (re.compile(r'getattr\s*\([^)]*,\s*["\']__'), "getattr mit dunder"),
            (re.compile(r'\.__class__'), "__class__ Zugriff"),
            (re.compile(r'\.__bases__'), "__bases__ Zugriff"),
            (re.compile(r'\.__subclasses__\s*\('), "__subclasses__() Aufruf"),
            (re.compile(r'\.__mro__'), "__mro__ Zugriff"),
            (re.compile(r'\.__globals__'), "__globals__ Zugriff"),
            (re.compile(r'\.__code__'), "__code__ Zugriff"),
            (re.compile(r'\.__builtins__'), "__builtins__ Zugriff"),

            # Pickle-Angriffe
            (re.compile(r'pickle\.loads?\s*\('), "pickle Deserialisierung"),
            (re.compile(r'cPickle\.loads?\s*\('), "cPickle Deserialisierung"),
            (re.compile(r'dill\.loads?\s*\('), "dill Deserialisierung"),
            (re.compile(r'marshal\.loads?\s*\('), "marshal Deserialisierung"),

            # Environment-Manipulation
            (re.compile(r'os\.environ'), "os.environ Zugriff"),
            (re.compile(r'os\.getenv\s*\('), "os.getenv() Aufruf"),
            (re.compile(r'os\.putenv\s*\('), "os.putenv() Aufruf"),

            # Prozess-Manipulation
            (re.compile(r'os\.kill\s*\('), "os.kill() Aufruf"),
            (re.compile(r'signal\.\w+'), "signal Modul Zugriff"),
            (re.compile(r'multiprocessing\.\w+'), "multiprocessing Zugriff"),

            # Base64-verschleierte Befehle
            (re.compile(r'base64\.(b64)?decode\s*\('), "base64 Dekodierung"),
            (re.compile(r'codecs\.decode\s*\('), "codecs Dekodierung"),

            # Gefaehrliche String-Formatierung
            (re.compile(r'format\s*\([^)]*\{[^}]*\.__'), "format mit dunder"),
            (re.compile(r'f["\'][^"\']*\{[^}]*\.__'), "f-string mit dunder"),
        ]

    # =========================================================================
    # ERLAUBTE MODULE (Whitelist)
    # =========================================================================
    # NUR diese Module duerfen importiert werden

    allowed_modules: frozenset = field(default_factory=lambda: frozenset({
        # Mathematik
        "math",
        "statistics",
        "decimal",
        "fractions",

        # Datenstrukturen
        "collections",
        "heapq",
        "bisect",
        "array",

        # Strings
        "string",
        "re",
        "difflib",
        "textwrap",
        "unicodedata",

        # Zeit (nur lesend)
        "datetime",
        "calendar",
        "time",  # nur time.time(), time.sleep()

        # Datenverarbeitung
        "json",
        "csv",

        # Itertools
        "itertools",
        "functools",
        "operator",

        # Typen
        "typing",
        "dataclasses",
        "enum",
        "copy",

        # Zufallszahlen
        "random",
    }))


# Standard-Instanz
DEFAULT_SECURITY = SecurityConfig()


def check_code_safety(code: str, config: SecurityConfig = None) -> tuple[bool, list[str]]:
    """
    Prueft Code auf Sicherheitsverletzungen.

    Args:
        code: Der zu pruefende Python-Code
        config: Optionale Sicherheitskonfiguration

    Returns:
        Tuple aus (ist_sicher: bool, liste_der_verletzungen: list[str])
    """
    if config is None:
        config = DEFAULT_SECURITY

    violations = []

    # Pruefe auf verbotene Imports
    import_pattern = re.compile(r'(?:^|\n)\s*(?:import|from)\s+(\w+)')
    for match in import_pattern.finditer(code):
        module = match.group(1)
        if module in config.blocked_modules:
            violations.append(f"Verbotener Import: {module}")
        elif module not in config.allowed_modules:
            violations.append(f"Nicht erlaubter Import: {module}")

    # Pruefe auf verbotene Muster
    for pattern, description in config.blocked_patterns:
        if pattern.search(code):
            violations.append(f"Verbotenes Muster: {description}")

    # Pruefe auf verbotene Builtins als Funktionsaufrufe
    for builtin in config.blocked_builtins:
        builtin_pattern = re.compile(rf'\b{builtin}\s*\(')
        if builtin_pattern.search(code):
            violations.append(f"Verbotener Builtin-Aufruf: {builtin}()")

    is_safe = len(violations) == 0
    return is_safe, violations


def sanitize_output(output: str, max_length: int = 10000) -> str:
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
        (re.compile(r'password["\']?\s*[:=]\s*["\']?[^"\'\\s]+'), 'password=***'),
        (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[^"\'\\s]+', re.I), 'api_key=***'),
        (re.compile(r'secret["\']?\s*[:=]\s*["\']?[^"\'\\s]+', re.I), 'secret=***'),
        (re.compile(r'token["\']?\s*[:=]\s*["\']?[^"\'\\s]+', re.I), 'token=***'),
    ]

    for pattern, replacement in sensitive_patterns:
        output = pattern.sub(replacement, output)

    return output
