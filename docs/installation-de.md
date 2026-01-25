# RLM Installationsanleitung (Deutsch)

Detaillierte Anleitung zur lokalen Installation und Inbetriebnahme des RLM
mit Docker-REPL und einem lokalen LLM-Server.

## Inhaltsverzeichnis

1. [Ueberblick](#1-ueberblick)
2. [Systemvoraussetzungen](#2-systemvoraussetzungen)
3. [Installation](#3-installation)
4. [LLM-Server einrichten](#4-llm-server-einrichten)
5. [Docker REPL Setup](#5-docker-repl-setup)
6. [RLM verwenden](#6-rlm-verwenden)
7. [Sicherheitsrichtlinien](#7-sicherheitsrichtlinien)
8. [Fehlerbehebung](#8-fehlerbehebung)

---

## 1. Ueberblick

### Was ist RLM?

**RLM (Recursive Language Models)** ist ein Inferenz-Paradigma, das
Sprachmodellen ermoeglicht, nahezu unbegrenzte Kontextlaengen zu verarbeiten.

Statt eines einfachen LLM-Aufrufs:
```python
llm.completion(prompt, model)
```

Verwendet RLM eine REPL-Umgebung:
```python
rlm.completion(prompt, model)
```

Das LM kann dabei:
- Kontext als Variable in der REPL speichern
- Den Kontext programmatisch inspizieren und transformieren
- Sub-LM-Aufrufe aus der REPL heraus starten

### Unterstuetzte Umgebungen

| Umgebung | Beschreibung |
|----------|--------------|
| `local`  | Python REPL im gleichen Prozess |
| `docker` | Isolierter Docker-Container (empfohlen) |
| `modal`  | Modal Cloud Sandbox |
| `prime`  | Prime Intellect Sandbox |
| `daytona`| Daytona Sandbox |

---

## 2. Systemvoraussetzungen

### Hardware

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| RAM | 4 GB | 16 GB+ |
| Festplatte | 5 GB | 50 GB+ |
| CPU | 4 Kerne | 8+ Kerne |

### Software

```bash
# Python 3.11+ pruefen
python3 --version

# Git pruefen
git --version

# Docker pruefen (fuer Docker-REPL)
docker --version
docker compose version
```

---

## 3. Installation

### Schritt 1: Repository klonen

```bash
git clone https://github.com/alexzhang/rlm.git
cd rlm
```

### Schritt 2: Virtuelle Umgebung erstellen

**Option A: Mit uv (empfohlen)**
```bash
# uv installieren
curl -LsSf https://astral.sh/uv/install.sh | sh

# Virtuelle Umgebung erstellen
uv venv --python 3.11
source .venv/bin/activate

# Abhaengigkeiten installieren
uv sync
```

**Option B: Mit pip**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Schritt 3: Installation pruefen

```bash
# Import testen
python -c "from rlm import RLM; print('OK')"

# Tests ausfuehren
pytest tests/ -v
```

---

## 4. LLM-Server einrichten

### Unsere Konfiguration

| Parameter | Wert |
|-----------|------|
| Base URL | `http://0.0.0.0:5567/v1` |
| API Key | `dummy` |
| Model | `GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf` |
| Max Context | 4096 Tokens |
| Max Output | 1024 Tokens |
| RAG Limit | 1000 Zeichen |
| Error Log Limit | 500 Zeichen |

### llama.cpp Server starten

```bash
./llama-server \
    --model ./models/GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf \
    --host 0.0.0.0 \
    --port 5567 \
    --ctx-size 4096 \
    --n-predict 1024 \
    --threads 8
```

### Server testen

```bash
curl http://0.0.0.0:5567/v1/models
```

### Umgebungsvariablen (.env)

Kopiere `.env.example` zu `.env`:

```bash
cp .env.example .env
```

---

## 5. Docker REPL Setup

### Warum Docker-REPL?

- **Isolation**: Code laeuft in separatem Container
- **Sicherheit**: Kein Zugriff auf Host-System
- **Ressourcenlimits**: CPU, Memory begrenzt

### Docker-Image bauen

```bash
docker build -t rlm-sandbox -f Dockerfile.sandbox .
```

### Container testen

```bash
docker run --rm rlm-sandbox python -c "print('Hello from sandbox!')"
```

### Mit docker-compose

```bash
# Starten
docker compose up -d rlm-sandbox

# Logs anzeigen
docker compose logs -f rlm-sandbox

# Stoppen
docker compose down
```

---

## 6. RLM verwenden

### Einfaches Beispiel (Local REPL)

```python
from rlm import RLM

# RLM initialisieren mit unserem lokalen LLM
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "base_url": "http://0.0.0.0:5567/v1",
        "api_key": "dummy",
        "model_name": "GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf",
    },
    environment="local",
    max_iterations=10,
    verbose=True,
)

# Kontext definieren
context = """
Verkaufsdaten 2024:
- Januar: 50.000 EUR
- Februar: 45.000 EUR
- Maerz: 62.000 EUR
"""

# Completion ausfuehren
result = rlm.completion(
    prompt=context,
    root_prompt="Berechne den Gesamtumsatz.",
)

print(result.response)
```

### Mit Docker-REPL (empfohlen)

```python
from rlm import RLM

# RLM mit Docker-REPL
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "base_url": "http://0.0.0.0:5567/v1",
        "api_key": "dummy",
        "model_name": "GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf",
    },
    environment="docker",  # <-- Docker-REPL
    environment_kwargs={
        "image": "rlm-sandbox",
    },
    max_iterations=10,
    verbose=True,
)

result = rlm.completion(
    prompt="Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21",
    root_prompt="Berechne die 15. Fibonacci-Zahl.",
)

print(result.response)
```

### Mit Logging

```python
from rlm import RLM
from rlm.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "base_url": "http://0.0.0.0:5567/v1",
        "api_key": "dummy",
        "model_name": "GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf",
    },
    environment="docker",
    logger=logger,
    verbose=True,
)

# Logs in ./logs/*.jsonl
# Visualisieren mit: cd visualizer && npm run dev
```

### Vorhandene Beispiele ausfuehren

```bash
# Docker REPL Beispiel
python examples/docker_repl_example.py

# LM in REPL Beispiel
python examples/lm_in_repl.py
```

---

## 7. Sicherheitsrichtlinien

### VERBOTENE Befehle in der REPL

Die folgenden Befehle duerfen **NIEMALS** in der REPL ausgefuehrt werden:

#### Verbotene Python-Module

```
VERBOTEN - Systemzugriff:
  os, sys, subprocess, shutil, pathlib

VERBOTEN - Netzwerk:
  socket, requests, urllib, http, httpx, aiohttp,
  ftplib, smtplib, telnetlib, ssl

VERBOTEN - Code-Ausfuehrung:
  code, codeop, ast, dis, inspect, importlib,
  pkgutil, modulefinder, compile

VERBOTEN - Prozesse/Threads:
  multiprocessing, threading, concurrent, asyncio, signal

VERBOTEN - Dateisystem:
  io, tempfile, glob, fnmatch, fileinput, stat

VERBOTEN - Serialisierung (Code-Injection-Risiko):
  pickle, cPickle, shelve, marshal, dill

VERBOTEN - Gefaehrliche Module:
  ctypes, pty, tty, termios, resource, gc,
  traceback, linecache, builtins, runpy, pdb
```

#### Verbotene Python-Builtins

```
VERBOTEN:
  eval()        - Beliebigen Code ausfuehren
  exec()        - Beliebigen Code ausfuehren
  compile()     - Code kompilieren
  open()        - Dateizugriff
  __import__()  - Dynamische Imports
  input()       - Benutzereingabe
  globals()     - Globale Variablen
  locals()      - Lokale Variablen
  getattr()     - Attributzugriff
  setattr()     - Attribute setzen
  delattr()     - Attribute loeschen
  breakpoint()  - Debugger
  exit()        - Prozess beenden
  quit()        - Prozess beenden
```

#### Verbotene Shell-Befehle (Docker-REPL)

```
VERBOTEN - Systemzerstoerung:
  rm, rmdir, dd, mkfs, fdisk, parted, wipefs, shred

VERBOTEN - Netzwerk:
  nc, netcat, ncat, curl, wget, ssh, scp, sftp,
  rsync, ftp, telnet, nmap, ping, traceroute

VERBOTEN - Privilege Escalation:
  sudo, su, doas, pkexec, chmod, chown, chgrp,
  chattr, setcap, getcap

VERBOTEN - Container-Ausbruch:
  docker, podman, kubectl, crictl, containerd, runc,
  mount, umount, chroot, pivot_root, unshare, nsenter

VERBOTEN - Prozessmanipulation:
  kill, killall, pkill, renice, nohup, screen,
  tmux, at, batch, cron, crontab

VERBOTEN - Paketmanager:
  apt, apt-get, dpkg, yum, dnf, rpm, pacman,
  pip, pip3, npm, yarn, gem

VERBOTEN - Interpreter:
  python, python3, perl, ruby, node, php,
  bash, sh, zsh, fish
```

#### Verbotene Code-Muster

```python
# VERBOTEN - Shell-Escapes
os.system("...")
os.popen("...")
os.spawn*("...")
os.exec*("...")
subprocess.run(...)
subprocess.call(...)
subprocess.Popen(...)

# VERBOTEN - Reflection-Angriffe
obj.__class__
obj.__bases__
obj.__subclasses__()
obj.__mro__
obj.__globals__
obj.__code__
obj.__builtins__

# VERBOTEN - Deserialisierung
pickle.load(...)
pickle.loads(...)
dill.load(...)
marshal.loads(...)

# VERBOTEN - Environment-Zugriff
os.environ
os.getenv(...)
os.putenv(...)

# VERBOTEN - Verschleierte Befehle
base64.b64decode(...)
codecs.decode(...)
```

### Erlaubte Module (Whitelist)

```
ERLAUBT - Mathematik:
  math, statistics, decimal, fractions

ERLAUBT - Datenstrukturen:
  collections, heapq, bisect, array

ERLAUBT - Strings:
  string, re, difflib, textwrap, unicodedata

ERLAUBT - Zeit (nur lesend):
  datetime, calendar, time

ERLAUBT - Datenverarbeitung:
  json, csv

ERLAUBT - Funktionale Programmierung:
  itertools, functools, operator

ERLAUBT - Typen:
  typing, dataclasses, enum, copy

ERLAUBT - Zufallszahlen:
  random
```

---

## 8. Fehlerbehebung

### LLM-Server nicht erreichbar

```bash
# Server-Status pruefen
curl http://0.0.0.0:5567/v1/models

# Port pruefen
netstat -tlnp | grep 5567

# Server neu starten
./llama-server --model ... --host 0.0.0.0 --port 5567
```

### Docker-Fehler

```bash
# Docker-Dienst starten
sudo systemctl start docker

# Benutzer zur Docker-Gruppe
sudo usermod -aG docker $USER
newgrp docker

# Image bauen
docker build -t rlm-sandbox -f Dockerfile.sandbox .
```

### Python Import-Fehler

```bash
# Virtuelle Umgebung aktivieren
source .venv/bin/activate

# Paket installieren
pip install -e .
```

### Token-Limit ueberschritten

```python
# Kontext kuerzen
MAX_CONTEXT = 3000  # Platz fuer Output lassen
if len(context) > MAX_CONTEXT:
    context = context[:MAX_CONTEXT] + "..."
```

---

## Weitere Ressourcen

- **Englische Doku**: `docs/getting-started.md`
- **API-Referenz**: `docs/api/rlm.md`
- **Entwickler-Guide**: `AGENTS.md`
- **Beispiele**: `examples/`
- **Visualizer**: `visualizer/`

---

*Letzte Aktualisierung: Januar 2025*
