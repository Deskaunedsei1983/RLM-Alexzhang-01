# RLM Installationsanleitung (Deutsch)

## Inhaltsverzeichnis

1. [Ueberblick](#1-ueberblick)
2. [Systemvoraussetzungen](#2-systemvoraussetzungen)
3. [Installation](#3-installation)
4. [LLM-Server Konfiguration](#4-llm-server-konfiguration)
5. [Docker REPL Setup](#5-docker-repl-setup)
6. [Sicherheitsrichtlinien](#6-sicherheitsrichtlinien)
7. [Erste Schritte](#7-erste-schritte)
8. [Fehlerbehebung](#8-fehlerbehebung)

---

## 1. Ueberblick

### Was ist RLM?

**RLM (Recursive Language Models)** ist ein Inferenz-Paradigma, das Sprachmodellen
ermoeglicht, nahezu unbegrenzte Kontextlaengen zu verarbeiten. Anstatt eines
einfachen `llm.completion(prompt, model)` Aufrufs verwendet RLM:

```python
rlm.completion(prompt, model)
```

Dabei:
- Lagert das LM Kontext als Variable in einer REPL-Umgebung aus
- Kann das LM den Kontext programmatisch inspizieren und transformieren
- Kann das LM Sub-LM-Aufrufe aus der REPL heraus starten
- Werden verschiedene Ausfuehrungsumgebungen unterstuetzt (local, Docker, Modal)

### Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                         RLM Core                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ LM Handler  │──│   Client    │──│    LLM Server       │  │
│  │ (TCP/HTTP)  │  │ (OpenAI)    │  │ (lokal/remote)      │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘  │
│         │                                                    │
│  ┌──────┴──────────────────────────────────────────────┐    │
│  │              REPL Environment                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │    │
│  │  │  Local   │  │  Docker  │  │  Modal/Prime/    │   │    │
│  │  │  REPL    │  │  REPL    │  │  Daytona         │   │    │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Systemvoraussetzungen

### Minimale Anforderungen

| Komponente       | Minimum           | Empfohlen         |
|------------------|-------------------|-------------------|
| Python           | 3.11              | 3.11+             |
| RAM              | 4 GB              | 16 GB+            |
| Festplatte       | 5 GB              | 50 GB+            |
| Docker           | 20.10+            | 24.0+             |
| Betriebssystem   | Linux/macOS       | Linux (Ubuntu 22.04+) |

### Software-Abhaengigkeiten

```bash
# Python 3.11+
python3 --version  # Muss 3.11.x oder hoeher sein

# Git
git --version

# Docker (fuer Docker-REPL)
docker --version
docker compose version

# uv (empfohlener Paketmanager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### LLM-Server

Fuer den lokalen Betrieb wird ein OpenAI-kompatibler LLM-Server benoetigt:

- **llama.cpp** (empfohlen): https://github.com/ggerganov/llama.cpp
- **vLLM**: https://github.com/vllm-project/vllm
- **Ollama**: https://ollama.ai/
- **text-generation-webui**: https://github.com/oobabooga/text-generation-webui

---

## 3. Installation

### 3.1 Repository klonen

```bash
# Repository klonen
git clone https://github.com/alexzhang/rlm.git
cd rlm

# Oder mit SSH
git clone git@github.com:alexzhang/rlm.git
cd rlm
```

### 3.2 Mit uv (empfohlen)

```bash
# uv installieren (falls noch nicht vorhanden)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Virtuelle Umgebung erstellen und aktivieren
uv venv --python 3.11
source .venv/bin/activate

# Abhaengigkeiten installieren
uv sync

# Fuer Entwicklung
uv sync --group dev --group test
```

### 3.3 Mit pip (alternativ)

```bash
# Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate

# Abhaengigkeiten installieren
pip install -e .

# Fuer Entwicklung
pip install -e ".[dev,test]"
```

### 3.4 Installation verifizieren

```bash
# Tests ausfuehren
uv run pytest tests/ -v

# Oder mit pip
pytest tests/ -v

# Import testen
python -c "from rlm import RLM; print('RLM erfolgreich importiert!')"
```

---

## 4. LLM-Server Konfiguration

### 4.1 Unsere LLM-Konfiguration

Diese Installationsanleitung verwendet folgende LLM-Einstellungen:

| Parameter         | Wert                                        |
|-------------------|---------------------------------------------|
| Base URL          | `http://0.0.0.0:5567/v1`                    |
| API Key           | `dummy`                                     |
| Model Name        | `GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf`|
| Max Context       | 4096 Tokens                                 |
| Max Output        | 1024 Tokens                                 |
| RAG Limit         | 1000 Zeichen                                |
| Error Log Limit   | 500 Zeichen                                 |

### 4.2 llama.cpp Server starten

```bash
# Server starten
./llama-server \
    --model ./models/GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf \
    --host 0.0.0.0 \
    --port 5567 \
    --ctx-size 4096 \
    --n-predict 1024 \
    --threads 8

# Server-Verfuegbarkeit testen
curl http://0.0.0.0:5567/v1/models
```

### 4.3 Umgebungsvariablen (.env)

Erstelle eine `.env` Datei im Projektverzeichnis:

```bash
# .env Datei
# ========================

# Lokaler LLM-Server
RLM_LLM_BASE_URL=http://0.0.0.0:5567/v1
RLM_LLM_API_KEY=dummy
RLM_LLM_MODEL_NAME=GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf

# Token-Limits
RLM_MAX_CTX_WINDOW=4096
RLM_MAX_OUTPUT_TOKENS=1024

# Docker-Einstellungen
RLM_DOCKER_WORKSPACE_DIR=./.rlm_workspace

# Logging
RLM_LOG_LEVEL=INFO
```

### 4.4 Konfiguration in Python verwenden

```python
from config.local_llm_config import LLMConfig, DEFAULT_CONFIG

# Standard-Konfiguration verwenden
config = DEFAULT_CONFIG

# Oder eigene Konfiguration erstellen
config = LLMConfig(
    base_url="http://0.0.0.0:5567/v1",
    api_key="dummy",
    model_name="GLM-4.7-Flash-REAP-23B-A3B-UD-Q2_K_XL.gguf",
    max_ctx_window=4096,
    max_output_tokens=1024,
    rag_limit=1000,
    error_log_limit=500,
)

# Validierung
warnings = config.validate()
if warnings:
    print("Warnungen:", warnings)
```

---

## 5. Docker REPL Setup

### 5.1 Warum Docker REPL?

Die Docker-REPL bietet:

- **Isolation**: Code laeuft in einem separaten Container
- **Sicherheit**: Kein Zugriff auf das Host-System
- **Ressourcenlimits**: CPU, Memory, PIDs begrenzt
- **Reproduzierbarkeit**: Gleiche Umgebung ueberall

### 5.2 Docker-Image bauen

```bash
# Standard-Image bauen
docker build -t rlm-sandbox -f Dockerfile.sandbox .

# Oder mit docker compose
docker compose build rlm-sandbox
```

### 5.3 Container testen

```bash
# Container starten
docker run -it --rm rlm-sandbox python -c "print('Hello from sandbox!')"

# Interaktiver Test
docker run -it --rm rlm-sandbox /bin/bash
```

### 5.4 RLM mit Docker-REPL verwenden

```python
from rlm import RLM
from config.local_llm_config import DEFAULT_CONFIG

# RLM mit Docker-REPL initialisieren
rlm = RLM(
    backend="openai",
    backend_kwargs=DEFAULT_CONFIG.to_backend_kwargs(),
    environment="docker",  # <-- Docker-REPL aktivieren
    environment_kwargs={
        "image": "rlm-sandbox",  # Unser gebautes Image
    },
    max_iterations=10,
    verbose=True,
)

# Completion ausfuehren
result = rlm.completion(
    prompt="Deine Daten hier...",
    root_prompt="Deine Frage hier...",
)
```

### 5.5 Docker Compose Setup

```bash
# Alle Dienste starten
docker compose up -d

# Nur Sandbox starten
docker compose up -d rlm-sandbox

# Logs anzeigen
docker compose logs -f rlm-sandbox

# Dienste stoppen
docker compose down
```

---

## 6. Sicherheitsrichtlinien

### 6.1 VERBOTENE Befehle in der REPL

Die folgenden Befehle duerfen **NIEMALS** in der REPL-Umgebung ausgefuehrt werden:

#### Python-Module (VERBOTEN)

```python
# VERBOTEN - Systemzugriff
import os
import sys
import subprocess
import shutil

# VERBOTEN - Netzwerk
import socket
import requests
import urllib
import http

# VERBOTEN - Code-Ausfuehrung
import code
import compile
import ast
import importlib

# VERBOTEN - Prozesse
import multiprocessing
import threading
import signal

# VERBOTEN - Dateisystem
import io
import tempfile
import glob

# VERBOTEN - Serialisierung (Code-Injection-Risiko)
import pickle
import dill
import marshal
```

#### Python-Builtins (VERBOTEN)

```python
# VERBOTEN - Diese Builtins werden blockiert
eval()           # Beliebigen Code ausfuehren
exec()           # Beliebigen Code ausfuehren
compile()        # Code kompilieren
open()           # Dateizugriff
__import__()     # Dynamische Imports
input()          # Benutzereingabe
globals()        # Globale Variablen
locals()         # Lokale Variablen
getattr()        # Attributzugriff (kann missbraucht werden)
setattr()        # Attribute setzen
delattr()        # Attribute loeschen
breakpoint()     # Debugger
exit()           # Prozess beenden
quit()           # Prozess beenden
```

#### Shell-Befehle (Docker-REPL - VERBOTEN)

```bash
# VERBOTEN - Systemzerstoerung
rm, dd, mkfs, fdisk, shred, wipefs

# VERBOTEN - Netzwerk-Angriffe
nc, netcat, curl, wget, ssh, scp, nmap, ping

# VERBOTEN - Privilege Escalation
sudo, su, chmod, chown, setcap

# VERBOTEN - Container-Ausbruch
docker, kubectl, mount, chroot, nsenter

# VERBOTEN - Prozessmanipulation
kill, killall, pkill, cron, crontab

# VERBOTEN - Paketmanager
apt, pip, npm, gem

# VERBOTEN - Interpreter
python, bash, sh, perl, ruby, node
```

### 6.2 Sicherheits-Patterns (VERBOTEN)

```python
# VERBOTEN - Shell-Escapes
os.system("...")
os.popen("...")
subprocess.run(...)

# VERBOTEN - Reflection-Angriffe
obj.__class__.__bases__
obj.__subclasses__()
obj.__globals__
obj.__builtins__

# VERBOTEN - Pickle-Angriffe
pickle.loads(data)
dill.loads(data)

# VERBOTEN - Base64-verschleierte Befehle
base64.b64decode("...")
```

### 6.3 Sicherheitspruefung verwenden

```python
from config.security import check_code_safety, DEFAULT_SECURITY

# Code pruefen
code = """
import os
os.system("rm -rf /")
"""

is_safe, violations = check_code_safety(code)
if not is_safe:
    print("GEFAEHRLICHER CODE ERKANNT!")
    for violation in violations:
        print(f"  - {violation}")
```

### 6.4 Docker-Sicherheitsprofile

```python
from config.docker_config import DockerProfiles

# Maximale Sicherheit (Produktion)
secure_config = DockerProfiles.secure()
# - network_mode: none
# - read_only: True
# - memory: 256MB
# - no_new_privileges: True

# Entwicklungsmodus
dev_config = DockerProfiles.development()
# - network_mode: none
# - memory: 1GB
# - read_only: False

# Minimale Ressourcen
minimal_config = DockerProfiles.minimal()
# - memory: 128MB
# - cpu: 0.25
# - timeout: 15s
```

---

## 7. Erste Schritte

### 7.1 Einfaches Beispiel (Local REPL)

```python
from rlm import RLM
from config.local_llm_config import DEFAULT_CONFIG

# RLM initialisieren
rlm = RLM(
    backend="openai",
    backend_kwargs=DEFAULT_CONFIG.to_backend_kwargs(),
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
- April: 58.000 EUR
"""

# Frage stellen
result = rlm.completion(
    prompt=context,
    root_prompt="Berechne den Gesamtumsatz und Durchschnitt.",
)

print(result.response)
```

### 7.2 Docker REPL Beispiel

```python
from rlm import RLM
from config.local_llm_config import DEFAULT_CONFIG

# RLM mit Docker-REPL
rlm = RLM(
    backend="openai",
    backend_kwargs=DEFAULT_CONFIG.to_backend_kwargs(),
    environment="docker",
    environment_kwargs={
        "image": "rlm-sandbox",
    },
    max_iterations=10,
    verbose=True,
)

# Mathematische Berechnung
context = "Fibonacci-Zahlen: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34"
query = "Berechne die 15. Fibonacci-Zahl."

result = rlm.completion(prompt=context, root_prompt=query)
print(result.response)
```

### 7.3 Mit Logging

```python
from rlm import RLM
from rlm.logger import RLMLogger
from config.local_llm_config import DEFAULT_CONFIG

# Logger initialisieren
logger = RLMLogger(log_dir="./logs")

# RLM mit Logging
rlm = RLM(
    backend="openai",
    backend_kwargs=DEFAULT_CONFIG.to_backend_kwargs(),
    environment="docker",
    environment_kwargs={"image": "rlm-sandbox"},
    logger=logger,
    verbose=True,
)

result = rlm.completion(prompt="...", root_prompt="...")

# Logs finden sich in ./logs/*.jsonl
# Visualisieren mit dem Visualizer
```

### 7.4 Beispielskripte ausfuehren

```bash
# Lokales Quickstart
python examples/quickstart_local.py

# Docker REPL Beispiel
python examples/docker_repl_local.py

# Weitere Beispiele
python examples/docker_repl_example.py
python examples/lm_in_repl.py
```

---

## 8. Fehlerbehebung

### 8.1 LLM-Server nicht erreichbar

**Problem**: `Connection refused` oder `timeout`

**Loesung**:
```bash
# Server-Status pruefen
curl http://0.0.0.0:5567/v1/models

# Firewall pruefen
sudo ufw status

# Port pruefen
netstat -tlnp | grep 5567

# Server neu starten
./llama-server --model ... --host 0.0.0.0 --port 5567
```

### 8.2 Docker-Fehler

**Problem**: `Cannot connect to Docker daemon`

**Loesung**:
```bash
# Docker-Dienst starten
sudo systemctl start docker

# Benutzer zur Docker-Gruppe hinzufuegen
sudo usermod -aG docker $USER
newgrp docker

# Docker-Status pruefen
docker info
```

**Problem**: `Image not found`

**Loesung**:
```bash
# Image bauen
docker build -t rlm-sandbox -f Dockerfile.sandbox .

# Image pruefen
docker images | grep rlm-sandbox
```

### 8.3 Python-Fehler

**Problem**: `ModuleNotFoundError: No module named 'rlm'`

**Loesung**:
```bash
# Im Projektverzeichnis sein
cd /pfad/zu/rlm

# Virtuelle Umgebung aktivieren
source .venv/bin/activate

# Paket installieren
pip install -e .
```

**Problem**: `Python version not supported`

**Loesung**:
```bash
# Python-Version pruefen
python3 --version

# Python 3.11 installieren (Ubuntu)
sudo apt update
sudo apt install python3.11 python3.11-venv

# Neue venv mit Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
```

### 8.4 Speicherprobleme

**Problem**: `Out of memory`

**Loesung**:
```bash
# Kleineres Modell verwenden (Q2 statt Q4)
# Oder max_ctx_window reduzieren

# In der Konfiguration:
config = LLMConfig(
    max_ctx_window=2048,  # Reduziert
    max_output_tokens=512,
)
```

### 8.5 Token-Limit ueberschritten

**Problem**: `Context length exceeded`

**Loesung**:
```python
# RAG-Limit in der Konfiguration anpassen
config = LLMConfig(
    rag_limit=500,  # Reduziert von 1000
)

# Oder Kontext vor der Uebergabe kuerzen
if len(context) > config.rag_limit:
    context = context[:config.rag_limit] + "..."
```

---

## Weitere Ressourcen

- **Dokumentation**: `docs/getting-started.md`
- **API-Referenz**: `docs/api/rlm.md`
- **Entwickler-Guide**: `AGENTS.md`
- **Beispiele**: `examples/`
- **Tests**: `tests/`
- **Visualizer**: `visualizer/`

---

## Support

Bei Fragen oder Problemen:

1. GitHub Issues: https://github.com/alexzhang/rlm/issues
2. Dokumentation: https://rlm.readthedocs.io/
3. Paper: ArXiv 2512.24601

---

*Letzte Aktualisierung: 2025*
