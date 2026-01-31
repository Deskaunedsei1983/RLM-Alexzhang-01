"""
RLM Streamlit GUI

Eine grafische Benutzeroberflaeche fuer das RLM-System.

Starten mit:
    streamlit run gui/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Projektpfad hinzufuegen
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.rlm_backend import (
    RLMBackend,
    RLMConfig,
    EXAMPLE_CONTEXTS,
    EXAMPLE_TASKS,
)


# =============================================================================
# Streamlit Konfiguration
# =============================================================================
st.set_page_config(
    page_title="RLM - Recursive Language Model",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Session State initialisieren
# =============================================================================
if "config" not in st.session_state:
    st.session_state.config = RLMConfig.from_env()

if "backend" not in st.session_state:
    st.session_state.backend = None

if "result" not in st.session_state:
    st.session_state.result = None

if "history" not in st.session_state:
    st.session_state.history = []


# =============================================================================
# Sidebar - Konfiguration
# =============================================================================
with st.sidebar:
    st.title("⚙️ Konfiguration")

    st.subheader("LLM Server")
    base_url = st.text_input(
        "Base URL",
        value=st.session_state.config.base_url,
        help="URL zum OpenAI-kompatiblen LLM-Server"
    )
    api_key = st.text_input(
        "API Key",
        value=st.session_state.config.api_key,
        type="password",
        help="API-Schluessel (bei lokalem Server oft 'dummy')"
    )
    model_name = st.text_input(
        "Modell",
        value=st.session_state.config.model_name,
        help="Name des Modells"
    )

    st.subheader("RLM Einstellungen")
    environment = st.selectbox(
        "Umgebung",
        options=["docker", "local"],
        index=0 if st.session_state.config.environment == "docker" else 1,
        help="Docker = sicher isoliert, Local = schneller"
    )
    max_iterations = st.slider(
        "Max Iterationen",
        min_value=5,
        max_value=30,
        value=st.session_state.config.max_iterations,
        help="Maximale Anzahl der REPL-Iterationen"
    )

    st.divider()

    # Status-Checks
    st.subheader("Status")

    # Config aktualisieren
    st.session_state.config = RLMConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        environment=environment,
        max_iterations=max_iterations,
    )

    # Backend initialisieren
    if st.session_state.backend is None:
        st.session_state.backend = RLMBackend(st.session_state.config)
    else:
        st.session_state.backend.config = st.session_state.config

    # LLM-Server pruefen
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄", help="Server pruefen"):
            pass  # Trigger rerun
    with col2:
        llm_ok, llm_msg = st.session_state.backend.check_llm_server()
        if llm_ok:
            st.success(llm_msg)
        else:
            st.error(llm_msg)

    # Docker pruefen (nur wenn Docker ausgewaehlt)
    if environment == "docker":
        docker_ok, docker_msg = st.session_state.backend.check_docker()
        if docker_ok:
            st.success(docker_msg)
        else:
            st.error(docker_msg)


# =============================================================================
# Hauptbereich
# =============================================================================
st.title("🔄 RLM - Recursive Language Model")
st.markdown("""
RLM ermoeglicht es Sprachmodellen, Code in einer REPL-Umgebung auszufuehren
und sich selbst rekursiv aufzurufen.
""")

# Tabs fuer verschiedene Bereiche
tab_main, tab_logs, tab_help = st.tabs(["📝 Ausfuehrung", "📊 Logs", "❓ Hilfe"])


# =============================================================================
# Tab: Ausfuehrung
# =============================================================================
with tab_main:
    # Beispiel-Auswahl
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("**Beispiel laden:**")
    with col2:
        example = st.selectbox(
            "Beispiel",
            options=["-- Eigene Eingabe --"] + list(EXAMPLE_CONTEXTS.keys()),
            label_visibility="collapsed",
        )

    # Kontext und Aufgabe
    col_context, col_task = st.columns(2)

    with col_context:
        st.subheader("📄 Kontext / Daten")
        if example != "-- Eigene Eingabe --":
            default_context = EXAMPLE_CONTEXTS.get(example, "")
        else:
            default_context = ""

        context = st.text_area(
            "Kontext",
            value=default_context,
            height=300,
            label_visibility="collapsed",
            placeholder="Gib hier deinen Kontext ein (Daten, Code, Text...)",
        )

    with col_task:
        st.subheader("🎯 Aufgabe / Frage")
        if example != "-- Eigene Eingabe --":
            default_task = EXAMPLE_TASKS.get(example, "")
        else:
            default_task = ""

        aufgabe = st.text_area(
            "Aufgabe",
            value=default_task,
            height=300,
            label_visibility="collapsed",
            placeholder="Was soll mit dem Kontext gemacht werden?",
        )

    # Ausfuehren Button
    st.divider()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_button = st.button(
            "▶️ Ausfuehren",
            type="primary",
            use_container_width=True,
            disabled=not context or not aufgabe,
        )

    with col_info:
        if not context:
            st.warning("Bitte Kontext eingeben")
        elif not aufgabe:
            st.warning("Bitte Aufgabe eingeben")
        else:
            st.info(f"Umgebung: {environment} | Max Iterationen: {max_iterations}")

    # Ausfuehrung
    if run_button:
        with st.spinner("RLM laeuft... Dies kann einige Minuten dauern."):
            result = st.session_state.backend.run_completion(context, aufgabe)
            st.session_state.result = result

            # Zur History hinzufuegen
            st.session_state.history.append({
                "context": context[:100] + "..." if len(context) > 100 else context,
                "aufgabe": aufgabe,
                "result": result,
            })

    # Ergebnis anzeigen
    if st.session_state.result:
        result = st.session_state.result

        st.divider()
        st.subheader("📤 Ergebnis")

        if result.success:
            # Erfolg
            col_res, col_meta = st.columns([3, 1])

            with col_res:
                st.markdown("**Antwort:**")
                st.markdown(result.response)

            with col_meta:
                st.metric("Zeit", f"{result.execution_time:.1f}s")
                if result.iterations:
                    st.metric("Iterationen", result.iterations)
                if result.log_file:
                    st.caption(f"Log: {result.log_file}")

        else:
            # Fehler
            st.error(f"Fehler: {result.error}")

            st.markdown("**Checkliste:**")
            st.markdown(f"""
- [ ] LLM-Server laeuft auf `{st.session_state.config.base_url}`?
- [ ] Docker-Daemon gestartet? (bei Docker-Umgebung)
- [ ] Modell `{st.session_state.config.model_name}` geladen?
            """)


# =============================================================================
# Tab: Logs
# =============================================================================
with tab_logs:
    st.subheader("📊 Log-Dateien")

    log_files = st.session_state.backend.get_log_files()

    if not log_files:
        st.info("Keine Log-Dateien gefunden. Fuehre zuerst eine Completion aus.")
    else:
        selected_log = st.selectbox(
            "Log-Datei auswaehlen",
            options=log_files,
            format_func=lambda x: x.name,
        )

        if selected_log:
            entries = st.session_state.backend.read_log_file(str(selected_log))

            if entries:
                st.markdown(f"**{len(entries)} Eintraege**")

                for i, entry in enumerate(entries):
                    with st.expander(f"Eintrag {i+1}", expanded=i==0):
                        st.json(entry)
            else:
                st.warning("Konnte Log-Datei nicht lesen")

    st.divider()

    st.subheader("📜 Session-History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Ausfuehrung {len(st.session_state.history) - i}: {item['aufgabe'][:50]}..."):
                st.markdown(f"**Kontext:** {item['context']}")
                st.markdown(f"**Aufgabe:** {item['aufgabe']}")
                if item['result'].success:
                    st.success(item['result'].response[:500] + "..." if len(item['result'].response) > 500 else item['result'].response)
                else:
                    st.error(item['result'].error)
    else:
        st.info("Noch keine Ausfuehrungen in dieser Session")


# =============================================================================
# Tab: Hilfe
# =============================================================================
with tab_help:
    st.subheader("❓ Hilfe")

    st.markdown("""
### Was ist RLM?

**RLM (Recursive Language Models)** ist ein Inferenz-Paradigma, das
Sprachmodellen ermoeglicht, Code in einer REPL-Umgebung auszufuehren.

### Wie funktioniert es?

1. **Kontext**: Du gibst Daten/Code/Text ein
2. **Aufgabe**: Du beschreibst, was damit gemacht werden soll
3. **RLM-Loop**: Das LLM generiert Python-Code, fuehrt ihn aus, sieht das Ergebnis, generiert mehr Code...
4. **Ergebnis**: Das LLM gibt eine finale Antwort

### Umgebungen

| Umgebung | Beschreibung |
|----------|--------------|
| **docker** | Code laeuft isoliert in Docker-Container (sicher) |
| **local** | Code laeuft im gleichen Prozess (schneller) |

### Sicherheit (Docker)

In der Docker-Umgebung sind gefaehrliche Befehle blockiert:
- `os.system()`, `subprocess`, etc.
- `eval()`, `exec()`, `open()`
- Netzwerkzugriffe

### Troubleshooting

**LLM-Server nicht erreichbar?**
```bash
curl http://0.0.0.0:5567/v1/models
```

**Docker-Fehler?**
```bash
docker info
```

**Mehr Infos?**
Siehe `docs/installation-de.md`
    """)


# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption("RLM GUI v1.0 | Powered by Streamlit")
