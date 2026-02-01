"""
RLM Streamlit GUI

Eine grafische Benutzeroberflaeche fuer das RLM-System.

Starten mit:
    streamlit run gui/app.py --server.port 8052
"""

import streamlit as st
import sys
import time
from pathlib import Path

# Projektpfad hinzufuegen
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.rlm_backend import (
    RLMBackend,
    RLMConfig,
    EXAMPLE_CONTEXTS,
    EXAMPLE_TASKS,
)
from gui.workflow_engine import WorkflowConfig, WorkflowStage
from gui.documentation_workflow import (
    DocumentationWorkflow,
    WorkflowProgress,
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

if "workflow_running" not in st.session_state:
    st.session_state.workflow_running = False

if "workflow_progress" not in st.session_state:
    st.session_state.workflow_progress = []

if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None


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

    st.divider()
    st.caption("Port: 8052")


# =============================================================================
# Hauptbereich
# =============================================================================
st.title("🔄 RLM - Recursive Language Model")
st.markdown("""
RLM ermoeglicht es Sprachmodellen, Code in einer REPL-Umgebung auszufuehren
und sich selbst rekursiv aufzurufen.
""")

# Tabs fuer verschiedene Bereiche
tab_main, tab_workflow, tab_logs, tab_help = st.tabs([
    "📝 Einfache Ausfuehrung",
    "📁 Dokumentations-Workflow",
    "📊 Logs",
    "❓ Hilfe"
])


# =============================================================================
# Tab: Einfache Ausfuehrung
# =============================================================================
with tab_main:
    # Kontext-Quelle auswaehlen
    st.subheader("📄 Kontext / Daten")

    context_source = st.radio(
        "Kontext-Quelle",
        options=["✏️ Text eingeben", "📁 Datei(en) hochladen", "📋 Beispiel laden"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Kontext basierend auf Quelle
    context = ""
    uploaded_files_info = ""

    if context_source == "📋 Beispiel laden":
        example = st.selectbox(
            "Beispiel auswaehlen",
            options=list(EXAMPLE_CONTEXTS.keys()),
        )
        context = EXAMPLE_CONTEXTS.get(example, "")
        default_task = EXAMPLE_TASKS.get(example, "")

        st.text_area(
            "Vorschau",
            value=context,
            height=200,
            disabled=True,
        )

    elif context_source == "📁 Datei(en) hochladen":
        uploaded_files = st.file_uploader(
            "Dateien hochladen",
            type=["txt", "py", "js", "ts", "json", "yaml", "yml", "md", "csv", "xml", "html", "css", "java", "c", "cpp", "h", "go", "rs", "sh"],
            accept_multiple_files=True,
            help="Unterstuetzte Formate: Text, Code, Markdown, JSON, YAML, CSV, etc.",
        )

        if uploaded_files:
            # Dateien einlesen und kombinieren
            file_contents = []
            total_size = 0

            for uploaded_file in uploaded_files:
                try:
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                    file_size = len(content)
                    total_size += file_size

                    # Datei-Header hinzufuegen
                    file_contents.append(f"### Datei: {uploaded_file.name} ({file_size} Zeichen)\n```\n{content}\n```")

                    uploaded_files_info += f"- {uploaded_file.name} ({file_size} Zeichen)\n"
                except Exception as e:
                    st.warning(f"Konnte {uploaded_file.name} nicht lesen: {e}")

            if file_contents:
                context = "\n\n".join(file_contents)

                # Info anzeigen
                st.success(f"✅ {len(uploaded_files)} Datei(en) geladen ({total_size} Zeichen gesamt)")

                with st.expander("📄 Geladene Dateien anzeigen"):
                    st.markdown(uploaded_files_info)
                    st.text_area(
                        "Inhalt (Vorschau)",
                        value=context[:5000] + ("..." if len(context) > 5000 else ""),
                        height=200,
                        disabled=True,
                    )

                # Warnung bei grossem Kontext
                if total_size > 3000:
                    st.warning(f"⚠️ Grosser Kontext ({total_size} Zeichen). Bei Token-Limit wird automatisch gekuerzt.")

        default_task = "Analysiere die hochgeladenen Dateien und beschreibe deren Inhalt und Zweck."

    else:  # Text eingeben
        context = st.text_area(
            "Kontext eingeben",
            value="",
            height=250,
            placeholder="Gib hier deinen Kontext ein (Daten, Code, Text...)",
            label_visibility="collapsed",
        )
        default_task = ""

    st.divider()

    # Aufgabe
    st.subheader("🎯 Aufgabe / Frage")

    aufgabe = st.text_area(
        "Aufgabe",
        value=default_task if 'default_task' in dir() else "",
        height=100,
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
            st.warning("Bitte Kontext eingeben oder Datei hochladen")
        elif not aufgabe:
            st.warning("Bitte Aufgabe eingeben")
        else:
            ctx_len = len(context)
            st.info(f"Umgebung: {environment} | Kontext: {ctx_len} Zeichen | Max Iter: {max_iterations}")

    # Ausfuehrung
    if run_button:
        with st.spinner("RLM laeuft... Dies kann einige Minuten dauern."):
            result = st.session_state.backend.run_completion(context, aufgabe)
            st.session_state.result = result

            # Zur History hinzufuegen
            ctx_preview = uploaded_files_info if uploaded_files_info else (context[:100] + "..." if len(context) > 100 else context)
            st.session_state.history.append({
                "context": ctx_preview,
                "aufgabe": aufgabe,
                "result": result,
            })

    # Ergebnis anzeigen
    if st.session_state.result:
        result = st.session_state.result

        st.divider()
        st.subheader("📤 Ergebnis")

        if result.success:
            col_res, col_meta = st.columns([3, 1])

            with col_res:
                st.markdown("**Antwort:**")
                st.markdown(result.response)

            with col_meta:
                st.metric("Zeit", f"{result.execution_time:.1f}s")
                if result.iterations:
                    st.metric("Iterationen", result.iterations)

        else:
            st.error(f"Fehler: {result.error}")


# =============================================================================
# Tab: Dokumentations-Workflow
# =============================================================================
with tab_workflow:
    st.subheader("📁 Automatische Dokumentations-Erstellung")

    st.markdown("""
    Dieser Workflow analysiert ein Verzeichnis und erstellt automatisch
    eine detaillierte Dokumentation. Das funktioniert in mehreren Phasen:

    1. **Discover**: Dateien finden und auflisten
    2. **Categorize**: Nach Typ gruppieren
    3. **Analyze**: Jede Datei analysieren (mit Chunking fuer grosse Dateien)
    4. **Summarize**: Modul-Zusammenfassungen erstellen
    5. **Document**: Finale Dokumentation generieren
    """)

    st.divider()

    # Pfad-Eingabe
    col_path, col_browse = st.columns([4, 1])

    with col_path:
        source_path = st.text_input(
            "📂 Quellverzeichnis",
            value="/home/user/RLM-Alexzhang-01",
            help="Pfad zum zu dokumentierenden Verzeichnis",
        )

    with col_browse:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📁 Pruefen"):
            if Path(source_path).exists():
                st.success("✓ Pfad existiert")
            else:
                st.error("✗ Pfad nicht gefunden")

    # Erweiterte Optionen
    with st.expander("⚙️ Erweiterte Optionen"):
        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            output_path = st.text_input(
                "Ausgabe-Verzeichnis",
                value="./documentation",
            )
            max_file_size = st.number_input(
                "Max. Dateigroesse (KB)",
                min_value=10,
                max_value=1000,
                value=100,
            )

        with col_opt2:
            file_extensions = st.text_input(
                "Dateiendungen (kommagetrennt)",
                value=".py, .js, .ts, .md, .json, .yaml",
            )
            ignore_patterns = st.text_input(
                "Ignorieren (kommagetrennt)",
                value="__pycache__, node_modules, .git, .venv",
            )

    st.divider()

    # Workflow starten
    col_start, col_stop = st.columns([1, 1])

    with col_start:
        start_workflow = st.button(
            "🚀 Workflow starten",
            type="primary",
            use_container_width=True,
            disabled=not source_path or st.session_state.workflow_running,
        )

    with col_stop:
        stop_workflow = st.button(
            "⏹️ Stoppen",
            use_container_width=True,
            disabled=not st.session_state.workflow_running,
        )

    # Workflow ausfuehren
    if start_workflow and source_path:
        st.session_state.workflow_running = True
        st.session_state.workflow_progress = []
        st.session_state.workflow_result = None

        # Konfiguration erstellen
        extensions = [e.strip() for e in file_extensions.split(",")]
        ignores = [i.strip() for i in ignore_patterns.split(",")]

        workflow_config = WorkflowConfig(
            source_path=source_path,
            output_path=output_path,
            file_extensions=extensions,
            ignore_patterns=ignores,
            max_file_size=max_file_size * 1024,
        )

        # Progress-Container
        progress_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Workflow erstellen und ausfuehren
        workflow = DocumentationWorkflow(
            backend=st.session_state.backend,
            workflow_config=workflow_config,
        )

        try:
            for progress in workflow.run():
                # Progress anzeigen
                progress_bar.progress(progress.progress)
                status_text.markdown(f"**{progress.stage.upper()}**: {progress.message}")

                st.session_state.workflow_progress.append(progress)

                if progress.is_error:
                    st.error(progress.detail or progress.message)

                # Check fuer Stop
                if stop_workflow:
                    workflow.stop()
                    break

            st.session_state.workflow_result = workflow.state
            st.session_state.workflow_running = False

            if workflow.state.final_documentation:
                st.success("✅ Dokumentation erfolgreich erstellt!")

        except Exception as e:
            st.error(f"Workflow-Fehler: {e}")
            st.session_state.workflow_running = False

    # Ergebnis anzeigen
    if st.session_state.workflow_result:
        state = st.session_state.workflow_result

        st.divider()
        st.subheader("📄 Ergebnis")

        # Statistiken
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Dateien", len(state.files))
        with col_stat2:
            st.metric("Kategorien", len(state.categories))
        with col_stat3:
            st.metric("Zusammenfassungen", len(state.summaries))

        # Dokumentation anzeigen
        if state.final_documentation:
            with st.expander("📖 Generierte Dokumentation", expanded=True):
                st.markdown(state.final_documentation)

                # Download-Button
                st.download_button(
                    "⬇️ Als Markdown herunterladen",
                    data=state.final_documentation,
                    file_name="DOCUMENTATION.md",
                    mime="text/markdown",
                )

        # Fehler anzeigen
        if state.errors:
            with st.expander(f"⚠️ Fehler ({len(state.errors)})"):
                for error in state.errors:
                    st.warning(error)


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

### Wie funktioniert der Dokumentations-Workflow?

Das Problem: Unser LLM hat ein begrenztes Kontextfenster (4096 Tokens).
Ein grosses Projekt passt nicht auf einmal hinein.

**Loesung: Hierarchische Analyse mit Wissens-Akkumulation**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Dateien    │ --> │ Kategorien  │ --> │  Analysen   │
│  finden     │     │  bilden     │     │ (chunked)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│   Finale    │ <-- │   Modul-    │ <----------┘
│   Doku      │     │   Summaries │
└─────────────┘     └─────────────┘
```

**Strategien:**
1. **Chunking**: Grosse Dateien werden in Teile aufgeteilt
2. **Rolling Summary**: Alte Infos werden komprimiert
3. **Hierarchisch**: Erst Struktur, dann Details
4. **Akkumulierend**: Wissen wird aufgebaut

### Kontextmanagement

| Problem | Loesung |
|---------|---------|
| Datei zu gross | Chunking (2000 Zeichen/Chunk) |
| Zu viele Dateien | Kategorisierung + Sampling |
| Wissen geht verloren | Knowledge Buffer |
| Buffer zu gross | Komprimierung alter Eintraege |

### Troubleshooting

**Workflow bricht ab?**
- Pruefe LLM-Server Verbindung
- Reduziere Max. Dateigroesse
- Ignoriere mehr Verzeichnisse

**Dokumentation unvollstaendig?**
- Erhoehe Max. Iterationen
- Pruefe Token-Limits

**Mehr Infos?**
Siehe `docs/installation-de.md`
    """)


# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption("RLM GUI v2.0 | Port 8052 | Powered by Streamlit")
