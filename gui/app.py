"""
RLM Streamlit GUI

Eine grafische Benutzeroberflaeche fuer das RLM-System.

Starten mit:
    streamlit run gui/app.py --server.port 8052
"""

import streamlit as st
import sys
import time
import gc
from pathlib import Path

# WICHTIG: Rekursionslimit fuer grosse Dateimengen erhoehen
# Das RLM-Framework nutzt intern Rekursion bei vielen Iterationen
sys.setrecursionlimit(100000)

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
from gui.rlm_workflow import (
    RLMCentricWorkflow,
    RLMWorkflowProgress,
)
from gui.memory_system import MemorySystem
from gui.smart_document_processor import (
    SmartDocumentProcessor,
    ProcessingConfig,
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

if "memory_system" not in st.session_state:
    st.session_state.memory_system = MemorySystem(memory_dir="./gui/memory")

if "smart_processor" not in st.session_state:
    st.session_state.smart_processor = None

if "smart_result" not in st.session_state:
    st.session_state.smart_result = None


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
        min_value=10,
        max_value=500,
        value=min(st.session_state.config.max_iterations, 500),
        help="Maximale Anzahl der REPL-Iterationen (hoch setzen fuer grosse Projekte!)"
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
tab_main, tab_workflow, tab_memory, tab_logs, tab_help = st.tabs([
    "📝 Einfache Ausfuehrung",
    "📁 Dokumentations-Workflow",
    "🧠 Memory & Smart Processing",
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
                    # WICHTIG: getvalue() statt read() verwenden!
                    # read() funktioniert nur beim ersten Aufruf, getvalue() immer
                    content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
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

                # Smart Processing Option fuer grosse Dateien
                if total_size > 3000:
                    st.warning(f"⚠️ Grosser Kontext ({total_size} Zeichen).")
                    use_smart_processing = st.checkbox(
                        "🧠 Smart Processing aktivieren (Chunking + Memory)",
                        value=True,
                        help="Verarbeitet grosse Dateien intelligent mit Chunking und Memory-System"
                    )
                else:
                    use_smart_processing = False

        default_task = "Analysiere die hochgeladenen Dateien und beschreibe deren Inhalt und Zweck."

    elif context_source == "✏️ Text eingeben":
        # Text manuell eingeben
        context = st.text_area(
            "Kontext eingeben",
            value="",
            height=250,
            placeholder="Gib hier deinen Kontext ein (Daten, Code, Text...)",
            label_visibility="collapsed",
        )
        default_task = ""
        use_smart_processing = False
        uploaded_files = None

    # Fallback fuer Variablen
    if 'use_smart_processing' not in dir():
        use_smart_processing = False
    if 'uploaded_files' not in dir():
        uploaded_files = None
    if 'default_task' not in dir():
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
        # Pruefe ob Smart Processing verwendet werden soll
        if use_smart_processing and len(context) > 3000:
            # Smart Processing mit Chunking und Memory
            st.info("🧠 Smart Processing aktiv - Verarbeite mit Chunking und Memory...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(progress: float, message: str):
                progress_bar.progress(progress)
                status_text.text(message)

            # Smart Processor initialisieren
            processor_config = ProcessingConfig(
                max_chunk_size=2500,
                chunk_overlap=300,
                max_context_size=2000,
                progress_callback=progress_callback,
            )

            processor = SmartDocumentProcessor(
                backend=st.session_state.backend,
                memory=st.session_state.memory_system,
                config=processor_config,
            )

            # Verarbeite jede hochgeladene Datei
            if uploaded_files:
                all_results = []
                for uploaded_file in uploaded_files:
                    content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                    doc_result = processor.process_document(
                        content=content,
                        filename=uploaded_file.name,
                        aufgabe=aufgabe,
                    )
                    all_results.append(doc_result)

                # Kombiniere Ergebnisse
                combined_answer = "\n\n---\n\n".join([
                    f"## {r.filename}\n{r.answer}"
                    for r in all_results
                ])

                from gui.rlm_backend import RLMResult
                st.session_state.result = RLMResult(
                    success=True,
                    response=combined_answer,
                    execution_time=sum(r.processing_time for r in all_results),
                )
                st.session_state.smart_result = all_results
            else:
                # Einzelner Text
                doc_result = processor.process_document(
                    content=context,
                    filename="input.txt",
                    aufgabe=aufgabe,
                )
                from gui.rlm_backend import RLMResult
                st.session_state.result = RLMResult(
                    success=doc_result.success,
                    response=doc_result.answer,
                    execution_time=doc_result.processing_time,
                    error=doc_result.error,
                )
                st.session_state.smart_result = [doc_result]

            progress_bar.progress(1.0)
            status_text.text("✅ Verarbeitung abgeschlossen!")

        else:
            # Normale Verarbeitung
            with st.spinner("RLM laeuft... Dies kann einige Minuten dauern."):
                result = st.session_state.backend.run_completion(context, aufgabe)
                st.session_state.result = result
                st.session_state.smart_result = None

        # Zur History hinzufuegen
        ctx_preview = uploaded_files_info if uploaded_files_info else (context[:100] + "..." if len(context) > 100 else context)
        st.session_state.history.append({
            "context": ctx_preview,
            "aufgabe": aufgabe,
            "result": st.session_state.result,
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

                # Smart Processing Details
                if st.session_state.smart_result:
                    total_chunks = sum(r.total_chunks for r in st.session_state.smart_result)
                    st.metric("Chunks", total_chunks)

                    # Memory Stats
                    mem_stats = st.session_state.memory_system.get_stats()
                    st.metric("Memory (ST/LT)", f"{mem_stats.short_term_entries}/{mem_stats.long_term_entries}")

            # Smart Processing Details anzeigen
            if st.session_state.smart_result:
                with st.expander("🧠 Smart Processing Details"):
                    for doc_result in st.session_state.smart_result:
                        st.markdown(f"**{doc_result.filename}**")
                        st.markdown(f"- Chunks: {doc_result.total_chunks}")
                        st.markdown(f"- Zeit: {doc_result.processing_time:.1f}s")

                        if doc_result.chunk_results:
                            st.markdown("**Chunk-Zusammenfassungen:**")
                            for cr in doc_result.chunk_results[:3]:
                                st.markdown(f"  - Teil {cr.chunk_num}: {cr.summary[:100]}...")

        else:
            st.error(f"Fehler: {result.error}")


# =============================================================================
# Tab: Dokumentations-Workflow
# =============================================================================
with tab_workflow:
    st.subheader("📁 Automatische Dokumentations-Erstellung")

    # Auswahl des Workflow-Modus
    workflow_mode = st.radio(
        "Workflow-Modus",
        options=["🔄 RLM-Centric (empfohlen)", "🤖 Agentic (Python-gesteuert)"],
        horizontal=True,
        help="RLM-Centric: Das LLM steuert den Prozess selbst via REPL. Agentic: Python iteriert ueber Dateien."
    )

    if "RLM-Centric" in workflow_mode:
        st.markdown("""
        **RLM-Centric Modus** - Das LLM steuert den gesamten Prozess:

        1. Dein Projektverzeichnis wird in Docker gemountet: `/project/`
        2. **EIN** RLM-Aufruf - LLM liest Dateien selbst via REPL
        3. LLM chunked grosse Dateien **SELBST**
        4. LLM ruft sich **SELBST** rekursiv auf (`llm_query()`)
        5. LLM aggregiert und dokumentiert

        ✅ Ideal fuer grosse Projekte (100k+ Dateien)
        ✅ LLM hat Zugriff auf ALLE Dateien
        ✅ Echtes RLM-Paradigma
        """)
    else:
        st.markdown("""
        **Agentic Modus** - Python steuert den Prozess:

        1. **Discover**: Dateien finden und auflisten
        2. **Categorize**: Nach Typ gruppieren
        3. **Analyze**: Jede Datei einzeln analysieren
        4. **Summarize**: Modul-Zusammenfassungen
        5. **Document**: Finale Dokumentation

        ⚠️ Bei vielen Dateien langsamer
        ⚠️ Moegliche Stack-Probleme bei 10k+ Dateien
        """)

    st.divider()

    # Pfad-Eingabe - WICHTIG fuer RLM-Centric
    if "RLM-Centric" in workflow_mode:
        st.markdown("""
        **📂 Projektverzeichnis fuer RLM-Analyse**

        Dieses Verzeichnis wird in den Docker-Container gemountet.
        Das LLM kann dann ALLE Dateien darin lesen unter `/project/`
        """)

    col_path, col_browse = st.columns([4, 1])

    with col_path:
        source_path = st.text_input(
            "📂 Quellverzeichnis (wird in Docker gemountet)" if "RLM-Centric" in workflow_mode else "📂 Quellverzeichnis",
            value=st.session_state.get("last_source_path", "/home/user"),
            help="Absoluter Pfad zum Projektverzeichnis. Im RLM-Centric Modus wird dies unter /project/ im Container verfuegbar.",
            key="source_path_input",
        )
        # Speichere fuer naechstes Mal
        st.session_state["last_source_path"] = source_path

    with col_browse:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📁 Pruefen"):
            if Path(source_path).exists():
                file_count = sum(1 for _ in Path(source_path).rglob("*") if _.is_file())
                st.success(f"✓ {file_count} Dateien")
            else:
                st.error("✗ Pfad nicht gefunden")

    # RLM-Centric: Zeige Mount-Info
    if "RLM-Centric" in workflow_mode and Path(source_path).exists():
        st.info(f"🐳 Docker-Mount: `{source_path}` → `/project/`")

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

        # Analysetiefe-Regler
        st.markdown("**Analysetiefe** (fuer grosse Projekte reduzieren)")
        analysis_depth = st.slider(
            "Analysetiefe",
            min_value=1,
            max_value=4,
            value=2,
            help="""
            1 = Nur Dateiliste + Struktur (sehr schnell, 100k+ Dateien)
            2 = + Dateianalyse (Standard)
            3 = + Modul-Zusammenfassungen
            4 = + Vollstaendige LLM-Dokumentation
            """,
            label_visibility="collapsed",
        )

        depth_labels = {
            1: "🚀 Schnell: Nur Struktur (fuer 100k+ Dateien)",
            2: "⚖️ Standard: Struktur + Dateianalyse",
            3: "📊 Ausfuehrlich: + Modul-Summaries",
            4: "📖 Vollstaendig: + LLM-Dokumentation",
        }
        st.info(depth_labels.get(analysis_depth, ""))

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

        # Workflow basierend auf Modus erstellen
        use_rlm_centric = "RLM-Centric" in workflow_mode

        if use_rlm_centric:
            # RLM-Centric: Das LLM steuert den Prozess
            st.info("🔄 RLM-Centric Modus: Das LLM steuert die Analyse via REPL...")
            workflow = RLMCentricWorkflow(
                backend=st.session_state.backend,
                config=workflow_config,
                memory=st.session_state.memory_system,
            )
        else:
            # Agentic: Python steuert den Prozess
            workflow = DocumentationWorkflow(
                backend=st.session_state.backend,
                workflow_config=workflow_config,
                memory_system=st.session_state.memory_system,
                analysis_depth=analysis_depth,
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
# Tab: Memory & Smart Processing
# =============================================================================
with tab_memory:
    st.subheader("🧠 Memory System & Smart Document Processing")

    st.markdown("""
    Das Memory-System speichert Wissen aus der Dokumentverarbeitung:

    - **Short-Term Memory**: Session-basiert, wird bei Neustart geloescht
    - **Long-Term Memory**: Persistent, bleibt erhalten

    Bei der Smart-Verarbeitung grosser Dokumente werden Zwischenergebnisse
    im Memory gespeichert und fuer spaetere Anfragen wiederverwendet.
    """)

    st.divider()

    # Memory Statistiken
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

    mem_stats = st.session_state.memory_system.get_stats()

    with col_stats1:
        st.metric("Short-Term Eintraege", mem_stats.short_term_entries)
    with col_stats2:
        st.metric("Long-Term Eintraege", mem_stats.long_term_entries)
    with col_stats3:
        st.metric("Gesamt Zeichen", f"{mem_stats.total_chars:,}")
    with col_stats4:
        if mem_stats.newest_entry:
            st.metric("Letzter Eintrag", mem_stats.newest_entry[:16])
        else:
            st.metric("Letzter Eintrag", "-")

    st.divider()

    # Memory Inhalt anzeigen
    col_mem1, col_mem2 = st.columns(2)

    with col_mem1:
        st.markdown("### 📋 Short-Term Memory")
        short_term = st.session_state.memory_system.get_all_short_term()
        if short_term:
            for entry in short_term[-10:]:  # Letzte 10
                with st.expander(f"{entry.key} ({entry.entry_type})"):
                    st.markdown(f"**Quelle:** {entry.source}")
                    st.markdown(f"**Zeit:** {entry.timestamp}")
                    st.markdown(f"**Relevanz:** {entry.relevance:.2f}")
                    st.text_area("Inhalt", entry.content, height=100, disabled=True, key=f"st_{entry.key}")
        else:
            st.info("Keine Short-Term Eintraege")

    with col_mem2:
        st.markdown("### 💾 Long-Term Memory")
        long_term = st.session_state.memory_system.get_all_long_term()
        if long_term:
            for entry in long_term[-10:]:  # Letzte 10
                with st.expander(f"{entry.key} ({entry.entry_type})"):
                    st.markdown(f"**Quelle:** {entry.source}")
                    st.markdown(f"**Zeit:** {entry.timestamp}")
                    st.markdown(f"**Relevanz:** {entry.relevance:.2f}")
                    st.text_area("Inhalt", entry.content, height=100, disabled=True, key=f"lt_{entry.key}")
        else:
            st.info("Keine Long-Term Eintraege")

    st.divider()

    # Memory Management
    st.markdown("### 🗑️ Memory Management")

    col_clear1, col_clear2, col_clear3 = st.columns(3)

    with col_clear1:
        if st.button("🧹 Short-Term loeschen", use_container_width=True):
            st.session_state.memory_system.clear_short_term()
            st.success("Short-Term Memory geloescht!")
            st.rerun()

    with col_clear2:
        if st.button("🧹 Long-Term loeschen", use_container_width=True):
            st.session_state.memory_system.clear_long_term()
            st.success("Long-Term Memory geloescht!")
            st.rerun()

    with col_clear3:
        if st.button("🧹 Alles loeschen", type="secondary", use_container_width=True):
            st.session_state.memory_system.clear_all()
            st.success("Gesamtes Memory geloescht!")
            st.rerun()

    st.divider()

    # Smart Processing Konfiguration
    st.markdown("### ⚙️ Smart Processing Konfiguration")

    with st.expander("Einstellungen anpassen"):
        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            chunk_size = st.slider(
                "Chunk-Groesse (Zeichen)",
                min_value=1000,
                max_value=5000,
                value=2500,
                step=500,
                help="Maximale Groesse eines Chunks"
            )
            chunk_overlap = st.slider(
                "Chunk-Ueberlappung (Zeichen)",
                min_value=100,
                max_value=500,
                value=300,
                step=50,
                help="Ueberlappung zwischen Chunks"
            )

        with col_cfg2:
            max_context = st.slider(
                "Max. Kontext-Groesse (Zeichen)",
                min_value=1000,
                max_value=4000,
                value=2000,
                step=500,
                help="Maximale Groesse des akkumulierten Kontexts"
            )
            auto_promote = st.slider(
                "Auto-Promote Schwelle",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Relevanz-Schwelle fuer automatische Long-Term Speicherung"
            )

        st.info(f"Aktuelle Konfiguration: Chunks {chunk_size} Zeichen, Overlap {chunk_overlap}, Kontext max {max_context}")


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

### 🧠 Smart Document Processing

Fuer grosse Dateien (>3000 Zeichen) steht Smart Processing zur Verfuegung:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Grosse     │ --> │  Chunking   │ --> │  Pro Chunk  │
│  Datei      │     │  (2500 Z.)  │     │  Analysieren│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│   Finale    │ <-- │   Memory    │ <----------┘
│   Synthese  │     │  Speichern  │
└─────────────┘     └─────────────┘
```

**Memory-System:**
- **Short-Term**: Temporaer, Session-basiert
- **Long-Term**: Persistent, bleibt nach Neustart

**Vorteile:**
- Verarbeitet Dateien beliebiger Groesse
- Akkumuliert Wissen ueber Chunks
- Wiederverwendung bei aehnlichen Anfragen

### Troubleshooting

**Workflow bricht ab?**
- Pruefe LLM-Server Verbindung
- Reduziere Max. Dateigroesse
- Ignoriere mehr Verzeichnisse

**Dokumentation unvollstaendig?**
- Erhoehe Max. Iterationen
- Pruefe Token-Limits

**Smart Processing langsam?**
- Reduziere Chunk-Groesse
- Weniger Chunks = schneller

**Mehr Infos?**
Siehe `docs/installation-de.md`
    """)


# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption("RLM GUI v2.1 | Port 8052 | Smart Processing + Memory | Powered by Streamlit")
