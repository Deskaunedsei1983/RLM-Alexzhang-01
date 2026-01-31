"""
RLM GUI Modul

Streamlit-basierte Benutzeroberflaeche fuer das RLM-System.

Starten:
    streamlit run gui/app.py
"""

from gui.rlm_backend import RLMBackend, RLMConfig, RLMResult

__all__ = ["RLMBackend", "RLMConfig", "RLMResult"]
