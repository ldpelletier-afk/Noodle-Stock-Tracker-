"""Noodle Stock Tracker — Portfolio Tracker entry point.

Thin dispatcher for the tracking/market-data half of the app: holdings,
watchlists, prices, risk, and valuation. No LLM/RAG dependency — install
requirements-core.txt to run this alone.
"""
import threading as _threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from ui.common import bg_prefetch, load_data

st.set_page_config(page_title="Noodle Stock Tracker — Portfolio", layout="wide")

st.title("Noodle Stock Tracker: Portfolio")

app_data = load_data()

# ── Lazy-load session state ──────────────────────────────────────────────────
# The startup prefetch (below) warms every tab's caches in the background, so
# mark the live-data tabs ready up-front. Tabs then render live data on first
# visit — served from the prefetched cache — instead of gating behind a manual
# 'Power-load all' click on the Dashboard.
if "_lazy_loaded" not in st.session_state:
    st.session_state._lazy_loaded = {"market_watch", "favorites", "asset_tracker"}


# ── Background prefetch ──────────────────────────────────────────────────────
if "_bg_prefetch_started" not in st.session_state:
    st.session_state._bg_prefetch_started = True
    _prefetch_thread = _threading.Thread(
        target=bg_prefetch, args=(app_data,), daemon=True
    )
    # Attach the current script context so cache writes land in st.cache_data
    # (the cache the tabs read) instead of the context-less fallback dict.
    add_script_run_ctx(_prefetch_thread, get_script_run_ctx())
    _prefetch_thread.start()


# ── Navigation ────────────────────────────────────────────────────────────────
_TABS = [
    "🏠 Dashboard",
    "💼 Asset Tracker",
    "📈 Market Watch",
    "⭐ Favorites",
    "📒 History",
    "🛡️ Risk",
    "⚖️ Valuation",
    "🏢 Peer Matrix",
]

st.session_state.setdefault("active_tab", _TABS[0])
if st.session_state["active_tab"] not in _TABS:
    st.session_state["active_tab"] = _TABS[0]

st.radio(
    "Section",
    options=_TABS,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)
_active = st.session_state["active_tab"]


# ── Tab dispatch ──────────────────────────────────────────────────────────────
if _active == "🏠 Dashboard":
    from ui.dashboard import render
    render(app_data)

elif _active == "💼 Asset Tracker":
    from ui.asset_tracker import render
    render(app_data)

elif _active == "📈 Market Watch":
    from ui.market_watch import render
    render(app_data)

elif _active == "⭐ Favorites":
    from ui.favorites import render
    render(app_data)

elif _active == "📒 History":
    from ui.history import render
    render(app_data)

elif _active == "🛡️ Risk":
    from ui.risk_tab import render
    render(app_data)

elif _active == "⚖️ Valuation":
    from ui.valuation import render
    render(app_data)

elif _active == "🏢 Peer Matrix":
    from ui.peer_matrix import render
    render(app_data)
