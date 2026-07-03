"""Noodle Stock Tracker — main entry point.

Thin dispatcher: sets up the page, loads data, runs the sidebar and navigation,
then delegates to the appropriate tab module in the ui/ package.
"""
import os
import threading as _threading

import streamlit as st

import llm_router as _llm_router
from ui.common import DB_DIR, UPLOAD_DIR, bg_prefetch, load_data

# Ensure required directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="The True Oracle", layout="wide")

st.title("The True Oracle: Valuation & Tracking")

app_data = load_data()

# ── Lazy-load session state ──────────────────────────────────────────────────
if "_lazy_loaded" not in st.session_state:
    st.session_state._lazy_loaded = set()


# ── Background prefetch ──────────────────────────────────────────────────────
if "_bg_prefetch_started" not in st.session_state:
    st.session_state._bg_prefetch_started = True
    _threading.Thread(
        target=bg_prefetch, args=(app_data,), daemon=True
    ).start()


# ── Sidebar: LLM backend selector ────────────────────────────────────────────
_llm_options = _llm_router.available_backends()
_llm_labels = {bid: _llm_router.BACKENDS[bid]["label"] for bid in _llm_options}

_default_backend = st.session_state.get("llm_backend") or _llm_router.get_backend()
if _default_backend not in _llm_options:
    _default_backend = _llm_options[0]

with st.sidebar:
    st.markdown("### 🤖 LLM Engine")
    _chosen = st.selectbox(
        "Active model for synthesis & analysis",
        options=_llm_options,
        index=_llm_options.index(_default_backend),
        format_func=lambda bid: _llm_labels.get(bid, bid),
        key="_llm_backend_select",
        label_visibility="collapsed",
    )
    st.session_state.llm_backend = _chosen
    _llm_router.set_backend(_chosen)

    if not _llm_router.has_anthropic_key():
        st.caption(
            "💡 Add `ANTHROPIC_API_KEY=...` to `.env` to unlock Claude models. "
            "Until then, Ollama is the only option."
        )
    else:
        _provider = _llm_router.BACKENDS[_chosen]["provider"]
        if _provider == "anthropic":
            st.caption("☁️ Calls bill against your Anthropic API credits.")
        else:
            st.caption("💻 Running locally via Ollama — free, no network calls.")


# ── Two-level navigation ──────────────────────────────────────────────────────
_SUPERGROUPS = ["💰 Finance", "🌍 Catalysts"]
_FINANCE_TABS = [
    "🏠 Dashboard",
    "💼 Asset Tracker",
    "📈 Market Watch",
    "⭐ Favorites",
    "📒 History",
    "🛡️ Risk",
    "⚖️ Valuation",
    "📰 Intelligence",
    "🏢 Peer Matrix",
    "🌐 Global Markets",
    "📚 The Library",
    "🧠 Analyst",
]
_CATALYST_TABS = [
    "🎯 Catalyst Calendar",
    "🏛️ Monetary Policy",
    "🏗️ Federal Contracts",
    "⚖️ Court Docket",
    "📰 Catalyst News",
]

st.session_state.setdefault("supergroup", _SUPERGROUPS[0])
st.session_state.setdefault("active_finance_tab", _FINANCE_TABS[0])
st.session_state.setdefault("active_catalyst_tab", _CATALYST_TABS[0])
if st.session_state["supergroup"] not in _SUPERGROUPS:
    st.session_state["supergroup"] = _SUPERGROUPS[0]
if st.session_state["active_finance_tab"] not in _FINANCE_TABS:
    st.session_state["active_finance_tab"] = _FINANCE_TABS[0]
if st.session_state["active_catalyst_tab"] not in _CATALYST_TABS:
    st.session_state["active_catalyst_tab"] = _CATALYST_TABS[0]

st.radio(
    "Section",
    options=_SUPERGROUPS,
    horizontal=True,
    label_visibility="collapsed",
    key="supergroup",
)
_supergroup = st.session_state["supergroup"]

if _supergroup == "💰 Finance":
    st.radio(
        "Finance section",
        options=_FINANCE_TABS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_finance_tab",
    )
    _active = st.session_state["active_finance_tab"]
else:
    st.radio(
        "Catalyst section",
        options=_CATALYST_TABS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_catalyst_tab",
    )
    _active = st.session_state["active_catalyst_tab"]


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

elif _active == "📰 Intelligence":
    from ui.intelligence import render
    render(app_data)

elif _active == "🏢 Peer Matrix":
    from ui.peer_matrix import render
    render(app_data)

elif _active == "🌐 Global Markets":
    from ui.global_markets import render
    render(app_data)

elif _active == "📚 The Library":
    from ui.library import render
    render(app_data)

elif _active == "🧠 Analyst":
    from ui.analyst import render
    render(app_data)

elif _active == "🎯 Catalyst Calendar":
    from ui.catalysts import render
    render(app_data)

elif _active == "🏛️ Monetary Policy":
    from ui.monetary_policy import render
    render(app_data)

elif _active == "🏗️ Federal Contracts":
    from ui.federal_contracts import render
    render(app_data)

elif _active == "⚖️ Court Docket":
    from ui.court_docket import render
    render(app_data)

elif _active == "📰 Catalyst News":
    from ui.catalyst_news import render
    render(app_data)
