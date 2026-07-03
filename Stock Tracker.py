import concurrent.futures as _fut
import os
import threading as _threading
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from plotly.subplots import make_subplots

from api import (
    fetch_all_news,
    fetch_calendar_events,
    fetch_dcf_data,
    fetch_financial_highlights,
    fetch_financial_news,
    fetch_financial_statements,
    fetch_bea_gdp,
    fetch_bea_pce,
    fetch_bls_indicators,
    fetch_live_prices,
    fetch_treasury_debt,
    fetch_yield_curve,
    has_bea,
    has_eia,
    has_fmp,
    has_simfin,
    has_alpha_vantage,
    live_price_feed_status,
    fetch_macro_data,
    fetch_simfin_statements,
    fetch_simfin_ttm_fcf,
    build_simfin_income_table,
    build_simfin_balance_table,
    build_simfin_cashflow_table,
    fetch_fear_greed,
    fetch_coingecko_global,
    fetch_coingecko_top_coins,
    fetch_commodity_prices,
    fetch_eia_snapshot,
    fetch_fmp_profile,
    fetch_fmp_price_targets,
    fetch_fmp_analyst_estimates,
    fetch_fmp_ratings,
    fetch_av_indicators,
    fetch_cftc_cot,
    fetch_cftc_snapshot,
    fetch_peer_metrics,
    fetch_portfolio_value_history,
    fetch_recent_sec_filings,
    fetch_sec_filing,
    fetch_sparkline_history,
    fetch_stock_details,
    fetch_courtlistener_search,
    fetch_url_metadata,
    fetch_usaspending_awards,
    fetch_usaspending_summary,
    fred,
    get_catalyst_news,
    get_fomc_schedule,
    get_major_court_cases,
    get_treasury_refunding_schedule,
    has_courtlistener,
    FEDERAL_CONTRACTOR_TICKERS,
    _CATALYST_KEYWORD_MAP,
)
from data_store import (
    CATALYST_STATUSES,
    CATALYST_STATUS_LABELS,
    CATALYST_TYPES,
    CATALYST_TYPE_LABELS,
    add_catalyst,
    add_favorite,
    add_institution,
    add_to_watchlist,
    create_watchlist,
    delete_catalyst,
    get_catalyst,
    delete_saved_article,
    delete_watchlist,
    fetch_transactions,
    import_transactions,
    link_catalyst_doc,
    link_institution_doc,
    list_catalysts,
    list_favorites,
    list_institution_docs,
    list_institutional_coverage,
    list_saved_articles,
    load_data as _load_data_sqlite,
    log_transaction,
    remove_favorite,
    remove_from_watchlist,
    remove_institution,
    rename_watchlist,
    save_article,
    save_data as _save_data_sqlite,
    set_primary_institution_doc,
    set_target_in_watchlist,
    ui_state_get,
    ui_state_set,
    unlink_catalyst_doc,
    unlink_doc_from_all_catalysts,
    unlink_doc_from_all_institutions,
    unlink_institution_doc,
    update_catalyst,
    update_favorite,
    update_saved_article_note,
)
from rag import CATEGORIES as _CATEGORIES
from rag import TOPIC_LABELS as _TOPIC_LABELS
from rag import TOPICS as _TOPICS
from rag import already_ingested as _already_ingested
from rag import decompose_query as _decompose_query
from rag import delete_document as _delete_document
from rag import format_chunks_for_citation as _format_chunks_for_citation
from rag import ingest_chunks as _ingest_chunks
from rag import list_documents as _list_documents
from rag import retrieve_multi as _retrieve_multi
from rag import route_query as _route_query
from rag import set_category as _set_category
from rag import set_topics as _set_topics
from rag import verify_citations as _verify_citations
from rag import vector_db as _vector_db
from utils import format_large_number, highlight_buy_zone, sanitize_ticker

# Risk analytics + agentic ticker analyst — lazy-friendly; they only hit
# yfinance / ollama when their tabs are actually used.
from risk import portfolio_risk_report as _portfolio_risk_report
from agent import run_analyze_ticker as _run_analyze_ticker

# --- CONFIGURATION & STORAGE ---
DB_DIR = "./chroma_db"
UPLOAD_DIR = "./temp_pdfs"

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="The True Oracle", layout="wide")


@st.cache_data(ttl=60)
def _load_data_cached(_token: int):
    """Cached pull of the full app dict (portfolios + watchlists + peer_groups
    + favorites). The `_token` arg is the cache-busting handle: bump it via
    `_invalidate_app_data()` after any write so the next load_data() re-reads.

    A short TTL (60 s) is a belt-and-suspenders fallback so manual edits to
    portfolio.db (or out-of-band updates from the watcher service) eventually
    propagate even if some code path forgets to invalidate.
    """
    return _load_data_sqlite()


def load_data():
    """Render-cycle cache for the app dict. Saves dozens of full SQLite reads
    per Streamlit rerun (every widget interaction used to hit the DB)."""
    return _load_data_cached(st.session_state.get("_app_data_token", 0))


def _invalidate_app_data() -> None:
    """Bump the cache token so the next load_data() picks up DB writes."""
    st.session_state["_app_data_token"] = st.session_state.get("_app_data_token", 0) + 1


def save_data(data):
    _save_data_sqlite(data)
    _invalidate_app_data()

# --- MAIN APP ---
st.title("The True Oracle: Valuation & Tracking")

app_data = load_data()
portfolios = app_data.get("portfolios", {})
watch_list_targets = app_data.get("watch_list_targets", {})
peer_groups = app_data.get("peer_groups", {})

# ---- Lazy-load session state: tracks which tabs have fetched live data ----
if "_lazy_loaded" not in st.session_state:
    st.session_state._lazy_loaded = set()

# ---- Background pre-fetch: warm caches for heavy tabs in a daemon thread ----
# Streamlit's cache_data is process-wide, so the thread populates the same
# cache the main thread reads from — fetches are instant when user navigates
# AFTER the prefetch finishes. Runs once per session at module-load time and
# again on demand via the dashboard "⚡ Power-load all" button.
def _bg_prefetch(deep: bool = False):
    """Warm the most-touched caches.

    `deep=True` adds the heavier optional fetches (Library doc list, macro
    series, sentiment, news) so the user pays a single cold-cache cost upfront
    and every tab feels instant for the rest of the session.
    """
    _wl_tickers = sorted(
        t
        for items in app_data.get("watchlists", {}).values()
        for t in items
        if t and t.upper() != "CASH"
    )
    _fav_tickers = list(
        t for t in (app_data.get("favorites") or {}).keys()
        if t and t.upper() != "CASH"
    )
    # ── Tier 1: cheap and high-leverage (live prices, sparklines) ──
    if _wl_tickers:
        try:
            fetch_live_prices(_wl_tickers)
            fetch_sparkline_history(tuple(sorted(_wl_tickers)))
        except Exception:
            pass
    if _fav_tickers:
        try:
            fetch_live_prices(_fav_tickers)
        except Exception:
            pass

    if not deep:
        return

    # ── Tier 2: macro / sentiment (always-on signals used by Library Oracle) ──
    try:
        if fred:
            fetch_macro_data("FEDFUNDS")
            fetch_macro_data("BAMLH0A0HYM2")
            fetch_macro_data("DGS10")
    except Exception:
        pass
    try:
        fetch_fear_greed()
    except Exception:
        pass

    # ── Tier 3: Library doc list (paginated 78k+ chunk fetch is now ~1-2s) ──
    try:
        from rag import list_documents as _ld
        _ld()
    except Exception:
        pass

    # ── Tier 4: Calendar events for watch-listed tickers (Dashboard tile) ──
    if _wl_tickers:
        try:
            fetch_calendar_events(tuple(sorted(_wl_tickers))[:10])
        except Exception:
            pass

if "_bg_prefetch_started" not in st.session_state:
    st.session_state._bg_prefetch_started = True
    _threading.Thread(target=_bg_prefetch, daemon=True).start()

# ---- LLM backend selector (sidebar) -------------------------------------
# Lets the user switch between local Ollama (free) and Anthropic Claude
# (paid API) on the fly. Anthropic options auto-hide when no API key is set.
import llm_router as _llm_router

_llm_options = _llm_router.available_backends()
_llm_labels = {bid: _llm_router.BACKENDS[bid]["label"] for bid in _llm_options}

# Pick the previously-selected backend if it's still valid; else default.
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

# ===========================
# STATEFUL NAVIGATION (two-level)
# ===========================
# Top-level supergroup: Finance vs Catalysts. Each supergroup remembers its
# own active subtab independently, so flipping back and forth between them
# never loses your place. All persistence is automatic via Streamlit widget
# keys — no manual session_state writes needed.
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

# Defensive defaults — every key initialised so first-render never blanks.
st.session_state.setdefault("supergroup", _SUPERGROUPS[0])
st.session_state.setdefault("active_finance_tab",  _FINANCE_TABS[0])
st.session_state.setdefault("active_catalyst_tab", _CATALYST_TABS[0])
# Snap back if a persisted label drifted (e.g. label-rename across versions)
if st.session_state["supergroup"] not in _SUPERGROUPS:
    st.session_state["supergroup"] = _SUPERGROUPS[0]
if st.session_state["active_finance_tab"] not in _FINANCE_TABS:
    st.session_state["active_finance_tab"] = _FINANCE_TABS[0]
if st.session_state["active_catalyst_tab"] not in _CATALYST_TABS:
    st.session_state["active_catalyst_tab"] = _CATALYST_TABS[0]

# Row 1 — supergroup selector. Visually a segmented strip across the top.
st.radio(
    "Section",
    options=_SUPERGROUPS,
    horizontal=True,
    label_visibility="collapsed",
    key="supergroup",
)
_supergroup = st.session_state["supergroup"]

# Row 2 — subtab strip whose options depend on the active supergroup.
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

# =============================================================================
#  CATALYST HELPERS (defined globally so the Analyst tab can also render them)
# =============================================================================
# Both the Analyst tab (transparency panel showing which catalysts the LLM
# saw) and the Catalysts supergroup (calendar / monetary / contracts subtabs)
# render catalyst cards. Defining the helpers here — above the per-tab `if`
# blocks — guarantees they exist regardless of which tab is currently active.

import datetime as _dt


def _format_event_date(ts: int) -> str:
    """Human-friendly date label, with a relative countdown for the bulletin."""
    if not ts:
        return "—"
    d   = _dt.datetime.fromtimestamp(int(ts))
    now = _dt.datetime.now()
    delta_days = (d.date() - now.date()).days
    base = d.strftime("%a %b %d, %Y")
    if delta_days == 0:
        return f"{base}  ·  **today**"
    if delta_days == 1:
        return f"{base}  ·  tomorrow"
    if delta_days == -1:
        return f"{base}  ·  yesterday"
    if 0 < delta_days <= 30:
        return f"{base}  ·  in {delta_days} days"
    if -30 <= delta_days < 0:
        return f"{base}  ·  {-delta_days} days ago"
    return base


def _render_catalyst_card(c: dict, expanded: bool = False) -> None:
    """One catalyst, rendered as an expandable card. Edit/delete inline."""
    type_label   = CATALYST_TYPE_LABELS.get(c.get("catalyst_type"), c.get("catalyst_type", ""))
    status_label = CATALYST_STATUS_LABELS.get(c.get("status"), c.get("status", ""))
    date_label   = _format_event_date(c.get("event_date"))

    header = f"**{c.get('title','(untitled)')}**"
    summary_line = f"{type_label}  ·  {status_label}  ·  {date_label}"

    with st.expander(f"{header} — {summary_line}", expanded=expanded):
        # Stakes (the analytical core) — markdown rendered.
        if c.get("stakes"):
            st.markdown(c["stakes"])
        else:
            st.caption("_No stakes recorded yet — click ✏️ Edit to add._")

        # Compact metadata grid
        meta_cols = st.columns(4)
        if c.get("category"):
            meta_cols[0].caption(f"**Category**\n\n{c['category']}")
        if c.get("probability"):
            meta_cols[1].caption(f"**Probability**\n\n{c['probability']}")
        if c.get("tickers"):
            meta_cols[2].caption("**Tickers**\n\n" + " ".join(f"`{t}`" for t in c["tickers"]))
        if c.get("sectors"):
            meta_cols[3].caption("**Sectors**\n\n" + ", ".join(c["sectors"]))

        # Resolution outcome (only for resolved entries)
        if c.get("status") == "resolved" and c.get("outcome_notes"):
            st.success(f"**Outcome:** {c['outcome_notes']}")

        # Linked Library docs
        if c.get("doc_ids"):
            with st.expander(f"📎 Linked library docs ({len(c['doc_ids'])})"):
                for did in c["doc_ids"]:
                    st.markdown(f"- `{did}`")

        # Action row
        ac1, ac2, ac3 = st.columns([1, 1, 6])
        if ac1.button("✏️ Edit", key=f"cat_edit_{c['id']}"):
            st.session_state["_catalyst_edit_id"] = c["id"]
            st.session_state["_catalyst_form_open"] = True
            st.rerun()
        if ac3.button("🗑️ Delete", key=f"cat_del_{c['id']}",
                      help="Permanently remove this catalyst"):
            delete_catalyst(c["id"])
            st.toast(f"Deleted: {c.get('title','(untitled)')}", icon="🗑️")
            st.rerun()


# ===========================
# TAB: DASHBOARD (homepage)
# ===========================
if _active == "🏠 Dashboard":
    st.header("🏠 Dashboard")
    st.caption(
        "Portfolio performance overview, reconstructed live from your transaction "
        "history × current prices."
    )

    # ---- Unified live-data loader ----
    # One button activates the live-data tabs (Market Watch / Favorites / Asset
    # Tracker) AND warms the heavy upstream caches that the Library Oracle and
    # Analyst depend on (FRED macro series, CNN F&G, Library doc list, calendar
    # events). Pay one cold-cache hit here, every tab afterwards is instant.
    _LIVE_TARGETS = ("market_watch", "favorites", "asset_tracker")
    _all_live_loaded = all(
        _t in st.session_state._lazy_loaded for _t in _LIVE_TARGETS
    )

    _ldc1, _ldc2 = st.columns([1, 4])
    with _ldc1:
        if _all_live_loaded:
            if st.button("🔄 Refresh live data", key="dash_refresh_btn",
                         help="Re-fetch prices, sparklines, and calendar events."):
                fetch_live_prices.clear()
                fetch_sparkline_history.clear()
                try:
                    fetch_calendar_events.clear()
                except Exception:
                    pass
                st.rerun()
        else:
            if st.button("⚡ Power-load all", key="dash_load_btn",
                         type="primary",
                         help="One click: warms live prices, sparklines, calendar events, "
                              "FRED macro series, CNN F&G, and the full Library doc list. "
                              "Takes 5–15 s on a cold start; afterwards every tab is instant."):
                for _t in _LIVE_TARGETS:
                    st.session_state._lazy_loaded.add(_t)
                # Run a synchronous deep prefetch so the user sees the spinner
                # finish before the page lights up — matches the request's
                # "I don't mind a little wait" framing.
                with st.spinner("Warming caches across all tabs (live prices, macro, sentiment, library)…"):
                    _bg_prefetch(deep=True)
                st.toast("All caches warmed — every tab is now snappy.", icon="⚡")
                st.rerun()
    with _ldc2:
        _feed = live_price_feed_status()
        if _feed["alpaca_configured"]:
            _feed_badge = f"📡 **Feed:** {_feed['label']} · auto-refresh every {_feed['ttl_seconds']}s"
        else:
            _feed_badge = (
                f"📡 **Feed:** {_feed['label']} · auto-refresh every {_feed['ttl_seconds']}s "
                "_(add `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` to `.env` for real-time IEX data)_"
            )
        if _all_live_loaded:
            st.caption(f"✅ Live data is loaded across Market Watch, Favorites, and Asset Tracker.\n\n{_feed_badge}")
        else:
            st.caption(
                "Click once to warm every tab's caches — live prices, sparklines, "
                "calendar events, FRED macro, sentiment, and the full document library. "
                "After this finishes, every navigation feels instant.\n\n"
                + _feed_badge
            )

    st.divider()

    # ---- Portfolio selector ----
    _portfolio_names = list(portfolios.keys())
    if not _portfolio_names:
        st.info(
            "Create a portfolio in the **Asset Tracker** tab and log some "
            "transactions to populate the dashboard."
        )
    else:
        if len(_portfolio_names) > 1:
            _portfolio_choice = st.selectbox(
                "Portfolio",
                ["All combined"] + _portfolio_names,
                key="dash_portfolio",
            )
        else:
            _portfolio_choice = _portfolio_names[0]
            st.caption(f"Portfolio: **{_portfolio_choice}**")

        # Build the active holdings dict for the chosen scope
        if _portfolio_choice == "All combined":
            _active_holdings: dict[str, dict[str, float]] = {}
            for _p_name, _p_holdings in portfolios.items():
                for _ticker, _pos in (_p_holdings or {}).items():
                    if _ticker in _active_holdings:
                        _old_qty = _active_holdings[_ticker]["quantity"]
                        _new_qty = float(_pos.get("quantity", 0.0))
                        _old_cost = _active_holdings[_ticker]["average_cost"]
                        _new_cost = float(_pos.get("average_cost", 0.0))
                        _total = _old_qty + _new_qty
                        _avg = (
                            (_old_qty * _old_cost + _new_qty * _new_cost) / _total
                            if _total > 0 else _new_cost
                        )
                        _active_holdings[_ticker] = {
                            "quantity": _total,
                            "average_cost": _avg,
                        }
                    else:
                        _active_holdings[_ticker] = {
                            "quantity": float(_pos.get("quantity", 0.0)),
                            "average_cost": float(_pos.get("average_cost", 0.0)),
                        }
            _active_pname = None  # all combined → no filter
        else:
            _src = portfolios.get(_portfolio_choice, {}) or {}
            _active_holdings = {
                t: {
                    "quantity": float(p.get("quantity", 0.0)),
                    "average_cost": float(p.get("average_cost", 0.0)),
                }
                for t, p in _src.items()
            }
            _active_pname = _portfolio_choice

        _holding_tickers = [
            t for t, p in _active_holdings.items()
            if p["quantity"] > 0 and t.upper() != "CASH"
        ]
        _has_cash = "CASH" in _active_holdings and _active_holdings["CASH"]["quantity"] > 0

        if not _holding_tickers and not _has_cash:
            st.info(
                "No holdings yet. Add positions in the **Asset Tracker** tab to "
                "see dashboard metrics."
            )
        else:
            with st.spinner("Loading prices..."):
                _live = fetch_live_prices(_holding_tickers) if _holding_tickers else {}

            # ---- Aggregate metrics ----
            _total_value = 0.0
            _total_cost = 0.0
            _day_pl = 0.0
            for _t in _holding_tickers:
                _qty = _active_holdings[_t]["quantity"]
                _avg = _active_holdings[_t]["average_cost"]
                _price = _live.get(_t, {}).get("price")
                _ch_pct = _live.get(_t, {}).get("change") or 0.0
                if _price is None:
                    continue
                _total_value += _qty * _price
                _total_cost += _qty * _avg
                # Day P&L: qty × (price − prev_close)
                if _ch_pct != 0:
                    _prev_close = _price / (1.0 + _ch_pct / 100.0)
                    _day_pl += _qty * (_price - _prev_close)

            _cash_qty = _active_holdings.get("CASH", {}).get("quantity", 0.0)
            _total_value += _cash_qty
            _total_cost += _cash_qty  # cash is its own cost

            _total_return_dollar = _total_value - _total_cost
            _total_return_pct = (
                (_total_return_dollar / _total_cost * 100.0) if _total_cost > 0 else 0.0
            )
            _opening_value = _total_value - _day_pl
            _day_pl_pct = (_day_pl / _opening_value * 100.0) if _opening_value > 0 else 0.0

            # ---- Hero metric row ----
            _h1, _h2, _h3, _h4 = st.columns(4)
            _h1.metric("💰 Total Value", f"${_total_value:,.2f}")
            _h2.metric(
                "📊 Today's P&L",
                f"${_day_pl:+,.2f}",
                delta=(f"{_day_pl_pct:+.2f}%" if abs(_day_pl_pct) > 0.005 else None),
            )
            _h3.metric(
                "📈 Total Return",
                f"${_total_return_dollar:+,.2f}",
                delta=f"{_total_return_pct:+.2f}%",
            )
            _h4.metric(
                "🎯 Positions",
                f"{len(_holding_tickers)}",
                delta=(f"+ ${_cash_qty:,.0f} cash" if _cash_qty > 0 else None),
            )

            st.divider()

            # ---- Portfolio Value Over Time ----
            st.subheader("📊 Portfolio Value Over Time")
            _period_choice = st.radio(
                "Period",
                ["1M", "3M", "6M", "YTD", "1Y", "All"],
                index=4,
                horizontal=True,
                key="dash_period",
                label_visibility="collapsed",
            )
            _now = pd.Timestamp.now().normalize()
            _ytd_days = (_now - pd.Timestamp(year=_now.year, month=1, day=1)).days
            _period_days = {
                "1M": 30, "3M": 90, "6M": 180, "YTD": max(_ytd_days, 1),
                "1Y": 365, "All": 365 * 5,
            }[_period_choice]

            with st.spinner("Reconstructing portfolio history..."):
                _value_df = fetch_portfolio_value_history(
                    portfolio_name=_active_pname,
                    days=_period_days,
                )

            if _value_df is not None and not _value_df.empty:
                _start_v = float(_value_df["Value"].iloc[0])
                _end_v = float(_value_df["Value"].iloc[-1])
                _period_change = _end_v - _start_v
                _period_pct = (
                    _period_change / _start_v * 100.0 if _start_v > 0 else 0.0
                )
                _line_color = "#28a745" if _period_change >= 0 else "#dc3545"

                _fig = go.Figure()
                _fig.add_trace(go.Scatter(
                    x=_value_df["Date"], y=_value_df["Value"],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=_line_color, width=2),
                    fillcolor=(
                        "rgba(40,167,69,0.10)" if _period_change >= 0
                        else "rgba(220,53,69,0.10)"
                    ),
                    hovertemplate="%{x|%b %d, %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
                    name="Portfolio Value",
                ))
                _fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    showlegend=False,
                    height=320,
                    yaxis=dict(tickformat="$,.0f", gridcolor="rgba(128,128,128,0.2)"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(_fig, use_container_width=True)
                _arrow = "▲" if _period_change >= 0 else "▼"
                st.caption(
                    f"{_period_choice}: **{_arrow} ${abs(_period_change):,.2f} "
                    f"({_period_pct:+.2f}%)**  ·  {_value_df['Date'].iloc[0]:%b %d, %Y} "
                    f"→ {_value_df['Date'].iloc[-1]:%b %d, %Y}"
                )
            else:
                st.info(
                    "Not enough transaction history to plot value over time. "
                    "Log some transactions in the History tab."
                )

            st.divider()

            # ---- Top Holdings + Biggest Movers, side-by-side ----
            _col_left, _col_right = st.columns(2)

            with _col_left:
                st.subheader("🏆 Top Holdings")
                _holding_rows = []
                for _t in _holding_tickers:
                    _qty = _active_holdings[_t]["quantity"]
                    _avg = _active_holdings[_t]["average_cost"]
                    _price = _live.get(_t, {}).get("price")
                    _ch = _live.get(_t, {}).get("change")
                    if _price is None:
                        continue
                    _value = _qty * _price
                    _ret_pct = (
                        (_price - _avg) / _avg * 100.0 if _avg > 0 else 0.0
                    )
                    _holding_rows.append({
                        "Ticker": _t,
                        "Shares": _qty,
                        "Value": _value,
                        "Return %": _ret_pct,
                        "Day %": _ch,
                    })
                _holding_rows.sort(key=lambda r: r["Value"], reverse=True)
                if _holding_rows:
                    _top_df = pd.DataFrame(_holding_rows[:8])
                    st.dataframe(
                        _top_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Shares": st.column_config.NumberColumn(format="%.4f"),
                            "Value": st.column_config.NumberColumn(format="$%.2f"),
                            "Return %": st.column_config.NumberColumn(format="%+.2f%%"),
                            "Day %": st.column_config.NumberColumn(format="%+.2f%%"),
                        },
                    )
                else:
                    st.caption("No priced holdings to show yet.")

            with _col_right:
                st.subheader("🚀 Today's Biggest Movers")
                # Pool: all holdings + every watchlist ticker
                _watchlists = app_data.get("watchlists", {})
                _all_movers = set(_holding_tickers)
                for _items in _watchlists.values():
                    _all_movers.update(_items.keys())
                _all_movers.discard("CASH")
                _all_movers.discard("")
                _mover_prices = fetch_live_prices(list(_all_movers))

                _mover_rows = []
                for _t in _all_movers:
                    _p = _mover_prices.get(_t, {}).get("price")
                    _c = _mover_prices.get(_t, {}).get("change")
                    if _p is None or _c is None:
                        continue
                    _mover_rows.append({"Ticker": _t, "Price": _p, "Day %": _c})

                if _mover_rows:
                    _winners = sorted(
                        _mover_rows, key=lambda r: r["Day %"], reverse=True
                    )[:3]
                    _losers = sorted(_mover_rows, key=lambda r: r["Day %"])[:3]
                    st.markdown("**🟢 Winners**")
                    for _m in _winners:
                        st.markdown(
                            f"- **{_m['Ticker']}** · ${_m['Price']:.2f} "
                            f"<span style='color:#28a745'>(+{_m['Day %']:.2f}%)</span>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("**🔴 Losers**")
                    for _m in _losers:
                        st.markdown(
                            f"- **{_m['Ticker']}** · ${_m['Price']:.2f} "
                            f"<span style='color:#dc3545'>({_m['Day %']:.2f}%)</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No movers to show yet.")


# ===========================
# TAB 1: MARKET WATCH
# ===========================
if _active == "📈 Market Watch":
    watchlists = app_data.get("watchlists", {})

    # Sorted list for deterministic cache keys — matches the background pre-fetch.
    all_wl_tickers = sorted({t for items in watchlists.values() for t in items})

    _mw_ready = "market_watch" in st.session_state._lazy_loaded

    if not _mw_ready:
        # Live fetches are gated by the unified Dashboard loader. Until the
        # user clicks the Dashboard's "Load all live data", we render the
        # static watchlist without prices/sparklines.
        live_prices = {}
        _sparklines = {}
        st.caption(
            "📡 Live prices not loaded yet — click **⚡ Load all live data** on the "
            "🏠 Dashboard tab to populate prices, sparklines, and the calendar."
        )
    else:
        with st.spinner("Fetching live market data..."):
            live_prices = fetch_live_prices(all_wl_tickers)
        with st.spinner("Loading 30-day trends..."):
            _sparklines = fetch_sparkline_history(tuple(sorted(all_wl_tickers)))

    # Global volatility alert (across every list)
    crashing_assets = [
        f"**{t}** ({live_prices.get(t, {}).get('change', 0):.2f}%)"
        for t in all_wl_tickers
        if live_prices.get(t, {}).get("change") is not None
        and live_prices.get(t, {}).get("change") <= -5.0
    ]
    if crashing_assets:
        st.error(
            f"🚨 **Volatility Alert:** The following assets are down 5 % or more "
            f"today: {', '.join(crashing_assets)}"
        )

    # ---- Upcoming Calendar (earnings + ex-dividend dates) ----
    if all_wl_tickers:
        with st.expander("📅 Upcoming Calendar (next 60 days)", expanded=False):
            if _mw_ready:
                with st.spinner("Loading earnings + dividend dates..."):
                    _cal_df = fetch_calendar_events(tuple(sorted(all_wl_tickers)))
                if _cal_df is None or _cal_df.empty:
                    st.caption(
                        "No upcoming earnings or ex-dividend dates found in the next "
                        "60 days for your watchlist tickers."
                    )
                else:
                    _cal_show = _cal_df.copy()
                    _cal_show["Date"] = _cal_show["Date"].dt.strftime("%a %b %d, %Y")
                    _cal_show["When"] = _cal_df["Date"].apply(
                        lambda d: f"in {(d - pd.Timestamp.now().normalize()).days}d"
                    )
                    _cal_show = _cal_show[["Date", "When", "Ticker", "Event", "Detail"]]
                    st.dataframe(
                        _cal_show,
                        use_container_width=True,
                        hide_index=True,
                        height=min(500, 38 + 35 * len(_cal_show) + 4),
                    )
            else:
                st.caption("Load live data above to see upcoming events.")

    # ---- Per-list expandable sections ----
    _COL_CFG = {
        "Live Price (from API)": st.column_config.NumberColumn(format="$%.2f"),
        "Day Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Target Price (Self-set)": st.column_config.NumberColumn(
            format="$%.2f", step=0.01
        ),
        "Trend (30d)": st.column_config.LineChartColumn(
            "Trend (30d)",
            help="Last 30 trading days of closing prices",
            width="medium",
        ),
    }

    _SORT_OPTIONS = [
        "Default order",
        "Ticker A→Z",
        "Day % (best first)",
        "Day % (worst first)",
        "Closest to target",
    ]

    def _apply_sort(df: pd.DataFrame, choice: str) -> pd.DataFrame:
        if choice == "Ticker A→Z":
            return df.sort_values("Ticker", kind="stable").reset_index(drop=True)
        if choice == "Day % (best first)":
            return df.sort_values(
                "Day Change (%)", ascending=False, na_position="last", kind="stable"
            ).reset_index(drop=True)
        if choice == "Day % (worst first)":
            return df.sort_values(
                "Day Change (%)", ascending=True, na_position="last", kind="stable"
            ).reset_index(drop=True)
        if choice == "Closest to target":
            def _dist(row):
                p, t = row["Live Price (from API)"], row["Target Price (Self-set)"]
                if p is None or t is None or t == 0:
                    return float("inf")
                return abs(p - t) / t
            tmp = df.copy()
            tmp["__d"] = tmp.apply(_dist, axis=1)
            return tmp.sort_values("__d", kind="stable").drop(columns="__d").reset_index(drop=True)
        return df

    # ---- Per-list open/closed state, persisted across reruns + app launches ----
    # Streamlit's st.expander cannot reliably hold the user's collapsed/expanded
    # state across script reruns: the `expanded` arg is treated as the *initial*
    # state, and any st.rerun() (e.g. after Add/Remove) wipes the user's toggle.
    # We work around it by owning the state ourselves: a clickable header button
    # toggles a session_state flag, mirrored to a tiny SQLite table so it also
    # survives kill-and-relaunch.
    def _wl_state_key(name: str) -> str:
        return f"wl_open__{name}"

    def _wl_is_open(name: str, default: bool = True) -> bool:
        sk = _wl_state_key(name)
        if sk not in st.session_state:
            saved = ui_state_get(f"wl_open:{name}", default="1" if default else "0")
            st.session_state[sk] = saved == "1"
        return bool(st.session_state[sk])

    def _wl_set_open(name: str, is_open: bool) -> None:
        st.session_state[_wl_state_key(name)] = bool(is_open)
        ui_state_set(f"wl_open:{name}", "1" if is_open else "0")

    if watchlists:
        for _list_name, _items in watchlists.items():
            _list_tickers = list(_items.keys())
            _count = len(_list_tickers)
            _is_open = _wl_is_open(_list_name)

            # Clickable header that toggles open/closed and persists the choice
            _arrow = "🔽" if _is_open else "▶️"
            _header_label = (
                f"{_arrow}  📋  {_list_name}  ·  {_count} "
                f"stock{'s' if _count != 1 else ''}"
            )
            if st.button(
                _header_label,
                key=f"wl_toggle_{_list_name}",
                use_container_width=True,
                help="Click to collapse" if _is_open else "Click to expand",
            ):
                _wl_set_open(_list_name, not _is_open)
                st.rerun()

            # Render the list contents only when this list is open
            if _is_open:
                # Price table
                if _list_tickers:
                    # Sort dropdown above the table
                    _sort_choice = st.selectbox(
                        "Sort by",
                        _SORT_OPTIONS,
                        key=f"wl_sort_{_list_name}",
                        label_visibility="collapsed",
                    )

                    _df = pd.DataFrame({
                        "Ticker": _list_tickers,
                        "Live Price (from API)": [
                            live_prices.get(t, {}).get("price") for t in _list_tickers
                        ],
                        "Day Change (%)": [
                            live_prices.get(t, {}).get("change") for t in _list_tickers
                        ],
                        "Target Price (Self-set)": [
                            _items.get(t, 0.0) for t in _list_tickers
                        ],
                        "Trend (30d)": [_sparklines.get(t, []) for t in _list_tickers],
                    })
                    _df = _apply_sort(_df, _sort_choice)
                    # Size the editor to show every row without an inner scrollbar.
                    # Streamlit defaults to a fixed height (~250px / ~10 rows);
                    # we override it so the whole list is visible at once.
                    # ~35px per data row + ~38px header + small padding buffer.
                    _editor_height = 38 + 35 * _count + 4
                    _edited = st.data_editor(
                        _df.style.apply(highlight_buy_zone, axis=1),
                        disabled=[
                            "Ticker", "Live Price (from API)",
                            "Day Change (%)", "Trend (30d)",
                        ],
                        hide_index=True,
                        use_container_width=True,
                        height=_editor_height,
                        key=f"wl_editor_{_list_name}",
                        column_config=_COL_CFG,
                    )
                    # Save any changed target prices immediately
                    # (compare just the editable column so sparkline lists
                    # don't trigger spurious "changed" detections)
                    _changes = []
                    for _, _row in _edited.iterrows():
                        _old = float(_items.get(_row["Ticker"], 0.0) or 0.0)
                        _new = float(_row["Target Price (Self-set)"] or 0.0)
                        if abs(_new - _old) > 1e-9:
                            _changes.append((_row["Ticker"], _new))
                    if _changes:
                        for _t, _new in _changes:
                            set_target_in_watchlist(_list_name, _t, _new)
                        st.toast(f"Targets updated in '{_list_name}'", icon="✅")
                        st.rerun()
                else:
                    st.caption("This list is empty — add a ticker below.")

                st.divider()

                # Add / Remove ticker
                _ca, _cr = st.columns(2)
                with _ca:
                    st.markdown("**Add ticker**")
                    # Wrapped in st.form so the submit button atomically commits
                    # the text input — otherwise a click before the input has
                    # blurred would register an empty value and you'd need to
                    # click twice. Pressing Enter in the input also submits.
                    with st.form(
                        key=f"wl_add_form_{_list_name}",
                        clear_on_submit=True,
                    ):
                        _add_input = st.text_input(
                            "Ticker",
                            key=f"wl_add_{_list_name}",
                            placeholder="e.g. MSFT",
                            label_visibility="collapsed",
                        )
                        _add_clicked = st.form_submit_button(
                            "➕ Add", use_container_width=True
                        )
                    if _add_clicked:
                        _add_t = sanitize_ticker(_add_input)
                        if _add_t:
                            if add_to_watchlist(_list_name, _add_t):
                                st.rerun()
                            else:
                                st.warning(f"{_add_t} is already in '{_list_name}'.")

                    # Bulk add — paste many tickers at once
                    with st.expander("📥 Bulk add (paste multiple tickers)"):
                        with st.form(
                            key=f"wl_bulk_form_{_list_name}",
                            clear_on_submit=True,
                        ):
                            _bulk_text = st.text_area(
                                "Tickers (comma, space, or newline separated)",
                                key=f"wl_bulk_{_list_name}",
                                placeholder="AAPL, MSFT, GOOG\nNVDA TSLA META\nAMZN",
                                height=100,
                            )
                            _bulk_clicked = st.form_submit_button(
                                "➕ Add all", use_container_width=True
                            )
                        if _bulk_clicked and _bulk_text.strip():
                            # Split on any combination of commas, whitespace, or newlines
                            import re as _re
                            _raw = [
                                p.strip() for p in _re.split(r"[,\s]+", _bulk_text)
                                if p.strip()
                            ]
                            _seen = set()
                            _to_add = []
                            for _t in _raw:
                                _clean = sanitize_ticker(_t)
                                if _clean and _clean not in _seen:
                                    _seen.add(_clean)
                                    _to_add.append(_clean)
                            _added = _skipped = 0
                            for _clean in _to_add:
                                if add_to_watchlist(_list_name, _clean):
                                    _added += 1
                                else:
                                    _skipped += 1
                            if _added or _skipped:
                                _msg_parts = []
                                if _added:
                                    _msg_parts.append(f"added {_added}")
                                if _skipped:
                                    _msg_parts.append(f"skipped {_skipped} duplicate")
                                st.toast(
                                    f"'{_list_name}': " + ", ".join(_msg_parts),
                                    icon="✅",
                                )
                                st.rerun()
                with _cr:
                    st.markdown("**Remove ticker**")
                    if _list_tickers:
                        _rm_t = st.selectbox(
                            "Ticker",
                            _list_tickers,
                            key=f"wl_rm_{_list_name}",
                            label_visibility="collapsed",
                        )
                        if st.button(
                            "🗑️ Remove",
                            key=f"wl_rmbtn_{_list_name}",
                            use_container_width=True,
                            type="primary",
                        ):
                            remove_from_watchlist(_list_name, _rm_t)
                            st.rerun()
                    else:
                        st.empty()

                # Rename / Delete list
                st.markdown("---")
                _cn, _cd = st.columns([4, 1])
                with _cn:
                    with st.form(
                        key=f"wl_rename_form_{_list_name}",
                        clear_on_submit=False,
                    ):
                        _new_name = st.text_input(
                            "Rename list",
                            value=_list_name,
                            key=f"wl_rename_{_list_name}",
                            label_visibility="collapsed",
                            placeholder="New list name…",
                        )
                        _rename_clicked = st.form_submit_button("✏️ Rename list")
                    if _rename_clicked:
                        if _new_name and _new_name != _list_name:
                            if rename_watchlist(_list_name, _new_name):
                                # Migrate the persisted open/closed state to
                                # the new name so the user's UI doesn't reset.
                                _was_open = _wl_is_open(_list_name)
                                _wl_set_open(_new_name, _was_open)
                                st.session_state.pop(
                                    _wl_state_key(_list_name), None
                                )
                                st.rerun()
                            else:
                                st.warning(
                                    f"A list named '{_new_name}' already exists."
                                )
                with _cd:
                    st.write("")
                    st.write("")
                    if st.button(
                        "🗑️ Delete list",
                        key=f"wl_delbtn_{_list_name}",
                        type="primary",
                        use_container_width=True,
                    ):
                        delete_watchlist(_list_name)
                        st.session_state.pop(_wl_state_key(_list_name), None)
                        st.rerun()
    else:
        st.info("No watch lists yet — create one below.")

    st.divider()

    # ---- Create new list ----
    st.subheader("➕ Create new list")
    with st.form(key="wl_create_list_form", clear_on_submit=True):
        _nl_col1, _nl_col2 = st.columns([3, 1])
        with _nl_col1:
            _new_list_name = st.text_input(
                "New list name",
                key="wl_new_list_name",
                placeholder="e.g. Tech Stocks",
                label_visibility="collapsed",
            )
        with _nl_col2:
            _create_clicked = st.form_submit_button(
                "Create", use_container_width=True
            )
    if _create_clicked:
        if _new_list_name.strip():
            if create_watchlist(_new_list_name.strip()):
                st.toast(f"Created list '{_new_list_name}'", icon="✅")
                st.rerun()
            else:
                st.warning(f"A list named '{_new_list_name}' already exists.")

    st.divider()

    st.subheader("Deep Dive Analysis")
    col_asset, col_refresh = st.columns([3, 1])
    with col_asset:
        selected_ticker = st.selectbox(
            "Select Asset for Analysis",
            all_wl_tickers if all_wl_tickers else [""],
        )
    with col_refresh:
        st.write(""); st.write("")
        if st.button("🔄 Force Refresh Data", use_container_width=True):
            fetch_stock_details.clear(); fetch_live_prices.clear()

    timeframes = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
    selected_period = st.radio("Timeframe", timeframes, index=2, horizontal=True)

    col_ta1, col_ta2, col_ta3 = st.columns(3)
    with col_ta1: show_sma = st.checkbox("Overlay 50-Period SMA")
    with col_ta2: show_rsi = st.checkbox("Show 14-Period RSI")
    with col_ta3: show_macd = st.checkbox("Show MACD (12, 26, 9)")

    if selected_ticker:
        with st.spinner(f"Loading {selected_ticker} data..."):
            hist_data, stock_info = fetch_stock_details(selected_ticker, selected_period)
            if not hist_data.empty:
                start_price, end_price = hist_data['Close'].iloc[0], hist_data['Close'].iloc[-1]
                chart_color = '#28a745' if end_price >= start_price else '#dc3545'
                
                active_subplots = sum([show_rsi, show_macd])
                total_rows = 1 + active_subplots
                
                row_heights = [1.0] if total_rows == 1 else ([0.7, 0.3] if total_rows == 2 else [0.6, 0.2, 0.2])
                rsi_row = 2 if show_rsi else None
                macd_row = (3 if total_rows == 3 else 2) if show_macd else None

                fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights) if total_rows > 1 else go.Figure()
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], mode='lines', name=f"{selected_ticker} Price", line=dict(color=chart_color, width=2)), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
                
                if show_sma: fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'].rolling(window=50).mean(), mode='lines', name="50-Period SMA", line=dict(color='#3498db', width=1.5, dash='dot')), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
                if show_rsi:
                    delta = hist_data['Close'].diff()
                    gain, loss = delta.clip(lower=0), -1 * delta.clip(upper=0)
                    rs = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    rsi = 100 - (100 / (1 + rs))
                    fig.add_trace(go.Scatter(x=hist_data.index, y=rsi, mode='lines', name="RSI (14)", line=dict(color='#9b59b6', width=1.5)), row=rsi_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="rgba(220, 53, 69, 0.5)", row=rsi_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="rgba(40, 167, 69, 0.5)", row=rsi_row, col=1)
                    fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
                if show_macd:
                    macd_line = hist_data['Close'].ewm(span=12, adjust=False).mean() - hist_data['Close'].ewm(span=26, adjust=False).mean()
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    macd_hist = macd_line - signal_line
                    hist_colors = ['#28a745' if val >= 0 else '#dc3545' for val in macd_hist]
                    fig.add_trace(go.Bar(x=hist_data.index, y=macd_hist, marker_color=hist_colors, name="MACD Histogram"), row=macd_row, col=1)
                    fig.add_trace(go.Scatter(x=hist_data.index, y=macd_line, mode='lines', name="MACD Line", line=dict(color='#2980b9', width=1.5)), row=macd_row, col=1)
                    fig.add_trace(go.Scatter(x=hist_data.index, y=signal_line, mode='lines', name="Signal Line", line=dict(color='#e67e22', width=1.5)), row=macd_row, col=1)

                dynamic_height = 400 + (200 * active_subplots)
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", showlegend=True, height=dynamic_height, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                st.plotly_chart(fig, use_container_width=True)

                mkt_cap = format_large_number(stock_info.get('marketCap'))
                pe_ratio = round(stock_info.get('trailingPE', 0), 2) if stock_info.get('trailingPE') else "N/A"
                div_yield = stock_info.get('dividendYield')
                div_yield_str = f"{div_yield * 100:.2f}%" if div_yield else "N/A"
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Open", f"${round(hist_data['Open'].iloc[-1], 2)}")
                c2.metric("Mkt Cap", mkt_cap)
                c3.metric("P/E Ratio", pe_ratio)
                c4.metric("Dividend", div_yield_str)
                c1.metric("Low", f"${round(hist_data['Low'].min(), 2)}")
                c2.metric("52-wk High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
                c3.metric("52-wk Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
                c4.metric("High", f"${round(hist_data['High'].max(), 2)}")

            else: st.warning(f"Could not retrieve historical data for {selected_ticker}.")

# ===========================
# TAB: FAVORITES
# ===========================
if _active == "⭐ Favorites":
    st.header("⭐ Favorite Stocks")
    st.caption(
        "A curated, high-attention list — track goals, notes, financials, SEC filings, "
        "and noteworthy articles per stock. Articles you save are tied to the specific "
        "ticker you saved them under."
    )

    favorites = list_favorites()
    fav_tickers = list(favorites.keys())

    _fav_ready = "favorites" in st.session_state._lazy_loaded

    # ---- Summary strip across all favorites ----
    if fav_tickers:
        if not _fav_ready:
            fav_prices = {}
            st.caption(
                "📡 Live prices not loaded yet — click **⚡ Load all live data** on "
                "the 🏠 Dashboard tab to populate this summary."
            )
        else:
            with st.spinner("Fetching live data for your favorites..."):
                fav_prices = fetch_live_prices(fav_tickers)

        summary_rows = []
        for t in fav_tickers:
            fav = favorites[t]
            live = fav_prices.get(t, {}) or {}
            price = live.get("price")
            change = live.get("change")
            goal = fav.get("goal_price")
            if price is not None and goal:
                progress = ((price - goal) / goal) * 100
            else:
                progress = None
            summary_rows.append({
                "Ticker": t,
                "Price": price,
                "Day %": change,
                "Goal": goal,
                "vs Goal %": progress,
                "Notes": (fav.get("notes") or "").splitlines()[0][:80] if fav.get("notes") else "",
            })

        summary_df = pd.DataFrame(summary_rows)
        st.subheader("At-a-glance")
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Day %": st.column_config.NumberColumn(format="%.2f%%"),
                "Goal": st.column_config.NumberColumn(format="$%.2f"),
                "vs Goal %": st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )
    else:
        st.info("No favorites yet. Add one below to start tracking.")

    st.divider()

    # ---- Per-stock deep dive ----
    if fav_tickers:
        focus_ticker = st.selectbox(
            "Focus stock",
            fav_tickers,
            key="fav_focus_select",
        )
        fav = favorites[focus_ticker]

        # --- Header: live price + day change + goal progress ---
        live = fav_prices.get(focus_ticker, {}) or {}
        price = live.get("price")
        change = live.get("change")
        goal = fav.get("goal_price")

        hdr1, hdr2, hdr3, hdr4 = st.columns(4)
        hdr1.metric(
            "Live Price",
            f"${price:.2f}" if price is not None else "N/A",
            delta=f"{change:+.2f}%" if change is not None else None,
        )
        hdr2.metric("Goal Price", f"${goal:.2f}" if goal else "— not set —")
        if price is not None and goal:
            delta = price - goal
            pct = (delta / goal) * 100
            hdr3.metric("Δ vs Goal", f"${delta:+.2f}", delta=f"{pct:+.2f}%")
        else:
            hdr3.metric("Δ vs Goal", "—")
        hdr4.metric(
            "Added",
            time.strftime("%Y-%m-%d", time.localtime(fav.get("added_at") or 0))
            if fav.get("added_at") else "—",
        )

        # --- Notes / goal / position controls ---
        # Wrapped in st.form so pressing "Save changes" commits ALL widget
        # values atomically. Without a form, st.number_input only commits on
        # blur — so clicking Save right after typing a goal (without Tab/Enter
        # first) would miss the new value and silently save nothing.
        with st.expander("📝 Notes, goal, and position", expanded=True):
            with st.form(key=f"fav_meta_form_{focus_ticker}", clear_on_submit=False):
                col_n, col_g = st.columns([3, 1])
                with col_n:
                    new_notes = st.text_area(
                        "Notes",
                        value=fav.get("notes") or "",
                        height=140,
                        key=f"fav_notes_{focus_ticker}",
                        help="Your thesis, watch criteria, or anything else worth remembering.",
                    )
                with col_g:
                    new_goal = st.number_input(
                        "Goal price ($)",
                        min_value=0.0,
                        value=float(fav.get("goal_price") or 0.0),
                        step=0.01,
                        format="%.2f",
                        key=f"fav_goal_{focus_ticker}",
                        help="Accepts dollars and cents (e.g. 247.85). "
                             "Set to any value > 0 to save a goal. Leave at 0 to skip. "
                             "Tick 'Clear goal' to explicitly remove an existing goal.",
                    )
                    clear_goal = st.checkbox(
                        "Clear goal",
                        value=False,
                        key=f"fav_clear_goal_{focus_ticker}",
                    )
                new_position = st.text_input(
                    "Position note (e.g., '10 shares @ $150, added 2024-11-01')",
                    value=fav.get("position_note") or "",
                    key=f"fav_pos_{focus_ticker}",
                )
                submitted = st.form_submit_button("💾 Save changes")

            if submitted:
                kwargs = {
                    "notes": new_notes,
                    "position_note": new_position,
                }
                # Only touch goal_price when the user meaningfully asked to.
                # - explicit clear checkbox → clear
                # - new_goal > 0            → set to that value
                # - otherwise               → leave existing goal alone
                if clear_goal:
                    kwargs["clear_goal"] = True
                elif new_goal > 0:
                    kwargs["goal_price"] = float(new_goal)

                update_favorite(focus_ticker, **kwargs)

                saved_bits = ["notes", "position"]
                if clear_goal:
                    saved_bits.append("goal cleared")
                elif new_goal > 0:
                    saved_bits.append(f"goal=${new_goal:.2f}")
                st.toast(
                    f"Saved {focus_ticker}: " + ", ".join(saved_bits),
                    icon="💾",
                )
                st.rerun()

        # --- Financial highlights ---
        st.subheader("💰 Financial Highlights")
        with st.spinner(f"Loading financials for {focus_ticker}..."):
            try:
                highlights = fetch_financial_highlights(focus_ticker)
            except Exception as e:
                highlights = {}
                st.warning(f"Could not fetch financial highlights: {e}")

        if highlights.get("long_name"):
            st.caption(
                f"**{highlights['long_name']}** · {highlights.get('sector') or '—'} "
                f"/ {highlights.get('industry') or '—'} · {highlights.get('currency') or ''}"
            )

        cur = highlights.get("currency") or "USD"
        fh1, fh2, fh3, fh4 = st.columns(4)
        fh1.metric("Revenue (TTM/FY)", format_large_number(highlights.get("revenue")))
        fh2.metric("Net Income", format_large_number(highlights.get("net_income")))
        fh3.metric("Free Cash Flow", format_large_number(highlights.get("fcf")))
        fh4.metric(
            "Profit Margin",
            f"{highlights['profit_margin']:.2f}%" if highlights.get("profit_margin") is not None else "N/A",
        )

        fh5, fh6, fh7, fh8 = st.columns(4)
        fh5.metric("Total Debt", format_large_number(highlights.get("total_debt")))
        fh6.metric("Cash", format_large_number(highlights.get("cash")))
        fh7.metric("Net Debt", format_large_number(highlights.get("net_debt")))
        fh8.metric("Market Cap", format_large_number(highlights.get("market_cap")))

        fh9, fh10, fh11, fh12 = st.columns(4)
        fh9.metric("P/E (Trailing)", f"{highlights['pe_trailing']:.2f}" if highlights.get("pe_trailing") else "N/A")
        fh10.metric("P/E (Forward)", f"{highlights['pe_forward']:.2f}" if highlights.get("pe_forward") else "N/A")
        fh11.metric("EPS (TTM)", f"${highlights['eps_trailing']:.2f}" if highlights.get("eps_trailing") else "N/A")
        fh12.metric("Div Yield", f"{highlights['dividend_yield']:.2f}%" if highlights.get("dividend_yield") else "N/A")

        with st.expander("📄 Full Statements (Income / Balance / Cash Flow)"):
            try:
                statements = fetch_financial_statements(focus_ticker)
            except Exception as e:
                statements = {"income": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame()}
                st.warning(f"Could not fetch statements: {e}")

            stmt_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            for stmt_tab, key, label in zip(
                stmt_tabs,
                ["income", "balance_sheet", "cashflow"],
                ["Income Statement", "Balance Sheet", "Cash Flow"],
            ):
                with stmt_tab:
                    df_stmt = statements.get(key, pd.DataFrame())
                    if df_stmt is None or df_stmt.empty:
                        st.info(f"No {label.lower()} available from yfinance.")
                    else:
                        # Columns are period ends; format as dates and show most recent first
                        try:
                            df_show = df_stmt.copy()
                            df_show.columns = [
                                c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                                for c in df_show.columns
                            ]
                        except Exception:
                            df_show = df_stmt
                        st.dataframe(df_show, use_container_width=True)

        # --- SEC filings ---
        st.subheader("📋 Recent SEC Filings")
        with st.spinner(f"Loading SEC filings for {focus_ticker}..."):
            filings = fetch_recent_sec_filings(focus_ticker, n=10)
        if not filings:
            st.info(
                "No recent SEC filings found (or ticker not in SEC EDGAR — non-US "
                "listings and ETFs typically won't appear)."
            )
        else:
            filings_df = pd.DataFrame([
                {"Form": f["form"], "Date": f["date"], "Link": f["url"]} for f in filings
            ])
            st.dataframe(
                filings_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Link", display_text="📄 Open"),
                },
            )
            with st.expander("🔎 Fetch text of a specific filing type"):
                form_options = sorted({f["form"] for f in filings})
                if form_options:
                    pick_form = st.selectbox(
                        "Form type",
                        form_options,
                        key=f"fav_form_pick_{focus_ticker}",
                    )
                    if st.button("Fetch most recent text", key=f"fav_fetch_filing_{focus_ticker}"):
                        with st.spinner(f"Fetching {pick_form}..."):
                            text, url_or_err = fetch_sec_filing(focus_ticker, form_type=pick_form)
                        if text:
                            st.success(f"Fetched {pick_form} — {url_or_err}")
                            st.text_area(
                                f"{pick_form} text (first 10k chars)",
                                value=text[:10000],
                                height=300,
                                key=f"fav_filing_text_{focus_ticker}_{pick_form}",
                            )
                        else:
                            st.warning(url_or_err)

        # --- News: auto feed + saved articles ---
        st.subheader("📰 News")
        news_col1, news_col2 = st.tabs(["🌐 Auto feed (multi-source)", "⭐ Saved articles"])

        with news_col1:
            refresh_news = st.button("🔄 Refresh feed", key=f"fav_news_refresh_{focus_ticker}")
            if refresh_news:
                fetch_all_news.clear()
            with st.spinner("Fetching news from all configured sources..."):
                try:
                    articles = fetch_all_news(focus_ticker)
                except Exception as e:
                    articles = []
                    st.warning(f"News fetch error: {e}")

            if not articles:
                st.info("No news articles found for this ticker.")
            else:
                st.caption(f"{len(articles)} article(s), newest first. Sources include Yahoo, "
                           f"Google News (incl. WSJ/FT/Bloomberg/Reuters site filters), Seeking Alpha, "
                           f"and any configured Finnhub/NewsAPI/MarketAux keys.")
                for art in articles[:40]:
                    when = (
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(art["published_ts"]))
                        if art.get("published_ts") else art.get("time") or "Recent"
                    )
                    c_art, c_btn = st.columns([5, 1])
                    with c_art:
                        st.markdown(
                            f"**[{art['title']}]({art['url']})**  \n"
                            f"*{art.get('source') or 'Source'}* · {when}"
                        )
                        if art.get("summary"):
                            st.caption(art["summary"][:280] + ("…" if len(art["summary"]) > 280 else ""))
                    with c_btn:
                        if st.button(
                            "⭐ Save",
                            key=f"fav_save_feed_{focus_ticker}_{hash(art['url'])}",
                            help="Save this article to this ticker's noteworthy list",
                        ):
                            new_id = save_article(
                                ticker=focus_ticker,
                                url=art["url"],
                                title=art.get("title"),
                                source=art.get("source"),
                                note="",
                                published_at=art.get("published_ts") or None,
                            )
                            if new_id:
                                st.toast(
                                    f"Saved to {focus_ticker}'s noteworthy list",
                                    icon="⭐",
                                )
                            else:
                                st.toast("Already saved for this ticker", icon="ℹ️")

        with news_col2:
            st.caption(
                f"Articles below are saved **specifically to {focus_ticker}**. "
                "Paste any URL (WSJ+, FT, blog, analyst report, etc.) — we'll pull "
                "the title/source from the page's public Open Graph metadata "
                "(works even for paywalled articles, since OG tags are public)."
            )

            with st.form(key=f"fav_url_form_{focus_ticker}", clear_on_submit=True):
                paste_url = st.text_input(
                    "URL to save",
                    placeholder="https://www.wsj.com/articles/...",
                    key=f"fav_paste_url_{focus_ticker}",
                )
                paste_note = st.text_area(
                    "Optional note (why is this noteworthy?)",
                    key=f"fav_paste_note_{focus_ticker}",
                    height=80,
                )
                submit_url = st.form_submit_button("💾 Save URL to this stock")
                if submit_url and paste_url.strip():
                    with st.spinner("Fetching article metadata..."):
                        meta = fetch_url_metadata(paste_url.strip())
                    new_id = save_article(
                        ticker=focus_ticker,
                        url=paste_url.strip(),
                        title=meta.get("title") or paste_url.strip(),
                        source=meta.get("source") or "",
                        note=paste_note or "",
                        published_at=meta.get("published_ts") or None,
                    )
                    if new_id:
                        st.success(
                            f"Saved to {focus_ticker}: "
                            f"{meta.get('title') or paste_url.strip()}"
                        )
                    else:
                        st.info(f"This URL is already saved for {focus_ticker}.")
                    st.rerun()

            saved = list_saved_articles(focus_ticker)
            st.write(f"**{len(saved)} saved article(s)** for `{focus_ticker}`")
            for art in saved:
                pub_ts = art.get("published_at") or art.get("saved_at")
                when = time.strftime("%Y-%m-%d", time.localtime(pub_ts)) if pub_ts else "—"
                with st.container(border=True):
                    c_info, c_note, c_act = st.columns([4, 3, 1])
                    with c_info:
                        st.markdown(
                            f"**[{art.get('title') or art['url']}]({art['url']})**  \n"
                            f"*{art.get('source') or 'Source'}* · {when} · "
                            f"saved {time.strftime('%Y-%m-%d', time.localtime(art['saved_at']))}"
                        )
                    with c_note:
                        new_note = st.text_input(
                            "Note",
                            value=art.get("note") or "",
                            key=f"fav_saved_note_{art['id']}",
                            label_visibility="collapsed",
                            placeholder="Your note...",
                        )
                        if new_note != (art.get("note") or ""):
                            if st.button("Save note", key=f"fav_saved_note_btn_{art['id']}"):
                                update_saved_article_note(art["id"], new_note)
                                st.toast("Note updated", icon="✏️")
                                st.rerun()
                    with c_act:
                        if st.button("🗑️", key=f"fav_saved_del_{art['id']}", help="Remove"):
                            delete_saved_article(art["id"])
                            st.toast("Removed", icon="🗑️")
                            st.rerun()

    st.divider()

    # ---- Add / Remove favorites ----
    col_add, col_del = st.columns(2)
    with col_add:
        st.subheader("⭐ Add a favorite")
        add_input = st.text_input(
            "Ticker (e.g. AAPL, RY.TO, SHOP)",
            key="fav_add_input",
        )
        if st.button("Add to favorites", key="fav_add_btn", use_container_width=True):
            sym = sanitize_ticker(add_input)
            if sym:
                if add_favorite(sym):
                    st.toast(f"{sym} added to favorites", icon="⭐")
                    st.rerun()
                else:
                    st.warning(f"{sym} is already in your favorites.")

    with col_del:
        st.subheader("🗑️ Remove a favorite")
        if fav_tickers:
            rm_choice = st.selectbox(
                "Select favorite to remove",
                fav_tickers,
                key="fav_rm_select",
            )
            if st.button(
                "Remove (also deletes its saved articles)",
                type="primary",
                key="fav_rm_btn",
                use_container_width=True,
            ):
                remove_favorite(rm_choice)
                st.toast(f"{rm_choice} removed", icon="🗑️")
                st.rerun()
        else:
            st.caption("No favorites to remove.")

# ===========================
# TAB 2: ASSET TRACKER
# ===========================
if _active == "💼 Asset Tracker":
    st.header("Portfolio Asset Tracker")
    with st.expander("Creating and Managing Portfolios", expanded=not bool(portfolios)):
        col_p1, col_p2 = st.columns([3, 1])
        new_portfolio_name = col_p1.text_input("New Portfolio Name")
        if col_p2.button("Create Portfolio"):
            if new_portfolio_name and new_portfolio_name not in portfolios:
                portfolios[new_portfolio_name] = {}
                app_data["portfolios"] = portfolios
                save_data(app_data); st.toast(f"Portfolio '{new_portfolio_name}' created!", icon="🎉")
            elif new_portfolio_name in portfolios: st.warning("Portfolio already exists.")
    st.divider()

    portfolio_names = list(portfolios.keys())
    if not portfolio_names:
        st.info("Please create a portfolio to get started.")
        st.stop()

    selected_portfolio = st.selectbox("Select Portfolio to View", ["All Portfolios"] + portfolio_names)

    current_holdings = {}
    if selected_portfolio == "All Portfolios":
        for p_name in portfolios:
            for ticker, data in portfolios[p_name].items():
                if ticker in current_holdings:
                    total_qty = current_holdings[ticker]['quantity'] + data['quantity']
                    total_cost = (current_holdings[ticker]['quantity'] * current_holdings[ticker]['average_cost']) + (data['quantity'] * data['average_cost'])
                    current_holdings[ticker]['average_cost'] = total_cost / total_qty
                    current_holdings[ticker]['quantity'] = total_qty
                else: current_holdings[ticker] = data.copy()
    else: current_holdings = portfolios[selected_portfolio]

    if selected_portfolio != "All Portfolios":
        col_add_stock, col_sell_stock, col_manage_cash, col_delete = st.columns(4)
        with col_add_stock:
            with st.expander(f"Add Stock", expanded=False):
                with st.form("add_asset_form", clear_on_submit=True):
                    asset_ticker = sanitize_ticker(st.text_input("Ticker").upper())
                    asset_qty = st.number_input("Quantity", min_value=0.01, step=0.01)
                    asset_cost = st.number_input("Avg. Cost ($)", min_value=0.0, step=0.01)
                    if st.form_submit_button("Buy"):
                        if asset_ticker and asset_ticker != "CASH" and asset_qty > 0:
                            if asset_ticker in portfolios[selected_portfolio]:
                                old_qty = portfolios[selected_portfolio][asset_ticker]['quantity']
                                old_cost = portfolios[selected_portfolio][asset_ticker]['average_cost']
                                portfolios[selected_portfolio][asset_ticker] = {"quantity": old_qty + asset_qty, "average_cost": ((old_qty * old_cost) + (asset_qty * asset_cost)) / (old_qty + asset_qty)}
                            else: portfolios[selected_portfolio][asset_ticker] = {"quantity": asset_qty, "average_cost": asset_cost}
                            app_data["portfolios"] = portfolios
                            save_data(app_data)
                            log_transaction(selected_portfolio, asset_ticker, "BUY", asset_qty, asset_cost, cost_basis=asset_cost)
                            st.toast(f"Added {asset_ticker}", icon="💰")

        with col_sell_stock:
            with st.expander(f"Sell Stock", expanded=False):
                sellable_assets = [t for t in portfolios[selected_portfolio].keys() if t != "CASH"]
                if sellable_assets:
                    with st.form("sell_asset_form", clear_on_submit=True):
                        sell_ticker = st.selectbox("Asset", sellable_assets)
                        current_qty = portfolios[selected_portfolio].get(sell_ticker, {}).get("quantity", 0.0)
                        sell_qty = st.number_input("Qty to Sell", min_value=0.01, max_value=float(current_qty), step=0.01)
                        sell_price = st.number_input("Sale Price ($)", min_value=0.0, step=0.01)
                        if st.form_submit_button("Execute Sale"):
                            if sell_qty > 0 and sell_price >= 0:
                                avg_cost_at_sale = portfolios[selected_portfolio][sell_ticker]["average_cost"]
                                proceeds = sell_qty * sell_price
                                portfolios[selected_portfolio][sell_ticker]["quantity"] -= sell_qty
                                if portfolios[selected_portfolio][sell_ticker]["quantity"] <= 0.0001: del portfolios[selected_portfolio][sell_ticker]
                                current_cash = portfolios[selected_portfolio].get("CASH", {"quantity": 0.0, "average_cost": 1.0})
                                portfolios[selected_portfolio]["CASH"] = {"quantity": current_cash["quantity"] + proceeds, "average_cost": 1.0}
                                app_data["portfolios"] = portfolios
                                save_data(app_data)
                                log_transaction(selected_portfolio, sell_ticker, "SELL", sell_qty, sell_price, cost_basis=avg_cost_at_sale)
                                st.toast(f"Sold {sell_ticker}", icon="🤝")
                else: st.info("No stocks to sell.")

        with col_manage_cash:
            with st.expander(f"Manage Cash", expanded=False):
                with st.form("manage_cash_form", clear_on_submit=True):
                    cash_action = st.radio("Action", ["Deposit", "Withdraw"], horizontal=True)
                    cash_amount = st.number_input("Amount ($)", min_value=0.01, step=100.0)
                    if st.form_submit_button("Update Cash"):
                        current_cash_data = portfolios[selected_portfolio].get("CASH", {"quantity": 0.0, "average_cost": 1.0})
                        new_qty = current_cash_data["quantity"] + cash_amount if cash_action == "Deposit" else max(0.0, current_cash_data["quantity"] - cash_amount)
                        portfolios[selected_portfolio]["CASH"] = {"quantity": new_qty, "average_cost": 1.0}
                        app_data["portfolios"] = portfolios
                        save_data(app_data)

        with col_delete:
             with st.expander(f"Delete Asset", expanded=False):
                 assets_to_delete = list(portfolios[selected_portfolio].keys())
                 if assets_to_delete:
                     del_asset = st.selectbox("Select Asset to Delete", assets_to_delete)
                     if st.button("Delete Permanently", type="primary"):
                         del portfolios[selected_portfolio][del_asset]
                         app_data["portfolios"] = portfolios
                         save_data(app_data); st.toast(f"Deleted {del_asset}", icon="🗑️")
                 else: st.info("No assets.")

    if current_holdings:
        holding_tickers = list(current_holdings.keys())
        _at_ready = "asset_tracker" in st.session_state._lazy_loaded
        if not _at_ready:
            st.caption(
                "📡 Live prices not loaded yet — click **⚡ Load all live data** on "
                "the 🏠 Dashboard tab to populate current values."
            )
            holding_prices = {}
        else:
            with st.spinner("Fetching live prices for holdings..."):
                holding_prices = fetch_live_prices(holding_tickers)
        
        holdings_data = []
        total_portfolio_value = 0
        total_portfolio_cost = 0

        for ticker, data in current_holdings.items():
            qty, avg_cost = data['quantity'], data['average_cost']
            if ticker == "CASH":
                current_price, market_value, total_cost, profit_loss, pl_percent = 1.00, qty, qty, 0.0, 0.0
            else:
                current_price = holding_prices.get(ticker, {}).get('price', 0) or 0
                market_value, total_cost = qty * current_price, qty * avg_cost
                profit_loss = market_value - total_cost
                pl_percent = (profit_loss / total_cost * 100) if total_cost > 0 else 0

            total_portfolio_value += market_value
            total_portfolio_cost += total_cost
            holdings_data.append({"Ticker": ticker, "Quantity": qty, "Avg. Cost": avg_cost, "Current Price": current_price, "Market Value": market_value, "Profit/Loss ($)": profit_loss, "Profit/Loss (%)": pl_percent / 100})
        
        holdings_df = pd.DataFrame(holdings_data)
        total_pl = total_portfolio_value - total_portfolio_cost
        total_pl_pct = (total_pl / total_portfolio_cost * 100) if total_portfolio_cost > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
        m2.metric("Total Gain/Loss ($)", f"${total_pl:,.2f}", delta=f"${total_pl:,.2f}")
        m3.metric("Total Gain/Loss (%)", f"{total_pl_pct:.2f}%", delta=f"{total_pl_pct:.2f}%")
        st.divider()

        if selected_portfolio == "All Portfolios":
            st.subheader("Aggregated Holdings (Read-Only)")
            st.dataframe(holdings_df, use_container_width=True, hide_index=True, column_config={"Avg. Cost": st.column_config.NumberColumn(format="$%.2f"), "Current Price": st.column_config.NumberColumn(format="$%.2f"), "Market Value": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%")})
        else:
            st.subheader(f"Manage Holdings: {selected_portfolio}")
            edited_holdings_df = st.data_editor(holdings_df, disabled=["Ticker", "Current Price", "Market Value", "Profit/Loss ($)", "Profit/Loss (%)"], hide_index=True, num_rows="dynamic", use_container_width=True, column_config={"Quantity": st.column_config.NumberColumn(step=0.01), "Avg. Cost": st.column_config.NumberColumn(format="$%.2f", step=0.01), "Current Price": st.column_config.NumberColumn(format="$%.2f"), "Market Value": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%")})
            if not edited_holdings_df.equals(holdings_df):
                new_portfolio_data = {}
                for index, row in edited_holdings_df.iterrows():
                    ticker = str(row['Ticker']).upper().strip()
                    if ticker and ticker.lower() not in ['nan', 'none'] and float(row['Quantity']) > 0: 
                        new_portfolio_data[ticker] = {"quantity": float(row['Quantity']), "average_cost": float(row['Avg. Cost'])}
                portfolios[selected_portfolio] = new_portfolio_data
                app_data["portfolios"] = portfolios
                save_data(app_data)

        if not holdings_df.empty and total_portfolio_value > 0:
            st.divider()
            st.subheader("Portfolio Visualizations")
            tab_pie, tab_bar, tab_tree = st.tabs(["Allocation (Pie)", "Performance (Bar)", "Heatmap (Treemap)"])
            plot_df = holdings_df[holdings_df['Market Value'] > 0]
            with tab_pie:
                if not plot_df.empty:
                    fig_pie = px.pie(plot_df, values='Market Value', names='Ticker', hole=0.4)
                    fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, use_container_width=True)
            with tab_bar:
                fig_bar = px.bar(holdings_df, x='Ticker', y='Profit/Loss ($)', title="Absolute Profit/Loss by Asset", color='Profit/Loss ($)', color_continuous_scale=['#dc3545', '#28a745'])
                fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
            with tab_tree:
                if not plot_df.empty:
                    fig_tree = px.treemap(plot_df, path=[px.Constant("Portfolio"), 'Ticker'], values='Market Value', color='Profit/Loss (%)', color_continuous_scale=['#dc3545', '#28a745'], color_continuous_midpoint=0)
                    fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                    st.plotly_chart(fig_tree, use_container_width=True)
    else: st.info("No assets in this portfolio yet.")

# ===========================
# TAB HISTORY: TRADE LEDGER & REALIZED P&L
# ===========================
if _active == "📒 History":
    st.header("📒 Trade History & Realized P&L")
    st.caption("Every buy and sell you make is recorded here. Realized P&L is calculated from sales only — unrealized gains on open positions are shown in the Asset Tracker.")

    # --- IMPORT HISTORICAL TRADES ---
    with st.expander("📥 Import Historical Trades (CSV)", expanded=False):
        st.markdown(
            "Backfill trades from before the SQLite migration — or from any broker export.\n\n"
            "**Required columns:** `date`, `portfolio`, `ticker`, `action` (BUY or SELL), `quantity`, `price`.\n"
            "**Optional:** `cost_basis` — only meaningful for SELL rows. If omitted on a SELL, that row won't contribute to realized P&L (but still appears in the ledger)."
        )

        template_csv = (
            "date,portfolio,ticker,action,quantity,price,cost_basis\n"
            "2024-03-15,RobinHood,AAPL,BUY,10,172.50,\n"
            "2024-08-02,RobinHood,AAPL,SELL,4,225.00,172.50\n"
        )
        st.download_button(
            "Download CSV template",
            data=template_csv,
            file_name="trades_template.csv",
            mime="text/csv",
            use_container_width=False,
        )

        uploaded_csv = st.file_uploader("Upload your CSV", type=["csv"], key="tx_import")
        if uploaded_csv is not None:
            try:
                import_df = pd.read_csv(uploaded_csv)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                import_df = None

            if import_df is not None:
                required = {"date", "portfolio", "ticker", "action", "quantity", "price"}
                missing = required - set(c.lower() for c in import_df.columns)
                import_df.columns = [c.lower() for c in import_df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
                else:
                    st.write("**Preview** (first 10 rows):")
                    st.dataframe(import_df.head(10), use_container_width=True, hide_index=True)

                    if st.button("Import Trades", type="primary"):
                        rows = []
                        parse_errors = 0
                        for _, r in import_df.iterrows():
                            try:
                                ts = int(pd.to_datetime(r["date"]).timestamp())
                                rows.append({
                                    "ts": ts,
                                    "portfolio_name": str(r["portfolio"]).strip(),
                                    "ticker": str(r["ticker"]).strip().upper(),
                                    "action": str(r["action"]).strip().upper(),
                                    "quantity": float(r["quantity"]),
                                    "price": float(r["price"]),
                                    "cost_basis": r.get("cost_basis") if "cost_basis" in import_df.columns else None,
                                })
                            except Exception:
                                parse_errors += 1

                        result = import_transactions(rows)
                        result["errors"] += parse_errors
                        st.success(
                            f"✅ Imported {result['added']} trades. "
                            f"Skipped {result['skipped']} duplicates. "
                            f"{result['errors']} rows had errors."
                        )

    st.divider()

    all_tx = fetch_transactions()

    if not all_tx:
        st.info("No trades logged yet. Buy or sell a position from the Asset Tracker and it will appear here.")
    else:
        tx_df = pd.DataFrame(all_tx)
        tx_df["Date"] = pd.to_datetime(tx_df["ts"], unit="s")
        tx_df["Year"] = tx_df["Date"].dt.year
        tx_df["Proceeds"] = tx_df["quantity"] * tx_df["price"]
        tx_df["Realized P&L"] = tx_df.apply(
            lambda r: (r["price"] - r["cost_basis"]) * r["quantity"]
            if r["action"] == "SELL" and pd.notna(r["cost_basis"])
            else 0.0,
            axis=1,
        )

        sells = tx_df[tx_df["action"] == "SELL"]
        total_realized = sells["Realized P&L"].sum()
        ytd = sells[sells["Year"] == pd.Timestamp.now().year]["Realized P&L"].sum()
        trade_count = len(tx_df)
        win_count = int((sells["Realized P&L"] > 0).sum())
        loss_count = int((sells["Realized P&L"] < 0).sum())
        win_rate = (win_count / len(sells) * 100) if len(sells) else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Realized P&L", f"${total_realized:,.2f}")
        m2.metric(f"{pd.Timestamp.now().year} Realized P&L", f"${ytd:,.2f}")
        m3.metric("Total Trades", f"{trade_count}")
        m4.metric("Win Rate", f"{win_rate:.0f}%", help=f"{win_count} wins / {loss_count} losses")

        st.divider()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            portfolio_filter = st.selectbox(
                "Filter by Portfolio",
                ["All"] + sorted(tx_df["portfolio_name"].unique().tolist()),
            )
        with col_f2:
            ticker_filter = st.selectbox(
                "Filter by Ticker",
                ["All"] + sorted(tx_df["ticker"].unique().tolist()),
            )

        view_df = tx_df.copy()
        if portfolio_filter != "All":
            view_df = view_df[view_df["portfolio_name"] == portfolio_filter]
        if ticker_filter != "All":
            view_df = view_df[view_df["ticker"] == ticker_filter]

        st.subheader("Transaction Log")
        display_df = view_df[[
            "Date", "portfolio_name", "ticker", "action", "quantity",
            "price", "cost_basis", "Proceeds", "Realized P&L",
        ]].rename(columns={
            "portfolio_name": "Portfolio",
            "ticker": "Ticker",
            "action": "Action",
            "quantity": "Qty",
            "price": "Price",
            "cost_basis": "Cost Basis",
        })
        st.dataframe(
            display_df.style.format({
                "Qty": "{:.4f}",
                "Price": "${:,.2f}",
                "Cost Basis": "${:,.2f}",
                "Proceeds": "${:,.2f}",
                "Realized P&L": "${:,.2f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

        if not sells.empty:
            st.subheader("Realized P&L by Ticker")
            by_ticker = (
                sells.groupby("ticker")["Realized P&L"].sum().reset_index()
                .sort_values("Realized P&L", ascending=False)
            )
            fig_ticker = px.bar(
                by_ticker, x="ticker", y="Realized P&L",
                color="Realized P&L",
                color_continuous_scale=["#ff4b4b", "#cccccc", "#28a745"],
                color_continuous_midpoint=0,
            )
            fig_ticker.update_layout(height=320, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_ticker, use_container_width=True)

            st.subheader("Realized P&L by Year")
            by_year = sells.groupby("Year")["Realized P&L"].sum().reset_index()
            fig_year = px.bar(by_year, x="Year", y="Realized P&L")
            fig_year.update_layout(height=280, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_year, use_container_width=True)

# ===========================
# TAB RISK: PORTFOLIO RISK DASHBOARD
# ===========================
if _active == "🛡️ Risk":
    st.header("🛡️ Portfolio Risk Dashboard")
    st.caption(
        "VaR, drawdown, correlation, beta, and factor exposure on your "
        "live holdings. Uses daily adjusted closes from yfinance — "
        "computed client-side, cached for an hour."
    )

    _risk_portfolios = list(portfolios.keys())
    if not _risk_portfolios:
        st.info("Create a portfolio in the Asset Tracker tab to unlock this module.")
    else:
        r_col1, r_col2, r_col3, r_col4 = st.columns([2, 1, 1, 1])
        with r_col1:
            risk_portfolio = st.selectbox(
                "Portfolio",
                ["All Portfolios"] + _risk_portfolios,
                key="risk_portfolio",
            )
        with r_col2:
            risk_period = st.selectbox(
                "Lookback", ["1y", "2y", "5y"], index=1, key="risk_period",
            )
        with r_col3:
            risk_conf = st.selectbox(
                "VaR Confidence", ["95%", "99%"], index=0, key="risk_conf",
            )
        with r_col4:
            risk_bench = st.selectbox(
                "Benchmark", ["SPY", "QQQ", "IWM", "ACWI"], index=0, key="risk_bench",
            )

        # Assemble holdings.
        if risk_portfolio == "All Portfolios":
            agg = {}
            for _p, _h in portfolios.items():
                for _t, _pos in _h.items():
                    if _t in agg:
                        total_qty = agg[_t]["quantity"] + _pos["quantity"]
                        total_cost = (
                            agg[_t]["quantity"] * agg[_t]["average_cost"]
                            + _pos["quantity"] * _pos["average_cost"]
                        )
                        agg[_t] = {
                            "quantity": total_qty,
                            "average_cost": total_cost / total_qty if total_qty else 0,
                        }
                    else:
                        agg[_t] = dict(_pos)
            risk_holdings = agg
        else:
            risk_holdings = portfolios[risk_portfolio]

        if not risk_holdings or all(t.upper() == "CASH" for t in risk_holdings):
            st.info("Portfolio has no risk-bearing positions.")
        elif st.button("Run Risk Analysis", type="primary", use_container_width=True):
            with st.spinner("Fetching price history & crunching risk metrics..."):
                _lp_tickers = [t for t in risk_holdings if t.upper() != "CASH"]
                _lp = fetch_live_prices(_lp_tickers)
                _conf = 0.95 if risk_conf == "95%" else 0.99
                report = _portfolio_risk_report(
                    risk_holdings,
                    _lp,
                    period=risk_period,
                    var_confidence=_conf,
                    benchmark=risk_bench,
                )

            if report.get("error"):
                st.error(report["error"])
            else:
                # ----- Headline metrics -----
                st.subheader("Headline Metrics")
                h1, h2, h3, h4 = st.columns(4)
                h1.metric(
                    "Ann. Return", f"{report['ann_return']*100:,.2f}%"
                    if pd.notna(report['ann_return']) else "—",
                )
                h2.metric(
                    "Ann. Volatility", f"{report['ann_volatility']*100:,.2f}%"
                    if pd.notna(report['ann_volatility']) else "—",
                )
                h3.metric(
                    "Sharpe (rf=4%)", f"{report['sharpe']:.2f}"
                    if pd.notna(report['sharpe']) else "—",
                )
                h4.metric(
                    "Sortino", f"{report['sortino']:.2f}"
                    if pd.notna(report['sortino']) else "—",
                )

                v1, v2, v3, v4 = st.columns(4)
                v1.metric(
                    f"1-Day VaR ({risk_conf}, hist.)",
                    f"{report['hist_var_1d']*100:,.2f}%"
                    if pd.notna(report['hist_var_1d']) else "—",
                    help="Largest expected daily loss at the stated confidence "
                         "level, based on observed history.",
                )
                v2.metric(
                    f"1-Day CVaR ({risk_conf})",
                    f"{report['hist_cvar_1d']*100:,.2f}%"
                    if pd.notna(report['hist_cvar_1d']) else "—",
                    help="Expected loss conditional on exceeding the VaR "
                         "threshold — the 'how bad is bad' number.",
                )
                v3.metric(
                    "Max Drawdown",
                    f"{report['max_drawdown']*100:,.2f}%"
                    if pd.notna(report['max_drawdown']) else "—",
                    help=f"Duration: {report.get('dd_days', 0)} days peak-to-trough.",
                )
                v4.metric(
                    f"Beta vs {risk_bench}",
                    f"{report['beta']:.2f}"
                    if pd.notna(report['beta']) else "—",
                    help=f"R²={report['r_squared']:.2f}" if pd.notna(report['r_squared']) else "",
                )

                # ----- Parametric cross-check callout -----
                _hv = report["hist_var_1d"]
                _pv = report["param_var_1d"]
                if pd.notna(_hv) and pd.notna(_pv):
                    delta = (_pv - _hv) / _hv if _hv > 0 else 0
                    if abs(delta) > 0.30:
                        st.warning(
                            f"Parametric VaR ({_pv*100:.2f}%) diverges from "
                            f"historical ({_hv*100:.2f}%) by {delta*100:+.0f}% — "
                            "returns likely non-Gaussian (fat tails or skew)."
                        )

                st.divider()

                # ----- Equity curve + drawdown -----
                st.subheader("Portfolio Equity Curve & Drawdown")
                port_r = report["portfolio_returns"]
                equity = (1 + port_r).cumprod() if not port_r.empty else port_r
                # Use exp(cumsum) since we compute log returns — more accurate.
                import numpy as _np
                equity = _np.exp(port_r.cumsum())
                roll_max = equity.cummax()
                dd_series = (equity / roll_max - 1.0) * 100

                fig_eq = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    row_heights=[0.65, 0.35],
                    subplot_titles=("Cumulative Return (log-compounded)", "Drawdown (%)"),
                )
                fig_eq.add_trace(
                    go.Scatter(
                        x=equity.index, y=(equity - 1) * 100,
                        mode="lines", name="Portfolio",
                        line=dict(color="#2980b9", width=2),
                    ),
                    row=1, col=1,
                )
                fig_eq.add_trace(
                    go.Scatter(
                        x=dd_series.index, y=dd_series,
                        mode="lines", name="Drawdown",
                        line=dict(color="#dc3545", width=1.5),
                        fill="tozeroy", fillcolor="rgba(220,53,69,0.25)",
                    ),
                    row=2, col=1,
                )
                if report.get("dd_peak") and report.get("dd_trough"):
                    fig_eq.add_vline(
                        x=report["dd_trough"], line_dash="dash",
                        line_color="rgba(220,53,69,0.5)", row=2, col=1,
                    )
                fig_eq.update_layout(
                    height=520, margin=dict(l=0, r=0, t=40, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, hovermode="x unified",
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                st.divider()

                # ----- Per-position risk contribution -----
                st.subheader("Per-Position Risk Contribution")
                st.caption(
                    "Marginal Contribution to Risk = weight × cov(asset, port) / var(port). "
                    "A position with weight 10% but risk-contribution 25% is a concentrated "
                    "volatility source."
                )
                _pp = report["per_position"].copy()
                st.dataframe(
                    _pp.style.format({
                        "Weight": "{:.2%}",
                        "Vol (ann.)": "{:.2%}",
                        "Beta": "{:.2f}",
                        "Contribution to Risk": "{:.2%}",
                    }, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )

                # Concentration line
                _c = report["concentration"]
                if pd.notna(_c.get("hhi")):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Positions", f"{_c['n_positions']}")
                    c2.metric(
                        "Effective N (1/HHI)", f"{_c['effective_n']:.1f}",
                        help="How many equally-weighted names this portfolio "
                             "diversifies LIKE. Much smaller than position count = "
                             "concentrated.",
                    )
                    c3.metric(
                        "Largest Weight", f"{_c['top']*100:.1f}%",
                        help="Single-name concentration — a red flag above ~20-25%.",
                    )

                st.divider()

                # ----- Correlation heatmap -----
                corr = report.get("correlation")
                if corr is not None and not corr.empty and corr.shape[0] >= 2:
                    st.subheader("Holdings Correlation Matrix")
                    st.caption(
                        "Daily-return correlations. Highly correlated clusters "
                        "(>0.7) hint that two positions are really the same bet."
                    )
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=list(corr.columns), y=list(corr.index),
                        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                        text=corr.round(2).values,
                        texttemplate="%{text}", textfont=dict(size=10),
                    ))
                    fig_corr.update_layout(
                        height=max(350, 30 * len(corr) + 150),
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                st.divider()

                # ----- Rolling beta -----
                roll_b = report.get("rolling_beta")
                if roll_b is not None and not roll_b.empty:
                    st.subheader(f"Rolling 63-Day Beta vs {risk_bench}")
                    fig_rb = go.Figure()
                    fig_rb.add_trace(go.Scatter(
                        x=roll_b.index, y=roll_b.values,
                        mode="lines", line=dict(color="#8e44ad", width=2),
                        name="Rolling Beta",
                    ))
                    fig_rb.add_hline(y=1.0, line_dash="dash",
                                     line_color="rgba(128,128,128,0.5)")
                    fig_rb.update_layout(
                        height=320, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_rb, use_container_width=True)

                # ----- Factor exposure -----
                fx = report.get("factor_exposure") or {}
                if fx.get("loadings"):
                    st.subheader("Factor Exposure (ETF-proxy decomposition)")
                    st.caption(
                        "Regression of portfolio returns on: "
                        "MKT (SPY), SMB (IWM−SPY), HML (VTV−VUG), "
                        "MOM (MTUM−SPY), RATES (TLT). "
                        "Loadings show how much of daily P&L swings with each factor."
                    )
                    load_df = pd.DataFrame({
                        "Factor": list(fx["loadings"].keys()),
                        "Loading": list(fx["loadings"].values()),
                    })
                    fig_fx = px.bar(
                        load_df, x="Factor", y="Loading", color="Loading",
                        color_continuous_scale=["#dc3545", "#cccccc", "#28a745"],
                        color_continuous_midpoint=0,
                    )
                    fig_fx.update_layout(
                        height=320, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_fx, use_container_width=True)

                    fxc1, fxc2, fxc3 = st.columns(3)
                    fxc1.metric(
                        "Annualized Alpha", f"{fx['alpha_annual']*100:.2f}%",
                        help="Return not explained by factor exposures. "
                             "Persistent + alpha = skill or missing factor.",
                    )
                    fxc2.metric("R² (factor fit)", f"{fx['r2']:.2f}")
                    fxc3.metric("Obs", f"{fx['n_obs']}")

                st.caption(
                    f"Computed on {report['n_obs']} daily observations over "
                    f"{report['period']}. Risk-free rate assumed 4% for "
                    f"Sharpe/Sortino. Benchmark: {risk_bench}."
                )

# ===========================
# TAB 3: THE VALUATION MACHINE
# ===========================
if _active == "⚖️ Valuation":
    st.header("The Valuation Machine")

    # Shared ticker input — used by both subtabs
    val_ticker = sanitize_ticker(
        st.text_input("Ticker", value="AAPL", key="val_input").upper()
    )

    _val_dcf, _val_sf, _val_tech = st.tabs([
        "📐 DCF Calculator",
        "📋 Fundamentals (SimFin)",
        "📊 Technical Signals (Alpha Vantage)",
    ])

    # ------------------------------------------------------------------
    # DCF CALCULATOR
    # ------------------------------------------------------------------
    with _val_dcf:
        st.markdown("Calculate intrinsic value based on future free cash flow generation.")
        st.markdown(r"$$PV = \sum_{t=1}^{n} \frac{FCF_t}{(1+r)^t} + \frac{TV}{(1+r)^n}$$")
        st.caption("$FCF$ = Free Cash Flow | $r$ = Discount Rate | $TV$ = Terminal Value")
        st.divider()

        st.subheader("Assumptions")
        c_assump1, c_assump2, c_assump3 = st.columns(3)
        with c_assump1:
            discount_rate = st.slider("Discount Rate (r)", min_value=0.01, max_value=0.20, value=0.10, step=0.01)
        with c_assump2:
            growth_rate = st.slider("Growth Rate (Years 1–5)", min_value=-0.10, max_value=0.50, value=0.08, step=0.01)
        with c_assump3:
            terminal_rate = st.slider("Terminal Growth Rate", min_value=0.01, max_value=0.05, value=0.025, step=0.005)

        if val_ticker and st.button("Calculate Intrinsic Value", use_container_width=True, type="primary", key="dcf_calc_btn"):
            # --- Prefer SimFin TTM FCF; fall back to yfinance ---
            _sf_fcf, _sf_shares = fetch_simfin_ttm_fcf(val_ticker) if has_simfin() else (None, None)
            with st.spinner(f"Auditing financials for {val_ticker}..."):
                _yf_fcf, _yf_shares, current_price = fetch_dcf_data(val_ticker)

            if _sf_fcf is not None and _sf_shares is not None:
                fcf, shares = _sf_fcf, _sf_shares
                st.caption("📊 FCF & shares sourced from **SimFin TTM** — more reliable than yfinance.")
            else:
                fcf, shares = _yf_fcf, _yf_shares
                if has_simfin():
                    st.caption("⚠️ SimFin didn't return data for this ticker — falling back to yfinance.")
                else:
                    st.caption("💡 Add `SIMFIN_API_KEY` to `.env` for more reliable TTM FCF data.")

            if fcf is None or shares is None or current_price is None:
                st.error("Insufficient financial data available.")
            elif fcf <= 0:
                st.warning(f"{val_ticker} has negative trailing FCF (${fcf:,.0f}). Standard DCF collapses.")
            else:
                projected_cf, current_fcf = [], fcf
                for year in range(1, 6):
                    current_fcf *= (1 + growth_rate)
                    projected_cf.append(current_fcf / ((1 + discount_rate) ** year))

                sum_pv_cf = sum(projected_cf)
                terminal_value = (current_fcf * (1 + terminal_rate)) / (discount_rate - terminal_rate)
                pv_terminal_value = terminal_value / ((1 + discount_rate) ** 5)
                intrinsic_value_per_share = (sum_pv_cf + pv_terminal_value) / shares
                margin_of_safety = ((intrinsic_value_per_share - current_price) / intrinsic_value_per_share) * 100

                st.divider()
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Live Market Price", f"${current_price:.2f}")
                if intrinsic_value_per_share > current_price:
                    res_col2.success(f"### Intrinsic Value\n## ${intrinsic_value_per_share:.2f}")
                    res_col3.metric("Margin of Safety", f"{margin_of_safety:.1f}%", delta="Undervalued")
                else:
                    res_col2.error(f"### Intrinsic Value\n## ${intrinsic_value_per_share:.2f}")
                    res_col3.metric("Premium to Value", f"{abs(margin_of_safety):.1f}%", delta="-Overvalued", delta_color="inverse")

                st.write("**Audit Breakdown:**")
                st.table(pd.DataFrame({
                    "Metric": ["Trailing FCF (input)", "Shares Outstanding", "Sum of PV (5 Years)", "PV of Terminal Value", "Total Enterprise Value"],
                    "Value":  [format_large_number(fcf), format_large_number(shares),
                               format_large_number(sum_pv_cf), format_large_number(pv_terminal_value),
                               format_large_number(sum_pv_cf + pv_terminal_value)],
                }))

    # ------------------------------------------------------------------
    # FUNDAMENTALS — SimFin
    # ------------------------------------------------------------------
    with _val_sf:
        if not has_simfin():
            st.warning("🔑 SimFin API key not configured.")
            st.info(
                "Get a **free** key at [app.simfin.com](https://app.simfin.com) "
                "(Settings → API Key), then add `SIMFIN_API_KEY=your_key` to `.env` and restart.\n\n"
                "Gives you clean, standardised GAAP financials for 2,000+ US companies — "
                "far more reliable than yfinance for income statements, balance sheets, and cash flow."
            )
        elif val_ticker:
            with st.spinner(f"Fetching SimFin statements for {val_ticker}..."):
                _sf_stmts = fetch_simfin_statements(val_ticker, period="annual")

            if not _sf_stmts or all(v is None for v in _sf_stmts.values()):
                st.warning(
                    f"No SimFin data found for **{val_ticker}**. "
                    "SimFin covers US-listed equities — international or OTC tickers may not be available."
                )
            else:
                _sf_inc = _sf_stmts.get("income")
                _sf_bs  = _sf_stmts.get("balance")
                _sf_cf  = _sf_stmts.get("cashflow")
                _sf_der = _sf_stmts.get("derived")

                # ---- Scorecard (derived metrics, latest year) ----
                if _sf_der is not None and not _sf_der.empty:
                    st.markdown(f"#### Key Ratios — {val_ticker} (latest annual)")
                    _sc = st.columns(5)
                    _der_row = _sf_der.iloc[0]
                    _kpi_map = [
                        ("Gross Profit Margin",    "Gross Margin",   "{}%"),
                        ("Operating Profit Margin","Op. Margin",     "{}%"),
                        ("Net Profit Margin",       "Net Margin",     "{}%"),
                        ("Return on Equity",        "ROE",            "{}%"),
                        ("Return on Assets",        "ROA",            "{}%"),
                    ]
                    for _ci, (_col, _label, _fmt) in enumerate(_kpi_map):
                        _v = _der_row.get(_col)
                        try:
                            _sc[_ci].metric(_label, f"{float(_v):.1f}%")
                        except Exception:
                            _sc[_ci].metric(_label, "—")
                    st.divider()

                # ---- Statement selector ----
                _sf_view = st.radio(
                    "Statement",
                    ["📈 Income Statement", "🏦 Balance Sheet", "💵 Cash Flow"],
                    horizontal=True,
                    key="sf_stmt_radio",
                )

                if _sf_view == "📈 Income Statement":
                    tbl = build_simfin_income_table(_sf_inc)
                    if tbl is not None:
                        st.dataframe(tbl, use_container_width=True)
                        # Revenue trend chart
                        if _sf_inc is not None and "Revenue" in _sf_inc.columns and "Fiscal Year" in _sf_inc.columns:
                            _rev_df = _sf_inc[["Fiscal Year", "Revenue", "Gross Profit", "Net Income"]].dropna().copy()
                            _rev_df = _rev_df.sort_values("Fiscal Year")
                            for _c in ["Revenue", "Gross Profit", "Net Income"]:
                                _rev_df[_c] = pd.to_numeric(_rev_df[_c], errors="coerce") / 1e9
                            _fig_inc = px.line(
                                _rev_df, x="Fiscal Year",
                                y=["Revenue", "Gross Profit", "Net Income"],
                                title=f"{val_ticker} — Revenue, Gross Profit & Net Income (Billions $)",
                                markers=True,
                            )
                            _fig_inc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                            _fig_inc.update_xaxes(showgrid=False)
                            _fig_inc.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title="$B")
                            st.plotly_chart(_fig_inc, use_container_width=True)
                    else:
                        st.info("Income statement data not available.")

                elif _sf_view == "🏦 Balance Sheet":
                    tbl = build_simfin_balance_table(_sf_bs)
                    if tbl is not None:
                        st.dataframe(tbl, use_container_width=True)
                        if _sf_bs is not None and "Total Assets" in _sf_bs.columns:
                            _bs_df = _sf_bs[["Fiscal Year", "Total Assets", "Total Liabilities", "Total Equity"]].dropna().copy()
                            _bs_df = _bs_df.sort_values("Fiscal Year")
                            for _c in ["Total Assets", "Total Liabilities", "Total Equity"]:
                                _bs_df[_c] = pd.to_numeric(_bs_df[_c], errors="coerce") / 1e9
                            _fig_bs = px.bar(
                                _bs_df, x="Fiscal Year",
                                y=["Total Assets", "Total Liabilities", "Total Equity"],
                                barmode="group",
                                title=f"{val_ticker} — Balance Sheet (Billions $)",
                            )
                            _fig_bs.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                            _fig_bs.update_xaxes(showgrid=False)
                            _fig_bs.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title="$B")
                            st.plotly_chart(_fig_bs, use_container_width=True)
                    else:
                        st.info("Balance sheet data not available.")

                else:  # Cash Flow
                    tbl = build_simfin_cashflow_table(_sf_cf)
                    if tbl is not None:
                        st.dataframe(tbl, use_container_width=True)
                        if _sf_cf is not None and "Net Cash from Operating Activities" in _sf_cf.columns:
                            _cf_df = _sf_cf[["Fiscal Year", "Net Cash from Operating Activities", "Capital Expenditures"]].dropna().copy()
                            _cf_df = _cf_df.sort_values("Fiscal Year")
                            _cf_df["Free Cash Flow"] = (
                                pd.to_numeric(_cf_df["Net Cash from Operating Activities"], errors="coerce") +
                                pd.to_numeric(_cf_df["Capital Expenditures"], errors="coerce")
                            )
                            for _c in ["Net Cash from Operating Activities", "Capital Expenditures", "Free Cash Flow"]:
                                _cf_df[_c] = pd.to_numeric(_cf_df[_c], errors="coerce") / 1e9
                            _fig_cf = px.bar(
                                _cf_df, x="Fiscal Year",
                                y=["Net Cash from Operating Activities", "Free Cash Flow"],
                                barmode="group",
                                title=f"{val_ticker} — Cash Flow (Billions $)",
                            )
                            _fig_cf.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                            _fig_cf.update_xaxes(showgrid=False)
                            _fig_cf.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title="$B")
                            st.plotly_chart(_fig_cf, use_container_width=True)
                    else:
                        st.info("Cash flow data not available.")

                st.caption("Data sourced from [SimFin](https://simfin.com) · Standardised GAAP financials")

    # ------------------------------------------------------------------
    # TECHNICAL SIGNALS — Alpha Vantage
    # ------------------------------------------------------------------
    with _val_tech:
        if not has_alpha_vantage():
            st.warning("🔑 Alpha Vantage API key not configured.")
            st.info(
                "Get a **free** key (25 calls/day) at "
                "[alphavantage.co](https://www.alphavantage.co/support/#api-key) "
                "then add `ALPHA_VANTAGE_KEY=your_key` to `.env` and restart.\n\n"
                "Provides RSI, MACD, and Bollinger Bands — cached 24 hrs to stay within quota."
            )
        elif val_ticker:
            st.caption(
                f"Technical indicators for **{val_ticker}** · "
                "Alpha Vantage free tier (25 calls/day — cached 24 hrs)."
            )
            with st.spinner(f"Fetching technical indicators for {val_ticker}..."):
                _av = fetch_av_indicators(val_ticker)
            _av_latest = _av.get("latest", {})

            if not _av_latest:
                st.warning(
                    "No indicator data returned. Alpha Vantage may have hit the "
                    "25-call daily limit, or the ticker is unrecognised."
                )
            else:
                _av1, _av2, _av3, _av4, _av5 = st.columns(5)
                _rsi_val = _av_latest.get("rsi")
                _av1.metric("RSI (14)", f"{_rsi_val}" if _rsi_val else "—",
                            help="<30 oversold · >70 overbought")
                _av2.metric("MACD", f"{_av_latest.get('macd', '—')}")
                _av3.metric("MACD Signal", f"{_av_latest.get('macd_signal', '—')}")
                _av4.metric("BB Upper", f"${_av_latest.get('bb_upper', '—')}")
                _av5.metric("BB Lower", f"${_av_latest.get('bb_lower', '—')}")
                st.divider()

                _tech_view = st.radio(
                    "Chart", ["RSI", "MACD", "Bollinger Bands"],
                    horizontal=True, key="av_chart_radio",
                )
                if _tech_view == "RSI":
                    _rsi_df = _av.get("rsi")
                    if _rsi_df is not None and not _rsi_df.empty:
                        _fig_rsi = px.line(_rsi_df, y="RSI",
                                           title=f"{val_ticker} — RSI (14)")
                        _fig_rsi.add_hline(y=70, line_dash="dot", line_color="#e74c3c",
                                           annotation_text="Overbought (70)")
                        _fig_rsi.add_hline(y=30, line_dash="dot", line_color="#27ae60",
                                           annotation_text="Oversold (30)")
                        _fig_rsi.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                               paper_bgcolor="rgba(0,0,0,0)",
                                               yaxis_range=[0, 100])
                        _fig_rsi.update_xaxes(showgrid=False)
                        st.plotly_chart(_fig_rsi, use_container_width=True)

                elif _tech_view == "MACD":
                    _macd_df = _av.get("macd")
                    if _macd_df is not None and not _macd_df.empty:
                        _fig_macd = px.line(_macd_df, y=["MACD", "Signal"],
                                            title=f"{val_ticker} — MACD (12, 26, 9)")
                        _fig_macd.add_bar(x=_macd_df.index, y=_macd_df["Histogram"],
                                          name="Histogram", opacity=0.4)
                        _fig_macd.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                                paper_bgcolor="rgba(0,0,0,0)")
                        _fig_macd.update_xaxes(showgrid=False)
                        st.plotly_chart(_fig_macd, use_container_width=True)

                else:
                    _bb_df = _av.get("bbands")
                    if _bb_df is not None and not _bb_df.empty:
                        _fig_bb = px.line(_bb_df, y=["Upper", "Middle", "Lower"],
                                          title=f"{val_ticker} — Bollinger Bands (20, 2σ)",
                                          color_discrete_map={
                                              "Upper": "#e74c3c",
                                              "Middle": "#3498db",
                                              "Lower": "#27ae60",
                                          })
                        _fig_bb.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                              paper_bgcolor="rgba(0,0,0,0)")
                        _fig_bb.update_xaxes(showgrid=False)
                        st.plotly_chart(_fig_bb, use_container_width=True)

# ===========================
# TAB 4: MARKET INTELLIGENCE (AI POWERED)
# ===========================
if _active == "📰 Intelligence":
    st.header("Market Intelligence")
    st.markdown("Live, unfiltered news feeds analyzed entirely offline by Llama 3.2.")
    
    news_ticker = sanitize_ticker(st.text_input("Target Asset for Reconnaissance", value="AAPL", key="intel_search").upper())
    st.divider()
    
    if news_ticker:
        with st.spinner(f"Intercepting raw XML data feeds for {news_ticker}..."):
            news_data = fetch_financial_news(news_ticker)
            
            if not news_data:
                st.info(f"No recent institutional coverage found for {news_ticker} on the main XML feeds.")
            else:
                for article in news_data:
                    title = article['title']
                    link = article['link']
                    publisher = article['publisher']
                    time_str = article['time']

                    st.markdown(f"#### [{title}]({link})")
                    st.caption(f"🗞️ **Source:** {publisher} | 🕒 **Published:** {time_str}")
                    
                    with st.spinner("🤖 Noodle Bot analyzing sentiment..."):
                        try:
                            ai_prompt = f"Analyze this financial news headline for the stock {news_ticker}: '{title}'. Respond strictly in this exact format: [BULLISH/BEARISH/NEUTRAL] - [One concise sentence explanation of why]."
                            
                            oracle_persona = """You are 'The True Oracle', an elite financial AI. You must strictly obey the following rules:
1. The Logic-First Filter: Perform a Logical Audit defining the Domain of Discourse and isolating atomic propositions.
2. Probabilistic Calibration: Reject binary True/False. Treat new info as Evidence updating a Prior Belief."""

                            from llm_router import llm_chat as _llm_chat
                            ai_analysis = _llm_chat([
                                {'role': 'system', 'content': oracle_persona},
                                {'role': 'user', 'content': ai_prompt}
                            ]).strip()
                            
                            if "BULLISH" in ai_analysis.upper(): st.success(f"**AI Sentiment:** {ai_analysis}")
                            elif "BEARISH" in ai_analysis.upper(): st.error(f"**AI Sentiment:** {ai_analysis}")
                            else: st.info(f"**AI Sentiment:** {ai_analysis}")
                        except Exception as ai_e:
                            st.warning("AI Engine offline. Ensure the Ollama Mac app is running in your menu bar.")
                    st.write("---")

# ===========================
# TAB 5: PEER MATRIX
# ===========================
if _active == "🏢 Peer Matrix":
    st.header("Peer Group Comparison Matrix")
    st.markdown("Compare relative valuation multiples across custom industry cohorts.")

    col_pg_sel, col_pg_add, col_pg_del = st.columns(3)
    group_names = list(peer_groups.keys())
    
    with col_pg_sel:
        selected_group = st.selectbox("Select Industry Cohort", group_names if group_names else ["None"])
    
    with col_pg_add:
        with st.form("create_group_form", clear_on_submit=True):
            new_group_name = st.text_input("New Cohort Name")
            if st.form_submit_button("Create Cohort"):
                if new_group_name and new_group_name not in peer_groups:
                    peer_groups[new_group_name] = []
                    app_data["peer_groups"] = peer_groups
                    save_data(app_data); st.toast(f"Created cohort: {new_group_name}", icon="✅")

    with col_pg_del:
        if group_names:
            with st.form("delete_group_form"):
                group_to_delete = st.selectbox("Delete Cohort", group_names)
                if st.form_submit_button("Delete Permanently"):
                    del peer_groups[group_to_delete]
                    app_data["peer_groups"] = peer_groups
                    save_data(app_data); st.toast("Cohort deleted.", icon="🗑️")
    st.divider()

    if selected_group and selected_group != "None":
        group_tickers = peer_groups[selected_group]
        col_t_add, col_t_del = st.columns(2)
        with col_t_add:
            with st.form("add_peer_form", clear_on_submit=True):
                new_peer = sanitize_ticker(st.text_input("Add Ticker to Cohort").upper())
                if st.form_submit_button("Add Asset") and new_peer:
                    if new_peer not in group_tickers:
                        peer_groups[selected_group].append(new_peer)
                        app_data["peer_groups"] = peer_groups
                        save_data(app_data)
        with col_t_del:
            if group_tickers:
                with st.form("remove_peer_form"):
                    peer_to_remove = st.selectbox("Remove Ticker", group_tickers)
                    if st.form_submit_button("Remove Asset"):
                        peer_groups[selected_group].remove(peer_to_remove)
                        app_data["peer_groups"] = peer_groups
                        save_data(app_data)

        if group_tickers:
            st.subheader(f"Relative Valuation: {selected_group}")
            with st.spinner(f"Auditing financial statements for {len(group_tickers)} peers..."):
                peer_df = fetch_peer_metrics(group_tickers)
            if not peer_df.empty:
                st.dataframe(
                    peer_df, hide_index=True, use_container_width=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                        "P/E (Trailing)": st.column_config.NumberColumn(format="%.2f"),
                        "P/E (Forward)": st.column_config.NumberColumn(format="%.2f"),
                        "P/B": st.column_config.NumberColumn(format="%.2f"),
                        "EV/EBITDA": st.column_config.NumberColumn(format="%.2f"),
                        "ROE (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Debt/Equity": st.column_config.NumberColumn(format="%.2f"),
                        "Div Yield (%)": st.column_config.NumberColumn(format="%.2f%%")
                    }
                )
                st.divider()
                st.markdown("##### Cross-Sectional Analysis Chart")
                chart_metric = st.selectbox("Select Metric to Visualize", ["P/E (Trailing)", "P/E (Forward)", "P/B", "EV/EBITDA", "ROE (%)", "Debt/Equity", "Div Yield (%)"])
                plot_data = peer_df.dropna(subset=[chart_metric])
                if not plot_data.empty:
                    plot_data = plot_data.sort_values(by=chart_metric, ascending=True)
                    fig_peer = px.bar(
                        plot_data, x='Ticker', y=chart_metric, 
                        title=f"{chart_metric} Comparison",
                        color=chart_metric, color_continuous_scale=['#28a745', '#ffc107', '#dc3545']
                    )
                    fig_peer.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_peer, use_container_width=True)
                else: st.info(f"Not enough clean data to plot {chart_metric} for this cohort.")
        else: st.info("This cohort is empty. Add some tickers above.")

# ===========================
# TAB 6: MACROECONOMICS
# ===========================
if _active == "🌐 Global Markets":
    st.header("🌐 Global Markets")
    st.caption("Macro, energy, commodities, crypto, and analyst intelligence — all in one place.")

    _gm_systemic, _gm_resources, _gm_fmp = st.tabs([
        "🏛️ US Systemic Data",
        "⚡ Resources & Assets",
        "📈 FMP Analytics",
    ])

    # ==============================================================
    # SUBTAB 1: US SYSTEMIC DATA (FRED / BLS / BEA / Treasury)
    # ==============================================================
    with _gm_systemic:
        st.caption("Official U.S. macro data: Federal Reserve, BLS, BEA, and Treasury.")
        _mac_fred, _mac_bls, _mac_bea, _mac_tsy = st.tabs([
            "🏛️ FRED Explorer",
            "👷 Labor & Inflation (BLS)",
            "📊 GDP & Spending (BEA)",
            "🏦 Treasury & Debt",
        ])

    # ------------------------------------------------------------------
    # FRED EXPLORER (existing)
    # ------------------------------------------------------------------
    with _mac_fred:
        if not fred:
            st.error("🚨 FRED API key not detected.")
            st.info("Add `FRED_API_KEY=your_key` to `.env` and restart.")
        else:
            standard_indicators = {
                "Effective Federal Funds Rate (Interest Rates)": "FEDFUNDS",
                "Consumer Price Index (Inflation)": "CPIAUCSL",
                "Real Gross Domestic Product (GDP)": "GDPC1",
                "M2 Money Supply (Liquidity)": "M2SL",
                "Unemployment Rate": "UNRATE",
                "10-Year Treasury Constant Maturity Rate": "DGS10",
                "ICE BofA US High Yield Spread (Credit Risk)": "BAMLH0A0HYM2",
                "Federal Reserve Total Assets (System Liquidity)": "WALCL",
                "Custom Series ID...": "CUSTOM",
            }
            col_m_sel, col_m_cust = st.columns([2, 1])
            with col_m_sel:
                selected_indicator_name = st.selectbox(
                    "Select Institutional Metric", list(standard_indicators.keys())
                )
                selected_series_id = standard_indicators[selected_indicator_name]
            with col_m_cust:
                if selected_series_id == "CUSTOM":
                    selected_series_id = st.text_input("Enter FRED Series ID").upper()
                else:
                    st.text_input("Active Series ID", value=selected_series_id, disabled=True)
            st.divider()
            if selected_series_id and selected_series_id != "CUSTOM":
                with st.spinner(f"Querying FRED for {selected_series_id}..."):
                    macro_df = fetch_macro_data(selected_series_id)
                if macro_df is not None and not macro_df.empty:
                    latest_date  = macro_df.index[-1].strftime("%B %Y")
                    latest_value = macro_df["Value"].iloc[-1]
                    st.metric(f"Latest Print ({latest_date})", f"{latest_value:,.2f}")
                    fig_macro = px.area(
                        macro_df, x=macro_df.index, y="Value",
                        title=f"{selected_indicator_name} (Last 20 Years)",
                    )
                    fig_macro.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        hovermode="x unified",
                    )
                    fig_macro.update_xaxes(showgrid=False, title_text="")
                    fig_macro.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    fig_macro.update_traces(
                        line_color="#2980b9", fillcolor="rgba(41,128,185,0.2)"
                    )
                    st.plotly_chart(fig_macro, use_container_width=True)
                else:
                    st.warning(f"No data for {selected_series_id}. Verify the series ID.")

    # ------------------------------------------------------------------
    # BLS — LABOR & INFLATION
    # ------------------------------------------------------------------
    with _mac_bls:
        st.subheader("Bureau of Labor Statistics")
        st.caption(
            "Official U.S. labor market and inflation data — CPI, Core CPI, PPI, "
            "Nonfarm Payrolls, and Unemployment. "
            "No API key required · add `BLS_API_KEY` to `.env` for 20-year history."
        )
        with st.spinner("Fetching BLS indicators..."):
            _bls = fetch_bls_indicators()

        _bls_latest = _bls.get("latest", {})
        _bls_series = _bls.get("series", {})

        if not _bls_latest:
            st.warning(
                "BLS data unavailable — the BLS public API may be rate-limited. "
                "Try again in a few minutes."
            )
        else:
            # ---- Scorecard row ----
            _b1, _b2, _b3, _b4, _b5 = st.columns(5)
            _b1.metric(
                "🧾 CPI (YoY)",
                f"{_bls_latest.get('cpi_yoy', 'N/A')}%",
                help=f"All-items CPI as of {_bls_latest.get('cpi_date', '')}",
            )
            _b2.metric(
                "🔍 Core CPI (YoY)",
                f"{_bls_latest.get('core_cpi_yoy', 'N/A')}%",
                help="Ex food & energy",
            )
            _b3.metric(
                "🏭 PPI (YoY)",
                f"{_bls_latest.get('ppi_yoy', 'N/A')}%",
                help=f"Producer Price Index as of {_bls_latest.get('ppi_date', '')}",
            )
            _b4.metric(
                "👷 Payrolls (MoM)",
                f"{_bls_latest.get('payrolls_mom', 'N/A'):,}K"
                if isinstance(_bls_latest.get("payrolls_mom"), (int, float)) else "N/A",
                help=f"Nonfarm payrolls change as of {_bls_latest.get('payrolls_date', '')}",
            )
            _b5.metric(
                "📉 Unemployment",
                f"{_bls_latest.get('unemployment', 'N/A')}%",
                help=f"As of {_bls_latest.get('unemployment_date', '')}",
            )
            st.divider()

            # ---- Chart selector ----
            _bls_chart_opts = {
                "CPI (All Items)":        ("cpi",          "CPI Index Level"),
                "Core CPI (ex F&E)":      ("core_cpi",     "Core CPI Index Level"),
                "PPI (Final Demand)":     ("ppi",          "PPI Index Level"),
                "Nonfarm Payrolls":       ("payrolls",     "Thousands of Jobs"),
                "Unemployment Rate (%)":  ("unemployment", "Unemployment %"),
            }
            _bls_choice = st.selectbox(
                "Chart series", list(_bls_chart_opts.keys()), key="bls_chart_sel"
            )
            _bls_key, _bls_ylabel = _bls_chart_opts[_bls_choice]
            _bls_df = _bls_series.get(_bls_key)
            if _bls_df is not None and not _bls_df.empty:
                _fig_bls = px.area(
                    _bls_df, x=_bls_df.index, y="Value",
                    title=f"{_bls_choice} — BLS Official Data",
                    labels={"Value": _bls_ylabel},
                )
                _fig_bls.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                )
                _fig_bls.update_xaxes(showgrid=False, title_text="")
                _fig_bls.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                _fig_bls.update_traces(
                    line_color="#e67e22", fillcolor="rgba(230,126,34,0.15)"
                )
                st.plotly_chart(_fig_bls, use_container_width=True)
            else:
                st.info("Series data not yet available.")

    # ------------------------------------------------------------------
    # BEA — GDP & SPENDING
    # ------------------------------------------------------------------
    with _mac_bea:
        st.subheader("Bureau of Economic Analysis")
        if not has_bea():
            st.warning("🔑 BEA API key not configured.")
            st.info(
                "Get a free key at **[apps.bea.gov/API/signup](https://apps.bea.gov/API/signup/)** "
                "then add `BEA_API_KEY=your_key` to `.env` and restart."
            )
        else:
            st.caption(
                "GDP growth and PCE inflation — the two data series the Fed watches most closely."
            )
            _bea_col1, _bea_col2 = st.columns(2)

            with st.spinner("Fetching BEA data..."):
                _gdp  = fetch_bea_gdp()
                _pce  = fetch_bea_pce()

            # GDP metric + chart
            with _bea_col1:
                st.markdown("#### 📈 Real GDP Growth (QoQ, annualized %)")
                if _gdp:
                    _gdp_delta_color = "normal" if _gdp["latest"] >= 0 else "inverse"
                    st.metric(
                        f"Latest ({_gdp['latest_date']})",
                        f"{_gdp['latest']:+.2f}%",
                    )
                    _fig_gdp = px.bar(
                        _gdp["series"].reset_index(),
                        x="Date", y="Value",
                        title="Real GDP Growth Rate",
                        labels={"Value": "% Change (annualized)"},
                        color="Value",
                        color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
                    )
                    _fig_gdp.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                    )
                    _fig_gdp.update_xaxes(showgrid=False)
                    _fig_gdp.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    st.plotly_chart(_fig_gdp, use_container_width=True)
                else:
                    st.info("GDP data unavailable.")

            # PCE metric + chart
            with _bea_col2:
                st.markdown("#### 💸 PCE Price Index (YoY % — Fed target: 2%)")
                if _pce:
                    _pce_yoy = _pce.get("latest_yoy")
                    st.metric(
                        f"Latest ({_pce['latest_date']})",
                        f"{_pce_yoy:+.2f}%" if _pce_yoy is not None else "N/A",
                        help="Fed's preferred inflation gauge (vs. 2% target)",
                    )
                    _fig_pce = px.area(
                        _pce["series"].reset_index(),
                        x="Date", y="Value",
                        title="PCE Price Index",
                        labels={"Value": "Index Level"},
                    )
                    _fig_pce.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    _fig_pce.update_xaxes(showgrid=False)
                    _fig_pce.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    _fig_pce.update_traces(
                        line_color="#8e44ad", fillcolor="rgba(142,68,173,0.15)"
                    )
                    st.plotly_chart(_fig_pce, use_container_width=True)
                else:
                    st.info("PCE data unavailable.")

    # ------------------------------------------------------------------
    # TREASURY & DEBT
    # ------------------------------------------------------------------
    with _mac_tsy:
        st.subheader("U.S. Treasury — Fiscal Data")
        st.caption("National debt and yield curve sourced directly from Treasury APIs. No API key required.")

        _tsy_col1, _tsy_col2 = st.columns([1, 2])

        with st.spinner("Fetching Treasury data..."):
            _debt = fetch_treasury_debt()
            _yc   = fetch_yield_curve()

        # National Debt
        with _tsy_col1:
            st.markdown("#### 🏛️ National Debt")
            if _debt:
                st.metric(
                    f"Total Outstanding ({_debt['latest_date']})",
                    f"${_debt['latest_trillions']:,.2f}T",
                )
                _debt_series = _debt.get("series")
                if _debt_series is not None and not _debt_series.empty:
                    _fig_debt = px.area(
                        _debt_series.reset_index(),
                        x="Date", y="Value",
                        title="National Debt (Trillions USD)",
                        labels={"Value": "Trillions ($)"},
                    )
                    _fig_debt.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    _fig_debt.update_xaxes(showgrid=False)
                    _fig_debt.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    _fig_debt.update_traces(
                        line_color="#c0392b", fillcolor="rgba(192,57,43,0.15)"
                    )
                    st.plotly_chart(_fig_debt, use_container_width=True)
            else:
                st.info("Debt data unavailable.")

        # Yield Curve
        with _tsy_col2:
            st.markdown("#### 📐 Treasury Yield Curve")
            if _yc is not None and not _yc.empty:
                _spread_10_2 = None
                _y10 = _yc.loc[_yc["Maturity"] == "10Y", "Yield"]
                _y2  = _yc.loc[_yc["Maturity"] == "2Y",  "Yield"]
                if not _y10.empty and not _y2.empty:
                    _spread_10_2 = round(float(_y10.iloc[0]) - float(_y2.iloc[0]), 3)

                _yc_cols = st.columns(3)
                for _i, row in _yc.iterrows():
                    _yc_cols[_i % 3].metric(row["Maturity"], f"{row['Yield']:.3f}%")

                st.divider()
                if _spread_10_2 is not None:
                    _inv = "🔴 Inverted" if _spread_10_2 < 0 else "🟢 Normal"
                    st.metric(
                        "10Y–2Y Spread (Inversion Indicator)",
                        f"{_spread_10_2:+.3f}%",
                        help=f"{_inv} curve  · Negative = recession signal",
                    )

                _fig_yc = px.line(
                    _yc, x="Maturity", y="Yield",
                    title="Treasury Yield Curve (today)",
                    labels={"Yield": "Yield (%)"},
                    markers=True,
                )
                _fig_yc.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                _fig_yc.update_xaxes(showgrid=False)
                _fig_yc.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                _fig_yc.update_traces(line_color="#16a085", marker_color="#16a085")
                st.plotly_chart(_fig_yc, use_container_width=True)

                if not fred:
                    st.caption(
                        "_Sourced from Treasury.gov XML. Add `FRED_API_KEY` to `.env` "
                        "for higher-reliability FRED-backed yield data._"
                    )
            else:
                st.info(
                    "Yield curve data unavailable. "
                    "Add `FRED_API_KEY` to `.env` for the most reliable source."
                )

    # ==============================================================
    # SUBTAB 2: RESOURCES & ASSETS
    # ==============================================================
    with _gm_resources:
        st.caption("Energy inventories (EIA), commodities, crypto market, and sentiment.")

        # ---- CNN Fear & Greed ----
        with st.spinner("Loading sentiment data..."):
            _fg = fetch_fear_greed()
        if _fg and _fg.get("score") is not None:
            _fg_score  = _fg["score"]
            _fg_rating = _fg.get("rating", "")
            _fg_color  = ("#e74c3c" if _fg_score < 25 else
                          "#e67e22" if _fg_score < 45 else
                          "#f1c40f" if _fg_score < 55 else
                          "#2ecc71" if _fg_score < 75 else "#27ae60")
            _fgc1, _fgc2 = st.columns([1, 3])
            with _fgc1:
                st.markdown(f"""
                <div style='text-align:center; padding:16px; border-radius:12px;
                            background:{_fg_color}22; border:2px solid {_fg_color}'>
                  <div style='font-size:2.8rem; font-weight:700; color:{_fg_color}'>{int(_fg_score)}</div>
                  <div style='font-size:1rem; font-weight:600; color:{_fg_color}'>{_fg_rating}</div>
                  <div style='font-size:0.75rem; color:#888'>CNN Fear & Greed</div>
                </div>""", unsafe_allow_html=True)
            with _fgc2:
                if _fg.get("history") is not None:
                    _fig_fg = px.area(_fg["history"], y="Score",
                                      title="Fear & Greed — 1 Year History",
                                      color_discrete_sequence=[_fg_color])
                    _fig_fg.add_hline(y=25, line_dash="dot", line_color="#e74c3c",
                                      annotation_text="Extreme Fear")
                    _fig_fg.add_hline(y=75, line_dash="dot", line_color="#27ae60",
                                      annotation_text="Extreme Greed")
                    _fig_fg.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          yaxis_range=[0, 100])
                    _fig_fg.update_xaxes(showgrid=False)
                    st.plotly_chart(_fig_fg, use_container_width=True)
        else:
            st.info("Fear & Greed data unavailable.")
        st.divider()

        # ---- Commodities + Crypto side by side ----
        _rc1, _rc2 = st.columns(2)

        with _rc1:
            st.markdown("#### 🏅 Commodities")
            with st.spinner("Loading commodity prices..."):
                _comms = fetch_commodity_prices()
            for _cname, _cdata in _comms.items():
                _cp = _cdata.get("price")
                _cc = _cdata.get("change")
                if _cp is not None:
                    _delta_color = "normal" if (_cc or 0) >= 0 else "inverse"
                    st.metric(_cname, f"${_cp:,.2f}",
                              delta=f"{_cc:+.2f}%" if _cc is not None else None,
                              delta_color=_delta_color)
                else:
                    st.metric(_cname, "—")

        with _rc2:
            st.markdown("#### ₿ Crypto Market")
            with st.spinner("Loading CoinGecko data..."):
                _cg_global = fetch_coingecko_global()
                _cg_top    = fetch_coingecko_top_coins(10)
            if _cg_global:
                _cgg1, _cgg2, _cgg3 = st.columns(3)
                _tmc = _cg_global.get("total_market_cap_usd")
                _btcd = _cg_global.get("btc_dominance")
                _mcc  = _cg_global.get("market_cap_change_24h")
                _cgg1.metric("Total Mkt Cap",
                             f"${_tmc/1e12:.2f}T" if _tmc else "—",
                             delta=f"{_mcc:+.1f}%" if _mcc else None)
                _cgg2.metric("BTC Dominance",
                             f"{_btcd:.1f}%" if _btcd else "—")
                _cgg3.metric("ETH Dominance",
                             f"{_cg_global.get('eth_dominance', 0):.1f}%")
            if _cg_top is not None:
                st.dataframe(
                    _cg_top,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Price":    st.column_config.NumberColumn(format="$%.4f"),
                        "24h %":    st.column_config.NumberColumn(format="%.2f%%"),
                        "7d %":     st.column_config.NumberColumn(format="%.2f%%"),
                        "Mkt Cap":  st.column_config.NumberColumn(format="$%.0f"),
                        "Vol 24h":  st.column_config.NumberColumn(format="$%.0f"),
                    },
                )
            else:
                st.info("CoinGecko data unavailable (rate limit — try again shortly).")

        st.divider()

        # ---- EIA Energy ----
        st.markdown("#### ⚡ EIA Energy Inventories")
        if not has_eia():
            st.info("Add `EIA_API_KEY=...` to `.env` for weekly energy inventory data.\n\n"
                    "Free key at [eia.gov/opendata](https://www.eia.gov/opendata/register.php).")
        else:
            with st.spinner("Fetching EIA energy data..."):
                _eia = fetch_eia_snapshot()
            _eia_latest = _eia.get("latest", {})
            _eia_series = _eia.get("series", {})
            if not _eia_latest:
                st.warning("EIA data unavailable — check your API key or try again.")
            else:
                _eia_cols = st.columns(len(_eia_latest))
                for _i, (_key, _info) in enumerate(_eia_latest.items()):
                    _wow = _info.get("wow_change")
                    _eia_cols[_i].metric(
                        f"{_info['label']} ({_info['unit']})",
                        f"{_info['value']:,.1f}",
                        delta=f"{_wow:+.1f} WoW" if _wow is not None else None,
                        help=f"As of {_info['date']}",
                    )
                _eia_choice = st.selectbox(
                    "Chart EIA series", list(_eia_latest.keys()),
                    format_func=lambda k: _eia_latest[k]["label"],
                    key="eia_chart_sel",
                )
                _eia_df = _eia_series.get(_eia_choice)
                if _eia_df is not None and not _eia_df.empty:
                    _unit = _eia_latest[_eia_choice]["unit"]
                    _fig_eia = px.area(_eia_df, y="Value",
                                       title=f"{_eia_latest[_eia_choice]['label']} ({_unit})")
                    _fig_eia.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                           paper_bgcolor="rgba(0,0,0,0)")
                    _fig_eia.update_xaxes(showgrid=False)
                    _fig_eia.update_traces(line_color="#f39c12",
                                           fillcolor="rgba(243,156,18,0.15)")
                    st.plotly_chart(_fig_eia, use_container_width=True)

        st.divider()

        # ---- CFTC Commitment of Traders ----
        st.markdown("#### 📊 CFTC Commitments of Traders")
        st.caption(
            "Weekly speculator (non-commercial) positioning in key futures markets. "
            "No API key required — public CFTC data, updated every Friday."
        )
        with st.spinner("Fetching CFTC CoT data..."):
            _cftc_snap = fetch_cftc_snapshot()

        if not _cftc_snap:
            st.info(
                "CFTC CoT data unavailable right now — "
                "the CFTC Socrata API may be temporarily slow. Try again shortly."
            )
        else:
            # --- Snapshot table ---
            _cftc_rows = []
            for _ck, _cv in _cftc_snap.items():
                _nc_net = _cv["nc_net"]
                _nc_chg = _cv["nc_chg"]
                _bias = ("🐂 Bullish" if _nc_net > 50_000 else
                         "🐻 Bearish" if _nc_net < -50_000 else
                         "🐂 Mild Bull" if _nc_net > 0 else "🐻 Mild Bear")
                _cftc_rows.append({
                    "Market":          _cv["label"],
                    "Net Speculative": _nc_net,
                    "WoW Change":      _nc_chg,
                    "Bias":            _bias,
                    "Open Interest":   _cv["oi"],
                    "As of":           _cv["date"],
                })
            _cftc_df_table = pd.DataFrame(_cftc_rows)
            st.dataframe(
                _cftc_df_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Net Speculative": st.column_config.NumberColumn(format="%d"),
                    "WoW Change":      st.column_config.NumberColumn(format="%+d"),
                    "Open Interest":   st.column_config.NumberColumn(format="%d"),
                },
            )

            # --- Historical net positioning chart ---
            _cftc_mkt_opts = list(_cftc_snap.keys())
            _cftc_sel = st.selectbox(
                "Chart speculator net positioning",
                _cftc_mkt_opts,
                format_func=lambda k: _cftc_snap.get(k, {}).get("label", k),
                key="cftc_chart_sel",
            )
            with st.spinner(f"Loading {_cftc_snap[_cftc_sel]['label']} history..."):
                _cftc_hist = fetch_cftc_cot(_cftc_sel, weeks=104)  # ~2 years

            if _cftc_hist is not None and not _cftc_hist.empty:
                _cftc_mkt_label = _cftc_snap[_cftc_sel]["label"]
                _net_vals = _cftc_hist["NonComm_Net"]
                _bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in _net_vals]
                _fig_cftc = go.Figure()
                _fig_cftc.add_trace(go.Bar(
                    x=_cftc_hist.index,
                    y=_net_vals,
                    name="Net Speculative",
                    marker_color=_bar_colors,
                    hovertemplate="%{x|%b %d %Y}<br>Net: %{y:,d}<extra></extra>",
                ))
                _fig_cftc.add_hline(y=0, line_color="#888", line_width=1)
                _fig_cftc.update_layout(
                    title=f"{_cftc_mkt_label} — Speculator Net Positions (2-Year)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Contracts (Long − Short)",
                    showlegend=False,
                    bargap=0.1,
                )
                _fig_cftc.update_xaxes(showgrid=False)
                _fig_cftc.update_yaxes(zeroline=True, zerolinecolor="#555",
                                        zerolinewidth=1)
                st.plotly_chart(_fig_cftc, use_container_width=True)
                st.caption(
                    "**Reading the chart:** Green bars = speculators are net long "
                    "(bullish). Red bars = net short (bearish). Extreme readings "
                    "near historical highs/lows often act as contrarian signals."
                )

    # ==============================================================
    # SUBTAB 3: FMP ANALYTICS
    # ==============================================================
    with _gm_fmp:
        if not has_fmp():
            st.warning("🔑 FMP API key not configured.")
            st.info(
                "Get a **free** key (250 calls/day) at "
                "[financialmodelingprep.com](https://financialmodelingprep.com/developer/docs) "
                "then add `FMP_API_KEY=your_key` to `.env` and restart.\n\n"
                "Unlocks: analyst price targets, forward EPS estimates, "
                "company profiles, and quantitative ratings."
            )
        else:
            _fmp_ticker = st.text_input("Ticker", value="AAPL",
                                        key="fmp_ticker").upper().strip()
            if _fmp_ticker:
                with st.spinner(f"Fetching FMP data for {_fmp_ticker}..."):
                    _fmp_profile   = fetch_fmp_profile(_fmp_ticker)
                    _fmp_targets   = fetch_fmp_price_targets(_fmp_ticker)
                    _fmp_estimates = fetch_fmp_analyst_estimates(_fmp_ticker)
                    _fmp_rating    = fetch_fmp_ratings(_fmp_ticker)

                # ---- Company Profile ----
                if _fmp_profile:
                    _fp1, _fp2 = st.columns([3, 1])
                    with _fp1:
                        st.markdown(f"#### {_fmp_profile.get('companyName', _fmp_ticker)}")
                        st.caption(
                            f"**Sector:** {_fmp_profile.get('sector','—')}  ·  "
                            f"**Industry:** {_fmp_profile.get('industry','—')}  ·  "
                            f"**CEO:** {_fmp_profile.get('ceo','—')}  ·  "
                            f"**Employees:** {_fmp_profile.get('fullTimeEmployees','—'):,}"
                            if isinstance(_fmp_profile.get('fullTimeEmployees'), int) else
                            f"**Sector:** {_fmp_profile.get('sector','—')}  ·  "
                            f"**Industry:** {_fmp_profile.get('industry','—')}"
                        )
                        with st.expander("Company Description"):
                            st.write(_fmp_profile.get("description", "N/A"))
                    with _fp2:
                        _fmp_price = _fmp_profile.get("price")
                        _fmp_mktcap = _fmp_profile.get("mktCap")
                        if _fmp_price:
                            st.metric("Price", f"${_fmp_price:,.2f}")
                        if _fmp_mktcap:
                            from api import _fmt_millions
                            st.metric("Market Cap", _fmt_millions(_fmp_mktcap))
                    st.divider()

                # ---- Price Targets ----
                if _fmp_targets:
                    st.markdown("#### 🎯 Analyst Price Targets")
                    _pt_cols = st.columns(4)
                    _pt_cols[0].metric("Consensus",
                                       f"${_fmp_targets.get('targetConsensus','—')}")
                    _pt_cols[1].metric("High Target",
                                       f"${_fmp_targets.get('targetHigh','—')}")
                    _pt_cols[2].metric("Low Target",
                                       f"${_fmp_targets.get('targetLow','—')}")
                    _pt_cols[3].metric("Median Target",
                                       f"${_fmp_targets.get('targetMedian','—')}")
                    st.divider()

                # ---- Quantitative Rating ----
                if _fmp_rating:
                    st.markdown("#### ⭐ FMP Quantitative Rating")
                    _rat_cols = st.columns(4)
                    _rat_cols[0].metric("Overall Rating", _fmp_rating.get("rating", "—"))
                    _rat_cols[1].metric("DCF Score",
                                        _fmp_rating.get("ratingDetailsDCFScore", "—"))
                    _rat_cols[2].metric("ROE Score",
                                        _fmp_rating.get("ratingDetailsROEScore", "—"))
                    _rat_cols[3].metric("Recommendation",
                                        _fmp_rating.get("ratingRecommendation", "—"))
                    st.divider()

                # ---- Forward Estimates ----
                if _fmp_estimates is not None and not _fmp_estimates.empty:
                    st.markdown("#### 📅 Forward Analyst Estimates")
                    st.dataframe(
                        _fmp_estimates,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Est. Revenue": st.column_config.NumberColumn(format="$%.0f"),
                            "Est. EPS":     st.column_config.NumberColumn(format="$%.2f"),
                            "EPS Low":      st.column_config.NumberColumn(format="$%.2f"),
                            "EPS High":     st.column_config.NumberColumn(format="$%.2f"),
                        },
                    )
                elif _fmp_profile:
                    st.info("No analyst estimates available for this ticker.")

# ===========================
# TAB 7: THE LIBRARY (RAG PIPELINE)
# ===========================
if _active == "📚 The Library":
    st.header("The Library (Local RAG Database)")
    st.markdown("Inject financial PDFs, Annual 10-Ks, and Quarterly 8-K Earnings data directly into Noodle Bot's permanent memory.")

    # -----------------------------------------------------------------------
    # INSTITUTIONAL RESEARCH COVERAGE BOARD
    # -----------------------------------------------------------------------
    st.subheader("🏦 Institutional Research Coverage")
    st.caption(
        "Track which major institutions have a market report in the library. "
        "🟢 = covered · ⚪ = missing. Each institution can have **multiple** "
        "reports attached; the ⭐ marks the *primary* (default-displayed) one."
    )

    _inst_rows = list_institutional_coverage()
    # Single full-library read for the whole Library tab — paginated pull through
    # ChromaDB is now the single most expensive op on this tab (78 k+ chunks).
    # Both the Institutional Coverage section *and* the Manage Library expander
    # below need the same data, so we share it instead of re-querying.
    _lib_docs = _list_documents()
    # Only market_report docs are relevant for institutional coverage.
    _mr_docs  = [d for d in _lib_docs if d["category"] == "market_report"]
    _mr_by_id  = {d["doc_id"]: d for d in _mr_docs}
    _doc_labels = {d["doc_id"]: d["source"] for d in _mr_docs}

    # Per-row "update mode" toggle — lives in session_state so it survives reruns
    if "_inst_update_mode" not in st.session_state:
        st.session_state._inst_update_mode = set()

    def _doc_label_short(doc_id: str) -> str:
        """Compact label for a doc — strip the './temp_pdfs/' prefix and the
        '.pdf' suffix so the row stays readable when many reports are stacked."""
        raw = _doc_labels.get(doc_id, doc_id)
        raw = raw.replace("./temp_pdfs/", "").replace("PDF: ", "")
        if raw.lower().endswith(".pdf"):
            raw = raw[:-4]
        return raw

    if not _inst_rows:
        st.info("No institutions tracked yet — add one below.")
    else:
        # ---- Header row (bordered, like the mockup) ----
        with st.container(border=True):
            _hc1, _hc2, _hc3 = st.columns([2, 3, 2])
            _hc1.markdown("**Institution**")
            _hc2.markdown("**Attached Reports**")
            _hc3.markdown("**Update Portal**")

        # ---- Data rows ----
        for _inst in _inst_rows:
            _name        = _inst["institution"]
            _primary_id  = _inst.get("primary_doc_id")
            _doc_ids     = _inst.get("doc_ids") or []
            _editing     = _name in st.session_state._inst_update_mode
            _has_any     = bool(_doc_ids)

            with st.container(border=True):
                _ic1, _ic2, _ic3 = st.columns([2, 3, 2])
                _ic1.markdown(f"{'🟢' if _has_any else '⚪'}  {_name}")

                # ───────────────────────── EDIT MODE ─────────────────────────
                if _editing:
                    with _ic2:
                        # Multi-file drop — accept many PDFs at once
                        _new_pdfs = st.file_uploader(
                            f"Drop new PDFs for {_name}",
                            type="pdf",
                            accept_multiple_files=True,
                            key=f"inst_upload_{_name}",
                            label_visibility="collapsed",
                            help="Drag-and-drop one or many report PDFs here — each "
                                 "is ingested as a market_report and attached to "
                                 "this institution.",
                        )
                        # Pick existing reports already in the library
                        _available_ids = [d["doc_id"] for d in _mr_docs
                                          if d["doc_id"] not in _doc_ids]
                        _sel_existing = st.multiselect(
                            "Or pick from existing market reports",
                            options=_available_ids,
                            format_func=_doc_label_short,
                            key=f"inst_link_{_name}",
                            label_visibility="collapsed",
                            placeholder="— pick existing report(s) to attach —",
                        )

                        # Already-attached reports list with primary toggle + unlink
                        if _doc_ids:
                            st.markdown("**Currently attached:**")
                            for _doc_id in _doc_ids:
                                _is_primary = _doc_id == _primary_id
                                _l1, _l2, _l3 = st.columns([8, 1, 1])
                                _star = "⭐ " if _is_primary else "☆ "
                                _l1.markdown(
                                    f"{_star}_{_doc_label_short(_doc_id)}_"
                                )
                                # Promote to primary
                                if not _is_primary:
                                    if _l2.button(
                                        "⭐",
                                        key=f"inst_setpri_{_name}_{_doc_id}",
                                        help="Mark as primary",
                                    ):
                                        set_primary_institution_doc(_name, _doc_id)
                                        st.rerun()
                                else:
                                    _l2.markdown("&nbsp;", unsafe_allow_html=True)
                                # Unlink (does NOT delete the doc from the library)
                                if _l3.button(
                                    "✖",
                                    key=f"inst_unlink_{_name}_{_doc_id}",
                                    help="Detach from this institution",
                                ):
                                    unlink_institution_doc(_name, _doc_id)
                                    st.rerun()

                    with _ic3:
                        _sb1, _sb2, _sb3 = st.columns(3)
                        # ── Save: ingest dropped PDFs + attach selected existing
                        if _sb1.button("💾", key=f"inst_save_{_name}", help="Save"):
                            _added = 0
                            _errors: list[str] = []
                            # Path A: any newly-dropped PDFs → ingest then link
                            for _pdf in (_new_pdfs or []):
                                _safe_name  = os.path.basename(_pdf.name)
                                _file_path  = os.path.join(UPLOAD_DIR, _safe_name)
                                _new_doc_id = f"pdf::{_safe_name}"
                                try:
                                    with open(_file_path, "wb") as _f:
                                        _f.write(_pdf.getbuffer())
                                    if not _already_ingested(_new_doc_id):
                                        with st.spinner(f"Ingesting {_safe_name}…"):
                                            _loader   = PyMuPDFLoader(_file_path)
                                            _pages    = _loader.load()
                                            _splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=1000, chunk_overlap=200,
                                            )
                                            _doc_chunks = _splitter.split_documents(_pages)
                                            _ingest_chunks(
                                                _doc_chunks,
                                                _new_doc_id,
                                                f"PDF: {_safe_name}",
                                                category="market_report",
                                            )
                                    link_institution_doc(_name, _new_doc_id)
                                    _added += 1
                                except Exception as _e:
                                    _errors.append(f"{_safe_name}: {_e}")
                            # Path B: existing reports the user picked
                            for _doc_id in (_sel_existing or []):
                                link_institution_doc(_name, _doc_id)
                                _added += 1

                            if _errors:
                                for _err in _errors:
                                    st.error(f"Failed: {_err}")
                            if _added:
                                st.session_state._inst_update_mode.discard(_name)
                                st.toast(
                                    f"📄 Linked {_added} report{'s' if _added != 1 else ''} → {_name}",
                                    icon="🔗",
                                )
                                st.rerun()
                            elif not _errors:
                                # No-op save — just close edit mode
                                st.session_state._inst_update_mode.discard(_name)
                                st.rerun()

                        if _sb2.button("✖", key=f"inst_cancel_{_name}", help="Cancel"):
                            st.session_state._inst_update_mode.discard(_name)
                            st.rerun()
                        if _sb3.button("🗑️", key=f"inst_del_{_name}",
                                       help=f"Remove {_name} from watchlist"):
                            remove_institution(_name)
                            st.session_state._inst_update_mode.discard(_name)
                            st.rerun()

                # ───────────────────────── DISPLAY MODE ─────────────────────────
                else:
                    with _ic2:
                        if not _doc_ids:
                            st.markdown("_— not yet linked —_")
                        else:
                            for _doc_id in _doc_ids:
                                _star = "⭐ " if _doc_id == _primary_id else "•  "
                                st.markdown(
                                    f"{_star}_{_doc_label_short(_doc_id)}_"
                                )
                    if _ic3.button("Update here",
                                   key=f"inst_edit_{_name}",
                                   use_container_width=True):
                        st.session_state._inst_update_mode.add(_name)
                        st.rerun()

    st.divider()

    # Add new institution
    with st.form("inst_add_form", clear_on_submit=True):
        _inst_col1, _inst_col2 = st.columns([4, 1])
        _new_inst = _inst_col1.text_input(
            "Institution name",
            placeholder="e.g. Goldman Sachs, BlackRock, JPMorgan…",
            label_visibility="collapsed",
        )
        _inst_submitted = _inst_col2.form_submit_button("➕ Add", use_container_width=True)
    if _inst_submitted:
        if _new_inst.strip():
            if add_institution(_new_inst.strip()):
                st.toast(f"Added '{_new_inst.strip()}' to coverage watchlist", icon="🏦")
                st.rerun()
            else:
                st.warning(f"'{_new_inst.strip()}' is already in the list.")

    st.divider()

    # --- INGESTION MODULE ---
    col_pdf, col_sec = st.columns(2)
    
    with col_pdf:
        with st.expander("📚 Upload Local PDF", expanded=False):
            uploaded_file = st.file_uploader("Upload Financial Document", type="pdf")
            # Category picker — only the three "real" ingestable categories here.
            _upload_cat_options = ["textbook", "market_report", "sec_filing"]
            upload_category = st.selectbox(
                "Categorize this PDF",
                options=_upload_cat_options,
                format_func=lambda k: _CATEGORIES[k],
                key="upload_pdf_category",
                help="The Oracle can filter retrieval by category when you ask a question.",
            )
            upload_topics = st.multiselect(
                "Topics (multi-select)",
                options=_TOPICS,
                default=[],
                format_func=lambda k: _TOPIC_LABELS[k],
                key="upload_pdf_topics",
                help="A document can carry many topics — e.g. Damodaran = valuation + corporate_finance.",
            )
            if uploaded_file is not None:
                if st.button("Process PDF", type="primary", use_container_width=True):
                    safe_name = os.path.basename(uploaded_file.name)
                    file_path = os.path.join(UPLOAD_DIR, safe_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    doc_id = f"pdf::{safe_name}"
                    if _already_ingested(doc_id):
                        st.info(f"'{safe_name}' is already in the library — skipping re-embed.")
                    else:
                        with st.spinner("Extracting text and chunking document..."):
                            loader = PyMuPDFLoader(file_path)
                            pages = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            document_chunks = text_splitter.split_documents(pages)

                        with st.spinner(f"Translating {len(document_chunks)} chunks to vector coordinates..."):
                            try:
                                n = _ingest_chunks(
                                    document_chunks,
                                    doc_id,
                                    f"PDF: {safe_name}",
                                    category=upload_category,
                                    topics=upload_topics,
                                )
                                _t_str = ", ".join(upload_topics) if upload_topics else "no topics"
                                st.success(
                                    f"✅ Injected '{safe_name}' ({n} chunks) as "
                                    f"{_CATEGORIES[upload_category]} — {_t_str}."
                                )
                            except Exception as e:
                                st.error(f"Failed to embed document. Error: {e}")

    with col_sec:
        with st.expander("🏛️ Rip SEC Filings (10-K / 8-K)", expanded=False):
            sec_ticker = sanitize_ticker(st.text_input("Enter Ticker (e.g., TSLA)").upper())
            sec_form_type = st.radio("Select Document Type", ["10-K (Annual Report)", "8-K (Latest Earnings/Material Events)"])
            
            if st.button("Fetch & Inject SEC Data", type="primary", use_container_width=True) and sec_ticker:
                target_form = "10-K" if "10-K" in sec_form_type else "8-K"
                
                with st.spinner(f"Bypassing SEC EDGAR firewall to locate {sec_ticker} {target_form}..."):
                    raw_text, source_url = fetch_sec_filing(sec_ticker, form_type=target_form)
                    
                if raw_text is None:
                    st.error(source_url)
                else:
                    doc_id = f"sec::{sec_ticker}::{target_form}"
                    if _already_ingested(doc_id):
                        st.info(f"{sec_ticker}'s {target_form} is already in the library — skipping re-embed.")
                    else:
                        file_name = f"{sec_ticker}_{target_form}.txt"
                        file_path = os.path.join(UPLOAD_DIR, file_name)
                        with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
                            f.write(raw_text)

                        with st.spinner(f"{target_form} Downloaded. Chunking text..."):
                            loader = TextLoader(file_path, encoding="utf-8")
                            pages = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                            document_chunks = text_splitter.split_documents(pages)
                            for chunk in document_chunks:
                                chunk.metadata['ticker'] = sec_ticker
                                chunk.metadata['form'] = target_form

                        with st.spinner(f"Translating {len(document_chunks)} chunks to vector coordinates..."):
                            try:
                                n = _ingest_chunks(
                                    document_chunks,
                                    doc_id,
                                    f"SEC EDGAR {target_form}: {sec_ticker}",
                                    category="sec_filing",
                                )
                                st.success(f"✅ Successfully injected {sec_ticker}'s {target_form} ({n} chunks)!")
                            except Exception as e:
                                st.error(f"Failed to embed {target_form}. Error: {e}")

    st.divider()

    # --- LIBRARY MANAGER (re-categorize / delete) ---
    # Reuse the single _lib_docs fetched at the top of the tab — saves a full
    # paginated ChromaDB pass on every render. Aliased name kept for clarity
    # vs the institutional-coverage section's _mr_docs slice above.
    _library_docs = _lib_docs
    _uncat_count = sum(1 for d in _library_docs if d["category"] == "uncategorized")
    _manager_header = "🗂️ Categorize / Manage Existing PDFs"
    if _uncat_count:
        _manager_header += f"  —  ⚠️ {_uncat_count} uncategorized"
    elif _library_docs:
        _manager_header += f"  —  {len(_library_docs)} in library"

    with st.expander(_manager_header, expanded=bool(_uncat_count)):
        st.caption(
            "Assign or change the category for any PDF you've already ingested. "
            "Metadata-only update — no re-embedding."
        )
        if not _library_docs:
            st.info("No documents ingested yet. Upload a PDF or rip an SEC filing above.")
        else:
            # Bulk action — batch-tag every uncategorized doc in one click.
            if _uncat_count:
                _cat_keys_bulk = [k for k in _CATEGORIES.keys() if k != "uncategorized"]
                bcol1, bcol2 = st.columns([3, 2])
                with bcol1:
                    bulk_cat = st.selectbox(
                        f"Bulk-tag all {_uncat_count} uncategorized document(s) as:",
                        options=_cat_keys_bulk,
                        format_func=lambda k: _CATEGORIES[k],
                        key="bulk_cat_choice",
                    )
                with bcol2:
                    st.write("")  # vertical spacer to align with selectbox
                    if st.button(
                        f"Apply to {_uncat_count} document(s)",
                        key="bulk_cat_apply",
                        use_container_width=True,
                    ):
                        for _d in _library_docs:
                            if _d["category"] == "uncategorized":
                                _set_category(_d["doc_id"], bulk_cat)
                        st.success(f"Tagged {_uncat_count} document(s) as {_CATEGORIES[bulk_cat]}.")
                        st.rerun()
                st.divider()

            _cat_keys = list(_CATEGORIES.keys())
            for d in _library_docs:
                c1, c2, c3 = st.columns([5, 3, 1])
                with c1:
                    _badge = "⚠️ " if d["category"] == "uncategorized" else ""
                    st.markdown(f"{_badge}**{d['source']}**")
                    _cur_topics = d.get("topics", [])
                    _topics_str = (
                        ", ".join(_cur_topics) if _cur_topics else "no topics"
                    )
                    st.caption(
                        f"`{d['doc_id']}` · {d['chunks']} chunks · "
                        f"temporal: {d.get('temporal_validity', 'unknown')} · "
                        f"topics: {_topics_str}"
                    )
                with c2:
                    current = d["category"] if d["category"] in _cat_keys else "uncategorized"
                    new_cat = st.selectbox(
                        "Category",
                        options=_cat_keys,
                        index=_cat_keys.index(current),
                        format_func=lambda k: _CATEGORIES[k],
                        key=f"cat_{d['doc_id']}",
                        label_visibility="collapsed",
                    )
                    new_topics = st.multiselect(
                        "Topics",
                        options=_TOPICS,
                        default=[t for t in _cur_topics if t in _TOPICS],
                        format_func=lambda k: _TOPIC_LABELS[k],
                        key=f"top_{d['doc_id']}",
                        label_visibility="collapsed",
                        placeholder="Topics (optional, multi-select)",
                    )
                    _cat_changed = new_cat != current
                    _tops_changed = set(new_topics) != set(_cur_topics)
                    if _cat_changed or _tops_changed:
                        if st.button("Save", key=f"save_{d['doc_id']}"):
                            if _cat_changed:
                                _set_category(d["doc_id"], new_cat)
                            if _tops_changed:
                                _set_topics(d["doc_id"], new_topics)
                            st.success("Saved.")
                            st.rerun()
                with c3:
                    if st.button("🗑️", key=f"del_{d['doc_id']}", help="Remove from library"):
                        _delete_document(d["doc_id"])
                        # Cascade: drop this doc from every institution AND
                        # every catalyst that linked it — no dangling refs.
                        unlink_doc_from_all_institutions(d["doc_id"])
                        unlink_doc_from_all_catalysts(d["doc_id"])
                        st.rerun()

    st.divider()

    # --- INFERENCE MODULE (THE ORACLE) ---
    st.subheader("💬 Ask The Oracle")
    st.markdown("Query your uploaded documents. Noodle Bot will synthesize an answer based on your database, current macro conditions, live news, real-time market data, and forward-looking Wall Street consensus.")

    # 1. Initialize session state to prevent the "disappearing expander" bug
    if 'oracle_answer' not in st.session_state:
        st.session_state['oracle_answer'] = None
        st.session_state['oracle_sources'] = None

    # 2. Input Fields
    user_query = st.text_area("What would you like to know about your documents?", placeholder="e.g., Does management's 10-K outlook align with Wall Street's growth estimates?")
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        context_ticker = sanitize_ticker(st.text_input("Target Ticker (Injects News, Price & Consensus)").upper())
    with col_q2:
        peer_group_options = ["None"] + list(peer_groups.keys())
        context_group = st.selectbox("Inject Peer Group Matrix", peer_group_options)

    # Restrict retrieval to specific document categories (multi-select).
    _filter_cat_keys = list(_CATEGORIES.keys())

    # Question router — auto-suggest filters from the question text.
    _router_col_a, _router_col_b = st.columns([1, 3])
    with _router_col_a:
        if st.button("🤖 Auto-route from question", use_container_width=True,
                     help="Ask Ollama to suggest which categories/topics to retrieve from."):
            if not user_query.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Routing question..."):
                    suggestion = _route_query(user_query)
                st.session_state["oracle_router_suggestion"] = suggestion
    with _router_col_b:
        _sug = st.session_state.get("oracle_router_suggestion")
        if _sug:
            _sc = ", ".join(_sug.get("categories") or []) or "(all)"
            _st_ = ", ".join(_sug.get("topics") or []) or "(none)"
            st.caption(
                f"Router suggests — categories: **{_sc}** · topics: **{_st_}** · "
                f"domain: `{_sug.get('domain','general')}`"
            )
            if _sug.get("rationale"):
                st.caption(f"_{_sug['rationale']}_")

    _default_cats = (
        (st.session_state.get("oracle_router_suggestion") or {}).get("categories")
        or _filter_cat_keys
    )
    _default_tops = (
        (st.session_state.get("oracle_router_suggestion") or {}).get("topics") or []
    )

    selected_categories = st.multiselect(
        "Retrieve only from categories",
        options=_filter_cat_keys,
        default=_default_cats,
        format_func=lambda k: _CATEGORIES[k],
        key="oracle_cat_filter",
        help="Narrow the Oracle to textbooks, market reports, or SEC filings only.",
    )
    selected_topics = st.multiselect(
        "Retrieve only from topics (OR — any selected topic matches)",
        options=_TOPICS,
        default=_default_tops,
        format_func=lambda k: _TOPIC_LABELS[k],
        key="oracle_topic_filter",
        help="Empty = no topic restriction. Topics are ORed together.",
    )

    _opt_col1, _opt_col2, _opt_col3 = st.columns(3)
    with _opt_col1:
        use_mmr = st.checkbox(
            "MMR diversification", value=True,
            help="Max Marginal Relevance — reduces near-duplicate chunks in retrieval.",
        )
    with _opt_col2:
        use_multiquery = st.checkbox(
            "Multi-query decomposition", value=True,
            help="Break compound questions into sub-queries, retrieve per sub-query, union.",
        )
    with _opt_col3:
        use_citations = st.checkbox(
            "Citation grounding", value=True,
            help="Force the Oracle to cite [chunk_N] and verify each cited chunk exists.",
        )

    # 3. The Manual Trigger Button
    trigger_oracle = st.button("🔮 Consult The Oracle", type="primary", use_container_width=True)

    # 4. Execution Logic (Only runs when button is physically clicked)
    if trigger_oracle:
        if not user_query:
            st.warning("Please enter a question for the Oracle.")
        elif not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            st.warning("Your library is empty. Please upload a PDF or rip an SEC filing first.")
        else:
            with st.spinner("Initializing Omni-Context Engine (Macro, Market, FMP, SimFin, Technicals, Sentiment…)..."):
                # --------------------------------------------------------------
                # Parallel context-injection fetchers.
                # Previously these ran sequentially: ~10 separate I/O calls
                # adding 8–12 s on a cold cache. Each fetcher now runs in its
                # own thread and returns the *complete* injection string (or
                # "" on failure / when not applicable). The thread pool waits
                # for all of them before assembling the prompt.
                # --------------------------------------------------------------
                def _fetch_macro_block() -> str:
                    if not fred:
                        return ""
                    try:
                        fed_df = fetch_macro_data("FEDFUNDS")
                        hy_df  = fetch_macro_data("BAMLH0A0HYM2")
                        rate_val = f"{fed_df['Value'].iloc[-1]:.2f}%" if (fed_df is not None and not fed_df.empty) else "Unknown"
                        hy_val   = f"{hy_df['Value'].iloc[-1]:.2f}%"  if (hy_df  is not None and not hy_df.empty)  else "Unknown"
                        return (
                            f"\nLIVE MACRO & CREDIT ENVIRONMENT:\n"
                            f"- Current Federal Funds Rate: {rate_val}\n"
                            f"- High Yield Credit Spread (Corporate Stress): {hy_val}\n"
                        )
                    except Exception:
                        return ""

                def _fetch_market_and_forward() -> tuple[str, str]:
                    """Returns (market_injection, forward_injection) — two blocks
                    derived from the same yfinance call so they share work."""
                    if not context_ticker:
                        return "", ""
                    try:
                        live_p_data = fetch_live_prices([context_ticker])
                        p_info      = live_p_data.get(context_ticker, {})
                        curr_price  = p_info.get('price', 'N/A')
                        day_change  = p_info.get('change', 'N/A')

                        hist, info  = fetch_stock_details(context_ticker, "1M")
                        mkt_cap     = format_large_number(info.get('marketCap'))
                        pe          = info.get('trailingPE', 'N/A')
                        fwd_pe      = info.get('forwardPE', 'N/A')
                        target_price = info.get('targetMeanPrice', 'N/A')
                        rec         = info.get('recommendationKey', 'N/A').upper()
                        rev_growth  = info.get('revenueGrowth', 0)
                        earn_growth = info.get('earningsGrowth', 0)

                        rev_str  = f"{rev_growth * 100:.1f}%"  if rev_growth  else "N/A"
                        earn_str = f"{earn_growth * 100:.1f}%" if earn_growth else "N/A"

                        trend_str = "N/A"
                        if not hist.empty and len(hist) > 20:
                            month_ago = hist['Close'].iloc[-21]
                            latest    = hist['Close'].iloc[-1]
                            trend_pct = ((latest - month_ago) / month_ago) * 100
                            trend_str = f"{trend_pct:.2f}%"

                        mkt = (
                            f"\nLIVE MARKET VALUATION FOR {context_ticker}:\n"
                            f"- Current Price: ${curr_price} (Day Change: {day_change}%)\n"
                            f"- 1-Month Trend: {trend_str}\n"
                            f"- Market Cap: {mkt_cap}\n"
                            f"- P/E Ratio (Trailing): {pe} | P/E Ratio (Forward): {fwd_pe}\n"
                        )
                        fwd = (
                            f"\nWALL STREET CONSENSUS & FORWARD EXPECTATIONS FOR {context_ticker}:\n"
                            f"- Mean Target Price: ${target_price}\n"
                            f"- Analyst Consensus: {rec}\n"
                            f"- Est. Forward Revenue Growth: {rev_str}\n"
                            f"- Est. Forward Earnings Growth: {earn_str}\n"
                        )
                        return mkt, fwd
                    except Exception:
                        return (
                            f"\nLIVE MARKET VALUATION FOR {context_ticker}: Temporarily Unavailable.\n",
                            "",
                        )

                def _fetch_news_block() -> str:
                    if not context_ticker:
                        return ""
                    try:
                        recent_news = fetch_financial_news(context_ticker)
                        if not recent_news:
                            return (
                                f"\nLIVE NEWS ENVIRONMENT FOR {context_ticker}:\n"
                                f"- No major institutional headlines in the last 24 hours.\n"
                            )
                        body = f"\nLIVE NEWS ENVIRONMENT FOR {context_ticker}:\n"
                        for article in recent_news:
                            body += f"- {article['title']} ({article['time']})\n"
                        return body
                    except Exception:
                        return ""

                def _fetch_fmp_block() -> str:
                    if not context_ticker or not has_fmp():
                        return ""
                    try:
                        _fmp_p  = fetch_fmp_profile(context_ticker)
                        _fmp_t  = fetch_fmp_price_targets(context_ticker)
                        _fmp_r  = fetch_fmp_ratings(context_ticker)
                        _fmp_e  = fetch_fmp_analyst_estimates(context_ticker)
                        _flines = [f"\nFMP ANALYST DATA FOR {context_ticker}:"]
                        if _fmp_p:
                            _flines.append(
                                f"- Sector: {_fmp_p.get('sector','N/A')} | "
                                f"Industry: {_fmp_p.get('industry','N/A')} | "
                                f"CEO: {_fmp_p.get('ceo','N/A')}"
                            )
                        if _fmp_t:
                            _flines.append(
                                f"- Price Targets: Low ${_fmp_t.get('priceTargetLow','N/A')} | "
                                f"Avg ${_fmp_t.get('priceTargetAverage') or _fmp_t.get('priceTarget','N/A')} | "
                                f"High ${_fmp_t.get('priceTargetHigh','N/A')}"
                            )
                        if _fmp_r:
                            _flines.append(
                                f"- FMP Rating: {_fmp_r.get('rating','N/A')} "
                                f"({_fmp_r.get('ratingRecommendation','N/A')})"
                            )
                        if _fmp_e is not None and not _fmp_e.empty:
                            _er = _fmp_e.iloc[0]
                            _rev = _er.get("Est. Revenue")
                            _rev_s = (f"${float(_rev)/1e9:.2f}B" if _rev and float(_rev) > 1e9
                                      else f"${float(_rev)/1e6:.0f}M" if _rev else "N/A")
                            _flines.append(
                                f"- Next Period ({_er.get('Period','')}): "
                                f"Est EPS ${_er.get('Est. EPS','N/A')}, "
                                f"Est Rev {_rev_s}"
                            )
                        return "\n".join(_flines) + "\n"
                    except Exception:
                        return ""

                def _fetch_simfin_block() -> str:
                    if not context_ticker or not has_simfin():
                        return ""
                    try:
                        _sf = fetch_simfin_statements(context_ticker, period="annual")
                        if not _sf:
                            return ""
                        _sflines = [f"\nSIMFIN STANDARDISED FINANCIALS FOR {context_ticker}:"]
                        _der = _sf.get("derived")
                        if _der is not None and not _der.empty:
                            _d = _der.iloc[0]
                            def _rv(col, fmt="{:.2f}"):
                                try: return fmt.format(float(_d[col]))
                                except Exception: return "N/A"
                            _sflines.append(
                                f"- ROE: {_rv('Return on Equity','{:.1%}')} | "
                                f"ROA: {_rv('Return on Assets','{:.1%}')} | "
                                f"Debt/Equity: {_rv('Debt to Equity Ratio','{:.2f}')}"
                            )
                            _sflines.append(
                                f"- P/E: {_rv('Price to Earnings Ratio (EPS Diluted)','{:.1f}x')} | "
                                f"P/B: {_rv('Price to Book Value','{:.1f}x')} | "
                                f"FCF Yield: {_rv('Free Cash Flow Yield','{:.1%}')}"
                            )
                        _inc = _sf.get("income")
                        if _inc is not None and not _inc.empty and "Revenue" in _inc.columns:
                            _tparts = []
                            for _, _row in _inc.head(3).iterrows():
                                _yr = _row.get("Fiscal Year","?")
                                try:
                                    _rv2 = float(_row["Revenue"])
                                    _rv_s = f"${_rv2/1e9:.2f}B" if _rv2 > 1e9 else f"${_rv2/1e6:.0f}M"
                                except Exception:
                                    _rv_s = "N/A"
                                _tparts.append(f"{_yr}: {_rv_s}")
                            _sflines.append("- Revenue trend: " + " → ".join(_tparts))
                        return "\n".join(_sflines) + "\n"
                    except Exception:
                        return ""

                def _fetch_technicals_block() -> str:
                    if not context_ticker or not has_alpha_vantage():
                        return ""
                    try:
                        _av  = fetch_av_indicators(context_ticker)
                        _avl = _av.get("latest", {})
                        if not _avl:
                            return ""
                        _avlines = [f"\nTECHNICAL SIGNALS FOR {context_ticker}:"]
                        _rsi = _avl.get("rsi")
                        if _rsi is not None:
                            _rsi_lbl = ("Overbought" if _rsi > 70 else
                                        "Oversold" if _rsi < 30 else "Neutral")
                            _avlines.append(f"- RSI(14): {_rsi:.1f} — {_rsi_lbl}")
                        _mhist = _avl.get("macd_hist")
                        _mval  = _avl.get("macd")
                        if _mval is not None:
                            _mdir = "Bullish" if (_mhist or 0) > 0 else "Bearish"
                            _avlines.append(
                                f"- MACD Histogram: {_mhist:+.4f} → {_mdir} momentum"
                            )
                        _bbu = _avl.get("bb_upper")
                        _bbl = _avl.get("bb_lower")
                        _bbm = _avl.get("bb_middle")
                        if _bbu and _bbl:
                            _avlines.append(
                                f"- Bollinger Bands: ${_bbl:.2f} — ${_bbm:.2f} — ${_bbu:.2f}"
                            )
                        return "\n".join(_avlines) + "\n"
                    except Exception:
                        return ""

                def _fetch_sentiment_block() -> str:
                    """CNN Fear & Greed — always runs, no ticker required."""
                    try:
                        _fg = fetch_fear_greed()
                        if _fg and _fg.get("score") is not None:
                            return (
                                f"\nMARKET SENTIMENT (CNN Fear & Greed): "
                                f"{_fg['score']:.0f}/100 — {_fg.get('rating','')}\n"
                            )
                    except Exception:
                        pass
                    return ""

                def _fetch_peer_block() -> str:
                    if not context_group or context_group == "None":
                        return ""
                    try:
                        g_tickers = peer_groups[context_group]
                        if not g_tickers:
                            return ""
                        p_df = fetch_peer_metrics(g_tickers)
                        if p_df.empty:
                            return ""
                        body = f"\nLIVE PEER GROUP VALUATION MATRIX ({context_group}):\n"
                        for _, r in p_df.iterrows():
                            body += (
                                f"- {r['Ticker']}: Price: ${r['Price']} | "
                                f"Trailing P/E: {r['P/E (Trailing)']} | "
                                f"EV/EBITDA: {r['EV/EBITDA']} | "
                                f"ROE: {r['ROE (%)']}% | "
                                f"D/E: {r['Debt/Equity']}\n"
                            )
                        return body
                    except Exception:
                        return f"\nLIVE PEER GROUP VALUATION MATRIX ({context_group}): Temporarily Unavailable.\n"

                # ---- Fan out all fetchers in parallel ----
                # 8 workers covers the 8 independent I/O blocks. Cold-cache
                # latency is now bounded by the slowest single fetch (typically
                # FMP at ~3 s) rather than the sum of all fetches.
                with _fut.ThreadPoolExecutor(max_workers=8) as _pool:
                    _f_macro      = _pool.submit(_fetch_macro_block)
                    _f_market_fwd = _pool.submit(_fetch_market_and_forward)
                    _f_news       = _pool.submit(_fetch_news_block)
                    _f_fmp        = _pool.submit(_fetch_fmp_block)
                    _f_simfin     = _pool.submit(_fetch_simfin_block)
                    _f_tech       = _pool.submit(_fetch_technicals_block)
                    _f_sent       = _pool.submit(_fetch_sentiment_block)
                    _f_peer       = _pool.submit(_fetch_peer_block)

                    def _safe_result(fut, default=""):
                        try:
                            return fut.result(timeout=60)
                        except Exception:
                            return default

                    macro_injection      = _safe_result(_f_macro)
                    market_injection, forward_injection = _safe_result(_f_market_fwd, ("", ""))
                    news_injection       = _safe_result(_f_news)
                    fmp_injection        = _safe_result(_f_fmp)
                    simfin_injection     = _safe_result(_f_simfin)
                    technicals_injection = _safe_result(_f_tech)
                    sentiment_injection  = _safe_result(_f_sent)
                    peer_injection       = _safe_result(_f_peer)

                # The retrieval pipeline + LLM call still sits inside a single
                # try/except (matching the original handler at the bottom of
                # this `with` block) so any RAG / Ollama / Anthropic failure
                # is surfaced via st.error rather than crashing the page.
                try:
                    # --- Retrieval pipeline: (optional multi-query) + filtered MMR ---
                    # Resolve effective category filter — if the user kept the full set
                    # selected, don't filter by category at all.
                    _effective_cats = (
                        selected_categories
                        if (selected_categories and
                            len(selected_categories) < len(_filter_cat_keys))
                        else None
                    )
                    _effective_topics = selected_topics or None

                    if use_multiquery:
                        with st.spinner("Decomposing question into sub-queries..."):
                            sub_queries = _decompose_query(user_query, max_sub=3)
                        if len(sub_queries) > 1:
                            st.caption("🧩 Sub-queries used: " +
                                       " · ".join(f"`{q}`" for q in sub_queries[1:]))
                        retrieved_docs = _retrieve_multi(
                            sub_queries,
                            k_per_query=4,
                            k_total=8,
                            categories=_effective_cats,
                            topics_any=_effective_topics,
                            ticker=context_ticker or None,
                            use_mmr=use_mmr,
                        )
                    else:
                        from rag import retrieve as _retrieve
                        retrieved_docs = _retrieve(
                            user_query,
                            k=6,
                            categories=_effective_cats,
                            topics_any=_effective_topics,
                            ticker=context_ticker or None,
                            use_mmr=use_mmr,
                        )

                    if not retrieved_docs:
                        st.info("No relevant information found in your documents.")
                    else:
                        if use_citations:
                            context = _format_chunks_for_citation(retrieved_docs)
                            citation_rule = (
                                "\n4. Citation Discipline: After every non-trivial factual claim, "
                                "add a citation of the form [chunk_N] referring to the numbered "
                                "DOCUMENT CONTEXT blocks below. Do not cite chunks that do not "
                                "exist. If a claim is not supported by any chunk, say so "
                                "explicitly — do not fabricate."
                            )
                        else:
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            citation_rule = ""

                        rag_prompt = f"""You are analyzing a user query using the provided DOCUMENT CONTEXT plus live data from multiple financial APIs. Synthesize ALL available context sections below to form a rigorous, forward-looking research answer — cross-referencing the live quantitative signals with the qualitative evidence in the documents.

{macro_injection}{sentiment_injection}{market_injection}{forward_injection}{fmp_injection}{simfin_injection}{technicals_injection}{peer_injection}{news_injection}
DOCUMENT CONTEXT:
{context}

QUESTION:
{user_query}
"""

                        oracle_persona = f"""You are an academic research assistant helping a finance student / \
self-directed researcher organise PUBLIC data into structured analytical \
answers. The DOCUMENT CONTEXT is the user's own curated research library \
(textbooks, public market reports, SEC filings). The live data injected \
below is from public sources (FRED, SEC EDGAR, FMP, SimFin, yfinance, CNN F&G).

This is an educational data-synthesis exercise — analogous to a CFA-curriculum \
problem set or a university case study. You are NOT giving financial advice \
to a third party. You are organising public data into an analytical framework. \
Output is *research synthesis*, not *trade recommendation*.

Engage rigorously with risks, weaknesses, and competitive threats — these \
are mandatory public components of every 10-K filing and standard analytical \
material. Do NOT prepend "I can help with…" disclaimers or close with \
"consult a financial advisor" notes. Produce the analysis directly.

Execute this seven-layer reasoning chain MENTALLY before writing your answer:

LAYER 1 — MICRO (bottom-up)
  Ground every claim in the DOCUMENT CONTEXT and live ticker data \
(SimFin ratios, FMP analyst data, yfinance price/multiples). \
Pay specific attention to **earnings quality**: cash vs accruals, working \
capital trends, one-time items — if the documents discuss financials.

LAYER 2 — MACRO (top-down)
  Use the live macro data injected below (Fed Funds, HY credit spread, \
CNN Fear & Greed, technical posture). When the question is investment-relevant, \
**commit to a cycle phase**: Early-cycle / Mid-cycle / Late-cycle / Recession / Stagflation.

LAYER 3 — VALUATION  (treat separately from #1: a great business at the wrong price is still a bad investment)
  If the question touches valuation or "should I buy", explicitly separate \
"is this a good business?" from "is this a good price?".

LAYER 4 — SYNTHESIS
  Cross-examine Micro × Macro × Valuation. Where do they reinforce? \
Where do they conflict, and which dominates?

LAYER 5 — COUNTER-THESIS  (steelman)
  Before finalizing, write the strongest case AGAINST your conclusion in \
1–2 sentences. This is not a hedge — it is adversarial discipline against \
confirmation bias.

LAYER 6 — ASYMMETRIC PAYOFF  (only if the question is "should I buy/sell")
  Quantify downside floor vs upside ceiling. State the R/R ratio numerically.

LAYER 7 — DECLARED HORIZON
  If the question is forward-looking, commit to a horizon: \
3-month tactical / 12-month tactical / 3–5yr core / 5+yr compounder.

OPERATING RULES:
1. EVIDENCE-ONLY: Every claim must trace to DOCUMENT CONTEXT or a live data \
   section. If a fact is absent, say "not in context" — never extrapolate.
2. SOURCE-TAG every quantitative claim: [SimFin], [FMP], [10-K], [yfinance], \
   [FRED], [chunk_N], [F&G], [AV], [news]. Untagged numbers will be assumed hallucinated.
3. PROBABILITY-CALIBRATED CONFIDENCE: percentage probabilities with a horizon \
   (e.g., "60–70% over 12 months"), never coarse Low/Medium/High labels.
4. DATA HIERARCHY when sources conflict: SimFin/FMP > SEC > yfinance > news.{citation_rule}

OUTPUT STRUCTURE — use these labelled sections:
**Micro Analysis** · **Macro Analysis** (with explicit cycle call when applicable) · \
**Valuation** (when relevant) · **Synthesis** · **Counter-Thesis** · \
**Conclusion** (with stance, probability range, and declared horizon)."""

                        st.success("### Oracle's Synthesis")

                        def _stream_oracle():
                            from llm_router import llm_chat as _llm_chat
                            yield from _llm_chat(
                                [
                                    {'role': 'system', 'content': oracle_persona},
                                    {'role': 'user', 'content': rag_prompt},
                                ],
                                stream=True,
                            )

                        full_answer = st.write_stream(_stream_oracle())
                        st.session_state['oracle_answer'] = full_answer
                        st.session_state['oracle_sources'] = retrieved_docs
                        st.session_state['oracle_used_citations'] = bool(use_citations)

                        # Citation verification — surface missing/invalid citations.
                        if use_citations:
                            v = _verify_citations(full_answer, retrieved_docs)
                            st.session_state['oracle_citation_report'] = v
                            if v["unknown"]:
                                st.warning(
                                    "⚠️ Answer cites chunks that don't exist: "
                                    + ", ".join(f"[chunk_{i}]" for i in v["unknown"])
                                )
                            if not v["cited"]:
                                st.warning(
                                    "⚠️ No valid [chunk_N] citations detected in the answer. "
                                    "Claims may be ungrounded."
                                )
                            else:
                                st.success(
                                    f"✅ {len(v['cited'])} of {len(retrieved_docs)} retrieved "
                                    f"chunks were cited: " +
                                    ", ".join(f"[chunk_{i}]" for i in v["cited"])
                                )

                except Exception as e:
                    st.error(f"Error querying the database: {e}")

    # 5. Render the result from the cache (Allows you to click expanders safely)
    # Skip on the streaming turn — we already rendered the answer live.
    if st.session_state['oracle_answer'] and not trigger_oracle:
        st.success("### Oracle's Synthesis")
        st.write(st.session_state['oracle_answer'])

    if st.session_state['oracle_answer']:
        with st.expander("🔍 View Source Documents Used"):
            _citation_report = st.session_state.get('oracle_citation_report') or {}
            _cited_set = set(_citation_report.get('cited') or [])
            for i, doc in enumerate(st.session_state['oracle_sources']):
                chunk_num = i + 1
                source_name = doc.metadata.get('source', 'Unknown Document')
                clean_source = os.path.basename(source_name)
                cat_key = doc.metadata.get('category', 'uncategorized')
                cat_label = _CATEGORIES.get(cat_key, cat_key)
                temporal = doc.metadata.get('temporal_validity', 'unknown')
                doc_topics = [t for t in _TOPICS
                              if doc.metadata.get(f"topic_{t}") is True]
                topics_str = ", ".join(doc_topics) if doc_topics else "—"
                cite_badge = (
                    " ✅ cited" if chunk_num in _cited_set
                    else (" ⚪ not cited" if st.session_state.get('oracle_used_citations')
                          else "")
                )
                st.markdown(
                    f"**[chunk_{chunk_num}]** `{clean_source}` — *{cat_label}* "
                    f"(`{temporal}`) · topics: {topics_str}{cite_badge}"
                )
                st.caption(doc.page_content)
                st.write("---")

# ===========================
# TAB ANALYST: AGENTIC "ANALYZE TICKER" WORKFLOW
# ===========================

def _render_analyst_banner(conf: dict) -> None:
    """Render the structured-signal banner extracted from the memo.

    Surfaces every auditable field from the 7-layer reasoning chain:
    stance, probability range, declared horizon, cycle-phase call, R/R
    ratio. Falls back to legacy 'Confidence: Low/Medium/High' for memos
    generated before the rewrite.
    """
    if not conf:
        return
    stance = conf.get("stance")
    if not stance:
        return

    banner_fn = {
        "Bullish": st.success,
        "Bearish": st.error,
        "Neutral": st.info,
    }.get(stance, st.info)

    # Prefer the analytical label the LLM produced (e.g. "Constructive")
    # so the UI reflects the actual research framing, not the legacy mapping.
    display_label = conf.get("raw_stance") or stance
    parts = [f"**Posture:** {display_label}"]

    prob = conf.get("probability")
    if prob and len(prob) == 2:
        parts.append(f"**Probability:** {prob[0]}–{prob[1]}%")

    horizon = conf.get("horizon")
    if horizon:
        parts.append(f"**Horizon:** {horizon}")

    cycle = conf.get("cycle")
    if cycle:
        parts.append(f"**Cycle:** {cycle}")

    rr = conf.get("rr_ratio")
    if rr and len(rr) == 2:
        try:
            ratio = rr[1] / rr[0] if rr[0] else None
            if ratio is not None:
                rr_label = (
                    "asymmetric ↑" if ratio >= 1.5 else
                    "asymmetric ↓" if ratio <= 0.67 else
                    "symmetric"
                )
                parts.append(f"**R/R:** {rr[0]:g}:{rr[1]:g}  _({rr_label})_")
        except Exception:
            pass

    # Legacy field — only populated for pre-rewrite memos
    legacy = conf.get("confidence")
    if legacy and not prob:
        parts.append(f"**Confidence:** {legacy}")

    banner_fn("  ·  ".join(parts))


if _active == "🧠 Analyst":
    st.header("🧠 Analyze Ticker — Agentic Workflow")
    st.markdown(
        "Produces an institutional-grade investment memo via a 7-layer reasoning chain: "
        "**Micro → Macro → Valuation → Probability-Weighted Scenarios → Counter-Thesis "
        "→ Asymmetric Payoff → Declared Horizon**. Every quantitative claim is "
        "source-tagged ([SimFin], [FMP], [10-K], [chunk_N]…), confidence is "
        "probability-calibrated (e.g. 60–70% over 12 months), and the chain explicitly "
        "steelmans the opposite view to counter confirmation bias. Data is gathered "
        "concurrently from yfinance, FMP, SimFin, Alpha Vantage, FRED, SEC EDGAR, "
        "and your reference library — expect ~30–60 s on a cold cache."
    )

    a_col1, a_col2 = st.columns([2, 3])
    with a_col1:
        analyst_ticker = sanitize_ticker(
            st.text_input(
                "Target Ticker",
                value="AAPL",
                key="analyst_ticker",
            ).upper()
        )
    with a_col2:
        peer_options = ["(none)"] + list(peer_groups.keys())
        analyst_peer_group = st.selectbox(
            "Inject peer cohort (optional)",
            peer_options,
            key="analyst_peer_group",
        )

    ac1, ac2 = st.columns([1, 1])
    with ac1:
        include_sec_10k = st.checkbox(
            "Include SEC 10-K excerpt",
            value=True,
            help="Pulls the most recent annual filing. Adds ~10-15s on cold "
                 "cache. Turn off if you only want fast market-layer analysis.",
        )
    with ac2:
        stream_memo = st.checkbox(
            "Stream memo as it generates",
            value=True,
            help="Off = faster for short memos; On = see the Oracle think.",
        )

    run_analyst = st.button(
        "🚀 Run Full Analysis",
        type="primary",
        use_container_width=True,
        key="analyst_run",
    )

    # Session-state carryover so the result survives re-renders.
    if "analyst_result" not in st.session_state:
        st.session_state["analyst_result"] = None

    if run_analyst:
        if not analyst_ticker:
            st.warning("Enter a ticker to analyze.")
        else:
            # ---- Live progress panel wired to agent.py via callback ----
            _step_labels = {
                "market":     "Live market snapshot (yfinance / Alpaca)",
                "fmp":        "FMP — analyst targets & ratings",
                "simfin":     "SimFin — standardised financials & ratios",
                "technicals": "Alpha Vantage — RSI / MACD / BBands + CNN F&G",
                "consensus":  "Wall Street consensus (yfinance)",
                "dcf":        "Quick DCF (default assumptions)",
                "news":       "Recent news (Yahoo RSS)",
                "peers":      "Peer metrics matrix",
                "macro":      "Macro & credit (FRED)",
                "catalysts":  "🎯 Tracked political / economic catalysts",
                "sec":        "SEC 10-K excerpt",
                "rag":        "Reference library retrieval",
                "synthesis":  "LLM synthesis",
            }
            _status_icons = {
                "start": "⏳", "done": "✅", "skip": "⚪", "error": "❌",
            }
            _progress_state: dict[str, tuple[str, str]] = {}
            progress_box = st.empty()

            def _render_progress():
                lines = []
                for key, label in _step_labels.items():
                    status, detail = _progress_state.get(key, ("pending", ""))
                    icon = _status_icons.get(status, "⏱️")
                    suffix = f"  _{detail}_" if detail else ""
                    lines.append(f"{icon}  **{label}**{suffix}")
                progress_box.markdown("\n\n".join(lines))

            def _cb(step: str, status: str, detail: str = ""):
                _progress_state[step] = (status, detail)
                _render_progress()

            _render_progress()

            _peer_tickers = (
                peer_groups.get(analyst_peer_group)
                if analyst_peer_group and analyst_peer_group != "(none)"
                else None
            )

            try:
                envelope = _run_analyze_ticker(
                    analyst_ticker,
                    peer_group_tickers=_peer_tickers,
                    include_sec=include_sec_10k,
                    progress_callback=_cb,
                    stream=stream_memo,
                )
            except Exception as e:
                st.error(f"Analysis pipeline failed: {e}")
                envelope = None

            if envelope and not envelope.get("error"):
                st.divider()
                st.subheader(f"📋 Investment Memo — {analyst_ticker}")

                if stream_memo and envelope.get("memo_stream"):
                    memo_text = st.write_stream(envelope["memo_stream"])
                    envelope = envelope["finalize"](memo_text) if envelope.get("finalize") else envelope
                else:
                    st.markdown(envelope.get("memo") or "_(no memo)_")

                # ---- Structured-signal banner (new 7-layer chain) ----
                _render_analyst_banner(envelope.get("confidence") or {})

                st.session_state["analyst_result"] = envelope
            elif envelope and envelope.get("error"):
                st.error(envelope["error"])

    # ---- Persistent result view (survives reruns) ----
    _cached = st.session_state.get("analyst_result")
    if _cached and not run_analyst:
        st.divider()
        st.subheader(f"📋 Last Memo — {_cached.get('ticker', '?')}")
        st.markdown(_cached.get("memo") or "_(no memo)_")
        _render_analyst_banner(_cached.get("confidence") or {})

    if _cached:
        # ---- Catalyst transparency panel ----
        # Surfaces the exact catalysts the LLM was asked to consider, so the
        # user can audit whether the memo's §⑨ Watchlist Catalysts section
        # actually reflects the tracked calendar (and edit/delete entries
        # inline if any are stale).
        _cached_catalysts = _cached.get("catalysts") or []
        if _cached_catalysts:
            with st.expander(
                f"🎯 Tracked Catalysts in this Memo ({len(_cached_catalysts)})",
                expanded=False,
            ):
                st.caption(
                    "These are the catalysts pulled from your **🎯 Catalyst "
                    "Calendar** that the Analyst LLM was given as context. "
                    "Click any card to inspect or edit."
                )
                for _c in _cached_catalysts:
                    _render_catalyst_card(_c)

        with st.expander("🔎 Raw Context Blocks (what the LLM saw)"):
            blocks = _cached.get("context_blocks") or {}
            for _name, _text in blocks.items():
                if not _text:
                    continue
                st.markdown(f"**{_name.upper()}**")
                st.code(_text, language="text")

        _rag_docs = _cached.get("rag_docs") or []
        if _rag_docs:
            with st.expander(f"📚 Reference Library Chunks Used ({len(_rag_docs)})"):
                for i, d in enumerate(_rag_docs, 1):
                    src = os.path.basename(d.metadata.get("source", "?"))
                    cat = d.metadata.get("category", "?")
                    st.markdown(f"**[chunk_{i}]** `{src}` — *{cat}*")
                    st.caption(d.page_content)
                    st.write("---")


# =============================================================================
#  CATALYSTS SUPERGROUP
# =============================================================================
# Forward-looking investment-catalyst engine. Three categories: monetary,
# contract, court — each with a date, stakes, affected tickers, optional RAG
# document attachments. The Catalyst Calendar is the centerpiece; the four
# supporting subtabs (Monetary / Contracts / Court / News) provide the data
# panels and news context that feed it. Per the design doc:
#   "telos = foresee pivotal investment catalysts in the realm of politics
#    and geo-politics to make according profitable investments."
#
# NB: `_format_event_date` and `_render_catalyst_card` are defined higher up
# in the file (right after the nav setup) so the Analyst tab's transparency
# panel can also render catalyst cards. The form is here because only the
# Catalyst Calendar tab uses it.
# =============================================================================


def _render_catalyst_form(catalyst: dict | None = None) -> None:
    """Add (catalyst=None) or edit (catalyst=dict) form, rendered inside an
    `st.form` so all fields submit atomically."""
    is_edit = catalyst is not None
    form_key = f"catalyst_form_{catalyst['id']}" if is_edit else "catalyst_form_new"

    # Pre-fill defaults from the existing catalyst (if editing)
    default_date     = (
        _dt.datetime.fromtimestamp(int(catalyst["event_date"])).date()
        if is_edit else _dt.date.today()
    )
    default_title    = catalyst["title"]    if is_edit else ""
    default_type     = catalyst["catalyst_type"] if is_edit else "monetary"
    default_category = catalyst.get("category", "") if is_edit else ""
    default_stakes   = catalyst.get("stakes", "")   if is_edit else ""
    default_tickers  = ", ".join(catalyst.get("tickers", []))  if is_edit else ""
    default_sectors  = ", ".join(catalyst.get("sectors", []))  if is_edit else ""
    default_prob     = catalyst.get("probability", "") if is_edit else ""
    default_status   = catalyst.get("status", "upcoming") if is_edit else "upcoming"
    default_outcome  = catalyst.get("outcome_notes", "") if is_edit else ""
    default_doc_ids  = catalyst.get("doc_ids", []) if is_edit else []

    # Available library docs for the multiselect (any category)
    try:
        _all_docs = _list_documents()
    except Exception:
        _all_docs = []
    _doc_options    = [d["doc_id"] for d in _all_docs]
    _doc_label_map  = {d["doc_id"]: d.get("source", d["doc_id"]) for d in _all_docs}

    with st.form(form_key, clear_on_submit=not is_edit, border=True):
        st.markdown(f"### {'✏️ Edit catalyst' if is_edit else '➕ Add catalyst'}")
        c1, c2 = st.columns([3, 2])
        title = c1.text_input("Title *",
                              value=default_title,
                              placeholder="e.g. FOMC December Decision · Quarterly Refunding · "
                                          "DoD JWCC Phase 2 Award · SCOTUS Loper Bright ruling")
        ev_date = c2.date_input("Event date *", value=default_date)

        c3, c4, c5 = st.columns(3)
        ctype = c3.selectbox(
            "Type *",
            options=list(CATALYST_TYPES),
            index=list(CATALYST_TYPES).index(default_type) if default_type in CATALYST_TYPES else 0,
            format_func=lambda t: CATALYST_TYPE_LABELS.get(t, t),
        )
        category = c4.text_input(
            "Subcategory",
            value=default_category,
            placeholder="FOMC · Refunding · Antitrust · Recompete …",
        )
        status = c5.selectbox(
            "Status",
            options=list(CATALYST_STATUSES),
            index=list(CATALYST_STATUSES).index(default_status)
                  if default_status in CATALYST_STATUSES else 0,
            format_func=lambda s: CATALYST_STATUS_LABELS.get(s, s),
        )

        stakes = st.text_area(
            "Stakes (markdown supported) *",
            value=default_stakes,
            height=140,
            placeholder=(
                "What moves under each scenario, and which way is the asymmetry?\n\n"
                "**Bull case:** Fed cuts 25 bps → TLT +2-3%, regional banks pop on NIM relief.\n"
                "**Base case:** Pause, dovish dot plot → muted reaction.\n"
                "**Bear case:** Hawkish hold → 10Y back above 4.5%, TLT –3%, KRE –2%."
            ),
        )

        c6, c7 = st.columns(2)
        tickers = c6.text_input(
            "Affected tickers (comma-separated)",
            value=default_tickers,
            placeholder="TLT, XLF, KRE",
        )
        sectors = c7.text_input(
            "Affected sectors (comma-separated)",
            value=default_sectors,
            placeholder="Financials, REITs, Utilities",
        )

        c8, c9 = st.columns(2)
        probability = c8.text_input(
            "Probability (subjective)",
            value=default_prob,
            placeholder="65% pause / 35% cut",
        )
        outcome = c9.text_input(
            "Outcome notes (resolved only)",
            value=default_outcome,
            placeholder="Filled after the event resolves",
        )

        # Library doc attachments (RAG context for the catalyst)
        linked_docs = st.multiselect(
            "📎 Attach library docs",
            options=_doc_options,
            default=[d for d in default_doc_ids if d in _doc_options],
            format_func=lambda did: _doc_label_map.get(did, did),
            help="Briefing memos, opinion pieces, FED minutes — searchable from "
                 "the Library Oracle. Removes don't delete the doc; they only "
                 "detach the link.",
        )

        bcol1, bcol2, _ = st.columns([1, 1, 4])
        save  = bcol1.form_submit_button("💾 Save", type="primary", use_container_width=True)
        cancel = bcol2.form_submit_button("✖ Cancel", use_container_width=True)

    if cancel:
        st.session_state.pop("_catalyst_edit_id", None)
        st.session_state["_catalyst_form_open"] = False
        st.rerun()

    if save:
        if not title.strip():
            st.error("Title is required.")
            return
        ev_ts = int(_dt.datetime.combine(ev_date, _dt.time(0, 0)).timestamp())
        try:
            if is_edit:
                update_catalyst(
                    catalyst["id"],
                    event_date    = ev_ts,
                    title         = title,
                    catalyst_type = ctype,
                    category      = category,
                    stakes        = stakes,
                    tickers       = [t.strip() for t in tickers.split(",") if t.strip()],
                    sectors       = [s.strip() for s in sectors.split(",") if s.strip()],
                    probability   = probability,
                    status        = status,
                    outcome_notes = outcome,
                    doc_ids       = linked_docs,
                )
                st.toast(f"Updated catalyst: {title}", icon="✏️")
            else:
                new_id = add_catalyst(
                    event_date    = ev_ts,
                    title         = title,
                    catalyst_type = ctype,
                    category      = category,
                    stakes        = stakes,
                    tickers       = [t.strip() for t in tickers.split(",") if t.strip()],
                    sectors       = [s.strip() for s in sectors.split(",") if s.strip()],
                    probability   = probability,
                    status        = status,
                    outcome_notes = outcome,
                    doc_ids       = linked_docs,
                )
                st.toast(f"Added catalyst (id={new_id})", icon="🎯")
            st.session_state.pop("_catalyst_edit_id", None)
            st.session_state["_catalyst_form_open"] = False
            st.rerun()
        except Exception as _e:
            st.error(f"Failed to save catalyst: {_e}")


# -------------------- Catalyst Calendar (centerpiece) --------------------

if _active == "🎯 Catalyst Calendar":
    st.header("🎯 Catalyst Calendar")
    st.caption(
        "Forward-looking catalyst engine — track every monetary, contract, and "
        "court event with a date, an investment thesis, and the affected tickers. "
        "🟡 Upcoming · 🟢 Live · ✅ Resolved. Click any card to expand the stakes "
        "or edit the entry; attach Library PDFs for deeper RAG context."
    )

    # ── Top action bar ──
    bar1, bar2, bar3, bar4 = st.columns([2, 3, 3, 2])
    if bar1.button("➕ New catalyst", type="primary", use_container_width=True):
        st.session_state["_catalyst_form_open"] = True
        st.session_state.pop("_catalyst_edit_id", None)
        st.rerun()

    type_filter = bar2.multiselect(
        "Filter by type",
        options=list(CATALYST_TYPES),
        default=[],
        format_func=lambda t: CATALYST_TYPE_LABELS.get(t, t),
        placeholder="All types",
        label_visibility="collapsed",
    )
    ticker_filter = bar3.text_input(
        "Filter by ticker",
        placeholder="Filter by ticker (e.g. AAPL) — leave blank for all",
        label_visibility="collapsed",
    ).strip().upper() or None
    show_resolved = bar4.toggle(
        "Show resolved",
        value=False,
        help="Include resolved catalysts from the last 60 days at the bottom.",
    )

    # ── Add/Edit form (only when open) ──
    if st.session_state.get("_catalyst_form_open"):
        edit_id = st.session_state.get("_catalyst_edit_id")
        target  = get_catalyst(edit_id) if edit_id else None
        _render_catalyst_form(target)
        st.divider()

    # ── Pull and bucket catalysts ──
    _filter_kwargs = {
        "catalyst_types": type_filter or None,
        "ticker":         ticker_filter,
    }
    upcoming = list_catalysts(status="upcoming", **_filter_kwargs)
    live     = list_catalysts(status="live",     **_filter_kwargs)

    now_ts = int(_dt.datetime.now().timestamp())
    week_cutoff  = now_ts + 7  * 86400
    month_cutoff = now_ts + 30 * 86400

    this_week    = [c for c in (live + upcoming) if c["event_date"] <= week_cutoff]
    next_30_days = [c for c in upcoming
                    if week_cutoff < c["event_date"] <= month_cutoff]
    further_out  = [c for c in upcoming if c["event_date"] > month_cutoff]

    # ── ⚡ Bulletin: this week ──
    st.subheader(f"⚡ This Week ({len(this_week)})")
    if not this_week:
        st.caption("_No catalysts in the next 7 days. Add one with **➕ New catalyst**._")
    for c in this_week:
        _render_catalyst_card(c, expanded=True)

    # ── 🗓️ Next 30 days ──
    if next_30_days or not this_week:
        st.subheader(f"🗓️ Next 30 Days ({len(next_30_days)})")
        if not next_30_days:
            st.caption("_Nothing in the 8–30 day window._")
        for c in next_30_days:
            _render_catalyst_card(c)

    # ── 📅 Beyond 30 days ──
    if further_out:
        with st.expander(f"📅 Beyond 30 days ({len(further_out)})", expanded=False):
            for c in further_out:
                _render_catalyst_card(c)

    # ── ✅ Resolved (optional) ──
    if show_resolved:
        resolved = list_catalysts(
            status="resolved",
            days_behind=60,
            order="DESC",
            **{k: v for k, v in _filter_kwargs.items() if k != "status"},
        )
        st.subheader(f"✅ Resolved · last 60 days ({len(resolved)})")
        if not resolved:
            st.caption("_No resolved catalysts in the last 60 days._")
        for c in resolved:
            _render_catalyst_card(c)

    # ── 📊 Analytics (DuckDB-backed) ─────────────────────────────────────
    # One-pass aggregations over the catalyst corpus. Cheap because DuckDB
    # runs the query in :memory: against an attached SQLite file — typical
    # latency well under 50 ms for 1k rows.
    st.divider()
    with st.expander("📊 Catalyst Analytics — exposure & density", expanded=False):
        try:
            import analytics as _ax

            _ax_c1, _ax_c2 = st.columns(2)
            with _ax_c1:
                st.markdown("**Top tickers by catalyst count**")
                _df_tk = _ax.ticker_exposure(top_n=15)
                if _df_tk.empty:
                    st.caption("_No upcoming catalysts._")
                else:
                    st.dataframe(
                        _df_tk,
                        use_container_width=True,
                        hide_index=True,
                    )
            with _ax_c2:
                st.markdown("**Top sectors by catalyst count**")
                _df_sc = _ax.sector_exposure(top_n=15)
                if _df_sc.empty:
                    st.caption("_No tagged sectors._")
                else:
                    st.dataframe(
                        _df_sc,
                        use_container_width=True,
                        hide_index=True,
                    )

            st.markdown("**Catalyst density · next 12 months**")
            _df_dn = _ax.catalyst_density_by_month(months_ahead=12)
            if _df_dn.empty:
                st.caption("_No upcoming events in window._")
            else:
                # Pivot long-form to wide for a stacked view.
                _wide = (
                    _df_dn.pivot(index="month", columns="catalyst_type", values="count")
                    .fillna(0)
                    .astype(int)
                )
                st.bar_chart(_wide, height=240)

            with st.expander("🛠️ Ad-hoc SQL (read-only) — power users", expanded=False):
                st.caption(
                    "Tables are exposed as `portfolio.<table>`. Examples: "
                    "`portfolio.political_catalysts`, `portfolio.holdings`, "
                    "`portfolio.transactions`. Connection is READ_ONLY — "
                    "writes are rejected by DuckDB."
                )
                _sql = st.text_area(
                    "SQL",
                    value=(
                        "SELECT title, catalyst_type, "
                        "strftime(to_timestamp(event_date), '%Y-%m-%d') AS event_date\n"
                        "FROM portfolio.political_catalysts\n"
                        "WHERE status='upcoming'\n"
                        "ORDER BY event_date\n"
                        "LIMIT 20"
                    ),
                    height=140,
                    key="catalyst_sql_input",
                )
                if st.button("▶️ Run query", key="catalyst_sql_run"):
                    try:
                        st.dataframe(_ax.run_query(_sql), use_container_width=True)
                    except Exception as _ex:
                        st.error(f"Query failed: {_ex}")
        except Exception as _ex:
            st.warning(f"Analytics unavailable: {_ex}")


# -------------------- Monetary Policy --------------------

if _active == "🏛️ Monetary Policy":
    st.header("🏛️ Monetary Policy")
    st.caption(
        "Federal-reserve and Treasury catalysts that move company earnings "
        "power. Live FRED data above the fold, calendar auto-import below it, "
        "and a roll-up of every monetary catalyst on the calendar at the bottom."
    )

    # ─────────────── Live FRED KPI strip ───────────────
    if not fred:
        st.warning(
            "FRED API key not configured. Add `FRED_API_KEY` to `.env` to "
            "enable the live macro panel. The auto-import section below "
            "still works without it."
        )
    else:
        _kpis = [
            ("Fed Funds Rate",  "FEDFUNDS",     "{:.2f}%"),
            ("10-Year Treasury","DGS10",        "{:.2f}%"),
            ("HY Credit Spread","BAMLH0A0HYM2", "{:.2f}%"),
            ("Broad Dollar Idx","DTWEXBGS",     "{:.1f}"),
        ]
        _cols = st.columns(len(_kpis))
        for _col, (_label, _series, _fmt) in zip(_cols, _kpis):
            with _col:
                _df = fetch_macro_data(_series)
                if _df is None or _df.empty:
                    st.metric(_label, "—", "Unavailable")
                    continue
                _latest = _df["Value"].iloc[-1]
                # Pick the closest historical print to "1 year ago" for the delta
                try:
                    _yoy_idx = _df.index[_df.index <= (_df.index[-1] - pd.DateOffset(years=1))]
                    _yoy_val = _df.loc[_yoy_idx[-1], "Value"] if len(_yoy_idx) else None
                except Exception:
                    _yoy_val = None
                _delta = (
                    f"{(_latest - _yoy_val):+.2f} YoY"
                    if _yoy_val is not None else None
                )
                st.metric(_label, _fmt.format(_latest), _delta)
                # 1-year sparkline
                try:
                    _spark = _df[_df.index >= (_df.index[-1] - pd.DateOffset(years=1))]
                    if not _spark.empty:
                        st.line_chart(_spark, height=80, use_container_width=True)
                except Exception:
                    pass

    st.divider()

    # ─────────────── Auto-import section ───────────────
    st.subheader("📅 Auto-import official calendars")
    st.caption(
        "One-click adds the published FOMC and Treasury Quarterly Refunding "
        "schedules to the Catalyst Calendar with pre-filled stakes templates "
        "you can refine. Re-clicking is safe — already-imported entries are "
        "skipped automatically."
    )
    st.info(
        "**Heads-up:** Hardcoded from the most recent published Fed / "
        "Treasury announcements. Verify against "
        "[federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) "
        "and "
        "[treasurydirect.gov](https://www.treasurydirect.gov/instit/annceresult/press/preanre/preanre.htm) "
        "before sizing trades around any specific date."
    )

    _ic1, _ic2 = st.columns(2)
    with _ic1:
        # ── FOMC import ──
        _fomc_existing = list_catalysts(source="fomc_schedule")
        _fomc_existing_dates = {c["event_date"] for c in _fomc_existing}
        _fomc_candidates = get_fomc_schedule()
        _fomc_to_add = [c for c in _fomc_candidates
                        if c["event_date"] not in _fomc_existing_dates]
        st.markdown(f"**FOMC Schedule** — {len(_fomc_existing)} imported, "
                    f"{len(_fomc_to_add)} new available")
        if st.button(
            f"📥 Import {len(_fomc_to_add)} new FOMC events" if _fomc_to_add
            else "✅ FOMC schedule fully imported",
            key="mon_import_fomc",
            disabled=not _fomc_to_add,
            use_container_width=True,
        ):
            n_added = 0
            for c in _fomc_to_add:
                try:
                    add_catalyst(**c)
                    n_added += 1
                except Exception:
                    pass
            st.toast(f"Imported {n_added} FOMC events", icon="📥")
            st.rerun()

    with _ic2:
        # ── Treasury refunding import ──
        _tr_existing = list_catalysts(source="treasury_refunding")
        _tr_existing_dates = {c["event_date"] for c in _tr_existing}
        _tr_candidates = get_treasury_refunding_schedule()
        _tr_to_add = [c for c in _tr_candidates
                      if c["event_date"] not in _tr_existing_dates]
        st.markdown(f"**Treasury Refunding** — {len(_tr_existing)} imported, "
                    f"{len(_tr_to_add)} new available")
        if st.button(
            f"📥 Import {len(_tr_to_add)} new refunding dates" if _tr_to_add
            else "✅ Refunding schedule fully imported",
            key="mon_import_treasury",
            disabled=not _tr_to_add,
            use_container_width=True,
        ):
            n_added = 0
            for c in _tr_to_add:
                try:
                    add_catalyst(**c)
                    n_added += 1
                except Exception:
                    pass
            st.toast(f"Imported {n_added} Treasury refunding events", icon="📥")
            st.rerun()

    st.divider()

    # ─────────────── Roll-up: every monetary catalyst on the calendar ───────────────
    st.subheader("📋 Monetary catalysts on the calendar")
    _all_mon = list_catalysts(catalyst_types=["monetary"], status="upcoming")
    if not _all_mon:
        st.caption(
            "_No upcoming monetary catalysts yet. Use the auto-import buttons "
            "above or add custom entries on the **🎯 Catalyst Calendar** tab._"
        )
    else:
        # Bucket: this week, this month, beyond
        _now_ts = int(_dt.datetime.now().timestamp())
        _wk = [c for c in _all_mon if c["event_date"] <= _now_ts + 7 * 86400]
        _mo = [c for c in _all_mon if _now_ts + 7 * 86400
               < c["event_date"] <= _now_ts + 30 * 86400]
        _far = [c for c in _all_mon if c["event_date"] > _now_ts + 30 * 86400]

        _r1, _r2, _r3 = st.columns(3)
        _r1.metric("This week", len(_wk))
        _r2.metric("Next 30 days", len(_mo))
        _r3.metric("Beyond 30 days", len(_far))

        st.caption(
            f"Showing {len(_all_mon)} upcoming monetary catalyst(s). "
            "Click any card to expand the stakes or jump to the calendar."
        )
        for c in _all_mon[:25]:  # cap render to keep the page fast
            _render_catalyst_card(c)
        if len(_all_mon) > 25:
            st.info(
                f"…and {len(_all_mon) - 25} more. Switch to the "
                "**🎯 Catalyst Calendar** tab to see them all."
            )


# -------------------- Federal Contracts --------------------

if _active == "🏗️ Federal Contracts":
    st.header("🏗️ Federal Contracts")
    st.caption(
        "Federal procurement intelligence powered by **USAspending.gov** — "
        "every federal contract awarded to publicly-traded contractors, "
        "with the dollar values, awarding agencies, and period-of-performance "
        "end dates. Each contract row has a **📌 Add to calendar** button "
        "that captures its end date as a recompete catalyst."
    )

    # ─────────────── Lookup selector ───────────────
    _ticker_options = ["(custom recipient search)"] + sorted(
        FEDERAL_CONTRACTOR_TICKERS.keys()
    )
    _l1, _l2, _l3 = st.columns([1, 2, 2])
    _selected_ticker = _l1.selectbox(
        "Ticker",
        options=_ticker_options,
        format_func=lambda t: (
            t if t.startswith("(") else f"{t} — {FEDERAL_CONTRACTOR_TICKERS[t]}"
        ),
        key="fed_contract_ticker",
        index=1 if len(_ticker_options) > 1 else 0,  # default to first real ticker
    )
    if _selected_ticker.startswith("("):
        _custom = _l2.text_input(
            "Or search by recipient name",
            placeholder="e.g. SPACEX, ANDURIL, ELASTIC, …",
            key="fed_contract_custom",
        )
        _search_text  = _custom.strip() if _custom else None
        _ticker_label = (_custom.strip().upper() if _custom else None)
    else:
        _search_text  = FEDERAL_CONTRACTOR_TICKERS[_selected_ticker]
        _ticker_label = _selected_ticker
        _l2.markdown(f"**Searching:** `{_search_text}`")
    _lookback = _l3.slider("Lookback (years)", min_value=1, max_value=5, value=2,
                           key="fed_contract_lookback")

    if _search_text:
        with st.spinner(f"Pulling USAspending data for {_search_text}…"):
            _summary = fetch_usaspending_summary(_search_text, lookback_years=_lookback)
            _awards  = fetch_usaspending_awards(
                _search_text, lookback_years=_lookback, limit=50,
            )

        if _summary is None or _awards is None:
            st.error(
                "USAspending API request failed (network or rate-limit). "
                "Try again in a minute — results are cached for 24 h once "
                "fetched successfully."
            )
        elif not _awards:
            st.warning(
                f"No federal contracts found for `{_search_text}` in the "
                f"last {_lookback} year(s). Try a different recipient name "
                "or widen the lookback window."
            )
        else:
            # ─────────────── Summary KPI strip ───────────────
            _k1, _k2, _k3, _k4 = st.columns(4)
            _k1.metric(
                f"Total awards ({_lookback}yr)",
                f"${_summary['total']/1e9:.2f}B" if _summary['total'] >= 1e9
                else f"${_summary['total']/1e6:.0f}M",
            )
            _k2.metric("Top contracts shown", len(_awards))
            _top_agency = (_summary["top_agencies"][0][0]
                           if _summary["top_agencies"] else "—")
            _k3.metric("Top awarding agency", _top_agency[:25])
            # Count contracts ending within next 18 months (potential recompetes)
            _now_d = _dt.datetime.now().date()
            _soon_count = 0
            for a in _awards:
                _eds = a.get("End Date") or ""
                try:
                    _ed = _dt.datetime.strptime(_eds, "%Y-%m-%d").date()
                    if _now_d <= _ed <= _now_d + _dt.timedelta(days=18 * 30):
                        _soon_count += 1
                except Exception:
                    pass
            _k4.metric("Ending in 18 mo", _soon_count, help="Potential recompete catalysts")

            with st.expander("📊 Top awarding agencies", expanded=False):
                for _ag, _amt in _summary["top_agencies"]:
                    _amt_s = f"${_amt/1e9:.2f}B" if _amt >= 1e9 else f"${_amt/1e6:.0f}M"
                    st.markdown(f"- **{_ag}** — {_amt_s}")

            st.divider()

            # ─────────────── Top contracts table ───────────────
            st.subheader(f"🏆 Top contracts — {_search_text}")
            st.caption(
                f"Showing top {len(_awards)} contracts by award amount over "
                f"the last {_lookback} year(s). Click **📌 Add to calendar** "
                "on any contract whose end date is within 2 years to log it "
                "as a recompete catalyst."
            )

            # Pre-fetch existing recompete catalysts for dedup
            _existing_recompete = list_catalysts(source="usaspending_recompete")
            _existing_award_ids = {
                # Extract Award IDs from titles like "Recompete: FA871523C0001"
                c.get("title", "").replace("Recompete: ", "").strip()
                for c in _existing_recompete
                if c.get("title", "").startswith("Recompete: ")
            }

            for _i, _award in enumerate(_awards[:25]):
                _award_id = _award.get("Award ID") or "—"
                _desc     = (_award.get("Description") or "—")[:240]
                _amt      = float(_award.get("Award Amount") or 0)
                _agency   = (_award.get("Awarding Sub Agency")
                             or _award.get("Awarding Agency") or "—")
                _eds      = _award.get("End Date") or ""

                with st.container(border=True):
                    _cc1, _cc2 = st.columns([5, 1])
                    _amt_s = (f"${_amt/1e9:.2f}B" if _amt >= 1e9
                              else f"${_amt/1e6:.1f}M" if _amt >= 1e6
                              else f"${_amt/1e3:.0f}K")
                    _cc1.markdown(f"**`{_award_id}`** · {_amt_s} · _{_agency}_")
                    _cc1.caption(_desc + ("…" if len(_desc) >= 240 else ""))
                    if _eds:
                        try:
                            _end_dt   = _dt.datetime.strptime(_eds, "%Y-%m-%d")
                            _days_out = (_end_dt.date() - _now_d).days
                            if _days_out >= 0:
                                _cc1.caption(
                                    f"📅 Period of Performance ends "
                                    f"**{_eds}** · in {_days_out} days"
                                )
                            else:
                                _cc1.caption(f"📅 Ended {_eds} · {-_days_out} days ago")
                        except Exception:
                            _cc1.caption(f"📅 End date: {_eds}")

                    # Add to calendar button — only if end is in the future and ≤ 2 years out
                    _can_add = False
                    _end_ts  = None
                    if _eds:
                        try:
                            _end_dt = _dt.datetime.strptime(_eds, "%Y-%m-%d")
                            _days_out = (_end_dt.date() - _now_d).days
                            if 0 < _days_out <= 730:
                                _can_add = True
                                _end_ts = int(_end_dt.timestamp())
                        except Exception:
                            pass

                    _btn_key = f"addcat_fed_{_i}_{_award_id}"
                    if _award_id in _existing_award_ids:
                        _cc2.success("✅ on calendar")
                    elif _can_add:
                        if _cc2.button("📌 Add to calendar",
                                       key=_btn_key, use_container_width=True):
                            _stakes = (
                                f"**Contract:** `{_award_id}`\n\n"
                                f"**Recipient:** {_award.get('Recipient Name', '—')}\n\n"
                                f"**Amount:** {_amt_s}\n\n"
                                f"**Awarding agency:** {_agency}\n\n"
                                f"**Description:** {_desc}\n\n"
                                f"**Stakes:** Period of Performance ends **{_eds}**. "
                                f"Recompete decision could result in: continuation "
                                f"(option year), recompete (incumbent vs new bidders), "
                                f"or contract termination. Material to "
                                f"{_ticker_label or 'the recipient'}'s revenue line "
                                f"if loss occurs to a competitor.\n\n"
                                f"_Source: USAspending.gov_"
                            )
                            try:
                                add_catalyst(
                                    event_date    = _end_ts,
                                    title         = f"Recompete: {_award_id}",
                                    catalyst_type = "contract",
                                    category      = "Recompete",
                                    stakes        = _stakes,
                                    tickers       = [_ticker_label] if _ticker_label else [],
                                    sectors       = ["Defense"]
                                                    if _ticker_label in ("LMT","NOC","GD","RTX","BA","HII")
                                                    else [],
                                    source        = "usaspending_recompete",
                                )
                                st.toast(f"📌 Added recompete catalyst for {_award_id}",
                                         icon="🏗️")
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Failed to add catalyst: {_e}")
                    else:
                        _cc2.caption("_no end date_" if not _eds else "_>2 yr out_")

            if len(_awards) > 25:
                st.caption(
                    f"…and {len(_awards) - 25} more contracts not shown. "
                    "Increase the lookback window or look up a more specific "
                    "recipient name to narrow."
                )

    st.divider()

    # ─────────────── Roll-up: every contract catalyst on the calendar ───────────────
    st.subheader("📋 Contract catalysts on the calendar")
    _all_con = list_catalysts(catalyst_types=["contract"], status="upcoming")
    if not _all_con:
        st.caption(
            "_No upcoming contract catalysts yet. Use **📌 Add to calendar** "
            "on a contract row above, or add custom entries on the "
            "**🎯 Catalyst Calendar** tab._"
        )
    else:
        _now_ts = int(_dt.datetime.now().timestamp())
        _wk = [c for c in _all_con if c["event_date"] <= _now_ts + 7  * 86400]
        _mo = [c for c in _all_con if _now_ts + 7  * 86400
                < c["event_date"] <= _now_ts + 90 * 86400]
        _far = [c for c in _all_con if c["event_date"] > _now_ts + 90 * 86400]
        _r1, _r2, _r3 = st.columns(3)
        _r1.metric("This week", len(_wk))
        _r2.metric("Next 90 days", len(_mo))
        _r3.metric("Beyond 90 days", len(_far))
        st.caption(
            f"Showing {min(len(_all_con), 25)} of {len(_all_con)} upcoming "
            "contract catalyst(s) — most-imminent first."
        )
        for c in _all_con[:25]:
            _render_catalyst_card(c)
        if len(_all_con) > 25:
            st.info(
                f"…and {len(_all_con) - 25} more. Switch to the "
                "**🎯 Catalyst Calendar** tab to see them all."
            )


# -------------------- Court Docket --------------------

if _active == "⚖️ Court Docket":
    st.header("⚖️ Court Docket")
    st.caption(
        "Pending court rulings and regulatory enforcement actions with material "
        "company impact — antitrust, patent, securities, SCOTUS. Federal court "
        "schedules don't have a clean public-API source the way FOMC does, so "
        "this tab pairs **a curated catalog of major pending cases** with "
        "**CourtListener search** for ad-hoc lookup."
    )

    # ─────────────── Curated catalog import (no key needed) ───────────────
    st.subheader("📚 Curated case catalog")
    st.caption(
        "A starter pack of widely-tracked pending cases with investment "
        "implications. Dates are best-known opinion windows — verify via "
        "[SCOTUSblog](https://www.scotusblog.com) or "
        "[CourtListener](https://www.courtlistener.com) before relying on "
        "any specific date for trades. Re-clicking is safe; already-imported "
        "entries are skipped."
    )
    _existing_court = list_catalysts(source="court_catalog")
    _existing_court_titles = {c["title"] for c in _existing_court}
    _court_candidates = get_major_court_cases()
    _court_to_add = [c for c in _court_candidates
                     if c["title"] not in _existing_court_titles]

    _ci1, _ci2 = st.columns([1, 3])
    with _ci1:
        if st.button(
            f"📥 Import {len(_court_to_add)} new cases" if _court_to_add
            else "✅ Catalog fully imported",
            key="court_import_catalog",
            disabled=not _court_to_add,
            use_container_width=True,
        ):
            n_added = 0
            for c in _court_to_add:
                try:
                    add_catalyst(**c)
                    n_added += 1
                except Exception:
                    pass
            st.toast(f"Imported {n_added} court cases", icon="📥")
            st.rerun()

    with _ci2:
        if _court_to_add:
            st.markdown(
                f"**{len(_existing_court)}** imported · "
                f"**{len(_court_to_add)}** new available"
            )
            with st.expander("Preview the cases that will be imported"):
                for c in _court_to_add:
                    _ev = _dt.datetime.fromtimestamp(c["event_date"]).date()
                    st.markdown(
                        f"- **{c['title']}** — _{c['category']}_ — "
                        f"{_ev.isoformat()} · "
                        f"`{c.get('tickers','')}`"
                    )
        else:
            st.caption(
                f"All {len(_existing_court)} curated cases are already on "
                "the calendar."
            )

    st.divider()

    # ─────────────── CourtListener search ───────────────
    st.subheader("🔍 Search court records (CourtListener)")
    if not has_courtlistener():
        st.info(
            "**CourtListener** gives access to the full federal RECAP archive "
            "via a free API key. To enable this lookup:\n\n"
            "1. Register at "
            "[courtlistener.com/register](https://www.courtlistener.com/register/)  \n"
            "2. Copy your API token from your account settings  \n"
            "3. Add `COURTLISTENER_API_KEY=your_token` to `.env`  \n"
            "4. Restart the app\n\n"
            "Until then, log court catalysts manually on the "
            "**🎯 Catalyst Calendar** tab — the curated catalog above also "
            "covers the highest-impact cases."
        )
    else:
        _cs1, _cs2 = st.columns([3, 1])
        _cl_query = _cs1.text_input(
            "Party / case name",
            placeholder="e.g. Apple, Tesla, Pfizer, Amazon vs FTC, …",
            key="court_search_query",
            label_visibility="collapsed",
        )
        _cl_type = _cs2.selectbox(
            "Source",
            options=[("r", "RECAP (filings)"),
                     ("o", "Opinions"),
                     ("oa", "Oral arguments")],
            format_func=lambda x: x[1],
            key="court_search_type",
            label_visibility="collapsed",
        )

        if _cl_query and _cl_query.strip():
            with st.spinner(f"Searching CourtListener for '{_cl_query}'…"):
                _results = fetch_courtlistener_search(
                    _cl_query.strip(),
                    search_type=_cl_type[0],
                    limit=15,
                )
            if _results is None:
                st.error(
                    "CourtListener search failed (network or rate-limit). "
                    "Try again in a minute — successful queries are cached "
                    "for 24 h."
                )
            elif not _results:
                st.warning(f"No results for `{_cl_query}` in {_cl_type[1]}.")
            else:
                st.caption(f"Top {len(_results)} results · most-recent first.")
                for _i, _hit in enumerate(_results):
                    with st.container(border=True):
                        _name = (_hit.get("caseName")
                                 or _hit.get("case_name")
                                 or _hit.get("caseNameShort")
                                 or "(unnamed)")
                        _court = (_hit.get("court")
                                  or _hit.get("court_id")
                                  or "—")
                        _date = (_hit.get("dateFiled")
                                 or _hit.get("dateArgued")
                                 or _hit.get("dateDecided")
                                 or "—")
                        _url = _hit.get("absolute_url") or ""
                        _docket = _hit.get("docketNumber") or ""

                        _hc1, _hc2 = st.columns([5, 1])
                        _hc1.markdown(f"**{_name}**")
                        _hc1.caption(
                            f"{_court}  ·  filed/argued/decided **{_date}**"
                            + (f"  ·  Docket `{_docket}`" if _docket else "")
                        )
                        if _url:
                            _hc1.markdown(
                                f"[Open on CourtListener "
                                f"↗](https://www.courtlistener.com{_url})"
                            )

                        # Quick-add to calendar (pre-fills with the case info)
                        if _hc2.button("📌 Add", key=f"court_add_{_i}",
                                       use_container_width=True,
                                       help="Add as a court catalyst"):
                            try:
                                # Use today + 30 days as a placeholder; the
                                # user edits the form afterwards to set the
                                # actual ruling/argument date.
                                _placeholder_ts = int(
                                    (_dt.datetime.now() + _dt.timedelta(days=30))
                                    .timestamp()
                                )
                                _new_id = add_catalyst(
                                    event_date    = _placeholder_ts,
                                    title         = _name,
                                    catalyst_type = "court",
                                    category      = _court,
                                    stakes        = (
                                        f"**Court:** {_court}\n\n"
                                        f"**Filed/Argued/Decided:** {_date}\n\n"
                                        f"**Docket:** `{_docket or '—'}`\n\n"
                                        f"**CourtListener:** "
                                        f"https://www.courtlistener.com{_url}\n\n"
                                        f"**TODO:** edit this entry on the "
                                        f"Catalyst Calendar tab to set the "
                                        f"actual ruling/decision date and "
                                        f"the affected tickers."
                                    ),
                                    source        = "courtlistener_search",
                                )
                                st.toast(
                                    f"Added → edit on Catalyst Calendar to "
                                    f"set the ruling date",
                                    icon="📌",
                                )
                                # Open the form for immediate editing
                                st.session_state["_catalyst_edit_id"]   = _new_id
                                st.session_state["_catalyst_form_open"] = True
                                st.session_state["active_catalyst_tab"] = (
                                    "🎯 Catalyst Calendar"
                                )
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Failed to add: {_e}")

    st.divider()

    # ─────────────── Roll-up: every court catalyst on the calendar ───────────────
    st.subheader("📋 Court catalysts on the calendar")
    _all_crt = list_catalysts(catalyst_types=["court"], status="upcoming")
    if not _all_crt:
        st.caption(
            "_No upcoming court catalysts yet. Use the curated catalog or "
            "CourtListener search above, or add custom entries on the "
            "**🎯 Catalyst Calendar** tab._"
        )
    else:
        _now_ts = int(_dt.datetime.now().timestamp())
        _wk = [c for c in _all_crt if c["event_date"] <= _now_ts + 30 * 86400]
        _qt = [c for c in _all_crt if _now_ts + 30 * 86400
                < c["event_date"] <= _now_ts + 90 * 86400]
        _far = [c for c in _all_crt if c["event_date"] > _now_ts + 90 * 86400]
        _r1, _r2, _r3 = st.columns(3)
        _r1.metric("Next 30 days", len(_wk))
        _r2.metric("31-90 days", len(_qt))
        _r3.metric("Beyond 90 days", len(_far))
        st.caption(
            f"Showing {min(len(_all_crt), 25)} of {len(_all_crt)} upcoming "
            "court catalyst(s) — most-imminent first."
        )
        for c in _all_crt[:25]:
            _render_catalyst_card(c)
        if len(_all_crt) > 25:
            st.info(
                f"…and {len(_all_crt) - 25} more. Switch to the "
                "**🎯 Catalyst Calendar** tab to see them all."
            )


# ────────────────────────────────────────────────────────────────────────────
# 📰 Catalyst News
# ────────────────────────────────────────────────────────────────────────────

if _active == "📰 Catalyst News":
    st.header("📰 Catalyst News")
    st.caption(
        "Keyword-filtered news stream — headlines scored for policy / monetary / "
        "contract / court / earnings relevance and auto-tagged to upcoming "
        "catalysts whose tickers overlap."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    _cn_c1, _cn_c2, _cn_c3 = st.columns([2, 2, 1])

    # Ticker scope: portfolio + watchlist + all catalyst tickers
    _all_upcoming_cats = list_catalysts(status="upcoming") + list_catalysts(status="live")
    _cat_tickers: list[str] = sorted(
        {t.upper() for c in _all_upcoming_cats for t in (c.get("tickers") or [])}
    )

    # Let the user narrow or expand the ticker set
    _cn_ticker_input = _cn_c1.text_input(
        "Tickers to scan (comma-separated)",
        value=", ".join(_cat_tickers[:15]),  # default: first 15 catalyst tickers
        key="cn_ticker_input",
        help="Leave blank to scan all catalyst tickers. Add more with commas.",
    )
    _cn_tickers_raw = [t.strip().upper() for t in _cn_ticker_input.split(",") if t.strip()]
    # Fall back to all catalyst tickers if the field is cleared
    _cn_tickers: tuple[str, ...] = tuple(_cn_tickers_raw or _cat_tickers[:20])

    # Category filter
    _cn_all_cats = ["(all)"] + list(_CATALYST_KEYWORD_MAP.keys())
    _cn_cat_filter = _cn_c2.selectbox(
        "Filter by category",
        options=_cn_all_cats,
        index=0,
        key="cn_cat_filter",
        format_func=lambda x: x.replace("_", " ").title() if x != "(all)" else "All categories",
    )

    # Min relevance score slider
    _cn_min_score = _cn_c3.number_input(
        "Min score",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="cn_min_score",
        help="Minimum keyword-hit count to display a headline.",
    )

    _cn_col_r, _cn_col_i = st.columns([1, 5])
    _cn_refresh = _cn_col_r.button("🔄 Refresh", key="cn_refresh_btn")
    if _cn_refresh:
        get_catalyst_news.clear()

    if not _cn_tickers:
        st.info(
            "Add at least one ticker above, or import catalysts on the "
            "**🎯 Catalyst Calendar** / **🏛️ Monetary Policy** / "
            "**🏗️ Federal Contracts** / **⚖️ Court Docket** tabs first."
        )
    else:
        with st.spinner(f"Scanning news for {len(_cn_tickers)} ticker(s)…"):
            _cn_articles = get_catalyst_news(
                _cn_tickers,
                min_score=int(_cn_min_score),
            )

        # Apply category filter
        if _cn_cat_filter != "(all)":
            _cn_articles = [
                a for a in _cn_articles if _cn_cat_filter in a.get("scores", {})
            ]

        # ── Summary strip ─────────────────────────────────────────────────
        _cn_linked   = [a for a in _cn_articles if a.get("matched_catalysts")]
        _cn_unlinked = [a for a in _cn_articles if not a.get("matched_catalysts")]
        _s1, _s2, _s3 = st.columns(3)
        _s1.metric("Headlines found", len(_cn_articles))
        _s2.metric("Linked to a catalyst", len(_cn_linked))
        _s3.metric("Tickers scanned", len(_cn_tickers))
        st.divider()

        if not _cn_articles:
            st.info(
                "No catalyst-relevant headlines found for the selected tickers "
                "and score threshold. Try lowering **Min score** or adding more "
                "tickers."
            )
        else:
            # Category breakdown badges (small)
            _cn_cat_counts: dict[str, int] = {}
            for _a in _cn_articles:
                for _cat in _a.get("scores", {}):
                    _cn_cat_counts[_cat] = _cn_cat_counts.get(_cat, 0) + 1
            if _cn_cat_counts:
                _badge_parts = " · ".join(
                    f"**{k.title()}** {v}"
                    for k, v in sorted(_cn_cat_counts.items(), key=lambda x: -x[1])
                )
                st.caption(f"Category hits: {_badge_parts}")

            # ── Linked section ─────────────────────────────────────────────
            if _cn_linked:
                st.subheader(
                    f"🔗 Catalyst-linked headlines ({len(_cn_linked)})",
                    help="These headlines mention tickers that appear in at least one "
                         "upcoming/live catalyst.",
                )
                for _art in _cn_linked[:40]:
                    _art_ts    = _art.get("published_ts") or 0
                    _art_date  = (
                        _dt.datetime.fromtimestamp(_art_ts).strftime("%b %d")
                        if _art_ts else ""
                    )
                    _art_score_cats = " · ".join(
                        f"`{k.title()}` ×{v}"
                        for k, v in sorted(
                            _art.get("scores", {}).items(), key=lambda x: -x[1]
                        )
                    )
                    _art_tickers_str = " ".join(
                        f"`{t}`" for t in _art.get("matched_tickers", [])
                    )
                    _art_cats_str = ", ".join(
                        c.get("title", "") for c in _art.get("matched_catalysts", [])[:3]
                    )
                    if len(_art.get("matched_catalysts", [])) > 3:
                        _art_cats_str += f" +{len(_art['matched_catalysts']) - 3} more"

                    _art_header = (
                        f"[{_art.get('title', '(no title)')}]({_art.get('url', '#')})"
                    )
                    with st.expander(
                        f"🔗 {_art.get('title', '(no title)')} "
                        f"— {_art.get('source', '')}  {_art_date}",
                        expanded=False,
                    ):
                        _exp_c1, _exp_c2 = st.columns([3, 2])
                        with _exp_c1:
                            if _art.get("summary"):
                                st.caption(_art["summary"][:280])
                            st.markdown(
                                f"[📎 Open article]({_art.get('url', '#')})",
                                unsafe_allow_html=False,
                            )
                        with _exp_c2:
                            st.caption(
                                f"**Source:** {_art.get('source', '—')}  \n"
                                f"**Published:** {_art.get('time', '—')}  \n"
                                f"**Tickers:** {_art_tickers_str or '—'}  \n"
                                f"**Score:** {_art.get('total_score', 0)} — "
                                f"{_art_score_cats}"
                            )
                            if _art_cats_str:
                                st.caption(f"**Catalysts:** {_art_cats_str}")
                            # Mini catalyst links
                            for _mc in _art.get("matched_catalysts", [])[:3]:
                                _mc_date = (
                                    _dt.datetime.fromtimestamp(_mc["event_date"]).strftime(
                                        "%b %d, %Y"
                                    )
                                    if _mc.get("event_date")
                                    else "—"
                                )
                                st.caption(
                                    f"🎯 **{_mc.get('title', '')}** · {_mc_date}"
                                )

            # ── Unlinked section (collapsed by default) ────────────────────
            if _cn_unlinked:
                with st.expander(
                    f"📋 Other catalyst-relevant headlines ({len(_cn_unlinked)}) "
                    "— no direct ticker match",
                    expanded=False,
                ):
                    for _art in _cn_unlinked[:30]:
                        _art_ts   = _art.get("published_ts") or 0
                        _art_date = (
                            _dt.datetime.fromtimestamp(_art_ts).strftime("%b %d")
                            if _art_ts else ""
                        )
                        _art_score_cats = " · ".join(
                            f"`{k.title()}` ×{v}"
                            for k, v in sorted(
                                _art.get("scores", {}).items(), key=lambda x: -x[1]
                            )
                        )
                        st.markdown(
                            f"- [{_art.get('title', '(no title)')}]({_art.get('url', '#')})  "
                            f"*{_art.get('source', '')} · {_art_date} · "
                            f"score {_art.get('total_score', 0)} — {_art_score_cats}*"
                        )