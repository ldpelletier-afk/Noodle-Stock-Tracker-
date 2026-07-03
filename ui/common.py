"""Shared helpers, constants, and data-access utilities for all UI tab modules."""
import datetime as _dt

import streamlit as st

from api import (
    fetch_calendar_events,
    fetch_fear_greed,
    fetch_live_prices,
    fetch_macro_data,
    fetch_sparkline_history,
    fred,
)
from data_store import (
    CATALYST_STATUS_LABELS,
    CATALYST_TYPE_LABELS,
    delete_catalyst,
    load_data as _load_data_sqlite,
    save_data as _save_data_sqlite,
)

DB_DIR = "./chroma_db"
UPLOAD_DIR = "./temp_pdfs"


@st.cache_data(ttl=60)
def _load_data_cached(_token: int):
    return _load_data_sqlite()


def load_data() -> dict:
    return _load_data_cached(st.session_state.get("_app_data_token", 0))


def _invalidate_app_data() -> None:
    st.session_state["_app_data_token"] = (
        st.session_state.get("_app_data_token", 0) + 1
    )


def save_data(data: dict) -> None:
    _save_data_sqlite(data)
    _invalidate_app_data()


def bg_prefetch(app_data: dict, deep: bool = False) -> None:
    """Warm the most-touched caches. Called from app.py background thread and
    from the Dashboard 'Power-load all' button (synchronous, with spinner)."""
    _wl_tickers = sorted(
        t
        for items in app_data.get("watchlists", {}).values()
        for t in items
        if t and t.upper() != "CASH"
    )
    _fav_tickers = list(
        t
        for t in (app_data.get("favorites") or {}).keys()
        if t and t.upper() != "CASH"
    )
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
    try:
        from rag import list_documents as _ld
        _ld()
    except Exception:
        pass
    if _wl_tickers:
        try:
            fetch_calendar_events(tuple(sorted(_wl_tickers))[:10])
        except Exception:
            pass


def _format_event_date(ts: int) -> str:
    if not ts:
        return "—"
    d = _dt.datetime.fromtimestamp(int(ts))
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
    """Expandable catalyst card with inline edit/delete."""
    type_label = CATALYST_TYPE_LABELS.get(c.get("catalyst_type"), c.get("catalyst_type", ""))
    status_label = CATALYST_STATUS_LABELS.get(c.get("status"), c.get("status", ""))
    date_label = _format_event_date(c.get("event_date"))

    header = f"**{c.get('title', '(untitled)')}**"
    summary_line = f"{type_label}  ·  {status_label}  ·  {date_label}"

    with st.expander(f"{header} — {summary_line}", expanded=expanded):
        if c.get("stakes"):
            st.markdown(c["stakes"])
        else:
            st.caption("_No stakes recorded yet — click ✏️ Edit to add._")

        meta_cols = st.columns(4)
        if c.get("category"):
            meta_cols[0].caption(f"**Category**\n\n{c['category']}")
        if c.get("probability"):
            meta_cols[1].caption(f"**Probability**\n\n{c['probability']}")
        if c.get("tickers"):
            meta_cols[2].caption("**Tickers**\n\n" + " ".join(f"`{t}`" for t in c["tickers"]))
        if c.get("sectors"):
            meta_cols[3].caption("**Sectors**\n\n" + ", ".join(c["sectors"]))

        if c.get("status") == "resolved" and c.get("outcome_notes"):
            st.success(f"**Outcome:** {c['outcome_notes']}")

        if c.get("doc_ids"):
            with st.expander(f"📎 Linked library docs ({len(c['doc_ids'])})"):
                for did in c["doc_ids"]:
                    st.markdown(f"- `{did}`")

        ac1, ac2, ac3 = st.columns([1, 1, 6])
        if ac1.button("✏️ Edit", key=f"cat_edit_{c['id']}"):
            st.session_state["_catalyst_edit_id"] = c["id"]
            st.session_state["_catalyst_form_open"] = True
            st.rerun()
        if ac3.button(
            "🗑️ Delete",
            key=f"cat_del_{c['id']}",
            help="Permanently remove this catalyst",
        ):
            delete_catalyst(c["id"])
            st.toast(f"Deleted: {c.get('title', '(untitled)')}", icon="🗑️")
            st.rerun()
