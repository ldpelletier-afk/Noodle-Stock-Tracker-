import datetime as _dt

import streamlit as st

from data_store import (
    CATALYST_STATUS_LABELS,
    CATALYST_STATUSES,
    CATALYST_TYPE_LABELS,
    CATALYST_TYPES,
    add_catalyst,
    get_catalyst,
    list_catalysts,
    update_catalyst,
)
from rag import list_documents as _list_documents
from ui.common import _render_catalyst_card


def _render_catalyst_form(catalyst: dict | None = None) -> None:
    is_edit = catalyst is not None
    form_key = f"catalyst_form_{catalyst['id']}" if is_edit else "catalyst_form_new"

    default_date = (
        _dt.datetime.fromtimestamp(int(catalyst["event_date"])).date()
        if is_edit else _dt.date.today()
    )
    default_title    = catalyst["title"]               if is_edit else ""
    default_type     = catalyst["catalyst_type"]        if is_edit else "monetary"
    default_category = catalyst.get("category", "")    if is_edit else ""
    default_stakes   = catalyst.get("stakes", "")      if is_edit else ""
    default_tickers  = ", ".join(catalyst.get("tickers", []))  if is_edit else ""
    default_sectors  = ", ".join(catalyst.get("sectors", []))  if is_edit else ""
    default_prob     = catalyst.get("probability", "")  if is_edit else ""
    default_status   = catalyst.get("status", "upcoming") if is_edit else "upcoming"
    default_outcome  = catalyst.get("outcome_notes", "") if is_edit else ""
    default_doc_ids  = catalyst.get("doc_ids", [])      if is_edit else []

    try:
        _all_docs = _list_documents()
    except Exception:
        _all_docs = []
    _doc_options   = [d["doc_id"] for d in _all_docs]
    _doc_label_map = {d["doc_id"]: d.get("source", d["doc_id"]) for d in _all_docs}

    with st.form(form_key, clear_on_submit=not is_edit, border=True):
        st.markdown(f"### {'✏️ Edit catalyst' if is_edit else '➕ Add catalyst'}")
        c1, c2 = st.columns([3, 2])
        title  = c1.text_input(
            "Title *", value=default_title,
            placeholder="e.g. FOMC December Decision · Quarterly Refunding · "
                        "DoD JWCC Phase 2 Award · SCOTUS Loper Bright ruling",
        )
        ev_date = c2.date_input("Event date *", value=default_date)

        c3, c4, c5 = st.columns(3)
        ctype = c3.selectbox(
            "Type *",
            options=list(CATALYST_TYPES),
            index=list(CATALYST_TYPES).index(default_type)
                  if default_type in CATALYST_TYPES else 0,
            format_func=lambda t: CATALYST_TYPE_LABELS.get(t, t),
        )
        category = c4.text_input(
            "Subcategory", value=default_category,
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
            "Stakes (markdown supported) *", value=default_stakes, height=140,
            placeholder=(
                "What moves under each scenario, and which way is the asymmetry?\n\n"
                "**Bull case:** Fed cuts 25 bps → TLT +2-3%, regional banks pop on NIM relief.\n"
                "**Base case:** Pause, dovish dot plot → muted reaction.\n"
                "**Bear case:** Hawkish hold → 10Y back above 4.5%, TLT –3%, KRE –2%."
            ),
        )

        c6, c7 = st.columns(2)
        tickers = c6.text_input(
            "Affected tickers (comma-separated)", value=default_tickers,
            placeholder="TLT, XLF, KRE",
        )
        sectors = c7.text_input(
            "Affected sectors (comma-separated)", value=default_sectors,
            placeholder="Financials, REITs, Utilities",
        )

        c8, c9 = st.columns(2)
        probability = c8.text_input(
            "Probability (subjective)", value=default_prob,
            placeholder="65% pause / 35% cut",
        )
        outcome = c9.text_input(
            "Outcome notes (resolved only)", value=default_outcome,
            placeholder="Filled after the event resolves",
        )

        linked_docs = st.multiselect(
            "📎 Attach library docs",
            options=_doc_options,
            default=[d for d in default_doc_ids if d in _doc_options],
            format_func=lambda did: _doc_label_map.get(did, did),
            help="Briefing memos, opinion pieces, FED minutes — searchable from "
                 "the Library Oracle.",
        )

        bcol1, bcol2, _ = st.columns([1, 1, 4])
        save   = bcol1.form_submit_button("💾 Save", type="primary", use_container_width=True)
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


def render(app_data: dict) -> None:
    st.header("🎯 Catalyst Calendar")
    st.caption(
        "Forward-looking catalyst engine — track every monetary, contract, and "
        "court event with a date, an investment thesis, and the affected tickers. "
        "🟡 Upcoming · 🟢 Live · ✅ Resolved. Click any card to expand the stakes "
        "or edit the entry; attach Library PDFs for deeper RAG context."
    )

    bar1, bar2, bar3, bar4 = st.columns([2, 3, 3, 2])
    if bar1.button("➕ New catalyst", type="primary", use_container_width=True):
        st.session_state["_catalyst_form_open"] = True
        st.session_state.pop("_catalyst_edit_id", None)
        st.rerun()

    type_filter = bar2.multiselect(
        "Filter by type", options=list(CATALYST_TYPES), default=[],
        format_func=lambda t: CATALYST_TYPE_LABELS.get(t, t),
        placeholder="All types", label_visibility="collapsed",
    )
    ticker_filter = bar3.text_input(
        "Filter by ticker",
        placeholder="Filter by ticker (e.g. AAPL) — leave blank for all",
        label_visibility="collapsed",
    ).strip().upper() or None
    show_resolved = bar4.toggle("Show resolved", value=False,
                                help="Include resolved catalysts from the last 60 days.")

    if st.session_state.get("_catalyst_form_open"):
        edit_id = st.session_state.get("_catalyst_edit_id")
        target  = get_catalyst(edit_id) if edit_id else None
        _render_catalyst_form(target)
        st.divider()

    _filter_kwargs = {
        "catalyst_types": type_filter or None,
        "ticker":         ticker_filter,
    }
    upcoming = list_catalysts(status="upcoming", **_filter_kwargs)
    live     = list_catalysts(status="live",     **_filter_kwargs)

    now_ts       = int(_dt.datetime.now().timestamp())
    week_cutoff  = now_ts + 7  * 86400
    month_cutoff = now_ts + 30 * 86400

    this_week    = [c for c in (live + upcoming) if c["event_date"] <= week_cutoff]
    next_30_days = [c for c in upcoming if week_cutoff < c["event_date"] <= month_cutoff]
    further_out  = [c for c in upcoming if c["event_date"] > month_cutoff]

    st.subheader(f"⚡ This Week ({len(this_week)})")
    if not this_week:
        st.caption("_No catalysts in the next 7 days. Add one with **➕ New catalyst**._")
    for c in this_week:
        _render_catalyst_card(c, expanded=True)

    if next_30_days or not this_week:
        st.subheader(f"🗓️ Next 30 Days ({len(next_30_days)})")
        if not next_30_days:
            st.caption("_Nothing in the 8–30 day window._")
        for c in next_30_days:
            _render_catalyst_card(c)

    if further_out:
        with st.expander(f"📅 Beyond 30 days ({len(further_out)})", expanded=False):
            for c in further_out:
                _render_catalyst_card(c)

    if show_resolved:
        resolved = list_catalysts(
            status="resolved", days_behind=60, order="DESC",
            **{k: v for k, v in _filter_kwargs.items() if k != "status"},
        )
        st.subheader(f"✅ Resolved · last 60 days ({len(resolved)})")
        if not resolved:
            st.caption("_No resolved catalysts in the last 60 days._")
        for c in resolved:
            _render_catalyst_card(c)

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
                    st.dataframe(_df_tk, use_container_width=True, hide_index=True)
            with _ax_c2:
                st.markdown("**Top sectors by catalyst count**")
                _df_sc = _ax.sector_exposure(top_n=15)
                if _df_sc.empty:
                    st.caption("_No tagged sectors._")
                else:
                    st.dataframe(_df_sc, use_container_width=True, hide_index=True)

            st.markdown("**Catalyst density · next 12 months**")
            _df_dn = _ax.catalyst_density_by_month(months_ahead=12)
            if _df_dn.empty:
                st.caption("_No upcoming events in window._")
            else:
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
                    "`portfolio.transactions`. Connection is READ_ONLY."
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
                    height=140, key="catalyst_sql_input",
                )
                if st.button("▶️ Run query", key="catalyst_sql_run"):
                    try:
                        st.dataframe(_ax.run_query(_sql), use_container_width=True)
                    except Exception as _ex:
                        st.error(f"Query failed: {_ex}")
        except Exception as _ex:
            st.warning(f"Analytics unavailable: {_ex}")
