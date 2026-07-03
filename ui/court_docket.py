import datetime as _dt

import streamlit as st

from api import fetch_courtlistener_search, has_courtlistener
from api import get_major_court_cases
from data_store import add_catalyst, list_catalysts
from ui.common import _render_catalyst_card


def render(app_data: dict) -> None:
    st.header("⚖️ Court Docket")
    st.caption(
        "Pending court rulings and regulatory enforcement actions with material "
        "company impact — antitrust, patent, securities, SCOTUS. Federal court "
        "schedules don't have a clean public-API source the way FOMC does, so "
        "this tab pairs **a curated catalog of major pending cases** with "
        "**CourtListener search** for ad-hoc lookup."
    )

    st.subheader("📚 Curated case catalog")
    st.caption(
        "A starter pack of widely-tracked pending cases with investment "
        "implications. Dates are best-known opinion windows — verify via "
        "[SCOTUSblog](https://www.scotusblog.com) or "
        "[CourtListener](https://www.courtlistener.com) before relying on "
        "any specific date for trades. Re-clicking is safe; already-imported "
        "entries are skipped."
    )
    _existing_court        = list_catalysts(source="court_catalog")
    _existing_court_titles = {c["title"] for c in _existing_court}
    _court_candidates      = get_major_court_cases()
    _court_to_add          = [c for c in _court_candidates
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
                f"All {len(_existing_court)} curated cases are already on the calendar."
            )

    st.divider()

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
            options=[("r", "RECAP (filings)"), ("o", "Opinions"), ("oa", "Oral arguments")],
            format_func=lambda x: x[1],
            key="court_search_type",
            label_visibility="collapsed",
        )

        if _cl_query and _cl_query.strip():
            with st.spinner(f"Searching CourtListener for '{_cl_query}'…"):
                _results = fetch_courtlistener_search(
                    _cl_query.strip(), search_type=_cl_type[0], limit=15
                )
            if _results is None:
                st.error(
                    "CourtListener search failed (network or rate-limit). "
                    "Try again in a minute — successful queries are cached for 24 h."
                )
            elif not _results:
                st.warning(f"No results for `{_cl_query}` in {_cl_type[1]}.")
            else:
                st.caption(f"Top {len(_results)} results · most-recent first.")
                for _i, _hit in enumerate(_results):
                    with st.container(border=True):
                        _name  = (_hit.get("caseName") or _hit.get("case_name")
                                  or _hit.get("caseNameShort") or "(unnamed)")
                        _court = _hit.get("court") or _hit.get("court_id") or "—"
                        _date  = (_hit.get("dateFiled") or _hit.get("dateArgued")
                                  or _hit.get("dateDecided") or "—")
                        _url    = _hit.get("absolute_url") or ""
                        _docket = _hit.get("docketNumber") or ""

                        _hc1, _hc2 = st.columns([5, 1])
                        _hc1.markdown(f"**{_name}**")
                        _hc1.caption(
                            f"{_court}  ·  filed/argued/decided **{_date}**"
                            + (f"  ·  Docket `{_docket}`" if _docket else "")
                        )
                        if _url:
                            _hc1.markdown(
                                f"[Open on CourtListener ↗](https://www.courtlistener.com{_url})"
                            )

                        if _hc2.button("📌 Add", key=f"court_add_{_i}",
                                       use_container_width=True,
                                       help="Add as a court catalyst"):
                            try:
                                _placeholder_ts = int(
                                    (_dt.datetime.now() + _dt.timedelta(days=30)).timestamp()
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
                                    "Added → edit on Catalyst Calendar to set the ruling date",
                                    icon="📌",
                                )
                                st.session_state["_catalyst_edit_id"]   = _new_id
                                st.session_state["_catalyst_form_open"] = True
                                st.session_state["active_catalyst_tab"] = "🎯 Catalyst Calendar"
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Failed to add: {_e}")

    st.divider()

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
        _wk  = [c for c in _all_crt if c["event_date"] <= _now_ts + 30 * 86400]
        _qt  = [c for c in _all_crt
                if _now_ts + 30 * 86400 < c["event_date"] <= _now_ts + 90 * 86400]
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
