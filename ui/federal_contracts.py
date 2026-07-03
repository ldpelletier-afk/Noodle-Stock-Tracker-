import datetime as _dt

import streamlit as st

from api import (
    FEDERAL_CONTRACTOR_TICKERS,
    fetch_usaspending_awards,
    fetch_usaspending_summary,
)
from data_store import add_catalyst, list_catalysts
from ui.common import _render_catalyst_card


def render(app_data: dict) -> None:
    st.header("🏗️ Federal Contracts")
    st.caption(
        "Federal procurement intelligence powered by **USAspending.gov** — "
        "every federal contract awarded to publicly-traded contractors, "
        "with the dollar values, awarding agencies, and period-of-performance "
        "end dates. Each contract row has a **📌 Add to calendar** button "
        "that captures its end date as a recompete catalyst."
    )

    _ticker_options = ["(custom recipient search)"] + sorted(FEDERAL_CONTRACTOR_TICKERS.keys())
    _l1, _l2, _l3 = st.columns([1, 2, 2])
    _selected_ticker = _l1.selectbox(
        "Ticker", options=_ticker_options,
        format_func=lambda t: (
            t if t.startswith("(") else f"{t} — {FEDERAL_CONTRACTOR_TICKERS[t]}"
        ),
        key="fed_contract_ticker",
        index=1 if len(_ticker_options) > 1 else 0,
    )
    if _selected_ticker.startswith("("):
        _custom = _l2.text_input(
            "Or search by recipient name",
            placeholder="e.g. SPACEX, ANDURIL, ELASTIC, …",
            key="fed_contract_custom",
        )
        _search_text  = _custom.strip() if _custom else None
        _ticker_label = _custom.strip().upper() if _custom else None
    else:
        _search_text  = FEDERAL_CONTRACTOR_TICKERS[_selected_ticker]
        _ticker_label = _selected_ticker
        _l2.markdown(f"**Searching:** `{_search_text}`")
    _lookback = _l3.slider("Lookback (years)", min_value=1, max_value=5, value=2,
                           key="fed_contract_lookback")

    if _search_text:
        with st.spinner(f"Pulling USAspending data for {_search_text}…"):
            _summary = fetch_usaspending_summary(_search_text, lookback_years=_lookback)
            _awards  = fetch_usaspending_awards(_search_text, lookback_years=_lookback, limit=50)

        if _summary is None or _awards is None:
            st.error(
                "USAspending API request failed (network or rate-limit). "
                "Try again in a minute — results are cached for 24 h once fetched successfully."
            )
        elif not _awards:
            st.warning(
                f"No federal contracts found for `{_search_text}` in the "
                f"last {_lookback} year(s). Try a different recipient name "
                "or widen the lookback window."
            )
        else:
            _k1, _k2, _k3, _k4 = st.columns(4)
            _k1.metric(
                f"Total awards ({_lookback}yr)",
                f"${_summary['total']/1e9:.2f}B" if _summary['total'] >= 1e9
                else f"${_summary['total']/1e6:.0f}M",
            )
            _k2.metric("Top contracts shown", len(_awards))
            _top_agency = (_summary["top_agencies"][0][0] if _summary["top_agencies"] else "—")
            _k3.metric("Top awarding agency", _top_agency[:25])
            _now_d      = _dt.datetime.now().date()
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

            st.subheader(f"🏆 Top contracts — {_search_text}")
            st.caption(
                f"Showing top {len(_awards)} contracts by award amount over "
                f"the last {_lookback} year(s). Click **📌 Add to calendar** "
                "on any contract whose end date is within 2 years to log it "
                "as a recompete catalyst."
            )

            _existing_recompete = list_catalysts(source="usaspending_recompete")
            _existing_award_ids = {
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
                    _amt_s = (
                        f"${_amt/1e9:.2f}B" if _amt >= 1e9 else
                        f"${_amt/1e6:.1f}M" if _amt >= 1e6 else
                        f"${_amt/1e3:.0f}K"
                    )
                    _cc1.markdown(f"**`{_award_id}`** · {_amt_s} · _{_agency}_")
                    _cc1.caption(_desc + ("…" if len(_desc) >= 240 else ""))
                    if _eds:
                        try:
                            _end_dt   = _dt.datetime.strptime(_eds, "%Y-%m-%d")
                            _days_out = (_end_dt.date() - _dt.datetime.now().date()).days
                            if _days_out >= 0:
                                _cc1.caption(
                                    f"📅 Period of Performance ends **{_eds}** · in {_days_out} days"
                                )
                            else:
                                _cc1.caption(f"📅 Ended {_eds} · {-_days_out} days ago")
                        except Exception:
                            _cc1.caption(f"📅 End date: {_eds}")

                    _can_add = False
                    _end_ts  = None
                    if _eds:
                        try:
                            _end_dt   = _dt.datetime.strptime(_eds, "%Y-%m-%d")
                            _days_out = (_end_dt.date() - _dt.datetime.now().date()).days
                            if 0 < _days_out <= 730:
                                _can_add = True
                                _end_ts  = int(_end_dt.timestamp())
                        except Exception:
                            pass

                    _btn_key = f"addcat_fed_{_i}_{_award_id}"
                    if _award_id in _existing_award_ids:
                        _cc2.success("✅ on calendar")
                    elif _can_add:
                        if _cc2.button("📌 Add to calendar", key=_btn_key,
                                       use_container_width=True):
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
                                    sectors       = (
                                        ["Defense"]
                                        if _ticker_label in ("LMT","NOC","GD","RTX","BA","HII")
                                        else []
                                    ),
                                    source        = "usaspending_recompete",
                                )
                                st.toast(f"📌 Added recompete catalyst for {_award_id}", icon="🏗️")
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Failed to add catalyst: {_e}")
                    else:
                        _cc2.caption("_no end date_" if not _eds else "_>2 yr out_")

            if len(_awards) > 25:
                st.caption(
                    f"…and {len(_awards) - 25} more contracts not shown. "
                    "Increase the lookback window or look up a more specific recipient name."
                )

    st.divider()

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
        _wk  = [c for c in _all_con if c["event_date"] <= _now_ts + 7  * 86400]
        _mo  = [c for c in _all_con
                if _now_ts + 7  * 86400 < c["event_date"] <= _now_ts + 90 * 86400]
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
