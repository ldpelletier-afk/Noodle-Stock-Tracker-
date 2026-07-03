import datetime as _dt

import pandas as pd
import streamlit as st

from api import fetch_macro_data, fred
from api import get_fomc_schedule, get_treasury_refunding_schedule
from data_store import add_catalyst, list_catalysts
from ui.common import _render_catalyst_card


def render(app_data: dict) -> None:
    st.header("🏛️ Monetary Policy")
    st.caption(
        "Federal-reserve and Treasury catalysts that move company earnings "
        "power. Live FRED data above the fold, calendar auto-import below it, "
        "and a roll-up of every monetary catalyst on the calendar at the bottom."
    )

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
                try:
                    _yoy_idx = _df.index[_df.index <= (_df.index[-1] - pd.DateOffset(years=1))]
                    _yoy_val = _df.loc[_yoy_idx[-1], "Value"] if len(_yoy_idx) else None
                except Exception:
                    _yoy_val = None
                _delta = f"{(_latest - _yoy_val):+.2f} YoY" if _yoy_val is not None else None
                st.metric(_label, _fmt.format(_latest), _delta)
                try:
                    _spark = _df[_df.index >= (_df.index[-1] - pd.DateOffset(years=1))]
                    if not _spark.empty:
                        st.line_chart(_spark, height=80, use_container_width=True)
                except Exception:
                    pass

    st.divider()

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
        _fomc_existing       = list_catalysts(source="fomc_schedule")
        _fomc_existing_dates = {c["event_date"] for c in _fomc_existing}
        _fomc_candidates     = get_fomc_schedule()
        _fomc_to_add         = [c for c in _fomc_candidates
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
        _tr_existing       = list_catalysts(source="treasury_refunding")
        _tr_existing_dates = {c["event_date"] for c in _tr_existing}
        _tr_candidates     = get_treasury_refunding_schedule()
        _tr_to_add         = [c for c in _tr_candidates
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

    st.subheader("📋 Monetary catalysts on the calendar")
    _all_mon = list_catalysts(catalyst_types=["monetary"], status="upcoming")
    if not _all_mon:
        st.caption(
            "_No upcoming monetary catalysts yet. Use the auto-import buttons "
            "above or add custom entries on the **🎯 Catalyst Calendar** tab._"
        )
    else:
        _now_ts = int(_dt.datetime.now().timestamp())
        _wk  = [c for c in _all_mon if c["event_date"] <= _now_ts + 7 * 86400]
        _mo  = [c for c in _all_mon
                if _now_ts + 7 * 86400 < c["event_date"] <= _now_ts + 30 * 86400]
        _far = [c for c in _all_mon if c["event_date"] > _now_ts + 30 * 86400]

        _r1, _r2, _r3 = st.columns(3)
        _r1.metric("This week", len(_wk))
        _r2.metric("Next 30 days", len(_mo))
        _r3.metric("Beyond 30 days", len(_far))

        st.caption(
            f"Showing {len(_all_mon)} upcoming monetary catalyst(s). "
            "Click any card to expand the stakes or jump to the calendar."
        )
        for c in _all_mon[:25]:
            _render_catalyst_card(c)
        if len(_all_mon) > 25:
            st.info(
                f"…and {len(_all_mon) - 25} more. Switch to the "
                "**🎯 Catalyst Calendar** tab to see them all."
            )
