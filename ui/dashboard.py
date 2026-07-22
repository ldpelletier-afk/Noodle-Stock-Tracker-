import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api import fetch_calendar_events, fetch_live_prices, fetch_portfolio_value_history, fetch_sparkline_history, live_price_feed_status
from ui.common import bg_prefetch


def render(app_data: dict) -> None:
    portfolios = app_data.get("portfolios", {})

    st.header("🏠 Dashboard")
    st.caption(
        "Portfolio performance overview, reconstructed live from your transaction "
        "history × current prices."
    )

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
                # Market Watch caches these per-session (see ui/market_watch.py)
                # to avoid re-fetching on every tab switch — bust that too.
                for _k in ("_mw_price_store", "_mw_spark_store", "_mw_cal_df", "_mw_cal_tickers"):
                    st.session_state.pop(_k, None)
                st.rerun()
        else:
            if st.button("⚡ Power-load all", key="dash_load_btn",
                         type="primary",
                         help="One click: warms live prices, sparklines, calendar events, "
                              "FRED macro series, CNN F&G, and the full Library doc list. "
                              "Takes 5–15 s on a cold start; afterwards every tab is instant."):
                for _t in _LIVE_TARGETS:
                    st.session_state._lazy_loaded.add(_t)
                with st.spinner("Warming caches across all tabs (live prices, macro, sentiment, library)…"):
                    bg_prefetch(app_data, deep=True)
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
            _active_pname = None
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
                if _ch_pct != 0:
                    _prev_close = _price / (1.0 + _ch_pct / 100.0)
                    _day_pl += _qty * (_price - _prev_close)

            _cash_qty = _active_holdings.get("CASH", {}).get("quantity", 0.0)
            _total_value += _cash_qty
            _total_cost += _cash_qty

            _total_return_dollar = _total_value - _total_cost
            _total_return_pct = (
                (_total_return_dollar / _total_cost * 100.0) if _total_cost > 0 else 0.0
            )
            _opening_value = _total_value - _day_pl
            _day_pl_pct = (_day_pl / _opening_value * 100.0) if _opening_value > 0 else 0.0

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
