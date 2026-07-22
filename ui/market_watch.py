import re as _re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from api import (
    fetch_calendar_events,
    fetch_live_prices,
    fetch_sparkline_history,
    fetch_stock_details,
)
from data_store import (
    add_to_watchlist,
    create_watchlist,
    delete_watchlist,
    remove_from_watchlist,
    rename_watchlist,
    set_target_in_watchlist,
    ui_state_get,
    ui_state_set,
)
from utils import format_large_number, highlight_buy_zone, sanitize_ticker


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


def render(app_data: dict) -> None:
    watchlists = app_data.get("watchlists", {})

    all_wl_tickers = sorted({t for items in watchlists.values() for t in items})

    _mw_ready = "market_watch" in st.session_state._lazy_loaded

    if not _mw_ready:
        live_prices = {}
        _sparklines = {}
        st.caption(
            "📡 Live prices not loaded yet — click **⚡ Load all live data** on the "
            "🏠 Dashboard tab to populate prices, sparklines, and the calendar."
        )
    else:
        # Session-scoped stores: once a ticker's price/sparkline is fetched this
        # session, every later visit to this tab reuses it instead of
        # re-fetching — even after the underlying TTL cache in api.py expires.
        # Only tickers newly added to a watchlist trigger a fetch. Use the
        # Dashboard's "🔄 Refresh live data" button to force a full refresh.
        _price_store = st.session_state.setdefault("_mw_price_store", {})
        _missing_prices = [t for t in all_wl_tickers if t not in _price_store]
        if _missing_prices:
            with st.spinner("Fetching live market data..."):
                _price_store.update(fetch_live_prices(_missing_prices))
        live_prices = _price_store

        _spark_store = st.session_state.setdefault("_mw_spark_store", {})
        _missing_sparks = [t for t in all_wl_tickers if t not in _spark_store]
        if _missing_sparks:
            with st.spinner("Loading 30-day trends..."):
                _spark_store.update(fetch_sparkline_history(tuple(sorted(_missing_sparks))))
        _sparklines = _spark_store

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

    if all_wl_tickers:
        with st.expander("📅 Upcoming Calendar (next 60 days)", expanded=False):
            if _mw_ready:
                _cal_ticker_key = tuple(sorted(all_wl_tickers))
                if st.session_state.get("_mw_cal_tickers") != _cal_ticker_key:
                    with st.spinner("Loading earnings + dividend dates..."):
                        st.session_state["_mw_cal_df"] = fetch_calendar_events(_cal_ticker_key)
                    st.session_state["_mw_cal_tickers"] = _cal_ticker_key
                _cal_df = st.session_state["_mw_cal_df"]
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

    _COL_CFG = {
        "Live Price (from API)": st.column_config.NumberColumn(format="$%.2f"),
        "Day Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Target Price (Self-set)": st.column_config.NumberColumn(format="$%.2f", step=0.01),
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

    if watchlists:
        for _list_name, _items in watchlists.items():
            _list_tickers = list(_items.keys())
            _count = len(_list_tickers)
            _is_open = _wl_is_open(_list_name)

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

            if _is_open:
                if _list_tickers:
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

                _ca, _cr = st.columns(2)
                with _ca:
                    st.markdown("**Add ticker**")
                    with st.form(key=f"wl_add_form_{_list_name}", clear_on_submit=True):
                        _add_input = st.text_input(
                            "Ticker",
                            key=f"wl_add_{_list_name}",
                            placeholder="e.g. MSFT",
                            label_visibility="collapsed",
                        )
                        _add_clicked = st.form_submit_button("➕ Add", use_container_width=True)
                    if _add_clicked:
                        _add_t = sanitize_ticker(_add_input)
                        if _add_t:
                            if add_to_watchlist(_list_name, _add_t):
                                st.rerun()
                            else:
                                st.warning(f"{_add_t} is already in '{_list_name}'.")

                    with st.expander("📥 Bulk add (paste multiple tickers)"):
                        with st.form(key=f"wl_bulk_form_{_list_name}", clear_on_submit=True):
                            _bulk_text = st.text_area(
                                "Tickers (comma, space, or newline separated)",
                                key=f"wl_bulk_{_list_name}",
                                placeholder="AAPL, MSFT, GOOG\nNVDA TSLA META\nAMZN",
                                height=100,
                            )
                            _bulk_clicked = st.form_submit_button("➕ Add all", use_container_width=True)
                        if _bulk_clicked and _bulk_text.strip():
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
                                    f"'{_list_name}': " + ", ".join(_msg_parts), icon="✅"
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

                st.markdown("---")
                _cn, _cd = st.columns([4, 1])
                with _cn:
                    with st.form(key=f"wl_rename_form_{_list_name}", clear_on_submit=False):
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
                                _was_open = _wl_is_open(_list_name)
                                _wl_set_open(_new_name, _was_open)
                                st.session_state.pop(_wl_state_key(_list_name), None)
                                st.rerun()
                            else:
                                st.warning(f"A list named '{_new_name}' already exists.")
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
            _create_clicked = st.form_submit_button("Create", use_container_width=True)
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
            fetch_stock_details.clear()
            fetch_live_prices.clear()
            for _k in ("_mw_price_store", "_mw_spark_store", "_mw_cal_df", "_mw_cal_tickers"):
                st.session_state.pop(_k, None)
            st.rerun()

    timeframes = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
    selected_period = st.radio("Timeframe", timeframes, index=2, horizontal=True)

    col_ta1, col_ta2, col_ta3 = st.columns(3)
    with col_ta1:
        show_sma = st.checkbox("Overlay 50-Period SMA")
    with col_ta2:
        show_rsi = st.checkbox("Show 14-Period RSI")
    with col_ta3:
        show_macd = st.checkbox("Show MACD (12, 26, 9)")

    if selected_ticker:
        with st.spinner(f"Loading {selected_ticker} data..."):
            hist_data, stock_info = fetch_stock_details(selected_ticker, selected_period)
            if not hist_data.empty:
                start_price = hist_data['Close'].iloc[0]
                end_price = hist_data['Close'].iloc[-1]
                chart_color = '#28a745' if end_price >= start_price else '#dc3545'

                active_subplots = sum([show_rsi, show_macd])
                total_rows = 1 + active_subplots

                row_heights = (
                    [1.0] if total_rows == 1
                    else ([0.7, 0.3] if total_rows == 2 else [0.6, 0.2, 0.2])
                )
                rsi_row = 2 if show_rsi else None
                macd_row = (3 if total_rows == 3 else 2) if show_macd else None

                if total_rows > 1:
                    fig = make_subplots(
                        rows=total_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=row_heights,
                    )
                else:
                    fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index, y=hist_data['Close'], mode='lines',
                        name=f"{selected_ticker} Price",
                        line=dict(color=chart_color, width=2),
                    ),
                    row=1 if total_rows > 1 else None,
                    col=1 if total_rows > 1 else None,
                )

                if show_sma:
                    fig.add_trace(
                        go.Scatter(
                            x=hist_data.index,
                            y=hist_data['Close'].rolling(window=50).mean(),
                            mode='lines', name="50-Period SMA",
                            line=dict(color='#3498db', width=1.5, dash='dot'),
                        ),
                        row=1 if total_rows > 1 else None,
                        col=1 if total_rows > 1 else None,
                    )
                if show_rsi:
                    delta = hist_data['Close'].diff()
                    gain = delta.clip(lower=0)
                    loss = -1 * delta.clip(upper=0)
                    rs = (
                        gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                        / loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    )
                    rsi = 100 - (100 / (1 + rs))
                    fig.add_trace(
                        go.Scatter(
                            x=hist_data.index, y=rsi, mode='lines', name="RSI (14)",
                            line=dict(color='#9b59b6', width=1.5),
                        ),
                        row=rsi_row, col=1,
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="rgba(220, 53, 69, 0.5)", row=rsi_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="rgba(40, 167, 69, 0.5)", row=rsi_row, col=1)
                    fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
                if show_macd:
                    macd_line = (
                        hist_data['Close'].ewm(span=12, adjust=False).mean()
                        - hist_data['Close'].ewm(span=26, adjust=False).mean()
                    )
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    macd_hist = macd_line - signal_line
                    hist_colors = ['#28a745' if val >= 0 else '#dc3545' for val in macd_hist]
                    fig.add_trace(
                        go.Bar(x=hist_data.index, y=macd_hist, marker_color=hist_colors, name="MACD Histogram"),
                        row=macd_row, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=hist_data.index, y=macd_line, mode='lines', name="MACD Line", line=dict(color='#2980b9', width=1.5)),
                        row=macd_row, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=hist_data.index, y=signal_line, mode='lines', name="Signal Line", line=dict(color='#e67e22', width=1.5)),
                        row=macd_row, col=1,
                    )

                dynamic_height = 400 + (200 * active_subplots)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified", showlegend=True, height=dynamic_height,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
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
            else:
                st.warning(f"Could not retrieve historical data for {selected_ticker}.")
