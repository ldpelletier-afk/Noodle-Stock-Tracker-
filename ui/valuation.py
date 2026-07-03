import pandas as pd
import plotly.express as px
import streamlit as st

from api import (
    build_simfin_balance_table,
    build_simfin_cashflow_table,
    build_simfin_income_table,
    fetch_av_indicators,
    fetch_dcf_data,
    fetch_simfin_statements,
    fetch_simfin_ttm_fcf,
    has_alpha_vantage,
    has_simfin,
)
from utils import format_large_number, sanitize_ticker


def render(app_data: dict) -> None:
    st.header("The Valuation Machine")

    val_ticker = sanitize_ticker(
        st.text_input("Ticker", value="AAPL", key="val_input").upper()
    )

    _val_dcf, _val_sf, _val_tech = st.tabs([
        "📐 DCF Calculator",
        "📋 Fundamentals (SimFin)",
        "📊 Technical Signals (Alpha Vantage)",
    ])

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
                margin_of_safety = (
                    (intrinsic_value_per_share - current_price) / intrinsic_value_per_share * 100
                )

                st.divider()
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Live Market Price", f"${current_price:.2f}")
                if intrinsic_value_per_share > current_price:
                    res_col2.success(f"### Intrinsic Value\n## ${intrinsic_value_per_share:.2f}")
                    res_col3.metric("Margin of Safety", f"{margin_of_safety:.1f}%", delta="Undervalued")
                else:
                    res_col2.error(f"### Intrinsic Value\n## ${intrinsic_value_per_share:.2f}")
                    res_col3.metric(
                        "Premium to Value", f"{abs(margin_of_safety):.1f}%",
                        delta="-Overvalued", delta_color="inverse",
                    )

                st.write("**Audit Breakdown:**")
                st.table(pd.DataFrame({
                    "Metric": ["Trailing FCF (input)", "Shares Outstanding", "Sum of PV (5 Years)", "PV of Terminal Value", "Total Enterprise Value"],
                    "Value": [
                        format_large_number(fcf), format_large_number(shares),
                        format_large_number(sum_pv_cf), format_large_number(pv_terminal_value),
                        format_large_number(sum_pv_cf + pv_terminal_value),
                    ],
                }))

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
                _sf_bs = _sf_stmts.get("balance")
                _sf_cf = _sf_stmts.get("cashflow")
                _sf_der = _sf_stmts.get("derived")

                if _sf_der is not None and not _sf_der.empty:
                    st.markdown(f"#### Key Ratios — {val_ticker} (latest annual)")
                    _sc = st.columns(5)
                    _der_row = _sf_der.iloc[0]
                    _kpi_map = [
                        ("Gross Profit Margin", "Gross Margin", "{}%"),
                        ("Operating Profit Margin", "Op. Margin", "{}%"),
                        ("Net Profit Margin", "Net Margin", "{}%"),
                        ("Return on Equity", "ROE", "{}%"),
                        ("Return on Assets", "ROA", "{}%"),
                    ]
                    for _ci, (_col, _label, _fmt) in enumerate(_kpi_map):
                        _v = _der_row.get(_col)
                        try:
                            _sc[_ci].metric(_label, f"{float(_v):.1f}%")
                        except Exception:
                            _sc[_ci].metric(_label, "—")
                    st.divider()

                _sf_view = st.radio(
                    "Statement",
                    ["📈 Income Statement", "🏦 Balance Sheet", "💵 Cash Flow"],
                    horizontal=True, key="sf_stmt_radio",
                )

                if _sf_view == "📈 Income Statement":
                    tbl = build_simfin_income_table(_sf_inc)
                    if tbl is not None:
                        st.dataframe(tbl, use_container_width=True)
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

                else:
                    tbl = build_simfin_cashflow_table(_sf_cf)
                    if tbl is not None:
                        st.dataframe(tbl, use_container_width=True)
                        if _sf_cf is not None and "Net Cash from Operating Activities" in _sf_cf.columns:
                            _cf_df = _sf_cf[["Fiscal Year", "Net Cash from Operating Activities", "Capital Expenditures"]].dropna().copy()
                            _cf_df = _cf_df.sort_values("Fiscal Year")
                            _cf_df["Free Cash Flow"] = (
                                pd.to_numeric(_cf_df["Net Cash from Operating Activities"], errors="coerce")
                                + pd.to_numeric(_cf_df["Capital Expenditures"], errors="coerce")
                            )
                            for _c in ["Net Cash from Operating Activities", "Capital Expenditures", "Free Cash Flow"]:
                                _cf_df[_c] = pd.to_numeric(_cf_df[_c], errors="coerce") / 1e9
                            _fig_cf = px.bar(
                                _cf_df, x="Fiscal Year",
                                y=["Net Cash from Operating Activities", "Free Cash Flow"],
                                barmode="group", title=f"{val_ticker} — Cash Flow (Billions $)",
                            )
                            _fig_cf.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                            _fig_cf.update_xaxes(showgrid=False)
                            _fig_cf.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title="$B")
                            st.plotly_chart(_fig_cf, use_container_width=True)
                    else:
                        st.info("Cash flow data not available.")

                st.caption("Data sourced from [SimFin](https://simfin.com) · Standardised GAAP financials")

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
                _av1.metric("RSI (14)", f"{_rsi_val}" if _rsi_val else "—", help="<30 oversold · >70 overbought")
                _av2.metric("MACD", f"{_av_latest.get('macd', '—')}")
                _av3.metric("MACD Signal", f"{_av_latest.get('macd_signal', '—')}")
                _av4.metric("BB Upper", f"${_av_latest.get('bb_upper', '—')}")
                _av5.metric("BB Lower", f"${_av_latest.get('bb_lower', '—')}")
                st.divider()

                _tech_view = st.radio(
                    "Chart", ["RSI", "MACD", "Bollinger Bands"], horizontal=True, key="av_chart_radio"
                )
                if _tech_view == "RSI":
                    _rsi_df = _av.get("rsi")
                    if _rsi_df is not None and not _rsi_df.empty:
                        _fig_rsi = px.line(_rsi_df, y="RSI", title=f"{val_ticker} — RSI (14)")
                        _fig_rsi.add_hline(y=70, line_dash="dot", line_color="#e74c3c", annotation_text="Overbought (70)")
                        _fig_rsi.add_hline(y=30, line_dash="dot", line_color="#27ae60", annotation_text="Oversold (30)")
                        _fig_rsi.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 100])
                        _fig_rsi.update_xaxes(showgrid=False)
                        st.plotly_chart(_fig_rsi, use_container_width=True)

                elif _tech_view == "MACD":
                    _macd_df = _av.get("macd")
                    if _macd_df is not None and not _macd_df.empty:
                        _fig_macd = px.line(_macd_df, y=["MACD", "Signal"], title=f"{val_ticker} — MACD (12, 26, 9)")
                        _fig_macd.add_bar(x=_macd_df.index, y=_macd_df["Histogram"], name="Histogram", opacity=0.4)
                        _fig_macd.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        _fig_macd.update_xaxes(showgrid=False)
                        st.plotly_chart(_fig_macd, use_container_width=True)

                else:
                    _bb_df = _av.get("bbands")
                    if _bb_df is not None and not _bb_df.empty:
                        _fig_bb = px.line(
                            _bb_df, y=["Upper", "Middle", "Lower"],
                            title=f"{val_ticker} — Bollinger Bands (20, 2σ)",
                            color_discrete_map={"Upper": "#e74c3c", "Middle": "#3498db", "Lower": "#27ae60"},
                        )
                        _fig_bb.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        _fig_bb.update_xaxes(showgrid=False)
                        st.plotly_chart(_fig_bb, use_container_width=True)
