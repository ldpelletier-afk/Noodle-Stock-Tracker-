import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api import (
    fetch_bea_gdp,
    fetch_bea_pce,
    fetch_bls_indicators,
    fetch_cftc_cot,
    fetch_cftc_snapshot,
    fetch_coingecko_global,
    fetch_coingecko_top_coins,
    fetch_commodity_prices,
    fetch_eia_snapshot,
    fetch_fear_greed,
    fetch_fmp_analyst_estimates,
    fetch_fmp_price_targets,
    fetch_fmp_profile,
    fetch_fmp_ratings,
    fetch_macro_data,
    fetch_treasury_debt,
    fetch_yield_curve,
    fred,
    has_bea,
    has_eia,
    has_fmp,
)


def render(app_data: dict) -> None:
    st.header("🌐 Global Markets")
    st.caption("Macro, energy, commodities, crypto, and analyst intelligence — all in one place.")

    _gm_systemic, _gm_resources, _gm_fmp = st.tabs([
        "🏛️ US Systemic Data",
        "⚡ Resources & Assets",
        "📈 FMP Analytics",
    ])

    with _gm_systemic:
        st.caption("Official U.S. macro data: Federal Reserve, BLS, BEA, and Treasury.")
        _mac_fred, _mac_bls, _mac_bea, _mac_tsy = st.tabs([
            "🏛️ FRED Explorer",
            "👷 Labor & Inflation (BLS)",
            "📊 GDP & Spending (BEA)",
            "🏦 Treasury & Debt",
        ])

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
                    latest_date = macro_df.index[-1].strftime("%B %Y")
                    latest_value = macro_df["Value"].iloc[-1]
                    st.metric(f"Latest Print ({latest_date})", f"{latest_value:,.2f}")
                    fig_macro = px.area(
                        macro_df, x=macro_df.index, y="Value",
                        title=f"{selected_indicator_name} (Last 20 Years)",
                    )
                    fig_macro.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        hovermode="x unified",
                    )
                    fig_macro.update_xaxes(showgrid=False, title_text="")
                    fig_macro.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    fig_macro.update_traces(line_color="#2980b9", fillcolor="rgba(41,128,185,0.2)")
                    st.plotly_chart(fig_macro, use_container_width=True)
                else:
                    st.warning(f"No data for {selected_series_id}. Verify the series ID.")

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
            _b1, _b2, _b3, _b4, _b5 = st.columns(5)
            _b1.metric("🧾 CPI (YoY)", f"{_bls_latest.get('cpi_yoy', 'N/A')}%",
                       help=f"All-items CPI as of {_bls_latest.get('cpi_date', '')}")
            _b2.metric("🔍 Core CPI (YoY)", f"{_bls_latest.get('core_cpi_yoy', 'N/A')}%",
                       help="Ex food & energy")
            _b3.metric("🏭 PPI (YoY)", f"{_bls_latest.get('ppi_yoy', 'N/A')}%",
                       help=f"Producer Price Index as of {_bls_latest.get('ppi_date', '')}")
            _b4.metric(
                "👷 Payrolls (MoM)",
                f"{_bls_latest.get('payrolls_mom', 'N/A'):,}K"
                if isinstance(_bls_latest.get("payrolls_mom"), (int, float)) else "N/A",
                help=f"Nonfarm payrolls change as of {_bls_latest.get('payrolls_date', '')}",
            )
            _b5.metric("📉 Unemployment", f"{_bls_latest.get('unemployment', 'N/A')}%",
                       help=f"As of {_bls_latest.get('unemployment_date', '')}")
            st.divider()

            _bls_chart_opts = {
                "CPI (All Items)": ("cpi", "CPI Index Level"),
                "Core CPI (ex F&E)": ("core_cpi", "Core CPI Index Level"),
                "PPI (Final Demand)": ("ppi", "PPI Index Level"),
                "Nonfarm Payrolls": ("payrolls", "Thousands of Jobs"),
                "Unemployment Rate (%)": ("unemployment", "Unemployment %"),
            }
            _bls_choice = st.selectbox("Chart series", list(_bls_chart_opts.keys()), key="bls_chart_sel")
            _bls_key, _bls_ylabel = _bls_chart_opts[_bls_choice]
            _bls_df = _bls_series.get(_bls_key)
            if _bls_df is not None and not _bls_df.empty:
                _fig_bls = px.area(
                    _bls_df, x=_bls_df.index, y="Value",
                    title=f"{_bls_choice} — BLS Official Data",
                    labels={"Value": _bls_ylabel},
                )
                _fig_bls.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified"
                )
                _fig_bls.update_xaxes(showgrid=False, title_text="")
                _fig_bls.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                _fig_bls.update_traces(line_color="#e67e22", fillcolor="rgba(230,126,34,0.15)")
                st.plotly_chart(_fig_bls, use_container_width=True)
            else:
                st.info("Series data not yet available.")

    with _mac_bea:
        st.subheader("Bureau of Economic Analysis")
        if not has_bea():
            st.warning("🔑 BEA API key not configured.")
            st.info(
                "Get a free key at **[apps.bea.gov/API/signup](https://apps.bea.gov/API/signup/)** "
                "then add `BEA_API_KEY=your_key` to `.env` and restart."
            )
        else:
            st.caption("GDP growth and PCE inflation — the two data series the Fed watches most closely.")
            _bea_col1, _bea_col2 = st.columns(2)
            with st.spinner("Fetching BEA data..."):
                _gdp = fetch_bea_gdp()
                _pce = fetch_bea_pce()

            with _bea_col1:
                st.markdown("#### 📈 Real GDP Growth (QoQ, annualized %)")
                if _gdp:
                    st.metric(f"Latest ({_gdp['latest_date']})", f"{_gdp['latest']:+.2f}%")
                    _fig_gdp = px.bar(
                        _gdp["series"].reset_index(), x="Date", y="Value",
                        title="Real GDP Growth Rate",
                        labels={"Value": "% Change (annualized)"},
                        color="Value",
                        color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
                    )
                    _fig_gdp.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                    )
                    _fig_gdp.update_xaxes(showgrid=False)
                    _fig_gdp.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    st.plotly_chart(_fig_gdp, use_container_width=True)
                else:
                    st.info("GDP data unavailable.")

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
                        _pce["series"].reset_index(), x="Date", y="Value",
                        title="PCE Price Index", labels={"Value": "Index Level"},
                    )
                    _fig_pce.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                    )
                    _fig_pce.update_xaxes(showgrid=False)
                    _fig_pce.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    _fig_pce.update_traces(line_color="#8e44ad", fillcolor="rgba(142,68,173,0.15)")
                    st.plotly_chart(_fig_pce, use_container_width=True)
                else:
                    st.info("PCE data unavailable.")

    with _mac_tsy:
        st.subheader("U.S. Treasury — Fiscal Data")
        st.caption("National debt and yield curve sourced directly from Treasury APIs. No API key required.")
        _tsy_col1, _tsy_col2 = st.columns([1, 2])
        with st.spinner("Fetching Treasury data..."):
            _debt = fetch_treasury_debt()
            _yc = fetch_yield_curve()

        with _tsy_col1:
            st.markdown("#### 🏛️ National Debt")
            if _debt:
                st.metric(
                    f"Total Outstanding ({_debt['latest_date']})", f"${_debt['latest_trillions']:,.2f}T"
                )
                _debt_series = _debt.get("series")
                if _debt_series is not None and not _debt_series.empty:
                    _fig_debt = px.area(
                        _debt_series.reset_index(), x="Date", y="Value",
                        title="National Debt (Trillions USD)", labels={"Value": "Trillions ($)"},
                    )
                    _fig_debt.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    _fig_debt.update_xaxes(showgrid=False)
                    _fig_debt.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
                    _fig_debt.update_traces(line_color="#c0392b", fillcolor="rgba(192,57,43,0.15)")
                    st.plotly_chart(_fig_debt, use_container_width=True)
            else:
                st.info("Debt data unavailable.")

        with _tsy_col2:
            st.markdown("#### 📐 Treasury Yield Curve")
            if _yc is not None and not _yc.empty:
                _spread_10_2 = None
                _y10 = _yc.loc[_yc["Maturity"] == "10Y", "Yield"]
                _y2 = _yc.loc[_yc["Maturity"] == "2Y", "Yield"]
                if not _y10.empty and not _y2.empty:
                    _spread_10_2 = round(float(_y10.iloc[0]) - float(_y2.iloc[0]), 3)
                _yc_cols = st.columns(3)
                for _i, row in _yc.iterrows():
                    _yc_cols[_i % 3].metric(row["Maturity"], f"{row['Yield']:.3f}%")
                st.divider()
                if _spread_10_2 is not None:
                    _inv = "🔴 Inverted" if _spread_10_2 < 0 else "🟢 Normal"
                    st.metric(
                        "10Y–2Y Spread (Inversion Indicator)", f"{_spread_10_2:+.3f}%",
                        help=f"{_inv} curve  · Negative = recession signal",
                    )
                _fig_yc = px.line(
                    _yc, x="Maturity", y="Yield", title="Treasury Yield Curve (today)",
                    labels={"Yield": "Yield (%)"}, markers=True,
                )
                _fig_yc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
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

    with _gm_resources:
        st.caption("Energy inventories (EIA), commodities, crypto market, and sentiment.")

        with st.spinner("Loading sentiment data..."):
            _fg = fetch_fear_greed()
        if _fg and _fg.get("score") is not None:
            _fg_score = _fg["score"]
            _fg_rating = _fg.get("rating", "")
            _fg_color = (
                "#e74c3c" if _fg_score < 25 else
                "#e67e22" if _fg_score < 45 else
                "#f1c40f" if _fg_score < 55 else
                "#2ecc71" if _fg_score < 75 else "#27ae60"
            )
            _fgc1, _fgc2 = st.columns([1, 3])
            with _fgc1:
                st.markdown(
                    f"<div style='text-align:center; padding:16px; border-radius:12px;"
                    f"background:{_fg_color}22; border:2px solid {_fg_color}'>"
                    f"<div style='font-size:2.8rem; font-weight:700; color:{_fg_color}'>{int(_fg_score)}</div>"
                    f"<div style='font-size:1rem; font-weight:600; color:{_fg_color}'>{_fg_rating}</div>"
                    f"<div style='font-size:0.75rem; color:#888'>CNN Fear & Greed</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with _fgc2:
                if _fg.get("history") is not None:
                    _fig_fg = px.area(
                        _fg["history"], y="Score",
                        title="Fear & Greed — 1 Year History",
                        color_discrete_sequence=[_fg_color],
                    )
                    _fig_fg.add_hline(y=25, line_dash="dot", line_color="#e74c3c", annotation_text="Extreme Fear")
                    _fig_fg.add_hline(y=75, line_dash="dot", line_color="#27ae60", annotation_text="Extreme Greed")
                    _fig_fg.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 100]
                    )
                    _fig_fg.update_xaxes(showgrid=False)
                    st.plotly_chart(_fig_fg, use_container_width=True)
        else:
            st.info("Fear & Greed data unavailable.")
        st.divider()

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
                _cg_top = fetch_coingecko_top_coins(10)
            if _cg_global:
                _cgg1, _cgg2, _cgg3 = st.columns(3)
                _tmc = _cg_global.get("total_market_cap_usd")
                _btcd = _cg_global.get("btc_dominance")
                _mcc = _cg_global.get("market_cap_change_24h")
                _cgg1.metric(
                    "Total Mkt Cap",
                    f"${_tmc/1e12:.2f}T" if _tmc else "—",
                    delta=f"{_mcc:+.1f}%" if _mcc else None,
                )
                _cgg2.metric("BTC Dominance", f"{_btcd:.1f}%" if _btcd else "—")
                _cgg3.metric("ETH Dominance", f"{_cg_global.get('eth_dominance', 0):.1f}%")
            if _cg_top is not None:
                st.dataframe(
                    _cg_top, hide_index=True, use_container_width=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="$%.4f"),
                        "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                        "7d %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Mkt Cap": st.column_config.NumberColumn(format="$%.0f"),
                        "Vol 24h": st.column_config.NumberColumn(format="$%.0f"),
                    },
                )
            else:
                st.info("CoinGecko data unavailable (rate limit — try again shortly).")

        st.divider()

        st.markdown("#### ⚡ EIA Energy Inventories")
        if not has_eia():
            st.info(
                "Add `EIA_API_KEY=...` to `.env` for weekly energy inventory data.\n\n"
                "Free key at [eia.gov/opendata](https://www.eia.gov/opendata/register.php)."
            )
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
                        f"{_info['label']} ({_info['unit']})", f"{_info['value']:,.1f}",
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
                    _fig_eia = px.area(
                        _eia_df, y="Value",
                        title=f"{_eia_latest[_eia_choice]['label']} ({_unit})",
                    )
                    _fig_eia.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    _fig_eia.update_xaxes(showgrid=False)
                    _fig_eia.update_traces(line_color="#f39c12", fillcolor="rgba(243,156,18,0.15)")
                    st.plotly_chart(_fig_eia, use_container_width=True)

        st.divider()

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
            _cftc_rows = []
            for _ck, _cv in _cftc_snap.items():
                _nc_net = _cv["nc_net"]
                _nc_chg = _cv["nc_chg"]
                _bias = (
                    "🐂 Bullish" if _nc_net > 50_000 else
                    "🐻 Bearish" if _nc_net < -50_000 else
                    "🐂 Mild Bull" if _nc_net > 0 else "🐻 Mild Bear"
                )
                _cftc_rows.append({
                    "Market": _cv["label"],
                    "Net Speculative": _nc_net,
                    "WoW Change": _nc_chg,
                    "Bias": _bias,
                    "Open Interest": _cv["oi"],
                    "As of": _cv["date"],
                })
            _cftc_df_table = pd.DataFrame(_cftc_rows)
            st.dataframe(
                _cftc_df_table, hide_index=True, use_container_width=True,
                column_config={
                    "Net Speculative": st.column_config.NumberColumn(format="%d"),
                    "WoW Change": st.column_config.NumberColumn(format="%+d"),
                    "Open Interest": st.column_config.NumberColumn(format="%d"),
                },
            )
            _cftc_mkt_opts = list(_cftc_snap.keys())
            _cftc_sel = st.selectbox(
                "Chart speculator net positioning",
                _cftc_mkt_opts,
                format_func=lambda k: _cftc_snap.get(k, {}).get("label", k),
                key="cftc_chart_sel",
            )
            with st.spinner(f"Loading {_cftc_snap[_cftc_sel]['label']} history..."):
                _cftc_hist = fetch_cftc_cot(_cftc_sel, weeks=104)

            if _cftc_hist is not None and not _cftc_hist.empty:
                _cftc_mkt_label = _cftc_snap[_cftc_sel]["label"]
                _net_vals = _cftc_hist["NonComm_Net"]
                _bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in _net_vals]
                _fig_cftc = go.Figure()
                _fig_cftc.add_trace(go.Bar(
                    x=_cftc_hist.index, y=_net_vals, name="Net Speculative",
                    marker_color=_bar_colors,
                    hovertemplate="%{x|%b %d %Y}<br>Net: %{y:,d}<extra></extra>",
                ))
                _fig_cftc.add_hline(y=0, line_color="#888", line_width=1)
                _fig_cftc.update_layout(
                    title=f"{_cftc_mkt_label} — Speculator Net Positions (2-Year)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Contracts (Long − Short)", showlegend=False, bargap=0.1,
                )
                _fig_cftc.update_xaxes(showgrid=False)
                _fig_cftc.update_yaxes(zeroline=True, zerolinecolor="#555", zerolinewidth=1)
                st.plotly_chart(_fig_cftc, use_container_width=True)
                st.caption(
                    "**Reading the chart:** Green bars = speculators are net long (bullish). "
                    "Red bars = net short (bearish). Extreme readings near historical highs/lows "
                    "often act as contrarian signals."
                )

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
            _fmp_ticker = st.text_input("Ticker", value="AAPL", key="fmp_ticker").upper().strip()
            if _fmp_ticker:
                with st.spinner(f"Fetching FMP data for {_fmp_ticker}..."):
                    _fmp_profile = fetch_fmp_profile(_fmp_ticker)
                    _fmp_targets = fetch_fmp_price_targets(_fmp_ticker)
                    _fmp_estimates = fetch_fmp_analyst_estimates(_fmp_ticker)
                    _fmp_rating = fetch_fmp_ratings(_fmp_ticker)

                if _fmp_profile:
                    _fp1, _fp2 = st.columns([3, 1])
                    with _fp1:
                        st.markdown(f"#### {_fmp_profile.get('companyName', _fmp_ticker)}")
                        _emp = _fmp_profile.get('fullTimeEmployees')
                        if isinstance(_emp, int):
                            st.caption(
                                f"**Sector:** {_fmp_profile.get('sector','—')}  ·  "
                                f"**Industry:** {_fmp_profile.get('industry','—')}  ·  "
                                f"**CEO:** {_fmp_profile.get('ceo','—')}  ·  "
                                f"**Employees:** {_emp:,}"
                            )
                        else:
                            st.caption(
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

                if _fmp_targets:
                    st.markdown("#### 🎯 Analyst Price Targets")
                    _pt_cols = st.columns(4)
                    _pt_cols[0].metric("Consensus", f"${_fmp_targets.get('targetConsensus','—')}")
                    _pt_cols[1].metric("High Target", f"${_fmp_targets.get('targetHigh','—')}")
                    _pt_cols[2].metric("Low Target", f"${_fmp_targets.get('targetLow','—')}")
                    _pt_cols[3].metric("Median Target", f"${_fmp_targets.get('targetMedian','—')}")
                    st.divider()

                if _fmp_rating:
                    st.markdown("#### ⭐ FMP Quantitative Rating")
                    _rat_cols = st.columns(4)
                    _rat_cols[0].metric("Overall Rating", _fmp_rating.get("rating", "—"))
                    _rat_cols[1].metric("DCF Score", _fmp_rating.get("ratingDetailsDCFScore", "—"))
                    _rat_cols[2].metric("ROE Score", _fmp_rating.get("ratingDetailsROEScore", "—"))
                    _rat_cols[3].metric("Recommendation", _fmp_rating.get("ratingRecommendation", "—"))
                    st.divider()

                if _fmp_estimates is not None and not _fmp_estimates.empty:
                    st.markdown("#### 📅 Forward Analyst Estimates")
                    st.dataframe(
                        _fmp_estimates, hide_index=True, use_container_width=True,
                        column_config={
                            "Est. Revenue": st.column_config.NumberColumn(format="$%.0f"),
                            "Est. EPS": st.column_config.NumberColumn(format="$%.2f"),
                            "EPS Low": st.column_config.NumberColumn(format="$%.2f"),
                            "EPS High": st.column_config.NumberColumn(format="$%.2f"),
                        },
                    )
                elif _fmp_profile:
                    st.info("No analyst estimates available for this ticker.")
