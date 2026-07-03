import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from api import fetch_live_prices
from risk import portfolio_risk_report as _portfolio_risk_report


def render(app_data: dict) -> None:
    portfolios = app_data.get("portfolios", {})

    st.header("🛡️ Portfolio Risk Dashboard")
    st.caption(
        "VaR, drawdown, correlation, beta, and factor exposure on your "
        "live holdings. Uses daily adjusted closes from yfinance — "
        "computed client-side, cached for an hour."
    )

    _risk_portfolios = list(portfolios.keys())
    if not _risk_portfolios:
        st.info("Create a portfolio in the Asset Tracker tab to unlock this module.")
        return

    r_col1, r_col2, r_col3, r_col4 = st.columns([2, 1, 1, 1])
    with r_col1:
        risk_portfolio = st.selectbox(
            "Portfolio", ["All Portfolios"] + _risk_portfolios, key="risk_portfolio"
        )
    with r_col2:
        risk_period = st.selectbox("Lookback", ["1y", "2y", "5y"], index=1, key="risk_period")
    with r_col3:
        risk_conf = st.selectbox("VaR Confidence", ["95%", "99%"], index=0, key="risk_conf")
    with r_col4:
        risk_bench = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "ACWI"], index=0, key="risk_bench")

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
        return

    if not st.button("Run Risk Analysis", type="primary", use_container_width=True):
        return

    with st.spinner("Fetching price history & crunching risk metrics..."):
        _lp_tickers = [t for t in risk_holdings if t.upper() != "CASH"]
        _lp = fetch_live_prices(_lp_tickers)
        _conf = 0.95 if risk_conf == "95%" else 0.99
        report = _portfolio_risk_report(
            risk_holdings, _lp,
            period=risk_period, var_confidence=_conf, benchmark=risk_bench,
        )

    if report.get("error"):
        st.error(report["error"])
        return

    st.subheader("Headline Metrics")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Ann. Return", f"{report['ann_return']*100:,.2f}%" if pd.notna(report['ann_return']) else "—")
    h2.metric("Ann. Volatility", f"{report['ann_volatility']*100:,.2f}%" if pd.notna(report['ann_volatility']) else "—")
    h3.metric("Sharpe (rf=4%)", f"{report['sharpe']:.2f}" if pd.notna(report['sharpe']) else "—")
    h4.metric("Sortino", f"{report['sortino']:.2f}" if pd.notna(report['sortino']) else "—")

    v1, v2, v3, v4 = st.columns(4)
    v1.metric(
        f"1-Day VaR ({risk_conf}, hist.)",
        f"{report['hist_var_1d']*100:,.2f}%" if pd.notna(report['hist_var_1d']) else "—",
        help="Largest expected daily loss at the stated confidence level, based on observed history.",
    )
    v2.metric(
        f"1-Day CVaR ({risk_conf})",
        f"{report['hist_cvar_1d']*100:,.2f}%" if pd.notna(report['hist_cvar_1d']) else "—",
        help="Expected loss conditional on exceeding the VaR threshold — the 'how bad is bad' number.",
    )
    v3.metric(
        "Max Drawdown",
        f"{report['max_drawdown']*100:,.2f}%" if pd.notna(report['max_drawdown']) else "—",
        help=f"Duration: {report.get('dd_days', 0)} days peak-to-trough.",
    )
    v4.metric(
        f"Beta vs {risk_bench}",
        f"{report['beta']:.2f}" if pd.notna(report['beta']) else "—",
        help=f"R²={report['r_squared']:.2f}" if pd.notna(report['r_squared']) else "",
    )

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

    st.subheader("Portfolio Equity Curve & Drawdown")
    import numpy as _np
    port_r = report["portfolio_returns"]
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
            x=equity.index, y=(equity - 1) * 100, mode="lines", name="Portfolio",
            line=dict(color="#2980b9", width=2),
        ),
        row=1, col=1,
    )
    fig_eq.add_trace(
        go.Scatter(
            x=dd_series.index, y=dd_series, mode="lines", name="Drawdown",
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

    st.subheader("Per-Position Risk Contribution")
    st.caption(
        "Marginal Contribution to Risk = weight × cov(asset, port) / var(port). "
        "A position with weight 10% but risk-contribution 25% is a concentrated volatility source."
    )
    _pp = report["per_position"].copy()
    st.dataframe(
        _pp.style.format({"Weight": "{:.2%}", "Vol (ann.)": "{:.2%}", "Beta": "{:.2f}", "Contribution to Risk": "{:.2%}"}, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    _c = report["concentration"]
    if pd.notna(_c.get("hhi")):
        c1, c2, c3 = st.columns(3)
        c1.metric("Positions", f"{_c['n_positions']}")
        c2.metric(
            "Effective N (1/HHI)", f"{_c['effective_n']:.1f}",
            help="How many equally-weighted names this portfolio diversifies LIKE. "
                 "Much smaller than position count = concentrated.",
        )
        c3.metric(
            "Largest Weight", f"{_c['top']*100:.1f}%",
            help="Single-name concentration — a red flag above ~20-25%.",
        )

    st.divider()

    corr = report.get("correlation")
    if corr is not None and not corr.empty and corr.shape[0] >= 2:
        st.subheader("Holdings Correlation Matrix")
        st.caption(
            "Daily-return correlations. Highly correlated clusters "
            "(>0.7) hint that two positions are really the same bet."
        )
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=10),
        ))
        fig_corr.update_layout(
            height=max(350, 30 * len(corr) + 150),
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    roll_b = report.get("rolling_beta")
    if roll_b is not None and not roll_b.empty:
        st.subheader(f"Rolling 63-Day Beta vs {risk_bench}")
        fig_rb = go.Figure()
        fig_rb.add_trace(go.Scatter(
            x=roll_b.index, y=roll_b.values,
            mode="lines", line=dict(color="#8e44ad", width=2), name="Rolling Beta",
        ))
        fig_rb.add_hline(y=1.0, line_dash="dash", line_color="rgba(128,128,128,0.5)")
        fig_rb.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_rb, use_container_width=True)

    fx = report.get("factor_exposure") or {}
    if fx.get("loadings"):
        st.subheader("Factor Exposure (ETF-proxy decomposition)")
        st.caption(
            "Regression of portfolio returns on: "
            "MKT (SPY), SMB (IWM−SPY), HML (VTV−VUG), MOM (MTUM−SPY), RATES (TLT). "
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
            help="Return not explained by factor exposures. Persistent + alpha = skill or missing factor.",
        )
        fxc2.metric("R² (factor fit)", f"{fx['r2']:.2f}")
        fxc3.metric("Obs", f"{fx['n_obs']}")

    st.caption(
        f"Computed on {report['n_obs']} daily observations over "
        f"{report['period']}. Risk-free rate assumed 4% for "
        f"Sharpe/Sortino. Benchmark: {risk_bench}."
    )
