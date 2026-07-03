import pandas as pd
import plotly.express as px
import streamlit as st

from api import fetch_live_prices
from data_store import (
    create_portfolio,
    log_transaction,
    remove_holding,
    replace_portfolio_holdings,
    upsert_holding,
)
from ui.common import _invalidate_app_data
from utils import sanitize_ticker


def render(app_data: dict) -> None:
    portfolios = app_data.get("portfolios", {})

    st.header("Portfolio Asset Tracker")
    with st.expander("Creating and Managing Portfolios", expanded=not bool(portfolios)):
        col_p1, col_p2 = st.columns([3, 1])
        new_portfolio_name = col_p1.text_input("New Portfolio Name")
        if col_p2.button("Create Portfolio"):
            if new_portfolio_name and new_portfolio_name not in portfolios:
                if create_portfolio(new_portfolio_name):
                    portfolios[new_portfolio_name] = {}
                    app_data["portfolios"] = portfolios
                    _invalidate_app_data()
                    st.toast(f"Portfolio '{new_portfolio_name}' created!", icon="🎉")
            elif new_portfolio_name in portfolios:
                st.warning("Portfolio already exists.")
    st.divider()

    portfolio_names = list(portfolios.keys())
    if not portfolio_names:
        st.info("Please create a portfolio to get started.")
        st.stop()

    selected_portfolio = st.selectbox(
        "Select Portfolio to View", ["All Portfolios"] + portfolio_names
    )

    current_holdings = {}
    if selected_portfolio == "All Portfolios":
        for p_name in portfolios:
            for ticker, data in portfolios[p_name].items():
                if ticker in current_holdings:
                    total_qty = current_holdings[ticker]['quantity'] + data['quantity']
                    total_cost = (
                        (current_holdings[ticker]['quantity'] * current_holdings[ticker]['average_cost'])
                        + (data['quantity'] * data['average_cost'])
                    )
                    current_holdings[ticker]['average_cost'] = total_cost / total_qty
                    current_holdings[ticker]['quantity'] = total_qty
                else:
                    current_holdings[ticker] = data.copy()
    else:
        current_holdings = portfolios[selected_portfolio]

    if selected_portfolio != "All Portfolios":
        col_add_stock, col_sell_stock, col_manage_cash, col_delete = st.columns(4)
        with col_add_stock:
            with st.expander("Add Stock", expanded=False):
                with st.form("add_asset_form", clear_on_submit=True):
                    asset_ticker = sanitize_ticker(st.text_input("Ticker").upper())
                    asset_qty = st.number_input("Quantity", min_value=0.01, step=0.01)
                    asset_cost = st.number_input("Avg. Cost ($)", min_value=0.0, step=0.01)
                    if st.form_submit_button("Buy"):
                        if asset_ticker and asset_ticker != "CASH" and asset_qty > 0:
                            if asset_ticker in portfolios[selected_portfolio]:
                                old_qty = portfolios[selected_portfolio][asset_ticker]['quantity']
                                old_cost = portfolios[selected_portfolio][asset_ticker]['average_cost']
                                new_qty = old_qty + asset_qty
                                new_avg = ((old_qty * old_cost) + (asset_qty * asset_cost)) / new_qty
                                portfolios[selected_portfolio][asset_ticker] = {
                                    "quantity": new_qty, "average_cost": new_avg,
                                }
                            else:
                                new_qty, new_avg = asset_qty, asset_cost
                                portfolios[selected_portfolio][asset_ticker] = {
                                    "quantity": new_qty, "average_cost": new_avg,
                                }
                            app_data["portfolios"] = portfolios
                            upsert_holding(selected_portfolio, asset_ticker, new_qty, new_avg)
                            _invalidate_app_data()
                            log_transaction(
                                selected_portfolio, asset_ticker, "BUY",
                                asset_qty, asset_cost, cost_basis=asset_cost,
                            )
                            st.toast(f"Added {asset_ticker}", icon="💰")

        with col_sell_stock:
            with st.expander("Sell Stock", expanded=False):
                sellable_assets = [
                    t for t in portfolios[selected_portfolio].keys() if t != "CASH"
                ]
                if sellable_assets:
                    with st.form("sell_asset_form", clear_on_submit=True):
                        sell_ticker = st.selectbox("Asset", sellable_assets)
                        current_qty = portfolios[selected_portfolio].get(
                            sell_ticker, {}
                        ).get("quantity", 0.0)
                        sell_qty = st.number_input(
                            "Qty to Sell", min_value=0.01, max_value=float(current_qty), step=0.01
                        )
                        sell_price = st.number_input("Sale Price ($)", min_value=0.0, step=0.01)
                        if st.form_submit_button("Execute Sale"):
                            if sell_qty > 0 and sell_price >= 0:
                                avg_cost_at_sale = portfolios[selected_portfolio][sell_ticker]["average_cost"]
                                proceeds = sell_qty * sell_price
                                portfolios[selected_portfolio][sell_ticker]["quantity"] -= sell_qty
                                if portfolios[selected_portfolio][sell_ticker]["quantity"] <= 0.0001:
                                    del portfolios[selected_portfolio][sell_ticker]
                                    remove_holding(selected_portfolio, sell_ticker)
                                else:
                                    remaining = portfolios[selected_portfolio][sell_ticker]
                                    upsert_holding(
                                        selected_portfolio, sell_ticker,
                                        remaining["quantity"], remaining["average_cost"],
                                    )
                                current_cash = portfolios[selected_portfolio].get(
                                    "CASH", {"quantity": 0.0, "average_cost": 1.0}
                                )
                                new_cash_qty = current_cash["quantity"] + proceeds
                                portfolios[selected_portfolio]["CASH"] = {
                                    "quantity": new_cash_qty, "average_cost": 1.0,
                                }
                                app_data["portfolios"] = portfolios
                                upsert_holding(selected_portfolio, "CASH", new_cash_qty, 1.0)
                                _invalidate_app_data()
                                log_transaction(
                                    selected_portfolio, sell_ticker, "SELL",
                                    sell_qty, sell_price, cost_basis=avg_cost_at_sale,
                                )
                                st.toast(f"Sold {sell_ticker}", icon="🤝")
                else:
                    st.info("No stocks to sell.")

        with col_manage_cash:
            with st.expander("Manage Cash", expanded=False):
                with st.form("manage_cash_form", clear_on_submit=True):
                    cash_action = st.radio("Action", ["Deposit", "Withdraw"], horizontal=True)
                    cash_amount = st.number_input("Amount ($)", min_value=0.01, step=100.0)
                    if st.form_submit_button("Update Cash"):
                        current_cash_data = portfolios[selected_portfolio].get(
                            "CASH", {"quantity": 0.0, "average_cost": 1.0}
                        )
                        new_qty = (
                            current_cash_data["quantity"] + cash_amount
                            if cash_action == "Deposit"
                            else max(0.0, current_cash_data["quantity"] - cash_amount)
                        )
                        portfolios[selected_portfolio]["CASH"] = {
                            "quantity": new_qty, "average_cost": 1.0
                        }
                        app_data["portfolios"] = portfolios
                        upsert_holding(selected_portfolio, "CASH", new_qty, 1.0)
                        _invalidate_app_data()

        with col_delete:
            with st.expander("Delete Asset", expanded=False):
                assets_to_delete = list(portfolios[selected_portfolio].keys())
                if assets_to_delete:
                    del_asset = st.selectbox("Select Asset to Delete", assets_to_delete)
                    if st.button("Delete Permanently", type="primary"):
                        del portfolios[selected_portfolio][del_asset]
                        app_data["portfolios"] = portfolios
                        remove_holding(selected_portfolio, del_asset)
                        _invalidate_app_data()
                        st.toast(f"Deleted {del_asset}", icon="🗑️")
                else:
                    st.info("No assets.")

    if current_holdings:
        holding_tickers = list(current_holdings.keys())
        _at_ready = "asset_tracker" in st.session_state._lazy_loaded
        if not _at_ready:
            st.caption(
                "📡 Live prices not loaded yet — click **⚡ Load all live data** on "
                "the 🏠 Dashboard tab to populate current values."
            )
            holding_prices = {}
        else:
            with st.spinner("Fetching live prices for holdings..."):
                holding_prices = fetch_live_prices(holding_tickers)

        holdings_data = []
        total_portfolio_value = 0
        total_portfolio_cost = 0

        for ticker, data in current_holdings.items():
            qty, avg_cost = data['quantity'], data['average_cost']
            if ticker == "CASH":
                current_price = 1.00
                market_value = qty
                total_cost = qty
                profit_loss = 0.0
                pl_percent = 0.0
            else:
                current_price = holding_prices.get(ticker, {}).get('price', 0) or 0
                market_value = qty * current_price
                total_cost = qty * avg_cost
                profit_loss = market_value - total_cost
                pl_percent = (profit_loss / total_cost * 100) if total_cost > 0 else 0

            total_portfolio_value += market_value
            total_portfolio_cost += total_cost
            holdings_data.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Avg. Cost": avg_cost,
                "Current Price": current_price,
                "Market Value": market_value,
                "Profit/Loss ($)": profit_loss,
                "Profit/Loss (%)": pl_percent / 100,
            })

        holdings_df = pd.DataFrame(holdings_data)
        total_pl = total_portfolio_value - total_portfolio_cost
        total_pl_pct = (total_pl / total_portfolio_cost * 100) if total_portfolio_cost > 0 else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
        m2.metric("Total Gain/Loss ($)", f"${total_pl:,.2f}", delta=f"${total_pl:,.2f}")
        m3.metric("Total Gain/Loss (%)", f"{total_pl_pct:.2f}%", delta=f"{total_pl_pct:.2f}%")
        st.divider()

        _col_cfg = {
            "Avg. Cost": st.column_config.NumberColumn(format="$%.2f"),
            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
            "Market Value": st.column_config.NumberColumn(format="$%.2f"),
            "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%"),
        }

        if selected_portfolio == "All Portfolios":
            st.subheader("Aggregated Holdings (Read-Only)")
            st.dataframe(holdings_df, use_container_width=True, hide_index=True, column_config=_col_cfg)
        else:
            st.subheader(f"Manage Holdings: {selected_portfolio}")
            edited_holdings_df = st.data_editor(
                holdings_df,
                disabled=["Ticker", "Current Price", "Market Value", "Profit/Loss ($)", "Profit/Loss (%)"],
                hide_index=True, num_rows="dynamic", use_container_width=True,
                column_config={
                    "Quantity": st.column_config.NumberColumn(step=0.01),
                    **_col_cfg,
                },
            )
            if not edited_holdings_df.equals(holdings_df):
                new_portfolio_data = {}
                for _, row in edited_holdings_df.iterrows():
                    ticker = str(row['Ticker']).upper().strip()
                    if ticker and ticker.lower() not in ['nan', 'none'] and float(row['Quantity']) > 0:
                        new_portfolio_data[ticker] = {
                            "quantity": float(row['Quantity']),
                            "average_cost": float(row['Avg. Cost']),
                        }
                portfolios[selected_portfolio] = new_portfolio_data
                app_data["portfolios"] = portfolios
                replace_portfolio_holdings(selected_portfolio, new_portfolio_data)
                _invalidate_app_data()

        if not holdings_df.empty and total_portfolio_value > 0:
            st.divider()
            st.subheader("Portfolio Visualizations")
            tab_pie, tab_bar, tab_tree = st.tabs(
                ["Allocation (Pie)", "Performance (Bar)", "Heatmap (Treemap)"]
            )
            plot_df = holdings_df[holdings_df['Market Value'] > 0]
            with tab_pie:
                if not plot_df.empty:
                    fig_pie = px.pie(plot_df, values='Market Value', names='Ticker', hole=0.4)
                    fig_pie.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            with tab_bar:
                fig_bar = px.bar(
                    holdings_df, x='Ticker', y='Profit/Loss ($)',
                    title="Absolute Profit/Loss by Asset",
                    color='Profit/Loss ($)',
                    color_continuous_scale=['#dc3545', '#28a745'],
                )
                fig_bar.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with tab_tree:
                if not plot_df.empty:
                    fig_tree = px.treemap(
                        plot_df, path=[px.Constant("Portfolio"), 'Ticker'],
                        values='Market Value', color='Profit/Loss (%)',
                        color_continuous_scale=['#dc3545', '#28a745'],
                        color_continuous_midpoint=0,
                    )
                    fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                    st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No assets in this portfolio yet.")
