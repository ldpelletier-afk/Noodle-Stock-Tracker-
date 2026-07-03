import pandas as pd
import plotly.express as px
import streamlit as st

from data_store import fetch_transactions, import_transactions


def render(app_data: dict) -> None:
    st.header("📒 Trade History & Realized P&L")
    st.caption(
        "Every buy and sell you make is recorded here. Realized P&L is calculated "
        "from sales only — unrealized gains on open positions are shown in the Asset Tracker."
    )

    with st.expander("📥 Import Historical Trades (CSV)", expanded=False):
        st.markdown(
            "Backfill trades from before the SQLite migration — or from any broker export.\n\n"
            "**Required columns:** `date`, `portfolio`, `ticker`, `action` (BUY or SELL), `quantity`, `price`.\n"
            "**Optional:** `cost_basis` — only meaningful for SELL rows. If omitted on a SELL, that row won't "
            "contribute to realized P&L (but still appears in the ledger)."
        )
        template_csv = (
            "date,portfolio,ticker,action,quantity,price,cost_basis\n"
            "2024-03-15,RobinHood,AAPL,BUY,10,172.50,\n"
            "2024-08-02,RobinHood,AAPL,SELL,4,225.00,172.50\n"
        )
        st.download_button(
            "Download CSV template",
            data=template_csv,
            file_name="trades_template.csv",
            mime="text/csv",
            use_container_width=False,
        )

        uploaded_csv = st.file_uploader("Upload your CSV", type=["csv"], key="tx_import")
        if uploaded_csv is not None:
            try:
                import_df = pd.read_csv(uploaded_csv)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                import_df = None

            if import_df is not None:
                required = {"date", "portfolio", "ticker", "action", "quantity", "price"}
                missing = required - set(c.lower() for c in import_df.columns)
                import_df.columns = [c.lower() for c in import_df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
                else:
                    st.write("**Preview** (first 10 rows):")
                    st.dataframe(import_df.head(10), use_container_width=True, hide_index=True)

                    if st.button("Import Trades", type="primary"):
                        rows = []
                        parse_errors = 0
                        for _, r in import_df.iterrows():
                            try:
                                ts = int(pd.to_datetime(r["date"]).timestamp())
                                rows.append({
                                    "ts": ts,
                                    "portfolio_name": str(r["portfolio"]).strip(),
                                    "ticker": str(r["ticker"]).strip().upper(),
                                    "action": str(r["action"]).strip().upper(),
                                    "quantity": float(r["quantity"]),
                                    "price": float(r["price"]),
                                    "cost_basis": (
                                        r.get("cost_basis")
                                        if "cost_basis" in import_df.columns else None
                                    ),
                                })
                            except Exception:
                                parse_errors += 1

                        result = import_transactions(rows)
                        result["errors"] += parse_errors
                        st.success(
                            f"✅ Imported {result['added']} trades. "
                            f"Skipped {result['skipped']} duplicates. "
                            f"{result['errors']} rows had errors."
                        )

    st.divider()

    all_tx = fetch_transactions()

    if not all_tx:
        st.info(
            "No trades logged yet. Buy or sell a position from the Asset Tracker "
            "and it will appear here."
        )
    else:
        tx_df = pd.DataFrame(all_tx)
        tx_df["Date"] = pd.to_datetime(tx_df["ts"], unit="s")
        tx_df["Year"] = tx_df["Date"].dt.year
        tx_df["Proceeds"] = tx_df["quantity"] * tx_df["price"]
        tx_df["Realized P&L"] = tx_df.apply(
            lambda r: (r["price"] - r["cost_basis"]) * r["quantity"]
            if r["action"] == "SELL" and pd.notna(r["cost_basis"])
            else 0.0,
            axis=1,
        )

        sells = tx_df[tx_df["action"] == "SELL"]
        total_realized = sells["Realized P&L"].sum()
        ytd = sells[sells["Year"] == pd.Timestamp.now().year]["Realized P&L"].sum()
        trade_count = len(tx_df)
        win_count = int((sells["Realized P&L"] > 0).sum())
        loss_count = int((sells["Realized P&L"] < 0).sum())
        win_rate = (win_count / len(sells) * 100) if len(sells) else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Realized P&L", f"${total_realized:,.2f}")
        m2.metric(f"{pd.Timestamp.now().year} Realized P&L", f"${ytd:,.2f}")
        m3.metric("Total Trades", f"{trade_count}")
        m4.metric("Win Rate", f"{win_rate:.0f}%", help=f"{win_count} wins / {loss_count} losses")

        st.divider()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            portfolio_filter = st.selectbox(
                "Filter by Portfolio",
                ["All"] + sorted(tx_df["portfolio_name"].unique().tolist()),
            )
        with col_f2:
            ticker_filter = st.selectbox(
                "Filter by Ticker",
                ["All"] + sorted(tx_df["ticker"].unique().tolist()),
            )

        view_df = tx_df.copy()
        if portfolio_filter != "All":
            view_df = view_df[view_df["portfolio_name"] == portfolio_filter]
        if ticker_filter != "All":
            view_df = view_df[view_df["ticker"] == ticker_filter]

        st.subheader("Transaction Log")
        display_df = view_df[[
            "Date", "portfolio_name", "ticker", "action", "quantity",
            "price", "cost_basis", "Proceeds", "Realized P&L",
        ]].rename(columns={
            "portfolio_name": "Portfolio",
            "ticker": "Ticker",
            "action": "Action",
            "quantity": "Qty",
            "price": "Price",
            "cost_basis": "Cost Basis",
        })
        st.dataframe(
            display_df.style.format({
                "Qty": "{:.4f}",
                "Price": "${:,.2f}",
                "Cost Basis": "${:,.2f}",
                "Proceeds": "${:,.2f}",
                "Realized P&L": "${:,.2f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

        if not sells.empty:
            st.subheader("Realized P&L by Ticker")
            by_ticker = (
                sells.groupby("ticker")["Realized P&L"].sum().reset_index()
                .sort_values("Realized P&L", ascending=False)
            )
            fig_ticker = px.bar(
                by_ticker, x="ticker", y="Realized P&L",
                color="Realized P&L",
                color_continuous_scale=["#ff4b4b", "#cccccc", "#28a745"],
                color_continuous_midpoint=0,
            )
            fig_ticker.update_layout(height=320, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_ticker, use_container_width=True)

            st.subheader("Realized P&L by Year")
            by_year = sells.groupby("Year")["Realized P&L"].sum().reset_index()
            fig_year = px.bar(by_year, x="Year", y="Realized P&L")
            fig_year.update_layout(height=280, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_year, use_container_width=True)
