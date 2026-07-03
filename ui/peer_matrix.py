import plotly.express as px
import streamlit as st

from api import fetch_peer_metrics
from data_store import (
    add_to_peer_group,
    create_peer_group,
    delete_peer_group,
    remove_from_peer_group,
)
from ui.common import _invalidate_app_data
from utils import sanitize_ticker


def render(app_data: dict) -> None:
    peer_groups = app_data.get("peer_groups", {})

    st.header("Peer Group Comparison Matrix")
    st.markdown("Compare relative valuation multiples across custom industry cohorts.")

    col_pg_sel, col_pg_add, col_pg_del = st.columns(3)
    group_names = list(peer_groups.keys())

    with col_pg_sel:
        selected_group = st.selectbox(
            "Select Industry Cohort", group_names if group_names else ["None"]
        )

    with col_pg_add:
        with st.form("create_group_form", clear_on_submit=True):
            new_group_name = st.text_input("New Cohort Name")
            if st.form_submit_button("Create Cohort"):
                if new_group_name and new_group_name not in peer_groups:
                    if create_peer_group(new_group_name):
                        peer_groups[new_group_name] = []
                        app_data["peer_groups"] = peer_groups
                        _invalidate_app_data()
                        st.toast(f"Created cohort: {new_group_name}", icon="✅")

    with col_pg_del:
        if group_names:
            with st.form("delete_group_form"):
                group_to_delete = st.selectbox("Delete Cohort", group_names)
                if st.form_submit_button("Delete Permanently"):
                    delete_peer_group(group_to_delete)
                    del peer_groups[group_to_delete]
                    app_data["peer_groups"] = peer_groups
                    _invalidate_app_data()
                    st.toast("Cohort deleted.", icon="🗑️")
    st.divider()

    if selected_group and selected_group != "None":
        group_tickers = peer_groups[selected_group]
        col_t_add, col_t_del = st.columns(2)
        with col_t_add:
            with st.form("add_peer_form", clear_on_submit=True):
                new_peer = sanitize_ticker(st.text_input("Add Ticker to Cohort").upper())
                if st.form_submit_button("Add Asset") and new_peer:
                    if new_peer not in group_tickers:
                        if add_to_peer_group(selected_group, new_peer):
                            peer_groups[selected_group].append(new_peer)
                            app_data["peer_groups"] = peer_groups
                            _invalidate_app_data()
        with col_t_del:
            if group_tickers:
                with st.form("remove_peer_form"):
                    peer_to_remove = st.selectbox("Remove Ticker", group_tickers)
                    if st.form_submit_button("Remove Asset"):
                        remove_from_peer_group(selected_group, peer_to_remove)
                        peer_groups[selected_group].remove(peer_to_remove)
                        app_data["peer_groups"] = peer_groups
                        _invalidate_app_data()

        if group_tickers:
            st.subheader(f"Relative Valuation: {selected_group}")
            with st.spinner(f"Auditing financial statements for {len(group_tickers)} peers..."):
                peer_df = fetch_peer_metrics(group_tickers)
            if not peer_df.empty:
                st.dataframe(
                    peer_df, hide_index=True, use_container_width=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                        "P/E (Trailing)": st.column_config.NumberColumn(format="%.2f"),
                        "P/E (Forward)": st.column_config.NumberColumn(format="%.2f"),
                        "P/B": st.column_config.NumberColumn(format="%.2f"),
                        "EV/EBITDA": st.column_config.NumberColumn(format="%.2f"),
                        "ROE (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Debt/Equity": st.column_config.NumberColumn(format="%.2f"),
                        "Div Yield (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )
                st.divider()
                st.markdown("##### Cross-Sectional Analysis Chart")
                chart_metric = st.selectbox(
                    "Select Metric to Visualize",
                    ["P/E (Trailing)", "P/E (Forward)", "P/B", "EV/EBITDA", "ROE (%)", "Debt/Equity", "Div Yield (%)"],
                )
                plot_data = peer_df.dropna(subset=[chart_metric])
                if not plot_data.empty:
                    plot_data = plot_data.sort_values(by=chart_metric, ascending=True)
                    fig_peer = px.bar(
                        plot_data, x='Ticker', y=chart_metric,
                        title=f"{chart_metric} Comparison",
                        color=chart_metric,
                        color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
                    )
                    fig_peer.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_peer, use_container_width=True)
                else:
                    st.info(f"Not enough clean data to plot {chart_metric} for this cohort.")
        else:
            st.info("This cohort is empty. Add some tickers above.")
