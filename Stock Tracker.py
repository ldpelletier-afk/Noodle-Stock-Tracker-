import os
import time

import ollama
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from plotly.subplots import make_subplots

from api import (
    fetch_all_news,
    fetch_dcf_data,
    fetch_financial_highlights,
    fetch_financial_news,
    fetch_financial_statements,
    fetch_live_prices,
    fetch_macro_data,
    fetch_peer_metrics,
    fetch_recent_sec_filings,
    fetch_sec_filing,
    fetch_stock_details,
    fetch_url_metadata,
    fred,
)
from data_store import (
    add_favorite,
    add_to_watchlist,
    create_watchlist,
    delete_saved_article,
    delete_watchlist,
    fetch_transactions,
    import_transactions,
    list_favorites,
    list_saved_articles,
    load_data as _load_data_sqlite,
    log_transaction,
    remove_favorite,
    remove_from_watchlist,
    rename_watchlist,
    save_article,
    save_data as _save_data_sqlite,
    set_target_in_watchlist,
    update_favorite,
    update_saved_article_note,
)
from rag import CATEGORIES as _CATEGORIES
from rag import TOPIC_LABELS as _TOPIC_LABELS
from rag import TOPICS as _TOPICS
from rag import already_ingested as _already_ingested
from rag import decompose_query as _decompose_query
from rag import delete_document as _delete_document
from rag import format_chunks_for_citation as _format_chunks_for_citation
from rag import ingest_chunks as _ingest_chunks
from rag import list_documents as _list_documents
from rag import retrieve_multi as _retrieve_multi
from rag import route_query as _route_query
from rag import set_category as _set_category
from rag import set_topics as _set_topics
from rag import verify_citations as _verify_citations
from rag import vector_db as _vector_db
from utils import format_large_number, highlight_buy_zone, sanitize_ticker

# Risk analytics + agentic ticker analyst — lazy-friendly; they only hit
# yfinance / ollama when their tabs are actually used.
from risk import portfolio_risk_report as _portfolio_risk_report
from agent import run_analyze_ticker as _run_analyze_ticker

# --- CONFIGURATION & STORAGE ---
DB_DIR = "./chroma_db"
UPLOAD_DIR = "./temp_pdfs"

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="The True Oracle", layout="wide")


def load_data():
    return _load_data_sqlite()


def save_data(data):
    _save_data_sqlite(data)

# --- MAIN APP ---
st.title("The True Oracle: Valuation & Tracking")

app_data = load_data()
portfolios = app_data.get("portfolios", {})
watch_list_targets = app_data.get("watch_list_targets", {})
peer_groups = app_data.get("peer_groups", {})

(
    tab1, tab_fav, tab2, tab_hist, tab_risk, tab3, tab4, tab5, tab6, tab7, tab_analyst
) = st.tabs([
    "📈 Market Watch", "⭐ Favorites", "💼 Asset Tracker", "📒 History", "🛡️ Risk",
    "⚖️ Valuation", "📰 Intelligence", "🏢 Peer Matrix", "🏦 Macro",
    "📚 The Library", "🧠 Analyst",
])

# ===========================
# TAB 1: MARKET WATCH
# ===========================
with tab1:
    watchlists = app_data.get("watchlists", {})

    # Collect every ticker across all lists for a single bulk price fetch.
    all_wl_tickers = list({t for items in watchlists.values() for t in items})

    with st.spinner("Fetching live market data..."):
        live_prices = fetch_live_prices(all_wl_tickers)

    # Global volatility alert (across every list)
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

    # ---- Per-list expandable sections ----
    _COL_CFG = {
        "Live Price (from API)": st.column_config.NumberColumn(format="$%.2f"),
        "Day Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Target Price (Self-set)": st.column_config.NumberColumn(
            format="$%.2f", step=0.01
        ),
    }

    if watchlists:
        for _list_name, _items in watchlists.items():
            _list_tickers = list(_items.keys())
            _label = (
                f"📋 {_list_name}  ·  {len(_list_tickers)} "
                f"stock{'s' if len(_list_tickers) != 1 else ''}"
            )
            with st.expander(_label, expanded=True):

                # Price table
                if _list_tickers:
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
                    })
                    _edited = st.data_editor(
                        _df.style.apply(highlight_buy_zone, axis=1),
                        disabled=["Ticker", "Live Price (from API)", "Day Change (%)"],
                        hide_index=True,
                        use_container_width=True,
                        key=f"wl_editor_{_list_name}",
                        column_config=_COL_CFG,
                    )
                    # Save any changed target prices immediately
                    if not _edited.equals(_df):
                        for _, _row in _edited.iterrows():
                            _old = float(_items.get(_row["Ticker"], 0.0) or 0.0)
                            _new = float(_row["Target Price (Self-set)"] or 0.0)
                            if abs(_new - _old) > 1e-9:
                                set_target_in_watchlist(
                                    _list_name, _row["Ticker"], _new
                                )
                        st.toast(f"Targets updated in '{_list_name}'", icon="✅")
                        st.rerun()
                else:
                    st.caption("This list is empty — add a ticker below.")

                st.divider()

                # Add / Remove ticker
                _ca, _cr = st.columns(2)
                with _ca:
                    st.markdown("**Add ticker**")
                    _add_t = sanitize_ticker(
                        st.text_input(
                            "Ticker",
                            key=f"wl_add_{_list_name}",
                            placeholder="e.g. MSFT",
                            label_visibility="collapsed",
                        )
                    )
                    if st.button(
                        "➕ Add", key=f"wl_addbtn_{_list_name}", use_container_width=True
                    ):
                        if _add_t:
                            if add_to_watchlist(_list_name, _add_t):
                                st.rerun()
                            else:
                                st.warning(f"{_add_t} is already in '{_list_name}'.")
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

                # Rename / Delete list
                st.markdown("---")
                _cn, _cd = st.columns([4, 1])
                with _cn:
                    _new_name = st.text_input(
                        "Rename list",
                        value=_list_name,
                        key=f"wl_rename_{_list_name}",
                        label_visibility="collapsed",
                        placeholder="New list name…",
                    )
                    if st.button(
                        "✏️ Rename list", key=f"wl_renamebtn_{_list_name}"
                    ):
                        if _new_name and _new_name != _list_name:
                            if rename_watchlist(_list_name, _new_name):
                                st.rerun()
                            else:
                                st.warning(
                                    f"A list named '{_new_name}' already exists."
                                )
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
                        st.rerun()
    else:
        st.info("No watch lists yet — create one below.")

    st.divider()

    # ---- Create new list ----
    st.subheader("➕ Create new list")
    _nl_col1, _nl_col2 = st.columns([3, 1])
    with _nl_col1:
        _new_list_name = st.text_input(
            "New list name",
            key="wl_new_list_name",
            placeholder="e.g. Tech Stocks",
            label_visibility="collapsed",
        )
    with _nl_col2:
        if st.button("Create", key="wl_create_list", use_container_width=True):
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
            fetch_stock_details.clear(); fetch_live_prices.clear()

    timeframes = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
    selected_period = st.radio("Timeframe", timeframes, index=2, horizontal=True)

    col_ta1, col_ta2, col_ta3 = st.columns(3)
    with col_ta1: show_sma = st.checkbox("Overlay 50-Period SMA")
    with col_ta2: show_rsi = st.checkbox("Show 14-Period RSI")
    with col_ta3: show_macd = st.checkbox("Show MACD (12, 26, 9)")

    if selected_ticker:
        with st.spinner(f"Loading {selected_ticker} data..."):
            hist_data, stock_info = fetch_stock_details(selected_ticker, selected_period)
            if not hist_data.empty:
                start_price, end_price = hist_data['Close'].iloc[0], hist_data['Close'].iloc[-1]
                chart_color = '#28a745' if end_price >= start_price else '#dc3545'
                
                active_subplots = sum([show_rsi, show_macd])
                total_rows = 1 + active_subplots
                
                row_heights = [1.0] if total_rows == 1 else ([0.7, 0.3] if total_rows == 2 else [0.6, 0.2, 0.2])
                rsi_row = 2 if show_rsi else None
                macd_row = (3 if total_rows == 3 else 2) if show_macd else None

                fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights) if total_rows > 1 else go.Figure()
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], mode='lines', name=f"{selected_ticker} Price", line=dict(color=chart_color, width=2)), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
                
                if show_sma: fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'].rolling(window=50).mean(), mode='lines', name="50-Period SMA", line=dict(color='#3498db', width=1.5, dash='dot')), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
                if show_rsi:
                    delta = hist_data['Close'].diff()
                    gain, loss = delta.clip(lower=0), -1 * delta.clip(upper=0)
                    rs = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    rsi = 100 - (100 / (1 + rs))
                    fig.add_trace(go.Scatter(x=hist_data.index, y=rsi, mode='lines', name="RSI (14)", line=dict(color='#9b59b6', width=1.5)), row=rsi_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="rgba(220, 53, 69, 0.5)", row=rsi_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="rgba(40, 167, 69, 0.5)", row=rsi_row, col=1)
                    fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
                if show_macd:
                    macd_line = hist_data['Close'].ewm(span=12, adjust=False).mean() - hist_data['Close'].ewm(span=26, adjust=False).mean()
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    macd_hist = macd_line - signal_line
                    hist_colors = ['#28a745' if val >= 0 else '#dc3545' for val in macd_hist]
                    fig.add_trace(go.Bar(x=hist_data.index, y=macd_hist, marker_color=hist_colors, name="MACD Histogram"), row=macd_row, col=1)
                    fig.add_trace(go.Scatter(x=hist_data.index, y=macd_line, mode='lines', name="MACD Line", line=dict(color='#2980b9', width=1.5)), row=macd_row, col=1)
                    fig.add_trace(go.Scatter(x=hist_data.index, y=signal_line, mode='lines', name="Signal Line", line=dict(color='#e67e22', width=1.5)), row=macd_row, col=1)

                dynamic_height = 400 + (200 * active_subplots)
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", showlegend=True, height=dynamic_height, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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

            else: st.warning(f"Could not retrieve historical data for {selected_ticker}.")

# ===========================
# TAB: FAVORITES
# ===========================
with tab_fav:
    st.header("⭐ Favorite Stocks")
    st.caption(
        "A curated, high-attention list — track goals, notes, financials, SEC filings, "
        "and noteworthy articles per stock. Articles you save are tied to the specific "
        "ticker you saved them under."
    )

    favorites = list_favorites()
    fav_tickers = list(favorites.keys())

    # ---- Summary strip across all favorites ----
    if fav_tickers:
        with st.spinner("Fetching live data for your favorites..."):
            fav_prices = fetch_live_prices(fav_tickers)

        summary_rows = []
        for t in fav_tickers:
            fav = favorites[t]
            live = fav_prices.get(t, {}) or {}
            price = live.get("price")
            change = live.get("change")
            goal = fav.get("goal_price")
            if price is not None and goal:
                progress = ((price - goal) / goal) * 100
            else:
                progress = None
            summary_rows.append({
                "Ticker": t,
                "Price": price,
                "Day %": change,
                "Goal": goal,
                "vs Goal %": progress,
                "Notes": (fav.get("notes") or "").splitlines()[0][:80] if fav.get("notes") else "",
            })

        summary_df = pd.DataFrame(summary_rows)
        st.subheader("At-a-glance")
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Day %": st.column_config.NumberColumn(format="%.2f%%"),
                "Goal": st.column_config.NumberColumn(format="$%.2f"),
                "vs Goal %": st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )
    else:
        st.info("No favorites yet. Add one below to start tracking.")

    st.divider()

    # ---- Per-stock deep dive ----
    if fav_tickers:
        focus_ticker = st.selectbox(
            "Focus stock",
            fav_tickers,
            key="fav_focus_select",
        )
        fav = favorites[focus_ticker]

        # --- Header: live price + day change + goal progress ---
        live = fav_prices.get(focus_ticker, {}) or {}
        price = live.get("price")
        change = live.get("change")
        goal = fav.get("goal_price")

        hdr1, hdr2, hdr3, hdr4 = st.columns(4)
        hdr1.metric(
            "Live Price",
            f"${price:.2f}" if price is not None else "N/A",
            delta=f"{change:+.2f}%" if change is not None else None,
        )
        hdr2.metric("Goal Price", f"${goal:.2f}" if goal else "— not set —")
        if price is not None and goal:
            delta = price - goal
            pct = (delta / goal) * 100
            hdr3.metric("Δ vs Goal", f"${delta:+.2f}", delta=f"{pct:+.2f}%")
        else:
            hdr3.metric("Δ vs Goal", "—")
        hdr4.metric(
            "Added",
            time.strftime("%Y-%m-%d", time.localtime(fav.get("added_at") or 0))
            if fav.get("added_at") else "—",
        )

        # --- Notes / goal / position controls ---
        # Wrapped in st.form so pressing "Save changes" commits ALL widget
        # values atomically. Without a form, st.number_input only commits on
        # blur — so clicking Save right after typing a goal (without Tab/Enter
        # first) would miss the new value and silently save nothing.
        with st.expander("📝 Notes, goal, and position", expanded=True):
            with st.form(key=f"fav_meta_form_{focus_ticker}", clear_on_submit=False):
                col_n, col_g = st.columns([3, 1])
                with col_n:
                    new_notes = st.text_area(
                        "Notes",
                        value=fav.get("notes") or "",
                        height=140,
                        key=f"fav_notes_{focus_ticker}",
                        help="Your thesis, watch criteria, or anything else worth remembering.",
                    )
                with col_g:
                    new_goal = st.number_input(
                        "Goal price ($)",
                        min_value=0.0,
                        value=float(fav.get("goal_price") or 0.0),
                        step=0.01,
                        format="%.2f",
                        key=f"fav_goal_{focus_ticker}",
                        help="Accepts dollars and cents (e.g. 247.85). "
                             "Set to any value > 0 to save a goal. Leave at 0 to skip. "
                             "Tick 'Clear goal' to explicitly remove an existing goal.",
                    )
                    clear_goal = st.checkbox(
                        "Clear goal",
                        value=False,
                        key=f"fav_clear_goal_{focus_ticker}",
                    )
                new_position = st.text_input(
                    "Position note (e.g., '10 shares @ $150, added 2024-11-01')",
                    value=fav.get("position_note") or "",
                    key=f"fav_pos_{focus_ticker}",
                )
                submitted = st.form_submit_button("💾 Save changes")

            if submitted:
                kwargs = {
                    "notes": new_notes,
                    "position_note": new_position,
                }
                # Only touch goal_price when the user meaningfully asked to.
                # - explicit clear checkbox → clear
                # - new_goal > 0            → set to that value
                # - otherwise               → leave existing goal alone
                if clear_goal:
                    kwargs["clear_goal"] = True
                elif new_goal > 0:
                    kwargs["goal_price"] = float(new_goal)

                update_favorite(focus_ticker, **kwargs)

                saved_bits = ["notes", "position"]
                if clear_goal:
                    saved_bits.append("goal cleared")
                elif new_goal > 0:
                    saved_bits.append(f"goal=${new_goal:.2f}")
                st.toast(
                    f"Saved {focus_ticker}: " + ", ".join(saved_bits),
                    icon="💾",
                )
                st.rerun()

        # --- Financial highlights ---
        st.subheader("💰 Financial Highlights")
        with st.spinner(f"Loading financials for {focus_ticker}..."):
            try:
                highlights = fetch_financial_highlights(focus_ticker)
            except Exception as e:
                highlights = {}
                st.warning(f"Could not fetch financial highlights: {e}")

        if highlights.get("long_name"):
            st.caption(
                f"**{highlights['long_name']}** · {highlights.get('sector') or '—'} "
                f"/ {highlights.get('industry') or '—'} · {highlights.get('currency') or ''}"
            )

        cur = highlights.get("currency") or "USD"
        fh1, fh2, fh3, fh4 = st.columns(4)
        fh1.metric("Revenue (TTM/FY)", format_large_number(highlights.get("revenue")))
        fh2.metric("Net Income", format_large_number(highlights.get("net_income")))
        fh3.metric("Free Cash Flow", format_large_number(highlights.get("fcf")))
        fh4.metric(
            "Profit Margin",
            f"{highlights['profit_margin']:.2f}%" if highlights.get("profit_margin") is not None else "N/A",
        )

        fh5, fh6, fh7, fh8 = st.columns(4)
        fh5.metric("Total Debt", format_large_number(highlights.get("total_debt")))
        fh6.metric("Cash", format_large_number(highlights.get("cash")))
        fh7.metric("Net Debt", format_large_number(highlights.get("net_debt")))
        fh8.metric("Market Cap", format_large_number(highlights.get("market_cap")))

        fh9, fh10, fh11, fh12 = st.columns(4)
        fh9.metric("P/E (Trailing)", f"{highlights['pe_trailing']:.2f}" if highlights.get("pe_trailing") else "N/A")
        fh10.metric("P/E (Forward)", f"{highlights['pe_forward']:.2f}" if highlights.get("pe_forward") else "N/A")
        fh11.metric("EPS (TTM)", f"${highlights['eps_trailing']:.2f}" if highlights.get("eps_trailing") else "N/A")
        fh12.metric("Div Yield", f"{highlights['dividend_yield']:.2f}%" if highlights.get("dividend_yield") else "N/A")

        with st.expander("📄 Full Statements (Income / Balance / Cash Flow)"):
            try:
                statements = fetch_financial_statements(focus_ticker)
            except Exception as e:
                statements = {"income": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame()}
                st.warning(f"Could not fetch statements: {e}")

            stmt_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            for stmt_tab, key, label in zip(
                stmt_tabs,
                ["income", "balance_sheet", "cashflow"],
                ["Income Statement", "Balance Sheet", "Cash Flow"],
            ):
                with stmt_tab:
                    df_stmt = statements.get(key, pd.DataFrame())
                    if df_stmt is None or df_stmt.empty:
                        st.info(f"No {label.lower()} available from yfinance.")
                    else:
                        # Columns are period ends; format as dates and show most recent first
                        try:
                            df_show = df_stmt.copy()
                            df_show.columns = [
                                c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                                for c in df_show.columns
                            ]
                        except Exception:
                            df_show = df_stmt
                        st.dataframe(df_show, use_container_width=True)

        # --- SEC filings ---
        st.subheader("📋 Recent SEC Filings")
        with st.spinner(f"Loading SEC filings for {focus_ticker}..."):
            filings = fetch_recent_sec_filings(focus_ticker, n=10)
        if not filings:
            st.info(
                "No recent SEC filings found (or ticker not in SEC EDGAR — non-US "
                "listings and ETFs typically won't appear)."
            )
        else:
            filings_df = pd.DataFrame([
                {"Form": f["form"], "Date": f["date"], "Link": f["url"]} for f in filings
            ])
            st.dataframe(
                filings_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Link", display_text="📄 Open"),
                },
            )
            with st.expander("🔎 Fetch text of a specific filing type"):
                form_options = sorted({f["form"] for f in filings})
                if form_options:
                    pick_form = st.selectbox(
                        "Form type",
                        form_options,
                        key=f"fav_form_pick_{focus_ticker}",
                    )
                    if st.button("Fetch most recent text", key=f"fav_fetch_filing_{focus_ticker}"):
                        with st.spinner(f"Fetching {pick_form}..."):
                            text, url_or_err = fetch_sec_filing(focus_ticker, form_type=pick_form)
                        if text:
                            st.success(f"Fetched {pick_form} — {url_or_err}")
                            st.text_area(
                                f"{pick_form} text (first 10k chars)",
                                value=text[:10000],
                                height=300,
                                key=f"fav_filing_text_{focus_ticker}_{pick_form}",
                            )
                        else:
                            st.warning(url_or_err)

        # --- News: auto feed + saved articles ---
        st.subheader("📰 News")
        news_col1, news_col2 = st.tabs(["🌐 Auto feed (multi-source)", "⭐ Saved articles"])

        with news_col1:
            refresh_news = st.button("🔄 Refresh feed", key=f"fav_news_refresh_{focus_ticker}")
            if refresh_news:
                fetch_all_news.clear()
            with st.spinner("Fetching news from all configured sources..."):
                try:
                    articles = fetch_all_news(focus_ticker)
                except Exception as e:
                    articles = []
                    st.warning(f"News fetch error: {e}")

            if not articles:
                st.info("No news articles found for this ticker.")
            else:
                st.caption(f"{len(articles)} article(s), newest first. Sources include Yahoo, "
                           f"Google News (incl. WSJ/FT/Bloomberg/Reuters site filters), Seeking Alpha, "
                           f"and any configured Finnhub/NewsAPI/MarketAux keys.")
                for art in articles[:40]:
                    when = (
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(art["published_ts"]))
                        if art.get("published_ts") else art.get("time") or "Recent"
                    )
                    c_art, c_btn = st.columns([5, 1])
                    with c_art:
                        st.markdown(
                            f"**[{art['title']}]({art['url']})**  \n"
                            f"*{art.get('source') or 'Source'}* · {when}"
                        )
                        if art.get("summary"):
                            st.caption(art["summary"][:280] + ("…" if len(art["summary"]) > 280 else ""))
                    with c_btn:
                        if st.button(
                            "⭐ Save",
                            key=f"fav_save_feed_{focus_ticker}_{hash(art['url'])}",
                            help="Save this article to this ticker's noteworthy list",
                        ):
                            new_id = save_article(
                                ticker=focus_ticker,
                                url=art["url"],
                                title=art.get("title"),
                                source=art.get("source"),
                                note="",
                                published_at=art.get("published_ts") or None,
                            )
                            if new_id:
                                st.toast(
                                    f"Saved to {focus_ticker}'s noteworthy list",
                                    icon="⭐",
                                )
                            else:
                                st.toast("Already saved for this ticker", icon="ℹ️")

        with news_col2:
            st.caption(
                f"Articles below are saved **specifically to {focus_ticker}**. "
                "Paste any URL (WSJ+, FT, blog, analyst report, etc.) — we'll pull "
                "the title/source from the page's public Open Graph metadata "
                "(works even for paywalled articles, since OG tags are public)."
            )

            with st.form(key=f"fav_url_form_{focus_ticker}", clear_on_submit=True):
                paste_url = st.text_input(
                    "URL to save",
                    placeholder="https://www.wsj.com/articles/...",
                    key=f"fav_paste_url_{focus_ticker}",
                )
                paste_note = st.text_area(
                    "Optional note (why is this noteworthy?)",
                    key=f"fav_paste_note_{focus_ticker}",
                    height=80,
                )
                submit_url = st.form_submit_button("💾 Save URL to this stock")
                if submit_url and paste_url.strip():
                    with st.spinner("Fetching article metadata..."):
                        meta = fetch_url_metadata(paste_url.strip())
                    new_id = save_article(
                        ticker=focus_ticker,
                        url=paste_url.strip(),
                        title=meta.get("title") or paste_url.strip(),
                        source=meta.get("source") or "",
                        note=paste_note or "",
                        published_at=meta.get("published_ts") or None,
                    )
                    if new_id:
                        st.success(
                            f"Saved to {focus_ticker}: "
                            f"{meta.get('title') or paste_url.strip()}"
                        )
                    else:
                        st.info(f"This URL is already saved for {focus_ticker}.")
                    st.rerun()

            saved = list_saved_articles(focus_ticker)
            st.write(f"**{len(saved)} saved article(s)** for `{focus_ticker}`")
            for art in saved:
                pub_ts = art.get("published_at") or art.get("saved_at")
                when = time.strftime("%Y-%m-%d", time.localtime(pub_ts)) if pub_ts else "—"
                with st.container(border=True):
                    c_info, c_note, c_act = st.columns([4, 3, 1])
                    with c_info:
                        st.markdown(
                            f"**[{art.get('title') or art['url']}]({art['url']})**  \n"
                            f"*{art.get('source') or 'Source'}* · {when} · "
                            f"saved {time.strftime('%Y-%m-%d', time.localtime(art['saved_at']))}"
                        )
                    with c_note:
                        new_note = st.text_input(
                            "Note",
                            value=art.get("note") or "",
                            key=f"fav_saved_note_{art['id']}",
                            label_visibility="collapsed",
                            placeholder="Your note...",
                        )
                        if new_note != (art.get("note") or ""):
                            if st.button("Save note", key=f"fav_saved_note_btn_{art['id']}"):
                                update_saved_article_note(art["id"], new_note)
                                st.toast("Note updated", icon="✏️")
                                st.rerun()
                    with c_act:
                        if st.button("🗑️", key=f"fav_saved_del_{art['id']}", help="Remove"):
                            delete_saved_article(art["id"])
                            st.toast("Removed", icon="🗑️")
                            st.rerun()

    st.divider()

    # ---- Add / Remove favorites ----
    col_add, col_del = st.columns(2)
    with col_add:
        st.subheader("⭐ Add a favorite")
        add_input = st.text_input(
            "Ticker (e.g. AAPL, RY.TO, SHOP)",
            key="fav_add_input",
        )
        if st.button("Add to favorites", key="fav_add_btn", use_container_width=True):
            sym = sanitize_ticker(add_input)
            if sym:
                if add_favorite(sym):
                    st.toast(f"{sym} added to favorites", icon="⭐")
                    st.rerun()
                else:
                    st.warning(f"{sym} is already in your favorites.")

    with col_del:
        st.subheader("🗑️ Remove a favorite")
        if fav_tickers:
            rm_choice = st.selectbox(
                "Select favorite to remove",
                fav_tickers,
                key="fav_rm_select",
            )
            if st.button(
                "Remove (also deletes its saved articles)",
                type="primary",
                key="fav_rm_btn",
                use_container_width=True,
            ):
                remove_favorite(rm_choice)
                st.toast(f"{rm_choice} removed", icon="🗑️")
                st.rerun()
        else:
            st.caption("No favorites to remove.")

# ===========================
# TAB 2: ASSET TRACKER
# ===========================
with tab2:
    st.header("Portfolio Asset Tracker")
    with st.expander("Creating and Managing Portfolios", expanded=not bool(portfolios)):
        col_p1, col_p2 = st.columns([3, 1])
        new_portfolio_name = col_p1.text_input("New Portfolio Name")
        if col_p2.button("Create Portfolio"):
            if new_portfolio_name and new_portfolio_name not in portfolios:
                portfolios[new_portfolio_name] = {}
                app_data["portfolios"] = portfolios
                save_data(app_data); st.toast(f"Portfolio '{new_portfolio_name}' created!", icon="🎉")
            elif new_portfolio_name in portfolios: st.warning("Portfolio already exists.")
    st.divider()

    portfolio_names = list(portfolios.keys())
    if not portfolio_names:
        st.info("Please create a portfolio to get started.")
        st.stop()

    selected_portfolio = st.selectbox("Select Portfolio to View", ["All Portfolios"] + portfolio_names)

    current_holdings = {}
    if selected_portfolio == "All Portfolios":
        for p_name in portfolios:
            for ticker, data in portfolios[p_name].items():
                if ticker in current_holdings:
                    total_qty = current_holdings[ticker]['quantity'] + data['quantity']
                    total_cost = (current_holdings[ticker]['quantity'] * current_holdings[ticker]['average_cost']) + (data['quantity'] * data['average_cost'])
                    current_holdings[ticker]['average_cost'] = total_cost / total_qty
                    current_holdings[ticker]['quantity'] = total_qty
                else: current_holdings[ticker] = data.copy()
    else: current_holdings = portfolios[selected_portfolio]

    if selected_portfolio != "All Portfolios":
        col_add_stock, col_sell_stock, col_manage_cash, col_delete = st.columns(4)
        with col_add_stock:
            with st.expander(f"Add Stock", expanded=False):
                with st.form("add_asset_form", clear_on_submit=True):
                    asset_ticker = sanitize_ticker(st.text_input("Ticker").upper())
                    asset_qty = st.number_input("Quantity", min_value=0.01, step=0.01)
                    asset_cost = st.number_input("Avg. Cost ($)", min_value=0.0, step=0.01)
                    if st.form_submit_button("Buy"):
                        if asset_ticker and asset_ticker != "CASH" and asset_qty > 0:
                            if asset_ticker in portfolios[selected_portfolio]:
                                old_qty = portfolios[selected_portfolio][asset_ticker]['quantity']
                                old_cost = portfolios[selected_portfolio][asset_ticker]['average_cost']
                                portfolios[selected_portfolio][asset_ticker] = {"quantity": old_qty + asset_qty, "average_cost": ((old_qty * old_cost) + (asset_qty * asset_cost)) / (old_qty + asset_qty)}
                            else: portfolios[selected_portfolio][asset_ticker] = {"quantity": asset_qty, "average_cost": asset_cost}
                            app_data["portfolios"] = portfolios
                            save_data(app_data)
                            log_transaction(selected_portfolio, asset_ticker, "BUY", asset_qty, asset_cost, cost_basis=asset_cost)
                            st.toast(f"Added {asset_ticker}", icon="💰")

        with col_sell_stock:
            with st.expander(f"Sell Stock", expanded=False):
                sellable_assets = [t for t in portfolios[selected_portfolio].keys() if t != "CASH"]
                if sellable_assets:
                    with st.form("sell_asset_form", clear_on_submit=True):
                        sell_ticker = st.selectbox("Asset", sellable_assets)
                        current_qty = portfolios[selected_portfolio].get(sell_ticker, {}).get("quantity", 0.0)
                        sell_qty = st.number_input("Qty to Sell", min_value=0.01, max_value=float(current_qty), step=0.01)
                        sell_price = st.number_input("Sale Price ($)", min_value=0.0, step=0.01)
                        if st.form_submit_button("Execute Sale"):
                            if sell_qty > 0 and sell_price >= 0:
                                avg_cost_at_sale = portfolios[selected_portfolio][sell_ticker]["average_cost"]
                                proceeds = sell_qty * sell_price
                                portfolios[selected_portfolio][sell_ticker]["quantity"] -= sell_qty
                                if portfolios[selected_portfolio][sell_ticker]["quantity"] <= 0.0001: del portfolios[selected_portfolio][sell_ticker]
                                current_cash = portfolios[selected_portfolio].get("CASH", {"quantity": 0.0, "average_cost": 1.0})
                                portfolios[selected_portfolio]["CASH"] = {"quantity": current_cash["quantity"] + proceeds, "average_cost": 1.0}
                                app_data["portfolios"] = portfolios
                                save_data(app_data)
                                log_transaction(selected_portfolio, sell_ticker, "SELL", sell_qty, sell_price, cost_basis=avg_cost_at_sale)
                                st.toast(f"Sold {sell_ticker}", icon="🤝")
                else: st.info("No stocks to sell.")

        with col_manage_cash:
            with st.expander(f"Manage Cash", expanded=False):
                with st.form("manage_cash_form", clear_on_submit=True):
                    cash_action = st.radio("Action", ["Deposit", "Withdraw"], horizontal=True)
                    cash_amount = st.number_input("Amount ($)", min_value=0.01, step=100.0)
                    if st.form_submit_button("Update Cash"):
                        current_cash_data = portfolios[selected_portfolio].get("CASH", {"quantity": 0.0, "average_cost": 1.0})
                        new_qty = current_cash_data["quantity"] + cash_amount if cash_action == "Deposit" else max(0.0, current_cash_data["quantity"] - cash_amount)
                        portfolios[selected_portfolio]["CASH"] = {"quantity": new_qty, "average_cost": 1.0}
                        app_data["portfolios"] = portfolios
                        save_data(app_data)

        with col_delete:
             with st.expander(f"Delete Asset", expanded=False):
                 assets_to_delete = list(portfolios[selected_portfolio].keys())
                 if assets_to_delete:
                     del_asset = st.selectbox("Select Asset to Delete", assets_to_delete)
                     if st.button("Delete Permanently", type="primary"):
                         del portfolios[selected_portfolio][del_asset]
                         app_data["portfolios"] = portfolios
                         save_data(app_data); st.toast(f"Deleted {del_asset}", icon="🗑️")
                 else: st.info("No assets.")

    if current_holdings:
        holding_tickers = list(current_holdings.keys())
        with st.spinner("Fetching live prices for holdings..."):
            holding_prices = fetch_live_prices(holding_tickers)
        
        holdings_data = []
        total_portfolio_value = 0
        total_portfolio_cost = 0

        for ticker, data in current_holdings.items():
            qty, avg_cost = data['quantity'], data['average_cost']
            if ticker == "CASH":
                current_price, market_value, total_cost, profit_loss, pl_percent = 1.00, qty, qty, 0.0, 0.0
            else:
                current_price = holding_prices.get(ticker, {}).get('price', 0) or 0
                market_value, total_cost = qty * current_price, qty * avg_cost
                profit_loss = market_value - total_cost
                pl_percent = (profit_loss / total_cost * 100) if total_cost > 0 else 0

            total_portfolio_value += market_value
            total_portfolio_cost += total_cost
            holdings_data.append({"Ticker": ticker, "Quantity": qty, "Avg. Cost": avg_cost, "Current Price": current_price, "Market Value": market_value, "Profit/Loss ($)": profit_loss, "Profit/Loss (%)": pl_percent / 100})
        
        holdings_df = pd.DataFrame(holdings_data)
        total_pl = total_portfolio_value - total_portfolio_cost
        total_pl_pct = (total_pl / total_portfolio_cost * 100) if total_portfolio_cost > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
        m2.metric("Total Gain/Loss ($)", f"${total_pl:,.2f}", delta=f"${total_pl:,.2f}")
        m3.metric("Total Gain/Loss (%)", f"{total_pl_pct:.2f}%", delta=f"{total_pl_pct:.2f}%")
        st.divider()

        if selected_portfolio == "All Portfolios":
            st.subheader("Aggregated Holdings (Read-Only)")
            st.dataframe(holdings_df, use_container_width=True, hide_index=True, column_config={"Avg. Cost": st.column_config.NumberColumn(format="$%.2f"), "Current Price": st.column_config.NumberColumn(format="$%.2f"), "Market Value": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%")})
        else:
            st.subheader(f"Manage Holdings: {selected_portfolio}")
            edited_holdings_df = st.data_editor(holdings_df, disabled=["Ticker", "Current Price", "Market Value", "Profit/Loss ($)", "Profit/Loss (%)"], hide_index=True, num_rows="dynamic", use_container_width=True, column_config={"Quantity": st.column_config.NumberColumn(step=0.01), "Avg. Cost": st.column_config.NumberColumn(format="$%.2f", step=0.01), "Current Price": st.column_config.NumberColumn(format="$%.2f"), "Market Value": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"), "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%")})
            if not edited_holdings_df.equals(holdings_df):
                new_portfolio_data = {}
                for index, row in edited_holdings_df.iterrows():
                    ticker = str(row['Ticker']).upper().strip()
                    if ticker and ticker.lower() not in ['nan', 'none'] and float(row['Quantity']) > 0: 
                        new_portfolio_data[ticker] = {"quantity": float(row['Quantity']), "average_cost": float(row['Avg. Cost'])}
                portfolios[selected_portfolio] = new_portfolio_data
                app_data["portfolios"] = portfolios
                save_data(app_data)

        if not holdings_df.empty and total_portfolio_value > 0:
            st.divider()
            st.subheader("Portfolio Visualizations")
            tab_pie, tab_bar, tab_tree = st.tabs(["Allocation (Pie)", "Performance (Bar)", "Heatmap (Treemap)"])
            plot_df = holdings_df[holdings_df['Market Value'] > 0]
            with tab_pie:
                if not plot_df.empty:
                    fig_pie = px.pie(plot_df, values='Market Value', names='Ticker', hole=0.4)
                    fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, use_container_width=True)
            with tab_bar:
                fig_bar = px.bar(holdings_df, x='Ticker', y='Profit/Loss ($)', title="Absolute Profit/Loss by Asset", color='Profit/Loss ($)', color_continuous_scale=['#dc3545', '#28a745'])
                fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
            with tab_tree:
                if not plot_df.empty:
                    fig_tree = px.treemap(plot_df, path=[px.Constant("Portfolio"), 'Ticker'], values='Market Value', color='Profit/Loss (%)', color_continuous_scale=['#dc3545', '#28a745'], color_continuous_midpoint=0)
                    fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                    st.plotly_chart(fig_tree, use_container_width=True)
    else: st.info("No assets in this portfolio yet.")

# ===========================
# TAB HISTORY: TRADE LEDGER & REALIZED P&L
# ===========================
with tab_hist:
    st.header("📒 Trade History & Realized P&L")
    st.caption("Every buy and sell you make is recorded here. Realized P&L is calculated from sales only — unrealized gains on open positions are shown in the Asset Tracker.")

    # --- IMPORT HISTORICAL TRADES ---
    with st.expander("📥 Import Historical Trades (CSV)", expanded=False):
        st.markdown(
            "Backfill trades from before the SQLite migration — or from any broker export.\n\n"
            "**Required columns:** `date`, `portfolio`, `ticker`, `action` (BUY or SELL), `quantity`, `price`.\n"
            "**Optional:** `cost_basis` — only meaningful for SELL rows. If omitted on a SELL, that row won't contribute to realized P&L (but still appears in the ledger)."
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
                                    "cost_basis": r.get("cost_basis") if "cost_basis" in import_df.columns else None,
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
        st.info("No trades logged yet. Buy or sell a position from the Asset Tracker and it will appear here.")
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

# ===========================
# TAB RISK: PORTFOLIO RISK DASHBOARD
# ===========================
with tab_risk:
    st.header("🛡️ Portfolio Risk Dashboard")
    st.caption(
        "VaR, drawdown, correlation, beta, and factor exposure on your "
        "live holdings. Uses daily adjusted closes from yfinance — "
        "computed client-side, cached for an hour."
    )

    _risk_portfolios = list(portfolios.keys())
    if not _risk_portfolios:
        st.info("Create a portfolio in the Asset Tracker tab to unlock this module.")
    else:
        r_col1, r_col2, r_col3, r_col4 = st.columns([2, 1, 1, 1])
        with r_col1:
            risk_portfolio = st.selectbox(
                "Portfolio",
                ["All Portfolios"] + _risk_portfolios,
                key="risk_portfolio",
            )
        with r_col2:
            risk_period = st.selectbox(
                "Lookback", ["1y", "2y", "5y"], index=1, key="risk_period",
            )
        with r_col3:
            risk_conf = st.selectbox(
                "VaR Confidence", ["95%", "99%"], index=0, key="risk_conf",
            )
        with r_col4:
            risk_bench = st.selectbox(
                "Benchmark", ["SPY", "QQQ", "IWM", "ACWI"], index=0, key="risk_bench",
            )

        # Assemble holdings.
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
        elif st.button("Run Risk Analysis", type="primary", use_container_width=True):
            with st.spinner("Fetching price history & crunching risk metrics..."):
                _lp_tickers = [t for t in risk_holdings if t.upper() != "CASH"]
                _lp = fetch_live_prices(_lp_tickers)
                _conf = 0.95 if risk_conf == "95%" else 0.99
                report = _portfolio_risk_report(
                    risk_holdings,
                    _lp,
                    period=risk_period,
                    var_confidence=_conf,
                    benchmark=risk_bench,
                )

            if report.get("error"):
                st.error(report["error"])
            else:
                # ----- Headline metrics -----
                st.subheader("Headline Metrics")
                h1, h2, h3, h4 = st.columns(4)
                h1.metric(
                    "Ann. Return", f"{report['ann_return']*100:,.2f}%"
                    if pd.notna(report['ann_return']) else "—",
                )
                h2.metric(
                    "Ann. Volatility", f"{report['ann_volatility']*100:,.2f}%"
                    if pd.notna(report['ann_volatility']) else "—",
                )
                h3.metric(
                    "Sharpe (rf=4%)", f"{report['sharpe']:.2f}"
                    if pd.notna(report['sharpe']) else "—",
                )
                h4.metric(
                    "Sortino", f"{report['sortino']:.2f}"
                    if pd.notna(report['sortino']) else "—",
                )

                v1, v2, v3, v4 = st.columns(4)
                v1.metric(
                    f"1-Day VaR ({risk_conf}, hist.)",
                    f"{report['hist_var_1d']*100:,.2f}%"
                    if pd.notna(report['hist_var_1d']) else "—",
                    help="Largest expected daily loss at the stated confidence "
                         "level, based on observed history.",
                )
                v2.metric(
                    f"1-Day CVaR ({risk_conf})",
                    f"{report['hist_cvar_1d']*100:,.2f}%"
                    if pd.notna(report['hist_cvar_1d']) else "—",
                    help="Expected loss conditional on exceeding the VaR "
                         "threshold — the 'how bad is bad' number.",
                )
                v3.metric(
                    "Max Drawdown",
                    f"{report['max_drawdown']*100:,.2f}%"
                    if pd.notna(report['max_drawdown']) else "—",
                    help=f"Duration: {report.get('dd_days', 0)} days peak-to-trough.",
                )
                v4.metric(
                    f"Beta vs {risk_bench}",
                    f"{report['beta']:.2f}"
                    if pd.notna(report['beta']) else "—",
                    help=f"R²={report['r_squared']:.2f}" if pd.notna(report['r_squared']) else "",
                )

                # ----- Parametric cross-check callout -----
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

                # ----- Equity curve + drawdown -----
                st.subheader("Portfolio Equity Curve & Drawdown")
                port_r = report["portfolio_returns"]
                equity = (1 + port_r).cumprod() if not port_r.empty else port_r
                # Use exp(cumsum) since we compute log returns — more accurate.
                import numpy as _np
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
                        x=equity.index, y=(equity - 1) * 100,
                        mode="lines", name="Portfolio",
                        line=dict(color="#2980b9", width=2),
                    ),
                    row=1, col=1,
                )
                fig_eq.add_trace(
                    go.Scatter(
                        x=dd_series.index, y=dd_series,
                        mode="lines", name="Drawdown",
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

                # ----- Per-position risk contribution -----
                st.subheader("Per-Position Risk Contribution")
                st.caption(
                    "Marginal Contribution to Risk = weight × cov(asset, port) / var(port). "
                    "A position with weight 10% but risk-contribution 25% is a concentrated "
                    "volatility source."
                )
                _pp = report["per_position"].copy()
                st.dataframe(
                    _pp.style.format({
                        "Weight": "{:.2%}",
                        "Vol (ann.)": "{:.2%}",
                        "Beta": "{:.2f}",
                        "Contribution to Risk": "{:.2%}",
                    }, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )

                # Concentration line
                _c = report["concentration"]
                if pd.notna(_c.get("hhi")):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Positions", f"{_c['n_positions']}")
                    c2.metric(
                        "Effective N (1/HHI)", f"{_c['effective_n']:.1f}",
                        help="How many equally-weighted names this portfolio "
                             "diversifies LIKE. Much smaller than position count = "
                             "concentrated.",
                    )
                    c3.metric(
                        "Largest Weight", f"{_c['top']*100:.1f}%",
                        help="Single-name concentration — a red flag above ~20-25%.",
                    )

                st.divider()

                # ----- Correlation heatmap -----
                corr = report.get("correlation")
                if corr is not None and not corr.empty and corr.shape[0] >= 2:
                    st.subheader("Holdings Correlation Matrix")
                    st.caption(
                        "Daily-return correlations. Highly correlated clusters "
                        "(>0.7) hint that two positions are really the same bet."
                    )
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=list(corr.columns), y=list(corr.index),
                        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                        text=corr.round(2).values,
                        texttemplate="%{text}", textfont=dict(size=10),
                    ))
                    fig_corr.update_layout(
                        height=max(350, 30 * len(corr) + 150),
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                st.divider()

                # ----- Rolling beta -----
                roll_b = report.get("rolling_beta")
                if roll_b is not None and not roll_b.empty:
                    st.subheader(f"Rolling 63-Day Beta vs {risk_bench}")
                    fig_rb = go.Figure()
                    fig_rb.add_trace(go.Scatter(
                        x=roll_b.index, y=roll_b.values,
                        mode="lines", line=dict(color="#8e44ad", width=2),
                        name="Rolling Beta",
                    ))
                    fig_rb.add_hline(y=1.0, line_dash="dash",
                                     line_color="rgba(128,128,128,0.5)")
                    fig_rb.update_layout(
                        height=320, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_rb, use_container_width=True)

                # ----- Factor exposure -----
                fx = report.get("factor_exposure") or {}
                if fx.get("loadings"):
                    st.subheader("Factor Exposure (ETF-proxy decomposition)")
                    st.caption(
                        "Regression of portfolio returns on: "
                        "MKT (SPY), SMB (IWM−SPY), HML (VTV−VUG), "
                        "MOM (MTUM−SPY), RATES (TLT). "
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
                        help="Return not explained by factor exposures. "
                             "Persistent + alpha = skill or missing factor.",
                    )
                    fxc2.metric("R² (factor fit)", f"{fx['r2']:.2f}")
                    fxc3.metric("Obs", f"{fx['n_obs']}")

                st.caption(
                    f"Computed on {report['n_obs']} daily observations over "
                    f"{report['period']}. Risk-free rate assumed 4% for "
                    f"Sharpe/Sortino. Benchmark: {risk_bench}."
                )

# ===========================
# TAB 3: THE VALUATION MACHINE
# ===========================
with tab3:
    st.header("The Valuation Machine")
    st.markdown("Calculate the true Intrinsic Value of an asset based on its future cash generation capabilities.")
    
    st.subheader("Asset Selection")
    val_ticker = sanitize_ticker(st.text_input("Enter Ticker to Value", value="AAPL", key="val_input").upper())
    
    st.write("---")
    st.markdown("**The Math:**")
    st.markdown(r"$$PV = \sum_{t=1}^{n} \frac{FCF_t}{(1+r)^t} + \frac{TV}{(1+r)^n}$$")
    st.caption("$FCF$ = Free Cash Flow | $r$ = Discount Rate | $TV$ = Terminal Value")
    st.divider()
    
    st.subheader("Philosophical Assumptions")
    c_assump1, c_assump2, c_assump3 = st.columns(3)
    with c_assump1: discount_rate = st.slider("Discount Rate (r)", min_value=0.01, max_value=0.20, value=0.10, step=0.01)
    with c_assump2: growth_rate = st.slider("Growth Rate (Years 1-5)", min_value=-0.10, max_value=0.50, value=0.08, step=0.01)
    with c_assump3: terminal_rate = st.slider("Terminal Growth Rate", min_value=0.01, max_value=0.05, value=0.025, step=0.005)

    if val_ticker and st.button("Calculate Intrinsic Value", use_container_width=True, type="primary"):
        with st.spinner(f"Auditing financial statements for {val_ticker}..."):
            fcf, shares, current_price = fetch_dcf_data(val_ticker)
            if fcf is None or shares is None or current_price is None: st.error("Insufficient financial data available from Yahoo Finance.")
            elif fcf <= 0: st.warning(f"{val_ticker} currently generates negative Free Cash Flow. Standard DCF model collapses.")
            else:
                projected_cf, current_fcf = [], fcf
                for year in range(1, 6):
                    current_fcf *= (1 + growth_rate)
                    projected_cf.append(current_fcf / ((1 + discount_rate) ** year))
                
                sum_pv_cf = sum(projected_cf)
                terminal_value = (current_fcf * (1 + terminal_rate)) / (discount_rate - terminal_rate)
                pv_terminal_value = terminal_value / ((1 + discount_rate) ** 5)
                intrinsic_value_per_share = (sum_pv_cf + pv_terminal_value) / shares
                margin_of_safety = ((intrinsic_value_per_share - current_price) / intrinsic_value_per_share) * 100
                
                st.divider()
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Live Market Price", f"${current_price:.2f}")
                if intrinsic_value_per_share > current_price:
                    res_col2.success(f"### Intrinsic Value\n## ${intrinsic_value_per_share:.2f}")
                    res_col3.metric("Margin of Safety", f"{margin_of_safety:.1f}%", delta="Undervalued")
                else:
                    res_col2.error(f"### Intrinsic Value\n## ${intrinsic_value_per_share:.2f}")
                    res_col3.metric("Premium to Value", f"{abs(margin_of_safety):.1f}%", delta="-Overvalued", delta_color="inverse")

                st.write("**Audit Breakdown:**")
                st.table(pd.DataFrame({
                    "Metric": ["Trailing Free Cash Flow", "Shares Outstanding", "Sum of PV (5 Years)", "PV of Terminal Value", "Total Enterprise Value"],
                    "Value": [format_large_number(fcf), format_large_number(shares), format_large_number(sum_pv_cf), format_large_number(pv_terminal_value), format_large_number(sum_pv_cf + pv_terminal_value)]
                }))

# ===========================
# TAB 4: MARKET INTELLIGENCE (AI POWERED)
# ===========================
with tab4:
    st.header("Market Intelligence")
    st.markdown("Live, unfiltered news feeds analyzed entirely offline by Llama 3.2.")
    
    news_ticker = sanitize_ticker(st.text_input("Target Asset for Reconnaissance", value="AAPL", key="intel_search").upper())
    st.divider()
    
    if news_ticker:
        with st.spinner(f"Intercepting raw XML data feeds for {news_ticker}..."):
            news_data = fetch_financial_news(news_ticker)
            
            if not news_data:
                st.info(f"No recent institutional coverage found for {news_ticker} on the main XML feeds.")
            else:
                for article in news_data:
                    title = article['title']
                    link = article['link']
                    publisher = article['publisher']
                    time_str = article['time']

                    st.markdown(f"#### [{title}]({link})")
                    st.caption(f"🗞️ **Source:** {publisher} | 🕒 **Published:** {time_str}")
                    
                    with st.spinner("🤖 Noodle Bot analyzing sentiment..."):
                        try:
                            ai_prompt = f"Analyze this financial news headline for the stock {news_ticker}: '{title}'. Respond strictly in this exact format: [BULLISH/BEARISH/NEUTRAL] - [One concise sentence explanation of why]."
                            
                            oracle_persona = """You are 'The True Oracle', an elite financial AI. You must strictly obey the following rules:
1. The Logic-First Filter: Perform a Logical Audit defining the Domain of Discourse and isolating atomic propositions.
2. Probabilistic Calibration: Reject binary True/False. Treat new info as Evidence updating a Prior Belief."""

                            response = ollama.chat(model='llama3.2', messages=[
                                {'role': 'system', 'content': oracle_persona},
                                {'role': 'user', 'content': ai_prompt}
                            ])
                            ai_analysis = response['message']['content'].strip()
                            
                            if "BULLISH" in ai_analysis.upper(): st.success(f"**AI Sentiment:** {ai_analysis}")
                            elif "BEARISH" in ai_analysis.upper(): st.error(f"**AI Sentiment:** {ai_analysis}")
                            else: st.info(f"**AI Sentiment:** {ai_analysis}")
                        except Exception as ai_e:
                            st.warning("AI Engine offline. Ensure the Ollama Mac app is running in your menu bar.")
                    st.write("---")

# ===========================
# TAB 5: PEER MATRIX
# ===========================
with tab5:
    st.header("Peer Group Comparison Matrix")
    st.markdown("Compare relative valuation multiples across custom industry cohorts.")

    col_pg_sel, col_pg_add, col_pg_del = st.columns(3)
    group_names = list(peer_groups.keys())
    
    with col_pg_sel:
        selected_group = st.selectbox("Select Industry Cohort", group_names if group_names else ["None"])
    
    with col_pg_add:
        with st.form("create_group_form", clear_on_submit=True):
            new_group_name = st.text_input("New Cohort Name")
            if st.form_submit_button("Create Cohort"):
                if new_group_name and new_group_name not in peer_groups:
                    peer_groups[new_group_name] = []
                    app_data["peer_groups"] = peer_groups
                    save_data(app_data); st.toast(f"Created cohort: {new_group_name}", icon="✅")

    with col_pg_del:
        if group_names:
            with st.form("delete_group_form"):
                group_to_delete = st.selectbox("Delete Cohort", group_names)
                if st.form_submit_button("Delete Permanently"):
                    del peer_groups[group_to_delete]
                    app_data["peer_groups"] = peer_groups
                    save_data(app_data); st.toast("Cohort deleted.", icon="🗑️")
    st.divider()

    if selected_group and selected_group != "None":
        group_tickers = peer_groups[selected_group]
        col_t_add, col_t_del = st.columns(2)
        with col_t_add:
            with st.form("add_peer_form", clear_on_submit=True):
                new_peer = sanitize_ticker(st.text_input("Add Ticker to Cohort").upper())
                if st.form_submit_button("Add Asset") and new_peer:
                    if new_peer not in group_tickers:
                        peer_groups[selected_group].append(new_peer)
                        app_data["peer_groups"] = peer_groups
                        save_data(app_data)
        with col_t_del:
            if group_tickers:
                with st.form("remove_peer_form"):
                    peer_to_remove = st.selectbox("Remove Ticker", group_tickers)
                    if st.form_submit_button("Remove Asset"):
                        peer_groups[selected_group].remove(peer_to_remove)
                        app_data["peer_groups"] = peer_groups
                        save_data(app_data)

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
                        "Div Yield (%)": st.column_config.NumberColumn(format="%.2f%%")
                    }
                )
                st.divider()
                st.markdown("##### Cross-Sectional Analysis Chart")
                chart_metric = st.selectbox("Select Metric to Visualize", ["P/E (Trailing)", "P/E (Forward)", "P/B", "EV/EBITDA", "ROE (%)", "Debt/Equity", "Div Yield (%)"])
                plot_data = peer_df.dropna(subset=[chart_metric])
                if not plot_data.empty:
                    plot_data = plot_data.sort_values(by=chart_metric, ascending=True)
                    fig_peer = px.bar(
                        plot_data, x='Ticker', y=chart_metric, 
                        title=f"{chart_metric} Comparison",
                        color=chart_metric, color_continuous_scale=['#28a745', '#ffc107', '#dc3545']
                    )
                    fig_peer.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_peer, use_container_width=True)
                else: st.info(f"Not enough clean data to plot {chart_metric} for this cohort.")
        else: st.info("This cohort is empty. Add some tickers above.")

# ===========================
# TAB 6: MACROECONOMICS (FRED)
# ===========================
with tab6:
    st.header("Macroeconomic Indicators (FRED)")
    st.markdown("Analyze top-down systemic risks using direct data feeds from the Federal Reserve.")

    if not fred:
        st.error("🚨 **System Lock:** FRED API key not detected.")
        st.info("To unlock this module, add `FRED_API_KEY=your_key` to your `.env` file and restart the application.")
    else:
        st.subheader("Data Selection")
        standard_indicators = {
            "Effective Federal Funds Rate (Interest Rates)": "FEDFUNDS",
            "Consumer Price Index (Inflation)": "CPIAUCSL",
            "Real Gross Domestic Product (GDP)": "GDPC1",
            "M2 Money Supply (Liquidity)": "M2SL",
            "Unemployment Rate": "UNRATE",
            "10-Year Treasury Constant Maturity Rate": "DGS10",
            "ICE BofA US High Yield Spread (Credit Risk)": "BAMLH0A0HYM2", 
            "Federal Reserve Total Assets (System Liquidity)": "WALCL",
            "Custom Series ID...": "CUSTOM"
        }
        
        col_m_sel, col_m_cust = st.columns([2, 1])
        with col_m_sel:
            selected_indicator_name = st.selectbox("Select Institutional Metric", list(standard_indicators.keys()))
            selected_series_id = standard_indicators[selected_indicator_name]
            
        with col_m_cust:
            if selected_series_id == "CUSTOM":
                selected_series_id = st.text_input("Enter FRED Series ID").upper()
            else:
                st.text_input("Active Series ID", value=selected_series_id, disabled=True)

        st.divider()

        if selected_series_id and selected_series_id != "CUSTOM":
            with st.spinner(f"Querying Federal Reserve database for {selected_series_id}..."):
                macro_df = fetch_macro_data(selected_series_id)
                if macro_df is not None and not macro_df.empty:
                    latest_date = macro_df.index[-1].strftime('%B %Y')
                    latest_value = macro_df['Value'].iloc[-1]
                    st.metric(f"Latest Print ({latest_date})", f"{latest_value:,.2f}")
                    
                    fig_macro = px.area(macro_df, x=macro_df.index, y='Value', title=f"{selected_indicator_name} (Last 20 Years)")
                    fig_macro.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
                    fig_macro.update_xaxes(showgrid=False, title_text="")
                    fig_macro.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    fig_macro.update_traces(line_color='#2980b9', fillcolor='rgba(41, 128, 185, 0.2)')
                    st.plotly_chart(fig_macro, use_container_width=True)
                else:
                    st.warning(f"Could not retrieve data for Series ID: {selected_series_id}. Please verify the ID is correct.")

# ===========================
# TAB 7: THE LIBRARY (RAG PIPELINE)
# ===========================
with tab7:
    st.header("The Library (Local RAG Database)")
    st.markdown("Inject financial PDFs, Annual 10-Ks, and Quarterly 8-K Earnings data directly into Noodle Bot's permanent memory.")

    # --- INGESTION MODULE ---
    col_pdf, col_sec = st.columns(2)
    
    with col_pdf:
        with st.expander("📚 Upload Local PDF", expanded=False):
            uploaded_file = st.file_uploader("Upload Financial Document", type="pdf")
            # Category picker — only the three "real" ingestable categories here.
            _upload_cat_options = ["textbook", "market_report", "sec_filing"]
            upload_category = st.selectbox(
                "Categorize this PDF",
                options=_upload_cat_options,
                format_func=lambda k: _CATEGORIES[k],
                key="upload_pdf_category",
                help="The Oracle can filter retrieval by category when you ask a question.",
            )
            upload_topics = st.multiselect(
                "Topics (multi-select)",
                options=_TOPICS,
                default=[],
                format_func=lambda k: _TOPIC_LABELS[k],
                key="upload_pdf_topics",
                help="A document can carry many topics — e.g. Damodaran = valuation + corporate_finance.",
            )
            if uploaded_file is not None:
                if st.button("Process PDF", type="primary", use_container_width=True):
                    safe_name = os.path.basename(uploaded_file.name)
                    file_path = os.path.join(UPLOAD_DIR, safe_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    doc_id = f"pdf::{safe_name}"
                    if _already_ingested(doc_id):
                        st.info(f"'{safe_name}' is already in the library — skipping re-embed.")
                    else:
                        with st.spinner("Extracting text and chunking document..."):
                            loader = PyMuPDFLoader(file_path)
                            pages = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            document_chunks = text_splitter.split_documents(pages)

                        with st.spinner(f"Translating {len(document_chunks)} chunks to vector coordinates..."):
                            try:
                                n = _ingest_chunks(
                                    document_chunks,
                                    doc_id,
                                    f"PDF: {safe_name}",
                                    category=upload_category,
                                    topics=upload_topics,
                                )
                                _t_str = ", ".join(upload_topics) if upload_topics else "no topics"
                                st.success(
                                    f"✅ Injected '{safe_name}' ({n} chunks) as "
                                    f"{_CATEGORIES[upload_category]} — {_t_str}."
                                )
                            except Exception as e:
                                st.error(f"Failed to embed document. Error: {e}")

    with col_sec:
        with st.expander("🏛️ Rip SEC Filings (10-K / 8-K)", expanded=False):
            sec_ticker = sanitize_ticker(st.text_input("Enter Ticker (e.g., TSLA)").upper())
            sec_form_type = st.radio("Select Document Type", ["10-K (Annual Report)", "8-K (Latest Earnings/Material Events)"])
            
            if st.button("Fetch & Inject SEC Data", type="primary", use_container_width=True) and sec_ticker:
                target_form = "10-K" if "10-K" in sec_form_type else "8-K"
                
                with st.spinner(f"Bypassing SEC EDGAR firewall to locate {sec_ticker} {target_form}..."):
                    raw_text, source_url = fetch_sec_filing(sec_ticker, form_type=target_form)
                    
                if raw_text is None:
                    st.error(source_url)
                else:
                    doc_id = f"sec::{sec_ticker}::{target_form}"
                    if _already_ingested(doc_id):
                        st.info(f"{sec_ticker}'s {target_form} is already in the library — skipping re-embed.")
                    else:
                        file_name = f"{sec_ticker}_{target_form}.txt"
                        file_path = os.path.join(UPLOAD_DIR, file_name)
                        with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
                            f.write(raw_text)

                        with st.spinner(f"{target_form} Downloaded. Chunking text..."):
                            loader = TextLoader(file_path, encoding="utf-8")
                            pages = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                            document_chunks = text_splitter.split_documents(pages)
                            for chunk in document_chunks:
                                chunk.metadata['ticker'] = sec_ticker
                                chunk.metadata['form'] = target_form

                        with st.spinner(f"Translating {len(document_chunks)} chunks to vector coordinates..."):
                            try:
                                n = _ingest_chunks(
                                    document_chunks,
                                    doc_id,
                                    f"SEC EDGAR {target_form}: {sec_ticker}",
                                    category="sec_filing",
                                )
                                st.success(f"✅ Successfully injected {sec_ticker}'s {target_form} ({n} chunks)!")
                            except Exception as e:
                                st.error(f"Failed to embed {target_form}. Error: {e}")

    st.divider()

    # --- LIBRARY MANAGER (re-categorize / delete) ---
    _library_docs = _list_documents()
    _uncat_count = sum(1 for d in _library_docs if d["category"] == "uncategorized")
    _manager_header = "🗂️ Categorize / Manage Existing PDFs"
    if _uncat_count:
        _manager_header += f"  —  ⚠️ {_uncat_count} uncategorized"
    elif _library_docs:
        _manager_header += f"  —  {len(_library_docs)} in library"

    with st.expander(_manager_header, expanded=bool(_uncat_count)):
        st.caption(
            "Assign or change the category for any PDF you've already ingested. "
            "Metadata-only update — no re-embedding."
        )
        if not _library_docs:
            st.info("No documents ingested yet. Upload a PDF or rip an SEC filing above.")
        else:
            # Bulk action — batch-tag every uncategorized doc in one click.
            if _uncat_count:
                _cat_keys_bulk = [k for k in _CATEGORIES.keys() if k != "uncategorized"]
                bcol1, bcol2 = st.columns([3, 2])
                with bcol1:
                    bulk_cat = st.selectbox(
                        f"Bulk-tag all {_uncat_count} uncategorized document(s) as:",
                        options=_cat_keys_bulk,
                        format_func=lambda k: _CATEGORIES[k],
                        key="bulk_cat_choice",
                    )
                with bcol2:
                    st.write("")  # vertical spacer to align with selectbox
                    if st.button(
                        f"Apply to {_uncat_count} document(s)",
                        key="bulk_cat_apply",
                        use_container_width=True,
                    ):
                        for _d in _library_docs:
                            if _d["category"] == "uncategorized":
                                _set_category(_d["doc_id"], bulk_cat)
                        st.success(f"Tagged {_uncat_count} document(s) as {_CATEGORIES[bulk_cat]}.")
                        st.rerun()
                st.divider()

            _cat_keys = list(_CATEGORIES.keys())
            for d in _library_docs:
                c1, c2, c3 = st.columns([5, 3, 1])
                with c1:
                    _badge = "⚠️ " if d["category"] == "uncategorized" else ""
                    st.markdown(f"{_badge}**{d['source']}**")
                    _cur_topics = d.get("topics", [])
                    _topics_str = (
                        ", ".join(_cur_topics) if _cur_topics else "no topics"
                    )
                    st.caption(
                        f"`{d['doc_id']}` · {d['chunks']} chunks · "
                        f"temporal: {d.get('temporal_validity', 'unknown')} · "
                        f"topics: {_topics_str}"
                    )
                with c2:
                    current = d["category"] if d["category"] in _cat_keys else "uncategorized"
                    new_cat = st.selectbox(
                        "Category",
                        options=_cat_keys,
                        index=_cat_keys.index(current),
                        format_func=lambda k: _CATEGORIES[k],
                        key=f"cat_{d['doc_id']}",
                        label_visibility="collapsed",
                    )
                    new_topics = st.multiselect(
                        "Topics",
                        options=_TOPICS,
                        default=[t for t in _cur_topics if t in _TOPICS],
                        format_func=lambda k: _TOPIC_LABELS[k],
                        key=f"top_{d['doc_id']}",
                        label_visibility="collapsed",
                        placeholder="Topics (optional, multi-select)",
                    )
                    _cat_changed = new_cat != current
                    _tops_changed = set(new_topics) != set(_cur_topics)
                    if _cat_changed or _tops_changed:
                        if st.button("Save", key=f"save_{d['doc_id']}"):
                            if _cat_changed:
                                _set_category(d["doc_id"], new_cat)
                            if _tops_changed:
                                _set_topics(d["doc_id"], new_topics)
                            st.success("Saved.")
                            st.rerun()
                with c3:
                    if st.button("🗑️", key=f"del_{d['doc_id']}", help="Remove from library"):
                        _delete_document(d["doc_id"])
                        st.rerun()

    st.divider()

    # --- INFERENCE MODULE (THE ORACLE) ---
    st.subheader("💬 Ask The Oracle")
    st.markdown("Query your uploaded documents. Noodle Bot will synthesize an answer based on your database, current macro conditions, live news, real-time market data, and forward-looking Wall Street consensus.")

    # 1. Initialize session state to prevent the "disappearing expander" bug
    if 'oracle_answer' not in st.session_state:
        st.session_state['oracle_answer'] = None
        st.session_state['oracle_sources'] = None

    # 2. Input Fields
    user_query = st.text_area("What would you like to know about your documents?", placeholder="e.g., Does management's 10-K outlook align with Wall Street's growth estimates?")
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        context_ticker = sanitize_ticker(st.text_input("Target Ticker (Injects News, Price & Consensus)").upper())
    with col_q2:
        peer_group_options = ["None"] + list(peer_groups.keys())
        context_group = st.selectbox("Inject Peer Group Matrix", peer_group_options)

    # Restrict retrieval to specific document categories (multi-select).
    _filter_cat_keys = list(_CATEGORIES.keys())

    # Question router — auto-suggest filters from the question text.
    _router_col_a, _router_col_b = st.columns([1, 3])
    with _router_col_a:
        if st.button("🤖 Auto-route from question", use_container_width=True,
                     help="Ask Ollama to suggest which categories/topics to retrieve from."):
            if not user_query.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Routing question..."):
                    suggestion = _route_query(user_query)
                st.session_state["oracle_router_suggestion"] = suggestion
    with _router_col_b:
        _sug = st.session_state.get("oracle_router_suggestion")
        if _sug:
            _sc = ", ".join(_sug.get("categories") or []) or "(all)"
            _st_ = ", ".join(_sug.get("topics") or []) or "(none)"
            st.caption(
                f"Router suggests — categories: **{_sc}** · topics: **{_st_}** · "
                f"domain: `{_sug.get('domain','general')}`"
            )
            if _sug.get("rationale"):
                st.caption(f"_{_sug['rationale']}_")

    _default_cats = (
        (st.session_state.get("oracle_router_suggestion") or {}).get("categories")
        or _filter_cat_keys
    )
    _default_tops = (
        (st.session_state.get("oracle_router_suggestion") or {}).get("topics") or []
    )

    selected_categories = st.multiselect(
        "Retrieve only from categories",
        options=_filter_cat_keys,
        default=_default_cats,
        format_func=lambda k: _CATEGORIES[k],
        key="oracle_cat_filter",
        help="Narrow the Oracle to textbooks, market reports, or SEC filings only.",
    )
    selected_topics = st.multiselect(
        "Retrieve only from topics (OR — any selected topic matches)",
        options=_TOPICS,
        default=_default_tops,
        format_func=lambda k: _TOPIC_LABELS[k],
        key="oracle_topic_filter",
        help="Empty = no topic restriction. Topics are ORed together.",
    )

    _opt_col1, _opt_col2, _opt_col3 = st.columns(3)
    with _opt_col1:
        use_mmr = st.checkbox(
            "MMR diversification", value=True,
            help="Max Marginal Relevance — reduces near-duplicate chunks in retrieval.",
        )
    with _opt_col2:
        use_multiquery = st.checkbox(
            "Multi-query decomposition", value=True,
            help="Break compound questions into sub-queries, retrieve per sub-query, union.",
        )
    with _opt_col3:
        use_citations = st.checkbox(
            "Citation grounding", value=True,
            help="Force the Oracle to cite [chunk_N] and verify each cited chunk exists.",
        )

    # 3. The Manual Trigger Button
    trigger_oracle = st.button("🔮 Consult The Oracle", type="primary", use_container_width=True)

    # 4. Execution Logic (Only runs when button is physically clicked)
    if trigger_oracle:
        if not user_query:
            st.warning("Please enter a question for the Oracle.")
        elif not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            st.warning("Your library is empty. Please upload a PDF or rip an SEC filing first.")
        else:
            with st.spinner("Initializing Omni-Context Engine (Macro, Market, Peer, and Wall Street Consensus)..."):
                try:
                    macro_injection = ""
                    if fred:
                        try:
                            fed_df = fetch_macro_data("FEDFUNDS")
                            hy_df = fetch_macro_data("BAMLH0A0HYM2")
                            rate_val = f"{fed_df['Value'].iloc[-1]:.2f}%" if (fed_df is not None and not fed_df.empty) else "Unknown"
                            hy_val = f"{hy_df['Value'].iloc[-1]:.2f}%" if (hy_df is not None and not hy_df.empty) else "Unknown"
                            macro_injection = f"\nLIVE MACRO & CREDIT ENVIRONMENT:\n- Current Federal Funds Rate: {rate_val}\n- High Yield Credit Spread (Corporate Stress): {hy_val}\n"
                        except:
                            pass 

                    news_injection = ""
                    market_injection = ""
                    forward_injection = ""
                    if context_ticker:
                        try:
                            live_p_data = fetch_live_prices([context_ticker])
                            p_info = live_p_data.get(context_ticker, {})
                            curr_price = p_info.get('price', 'N/A')
                            day_change = p_info.get('change', 'N/A')
                            
                            hist, info = fetch_stock_details(context_ticker, "1M")
                            mkt_cap = format_large_number(info.get('marketCap'))
                            pe = info.get('trailingPE', 'N/A')
                            fwd_pe = info.get('forwardPE', 'N/A')
                            
                            target_price = info.get('targetMeanPrice', 'N/A')
                            rec = info.get('recommendationKey', 'N/A').upper()
                            rev_growth = info.get('revenueGrowth', 0)
                            earn_growth = info.get('earningsGrowth', 0)
                            
                            rev_str = f"{rev_growth * 100:.1f}%" if rev_growth else "N/A"
                            earn_str = f"{earn_growth * 100:.1f}%" if earn_growth else "N/A"
                            
                            trend_str = "N/A"
                            if not hist.empty and len(hist) > 20:
                                month_ago = hist['Close'].iloc[-21]
                                latest = hist['Close'].iloc[-1]
                                trend_pct = ((latest - month_ago) / month_ago) * 100
                                trend_str = f"{trend_pct:.2f}%"

                            market_injection = f"""
LIVE MARKET VALUATION FOR {context_ticker}:
- Current Price: ${curr_price} (Day Change: {day_change}%)
- 1-Month Trend: {trend_str}
- Market Cap: {mkt_cap}
- P/E Ratio (Trailing): {pe} | P/E Ratio (Forward): {fwd_pe}
"""
                            forward_injection = f"""
WALL STREET CONSENSUS & FORWARD EXPECTATIONS FOR {context_ticker}:
- Mean Target Price: ${target_price}
- Analyst Consensus: {rec}
- Est. Forward Revenue Growth: {rev_str}
- Est. Forward Earnings Growth: {earn_str}
"""
                        except Exception:
                            market_injection = f"\nLIVE MARKET VALUATION FOR {context_ticker}: Temporarily Unavailable.\n"

                        try:
                            recent_news = fetch_financial_news(context_ticker)
                            if recent_news:
                                news_injection = f"\nLIVE NEWS ENVIRONMENT FOR {context_ticker}:\n"
                                for article in recent_news:
                                    news_injection += f"- {article['title']} ({article['time']})\n"
                            else:
                                news_injection = f"\nLIVE NEWS ENVIRONMENT FOR {context_ticker}:\n- No major institutional headlines in the last 24 hours.\n"
                        except:
                            pass

                    peer_injection = ""
                    if context_group and context_group != "None":
                        try:
                            g_tickers = peer_groups[context_group]
                            if g_tickers:
                                p_df = fetch_peer_metrics(g_tickers)
                                if not p_df.empty:
                                    peer_injection = f"\nLIVE PEER GROUP VALUATION MATRIX ({context_group}):\n"
                                    for _, r in p_df.iterrows():
                                        peer_injection += f"- {r['Ticker']}: Price: ${r['Price']} | Trailing P/E: {r['P/E (Trailing)']} | EV/EBITDA: {r['EV/EBITDA']} | ROE: {r['ROE (%)']}% | D/E: {r['Debt/Equity']}\n"
                        except Exception:
                            peer_injection = f"\nLIVE PEER GROUP VALUATION MATRIX ({context_group}): Temporarily Unavailable.\n"
                
                    # --- Retrieval pipeline: (optional multi-query) + filtered MMR ---
                    # Resolve effective category filter — if the user kept the full set
                    # selected, don't filter by category at all.
                    _effective_cats = (
                        selected_categories
                        if (selected_categories and
                            len(selected_categories) < len(_filter_cat_keys))
                        else None
                    )
                    _effective_topics = selected_topics or None

                    if use_multiquery:
                        with st.spinner("Decomposing question into sub-queries..."):
                            sub_queries = _decompose_query(user_query, max_sub=3)
                        if len(sub_queries) > 1:
                            st.caption("🧩 Sub-queries used: " +
                                       " · ".join(f"`{q}`" for q in sub_queries[1:]))
                        retrieved_docs = _retrieve_multi(
                            sub_queries,
                            k_per_query=4,
                            k_total=8,
                            categories=_effective_cats,
                            topics_any=_effective_topics,
                            ticker=context_ticker or None,
                            use_mmr=use_mmr,
                        )
                    else:
                        from rag import retrieve as _retrieve
                        retrieved_docs = _retrieve(
                            user_query,
                            k=6,
                            categories=_effective_cats,
                            topics_any=_effective_topics,
                            ticker=context_ticker or None,
                            use_mmr=use_mmr,
                        )

                    if not retrieved_docs:
                        st.info("No relevant information found in your documents.")
                    else:
                        if use_citations:
                            context = _format_chunks_for_citation(retrieved_docs)
                            citation_rule = (
                                "\n4. Citation Discipline: After every non-trivial factual claim, "
                                "add a citation of the form [chunk_N] referring to the numbered "
                                "DOCUMENT CONTEXT blocks below. Do not cite chunks that do not "
                                "exist. If a claim is not supported by any chunk, say so "
                                "explicitly — do not fabricate."
                            )
                        else:
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            citation_rule = ""

                        rag_prompt = f"""You are analyzing a user query based strictly on the provided DOCUMENT CONTEXT. You MUST factor in the LIVE MACRO & CREDIT ENVIRONMENT, MARKET VALUATION, WALL STREET CONSENSUS, PEER GROUP MATRIX, and LIVE NEWS provided below to form a sophisticated, forward-looking thesis designed to maximize profit and identify market mispricings.

                        {macro_injection}
                        {market_injection}
                        {forward_injection}
                        {peer_injection}
                        {news_injection}

                        DOCUMENT CONTEXT:
                        {context}

                        QUESTION:
                        {user_query}
                        """

                        oracle_persona = f"""You are 'The True Oracle', an elite financial AI running on a Mac M2. You must strictly obey the following rules:
1. The Logic-First Filter: Before answering, perform a Logical Audit defining the Domain of Discourse and isolating atomic propositions. Explicitly list hidden premises (enthymemes).
2. Probabilistic Calibration: For empirical claims, reject binary True/False. Treat new info as Evidence updating a Prior Belief (Bayesian update). Provide estimated confidence intervals (e.g., Confidence: High, p > 0.8).
3. Output Structuring: Define ambiguous terms immediately; use numbered steps for reasoning chains; halt and flag logical contradictions.{citation_rule}"""

                        st.success("### Oracle's Synthesis")

                        def _stream_oracle():
                            for chunk in ollama.chat(
                                model='llama3.2',
                                messages=[
                                    {'role': 'system', 'content': oracle_persona},
                                    {'role': 'user', 'content': rag_prompt},
                                ],
                                stream=True,
                            ):
                                yield chunk['message']['content']

                        full_answer = st.write_stream(_stream_oracle())
                        st.session_state['oracle_answer'] = full_answer
                        st.session_state['oracle_sources'] = retrieved_docs
                        st.session_state['oracle_used_citations'] = bool(use_citations)

                        # Citation verification — surface missing/invalid citations.
                        if use_citations:
                            v = _verify_citations(full_answer, retrieved_docs)
                            st.session_state['oracle_citation_report'] = v
                            if v["unknown"]:
                                st.warning(
                                    "⚠️ Answer cites chunks that don't exist: "
                                    + ", ".join(f"[chunk_{i}]" for i in v["unknown"])
                                )
                            if not v["cited"]:
                                st.warning(
                                    "⚠️ No valid [chunk_N] citations detected in the answer. "
                                    "Claims may be ungrounded."
                                )
                            else:
                                st.success(
                                    f"✅ {len(v['cited'])} of {len(retrieved_docs)} retrieved "
                                    f"chunks were cited: " +
                                    ", ".join(f"[chunk_{i}]" for i in v["cited"])
                                )

                except Exception as e:
                    st.error(f"Error querying the database: {e}")

    # 5. Render the result from the cache (Allows you to click expanders safely)
    # Skip on the streaming turn — we already rendered the answer live.
    if st.session_state['oracle_answer'] and not trigger_oracle:
        st.success("### Oracle's Synthesis")
        st.write(st.session_state['oracle_answer'])

    if st.session_state['oracle_answer']:
        with st.expander("🔍 View Source Documents Used"):
            _citation_report = st.session_state.get('oracle_citation_report') or {}
            _cited_set = set(_citation_report.get('cited') or [])
            for i, doc in enumerate(st.session_state['oracle_sources']):
                chunk_num = i + 1
                source_name = doc.metadata.get('source', 'Unknown Document')
                clean_source = os.path.basename(source_name)
                cat_key = doc.metadata.get('category', 'uncategorized')
                cat_label = _CATEGORIES.get(cat_key, cat_key)
                temporal = doc.metadata.get('temporal_validity', 'unknown')
                doc_topics = [t for t in _TOPICS
                              if doc.metadata.get(f"topic_{t}") is True]
                topics_str = ", ".join(doc_topics) if doc_topics else "—"
                cite_badge = (
                    " ✅ cited" if chunk_num in _cited_set
                    else (" ⚪ not cited" if st.session_state.get('oracle_used_citations')
                          else "")
                )
                st.markdown(
                    f"**[chunk_{chunk_num}]** `{clean_source}` — *{cat_label}* "
                    f"(`{temporal}`) · topics: {topics_str}{cite_badge}"
                )
                st.caption(doc.page_content)
                st.write("---")

# ===========================
# TAB ANALYST: AGENTIC "ANALYZE TICKER" WORKFLOW
# ===========================
with tab_analyst:
    st.header("🧠 Analyze Ticker — Agentic Workflow")
    st.markdown(
        "A full buy-side memo assembled from **market data + DCF + peers + "
        "macro + SEC 10-K + recent news + your reference library**, then "
        "synthesized by Llama 3.2 into a structured Bull / Bear / Confidence "
        "briefing. Runs data gathering concurrently — expect ~30-60 seconds "
        "on a cold cache, faster on repeat."
    )

    a_col1, a_col2 = st.columns([2, 3])
    with a_col1:
        analyst_ticker = sanitize_ticker(
            st.text_input(
                "Target Ticker",
                value="AAPL",
                key="analyst_ticker",
            ).upper()
        )
    with a_col2:
        peer_options = ["(none)"] + list(peer_groups.keys())
        analyst_peer_group = st.selectbox(
            "Inject peer cohort (optional)",
            peer_options,
            key="analyst_peer_group",
        )

    ac1, ac2 = st.columns([1, 1])
    with ac1:
        include_sec_10k = st.checkbox(
            "Include SEC 10-K excerpt",
            value=True,
            help="Pulls the most recent annual filing. Adds ~10-15s on cold "
                 "cache. Turn off if you only want fast market-layer analysis.",
        )
    with ac2:
        stream_memo = st.checkbox(
            "Stream memo as it generates",
            value=True,
            help="Off = faster for short memos; On = see the Oracle think.",
        )

    run_analyst = st.button(
        "🚀 Run Full Analysis",
        type="primary",
        use_container_width=True,
        key="analyst_run",
    )

    # Session-state carryover so the result survives re-renders.
    if "analyst_result" not in st.session_state:
        st.session_state["analyst_result"] = None

    if run_analyst:
        if not analyst_ticker:
            st.warning("Enter a ticker to analyze.")
        else:
            # ---- Live progress panel wired to agent.py via callback ----
            _step_labels = {
                "market":    "Live market snapshot (yfinance)",
                "consensus": "Wall Street consensus",
                "dcf":       "Quick DCF (default assumptions)",
                "news":      "Recent news (Yahoo RSS)",
                "peers":     "Peer metrics matrix",
                "macro":     "Macro & credit (FRED)",
                "sec":       "SEC 10-K excerpt",
                "rag":       "Reference library retrieval",
                "synthesis": "LLM synthesis",
            }
            _status_icons = {
                "start": "⏳", "done": "✅", "skip": "⚪", "error": "❌",
            }
            _progress_state: dict[str, tuple[str, str]] = {}
            progress_box = st.empty()

            def _render_progress():
                lines = []
                for key, label in _step_labels.items():
                    status, detail = _progress_state.get(key, ("pending", ""))
                    icon = _status_icons.get(status, "⏱️")
                    suffix = f"  _{detail}_" if detail else ""
                    lines.append(f"{icon}  **{label}**{suffix}")
                progress_box.markdown("\n\n".join(lines))

            def _cb(step: str, status: str, detail: str = ""):
                _progress_state[step] = (status, detail)
                _render_progress()

            _render_progress()

            _peer_tickers = (
                peer_groups.get(analyst_peer_group)
                if analyst_peer_group and analyst_peer_group != "(none)"
                else None
            )

            try:
                envelope = _run_analyze_ticker(
                    analyst_ticker,
                    peer_group_tickers=_peer_tickers,
                    include_sec=include_sec_10k,
                    progress_callback=_cb,
                    stream=stream_memo,
                )
            except Exception as e:
                st.error(f"Analysis pipeline failed: {e}")
                envelope = None

            if envelope and not envelope.get("error"):
                st.divider()
                st.subheader(f"📋 Investment Memo — {analyst_ticker}")

                if stream_memo and envelope.get("memo_stream"):
                    memo_text = st.write_stream(envelope["memo_stream"])
                    envelope = envelope["finalize"](memo_text) if envelope.get("finalize") else envelope
                else:
                    st.markdown(envelope.get("memo") or "_(no memo)_")

                # Confidence banner
                conf = (envelope.get("confidence") or {})
                stance = conf.get("stance")
                conf_level = conf.get("confidence")
                if stance and conf_level:
                    banner_fn = {
                        "Bullish": st.success,
                        "Bearish": st.error,
                        "Neutral": st.info,
                    }.get(stance, st.info)
                    banner_fn(f"**Final Stance:** {stance}  ·  **Confidence:** {conf_level}")

                st.session_state["analyst_result"] = envelope
            elif envelope and envelope.get("error"):
                st.error(envelope["error"])

    # ---- Persistent result view (survives reruns) ----
    _cached = st.session_state.get("analyst_result")
    if _cached and not run_analyst:
        st.divider()
        st.subheader(f"📋 Last Memo — {_cached.get('ticker', '?')}")
        st.markdown(_cached.get("memo") or "_(no memo)_")
        conf = (_cached.get("confidence") or {})
        if conf.get("stance") and conf.get("confidence"):
            banner_fn = {
                "Bullish": st.success,
                "Bearish": st.error,
                "Neutral": st.info,
            }.get(conf["stance"], st.info)
            banner_fn(
                f"**Final Stance:** {conf['stance']}  ·  "
                f"**Confidence:** {conf['confidence']}"
            )

    if _cached:
        with st.expander("🔎 Raw Context Blocks (what the LLM saw)"):
            blocks = _cached.get("context_blocks") or {}
            for _name, _text in blocks.items():
                if not _text:
                    continue
                st.markdown(f"**{_name.upper()}**")
                st.code(_text, language="text")

        _rag_docs = _cached.get("rag_docs") or []
        if _rag_docs:
            with st.expander(f"📚 Reference Library Chunks Used ({len(_rag_docs)})"):
                for i, d in enumerate(_rag_docs, 1):
                    src = os.path.basename(d.metadata.get("source", "?"))
                    cat = d.metadata.get("category", "?")
                    st.markdown(f"**[chunk_{i}]** `{src}` — *{cat}*")
                    st.caption(d.page_content)
                    st.write("---")