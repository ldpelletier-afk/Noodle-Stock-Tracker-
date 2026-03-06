import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime
from dotenv import load_dotenv

# --- CONFIGURATION & STORAGE ---
DATA_FILE = "portfolio.json"

st.set_page_config(page_title="The True Oracle", layout="wide")

def sanitize_ticker(ticker):
    """Converts common brokerage ticker formats to Yahoo Finance standards."""
    if not ticker: return ticker
    ticker = ticker.upper().strip()
    if ticker.endswith(":CA"):
        return ticker.replace(":CA", ".TO")
    return ticker

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            
            if data and isinstance(list(data.values())[0], (int, float)):
                new_data = {
                    "portfolios": {"My First Portfolio": {}},
                    "watch_list_targets": data 
                }
                save_data(new_data)
                data = new_data
                
            needs_save = False
            clean_targets = {}
            for t, val in data.get("watch_list_targets", {}).items():
                clean_t = sanitize_ticker(t)
                clean_targets[clean_t] = val
                if clean_t != t: needs_save = True
            data["watch_list_targets"] = clean_targets
            
            for p_name, p_data in data.get("portfolios", {}).items():
                clean_p_data = {}
                for t, t_data in p_data.items():
                    clean_t = sanitize_ticker(t)
                    clean_p_data[clean_t] = t_data
                    if clean_t != t: needs_save = True
                data["portfolios"][p_name] = clean_p_data
                
            if needs_save:
                save_data(data)
                
            return data
        except json.JSONDecodeError:
             return {"portfolios": {}, "watch_list_targets": {}}
    
    return {
        "portfolios": {"My First Portfolio": {}},
        "watch_list_targets": {"AAPL": 250.0, "GOOG": 250.0, "TSLA": 0.0}
    }

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- DATA FETCHING (CACHED) ---
@st.cache_data(ttl=60)
def fetch_live_prices(tickers):
    """Upgraded to fetch both price and daily % change."""
    prices = {}
    for ticker in tickers:
        if not ticker: continue
        if ticker.upper() == "CASH":
            prices[ticker] = {'price': 1.00, 'change': 0.0}
            continue
        try:
            stock = yf.Ticker(ticker)
            price = stock.fast_info.last_price
            prev_close = stock.fast_info.previous_close
            pct_change = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0
            prices[ticker] = {'price': round(price, 2), 'change': round(pct_change, 2)}
        except Exception:
            prices[ticker] = {'price': None, 'change': None}
    return prices

@st.cache_data(ttl=3600) 
def fetch_stock_details(ticker, period):
    stock = yf.Ticker(ticker)
    period_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "Max": "max"}
    yf_period = period_map.get(period, "1mo")
    try:
        interval = "1m" if yf_period == "1d" else "1d"
        hist = stock.history(period=yf_period, interval=interval)
    except:
        hist = pd.DataFrame()
    try:
        info = stock.info
    except:
        info = {}
    return hist, info

@st.cache_data(ttl=86400) 
def fetch_dcf_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    current_price = info.get('currentPrice') or info.get('previousClose')
    if not current_price:
        try: current_price = stock.fast_info.last_price
        except: current_price = None
        
    shares_out = info.get('sharesOutstanding')
    
    try:
        cf = stock.cash_flow
        if 'Free Cash Flow' in cf.index:
            fcf = cf.loc['Free Cash Flow'].iloc[0]
        else:
            op_cf = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0
            capex = cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0
            fcf = op_cf + capex 
    except Exception:
        fcf = None
        
    return fcf, shares_out, current_price

# --- STYLING LOGIC ---
def highlight_buy_zone(row):
    live = row.get('Live Price (from API)')
    target = row.get('Target Price (Self-set)')
    change = row.get('Day Change (%)')
    
    styles = [''] * len(row)
    
    # 1. Target hit -> Entire row solid green with white text for crisp contrast
    if pd.notna(live) and pd.notna(target) and target > 0.0 and live <= target:
        styles = ['background-color: #28a745; color: white;'] * len(row)
        
    # 2. Daily Drop -> Bold Red text for the ticker and the change column
    if pd.notna(change) and change <= -5.0:
        change_idx = list(row.index).index('Day Change (%)')
        ticker_idx = list(row.index).index('Ticker')
        
        # We append to any existing styles (like the green background) so they can stack
        styles[change_idx] += 'color: #ff4b4b; font-weight: bold;'
        styles[ticker_idx] += 'color: #ff4b4b; font-weight: bold;'
        
    return styles

def format_large_number(num):
    if num is None: return "N/A"
    if num >= 1e12: return f"{num/1e12:.2f}T"
    if num >= 1e9: return f"{num/1e9:.2f}B"
    if num >= 1e6: return f"{num/1e6:.2f}M"
    return f"{num:.2f}"

# --- MAIN APP ---
st.title("The True Oracle: Valuation & Tracking")

app_data = load_data()
portfolios = app_data.get("portfolios", {})
watch_list_targets = app_data.get("watch_list_targets", {})

tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Watch", "💼 Asset Tracker", "⚖️ The Valuation Machine", "📰 Market Intelligence"])

# ===========================
# TAB 1: MARKET WATCH 
# ===========================
with tab1:
    tickers = list(watch_list_targets.keys())

    with st.spinner("Fetching live market data..."):
        live_prices = fetch_live_prices(tickers)

    # --- VOLATILITY ALERT SYSTEM ---
    crashing_assets = []
    for t in tickers:
        pct_drop = live_prices.get(t, {}).get('change')
        if pct_drop is not None and pct_drop <= -5.0:
            crashing_assets.append(f"**{t}** ({pct_drop}%)")
    
    if crashing_assets:
        st.error(f"🚨 **Volatility Alert:** The following assets are down 5% or more today: {', '.join(crashing_assets)}")

    # Incorporating the nested dict structure
    df_data = {
        "Ticker": tickers,
        "Live Price (from API)": [live_prices.get(t, {}).get('price') for t in tickers],
        "Day Change (%)": [live_prices.get(t, {}).get('change') for t in tickers],
        "Target Price (Self-set)": [watch_list_targets.get(t) for t in tickers]
    }
    df = pd.DataFrame(df_data)

    st.subheader("Watch List & Price Targets")
    styled_df = df.style.apply(highlight_buy_zone, axis=1)

    edited_df = st.data_editor(
        styled_df,
        disabled=["Ticker", "Live Price (from API)", "Day Change (%)"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Live Price (from API)": st.column_config.NumberColumn(format="$%.2f"),
            "Day Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Target Price (Self-set)": st.column_config.NumberColumn(format="$%.2f", step=1.0)
        }
    )

    if not edited_df.equals(df):
        new_targets = dict(zip(edited_df["Ticker"], edited_df["Target Price (Self-set)"]))
        app_data["watch_list_targets"] = new_targets
        save_data(app_data)
        st.toast("Target prices saved!", icon="✅")

    st.divider()

    col_add, col_del = st.columns(2)
    with col_add:
        st.subheader("Add to Watch List")
        new_ticker = sanitize_ticker(st.text_input("Enter Ticker Symbol (e.g., MSFT)").upper())
        if st.button("Add Stock", use_container_width=True):
            if new_ticker and new_ticker not in watch_list_targets:
                watch_list_targets[new_ticker] = 0.0 
                app_data["watch_list_targets"] = watch_list_targets
                save_data(app_data)
                st.rerun() 
            elif new_ticker in watch_list_targets:
                st.warning("Ticker already on watch list.")
                
    with col_del:
        st.subheader("Remove from Watch List")
        if tickers:
            ticker_to_remove = st.selectbox("Select Asset to Delete", tickers)
            if st.button("Delete Stock", type="primary", use_container_width=True):
                del watch_list_targets[ticker_to_remove]
                app_data["watch_list_targets"] = watch_list_targets
                save_data(app_data)
                st.toast(f"Removed {ticker_to_remove}", icon="🗑️")
                st.rerun()
        else:
            st.info("Watch list is empty.")

    st.divider()

    st.subheader("Deep Dive Analysis")
    col_asset, col_refresh = st.columns([3, 1])
    with col_asset:
        selected_ticker = st.selectbox("Select Asset for Analysis", tickers if tickers else [""])
    with col_refresh:
        st.write("") 
        st.write("") 
        if st.button("🔄 Force Refresh Data", use_container_width=True):
            fetch_stock_details.clear() 
            fetch_live_prices.clear()   
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
                
                row_heights = [1.0]
                rsi_row, macd_row = None, None

                if total_rows == 3:
                    row_heights = [0.6, 0.2, 0.2]
                    rsi_row, macd_row = 2, 3
                elif total_rows == 2:
                    row_heights = [0.7, 0.3]
                    rsi_row = 2 if show_rsi else None
                    macd_row = 2 if show_macd else None

                fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights) if total_rows > 1 else go.Figure()
                
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], mode='lines', name=f"{selected_ticker} Price", line=dict(color=chart_color, width=2)), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
                
                if show_sma:
                    sma_50 = hist_data['Close'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(x=hist_data.index, y=sma_50, mode='lines', name="50-Period SMA", line=dict(color='#3498db', width=1.5, dash='dot')), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
                    
                if show_rsi:
                    delta = hist_data['Close'].diff()
                    gain = delta.clip(lower=0)
                    loss = -1 * delta.clip(upper=0)
                    ema_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    ema_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                    rs = ema_gain / ema_loss
                    rsi = 100 - (100 / (1 + rs))
                    fig.add_trace(go.Scatter(x=hist_data.index, y=rsi, mode='lines', name="RSI (14)", line=dict(color='#9b59b6', width=1.5)), row=rsi_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="rgba(220, 53, 69, 0.5)", row=rsi_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="rgba(40, 167, 69, 0.5)", row=rsi_row, col=1)
                    fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
                    
                if show_macd:
                    ema_12 = hist_data['Close'].ewm(span=12, adjust=False).mean()
                    ema_26 = hist_data['Close'].ewm(span=26, adjust=False).mean()
                    macd_line = ema_12 - ema_26
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

            else:
                st.warning(f"Could not retrieve historical data for {selected_ticker}.")

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
                save_data(app_data)
                st.toast(f"Portfolio '{new_portfolio_name}' created!", icon="🎉")
                st.rerun()
            elif new_portfolio_name in portfolios:
                st.warning("Portfolio already exists.")
    
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
                    total_cost = (current_holdings[ticker]['quantity'] * current_holdings[ticker]['average_cost']) + \
                                 (data['quantity'] * data['average_cost'])
                    current_holdings[ticker]['average_cost'] = total_cost / total_qty
                    current_holdings[ticker]['quantity'] = total_qty
                else:
                    current_holdings[ticker] = data.copy()
    else:
        current_holdings = portfolios[selected_portfolio]

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
                                new_total_qty = old_qty + asset_qty
                                new_avg_cost = ((old_qty * old_cost) + (asset_qty * asset_cost)) / new_total_qty
                                portfolios[selected_portfolio][asset_ticker] = {"quantity": new_total_qty, "average_cost": new_avg_cost}
                            else:
                                portfolios[selected_portfolio][asset_ticker] = {"quantity": asset_qty, "average_cost": asset_cost}
                            app_data["portfolios"] = portfolios
                            save_data(app_data)
                            st.toast(f"Added {asset_ticker}", icon="💰")
                            st.rerun()

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
                                proceeds = sell_qty * sell_price
                                portfolios[selected_portfolio][sell_ticker]["quantity"] -= sell_qty
                                if portfolios[selected_portfolio][sell_ticker]["quantity"] <= 0.0001:
                                    del portfolios[selected_portfolio][sell_ticker]
                                current_cash = portfolios[selected_portfolio].get("CASH", {"quantity": 0.0, "average_cost": 1.0})
                                portfolios[selected_portfolio]["CASH"] = {
                                    "quantity": current_cash["quantity"] + proceeds,
                                    "average_cost": 1.0
                                }
                                app_data["portfolios"] = portfolios
                                save_data(app_data)
                                st.toast(f"Sold {sell_ticker}", icon="🤝")
                                st.rerun()
                else:
                    st.info("No stocks to sell.")

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
                        st.rerun()

        with col_delete:
             with st.expander(f"Delete Asset", expanded=False):
                 assets_to_delete = list(portfolios[selected_portfolio].keys())
                 if assets_to_delete:
                     del_asset = st.selectbox("Select Asset to Delete", assets_to_delete)
                     if st.button("Delete Permanently", type="primary"):
                         del portfolios[selected_portfolio][del_asset]
                         app_data["portfolios"] = portfolios
                         save_data(app_data)
                         st.toast(f"Deleted {del_asset}", icon="🗑️")
                         st.rerun()
                 else:
                     st.info("No assets.")

    if current_holdings:
        holding_tickers = list(current_holdings.keys())
        with st.spinner("Fetching live prices for holdings..."):
            holding_prices = fetch_live_prices(holding_tickers)
        
        holdings_data = []
        total_portfolio_value = 0
        total_portfolio_cost = 0

        for ticker, data in current_holdings.items():
            qty = data['quantity']
            avg_cost = data['average_cost']
            
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
                "Ticker": ticker, "Quantity": qty, "Avg. Cost": avg_cost, 
                "Current Price": current_price, "Market Value": market_value,
                "Profit/Loss ($)": profit_loss, "Profit/Loss (%)": pl_percent / 100 
            })
        
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
            st.dataframe(holdings_df, use_container_width=True, hide_index=True, column_config={
                "Avg. Cost": st.column_config.NumberColumn(format="$%.2f"),
                "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%")
            })
        else:
            st.subheader(f"Manage Holdings: {selected_portfolio}")
            edited_holdings_df = st.data_editor(holdings_df, disabled=["Ticker", "Current Price", "Market Value", "Profit/Loss ($)", "Profit/Loss (%)"], hide_index=True, num_rows="dynamic", use_container_width=True, column_config={
                "Quantity": st.column_config.NumberColumn(step=0.01),
                "Avg. Cost": st.column_config.NumberColumn(format="$%.2f", step=0.01),
                "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%")
            })

            if not edited_holdings_df.equals(holdings_df):
                new_portfolio_data = {}
                for index, row in edited_holdings_df.iterrows():
                    ticker = str(row['Ticker']).upper().strip()
                    if ticker and ticker.lower() not in ['nan', 'none']:
                        qty = float(row['Quantity'])
                        if qty > 0: 
                            new_portfolio_data[ticker] = {"quantity": qty, "average_cost": float(row['Avg. Cost'])}
                portfolios[selected_portfolio] = new_portfolio_data
                app_data["portfolios"] = portfolios
                save_data(app_data)
                st.rerun()

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
                fig_bar = px.bar(holdings_df, x='Ticker', y='Profit/Loss ($)', 
                                 title="Absolute Profit/Loss by Asset",
                                 color='Profit/Loss ($)', 
                                 color_continuous_scale=['#dc3545', '#28a745'])
                fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with tab_tree:
                if not plot_df.empty:
                    fig_tree = px.treemap(plot_df, path=[px.Constant("Portfolio"), 'Ticker'], 
                                          values='Market Value', 
                                          color='Profit/Loss (%)',
                                          color_continuous_scale=['#dc3545', '#28a745'],
                                          color_continuous_midpoint=0)
                    fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                    st.plotly_chart(fig_tree, use_container_width=True)

    else:
        st.info("No assets in this portfolio yet.")

# ===========================
# TAB 3: THE VALUATION MACHINE
# ===========================
with tab3:
    st.header("The Valuation Machine")
    st.markdown("Calculate the true Intrinsic Value of an asset based on its future cash generation capabilities, independent of market sentiment.")
    
    st.subheader("Asset Selection")
    val_ticker = sanitize_ticker(st.text_input("Enter Ticker to Value", value="AAPL").upper())
    
    st.write("---")
    st.markdown("**The Math:**")
    st.markdown(r"$$PV = \sum_{t=1}^{n} \frac{FCF_t}{(1+r)^t} + \frac{TV}{(1+r)^n}$$")
    st.caption("$FCF$ = Free Cash Flow | $r$ = Discount Rate | $TV$ = Terminal Value")

    st.divider()
    
    st.subheader("Philosophical Assumptions")
    c_assump1, c_assump2, c_assump3 = st.columns(3)
    with c_assump1:
        discount_rate = st.slider("Discount Rate (r)", min_value=0.01, max_value=0.20, value=0.10, step=0.01, help="Your required rate of return. Higher risk = higher rate.")
    with c_assump2:
        growth_rate = st.slider("Growth Rate (Years 1-5)", min_value=-0.10, max_value=0.50, value=0.08, step=0.01, help="Expected annual growth of Free Cash Flow.")
    with c_assump3:
        terminal_rate = st.slider("Terminal Growth Rate", min_value=0.01, max_value=0.05, value=0.025, step=0.005, help="Perpetual growth rate after year 5. Should roughly track GDP.")

    if val_ticker and st.button("Calculate Intrinsic Value", use_container_width=True, type="primary"):
        with st.spinner(f"Auditing financial statements for {val_ticker}..."):
            fcf, shares, current_price = fetch_dcf_data(val_ticker)
            
            if fcf is None or shares is None or current_price is None:
                st.error("Insufficient financial data available from Yahoo Finance to complete a strict DCF model for this asset.")
            elif fcf <= 0:
                st.warning(f"{val_ticker} currently generates negative Free Cash Flow. A standard DCF model mathematically collapses under negative trailing cash flow.")
            else:
                projected_cf = []
                current_fcf = fcf
                
                for year in range(1, 6):
                    current_fcf = current_fcf * (1 + growth_rate)
                    pv_cf = current_fcf / ((1 + discount_rate) ** year)
                    projected_cf.append(pv_cf)
                
                sum_pv_cf = sum(projected_cf)
                
                final_year_fcf = current_fcf
                terminal_value = (final_year_fcf * (1 + terminal_rate)) / (discount_rate - terminal_rate)
                pv_terminal_value = terminal_value / ((1 + discount_rate) ** 5)
                
                total_intrinsic_value = sum_pv_cf + pv_terminal_value
                intrinsic_value_per_share = total_intrinsic_value / shares
                
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
                breakdown_data = {
                    "Metric": ["Trailing Free Cash Flow", "Shares Outstanding", "Sum of PV (5 Years)", "PV of Terminal Value", "Total Enterprise Value"],
                    "Value": [format_large_number(fcf), format_large_number(shares), format_large_number(sum_pv_cf), format_large_number(pv_terminal_value), format_large_number(total_intrinsic_value)]
                }
                st.table(pd.DataFrame(breakdown_data))

# ===========================
# TAB 4: MARKET INTELLIGENCE
# ===========================
with tab4:
    st.header("Market Intelligence")
    st.markdown("Live, unfiltered news feeds extracted directly from global financial publishers.")
    
    news_ticker = sanitize_ticker(st.text_input("Target Asset for Reconnaissance", value="AAPL").upper())
        
    st.divider()
    
    if news_ticker:
        with st.spinner(f"Intercepting wire traffic for {news_ticker}..."):
            try:
                news_data = yf.Ticker(news_ticker).news
                
                if not news_data:
                    st.info(f"No recent institutional coverage found for {news_ticker}.")
                else:
                    for article in news_data:
                        title = article.get('title', 'Headline Unavailable')
                        link = article.get('link', '#')
                        publisher = article.get('publisher', 'Unknown Publisher')
                        
                        timestamp = article.get('providerPublishTime')
                        if timestamp:
                            publish_time = datetime.datetime.fromtimestamp(timestamp)
                            time_str = publish_time.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            time_str = "Recent"

                        st.markdown(f"#### [{title}]({link})")
                        st.caption(f"🗞️ **Source:** {publisher} | 🕒 **Published:** {time_str}")
                        st.write("---")
            except Exception as e:
                st.error(f"Failed to establish secure connection to news server: {e}")