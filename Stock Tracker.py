import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots # NEW IMPORT

# --- CONFIGURATION & STORAGE ---
DATA_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {"AAPL": 250.0, "GOOG": 250.0, "TSLA": 0.0, "CLF": 0.0}

def save_portfolio(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

# --- DATA FETCHING (CACHED) ---
@st.cache_data(ttl=60)
def fetch_live_prices(tickers):
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            prices[ticker] = round(stock.fast_info['last_price'], 2)
        except Exception:
            prices[ticker] = None
    return prices

@st.cache_data(ttl=3600) # Cached for 1 hour to keep app fast
def fetch_stock_details(ticker, period):
    """Fetches historical chart data and fundamental info."""
    stock = yf.Ticker(ticker)
    
    # Map UI timeframes to yfinance timeframes
    period_map = {
        "1D": "1d", "5D": "5d", "1M": "1mo", "6M": "6mo",
        "YTD": "ytd", "1Y": "1y", "5Y": "5y", "Max": "max"
    }
    yf_period = period_map.get(period, "1mo")
    
    # Fetch history
    try:
        # Use 1-minute intervals for 1-day charts, otherwise daily
        interval = "1m" if yf_period == "1d" else "1d"
        hist = stock.history(period=yf_period, interval=interval)
    except:
        hist = pd.DataFrame()
        
    # Fetch deep info (P/E, Market Cap, etc.)
    try:
        info = stock.info
    except:
        info = {}
        
    return hist, info

# --- STYLING LOGIC ---
def highlight_buy_zone(row):
    live = row['Live Price (from API)']
    target = row['Target Price (Self-set)']
    if pd.isna(live) or pd.isna(target) or target == 0.0:
        return [''] * len(row)
    if live <= target:
        return ['background-color: rgba(39, 174, 96, 0.3)'] * len(row)
    return [''] * len(row)

# --- UI CONSTRUCTION ---
st.set_page_config(page_title="Stock Tracker", layout="centered")
st.title("Stock Price & Target Tracker")

portfolio = load_portfolio()
tickers = list(portfolio.keys())

with st.spinner("Fetching live market data..."):
    live_prices = fetch_live_prices(tickers)

df_data = {
    "Ticker": tickers,
    "Live Price (from API)": [live_prices.get(t) for t in tickers],
    "Target Price (Self-set)": [portfolio.get(t) for t in tickers]
}
df = pd.DataFrame(df_data)

# --- SECTION 1: MARKET WATCH TABLE ---
st.subheader("Market Watch")
styled_df = df.style.apply(highlight_buy_zone, axis=1)

edited_df = st.data_editor(
    styled_df,
    disabled=["Ticker", "Live Price (from API)"],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Live Price (from API)": st.column_config.NumberColumn(format="$%.2f"),
        "Target Price (Self-set)": st.column_config.NumberColumn(format="$%.2f", step=1.0)
    }
)

if not edited_df.equals(df):
    new_portfolio = dict(zip(edited_df["Ticker"], edited_df["Target Price (Self-set)"]))
    save_portfolio(new_portfolio)
    st.success("Target prices saved!")

st.divider()

# --- SECTION 2: ADD NEW STOCK ---
st.subheader("Add a New Stock")
col_input, col_btn = st.columns([3, 1])
with col_input:
    new_ticker = st.text_input("Enter Ticker Symbol (e.g., MSFT)").upper()
with col_btn:
    st.write("") 
    st.write("") 
    if st.button("Add Stock", use_container_width=True):
        if new_ticker and new_ticker not in portfolio:
            portfolio[new_ticker] = 0.0 
            save_portfolio(portfolio)
            st.rerun() 
        elif new_ticker in portfolio:
            st.warning("Ticker already tracking.")

# --- SECTION 3: DEEP DIVE & CHARTING ---
st.subheader("Deep Dive Analysis")

col_asset, col_refresh = st.columns([3, 1])
with col_asset:
    selected_ticker = st.selectbox("Select Asset", tickers)
with col_refresh:
    st.write("") 
    st.write("") 
    if st.button("🔄 Force Refresh", use_container_width=True):
        fetch_stock_details.clear() 
        fetch_live_prices.clear()   
        st.rerun() 

timeframes = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
selected_period = st.radio("Timeframe", timeframes, index=2, horizontal=True)

# Toggles for Technical Analysis (Now using 3 columns)
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
            
            # --- COLOR LOGIC ---
            start_price = hist_data['Close'].iloc[0]
            end_price = hist_data['Close'].iloc[-1]
            chart_color = '#28a745' if end_price >= start_price else '#dc3545'
            
            # --- DYNAMIC LAYOUT ROUTING ---
            # Calculate how many extra panels we need
            active_subplots = sum([show_rsi, show_macd])
            total_rows = 1 + active_subplots
            
            if total_rows == 3:
                row_heights = [0.6, 0.2, 0.2]
                rsi_row = 2
                macd_row = 3
            elif total_rows == 2:
                row_heights = [0.7, 0.3]
                rsi_row = 2 if show_rsi else None
                macd_row = 2 if show_macd else None
            else:
                row_heights = [1.0]
                rsi_row = None
                macd_row = None
                
            # Initialize Figure architecture
            if total_rows > 1:
                fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=row_heights)
            else:
                fig = go.Figure()
            
            # --- PRIMARY PRICE LINE ---
            fig.add_trace(go.Scatter(
                x=hist_data.index, 
                y=hist_data['Close'], 
                mode='lines',
                name=f"{selected_ticker} Price",
                line=dict(color=chart_color, width=2) 
            ), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
            
            # --- SMA LOGIC ---
            if show_sma:
                sma_50 = hist_data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=sma_50,
                    mode='lines',
                    name="50-Period SMA",
                    line=dict(color='#3498db', width=1.5, dash='dot')
                ), row=1 if total_rows > 1 else None, col=1 if total_rows > 1 else None)
                
            # --- RSI LOGIC ---
            if show_rsi:
                delta = hist_data['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -1 * delta.clip(upper=0)
                ema_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                ema_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                rs = ema_gain / ema_loss
                rsi = 100 - (100 / (1 + rs))
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=rsi, mode='lines',
                    name="RSI (14)", line=dict(color='#9b59b6', width=1.5)
                ), row=rsi_row, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(220, 53, 69, 0.5)", row=rsi_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(40, 167, 69, 0.5)", row=rsi_row, col=1)
                fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
                
            # --- MACD LOGIC ---
            if show_macd:
                # Calculate the EMAs and MACD vector math
                ema_12 = hist_data['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = hist_data['Close'].ewm(span=26, adjust=False).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = macd_line - signal_line
                
                # Assign dynamic colors to the histogram (Green for positive, Red for negative)
                hist_colors = ['#28a745' if val >= 0 else '#dc3545' for val in macd_hist]
                
                fig.add_trace(go.Bar(
                    x=hist_data.index, y=macd_hist, marker_color=hist_colors, name="MACD Histogram"
                ), row=macd_row, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=macd_line, mode='lines',
                    name="MACD Line", line=dict(color='#2980b9', width=1.5)
                ), row=macd_row, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index, y=signal_line, mode='lines',
                    name="Signal Line", line=dict(color='#e67e22', width=1.5)
                ), row=macd_row, col=1)
            
            # --- LAYOUT FORMATTING ---
            # Dynamically increase overall chart height to accommodate active subplots
            dynamic_height = 400 + (150 * active_subplots)
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                showlegend=show_sma or show_rsi or show_macd,
                height=dynamic_height,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Format numbers for readability
            def format_large_number(num):
                if num is None: return "N/A"
                if num >= 1e12: return f"{num/1e12:.2f}T"
                if num >= 1e9: return f"{num/1e9:.2f}B"
                if num >= 1e6: return f"{num/1e6:.2f}M"
                return f"{num:.2f}"

            mkt_cap = format_large_number(stock_info.get('marketCap'))
            pe_ratio = round(stock_info.get('trailingPE', 0), 2) if stock_info.get('trailingPE') else "N/A"
            div_yield = stock_info.get('dividendYield')
            div_yield_str = f"{div_yield * 100:.2f}%" if div_yield else "N/A"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Open", f"${round(hist_data['Open'].iloc[-1], 2) if not hist_data.empty else 'N/A'}")
                st.metric("Low", f"${round(hist_data['Low'].min(), 2) if not hist_data.empty else 'N/A'}")
            with col2:
                st.metric("Mkt Cap", mkt_cap)
                st.metric("52-wk High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
            with col3:
                st.metric("P/E Ratio", pe_ratio)
                st.metric("52-wk Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
            with col4:
                st.metric("Dividend", div_yield_str)
                st.metric("High", f"${round(hist_data['High'].max(), 2) if not hist_data.empty else 'N/A'}")

        else:
            st.warning(f"Could not retrieve historical data for {selected_ticker}.")

st.divider()


            