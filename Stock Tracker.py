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
import ollama
from fredapi import Fred

# --- RAG & SEC PIPELINE IMPORTS ---
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import requests
from bs4 import BeautifulSoup

from data_store import (
    load_data as _load_data_sqlite,
    save_data as _save_data_sqlite,
    log_transaction,
)

# --- CONFIGURATION & STORAGE ---
DB_DIR = "./chroma_db"
UPLOAD_DIR = "./temp_pdfs"

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="The True Oracle", layout="wide")

# --- SECURE API LOADING ---
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
try:
    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
except Exception:
    fred = None

def sanitize_ticker(ticker):
    if not ticker: return ticker
    ticker = ticker.upper().strip()
    if ticker.endswith(":CA"): return ticker.replace(":CA", ".TO")
    return ticker

def load_data():
    return _load_data_sqlite()

def save_data(data):
    _save_data_sqlite(data)

# --- RAG SINGLETONS ---
@st.cache_resource
def _embedding_engine():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def _vector_db():
    return Chroma(persist_directory=DB_DIR, embedding_function=_embedding_engine())

def _already_ingested(doc_id: str) -> bool:
    try:
        existing = _vector_db().get(where={"doc_id": doc_id}, limit=1)
        return bool(existing and existing.get("ids"))
    except Exception:
        return False

def _ingest_chunks(chunks, doc_id: str, source_label: str) -> int:
    for c in chunks:
        c.metadata["doc_id"] = doc_id
        c.metadata["source"] = source_label
    _vector_db().add_documents(chunks)
    return len(chunks)

# --- SEC EDGAR INTERCEPTOR ---
@st.cache_resource
def _sec_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers.update({
        'User-Agent': os.getenv('SEC_USER_AGENT', 'TheTrueOracle_Quantitative_Engine info@example.com'),
        'Accept-Encoding': 'gzip, deflate',
    })
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount('https://', adapter)
    return s

@st.cache_data(ttl=86400)
def _sec_ticker_map():
    sess = _sec_session()
    r = sess.get("https://www.sec.gov/files/company_tickers.json", timeout=10)
    return r.json()

def fetch_sec_filing(ticker, form_type="10-K"):
    """Fetches either 10-K (Annual) or 8-K (Latest Earnings/Events)"""
    sess = _sec_session()
    try:
        ticker_map = _sec_ticker_map()

        cik = None
        for key, company in ticker_map.items():
            if company['ticker'] == ticker.upper():
                cik = str(company['cik_str']).zfill(10)
                break

        if not cik:
            return None, "Ticker not found in SEC EDGAR database."

        subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        subs_response = sess.get(subs_url, timeout=10)
        filings = subs_response.json()['filings']['recent']

        for i, f_type in enumerate(filings['form']):
            if f_type == form_type:
                accession_number = filings['accessionNumber'][i].replace("-", "")
                primary_document = filings['primaryDocument'][i]
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"

                doc_response = sess.get(doc_url, timeout=15)
                soup = BeautifulSoup(doc_response.content, "html.parser")
                clean_text = soup.get_text(separator='\n', strip=True)
                return clean_text, doc_url
                
        return None, f"No {form_type} filing found for this company in recent history."
    except Exception as e:
        return None, f"SEC API Error: {str(e)}"

# --- DATA FETCHING (CACHED) ---
def fetch_financial_news(ticker):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    articles = []
    try:
        response = requests.get(url, headers=headers, timeout=5)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        for item in root.findall('.//item')[:5]:
            title = item.find('title').text if item.find('title') is not None else 'No Title'
            link = item.find('link').text if item.find('link') is not None else '#'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'Recent'
            articles.append({'title': title, 'link': link, 'time': pub_date, 'publisher': 'Financial Wire'})
    except Exception as e:
        pass
    return articles

@st.cache_data(ttl=60)
def fetch_live_prices(tickers):
    prices = {}
    real_tickers = []
    for ticker in tickers:
        if not ticker: continue
        if ticker.upper() == "CASH":
            prices[ticker] = {'price': 1.00, 'change': 0.0}
            continue
        real_tickers.append(ticker)

    if not real_tickers:
        return prices

    try:
        batch = yf.Tickers(" ".join(real_tickers))
    except Exception:
        batch = None

    for ticker in real_tickers:
        try:
            fast = batch.tickers[ticker].fast_info if batch else yf.Ticker(ticker).fast_info
            price = fast["last_price"]
            prev_close = fast["previous_close"]
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
    except: hist = pd.DataFrame()
    try: info = stock.info
    except: info = {}
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
        if 'Free Cash Flow' in cf.index: fcf = cf.loc['Free Cash Flow'].iloc[0]
        else: fcf = (cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0) + (cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0)
    except: fcf = None
    return fcf, shares_out, current_price

@st.cache_data(ttl=3600)
def fetch_peer_metrics(tickers):
    real_tickers = [t for t in tickers if t]
    if not real_tickers:
        return pd.DataFrame()
    try:
        batch = yf.Tickers(" ".join(real_tickers))
    except Exception:
        batch = None

    data = []
    for t in real_tickers:
        try:
            info = batch.tickers[t].info if batch else yf.Ticker(t).info
            data.append({
                "Ticker": t,
                "Price": info.get("currentPrice") or info.get("previousClose"),
                "P/E (Trailing)": info.get("trailingPE"),
                "P/E (Forward)": info.get("forwardPE"),
                "P/B": info.get("priceToBook"),
                "EV/EBITDA": info.get("enterpriseToEbitda"),
                "ROE (%)": (info.get("returnOnEquity", 0) * 100) if info.get("returnOnEquity") is not None else None,
                "Debt/Equity": info.get("debtToEquity"),
                "Div Yield (%)": (info.get("dividendYield", 0) * 100) if info.get("dividendYield") is not None else None
            })
        except Exception:
            data.append({"Ticker": t}) 
    return pd.DataFrame(data)

@st.cache_data(ttl=86400) 
def fetch_macro_data(series_id):
    if not fred: return None
    try:
        data = fred.get_series(series_id)
        df = pd.DataFrame(data, columns=['Value'])
        df.index.name = 'Date'
        df = df.dropna()
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=20)
        df = df[df.index >= cutoff_date]
        return df
    except Exception as e:
        return None

# --- STYLING LOGIC ---
def highlight_buy_zone(row):
    live = row.get('Live Price (from API)')
    target = row.get('Target Price (Self-set)')
    change = row.get('Day Change (%)')
    styles = [''] * len(row)
    if pd.notna(live) and pd.notna(target) and target > 0.0 and live <= target:
        styles = ['background-color: #28a745; color: white;'] * len(row)
    if pd.notna(change) and change <= -5.0:
        change_idx = list(row.index).index('Day Change (%)')
        ticker_idx = list(row.index).index('Ticker')
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
peer_groups = app_data.get("peer_groups", {})

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Market Watch", "💼 Asset Tracker", "⚖️ Valuation", 
    "📰 Intelligence", "🏢 Peer Matrix", "🏦 Macro", "📚 The Library"
])

# ===========================
# TAB 1: MARKET WATCH 
# ===========================
with tab1:
    tickers = list(watch_list_targets.keys())
    with st.spinner("Fetching live market data..."):
        live_prices = fetch_live_prices(tickers)

    crashing_assets = []
    for t in tickers:
        pct_drop = live_prices.get(t, {}).get('change')
        if pct_drop is not None and pct_drop <= -5.0:
            crashing_assets.append(f"**{t}** ({pct_drop}%)")
    
    if crashing_assets:
        st.error(f"🚨 **Volatility Alert:** The following assets are down 5% or more today: {', '.join(crashing_assets)}")

    df_data = {
        "Ticker": tickers,
        "Live Price (from API)": [live_prices.get(t, {}).get('price') for t in tickers],
        "Day Change (%)": [live_prices.get(t, {}).get('change') for t in tickers],
        "Target Price (Self-set)": [watch_list_targets.get(t) for t in tickers]
    }
    df = pd.DataFrame(df_data)

    st.subheader("Watch List & Price Targets")
    edited_df = st.data_editor(
        df.style.apply(highlight_buy_zone, axis=1),
        disabled=["Ticker", "Live Price (from API)", "Day Change (%)"],
        hide_index=True, use_container_width=True,
        column_config={
            "Live Price (from API)": st.column_config.NumberColumn(format="$%.2f"),
            "Day Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Target Price (Self-set)": st.column_config.NumberColumn(format="$%.2f", step=1.0)
        }
    )

    if not edited_df.equals(df):
        app_data["watch_list_targets"] = dict(zip(edited_df["Ticker"], edited_df["Target Price (Self-set)"]))
        save_data(app_data)
        st.toast("Target prices saved!", icon="✅")

    st.divider()

    col_add, col_del = st.columns(2)
    with col_add:
        st.subheader("Add to Watch List")
        new_ticker = sanitize_ticker(st.text_input("Enter Ticker Symbol (e.g., MSFT)", key="wl_add").upper())
        if st.button("Add Stock", use_container_width=True):
            if new_ticker and new_ticker not in watch_list_targets:
                watch_list_targets[new_ticker] = 0.0 
                app_data["watch_list_targets"] = watch_list_targets
                save_data(app_data); st.rerun() 
            elif new_ticker in watch_list_targets: st.warning("Ticker already on watch list.")
                
    with col_del:
        st.subheader("Remove from Watch List")
        if tickers:
            ticker_to_remove = st.selectbox("Select Asset to Delete", tickers, key="wl_del")
            if st.button("Delete Stock", type="primary", use_container_width=True):
                del watch_list_targets[ticker_to_remove]
                app_data["watch_list_targets"] = watch_list_targets
                save_data(app_data); st.toast(f"Removed {ticker_to_remove}", icon="🗑️"); st.rerun()
        else: st.info("Watch list is empty.")

    st.divider()

    st.subheader("Deep Dive Analysis")
    col_asset, col_refresh = st.columns([3, 1])
    with col_asset: selected_ticker = st.selectbox("Select Asset for Analysis", tickers if tickers else [""])
    with col_refresh:
        st.write(""); st.write("") 
        if st.button("🔄 Force Refresh Data", use_container_width=True):
            fetch_stock_details.clear(); fetch_live_prices.clear(); st.rerun() 

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
                save_data(app_data); st.toast(f"Portfolio '{new_portfolio_name}' created!", icon="🎉"); st.rerun()
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
                            log_transaction(selected_portfolio, asset_ticker, "BUY", asset_qty, asset_cost)
                            st.toast(f"Added {asset_ticker}", icon="💰"); st.rerun()

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
                                if portfolios[selected_portfolio][sell_ticker]["quantity"] <= 0.0001: del portfolios[selected_portfolio][sell_ticker]
                                current_cash = portfolios[selected_portfolio].get("CASH", {"quantity": 0.0, "average_cost": 1.0})
                                portfolios[selected_portfolio]["CASH"] = {"quantity": current_cash["quantity"] + proceeds, "average_cost": 1.0}
                                app_data["portfolios"] = portfolios
                                save_data(app_data)
                                log_transaction(selected_portfolio, sell_ticker, "SELL", sell_qty, sell_price)
                                st.toast(f"Sold {sell_ticker}", icon="🤝"); st.rerun()
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
                        save_data(app_data); st.rerun()

        with col_delete:
             with st.expander(f"Delete Asset", expanded=False):
                 assets_to_delete = list(portfolios[selected_portfolio].keys())
                 if assets_to_delete:
                     del_asset = st.selectbox("Select Asset to Delete", assets_to_delete)
                     if st.button("Delete Permanently", type="primary"):
                         del portfolios[selected_portfolio][del_asset]
                         app_data["portfolios"] = portfolios
                         save_data(app_data); st.toast(f"Deleted {del_asset}", icon="🗑️"); st.rerun()
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
                save_data(app_data); st.rerun()

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
                    save_data(app_data); st.toast(f"Created cohort: {new_group_name}", icon="✅"); st.rerun()

    with col_pg_del:
        if group_names:
            with st.form("delete_group_form"):
                group_to_delete = st.selectbox("Delete Cohort", group_names)
                if st.form_submit_button("Delete Permanently"):
                    del peer_groups[group_to_delete]
                    app_data["peer_groups"] = peer_groups
                    save_data(app_data); st.toast("Cohort deleted.", icon="🗑️"); st.rerun()
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
                        save_data(app_data); st.rerun()
        with col_t_del:
            if group_tickers:
                with st.form("remove_peer_form"):
                    peer_to_remove = st.selectbox("Remove Ticker", group_tickers)
                    if st.form_submit_button("Remove Asset"):
                        peer_groups[selected_group].remove(peer_to_remove)
                        app_data["peer_groups"] = peer_groups
                        save_data(app_data); st.rerun()

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
                                n = _ingest_chunks(document_chunks, doc_id, f"PDF: {safe_name}")
                                st.success(f"✅ Injected '{safe_name}' ({n} chunks) into the database!")
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
                                n = _ingest_chunks(document_chunks, doc_id, f"SEC EDGAR {target_form}: {sec_ticker}")
                                st.success(f"✅ Successfully injected {sec_ticker}'s {target_form} ({n} chunks)!")
                            except Exception as e:
                                st.error(f"Failed to embed {target_form}. Error: {e}")

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
                
                    db = _vector_db()

                    retrieved_docs = []
                    if context_ticker:
                        try:
                            retrieved_docs = db.similarity_search(
                                user_query, k=6, filter={"ticker": context_ticker}
                            )
                        except Exception:
                            retrieved_docs = []
                    if not retrieved_docs:
                        retrieved_docs = db.similarity_search(user_query, k=6)

                    if not retrieved_docs:
                        st.info("No relevant information found in your documents.")
                    else:
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        
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
                        
                        oracle_persona = """You are 'The True Oracle', an elite financial AI running on a Mac M2. You must strictly obey the following rules:
1. The Logic-First Filter: Before answering, perform a Logical Audit defining the Domain of Discourse and isolating atomic propositions. Explicitly list hidden premises (enthymemes).
2. Probabilistic Calibration: For empirical claims, reject binary True/False. Treat new info as Evidence updating a Prior Belief (Bayesian update). Provide estimated confidence intervals (e.g., Confidence: High, p > 0.8).
3. Output Structuring: Define ambiguous terms immediately; use numbered steps for reasoning chains; halt and flag logical contradictions."""

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

                except Exception as e:
                    st.error(f"Error querying the database: {e}")

    # 5. Render the result from the cache (Allows you to click expanders safely)
    # Skip on the streaming turn — we already rendered the answer live.
    if st.session_state['oracle_answer'] and not trigger_oracle:
        st.success("### Oracle's Synthesis")
        st.write(st.session_state['oracle_answer'])

    if st.session_state['oracle_answer']:
        with st.expander("🔍 View Source Documents Used"):
            for i, doc in enumerate(st.session_state['oracle_sources']):
                source_name = doc.metadata.get('source', 'Unknown Document')
                clean_source = os.path.basename(source_name)
                st.markdown(f"**Source {i+1}:** `{clean_source}`")
                st.caption(doc.page_content)
                st.write("---")