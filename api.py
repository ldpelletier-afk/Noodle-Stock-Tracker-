"""External data fetchers: yfinance, SEC EDGAR, FRED, Yahoo news RSS.

All Streamlit cache decorators live here so call sites don't need to know
about TTLs. Importing this module requires streamlit to be available.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
try:
    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
except Exception:
    fred = None


# -------- SEC EDGAR --------

@st.cache_resource
def _sec_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers.update({
        "User-Agent": os.getenv(
            "SEC_USER_AGENT", "TheTrueOracle_Quantitative_Engine info@example.com"
        ),
        "Accept-Encoding": "gzip, deflate",
    })
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    return s


@st.cache_data(ttl=86400)
def _sec_ticker_map():
    sess = _sec_session()
    r = sess.get("https://www.sec.gov/files/company_tickers.json", timeout=10)
    return r.json()


def fetch_sec_filing(ticker: str, form_type: str = "10-K"):
    """Return (clean_text, doc_url) for the most recent matching filing, or (None, error_message)."""
    sess = _sec_session()
    try:
        ticker_map = _sec_ticker_map()
        cik = None
        for _, company in ticker_map.items():
            if company["ticker"] == ticker.upper():
                cik = str(company["cik_str"]).zfill(10)
                break
        if not cik:
            return None, "Ticker not found in SEC EDGAR database."

        subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        filings = sess.get(subs_url, timeout=10).json()["filings"]["recent"]

        for i, f_type in enumerate(filings["form"]):
            if f_type == form_type:
                accession = filings["accessionNumber"][i].replace("-", "")
                primary = filings["primaryDocument"][i]
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary}"
                doc_response = sess.get(doc_url, timeout=15)
                soup = BeautifulSoup(doc_response.content, "html.parser")
                return soup.get_text(separator="\n", strip=True), doc_url

        return None, f"No {form_type} filing found for this company in recent history."
    except Exception as e:
        return None, f"SEC API Error: {e}"


# -------- Yahoo financial news RSS --------

def fetch_financial_news(ticker: str):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    articles = []
    try:
        response = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(response.content)
        for item in root.findall(".//item")[:5]:
            title = item.find("title").text if item.find("title") is not None else "No Title"
            link = item.find("link").text if item.find("link") is not None else "#"
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else "Recent"
            articles.append(
                {"title": title, "link": link, "time": pub_date, "publisher": "Financial Wire"}
            )
    except Exception:
        pass
    return articles


# -------- yfinance batched fetchers --------

@st.cache_data(ttl=60)
def fetch_live_prices(tickers):
    prices = {}
    real_tickers = []
    for ticker in tickers:
        if not ticker:
            continue
        if ticker.upper() == "CASH":
            prices[ticker] = {"price": 1.00, "change": 0.0}
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
            prices[ticker] = {"price": round(price, 2), "change": round(pct_change, 2)}
        except Exception:
            prices[ticker] = {"price": None, "change": None}
    return prices


@st.cache_data(ttl=3600)
def fetch_stock_details(ticker, period):
    stock = yf.Ticker(ticker)
    period_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "Max": "max"}
    yf_period = period_map.get(period, "1mo")
    try:
        interval = "1m" if yf_period == "1d" else "1d"
        hist = stock.history(period=yf_period, interval=interval)
    except Exception:
        hist = pd.DataFrame()
    try:
        info = stock.info
    except Exception:
        info = {}
    return hist, info


@st.cache_data(ttl=86400)
def fetch_dcf_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get("currentPrice") or info.get("previousClose")
    if not current_price:
        try:
            current_price = stock.fast_info.last_price
        except Exception:
            current_price = None
    shares_out = info.get("sharesOutstanding")
    try:
        cf = stock.cash_flow
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow"].iloc[0]
        else:
            fcf = (
                (cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else 0)
                + (cf.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cf.index else 0)
            )
    except Exception:
        fcf = None
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
                "Div Yield (%)": (info.get("dividendYield", 0) * 100) if info.get("dividendYield") is not None else None,
            })
        except Exception:
            data.append({"Ticker": t})
    return pd.DataFrame(data)


# -------- FRED macro --------

@st.cache_data(ttl=86400)
def fetch_macro_data(series_id):
    if not fred:
        return None
    try:
        data = fred.get_series(series_id)
        df = pd.DataFrame(data, columns=["Value"])
        df.index.name = "Date"
        df = df.dropna()
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=20)
        return df[df.index >= cutoff_date]
    except Exception:
        return None
