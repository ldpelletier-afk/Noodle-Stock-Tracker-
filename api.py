"""External data fetchers: yfinance, SEC EDGAR, FRED, Yahoo news RSS.

All Streamlit cache decorators live here so call sites don't need to know
about TTLs. Importing this module requires streamlit to be available.
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")

try:
    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
except Exception:
    fred = None

_DEFAULT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


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


@st.cache_data(ttl=3600)
def fetch_recent_sec_filings(ticker: str, n: int = 10) -> list[dict]:
    """List the most recent N SEC filings for a ticker.

    Returns list of {form, date, accession, url, primary_document}.
    """
    sess = _sec_session()
    try:
        ticker_map = _sec_ticker_map()
        cik = None
        for _, company in ticker_map.items():
            if company["ticker"] == ticker.upper():
                cik = str(company["cik_str"]).zfill(10)
                break
        if not cik:
            return []

        subs = sess.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=10
        ).json()
        recent = subs["filings"]["recent"]

        out = []
        for i in range(min(n, len(recent["form"]))):
            accession = recent["accessionNumber"][i]
            accession_nodash = accession.replace("-", "")
            primary = recent["primaryDocument"][i]
            out.append({
                "form": recent["form"][i],
                "date": recent["filingDate"][i],
                "accession": accession,
                "primary_document": primary,
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary}",
                "index_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=include&count=40",
            })
        return out
    except Exception:
        return []


# -------- Financial statements & highlights --------

@st.cache_data(ttl=86400)
def fetch_financial_statements(ticker: str) -> dict[str, pd.DataFrame]:
    """Annual income statement, balance sheet, and cash flow from yfinance.

    Returns {'income': DataFrame, 'balance_sheet': DataFrame, 'cashflow': DataFrame}.
    Columns are period end dates (newest first); rows are line items.
    Missing statements come back as empty DataFrames.
    """
    stock = yf.Ticker(ticker)
    out: dict[str, pd.DataFrame] = {}
    for key, attrs in (
        ("income", ("income_stmt", "financials")),
        ("balance_sheet", ("balance_sheet",)),
        ("cashflow", ("cashflow", "cash_flow")),
    ):
        df = pd.DataFrame()
        for attr in attrs:
            try:
                candidate = getattr(stock, attr, None)
                if candidate is not None and not candidate.empty:
                    df = candidate
                    break
            except Exception:
                continue
        out[key] = df
    return out


@st.cache_data(ttl=86400)
def fetch_financial_highlights(ticker: str) -> dict:
    """Flat dict of headline metrics for the Favorites summary strip."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception:
        info = {}

    def _latest(df, row_candidates):
        if df is None or df.empty:
            return None
        for name in row_candidates:
            if name in df.index:
                try:
                    val = df.loc[name].iloc[0]
                    return float(val) if pd.notna(val) else None
                except Exception:
                    continue
        return None

    statements = fetch_financial_statements(ticker)
    income = statements["income"]
    balance = statements["balance_sheet"]
    cashflow = statements["cashflow"]

    revenue = _latest(income, ["Total Revenue", "Revenue", "Operating Revenue"])
    gross_profit = _latest(income, ["Gross Profit"])
    net_income = _latest(income, ["Net Income", "Net Income Common Stockholders"])
    operating_income = _latest(income, ["Operating Income"])

    total_debt = _latest(balance, [
        "Total Debt",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    ])
    cash = _latest(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])

    fcf = _latest(cashflow, ["Free Cash Flow"])
    if fcf is None:
        op_cf = _latest(cashflow, ["Operating Cash Flow", "Cash Flow From Operations"])
        capex = _latest(cashflow, ["Capital Expenditure"])
        if op_cf is not None and capex is not None:
            fcf = op_cf + capex  # capex is negative

    return {
        "currency": info.get("financialCurrency") or info.get("currency"),
        "market_cap": info.get("marketCap"),
        "price": info.get("currentPrice") or info.get("previousClose"),
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "eps_trailing": info.get("trailingEps"),
        "dividend_yield": (info.get("dividendYield") or 0) * 100 if info.get("dividendYield") else None,
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "fcf": fcf,
        "total_debt": total_debt,
        "cash": cash,
        "net_debt": (total_debt - cash) if (total_debt is not None and cash is not None) else None,
        "profit_margin": (net_income / revenue * 100) if (revenue and net_income) else None,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "long_name": info.get("longName") or info.get("shortName"),
    }


# -------- News: multi-source adapters --------
#
# Each adapter returns list[Article] where Article is:
#   {"title": str, "url": str, "source": str, "time": str (human),
#    "published_ts": int (epoch, may be 0), "summary": str}
# fetch_all_news() runs adapters in parallel, dedupes by URL, sorts by ts desc.

def _parse_rss_date(s: str) -> int:
    if not s:
        return 0
    try:
        return int(parsedate_to_datetime(s).timestamp())
    except Exception:
        return 0


def _parse_rss(xml_bytes: bytes, source: str, limit: int = 20) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()
        if not title or not link:
            continue
        ts = _parse_rss_date(pub)
        out.append({
            "title": title,
            "url": link,
            "source": source,
            "time": pub or "Recent",
            "published_ts": ts,
            "summary": BeautifulSoup(desc, "html.parser").get_text(" ", strip=True) if desc else "",
        })
    return out


def _get_rss(url: str, source: str, limit: int = 20, timeout: int = 5) -> list[dict]:
    try:
        r = requests.get(url, headers={"User-Agent": _DEFAULT_UA}, timeout=timeout)
        if r.status_code != 200:
            return []
        return _parse_rss(r.content, source=source, limit=limit)
    except Exception:
        return []


def fetch_news_yahoo(ticker: str) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    return _get_rss(url, source="Yahoo Finance", limit=10)


def _google_news_rss(query: str, limit: int = 10) -> list[dict]:
    url = (
        "https://news.google.com/rss/search"
        f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    items = _get_rss(url, source="Google News", limit=limit)
    # Google News wraps the publisher name in the <source> element inside each <item>,
    # and puts it at the tail of the title as " - Publisher". Prefer that as source.
    for it in items:
        m = re.search(r"\s[-–]\s([^-–]+)$", it["title"])
        if m:
            it["source"] = m.group(1).strip()
            it["title"] = it["title"][: m.start()].strip()
    return items


def fetch_news_google(ticker: str) -> list[dict]:
    return _google_news_rss(f"{ticker} stock", limit=15)


def fetch_news_google_wsj(ticker: str) -> list[dict]:
    items = _google_news_rss(f"{ticker} site:wsj.com", limit=10)
    for it in items:
        it["source"] = "WSJ"
    return items


def fetch_news_google_ft(ticker: str) -> list[dict]:
    items = _google_news_rss(f"{ticker} site:ft.com", limit=10)
    for it in items:
        it["source"] = "Financial Times"
    return items


def fetch_news_google_bloomberg(ticker: str) -> list[dict]:
    items = _google_news_rss(f"{ticker} site:bloomberg.com", limit=10)
    for it in items:
        it["source"] = "Bloomberg"
    return items


def fetch_news_google_reuters(ticker: str) -> list[dict]:
    items = _google_news_rss(f"{ticker} site:reuters.com", limit=10)
    for it in items:
        it["source"] = "Reuters"
    return items


def fetch_news_seeking_alpha(ticker: str) -> list[dict]:
    url = f"https://seekingalpha.com/api/sa/combined/{ticker.upper()}.xml"
    return _get_rss(url, source="Seeking Alpha", limit=10)


def fetch_news_finnhub(ticker: str) -> list[dict]:
    if not FINNHUB_API_KEY:
        return []
    today = time.strftime("%Y-%m-%d")
    a_month_ago = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30 * 86400))
    url = (
        "https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={a_month_ago}&to={today}&token={FINNHUB_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        data = r.json() or []
    except Exception:
        return []
    out = []
    for item in data[:30]:
        ts = int(item.get("datetime") or 0)
        out.append({
            "title": (item.get("headline") or "").strip(),
            "url": (item.get("url") or "").strip(),
            "source": (item.get("source") or "Finnhub").strip(),
            "time": time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)) if ts else "Recent",
            "published_ts": ts,
            "summary": (item.get("summary") or "").strip(),
        })
    return [o for o in out if o["title"] and o["url"]]


def fetch_news_newsapi(ticker: str) -> list[dict]:
    if not NEWSAPI_KEY:
        return []
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(ticker)}"
        "&language=en&sortBy=publishedAt&pageSize=25"
        f"&apiKey={NEWSAPI_KEY}"
    )
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        items = (r.json() or {}).get("articles", [])
    except Exception:
        return []
    out = []
    for a in items:
        pub = a.get("publishedAt") or ""
        try:
            ts = int(pd.Timestamp(pub).timestamp()) if pub else 0
        except Exception:
            ts = 0
        out.append({
            "title": (a.get("title") or "").strip(),
            "url": (a.get("url") or "").strip(),
            "source": ((a.get("source") or {}).get("name") or "NewsAPI").strip(),
            "time": pub or "Recent",
            "published_ts": ts,
            "summary": (a.get("description") or "").strip(),
        })
    return [o for o in out if o["title"] and o["url"]]


def fetch_news_marketaux(ticker: str) -> list[dict]:
    if not MARKETAUX_API_KEY:
        return []
    url = (
        "https://api.marketaux.com/v1/news/all"
        f"?symbols={ticker}&language=en&limit=25&api_token={MARKETAUX_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        items = (r.json() or {}).get("data", [])
    except Exception:
        return []
    out = []
    for a in items:
        pub = a.get("published_at") or ""
        try:
            ts = int(pd.Timestamp(pub).timestamp()) if pub else 0
        except Exception:
            ts = 0
        out.append({
            "title": (a.get("title") or "").strip(),
            "url": (a.get("url") or "").strip(),
            "source": (a.get("source") or "MarketAux").strip(),
            "time": pub or "Recent",
            "published_ts": ts,
            "summary": (a.get("snippet") or a.get("description") or "").strip(),
        })
    return [o for o in out if o["title"] and o["url"]]


# Keep legacy function name for back-compat with the existing Intelligence tab.
def fetch_financial_news(ticker: str):
    """Legacy single-source Yahoo news (kept for tab4 Intelligence)."""
    articles = fetch_news_yahoo(ticker)[:5]
    # Translate keys back to the legacy shape expected by tab4.
    return [
        {
            "title": a["title"],
            "link": a["url"],
            "time": a["time"],
            "publisher": a["source"],
        }
        for a in articles
    ]


NEWS_ADAPTERS = [
    ("yahoo", fetch_news_yahoo),
    ("google", fetch_news_google),
    ("google-wsj", fetch_news_google_wsj),
    ("google-ft", fetch_news_google_ft),
    ("google-bloomberg", fetch_news_google_bloomberg),
    ("google-reuters", fetch_news_google_reuters),
    ("seeking-alpha", fetch_news_seeking_alpha),
    ("finnhub", fetch_news_finnhub),
    ("newsapi", fetch_news_newsapi),
    ("marketaux", fetch_news_marketaux),
]


@st.cache_data(ttl=600)
def fetch_all_news(ticker: str, enabled: tuple[str, ...] | None = None) -> list[dict]:
    """Run all enabled news adapters concurrently; dedupe by URL; sort by time desc.

    `enabled`: optional tuple of adapter names to include (tuple, not list, so
    streamlit can hash it for caching). If None, all adapters run.
    """
    adapters = NEWS_ADAPTERS
    if enabled is not None:
        allowed = set(enabled)
        adapters = [(n, fn) for n, fn in NEWS_ADAPTERS if n in allowed]

    all_items: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(adapters)) as ex:
        futures = {ex.submit(fn, ticker): name for name, fn in adapters}
        for fut in as_completed(futures):
            try:
                all_items.extend(fut.result() or [])
            except Exception:
                continue

    # Dedupe by URL (prefer the entry with better metadata — non-zero ts).
    by_url: dict[str, dict] = {}
    for item in all_items:
        key = item["url"]
        existing = by_url.get(key)
        if existing is None:
            by_url[key] = item
        elif item["published_ts"] and not existing["published_ts"]:
            by_url[key] = item

    out = list(by_url.values())
    out.sort(key=lambda a: a["published_ts"] or 0, reverse=True)
    return out


# -------- URL metadata extraction (for manually saved articles) --------

@st.cache_data(ttl=86400)
def fetch_url_metadata(url: str) -> dict:
    """Fetch og:title / og:site_name / og:published_time from a URL's HTML head.

    Works on paywalled articles (WSJ, FT, etc.) because Open Graph tags are
    served publicly for social previews. Returns a dict with string values
    (empty strings if not found), plus a derived 'source' from the domain.
    """
    out = {
        "title": "",
        "source": "",
        "description": "",
        "published_ts": 0,
    }
    if not url:
        return out
    # Derive source from domain as a fallback
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        out["source"] = _pretty_source_from_domain(domain)
    except Exception:
        pass

    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": _DEFAULT_UA,
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=6,
            allow_redirects=True,
        )
        if r.status_code != 200 or not r.text:
            return out
        soup = BeautifulSoup(r.text, "html.parser")

        def og(prop):
            tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
            return (tag["content"].strip() if tag and tag.get("content") else "")

        title = og("og:title") or (soup.title.string.strip() if soup.title and soup.title.string else "")
        site = og("og:site_name")
        desc = og("og:description") or og("description")
        pub = og("article:published_time") or og("og:published_time")

        if title:
            out["title"] = title
        if site:
            out["source"] = site
        if desc:
            out["description"] = desc
        if pub:
            try:
                out["published_ts"] = int(pd.Timestamp(pub).timestamp())
            except Exception:
                pass
    except Exception:
        pass
    return out


_DOMAIN_PRETTY = {
    "wsj.com": "WSJ",
    "ft.com": "Financial Times",
    "bloomberg.com": "Bloomberg",
    "reuters.com": "Reuters",
    "nytimes.com": "New York Times",
    "economist.com": "The Economist",
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "barrons.com": "Barron's",
    "seekingalpha.com": "Seeking Alpha",
    "finance.yahoo.com": "Yahoo Finance",
    "yahoo.com": "Yahoo",
    "reuters.com": "Reuters",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "theguardian.com": "The Guardian",
    "ap.org": "Associated Press",
    "apnews.com": "Associated Press",
    "foxbusiness.com": "Fox Business",
    "businessinsider.com": "Business Insider",
    "forbes.com": "Forbes",
    "fool.com": "The Motley Fool",
    "investors.com": "Investor's Business Daily",
    "sec.gov": "SEC",
}


def _pretty_source_from_domain(domain: str) -> str:
    if not domain:
        return ""
    domain = domain.lower()
    if domain in _DOMAIN_PRETTY:
        return _DOMAIN_PRETTY[domain]
    # Strip common country suffixes for pretty-printing
    base = domain.rsplit(":", 1)[0]
    parts = base.split(".")
    if len(parts) >= 2:
        return parts[-2].capitalize()
    return domain


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
def fetch_calendar_events(tickers, days_ahead: int = 60):
    """Fetch upcoming earnings + ex-dividend dates for the given tickers.

    Returns a DataFrame sorted by Date with columns:
      Ticker, Event, Date, Detail
    Only events within the next `days_ahead` days are kept.
    `tickers` should be a tuple (so streamlit can hash it for caching).
    """
    rows = []
    for ticker in tickers:
        if not ticker or ticker.upper() == "CASH":
            continue
        try:
            stock = yf.Ticker(ticker)
            try:
                cal = stock.calendar or {}
            except Exception:
                cal = {}
            try:
                info = stock.info or {}
            except Exception:
                info = {}

            # --- Earnings date (calendar may store a list of dates) ---
            earn_raw = cal.get("Earnings Date")
            earn_date = None
            if earn_raw:
                if isinstance(earn_raw, list) and earn_raw:
                    earn_date = earn_raw[0]
                else:
                    earn_date = earn_raw
            try:
                earn_date = pd.to_datetime(earn_date) if earn_date else None
            except Exception:
                earn_date = None
            if earn_date is not None:
                eps_est = cal.get("Earnings Average")
                detail = (
                    f"EPS est: ${eps_est:.2f}"
                    if isinstance(eps_est, (int, float))
                    else "Earnings"
                )
                rows.append({
                    "Ticker": ticker,
                    "Event": "📊 Earnings",
                    "Date": earn_date,
                    "Detail": detail,
                })

            # --- Ex-dividend date ---
            ex_raw = cal.get("Ex-Dividend Date") or info.get("exDividendDate")
            try:
                if isinstance(ex_raw, (int, float)):
                    ex_date = pd.to_datetime(ex_raw, unit="s")
                elif ex_raw:
                    ex_date = pd.to_datetime(ex_raw)
                else:
                    ex_date = None
            except Exception:
                ex_date = None
            if ex_date is not None:
                amount = info.get("lastDividendValue") or info.get("dividendRate")
                yield_raw = info.get("dividendYield") or 0
                # yfinance is inconsistent: sometimes 0.018 (=1.8%), sometimes 1.8
                yld_pct = yield_raw * 100 if 0 < yield_raw < 1 else yield_raw
                detail_parts = []
                if isinstance(amount, (int, float)) and amount > 0:
                    detail_parts.append(f"${amount:.2f}/share")
                if yld_pct:
                    detail_parts.append(f"{yld_pct:.2f}% yield")
                detail = " · ".join(detail_parts) or "Ex-dividend"
                rows.append({
                    "Ticker": ticker,
                    "Event": "💰 Ex-Div",
                    "Date": ex_date,
                    "Detail": detail,
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Event", "Date", "Detail"])

    df = pd.DataFrame(rows)
    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df[(df["Date"] >= today) & (df["Date"] <= cutoff)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(ttl=900)
def fetch_portfolio_value_history(portfolio_name=None, days: int = 365):
    """Reconstruct portfolio value at every business day for the past `days`.

    Pulls the transaction log, replays buys/sells chronologically (CASH "buy"
    = deposit, "sell" = withdrawal), and multiplies each day's holdings by
    historical close prices from yfinance. Returns a DataFrame with columns
    Date and Value.

    `portfolio_name=None` aggregates across every portfolio.
    """
    # Local import to avoid circular deps at module load
    from data_store import fetch_transactions

    txns = fetch_transactions(portfolio_name)
    if not txns:
        return pd.DataFrame(columns=["Date", "Value"])

    txns = sorted(txns, key=lambda r: r["ts"])
    today = pd.Timestamp.now().normalize()
    start_dt = today - pd.Timedelta(days=days)
    earliest_tx = pd.Timestamp(txns[0]["ts"], unit="s").normalize()
    fetch_start = min(start_dt, earliest_tx)

    # Tickers ever traded (CASH handled separately)
    tickers = sorted({t["ticker"] for t in txns if t["ticker"].upper() != "CASH"})
    price_series: dict[str, pd.Series] = {}

    if tickers:
        def _fetch_one(t):
            try:
                h = yf.Ticker(t).history(
                    start=fetch_start.strftime("%Y-%m-%d"),
                    end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                )
                if not h.empty and "Close" in h.columns:
                    s = h["Close"].copy()
                    if s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    s.index = s.index.normalize()
                    return t, s
            except Exception:
                pass
            return t, None

        with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as ex:
            futures = {ex.submit(_fetch_one, t): t for t in tickers}
            for fut in as_completed(futures):
                try:
                    tk, s = fut.result()
                    if s is not None and not s.empty:
                        price_series[tk] = s
                except Exception:
                    continue

    # Replay transactions across business days, snapshotting value
    business_days = pd.date_range(start_dt, today, freq="B")
    rows = []
    holdings: dict[str, float] = {}
    cash = 0.0
    tx_idx = 0

    for date in business_days:
        while (
            tx_idx < len(txns)
            and pd.Timestamp(txns[tx_idx]["ts"], unit="s").normalize() <= date
        ):
            tx = txns[tx_idx]
            t = tx["ticker"]
            q = float(tx["quantity"])
            p = float(tx["price"])
            action = tx["action"].upper()
            if t.upper() == "CASH":
                cash += q if action == "BUY" else -q
            else:
                if action == "BUY":
                    holdings[t] = holdings.get(t, 0.0) + q
                    cash -= q * p
                elif action == "SELL":
                    holdings[t] = holdings.get(t, 0.0) - q
                    cash += q * p
            tx_idx += 1

        value = cash
        for t, qty in holdings.items():
            if qty <= 0:
                continue
            s = price_series.get(t)
            if s is None or s.empty:
                continue
            idx = s.index.searchsorted(date, side="right") - 1
            if 0 <= idx < len(s):
                value += qty * float(s.iloc[idx])
        rows.append({"Date": date, "Value": value})

    return pd.DataFrame(rows)


@st.cache_data(ttl=900)
def fetch_sparkline_history(tickers, days: int = 30):
    """Fetch closing prices for the last `days` trading days for each ticker.

    Returns {ticker: [close prices, oldest first]}. Used to feed
    st.column_config.LineChartColumn for inline sparklines.
    `tickers` should be a tuple (hashable for caching).
    """
    real_tickers = [t for t in tickers if t and t.upper() != "CASH"]
    if not real_tickers:
        return {}

    out: dict[str, list[float]] = {}

    def _fetch_one(t):
        try:
            hist = yf.Ticker(t).history(period=f"{days + 5}d", interval="1d")
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna().tail(days).tolist()
                return t, [float(c) for c in closes]
        except Exception:
            pass
        return t, []

    with ThreadPoolExecutor(max_workers=min(10, len(real_tickers))) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in real_tickers}
        for fut in as_completed(futures):
            try:
                ticker, closes = fut.result()
                if closes:
                    out[ticker] = closes
            except Exception:
                continue
    return out


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
