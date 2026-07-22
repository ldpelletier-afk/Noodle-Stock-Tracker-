"""External data fetchers: yfinance, SEC EDGAR, FRED, Yahoo news RSS.

All cache decorators live here so call sites don't need to know about TTLs.
Uses the `cache` shim — works both inside and outside a Streamlit session.
"""
from __future__ import annotations

import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import pandas as pd
import requests
import yfinance as yf
from cache import cache_data, cache_resource
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
BLS_API_KEY        = os.getenv("BLS_API_KEY")         # optional — unlocks v2 (20yr history)
BEA_API_KEY        = os.getenv("BEA_API_KEY")         # required for BEA GDP/PCE data
SIMFIN_API_KEY     = os.getenv("SIMFIN_API_KEY")      # free at simfin.com
EIA_API_KEY        = os.getenv("EIA_API_KEY")          # free at eia.gov/opendata
FMP_API_KEY        = os.getenv("FMP_API_KEY")          # free at financialmodelingprep.com (250/day)
ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")  # 25/day free
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")  # free at courtlistener.com/register


def has_eia() -> bool:
    return bool((EIA_API_KEY or "").strip())

def has_fmp() -> bool:
    return bool((FMP_API_KEY or "").strip())

def has_alpha_vantage() -> bool:
    return bool((ALPHA_VANTAGE_KEY or "").strip())

def has_courtlistener() -> bool:
    return bool((COURTLISTENER_API_KEY or "").strip())

try:
    fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
except Exception:
    fred = None

_DEFAULT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


# -------- Alpaca real-time quotes (free IEX feed) --------
# Alpaca's free tier returns real-time IEX (Investors Exchange) quotes for US
# equities — typically <1 second latency vs. yfinance's ~15-min delay. Coverage
# is US-listed only; international/OTC tickers fall back to yfinance.

def has_alpaca() -> bool:
    return bool((ALPACA_API_KEY or "").strip() and (ALPACA_SECRET_KEY or "").strip())


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": (ALPACA_API_KEY or "").strip(),
        "APCA-API-SECRET-KEY": (ALPACA_SECRET_KEY or "").strip(),
        "Accept": "application/json",
    }


def _alpaca_normalize_symbol(ticker: str) -> str:
    """yfinance uses BRK-B; Alpaca uses BRK.B. Normalize before sending."""
    return ticker.upper().replace("-", ".")


def _alpaca_fetch_snapshots(tickers: list[str]) -> dict[str, dict]:
    """Return ``{ticker: {"price": float, "change": float}}`` for tickers that
    Alpaca recognizes; missing tickers (international, OTC, etc.) are simply
    absent from the result so the caller can fall back to yfinance.

    The /v2/stocks/snapshots endpoint accepts up to 100 symbols per call.
    """
    if not tickers or not has_alpaca():
        return {}

    # Map Alpaca symbol -> original (yfinance) symbol so we return the keys
    # the rest of the app expects.
    symbol_map: dict[str, str] = {}
    for t in tickers:
        symbol_map[_alpaca_normalize_symbol(t)] = t

    out: dict[str, dict] = {}
    # Chunk into 100-symbol batches.
    alpaca_syms = list(symbol_map.keys())
    for i in range(0, len(alpaca_syms), 100):
        chunk = alpaca_syms[i : i + 100]
        try:
            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/snapshots",
                params={"symbols": ",".join(chunk)},
                headers=_alpaca_headers(),
                timeout=6,
            )
            if r.status_code != 200:
                continue
            data = r.json() or {}
        except Exception:
            continue

        for sym, snap in data.items():
            if not isinstance(snap, dict):
                continue
            latest_trade = (snap.get("latestTrade") or {})
            prev_bar = (snap.get("prevDailyBar") or {})
            daily_bar = (snap.get("dailyBar") or {})

            price = latest_trade.get("p")
            # Fallback: today's close if no trade today.
            if price is None:
                price = daily_bar.get("c")
            prev_close = prev_bar.get("c")

            if price is None or prev_close in (None, 0):
                continue

            try:
                pct = ((float(price) - float(prev_close)) / float(prev_close)) * 100.0
            except Exception:
                continue

            original = symbol_map.get(sym, sym)
            out[original] = {
                "price": round(float(price), 2),
                "change": round(pct, 2),
            }
    return out


# -------- SEC EDGAR --------

@cache_resource
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


@cache_data(ttl=86400)
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


@cache_data(ttl=3600)
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

@cache_data(ttl=86400)
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


@cache_data(ttl=86400)
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


@cache_data(ttl=600)
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

@cache_data(ttl=86400)
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

    html = _fetch_html(url)
    if not html:
        return out
    try:
        # Trafilatura's metadata extractor pulls Open Graph + JSON-LD + bare
        # <title> in one pass, normalises author/date, and is more robust than
        # the manual og:* sweep we used to do.
        try:
            import trafilatura  # type: ignore
            meta = trafilatura.extract_metadata(html, default_url=url)
        except Exception:
            meta = None

        if meta is not None:
            if getattr(meta, "title", None):
                out["title"] = meta.title.strip()
            if getattr(meta, "sitename", None):
                out["source"] = meta.sitename.strip()
            if getattr(meta, "description", None):
                out["description"] = meta.description.strip()
            if getattr(meta, "date", None):
                try:
                    out["published_ts"] = int(pd.Timestamp(meta.date).timestamp())
                except Exception:
                    pass

        # Fall back to a manual og: sweep for any field trafilatura missed.
        if not (out["title"] and out["description"]):
            soup = BeautifulSoup(html, "html.parser")

            def og(prop):
                tag = (
                    soup.find("meta", attrs={"property": prop})
                    or soup.find("meta", attrs={"name": prop})
                )
                return (tag["content"].strip() if tag and tag.get("content") else "")

            if not out["title"]:
                out["title"] = og("og:title") or (
                    soup.title.string.strip() if soup.title and soup.title.string else ""
                )
            if not out["source"]:
                out["source"] = og("og:site_name") or out["source"]
            if not out["description"]:
                out["description"] = og("og:description") or og("description")
            if not out["published_ts"]:
                pub = og("article:published_time") or og("og:published_time")
                if pub:
                    try:
                        out["published_ts"] = int(pd.Timestamp(pub).timestamp())
                    except Exception:
                        pass
    except Exception:
        pass
    return out


# ── HTML fetching: requests fast-path + Playwright fallback ────────────────
#
# Most news / blog sites still serve their content in the initial HTML
# response, so a 6-second `requests.get` is the right default. But research-
# portal sites (Goldman Sachs, Morgan Stanley, some FT and Bloomberg pages)
# render the article via JavaScript after the initial document loads — those
# pages return a near-empty <body> to a static fetcher. ``_fetch_html_dynamic``
# spins up a headless Chromium via Playwright for those cases.

def _fetch_html(url: str, *, timeout: int = 6) -> str:
    """Static GET. Returns the HTML string or '' on failure."""
    if not url:
        return ""
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": _DEFAULT_UA,
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=timeout,
            allow_redirects=True,
        )
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        pass
    return ""


def _fetch_html_dynamic(url: str, *, timeout_s: float = 15.0) -> str:
    """Render ``url`` in a headless Chromium and return the post-JS HTML.

    Falls back to '' if Playwright isn't installed, the browser binary is
    missing, or the navigation times out — callers should treat the result
    as best-effort.
    """
    if not url:
        return ""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(user_agent=_DEFAULT_UA)
                page = context.new_page()
                page.set_default_navigation_timeout(int(timeout_s * 1000))
                page.goto(url, wait_until="domcontentloaded")
                # Give late-mount JS frameworks a beat to flush content.
                try:
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except Exception:
                    pass
                html = page.content()
            finally:
                browser.close()
        return html or ""
    except Exception:
        return ""


@cache_data(ttl=3600)
def fetch_article_text(url: str, *, force_dynamic: bool = False) -> dict:
    """Extract clean article text from ``url``.

    Tries a static ``requests.get`` first (fast, free, works on >90% of news
    sites). If trafilatura can't pull at least 500 characters of body text
    out of the result — the usual signature of a JS-rendered page — we
    re-render the URL through Playwright and try again. ``force_dynamic``
    skips the static attempt entirely.

    Returns a dict::

        {
            "url":          str,
            "text":         str,    # cleaned article body, paragraphs joined
            "title":        str,
            "source":       str,    # publisher name (best-effort)
            "published_ts": int,    # 0 when unknown
            "rendered":     "static" | "dynamic" | "none",
            "char_count":   int,
        }
    """
    out = {
        "url": url,
        "text": "",
        "title": "",
        "source": "",
        "published_ts": 0,
        "rendered": "none",
        "char_count": 0,
    }
    if not url:
        return out

    try:
        import trafilatura  # type: ignore
    except Exception:
        return out

    def _try(html: str, mode: str) -> bool:
        if not html:
            return False
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        ) or ""
        if len(text) < 500 and not force_dynamic:
            # Too short — probably a JS-rendered shell.
            return False
        out["text"] = text
        out["rendered"] = mode
        out["char_count"] = len(text)

        meta = trafilatura.extract_metadata(html, default_url=url)
        if meta is not None:
            out["title"] = (getattr(meta, "title", "") or "").strip()
            out["source"] = (getattr(meta, "sitename", "") or "").strip()
            if getattr(meta, "date", None):
                try:
                    out["published_ts"] = int(pd.Timestamp(meta.date).timestamp())
                except Exception:
                    pass
        return True

    if not force_dynamic:
        if _try(_fetch_html(url, timeout=8), "static"):
            return out

    # Static path either failed or was skipped — render with Playwright.
    if _try(_fetch_html_dynamic(url), "dynamic"):
        return out

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

# When Alpaca is configured we re-fetch every 30 seconds (real-time data is
# cheap to refresh); without it we keep the conservative 60-second TTL on
# yfinance to avoid hammering Yahoo. The function-level cache key still
# includes the tickers so different watchlists don't trample each other.
_LIVE_PRICE_TTL = 30 if (ALPACA_API_KEY and ALPACA_SECRET_KEY) else 60


# ── Per-ticker live-price store ───────────────────────────────────────────────
# A single, process-wide, thread-safe cache keyed by *individual* ticker (not by
# the exact list a caller passes). Because every Portfolio tab requests a
# different subset of tickers (holdings, one watchlist, favourites, …), keying
# the cache on the whole list — as a plain @cache_data would — means overlapping
# requests never share data and each tab pays its own cold fetch. Keying per
# ticker lets the first fetch of a symbol (including the startup prefetch, which
# runs in a background thread with no Streamlit context) serve every tab for the
# next ``_LIVE_PRICE_TTL`` seconds.
_live_price_store: dict[str, tuple[dict, float]] = {}
_live_price_lock = threading.Lock()


def _live_prices_from_store(tickers) -> dict:
    """Return ``{ticker: record}`` for tickers whose cached price is still fresh."""
    fresh = {}
    now = time.monotonic()
    with _live_price_lock:
        for ticker in tickers:
            entry = _live_price_store.get(ticker)
            if entry is not None:
                record, ts = entry
                if (now - ts) < _LIVE_PRICE_TTL:
                    fresh[ticker] = record
    return fresh


def _live_prices_to_store(records: dict) -> None:
    now = time.monotonic()
    with _live_price_lock:
        for ticker, record in records.items():
            _live_price_store[ticker] = (record, now)


def _fetch_live_prices_network(tickers) -> dict:
    """Batch-fetch prices for ``tickers`` (Alpaca real-time → yfinance fallback)."""
    prices = {}
    real_tickers = [t for t in tickers if t]
    if not real_tickers:
        return prices

    # ---- Step 1: Alpaca real-time (US-listed only) ----
    alpaca_prices = _alpaca_fetch_snapshots(real_tickers) if has_alpaca() else {}
    prices.update(alpaca_prices)

    # ---- Step 2: yfinance fallback for whatever Alpaca didn't return ----
    missing = [t for t in real_tickers if t not in alpaca_prices]
    if missing:
        try:
            batch = yf.Tickers(" ".join(missing))
        except Exception:
            batch = None

        for ticker in missing:
            try:
                fast = (
                    batch.tickers[ticker].fast_info
                    if batch
                    else yf.Ticker(ticker).fast_info
                )
                price = fast["last_price"]
                prev_close = fast["previous_close"]
                pct_change = (
                    ((price - prev_close) / prev_close) * 100 if prev_close else 0.0
                )
                prices[ticker] = {
                    "price": round(price, 2),
                    "change": round(pct_change, 2),
                }
            except Exception:
                prices[ticker] = {"price": None, "change": None}

    return prices


def fetch_live_prices(tickers):
    """Return ``{ticker: {"price": float|None, "change": float|None}}``.

    Backed by a per-ticker TTL store (see above) so overlapping ticker lists
    requested by different tabs share warmed data — the first fetch of a symbol,
    including the background prefetch at startup, serves every tab for the next
    ``_LIVE_PRICE_TTL`` seconds regardless of which subset each tab asks for.

    Resolution order per ticker:
      1. Alpaca free real-time IEX feed (when API keys are set in .env)
      2. yfinance ~15-minute delayed feed (fallback / international / OTC)
      3. None / None if both fail

    CASH is hard-coded to {"price": 1.00, "change": 0.0}.
    """
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

    cached = _live_prices_from_store(real_tickers)
    prices.update(cached)

    missing = [t for t in real_tickers if t not in cached]
    if missing:
        fetched = _fetch_live_prices_network(missing)
        _live_prices_to_store(fetched)
        prices.update(fetched)

    return prices


def _clear_live_prices() -> None:
    """Bust the per-ticker store (used by the Dashboard/Market Watch refresh)."""
    with _live_price_lock:
        _live_price_store.clear()


# Preserve the ``fetch_live_prices.clear()`` API the tabs already call.
fetch_live_prices.clear = _clear_live_prices


def live_price_feed_status(tickers: list[str] | None = None) -> dict:
    """Returns a small status dict describing which feed is active.

    Useful for the UI to render a feed badge:
      {"alpaca_configured": bool, "ttl_seconds": int, "label": str}
    """
    if has_alpaca():
        return {
            "alpaca_configured": True,
            "ttl_seconds": 30,
            "label": "Alpaca real-time (IEX)",
        }
    return {
        "alpaca_configured": False,
        "ttl_seconds": 60,
        "label": "yfinance (~15-min delayed)",
    }


@cache_data(ttl=3600)
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


@cache_data(ttl=900)
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


@cache_data(ttl=900)
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


@cache_data(ttl=3600)
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


@cache_data(ttl=86400)
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


@cache_data(ttl=3600)
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

@cache_data(ttl=86400)
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


# ============================================================
# BLS — Bureau of Labor Statistics
# ============================================================
# No key needed (v1); add BLS_API_KEY to .env for v2 (20-yr history).
# Series used:
#   CUSR0000SA0   CPI-U All Items (SA)
#   CUSR0000SA0L1E  Core CPI (ex food & energy, SA)
#   WPSFD4111     PPI Final Demand
#   CES0000000001 Total Nonfarm Payrolls (thousands)
#   LNS14000000   Unemployment Rate

BLS_SERIES = {
    "cpi":          "CUSR0000SA0",
    "core_cpi":     "CUSR0000SA0L1E",
    "ppi":          "WPSFD4111",
    "payrolls":     "CES0000000001",
    "unemployment": "LNS14000000",
}


@cache_data(ttl=86400)
def fetch_bls_series(series_ids: tuple, start_year: int | None = None,
                     end_year: int | None = None) -> dict[str, pd.DataFrame]:
    """Fetch one or more BLS series. Returns {series_id: DataFrame(Date, Value)}."""
    from datetime import datetime as _dt
    if start_year is None:
        start_year = _dt.now().year - (20 if BLS_API_KEY else 10)
    if end_year is None:
        end_year = _dt.now().year

    ver = "v2" if BLS_API_KEY else "v1"
    payload: dict = {
        "seriesid": list(series_ids),
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if BLS_API_KEY:
        payload["registrationkey"] = BLS_API_KEY

    try:
        r = requests.post(
            f"https://api.bls.gov/publicAPI/{ver}/timeseries/data/",
            json=payload,
            timeout=20,
        )
        data = r.json()
    except Exception:
        return {}

    if data.get("status") != "REQUEST_SUCCEEDED":
        return {}

    result: dict[str, pd.DataFrame] = {}
    for series in data.get("Results", {}).get("series", []):
        sid = series.get("seriesID", "")
        rows = []
        for obs in series.get("data", []):
            period = obs.get("period", "")
            if not period.startswith("M") or period == "M13":
                continue
            try:
                date = pd.Timestamp(year=int(obs["year"]),
                                    month=int(period[1:]), day=1)
                rows.append({"Date": date, "Value": float(obs["value"])})
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows).set_index("Date").sort_index()
            result[sid] = df
    return result


@cache_data(ttl=86400)
def fetch_bls_indicators() -> dict:
    """Return key BLS indicators with time-series and latest-print summary."""
    raw = fetch_bls_series(tuple(BLS_SERIES.values()))
    inv = {v: k for k, v in BLS_SERIES.items()}
    dfs = {inv[sid]: df for sid, df in raw.items() if sid in inv}
    latest: dict = {}

    def _yoy(df: pd.DataFrame, periods: int = 12) -> float | None:
        if df is None or len(df) < periods + 1:
            return None
        return round((df["Value"].iloc[-1] / df["Value"].iloc[-periods - 1] - 1) * 100, 2)

    if "cpi" in dfs:
        latest["cpi_yoy"]   = _yoy(dfs["cpi"])
        latest["cpi_date"]  = dfs["cpi"].index[-1].strftime("%B %Y")
    if "core_cpi" in dfs:
        latest["core_cpi_yoy"] = _yoy(dfs["core_cpi"])
    if "ppi" in dfs:
        latest["ppi_yoy"]  = _yoy(dfs["ppi"])
        latest["ppi_date"] = dfs["ppi"].index[-1].strftime("%B %Y")
    if "payrolls" in dfs and len(dfs["payrolls"]) >= 2:
        pay = dfs["payrolls"]
        latest["payrolls_mom"]  = round(pay["Value"].iloc[-1] - pay["Value"].iloc[-2], 0)
        latest["payrolls_date"] = pay.index[-1].strftime("%B %Y")
    if "unemployment" in dfs:
        unemp = dfs["unemployment"]
        latest["unemployment"]      = round(unemp["Value"].iloc[-1], 1)
        latest["unemployment_date"] = unemp.index[-1].strftime("%B %Y")

    return {"series": dfs, "latest": latest}


# ============================================================
# BEA — Bureau of Economic Analysis
# ============================================================
# Requires a free API key: https://apps.bea.gov/API/signup/
# Add BEA_API_KEY=... to .env

@cache_data(ttl=86400)
def _fetch_bea_nipa(table_name: str, frequency: str = "Q") -> pd.DataFrame | None:
    """Raw BEA NIPA data as a DataFrame of all lines."""
    if not BEA_API_KEY:
        return None
    try:
        r = requests.get(
            "https://apps.bea.gov/api/data/",
            params={
                "UserID":      BEA_API_KEY,
                "method":      "GetData",
                "datasetname": "NIPA",
                "TableName":   table_name,
                "Frequency":   frequency,
                "Year":        "ALL",
                "ResultFormat": "JSON",
            },
            timeout=20,
        )
        payload = r.json()
    except Exception:
        return None

    rows = (payload.get("BEAAPI", {})
                   .get("Results", {})
                   .get("Data", []))
    if not rows or not isinstance(rows, list):
        return None
    return pd.DataFrame(rows)


def _bea_parse_timeseries(df_raw: pd.DataFrame, line_num: str = "1") -> pd.DataFrame | None:
    """Extract a single line from a raw BEA NIPA DataFrame into (Date, Value)."""
    if df_raw is None or df_raw.empty:
        return None
    subset = df_raw[df_raw.get("LineNumber", pd.Series(dtype=str)) == line_num].copy()
    if subset.empty:
        return None
    rows = []
    for _, row in subset.iterrows():
        tp = str(row.get("TimePeriod", ""))
        try:
            if "M" in tp:
                date = pd.Timestamp(year=int(tp[:4]), month=int(tp[5:7]), day=1)
            elif "Q" in tp:
                q = int(tp[5])
                date = pd.Timestamp(year=int(tp[:4]), month=(q - 1) * 3 + 1, day=1)
            else:
                continue
            val = float(str(row.get("DataValue", "")).replace(",", ""))
            rows.append({"Date": date, "Value": val})
        except Exception:
            continue
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("Date").sort_index()


@cache_data(ttl=86400)
def fetch_bea_gdp() -> dict:
    """GDP percent change from preceding quarter (annualized).
    Source: BEA NIPA Table 1.1.1, Line 1.
    """
    raw = _fetch_bea_nipa("T10101", "Q")
    df = _bea_parse_timeseries(raw, "1")
    if df is None:
        return {}
    latest_val  = df["Value"].iloc[-1]
    latest_date = df.index[-1]
    q_label     = f"Q{(latest_date.month - 1) // 3 + 1} {latest_date.year}"
    return {"series": df, "latest": round(latest_val, 2), "latest_date": q_label}


@cache_data(ttl=86400)
def fetch_bea_pce() -> dict:
    """PCE price index — the Fed's preferred inflation gauge.
    Source: BEA NIPA Table 2.8.4 (quarterly price indexes), Line 1.
    Computes YoY % change from the index.
    """
    raw = _fetch_bea_nipa("T20804", "Q")
    df = _bea_parse_timeseries(raw, "1")
    if df is None:
        return {}
    yoy = None
    if len(df) >= 5:
        yoy = round((df["Value"].iloc[-1] / df["Value"].iloc[-5] - 1) * 100, 2)
    latest_date = df.index[-1]
    q_label     = f"Q{(latest_date.month - 1) // 3 + 1} {latest_date.year}"
    return {"series": df, "latest_yoy": yoy, "latest_date": q_label}


def has_bea() -> bool:
    return bool((BEA_API_KEY or "").strip())


# ============================================================
# U.S. Treasury — Fiscal Data API (no key required)
# ============================================================

@cache_data(ttl=3600)
def fetch_treasury_debt() -> dict:
    """National debt (total public debt outstanding) from Treasury Fiscal Data API.
    Returns latest value in trillions + a 30-point time series.
    """
    try:
        r = requests.get(
            "https://api.fiscaldata.treasury.gov/services/api/v1/"
            "accounting/od/debt_to_penny",
            params={
                "fields":      "record_date,tot_pub_debt_out_amt",
                "sort":        "-record_date",
                "page[size]":  "60",
            },
            timeout=12,
        )
        data = r.json()
    except Exception:
        return {}

    rows = []
    for item in data.get("data", []):
        try:
            date = pd.Timestamp(item["record_date"])
            amt  = float(item["tot_pub_debt_out_amt"]) / 1e12   # → trillions
            rows.append({"Date": date, "Value": amt})
        except Exception:
            continue
    if not rows:
        return {}
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return {
        "series":          df,
        "latest_trillions": round(df["Value"].iloc[-1], 2),
        "latest_date":     df.index[-1].strftime("%B %d, %Y"),
    }


@cache_data(ttl=3600)
def fetch_yield_curve() -> pd.DataFrame | None:
    """Today's Treasury yield curve (9 maturities) as DataFrame(Maturity, Yield).

    Priority order:
      1. FRED (if FRED_API_KEY is set) — most reliable
      2. Treasury.gov XML API — no key needed
    Returns None if both sources fail.
    """
    MATURITIES = [
        ("1M",  "DGS1MO"),  ("3M",  "DGS3MO"), ("6M",  "DGS6MO"),
        ("1Y",  "DGS1"),    ("2Y",  "DGS2"),    ("5Y",  "DGS5"),
        ("10Y", "DGS10"),   ("20Y", "DGS20"),   ("30Y", "DGS30"),
    ]
    ORDER = [m for m, _ in MATURITIES]

    points: list[dict] = []

    # ---- FRED path ----
    if fred:
        def _get_one(label_series):
            label, sid = label_series
            try:
                s = fred.get_series(sid)
                if s is not None and not s.empty:
                    val = float(s.dropna().iloc[-1])
                    return {"Maturity": label, "Yield": round(val, 3)}
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=9) as pool:
            for result in pool.map(_get_one, MATURITIES):
                if result:
                    points.append(result)

    # ---- Treasury.gov XML fallback ----
    if not points:
        try:
            from datetime import datetime as _dt
            year = _dt.now().year
            r = requests.get(
                "https://home.treasury.gov/resource-center/data-chart-center/"
                f"interest-rates/pages/xml?data=daily_treasury_yield_curve"
                f"&field_tdr_date_value={year}",
                timeout=12,
            )
            root = ET.fromstring(r.content)
            _NS = "http://schemas.microsoft.com/ado/2007/08/dataservices"
            _ATOM = "http://www.w3.org/2005/Atom"
            entries = root.findall(f".//{{{_ATOM}}}entry")
            if entries:
                props = entries[-1].find(
                    f".//{{{_NS[:-len('dataservices')]}"
                    f"dataservices/metadata}}properties"
                )
                if props is None:
                    # Try alternate namespace pattern
                    props = entries[-1].find(
                        ".//{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}properties"
                    )
                field_map = {
                    "BC_1MONTH": "1M",  "BC_3MONTH": "3M",  "BC_6MONTH": "6M",
                    "BC_1YEAR":  "1Y",  "BC_2YEAR":  "2Y",  "BC_5YEAR":  "5Y",
                    "BC_10YEAR": "10Y", "BC_20YEAR": "20Y", "BC_30YEAR": "30Y",
                }
                if props is not None:
                    for field, label in field_map.items():
                        el = props.find(f"{{{_NS}}}{field}")
                        if el is not None and el.text:
                            try:
                                points.append({"Maturity": label,
                                               "Yield": round(float(el.text), 3)})
                            except Exception:
                                pass
        except Exception:
            pass

    if not points:
        return None

    df = pd.DataFrame(points)
    order_map = {m: i for i, m in enumerate(ORDER)}
    df["_ord"] = df["Maturity"].map(order_map)
    return df.sort_values("_ord").drop("_ord", axis=1).reset_index(drop=True)


# ============================================================
# SimFin — Standardised Financial Statements
# ============================================================
# Free API key: https://app.simfin.com  (Settings → API Key)
# Add SIMFIN_API_KEY=... to .env
# Free tier: 2,000 calls/day, 1 req/sec.  All US-listed companies.
# Uses the REST API directly — no bulk CSV downloads, zero disk usage.

_SIMFIN_BASE = "https://backend.simfin.com/api/v3"


def has_simfin() -> bool:
    return bool((SIMFIN_API_KEY or "").strip())


def _simfin_get(endpoint: str, params: dict) -> dict | list | None:
    """Single authenticated GET against the SimFin v3 API."""
    if not has_simfin():
        return None
    try:
        r = requests.get(
            f"{_SIMFIN_BASE}/{endpoint.lstrip('/')}",
            params=params,
            headers={"Authorization": f"api-key {SIMFIN_API_KEY.strip()}"},
            timeout=15,
        )
        if r.status_code == 401:
            return None   # bad key
        if r.status_code == 404:
            return None   # ticker not found
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _sf_compact_to_df(raw: list | dict, stmt_type: str,
                      period_filter: str | None = "FY") -> pd.DataFrame | None:
    """Convert a SimFin compact-format API response into a tidy DataFrame.

    ``raw`` may be the full top-level list (one element per ticker) or the
    ``statements`` sub-list for a single ticker.
    """
    # Unwrap single-ticker envelope if needed
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "statements" in raw[0]:
        stmts = raw[0].get("statements", [])
    elif isinstance(raw, list):
        stmts = raw
    else:
        return None

    for stmt in stmts:
        # API returns uppercase ('PL', 'BS', 'CF', 'DERIVED') — compare case-insensitively
        if stmt.get("statement", "").upper() != stmt_type.upper():
            continue
        cols = stmt.get("columns", [])
        data = stmt.get("data", [])
        if not cols or not data:
            return None
        df = pd.DataFrame(data, columns=cols)
        # Filter to annual FY rows (skip Q1/Q2/Q3/Q4 when period_filter set)
        if period_filter and "Fiscal Period" in df.columns:
            df = df[df["Fiscal Period"] == period_filter].copy()
        if "Fiscal Year" in df.columns:
            df = df.sort_values("Fiscal Year", ascending=False)
        return df.reset_index(drop=True)
    return None


@cache_data(ttl=86400)
def fetch_simfin_statements(ticker: str, period: str = "annual") -> dict[str, pd.DataFrame | None]:
    """Return a dict with keys "income", "balance", "cashflow", "derived".

    Each value is a DataFrame (most recent year first) or None if unavailable.
    ``period`` may be "annual", "quarterly", or "ttm".
    """
    if not has_simfin():
        return {}

    # SimFin v3 uses "fy" for annual, "ttm" for TTM (not "annual")
    sf_period = {"annual": "fy"}.get(period, period)

    raw = _simfin_get(
        "companies/statements/compact",
        {"ticker": ticker.upper(), "statements": "pl,bs,cf,derived",
         "period": sf_period},
    )
    if not raw:
        return {}

    # period_filter: fy → "FY" rows only, ttm → "TTM" rows only
    pf = {"fy": "FY", "ttm": "TTM"}.get(sf_period)

    return {
        "income":   _sf_compact_to_df(raw, "pl",      period_filter=pf),
        "balance":  _sf_compact_to_df(raw, "bs",      period_filter=pf),
        "cashflow": _sf_compact_to_df(raw, "cf",      period_filter=pf),
        "derived":  _sf_compact_to_df(raw, "derived", period_filter=pf),
    }


@cache_data(ttl=86400)
def fetch_simfin_ttm_fcf(ticker: str) -> tuple[float | None, float | None]:
    """Return (ttm_free_cash_flow, diluted_shares) for use in DCF.

    Prefers the "derived" statement's "Free Cash Flow" field; falls back to
    operating_cf − capex from the cash flow statement.
    Returns (None, None) if SimFin isn't configured or ticker isn't found.
    """
    if not has_simfin():
        return (None, None)

    stmts = fetch_simfin_statements(ticker, period="ttm")
    fcf: float | None = None
    shares: float | None = None

    # --- FCF from derived statement ---
    derived = stmts.get("derived")
    if derived is not None and not derived.empty:
        for col in ("Free Cash Flow", "Free Cash Flow (TTM)"):
            if col in derived.columns:
                v = derived[col].iloc[0]
                if v is not None and str(v) not in ("", "None", "nan"):
                    try:
                        fcf = float(v)
                        break
                    except Exception:
                        pass

    # --- FCF fallback: Op CF − Capex ---
    if fcf is None:
        cf = stmts.get("cashflow")
        if cf is not None and not cf.empty:
            op_cf = capex = None
            for col in ("Net Cash from Operating Activities", "Operating Cash Flow"):
                if col in cf.columns:
                    try:
                        op_cf = float(cf[col].iloc[0])
                        break
                    except Exception:
                        pass
            for col in ("Capital Expenditures", "Purchases of Property, Plant & Equipment"):
                if col in cf.columns:
                    try:
                        capex = float(cf[col].iloc[0])
                        break
                    except Exception:
                        pass
            if op_cf is not None and capex is not None:
                fcf = op_cf + capex   # capex is usually negative in SimFin

    # --- Shares from income statement ---
    inc = stmts.get("income")
    if inc is not None and not inc.empty:
        for col in ("Shares (Diluted)", "Shares (Basic)"):
            if col in inc.columns:
                try:
                    shares = float(inc[col].iloc[0])
                    break
                except Exception:
                    pass

    return (fcf, shares)


def _fmt_millions(val) -> str:
    """Format a raw dollar value (in ones) to a readable string."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


def _pct(num, denom) -> str:
    try:
        return f"{float(num)/float(denom)*100:.1f}%"
    except Exception:
        return "—"


def build_simfin_income_table(df: pd.DataFrame, n_years: int = 6) -> pd.DataFrame | None:
    """Build a display-ready income statement table (metrics × years)."""
    if df is None or df.empty:
        return None
    df = df.head(n_years)
    years = df["Fiscal Year"].tolist() if "Fiscal Year" in df.columns else list(range(len(df)))

    def _get(col):
        if col in df.columns:
            return df[col].tolist()
        return [None] * len(df)

    rev  = _get("Revenue")
    gp   = _get("Gross Profit")
    oi   = _get("Operating Income (Loss)")
    ni   = _get("Net Income")

    rows = {
        "Revenue":           [_fmt_millions(v) for v in rev],
        "Gross Profit":      [_fmt_millions(v) for v in gp],
        "Gross Margin":      [_pct(g, r) for g, r in zip(gp, rev)],
        "Operating Income":  [_fmt_millions(v) for v in oi],
        "Op. Margin":        [_pct(o, r) for o, r in zip(oi, rev)],
        "Net Income":        [_fmt_millions(v) for v in ni],
        "Net Margin":        [_pct(n, r) for n, r in zip(ni, rev)],
    }
    return pd.DataFrame(rows, index=[str(y) for y in years]).T


def build_simfin_balance_table(df: pd.DataFrame, n_years: int = 6) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.head(n_years)
    years = df["Fiscal Year"].tolist() if "Fiscal Year" in df.columns else list(range(len(df)))

    def _get(col):
        return df[col].tolist() if col in df.columns else [None] * len(df)

    cash   = _get("Cash, Cash Equivalents & Short Term Investments")
    assets = _get("Total Assets")
    liab   = _get("Total Liabilities")
    equity = _get("Total Equity")

    rows = {
        "Cash & Short-term Investments": [_fmt_millions(v) for v in cash],
        "Total Assets":                  [_fmt_millions(v) for v in assets],
        "Total Liabilities":             [_fmt_millions(v) for v in liab],
        "Total Equity":                  [_fmt_millions(v) for v in equity],
        "Debt / Equity":                 [_pct(l, e).replace("%","x") if e else "—"
                                          for l, e in zip(liab, equity)],
    }
    return pd.DataFrame(rows, index=[str(y) for y in years]).T


def build_simfin_cashflow_table(df: pd.DataFrame, n_years: int = 6) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.head(n_years)
    years = df["Fiscal Year"].tolist() if "Fiscal Year" in df.columns else list(range(len(df)))

    def _get(col):
        return df[col].tolist() if col in df.columns else [None] * len(df)

    op_cf = _get("Net Cash from Operating Activities")
    capex = _get("Capital Expenditures")
    # FCF = Op CF + Capex (capex is negative)
    fcf_raw = []
    for o, c in zip(op_cf, capex):
        try:
            fcf_raw.append(float(o) + float(c))
        except Exception:
            fcf_raw.append(None)
    div   = _get("Dividends Paid")

    # Get revenue from income statement if needed for FCF margin — skip for now.
    rows = {
        "Operating Cash Flow":  [_fmt_millions(v) for v in op_cf],
        "Capital Expenditures": [_fmt_millions(v) for v in capex],
        "Free Cash Flow":       [_fmt_millions(v) for v in fcf_raw],
        "Dividends Paid":       [_fmt_millions(v) for v in div],
    }
    return pd.DataFrame(rows, index=[str(y) for y in years]).T


# ============================================================
# CNN Fear & Greed Index  (no key required)
# ============================================================

@cache_data(ttl=3600)
def fetch_fear_greed() -> dict:
    """CNN Fear & Greed composite sentiment (0–100).
    Returns {"score": float, "rating": str, "history": DataFrame|None}
    """
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": _DEFAULT_UA}, timeout=8,
        )
        data = r.json()
        fg     = data.get("fear_and_greed", {})
        score  = fg.get("score")
        rating = (fg.get("rating") or "").replace("_", " ").title()
        hist_rows = []
        for item in data.get("fear_and_greed_historical", {}).get("data", []):
            try:
                hist_rows.append({"Date": pd.Timestamp(item["x"], unit="ms"),
                                   "Score": float(item["y"])})
            except Exception:
                continue
        hist_df = (pd.DataFrame(hist_rows).set_index("Date").sort_index()
                   if hist_rows else None)
        return {"score": round(float(score), 1) if score is not None else None,
                "rating": rating or None, "history": hist_df}
    except Exception:
        return {}


# ============================================================
# CoinGecko — Crypto Market Data  (no key required)
# ============================================================

@cache_data(ttl=120)
def fetch_coingecko_global() -> dict:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global",
                         headers={"User-Agent": _DEFAULT_UA}, timeout=8)
        d = r.json().get("data", {})
        return {
            "total_market_cap_usd":  d.get("total_market_cap", {}).get("usd"),
            "total_volume_usd":      d.get("total_volume", {}).get("usd"),
            "btc_dominance":         d.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance":         d.get("market_cap_percentage", {}).get("eth"),
            "market_cap_change_24h": d.get("market_cap_change_percentage_24h_usd"),
        }
    except Exception:
        return {}


@cache_data(ttl=120)
def fetch_coingecko_top_coins(n: int = 10) -> pd.DataFrame | None:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": n, "page": 1, "sparkline": False,
                    "price_change_percentage": "24h,7d"},
            headers={"User-Agent": _DEFAULT_UA}, timeout=8,
        )
        rows = [{"Coin": f"{c.get('name')} ({(c.get('symbol') or '').upper()})",
                 "Price": c.get("current_price"),
                 "24h %": c.get("price_change_percentage_24h"),
                 "7d %":  c.get("price_change_percentage_7d_in_currency"),
                 "Mkt Cap": c.get("market_cap"),
                 "Vol 24h": c.get("total_volume")}
                for c in r.json()]
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


# ============================================================
# Commodities — Gold, Silver, Oil, Gas, Copper  (yfinance)
# ============================================================

@cache_data(ttl=60)
def fetch_commodity_prices() -> dict[str, dict]:
    SYMBOLS = {"Gold": "GC=F", "Silver": "SI=F",
               "Crude Oil (WTI)": "CL=F", "Natural Gas": "NG=F", "Copper": "HG=F"}
    result: dict[str, dict] = {}
    for name, sym in SYMBOLS.items():
        try:
            fast  = yf.Ticker(sym).fast_info
            price = fast["last_price"]
            prev  = fast["previous_close"]
            chg   = round((price - prev) / prev * 100, 2) if prev else 0.0
            result[name] = {"price": round(price, 2), "change": chg, "symbol": sym}
        except Exception:
            result[name] = {"price": None, "change": None, "symbol": sym}
    return result


# ============================================================
# EIA — Energy Information Administration
# ============================================================
# Free API key: https://www.eia.gov/opendata/register.php
# Add EIA_API_KEY=... to .env

_EIA_BASE = "https://api.eia.gov/v2"

# Key weekly petroleum / natural-gas series
EIA_SERIES = {
    "crude_stocks":     "PET.WCRSTUS1.W",
    "gasoline_stocks":  "PET.WGTSTUS1.W",
    "distillate_stocks":"PET.WDISTUS1.W",
    "natgas_storage":   "NG.NW2_EPG0_SWO_R48_BCF.W",
    "crude_production": "PET.MCRFPUS2.M",
}


@cache_data(ttl=86400)
def fetch_eia_series(series_id: str, length: int = 52) -> pd.DataFrame | None:
    """Fetch a single EIA v2 series. Returns DataFrame(Date, Value) or None."""
    if not has_eia():
        return None
    try:
        r = requests.get(
            f"{_EIA_BASE}/seriesid/{series_id}",
            params={"api_key": EIA_API_KEY, "frequency": "weekly",
                    "data[0]": "value", "sort[0][column]": "period",
                    "sort[0][direction]": "desc", "length": length},
            timeout=12,
        )
        rows_raw = r.json().get("response", {}).get("data", [])
        rows = []
        for item in rows_raw:
            try:
                rows.append({"Date": pd.Timestamp(item["period"]),
                              "Value": float(item["value"])})
            except Exception:
                continue
        if not rows:
            return None
        return pd.DataFrame(rows).set_index("Date").sort_index()
    except Exception:
        return None


@cache_data(ttl=86400)
def fetch_eia_snapshot() -> dict:
    """Return latest prints for key EIA energy series."""
    if not has_eia():
        return {}
    result: dict = {"series": {}, "latest": {}}
    labels = {
        "crude_stocks":      ("Crude Oil Stocks",      "M bbl"),
        "gasoline_stocks":   ("Gasoline Stocks",        "M bbl"),
        "distillate_stocks": ("Distillate Fuel Stocks", "M bbl"),
        "natgas_storage":    ("Natural Gas Storage",    "Bcf"),
        "crude_production":  ("US Crude Production",   "Mb/d"),
    }
    for key, series_id in EIA_SERIES.items():
        df = fetch_eia_series(series_id)
        if df is not None and not df.empty:
            result["series"][key] = df
            label, unit = labels[key]
            val = df["Value"].iloc[-1]
            date_str = df.index[-1].strftime("%b %d, %Y")
            # WoW change
            chg = None
            if len(df) >= 2:
                prev = df["Value"].iloc[-2]
                chg = round(val - prev, 1)
            result["latest"][key] = {
                "label": label, "unit": unit,
                "value": round(val, 1), "date": date_str, "wow_change": chg,
            }
    return result


# ============================================================
# FMP — Financial Modeling Prep
# ============================================================
# Free API key: https://financialmodelingprep.com/developer/docs
# 250 calls/day on free tier. Add FMP_API_KEY=... to .env

_FMP_BASE = "https://financialmodelingprep.com/api"


def _fmp_get(endpoint: str, params: dict | None = None) -> list | dict | None:
    if not has_fmp():
        return None
    try:
        p = {"apikey": FMP_API_KEY}
        if params:
            p.update(params)
        r = requests.get(f"{_FMP_BASE}/{endpoint.lstrip('/')}", params=p, timeout=12)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


@cache_data(ttl=86400)
def fetch_fmp_profile(ticker: str) -> dict:
    """Company profile: description, sector, CEO, employees, exchange."""
    data = _fmp_get(f"v3/profile/{ticker.upper()}")
    if not data or not isinstance(data, list):
        return {}
    return data[0] if data else {}


@cache_data(ttl=86400)
def fetch_fmp_price_targets(ticker: str) -> dict:
    """Latest analyst price targets: low / avg / high / consensus."""
    data = _fmp_get("v4/price-target-consensus", {"symbol": ticker.upper()})
    if not data or not isinstance(data, list):
        return {}
    return data[0] if data else {}


@cache_data(ttl=86400)
def fetch_fmp_analyst_estimates(ticker: str) -> pd.DataFrame | None:
    """Forward EPS and revenue analyst estimates (next 4 quarters + 2 years)."""
    data = _fmp_get(f"v3/analyst-estimates/{ticker.upper()}", {"limit": 8})
    if not data or not isinstance(data, list):
        return None
    rows = []
    for item in data:
        rows.append({
            "Period":          item.get("date", ""),
            "Est. Revenue":    item.get("estimatedRevenueAvg"),
            "Est. EPS":        item.get("estimatedEpsAvg"),
            "EPS Low":         item.get("estimatedEpsLow"),
            "EPS High":        item.get("estimatedEpsHigh"),
            "# Analysts":      item.get("numberAnalystEstimatedEps"),
        })
    return pd.DataFrame(rows) if rows else None


@cache_data(ttl=86400)
def fetch_fmp_ratings(ticker: str) -> dict:
    """FMP composite DCF + quantitative rating for a ticker."""
    data = _fmp_get(f"v3/rating/{ticker.upper()}")
    if not data or not isinstance(data, list):
        return {}
    return data[0] if data else {}


# ============================================================
# Alpha Vantage — Technical Indicators
# ============================================================
# Free key: https://www.alphavantage.co/support/#api-key  (25 calls/day)
# Add ALPHA_VANTAGE_KEY=... to .env
# Cache TTL = 24h to stay well within the daily quota.

_AV_BASE = "https://www.alphavantage.co/query"


def _av_get(params: dict) -> dict | None:
    if not has_alpha_vantage():
        return None
    try:
        p = {"apikey": ALPHA_VANTAGE_KEY}
        p.update(params)
        r = requests.get(_AV_BASE, params=p, timeout=15)
        data = r.json()
        # AV signals errors as {"Note": "..."} or {"Information": "..."}
        if "Note" in data or "Information" in data:
            return None
        return data
    except Exception:
        return None


@cache_data(ttl=86400)
def fetch_av_rsi(ticker: str, period: int = 14) -> pd.DataFrame | None:
    """RSI(period) daily values for ticker. Returns DataFrame(Date, RSI)."""
    data = _av_get({"function": "RSI", "symbol": ticker.upper(),
                    "interval": "daily", "time_period": period,
                    "series_type": "close"})
    if not data:
        return None
    raw = data.get("Technical Analysis: RSI", {})
    if not raw:
        return None
    rows = [{"Date": pd.Timestamp(d), "RSI": float(v["RSI"])}
            for d, v in raw.items()]
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df.tail(252)   # ~1 year of daily values


@cache_data(ttl=86400)
def fetch_av_macd(ticker: str) -> pd.DataFrame | None:
    """MACD (12,26,9) daily values. Returns DataFrame(Date, MACD, Signal, Hist)."""
    data = _av_get({"function": "MACD", "symbol": ticker.upper(),
                    "interval": "daily", "series_type": "close"})
    if not data:
        return None
    raw = data.get("Technical Analysis: MACD", {})
    if not raw:
        return None
    rows = [{"Date": pd.Timestamp(d),
             "MACD": float(v["MACD"]),
             "Signal": float(v["MACD_Signal"]),
             "Histogram": float(v["MACD_Hist"])}
            for d, v in raw.items()]
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df.tail(252)


@cache_data(ttl=86400)
def fetch_av_bbands(ticker: str, period: int = 20) -> pd.DataFrame | None:
    """Bollinger Bands (period, 2σ) daily. Returns DataFrame(Date, Upper, Middle, Lower)."""
    data = _av_get({"function": "BBANDS", "symbol": ticker.upper(),
                    "interval": "daily", "time_period": period,
                    "series_type": "close", "nbdevup": 2, "nbdevdn": 2})
    if not data:
        return None
    raw = data.get("Technical Analysis: BBANDS", {})
    if not raw:
        return None
    rows = [{"Date": pd.Timestamp(d),
             "Upper": float(v["Real Upper Band"]),
             "Middle": float(v["Real Middle Band"]),
             "Lower": float(v["Real Lower Band"])}
            for d, v in raw.items()]
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df.tail(252)


@cache_data(ttl=86400)
def fetch_av_indicators(ticker: str) -> dict:
    """Fetch RSI + MACD + BBands in one cached call to minimise quota usage.
    Returns {"rsi": df, "macd": df, "bbands": df, "latest": {...}}
    """
    rsi    = fetch_av_rsi(ticker)
    macd   = fetch_av_macd(ticker)
    bbands = fetch_av_bbands(ticker)

    latest: dict = {}
    if rsi is not None and not rsi.empty:
        latest["rsi"]      = round(rsi["RSI"].iloc[-1], 2)
        latest["rsi_date"] = rsi.index[-1].strftime("%b %d")
    if macd is not None and not macd.empty:
        latest["macd"]        = round(macd["MACD"].iloc[-1], 4)
        latest["macd_signal"] = round(macd["Signal"].iloc[-1], 4)
        latest["macd_hist"]   = round(macd["Histogram"].iloc[-1], 4)
    if bbands is not None and not bbands.empty:
        latest["bb_upper"]  = round(bbands["Upper"].iloc[-1], 2)
        latest["bb_middle"] = round(bbands["Middle"].iloc[-1], 2)
        latest["bb_lower"]  = round(bbands["Lower"].iloc[-1], 2)

    return {"rsi": rsi, "macd": macd, "bbands": bbands, "latest": latest}


# ============================================================
# CFTC — Commitments of Traders  (no key required)
# ============================================================
# Weekly report released every Friday afternoon (data for prior Tuesday).
# Source: CFTC Socrata Open Data API — dataset 6dca-aqww
#   "Futures-Only Commitments of Traders (Legacy)"
# No authentication required.

_CFTC_BASE = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# Market names exactly as they appear in the CFTC dataset
CFTC_MARKETS: dict[str, str] = {
    "gold":       "GOLD - COMMODITY EXCHANGE INC.",
    "crude_oil":  "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    "sp500":      "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "10yr_tnote": "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    "bitcoin":    "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
    "eur_usd":    "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "nat_gas":    "NATURAL GAS (HENRY HUB) - NEW YORK MERCANTILE EXCHANGE",
}

CFTC_LABELS: dict[str, str] = {
    "gold":       "Gold",
    "crude_oil":  "Crude Oil (WTI)",
    "sp500":      "S&P 500 E-Mini",
    "10yr_tnote": "10-Yr T-Note",
    "bitcoin":    "Bitcoin",
    "eur_usd":    "EUR/USD",
    "nat_gas":    "Natural Gas",
}


@cache_data(ttl=86400)
def fetch_cftc_cot(market_key: str, weeks: int = 52) -> pd.DataFrame | None:
    """Fetch CFTC Legacy CoT (futures-only) for *market_key* (see CFTC_MARKETS).

    Returns a DataFrame indexed by Date with columns:
    NonComm_Long, NonComm_Short, NonComm_Net, Comm_Long, Comm_Short,
    Comm_Net, NonRept_Long, NonRept_Short, OI.
    Returns None if the market key is unknown or the fetch fails.
    """
    market_name = CFTC_MARKETS.get(market_key)
    if not market_name:
        return None
    try:
        r = requests.get(
            _CFTC_BASE,
            params={
                "market_and_exchange_names": market_name,
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": weeks,
            },
            headers={"User-Agent": _DEFAULT_UA},
            timeout=20,
        )
        items = r.json()
        if not items or not isinstance(items, list):
            return None
        rows = []
        for item in items:
            try:
                rows.append({
                    "Date":          pd.Timestamp(item["report_date_as_yyyy_mm_dd"]),
                    "NonComm_Long":  int(float(item.get("noncomm_positions_long_all",  0))),
                    "NonComm_Short": int(float(item.get("noncomm_positions_short_all", 0))),
                    "Comm_Long":     int(float(item.get("comm_positions_long_all",     0))),
                    "Comm_Short":    int(float(item.get("comm_positions_short_all",    0))),
                    "NonRept_Long":  int(float(item.get("nonrept_positions_long_all",  0))),
                    "NonRept_Short": int(float(item.get("nonrept_positions_short_all", 0))),
                    "OI":            int(float(item.get("open_interest_all",           0))),
                })
            except Exception:
                continue
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("Date").sort_index()
        df["NonComm_Net"] = df["NonComm_Long"]  - df["NonComm_Short"]
        df["Comm_Net"]    = df["Comm_Long"]      - df["Comm_Short"]
        return df
    except Exception:
        return None


@cache_data(ttl=86400)
def fetch_cftc_snapshot() -> dict:
    """Return latest + WoW-change CoT positioning for all CFTC_MARKETS.

    Each key maps to:
    {"label", "date", "nc_long", "nc_short", "nc_net", "nc_chg", "oi"}
    """
    result: dict = {}
    for key, label in CFTC_LABELS.items():
        df = fetch_cftc_cot(key, weeks=3)  # need ≥ 2 rows for WoW change
        if df is None or df.empty:
            continue
        latest = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) >= 2 else None
        nc_net = int(latest["NonComm_Net"])
        chg    = int(nc_net - int(prev["NonComm_Net"])) if prev is not None else 0
        result[key] = {
            "label":    label,
            "date":     df.index[-1].strftime("%b %d, %Y"),
            "nc_long":  int(latest["NonComm_Long"]),
            "nc_short": int(latest["NonComm_Short"]),
            "nc_net":   nc_net,
            "nc_chg":   chg,
            "oi":       int(latest["OI"]),
        }
    return result


# ============================================================
# Monetary catalyst calendars — FOMC + Treasury refunding
# ============================================================
# These are the published dates for 2026/2027 from the Fed and Treasury.
# No clean public ICS or JSON feed exists for the FOMC schedule (the Fed
# publishes only HTML), so we maintain hardcoded lists with annual updates.
# The Catalyst Calendar's "import" path dedupes on (source, event_date) so
# re-running the import after each year's update is safe.
#
# IMPORTANT: dates are accurate as of the most recent Fed / Treasury
# announcement, but should be re-verified against:
#   • https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#   • https://www.treasurydirect.gov/instit/annceresult/press/preanre/preanre.htm
# before sizing trades around them.

import datetime as _dt_mod


def _ts(year: int, month: int, day: int) -> int:
    """Local-noon Unix timestamp for a calendar date — gives a consistent
    'this happens on this day' anchor without TZ ambiguity."""
    return int(_dt_mod.datetime(year, month, day, 12, 0, 0).timestamp())


def get_fomc_schedule() -> list[dict]:
    """Hardcoded FOMC meeting + SEP-release schedule.

    Each entry returns the second day of the meeting (when the rate
    announcement and press conference happen). The `has_sep` flag marks
    the four meetings per year that release a Summary of Economic
    Projections (Mar / Jun / Sep / Dec).

    Returns dicts shaped for direct ingestion into political_catalysts:
      {event_date, title, catalyst_type, category, stakes, source}
    """
    # (year, decision_month, decision_day, has_sep)
    meetings = [
        # 2026
        (2026,  1, 28, False),
        (2026,  3, 18, True),
        (2026,  4, 29, False),
        (2026,  6, 17, True),
        (2026,  7, 29, False),
        (2026,  9, 16, True),
        (2026, 10, 28, False),
        (2026, 12,  9, True),
        # 2027 (provisional — Fed publishes by mid-2026)
        (2027,  1, 27, False),
        (2027,  3, 17, True),
        (2027,  4, 28, False),
        (2027,  6, 16, True),
        (2027,  7, 28, False),
        (2027,  9, 15, True),
        (2027, 10, 27, False),
        (2027, 12,  8, True),
    ]
    out = []
    for y, m, d, has_sep in meetings:
        title = f"FOMC Decision · {_dt_mod.date(y, m, d).strftime('%b %Y')}"
        if has_sep:
            title += " (with SEP)"
        stakes = (
            "**Setup:** Decision and press conference at 2:00 PM ET.\n\n"
            "**Bull case:** Rate cut or dovish guidance → "
            "TLT / KRE / XHB / utilities pop on lower discount rate.\n\n"
            "**Base case:** Hold + balanced statement → muted reaction, "
            "focus shifts to the next meeting.\n\n"
            "**Bear case:** Hawkish hold or rate hike → 10Y back above "
            "current cycle highs; long duration and rate-sensitive "
            "sectors take the hit."
        )
        if has_sep:
            stakes += (
                "\n\n_SEP release at this meeting — dot-plot revisions can "
                "produce outsized moves vs the rate decision itself._"
            )
        out.append({
            "event_date":    _ts(y, m, d),
            "title":         title,
            "catalyst_type": "monetary",
            "category":      "FOMC",
            "stakes":        stakes,
            "tickers":       "TLT,XLF,KRE,XHB,XLU",
            "sectors":       "Financials,Banks,REITs,Utilities,Homebuilders",
            "source":        "fomc_schedule",
        })
    return out


# ============================================================
# USAspending.gov — federal contract intelligence
# ============================================================
# Public REST API at https://api.usaspending.gov — no key required, no
# authentication, generous rate limits. The /search/spending_by_award/
# endpoint takes a recipient_search_text filter and returns the top N
# contracts by amount over a configurable lookback window.
#
# Used by the Federal Contracts subtab to answer:
#   • "How much federal $ flows to ticker X right now?"
#   • "Which agencies are this company's biggest customers?"
#   • "Which active contracts end in the next 18 months and could
#      become recompete catalysts?"

_USASPENDING_BASE = "https://api.usaspending.gov/api/v2"

# Curated mapping of common federal-contractor tickers to their primary
# USAspending recipient-name search string. The right-hand strings are
# fuzzy-matched (case-insensitive) so corporate-name variants (e.g.
# "RAYTHEON COMPANY" vs "RTX CORPORATION") all match.
FEDERAL_CONTRACTOR_TICKERS: dict[str, str] = {
    # Defense primes
    "LMT":  "LOCKHEED MARTIN",
    "NOC":  "NORTHROP GRUMMAN",
    "GD":   "GENERAL DYNAMICS",
    "RTX":  "RTX CORPORATION",
    "BA":   "BOEING",
    "HII":  "HUNTINGTON INGALLS",
    # Pure-play defense IT / mid-tier
    "LDOS": "LEIDOS",
    "BAH":  "BOOZ ALLEN HAMILTON",
    "SAIC": "SAIC",
    "CACI": "CACI",
    "PLTR": "PALANTIR",
    # Specialty defense
    "AVAV": "AEROVIRONMENT",
    "KTOS": "KRATOS DEFENSE",
    "MRCY": "MERCURY SYSTEMS",
    "TDG":  "TRANSDIGM",
    "AXON": "AXON ENTERPRISE",
    "CW":   "CURTISS-WRIGHT",
    "HEI":  "HEICO",
    # Engineering & infrastructure
    "FLR":  "FLUOR",
    "J":    "JACOBS SOLUTIONS",
    "PWR":  "QUANTA SERVICES",
    # Healthcare federal contractors (Medicare / TRICARE / VA)
    "HUM":  "HUMANA",
    "UNH":  "UNITEDHEALTH",
    # Big-tech federal subsidiaries
    "MSFT": "MICROSOFT",
    "ORCL": "ORACLE",
    "IBM":  "INTERNATIONAL BUSINESS MACHINES",
    "GOOGL":"GOOGLE",
    "AMZN": "AMAZON WEB SERVICES",
}


@cache_data(ttl=86400, show_spinner=False)
def fetch_usaspending_awards(
    recipient_search_text: str,
    lookback_years: int = 2,
    limit: int = 50,
) -> list[dict] | None:
    """Top federal-contract awards for a recipient over the last N years.

    Returns a list of award dicts sorted by Award Amount descending, or
    None on network/API failure (so callers can distinguish 'no data' from
    'no results').
    """
    if not recipient_search_text:
        return None
    end   = _dt_mod.date.today()
    start = end - _dt_mod.timedelta(days=int(lookback_years * 365.25))

    payload = {
        "filters": {
            "recipient_search_text": [recipient_search_text],
            # A=BPA Call · B=Purchase Order · C=Delivery Order · D=Definitive Contract
            "award_type_codes": ["A", "B", "C", "D"],
            "time_period": [{
                "start_date": start.isoformat(),
                "end_date":   end.isoformat(),
            }],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Start Date",
            "End Date",
            "Award Amount",
            "Description",
            "Awarding Agency",
            "Awarding Sub Agency",
            "Contract Award Type",
        ],
        "limit": min(int(limit), 100),
        "page":  1,
        "sort":  "Award Amount",
        "order": "desc",
    }

    try:
        r = requests.post(
            f"{_USASPENDING_BASE}/search/spending_by_award/",
            json=payload, timeout=30,
        )
        r.raise_for_status()
        return r.json().get("results", []) or []
    except Exception:
        return None


@cache_data(ttl=86400, show_spinner=False)
def fetch_usaspending_summary(
    recipient_search_text: str,
    lookback_years: int = 2,
) -> dict | None:
    """Aggregate stats: total $, contract count, and top 5 awarding agencies.

    Built on top of fetch_usaspending_awards so a single API call powers both
    the summary tiles and the contract table.
    """
    awards = fetch_usaspending_awards(
        recipient_search_text, lookback_years=lookback_years, limit=100,
    )
    if awards is None:
        return None
    if not awards:
        return {"total": 0.0, "count": 0, "top_agencies": []}

    total = sum(float(a.get("Award Amount") or 0) for a in awards)
    by_agency: dict[str, float] = {}
    for a in awards:
        ag  = a.get("Awarding Sub Agency") or a.get("Awarding Agency") or "Unknown"
        amt = float(a.get("Award Amount") or 0)
        by_agency[ag] = by_agency.get(ag, 0) + amt
    top = sorted(by_agency.items(), key=lambda x: -x[1])[:5]

    return {"total": total, "count": len(awards), "top_agencies": top}


# ============================================================
# CourtListener — federal court records search
# ============================================================
# Free public REST API mirroring the federal court PACER system. Requires
# free API key registration at https://www.courtlistener.com/register/ —
# keys auth via "Authorization: Token <key>" header.
#
# Used by the Court Docket subtab to find pending federal cases by party
# name (e.g., search "Apple" → list of active dockets where Apple is a
# party, with court / filing date / case status).

_COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4"


@cache_data(ttl=86400, show_spinner=False)
def fetch_courtlistener_search(
    query: str,
    search_type: str = "r",   # r=RECAP filings · o=opinions · oa=oral arguments
    limit: int = 20,
) -> list[dict] | None:
    """Search CourtListener federal court records.

    Returns a list of result dicts on success, None when:
      • no API key configured
      • network/API failure
      • empty query
    Empty list when the query had no matches.
    """
    if not has_courtlistener() or not query or not query.strip():
        return None
    headers = {
        "Authorization": f"Token {COURTLISTENER_API_KEY.strip()}",
        "User-Agent": "NoodleStockTracker/1.0 (catalyst calendar)",
    }
    params = {"q": query.strip(), "type": search_type}
    try:
        r = requests.get(
            f"{_COURTLISTENER_BASE}/search/",
            headers=headers, params=params, timeout=20,
        )
        r.raise_for_status()
        return (r.json().get("results") or [])[:limit]
    except Exception:
        return None


# ============================================================
# Major SCOTUS / federal cases — curated catalog
# ============================================================
# Hardcoded reference list of widely-tracked pending cases that have
# investment implications. Maintained manually — court dockets are too
# unstructured for clean auto-import the way FOMC / Treasury are.
# Each entry returns dict shaped for political_catalysts ingestion.
#
# IMPORTANT: dates are best-known opinion-window estimates, not certainties.
# Court schedules slip. Verify via CourtListener / SCOTUSblog before sizing
# trades around any specific date.

def get_major_court_cases() -> list[dict]:
    """Curated list of pending federal court cases material to public
    company earnings or sector-wide regulation. Acts as a starter pack the
    user can selectively import; everything beyond this catalog is logged
    via manual entry on the Catalyst Calendar tab."""

    # SCOTUS October 2025 term decisions land May–June 2026.
    # Federal antitrust cases run on their own (slower) clocks.
    end_of_term_jun30 = _ts(2026, 6, 30)

    cases = [
        {
            "title": "DOJ v. Google · Search Antitrust Remedies",
            "event_date":    _ts(2026, 8, 15),
            "catalyst_type": "court",
            "category":      "Antitrust · Federal District",
            "stakes": (
                "**Court:** D.D.C. (Judge Mehta) · **Stage:** Remedies ruling.\n\n"
                "**Bull case (for GOOGL):** Narrow conduct remedy (no breakup, "
                "preserved default-search payments to AAPL) → relief rally.\n\n"
                "**Base case:** Behavioral remedies (data-sharing, default-search "
                "auction) → manageable hit, modest revenue drag.\n\n"
                "**Bear case:** Structural breakup or full ban on default-search "
                "payments → AAPL services revenue at risk (~$20B/yr from Google), "
                "GOOGL search moat impaired."
            ),
            "tickers": "GOOGL,GOOG,AAPL,MSFT,DDOG",
            "sectors": "Tech,Antitrust",
            "source":  "court_catalog",
        },
        {
            "title": "DOJ v. Google · Ad Tech Antitrust",
            "event_date":    _ts(2026, 9, 22),
            "catalyst_type": "court",
            "category":      "Antitrust · Federal District",
            "stakes": (
                "**Court:** E.D. Va. · **Stage:** Liability ruling expected.\n\n"
                "Targets Google's ad-tech stack (DoubleClick / publisher "
                "ad server / ad exchange). DoJ seeks divestiture of AdX and "
                "DFP.\n\n"
                "**Beneficiaries on adverse ruling:** TTD, MGNI, PUBM, CRTO."
            ),
            "tickers": "GOOGL,GOOG,TTD,MGNI,PUBM,CRTO",
            "sectors": "Tech,AdTech,Antitrust",
            "source":  "court_catalog",
        },
        {
            "title": "FTC v. Meta · Vertical Merger Antitrust",
            "event_date":    _ts(2026, 7, 31),
            "catalyst_type": "court",
            "category":      "Antitrust · Federal District",
            "stakes": (
                "FTC seeks unwinding of Instagram / WhatsApp acquisitions. "
                "Bench trial concluded; ruling pending.\n\n"
                "**Bear case (META):** Forced divestiture of Instagram → "
                "loss of ~50% of ad revenue stream. Tail risk only — base "
                "case is structural conduct remedies."
            ),
            "tickers": "META,SNAP,PINS",
            "sectors": "Tech,Social,Antitrust",
            "source":  "court_catalog",
        },
        {
            "title": "FTC v. Amazon · Antitrust Trial",
            "event_date":    _ts(2026, 10, 13),
            "catalyst_type": "court",
            "category":      "Antitrust · Federal District",
            "stakes": (
                "**Court:** W.D. Wash. · **Stage:** Trial proceedings.\n\n"
                "FTC alleges anti-competitive conduct in Amazon's "
                "third-party seller policies and Buy Box algorithm. "
                "Material to AMZN's ~$140B/yr 3P services revenue line."
            ),
            "tickers": "AMZN,EBAY,SHOP",
            "sectors": "Retail,Antitrust",
            "source":  "court_catalog",
        },
        {
            "title": "SCOTUS · Tariff Authority Challenges (IEEPA)",
            "event_date":    end_of_term_jun30,
            "catalyst_type": "court",
            "category":      "SCOTUS · Statutory",
            "stakes": (
                "Challenges to executive tariff authority under IEEPA. "
                "Outcome reshapes US-China / US-EU trade architecture.\n\n"
                "**Bull case (broad EM equity):** Tariffs struck → CNY rallies, "
                "China internet stocks (BABA, JD, PDD) catch a bid, "
                "industrial supply chains re-rate higher.\n\n"
                "**Bear case (US domestic-content):** Tariffs upheld → "
                "STLD, NUE, X, dollar firms; importers (BBY, DG) under pressure."
            ),
            "tickers": "BABA,JD,PDD,STLD,NUE,X,BBY,DG",
            "sectors": "Trade,Industrials,Steel,EM",
            "source":  "court_catalog",
        },
        {
            "title": "SCOTUS · Apple v. Epic — App Store Fees Cert",
            "event_date":    end_of_term_jun30,
            "catalyst_type": "court",
            "category":      "SCOTUS · Cert Decision",
            "stakes": (
                "Cert decision on Ninth Circuit's App Store anti-steering "
                "ruling. If cert granted, full review of App Store economics "
                "follows in 2026-27 term.\n\n"
                "Material to AAPL services revenue (~$80B/yr) and the "
                "broader app-store economy."
            ),
            "tickers": "AAPL,GOOGL,MSFT,EPIC",
            "sectors": "Tech,App Store,Antitrust",
            "source":  "court_catalog",
        },
        {
            "title": "AbbVie · Humira-class Patent Cliff Litigation",
            "event_date":    _ts(2026, 11, 18),
            "catalyst_type": "court",
            "category":      "Pharma · Patent",
            "stakes": (
                "Generic / biosimilar challenges to AbbVie's downstream "
                "rebate-and-bundle defenses around the Humira franchise. "
                "Resolves whether AbbVie's contracting practices keep "
                "biosimilars constrained to <30% market share."
            ),
            "tickers": "ABBV,AMGN,PFE,JNJ,TEVA,REGN",
            "sectors": "Pharma,Biotech,Patent",
            "source":  "court_catalog",
        },
        {
            "title": "SCOTUS · Securities Class Action Reform Cert",
            "event_date":    end_of_term_jun30,
            "catalyst_type": "court",
            "category":      "SCOTUS · Securities",
            "stakes": (
                "Pending cert / decisions on the pleading standard for "
                "securities fraud actions under PSLRA. Outcome materially "
                "shifts D&O litigation exposure for every public company "
                "and for the listed insurers (TRV, CB, AIG)."
            ),
            "tickers": "TRV,CB,AIG,WRB",
            "sectors": "Insurance,Securities,Litigation",
            "source":  "court_catalog",
        },
    ]
    return cases


def get_treasury_refunding_schedule() -> list[dict]:
    """Treasury Quarterly Refunding announcement dates.

    Held the first Wednesday of February, May, August, and November —
    Treasury announces the size and composition of the next quarter's
    coupon issuance. Material for the long-end of the curve, dollar,
    and bank capital trades.
    """
    # (year, month, day) of the announcement
    announcements = [
        (2026,  2,  4),
        (2026,  5,  6),
        (2026,  8,  5),
        (2026, 11,  4),
        (2027,  2,  3),
        (2027,  5,  5),
        (2027,  8,  4),
        (2027, 11,  3),
    ]
    out = []
    for y, m, d in announcements:
        quarter_label = {2: "Q1", 5: "Q2", 8: "Q3", 11: "Q4"}[m]
        title = f"Treasury Quarterly Refunding · {quarter_label} {y}"
        stakes = (
            "**What's announced:** Size and composition of coupon issuance "
            "for the upcoming quarter — long-bond mix, bill cap, buyback "
            "guidance.\n\n"
            "**Bull case (for duration):** Skew toward bills + smaller-than-"
            "expected coupon increases → 10Y/30Y rally, TLT pops.\n\n"
            "**Base case:** Issuance roughly matches dealer expectations → "
            "muted move; markets absorb the supply.\n\n"
            "**Bear case:** Larger coupon sizes or a hawkish refunding tone "
            "→ duration sell-off, dollar firms, banks under pressure on "
            "deposit competition.\n\n"
            "_The Treasury Borrowing Advisory Committee (TBAC) presentation "
            "the same morning often telegraphs the announcement._"
        )
        out.append({
            "event_date":    _ts(y, m, d),
            "title":         title,
            "catalyst_type": "monetary",
            "category":      "Treasury Refunding",
            "stakes":        stakes,
            "tickers":       "TLT,IEF,UUP,XLF",
            "sectors":       "Treasuries,Dollar,Financials",
            "source":        "treasury_refunding",
        })
    return out


# ── Catalyst News ──────────────────────────────────────────────────────────────

# Keywords that signal a headline is relevant to a catalyst-type.
_CATALYST_KEYWORD_MAP: dict[str, list[str]] = {
    "monetary": [
        "fed", "fomc", "federal reserve", "rate cut", "rate hike", "rate decision",
        "powell", "inflation", "cpi", "pce", "interest rate", "taper", "qe", "qt",
        "treasury", "refunding", "yield curve", "10-year", "10y", "bund", "ecb",
        "boe", "bank of japan", "boj", "monetary policy", "basis point", "bps",
        "sofr", "libor", "overnight rate", "dot plot", "jackson hole",
    ],
    "contract": [
        "contract", "award", "dod", "pentagon", "defense contract", "procurement",
        "rfp", "bid", "government contract", "federal contract", "usaspending",
        "nasa", "army", "navy", "air force", "darpa", "recompete", "gsa",
        "sole source", "task order", "idiq",
    ],
    "court": [
        "antitrust", "lawsuit", "litigation", "court", "judge", "ruling", "verdict",
        "doj", "ftc", "sec", "settlement", "injunction", "appeal", "scotus",
        "supreme court", "district court", "circuit court", "class action",
        "remedy", "breakup", "consent decree", "patent", "trademark", "infringement",
        "trial", "deposition",
    ],
    "earnings": [
        "earnings", "eps", "revenue", "guidance", "forecast", "outlook",
        "beat", "miss", "quarter", "quarterly", "fiscal year", "q1", "q2",
        "q3", "q4", "analyst estimate", "consensus",
    ],
    "regulatory": [
        "regulation", "regulatory", "rule", "rulemaking", "sec filing",
        "fda", "approval", "clearance", "nda", "bla", "510k", "clinical trial",
        "phase 2", "phase 3", "advisory committee", "adcom", "epa", "cfpb",
        "finra", "occ", "fdic", "fed stress test",
    ],
    "macro": [
        "gdp", "recession", "growth", "unemployment", "jobs report", "nfp",
        "nonfarm payroll", "consumer confidence", "pmi", "ism", "housing",
        "retail sales", "trade deficit", "tariff", "sanction", "geopolit",
        "supply chain", "commodity", "oil", "wti", "brent",
    ],
}

# Combined superset for the "any catalyst" filter
_ALL_CATALYST_KEYWORDS: list[str] = sorted(
    {kw for kws in _CATALYST_KEYWORD_MAP.values() for kw in kws}
)


def _score_headline(text: str) -> dict[str, int]:
    """Return a {category: hit_count} mapping for a headline + summary combo."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for cat, kws in _CATALYST_KEYWORD_MAP.items():
        hits = sum(1 for kw in kws if kw in lower)
        if hits:
            scores[cat] = hits
    return scores


@cache_data(ttl=600)
def get_catalyst_news(
    tickers: tuple[str, ...],
    *,
    max_per_ticker: int = 20,
    min_score: int = 1,
) -> list[dict]:
    """Fetch multi-ticker news, filter by catalyst-relevance keywords, and
    cross-reference each headline with any catalyst whose tickers overlap.

    Parameters
    ----------
    tickers:
        Tuple of ticker symbols to pull news for (tuple so Streamlit can hash it).
    max_per_ticker:
        Maximum articles per ticker before merging / deduplication.
    min_score:
        Minimum total keyword-hit count to include a headline.

    Returns
    -------
    List of article dicts with extra keys:
        ``scores``            – {category: hit_count} keyword breakdown
        ``total_score``       – sum of all hits (relevance rank)
        ``matched_tickers``   – tickers from the query that appear in the headline
        ``matched_catalysts`` – list of catalyst dicts whose tickers match
    """
    if not tickers:
        return []

    # ── 1. Fetch concurrently, one call per ticker ──────────────────────────
    all_items: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as ex:
        futures = {ex.submit(fetch_all_news, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            tk = futures[fut]
            try:
                items = fut.result() or []
                # Tag the originating ticker on each article for later matching.
                for it in items[:max_per_ticker]:
                    it = dict(it)
                    it.setdefault("_query_ticker", tk)
                    all_items.append(it)
            except Exception:
                continue

    # ── 2. Dedupe by URL ────────────────────────────────────────────────────
    by_url: dict[str, dict] = {}
    for item in all_items:
        key = item.get("url", "")
        if not key:
            continue
        existing = by_url.get(key)
        if existing is None:
            by_url[key] = item
        elif item.get("published_ts", 0) and not existing.get("published_ts", 0):
            by_url[key] = item

    items = list(by_url.values())

    # ── 3. Score and filter ─────────────────────────────────────────────────
    scored: list[dict] = []
    tickers_upper = {t.upper() for t in tickers}
    for item in items:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        scores = _score_headline(text)
        total = sum(scores.values())
        if total < min_score:
            continue

        # Which query tickers appear in the headline text?
        matched_tickers = sorted(
            t for t in tickers_upper if re.search(r"\b" + re.escape(t) + r"\b", text, re.I)
        )

        item = dict(item)
        item["scores"] = scores
        item["total_score"] = total
        item["matched_tickers"] = matched_tickers
        item["matched_catalysts"] = []  # filled in step 4
        scored.append(item)

    # ── 4. Cross-reference with live catalyst list ──────────────────────────
    # Import lazily to avoid a circular import at module load time.
    try:
        from data_store import list_catalysts as _list_catalysts
        upcoming = _list_catalysts(status="upcoming") + _list_catalysts(status="live")
    except Exception:
        upcoming = []

    for item in scored:
        headline_tickers = set(item["matched_tickers"])
        linked: list[dict] = []
        for cat in upcoming:
            cat_tickers = {t.upper() for t in (cat.get("tickers") or [])}
            if cat_tickers & headline_tickers:
                linked.append(cat)
        item["matched_catalysts"] = linked

    # ── 5. Sort: catalyst-linked first, then by total score, then by recency ─
    scored.sort(
        key=lambda a: (
            -len(a["matched_catalysts"]),
            -a["total_score"],
            -(a.get("published_ts") or 0),
        )
    )
    return scored
