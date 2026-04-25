"""Agentic 'Analyze Ticker' workflow.

Orchestrates the existing primitives (SEC, news, valuation, peers, macro,
RAG) into a single structured investment memo. The LLM is *only* used at
the end for synthesis — the data-gathering phase is deterministic and runs
concurrently where possible.

Public entry point:

    run_analyze_ticker(ticker, peer_group_tickers=None, period="1Y",
                       progress_callback=None) -> dict

Returns a dict with:
  - context:   the raw structured context strings (for transparency)
  - rag_docs:  list of langchain Documents used
  - memo:      the synthesized investment memo (string)
  - confidence: extracted confidence label if the LLM produced one
  - steps:     list of {step, status, detail} for the UI progress panel

`progress_callback(step_name, status)` is called with status in
{"start", "done", "skip", "error"} so the Streamlit tab can render a live
checklist while data is fetched.
"""
from __future__ import annotations

import concurrent.futures as _fut
import re
from typing import Callable, Optional

import ollama

from api import (
    fetch_dcf_data,
    fetch_financial_news,
    fetch_live_prices,
    fetch_macro_data,
    fetch_peer_metrics,
    fetch_sec_filing,
    fetch_stock_details,
    fred,
)
from utils import format_large_number

# Keep a separate handle so the user can swap models without touching the
# Oracle tab's existing pinned model.
ANALYST_MODEL = "llama3.2"

_ProgressCB = Optional[Callable[[str, str, str], None]]


def _notify(cb: _ProgressCB, step: str, status: str, detail: str = "") -> None:
    if cb:
        try:
            cb(step, status, detail)
        except Exception:
            pass


# ---------- Step functions (each returns a string context block or "") ----------

def _step_market(ticker: str, cb: _ProgressCB) -> str:
    _notify(cb, "market", "start")
    try:
        lp = fetch_live_prices([ticker]).get(ticker, {})
        price = lp.get("price")
        change = lp.get("change")
        hist, info = fetch_stock_details(ticker, "1Y")
        mkt_cap = format_large_number(info.get("marketCap"))
        pe_t = info.get("trailingPE")
        pe_f = info.get("forwardPE")
        fifty_low = info.get("fiftyTwoWeekLow")
        fifty_high = info.get("fiftyTwoWeekHigh")

        one_year_pct = None
        if hist is not None and not hist.empty and len(hist) > 20:
            one_year_pct = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100

        lines = [f"LIVE MARKET SNAPSHOT — {ticker}:"]
        lines.append(f"- Price: ${price}  (Day Change: {change}%)")
        if one_year_pct is not None:
            lines.append(f"- 1Y Total Return (price only): {one_year_pct:.2f}%")
        lines.append(f"- Market Cap: {mkt_cap}")
        lines.append(f"- P/E Trailing: {pe_t}  |  P/E Forward: {pe_f}")
        if fifty_low and fifty_high:
            lines.append(f"- 52-Week Range: ${fifty_low} – ${fifty_high}")
        _notify(cb, "market", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "market", "error", str(e))
        return ""


def _step_consensus(ticker: str, cb: _ProgressCB) -> str:
    _notify(cb, "consensus", "start")
    try:
        _, info = fetch_stock_details(ticker, "1M")
        tgt = info.get("targetMeanPrice")
        rec = (info.get("recommendationKey") or "").upper()
        rev_g = info.get("revenueGrowth")
        earn_g = info.get("earningsGrowth")
        n_analysts = info.get("numberOfAnalystOpinions")
        lines = [f"WALL STREET CONSENSUS — {ticker}:"]
        lines.append(f"- Analyst Mean Target: ${tgt}  ({n_analysts or '?'} analysts)")
        lines.append(f"- Consensus Rating: {rec or 'N/A'}")
        if rev_g is not None:
            lines.append(f"- Forward Revenue Growth Est.: {rev_g * 100:.2f}%")
        if earn_g is not None:
            lines.append(f"- Forward Earnings Growth Est.: {earn_g * 100:.2f}%")
        _notify(cb, "consensus", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "consensus", "error", str(e))
        return ""


def _step_dcf(
    ticker: str,
    cb: _ProgressCB,
    discount_rate: float = 0.10,
    growth_rate: float = 0.08,
    terminal_rate: float = 0.025,
) -> str:
    _notify(cb, "dcf", "start")
    try:
        fcf, shares, current_price = fetch_dcf_data(ticker)
        if not all([fcf, shares, current_price]):
            _notify(cb, "dcf", "skip", "insufficient financial data")
            return f"QUICK DCF — {ticker}: insufficient financial data from yfinance."
        if fcf <= 0:
            _notify(cb, "dcf", "skip", "negative FCF")
            return (
                f"QUICK DCF — {ticker}: trailing FCF is negative "
                f"({format_large_number(fcf)}), standard DCF collapses."
            )
        pv_sum = 0.0
        cf = fcf
        for year in range(1, 6):
            cf *= (1 + growth_rate)
            pv_sum += cf / ((1 + discount_rate) ** year)
        terminal = cf * (1 + terminal_rate) / (discount_rate - terminal_rate)
        pv_terminal = terminal / ((1 + discount_rate) ** 5)
        intrinsic = (pv_sum + pv_terminal) / shares
        mos = ((intrinsic - current_price) / intrinsic) * 100 if intrinsic else 0.0
        lines = [
            f"QUICK DCF — {ticker}  (r={discount_rate:.0%}, g={growth_rate:.0%}, "
            f"tg={terminal_rate:.1%}):",
            f"- Trailing FCF: {format_large_number(fcf)}",
            f"- 5-yr PV of Cash Flows: {format_large_number(pv_sum)}",
            f"- PV of Terminal Value: {format_large_number(pv_terminal)}",
            f"- Intrinsic Value / Share: ${intrinsic:.2f}  vs Market: ${current_price:.2f}",
            f"- Margin of Safety: {mos:.1f}%  "
            f"({'UNDERVALUED' if mos > 0 else 'OVERVALUED'})",
        ]
        _notify(cb, "dcf", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "dcf", "error", str(e))
        return ""


def _step_news(ticker: str, cb: _ProgressCB) -> str:
    _notify(cb, "news", "start")
    try:
        articles = fetch_financial_news(ticker) or []
        if not articles:
            _notify(cb, "news", "done", "no headlines")
            return f"RECENT NEWS — {ticker}: no recent institutional headlines."
        lines = [f"RECENT NEWS — {ticker}:"]
        for a in articles[:5]:
            lines.append(f"- {a.get('title', '')}  ({a.get('time', '')})")
        _notify(cb, "news", "done", f"{len(articles[:5])} items")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "news", "error", str(e))
        return ""


def _step_peers(
    ticker: str, peer_tickers: list[str] | None, cb: _ProgressCB
) -> str:
    if not peer_tickers:
        _notify(cb, "peers", "skip", "no peer group provided")
        return ""
    _notify(cb, "peers", "start")
    try:
        peers = list(peer_tickers)
        if ticker not in peers:
            peers = [ticker] + peers
        df = fetch_peer_metrics(peers)
        if df is None or df.empty:
            _notify(cb, "peers", "skip", "empty frame")
            return ""
        lines = [f"PEER MATRIX — {ticker} vs cohort:"]
        for _, r in df.iterrows():
            lines.append(
                f"- {r.get('Ticker')}: "
                f"P/E(T)={r.get('P/E (Trailing)')}, "
                f"P/E(F)={r.get('P/E (Forward)')}, "
                f"P/B={r.get('P/B')}, "
                f"EV/EBITDA={r.get('EV/EBITDA')}, "
                f"ROE={r.get('ROE (%)')}%, "
                f"D/E={r.get('Debt/Equity')}"
            )
        _notify(cb, "peers", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "peers", "error", str(e))
        return ""


def _step_macro(cb: _ProgressCB) -> str:
    _notify(cb, "macro", "start")
    if not fred:
        _notify(cb, "macro", "skip", "no FRED key")
        return ""
    try:
        fed_df = fetch_macro_data("FEDFUNDS")
        hy_df = fetch_macro_data("BAMLH0A0HYM2")
        ten_df = fetch_macro_data("DGS10")
        lines = ["MACRO & CREDIT ENVIRONMENT (latest FRED prints):"]
        if fed_df is not None and not fed_df.empty:
            lines.append(f"- Fed Funds Rate: {fed_df['Value'].iloc[-1]:.2f}%")
        if ten_df is not None and not ten_df.empty:
            lines.append(f"- 10Y Treasury: {ten_df['Value'].iloc[-1]:.2f}%")
        if hy_df is not None and not hy_df.empty:
            lines.append(f"- HY Credit Spread: {hy_df['Value'].iloc[-1]:.2f}%")
        _notify(cb, "macro", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "macro", "error", str(e))
        return ""


def _step_sec(ticker: str, cb: _ProgressCB, max_chars: int = 20000) -> str:
    """Pull the most recent 10-K text and truncate aggressively. Full
    filings are 100k+ chars — for the synthesis prompt we take the head
    (which usually contains Risk Factors, MD&A start) plus a slice from
    the middle so we get both the qualitative risks and some financials.
    """
    _notify(cb, "sec", "start")
    try:
        text, meta = fetch_sec_filing(ticker, form_type="10-K")
        if text is None:
            _notify(cb, "sec", "skip", str(meta))
            return f"SEC 10-K — {ticker}: {meta}"
        cleaned = re.sub(r"\s+", " ", text)
        if len(cleaned) <= max_chars:
            body = cleaned
        else:
            head = cleaned[: max_chars // 2]
            mid_start = len(cleaned) // 3
            mid = cleaned[mid_start : mid_start + max_chars // 2]
            body = head + "\n... [truncated] ...\n" + mid
        _notify(cb, "sec", "done", f"{len(cleaned):,} chars")
        return f"RECENT 10-K EXCERPT — {ticker}:\n{body}"
    except Exception as e:
        _notify(cb, "sec", "error", str(e))
        return ""


def _step_rag(ticker: str, query_hint: str, cb: _ProgressCB) -> tuple[str, list]:
    """Lazy-imported so agent.py can be loaded in environments where rag
    isn't configured yet."""
    _notify(cb, "rag", "start")
    try:
        from rag import retrieve
        query = (
            f"What do my reference documents say about {ticker}'s "
            f"business quality, risks, and competitive position? {query_hint}"
        )
        docs = retrieve(
            query,
            k=6,
            ticker=ticker,
            use_mmr=True,
        )
        if not docs:
            _notify(cb, "rag", "done", "0 chunks")
            return ("", [])
        lines = [f"REFERENCE LIBRARY EXCERPTS — retrieved {len(docs)} chunks:"]
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "?")
            cat = d.metadata.get("category", "?")
            excerpt = d.page_content.strip().replace("\n", " ")[:600]
            lines.append(f"[chunk_{i}] ({cat}) {src}\n{excerpt}")
        _notify(cb, "rag", "done", f"{len(docs)} chunks")
        return ("\n\n".join(lines), docs)
    except Exception as e:
        _notify(cb, "rag", "error", str(e))
        return ("", [])


# ---------- Synthesis ----------

_MEMO_SYSTEM = """You are 'The True Oracle', a rigorous buy-side analyst. You will be given
a bundle of structured context about a single ticker and must produce a
concise investment memo. You must strictly obey these rules:

1. Logic-First: Isolate atomic propositions from the context. Do not
   invent numbers or quotes — if something isn't in the context, say
   'not in context' instead of guessing.
2. Probabilistic Calibration: Reject binary True/False. Anchor any
   forward claim in evidence, and attach a confidence level (Low /
   Medium / High) with a one-line rationale.
3. Structured Output: Use the exact section headers below. Keep each
   section tight — 3 to 6 bullet points.
4. Citation Discipline: When quoting or paraphrasing a reference-library
   chunk, cite it as [chunk_N] matching the numbered excerpts provided.
   Do not invent chunk numbers.

Required sections (use these exact headers, in this order):

## Thesis in One Sentence
## Bull Case
## Bear Case
## Key Numbers
## Key Uncertainties
## Catalysts to Watch (next 6–12 months)
## Overall Confidence
   (One line: Bullish / Bearish / Neutral + Confidence: Low/Medium/High + brief rationale.)
"""


def _build_memo_prompt(ticker: str, contexts: list[str]) -> str:
    joined = "\n\n".join([c for c in contexts if c])
    return f"""TICKER UNDER REVIEW: {ticker}

STRUCTURED CONTEXT BUNDLE
=========================
{joined}

TASK: Produce the investment memo per your system instructions.
"""


_CONF_RE = re.compile(
    r"(Bullish|Bearish|Neutral).*?Confidence:\s*(Low|Medium|High)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_confidence(memo: str) -> dict:
    m = _CONF_RE.search(memo or "")
    if not m:
        return {"stance": None, "confidence": None}
    return {
        "stance": m.group(1).capitalize(),
        "confidence": m.group(2).capitalize(),
    }


# ---------- Public entry point ----------

def run_analyze_ticker(
    ticker: str,
    peer_group_tickers: list[str] | None = None,
    include_sec: bool = True,
    progress_callback: _ProgressCB = None,
    stream: bool = False,
):
    """Orchestrate the full analysis pipeline.

    If `stream=False`, returns a dict with the finished memo.
    If `stream=True`, returns a dict containing a generator under key
    `memo_stream` that the Streamlit tab can pass to `st.write_stream`;
    the caller is responsible for attaching the final text to the dict
    afterwards via the returned `finalize(full_text)` helper.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"error": "Empty ticker."}

    # Fan out the slow I/O steps concurrently. SEC is slowest (~15s network),
    # news/prices/peers/macro are ~1-3s each. Running in a thread pool cuts
    # the gather phase from ~30s to ~15s on a cold cache.
    results: dict[str, str] = {}
    with _fut.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            "market":    pool.submit(_step_market, ticker, progress_callback),
            "consensus": pool.submit(_step_consensus, ticker, progress_callback),
            "dcf":       pool.submit(_step_dcf, ticker, progress_callback),
            "news":      pool.submit(_step_news, ticker, progress_callback),
            "peers":     pool.submit(_step_peers, ticker, peer_group_tickers, progress_callback),
            "macro":     pool.submit(_step_macro, progress_callback),
        }
        if include_sec:
            futures["sec"] = pool.submit(_step_sec, ticker, progress_callback)
        for name, f in futures.items():
            try:
                results[name] = f.result(timeout=60)
            except Exception as e:
                _notify(progress_callback, name, "error", str(e))
                results[name] = ""

    # RAG runs *after* the others — its query benefits from knowing the
    # news/consensus context wouldn't be meaningful here, but keeping it
    # sequential avoids piling up Ollama embedding calls on top of the
    # pool's active threads.
    rag_hint = results.get("market", "")[:300]
    rag_ctx, rag_docs = _step_rag(ticker, rag_hint, progress_callback)
    results["rag"] = rag_ctx

    ordered_blocks = [
        results.get("market", ""),
        results.get("consensus", ""),
        results.get("dcf", ""),
        results.get("macro", ""),
        results.get("peers", ""),
        results.get("news", ""),
        results.get("sec", ""),
        results.get("rag", ""),
    ]
    prompt = _build_memo_prompt(ticker, ordered_blocks)

    _notify(progress_callback, "synthesis", "start")

    if stream:
        def _gen():
            try:
                for chunk in ollama.chat(
                    model=ANALYST_MODEL,
                    messages=[
                        {"role": "system", "content": _MEMO_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                ):
                    yield chunk["message"]["content"]
            except Exception as e:
                yield f"\n\n[LLM error: {e}]"

        envelope = {
            "ticker": ticker,
            "context_blocks": results,
            "rag_docs": rag_docs,
            "memo_stream": _gen(),
            "memo": None,
            "confidence": None,
        }

        def finalize(full_text: str) -> dict:
            envelope["memo"] = full_text
            envelope["confidence"] = _extract_confidence(full_text)
            _notify(progress_callback, "synthesis", "done")
            return envelope

        envelope["finalize"] = finalize
        return envelope

    try:
        resp = ollama.chat(
            model=ANALYST_MODEL,
            messages=[
                {"role": "system", "content": _MEMO_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        memo = resp["message"]["content"]
    except Exception as e:
        _notify(progress_callback, "synthesis", "error", str(e))
        return {
            "ticker": ticker,
            "context_blocks": results,
            "rag_docs": rag_docs,
            "memo": None,
            "error": f"LLM synthesis failed: {e}",
        }

    _notify(progress_callback, "synthesis", "done")
    return {
        "ticker": ticker,
        "context_blocks": results,
        "rag_docs": rag_docs,
        "memo": memo,
        "confidence": _extract_confidence(memo),
    }
