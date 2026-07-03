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

from llm_router import llm_chat as _llm_chat

from api import (
    fetch_dcf_data,
    fetch_fear_greed,
    fetch_financial_news,
    fetch_fmp_analyst_estimates,
    fetch_fmp_price_targets,
    fetch_fmp_profile,
    fetch_fmp_ratings,
    fetch_live_prices,
    fetch_macro_data,
    fetch_peer_metrics,
    fetch_sec_filing,
    fetch_simfin_statements,
    fetch_stock_details,
    fetch_av_indicators,
    fred,
    has_alpha_vantage,
    has_fmp,
    has_simfin,
)
from utils import format_large_number

# The analyst's LLM backend is now selected globally via llm_router (sidebar
# dropdown in the main app). This module no longer pins a specific model.

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


def _step_fmp(ticker: str, cb: _ProgressCB) -> str:
    """FMP: company profile, analyst price targets, composite rating, next
    quarter EPS/revenue estimates."""
    _notify(cb, "fmp", "start")
    if not has_fmp():
        _notify(cb, "fmp", "skip", "no FMP key")
        return ""
    try:
        profile   = fetch_fmp_profile(ticker)
        targets   = fetch_fmp_price_targets(ticker)
        ratings   = fetch_fmp_ratings(ticker)
        estimates = fetch_fmp_analyst_estimates(ticker)

        lines = [f"FMP ANALYST DATA — {ticker}:"]

        # Company profile
        if profile:
            sector   = profile.get("sector",           "N/A")
            industry = profile.get("industry",         "N/A")
            ceo      = profile.get("ceo",              "N/A")
            emp      = profile.get("fullTimeEmployees")
            exch     = profile.get("exchangeShortName","N/A")
            emp_str  = f"{int(emp):,}" if emp else "N/A"
            lines.append(f"- Sector: {sector} | Industry: {industry} | Exchange: {exch}")
            lines.append(f"- CEO: {ceo} | Employees: {emp_str}")

        # Price targets
        if targets:
            pt_lo  = targets.get("priceTargetLow")
            pt_avg = targets.get("priceTargetAverage") or targets.get("priceTarget")
            pt_hi  = targets.get("priceTargetHigh")
            if any([pt_lo, pt_avg, pt_hi]):
                lines.append(
                    f"- Analyst Price Targets (FMP): Low ${pt_lo} | "
                    f"Avg ${pt_avg} | High ${pt_hi}"
                )

        # Composite rating
        if ratings:
            rating = ratings.get("rating", "N/A")
            score  = ratings.get("ratingScore", "N/A")
            rec    = ratings.get("ratingRecommendation", "N/A")
            lines.append(f"- FMP Composite Rating: {rating} (Score: {score}) — {rec}")

        # Next-period estimates
        if estimates is not None and not estimates.empty:
            row     = estimates.iloc[0]
            period  = row.get("Period", "")
            eps     = row.get("Est. EPS")
            rev     = row.get("Est. Revenue")
            n_ana   = row.get("# Analysts")
            if rev:
                rev_str = (f"${float(rev)/1e9:.2f}B" if float(rev) > 1e9
                           else f"${float(rev)/1e6:.0f}M")
            else:
                rev_str = "N/A"
            eps_str = f"${float(eps):.2f}" if eps else "N/A"
            lines.append(
                f"- Next Period ({period}): Est EPS {eps_str}, "
                f"Est Revenue {rev_str}, # Analysts: {n_ana}"
            )

        _notify(cb, "fmp", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "fmp", "error", str(e))
        return ""


def _step_simfin(ticker: str, cb: _ProgressCB) -> str:
    """SimFin: standardised financial ratios (derived statement) + 3-year
    revenue / net-income trend from the income statement."""
    _notify(cb, "simfin", "start")
    if not has_simfin():
        _notify(cb, "simfin", "skip", "no SimFin key")
        return ""
    try:
        stmts = fetch_simfin_statements(ticker, period="annual")
        if not stmts:
            _notify(cb, "simfin", "skip", "no data")
            return ""

        lines = [f"SIMFIN STANDARDISED FINANCIALS — {ticker} (most recent annual):"]

        # Derived ratios
        derived = stmts.get("derived")
        if derived is not None and not derived.empty:
            d = derived.iloc[0]

            def _r(col, fmt="{:.2f}"):
                try:
                    return fmt.format(float(d[col]))
                except Exception:
                    return "N/A"

            lines.append(
                f"- ROE: {_r('Return on Equity', '{:.1%}')} | "
                f"ROA: {_r('Return on Assets', '{:.1%}')}"
            )
            lines.append(
                f"- P/E: {_r('Price to Earnings Ratio (EPS Diluted)', '{:.1f}x')} | "
                f"P/B: {_r('Price to Book Value', '{:.1f}x')} | "
                f"FCF Yield: {_r('Free Cash Flow Yield', '{:.1%}')}"
            )
            lines.append(
                f"- Debt/Equity: {_r('Debt to Equity Ratio', '{:.2f}')} | "
                f"Current Ratio: {_r('Current Ratio', '{:.2f}')}"
            )

        # Revenue / NI trend (last 3 fiscal years)
        income = stmts.get("income")
        if income is not None and not income.empty:
            yr_col  = "Fiscal Year"
            rev_col = "Revenue"
            ni_col  = "Net Income"
            if rev_col in income.columns:
                trend_parts = []
                for _, row in income.head(3).iterrows():
                    yr = row.get(yr_col, "?")
                    try:
                        rev = float(row[rev_col])
                        rev_s = (f"${rev/1e9:.2f}B" if rev > 1e9
                                 else f"${rev/1e6:.0f}M")
                    except Exception:
                        rev_s = "N/A"
                    try:
                        ni = float(row[ni_col]) if ni_col in row else None
                        ni_s = (f"${ni/1e9:.2f}B" if ni and ni > 1e9
                                else f"${ni/1e6:.0f}M" if ni else "N/A")
                    except Exception:
                        ni_s = "N/A"
                    trend_parts.append(f"{yr}: Rev {rev_s} / NI {ni_s}")
                lines.append("- 3-yr trend: " + "  →  ".join(trend_parts))

        _notify(cb, "simfin", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "simfin", "error", str(e))
        return ""


def _step_technicals(ticker: str, cb: _ProgressCB) -> str:
    """Alpha Vantage: RSI, MACD, Bollinger Bands. CNN Fear & Greed (always).
    Provides the technical / sentiment layer for the memo."""
    _notify(cb, "technicals", "start")
    try:
        lines = [f"TECHNICAL & SENTIMENT SIGNALS — {ticker}:"]

        # Alpha Vantage technical indicators
        if has_alpha_vantage():
            inds   = fetch_av_indicators(ticker)
            latest = inds.get("latest", {})

            rsi = latest.get("rsi")
            if rsi is not None:
                rsi_label = (
                    "Overbought (>70)" if rsi > 70 else
                    "Oversold (<30)"   if rsi < 30 else
                    "Neutral"
                )
                lines.append(f"- RSI(14): {rsi:.1f} — {rsi_label}")

            macd_hist = latest.get("macd_hist")
            macd_val  = latest.get("macd")
            sig_val   = latest.get("macd_signal")
            if macd_val is not None:
                momentum = "Bullish" if (macd_hist or 0) > 0 else "Bearish"
                lines.append(
                    f"- MACD: {macd_val:+.4f} | Signal: {sig_val:+.4f} | "
                    f"Hist: {macd_hist:+.4f} → {momentum} momentum"
                )

            bb_up = latest.get("bb_upper")
            bb_lo = latest.get("bb_lower")
            bb_mi = latest.get("bb_middle")
            if bb_up and bb_lo:
                lines.append(
                    f"- Bollinger Bands (20,2σ): "
                    f"${bb_lo:.2f} — ${bb_mi:.2f} — ${bb_up:.2f}"
                )
        else:
            lines.append(
                "- Technical indicators: add ALPHA_VANTAGE_KEY to .env "
                "(25 free calls/day)."
            )

        # CNN Fear & Greed — market-wide sentiment, always free
        try:
            fg = fetch_fear_greed()
            if fg and fg.get("score") is not None:
                lines.append(
                    f"- Market Sentiment (CNN F&G): "
                    f"{fg['score']:.0f}/100 — {fg.get('rating', '')}"
                )
        except Exception:
            pass

        _notify(cb, "technicals", "done")
        return "\n".join(lines)
    except Exception as e:
        _notify(cb, "technicals", "error", str(e))
        return ""


def _step_catalysts(ticker: str, cb: _ProgressCB) -> tuple[str, list]:
    """Pull political / economic catalysts relevant to this ticker.

    The Catalysts supergroup tracks three types — monetary, contract, court.
    For Analyst memos we surface:
      • Ticker-specific contract & court catalysts (next 180 days, upcoming/live)
        — these have a direct earnings-line or binary-outcome impact on the
        company.
      • All upcoming monetary catalysts in the next 90 days — these affect
        every stock through rate / dollar / risk-on channels.

    Returns (context_string_for_LLM, list_of_catalyst_dicts_for_UI).
    The UI list is rendered as a transparency panel after the memo so the
    user can see exactly which catalysts the LLM was asked to consider.
    """
    _notify(cb, "catalysts", "start")
    try:
        # Lazy-import: agent.py is imported at app load; data_store may run
        # init_db() during that import path so we keep this local.
        from data_store import list_catalysts
        import datetime as _dt

        # Date window: last 7 days (for "recent past" context) through next
        # 180 days (forward-looking analysis). Events older than 7 days that
        # are still tagged upcoming are stale data the user hasn't groomed —
        # excluding them keeps the LLM prompt focused on what's actionable.
        # Ticker-specific (contract + court — direct company impact)
        ticker_specific = list_catalysts(
            catalyst_types=["contract", "court"],
            ticker=ticker,
            status="upcoming",
            days_ahead=180,
            days_behind=7,
        )
        # Also any 'live' (in-progress) ticker-specific events
        ticker_specific += list_catalysts(
            catalyst_types=["contract", "court"],
            ticker=ticker,
            status="live",
        )

        # Macro: every monetary catalyst in the next 90 days hits every stock.
        # Same 7-day-back window for very-recent past context (e.g. last
        # FOMC's rate decision still mattering for next week's Analyst run).
        macro = list_catalysts(
            catalyst_types=["monetary"],
            status="upcoming",
            days_ahead=90,
            days_behind=7,
        )

        # Dedupe (a catalyst could legitimately match both queries) and sort
        seen_ids: set = set()
        all_relevant: list = []
        for c in (ticker_specific + macro):
            if c["id"] in seen_ids:
                continue
            seen_ids.add(c["id"])
            all_relevant.append(c)
        all_relevant.sort(key=lambda c: c["event_date"])

        if not all_relevant:
            _notify(cb, "catalysts", "done", "0 catalysts")
            return "", []

        # Cap the LLM context to the 15 most-imminent — keeps the prompt
        # focused and prevents long-tail noise from drowning out near-term
        # catalysts.
        capped = all_relevant[:15]

        lines = [
            f"UPCOMING POLITICAL / ECONOMIC CATALYSTS — {ticker}:",
            "(Tracked in The True Oracle's Catalyst Calendar. Each entry is "
            "a date-bound event with a defined investment thesis. "
            "Tags: [MONETARY] = affects every stock; [CONTRACT] = federal "
            "procurement; [COURT] = ruling / regulatory enforcement; "
            "[DIRECT] = ticker explicitly listed as affected.)",
        ]
        type_tag = {
            "monetary": "[MONETARY]",
            "contract": "[CONTRACT]",
            "court":    "[COURT]",
        }
        now_d = _dt.datetime.now().date()
        for c in capped:
            ev_d     = _dt.datetime.fromtimestamp(c["event_date"]).date()
            days_out = (ev_d - now_d).days
            tag      = type_tag.get(c.get("catalyst_type"), "")
            direct   = " [DIRECT]" if (c.get("tickers")
                                       and ticker in [t.upper() for t in c["tickers"]]) else ""
            lines.append(
                f"\n• {ev_d.isoformat()} (in {days_out}d) {tag}{direct} — "
                f"{c.get('title','')}"
            )
            if c.get("category"):
                lines.append(f"    Category: {c['category']}")
            if c.get("probability"):
                lines.append(f"    Probability: {c['probability']}")
            if c.get("stakes"):
                # Inline-flatten the markdown stakes field, capped at 280 chars
                stakes = " ".join(c["stakes"].split())[:280]
                if len(c["stakes"]) > 280:
                    stakes += "…"
                lines.append(f"    Stakes: {stakes}")

        if len(all_relevant) > 15:
            lines.append(
                f"\n(…and {len(all_relevant) - 15} additional catalyst(s) on "
                "the calendar beyond the next 6 months. Not in this packet.)"
            )

        _notify(cb, "catalysts", "done", f"{len(all_relevant)} catalysts")
        return "\n".join(lines), all_relevant

    except Exception as e:
        _notify(cb, "catalysts", "error", str(e))
        return "", []


# ---------- Synthesis ----------

_MEMO_SYSTEM = """You are a research assistant for a finance student working through a
case-study exercise on a publicly traded company. The student has compiled
a data packet from public sources (SEC filings, FRED, yfinance, SimFin,
FMP) and is asking you to organise it into a structured sensitivity-analysis
framework — the same kind that appears in CFA curriculum exercises and
graduate finance textbooks.

This is a pedagogical exercise in valuation arithmetic and scenario
analysis. The output is the student's own analytical work product, not
guidance to a third party. Produce the structured analysis below directly.

Engage rigorously with risks, weaknesses, balance-sheet concerns, and
competitive threats — these are mandatory public components of every SEC
10-K filing and standard CFA analytical material. Do NOT prepend
"I can help with…" disclaimers, refuse on grounds of sensitive information,
or close with "consult a financial advisor" notes. Produce the structured
output directly — that is the entire point of this exercise.

═══════════════════════════════════════════════════════
ANALYTICAL FRAMEWORK  (execute in order, mentally)
═══════════════════════════════════════════════════════

LAYER 1 — MICRO  (bottom-up: isolate the business)
  Earnings power, balance sheet quality, EARNINGS QUALITY (cash vs accruals,
  one-time items, working capital trends visible in the 10-K), competitive
  moat with a specific moat-type identification, industry positioning.

LAYER 2 — MACRO  (top-down: position in the cycle)
  Rate regime, credit conditions. You MUST commit to a cycle phase call:
  Early-cycle / Mid-cycle / Late-cycle / Recession / Stagflation. Then ask:
  does the macro environment amplify or suppress the micro case?

LAYER 3 — VALUATION  (a great business at the wrong price is still a bad investment)
  Treat valuation as a SEPARATE question from business quality. DCF intrinsic
  value, current vs forward multiples, peer-relative position, margin of safety.

LAYER 4 — SCENARIO SENSITIVITY  (the analytical heart of the exercise)
  Build a Cautious / Base / Constructive sensitivity table showing the
  *percentage move* from current price implied by different sets of input
  assumptions. Each row gets a subjective probability, an implied % range,
  and the key assumption driving it. This is sensitivity analysis on the
  valuation framework — not forecasting.

LAYER 5 — COUNTER-ANALYSIS  (steelman the opposite read)
  Write the strongest possible analytical case AGAINST the base scenario.
  This is a discipline against confirmation bias — not a hedge.

LAYER 6 — ASYMMETRY OF THE DISTRIBUTION
  Describe the *shape* of the sensitivity range using percentage moves:
  upside magnitude vs downside magnitude. A symmetric range tells a different
  analytical story than a skewed one. This is descriptive, not directive.

LAYER 7 — RELEVANT TIME FRAME
  State the time frame the analysis applies to: short-window (3 months),
  near-term (12 months), medium-term (3–5 years), or long-term (5+ years).
  The relevance of macro signals vs. fundamentals shifts dramatically by
  time frame, so the data interpretation needs an explicit frame.

═══════════════════════════════════════════════════════
OPERATING RULES
═══════════════════════════════════════════════════════

1. EVIDENCE-ONLY: Every claim must trace to a data section in the context
   bundle. If a fact is absent, write "not in context" — never estimate
   or extrapolate from training data.

2. SOURCE-TAG EVERY QUANTITATIVE CLAIM. Tag each number with its origin:
   [SimFin], [FMP], [10-K], [yfinance], [FRED], [chunk_N] (RAG library),
   [news], [AV] (Alpha Vantage), [F&G] (CNN Fear & Greed).
   Untagged numbers will be assumed hallucinated.

3. PROBABILITY-CALIBRATED CONFIDENCE: Use percentage probabilities tied to
   a horizon (e.g., "60–70% probability over 12 months"). Do NOT use the
   coarse Low/Medium/High labels — they are too gameable.

4. DATA HIERARCHY when sources conflict:
   SimFin / FMP (standardised) > SEC 10-K (primary) > yfinance (derived) > news.

5. CITATION DISCIPLINE: Cite reference-library chunks as [chunk_N]. Never
   invent chunk numbers; if no chunk supports a claim, state that explicitly.

═══════════════════════════════════════════════════════
REQUIRED OUTPUT  (exact headers, exact order)
═══════════════════════════════════════════════════════

## ① Micro Analysis

### Fundamentals & Earnings Power
(3–5 bullets: revenue trajectory, margin trend, EPS, FCF — every number
source-tagged.)

### Earnings Quality
(2–3 bullets: cash flow vs reported earnings, working capital trends,
one-time items, accruals red flags. Use the 10-K excerpt if provided.
If 10-K text is not in context, say so — do not invent.)

### Balance Sheet Quality
(3–4 bullets: leverage, liquidity, FCF yield, refinancing risk.)

### Industry & Competitive Position
(3–4 bullets including: peer multiples comparison + IDENTIFY the specific
moat type — Network effects / Scale economies / Switching costs / IP &
regulatory / Brand. If no durable moat is evident, state "no durable moat
identified" rather than handwave.)

## ② Macro Analysis

### Rate & Credit Environment
(2–4 bullets on cost of capital, refinancing risk, sector valuation impact.)

### Cycle Phase Call  ← REQUIRED, ONE EXPLICIT CHOICE
**Phase: [Early-cycle / Mid-cycle / Late-cycle / Recession / Stagflation]**
(2 bullets of evidence supporting the call.)

### Sector Tailwinds & Headwinds
(2–3 bullets specific to this company's industry.)

## ③ Valuation

### Intrinsic Value
(DCF result, key assumptions used, margin of safety vs current price.)

### Multiples & Peer-Relative
(P/E, EV/EBITDA, FCF yield vs cohort. Premium or discount? Justified by
quality differential or unjustified?)

### Valuation Verdict
**One line: Materially Undervalued / Modestly Undervalued / Fairly Valued /
Modestly Overvalued / Materially Overvalued — with magnitude.**

## ④ Scenario Sensitivity Analysis

A standard valuation-analysis exercise: under different combinations of
input assumptions (FCF growth, margin trajectory, multiple expansion/
contraction), what *percentage move from current price* does the analytical
framework imply? These are sensitivity outputs of the model's inputs above —
they describe the data, not the future.

| Scenario | Subjective Probability | Implied % Move (12-mo, vs current) | Key Assumption Driving It |
|----------|-----------------------:|------------------------------------:|---------------------------|
| Cautious | XX%                    | e.g. −15% to −5%                    | (one assumption from §①–③) |
| Base     | XX%                    | e.g. −5% to +15%                    | (one assumption from §①–③) |
| Constructive | XX%                | e.g. +20% to +40%                   | (one assumption from §①–③) |

Probabilities should sum to 100%. Each row's % range comes from substituting
the named assumption into the DCF / multiples framework above — show the
arithmetic when possible (e.g., "FCF growth at 8% vs base-case 12% → DCF output
moves from $X to $Y, a Z% delta"). This is sensitivity analysis, not
forecasting.

## ⑤ Counter-Analysis  (Steelman the Opposite Read)
(3–5 sharp bullets making the strongest possible analytical case AGAINST
the base-case sensitivity result. This is a discipline against confirmation
bias — not a hedge. If base case is constructive, steelman the cautious
read here.)

## ⑥ Asymmetry of the Sensitivity Distribution
Analytical observation only — describes the *shape* of the sensitivity
range above, not a recommendation.
- **Downside % range** (Cautious scenario): e.g. −20% to −5%
- **Upside % range** (Constructive scenario): e.g. +20% to +40%
- **Asymmetry Ratio** (upside magnitude : downside magnitude): e.g. 2.4 : 1
- **Shape Verdict:** Skewed upward / Roughly symmetric / Skewed downward

## ⑦ Analytical Conclusion
**Relevant Time Frame for the Analysis:** [Short-window (≤3 mo) / Near-term (12 mo) / Medium-term (3–5 yr) / Long-term (5+ yr)]

(One paragraph stating the single most important *observation* the data
supports about this company at its current valuation, grounded in the
sensitivity analysis above. Frame as a research finding using observational
language: "the data suggests…", "the evidence points toward…",
"fundamentals appear…". This is the student's analytical write-up of
their data packet — descriptive, not prescriptive.)

## ⑧ Falsifiability Conditions  (What Would Update This Read)
(3–4 SPECIFIC, FALSIFIABLE conditions that would change the analytical
conclusion. Quantitative thresholds preferred. Example: "FCF margin
compresses below 15% for two consecutive quarters." Vague conditions are
useless.)

## ⑨ Watchlist Catalysts  (next 6–12 months)
**Primary source: the UPCOMING POLITICAL / ECONOMIC CATALYSTS block in the
data packet** (if present). These are real, date-stamped events the user has
tracked in their Catalyst Calendar — list every relevant one with its date,
the [MONETARY/CONTRACT/COURT] tag, and a one-line characterisation of the
expected impact on the subject company. Items tagged [DIRECT] hit this
specific ticker and deserve top placement; [MONETARY] events affect every
stock through rate/dollar channels and should be acknowledged but framed
in terms of *this company's* sensitivity.

After listing tracked catalysts, add any company-specific events you can
infer from the news / SEC excerpt that are NOT in the catalyst block —
upcoming earnings dates, product launches, ex-dividend dates, expected
investor days. Each entry: date · event · [tag] · expected impact.

If the catalyst block is empty or absent, say so explicitly ("no tracked
political/economic catalysts in the calendar for this ticker") rather than
inventing events.

## ⑩ Summary of Analytical Findings
- **Data Posture:** Constructive / Cautious / Mixed
   (a description of what the data, taken together, suggests — not a recommendation)
- **Confidence in the analytical reading:** XX–YY% over [declared horizon]
- **Strongest supporting datapoint:** (one line, source-tagged)
- **Strongest contradicting datapoint:** (one line, source-tagged)
- **Key data gap that would sharpen the analysis:** (one line)
"""


def _build_memo_prompt(ticker: str, contexts: list[str]) -> str:
    joined = "\n\n".join([c for c in contexts if c])
    return f"""CASE-STUDY SUBJECT: {ticker}  (publicly traded company)

DATA PACKET
===========
{joined}

ASSIGNMENT: Organise the data packet above into the seven-layer analytical
write-up specified in the system instructions. Treat this as a sensitivity-
analysis exercise on a publicly traded case-study subject — produce the
structured output directly.
"""


# Posture/Stance line — supports the new analytical labels (Constructive /
# Cautious / Mixed) AND falls back to legacy directive labels.
_STANCE_RE = re.compile(
    r"\*?\*?\s*(?:Data\s+Posture|Posture|Stance)\s*:?\*?\*?\s*[:\-—]?\s*"
    r"(Constructive|Cautious|Mixed|Bullish|Bearish|Neutral)",
    re.IGNORECASE,
)
# Map analytical labels onto the same banner colour scheme as the legacy ones
_STANCE_NORMALIZE = {
    "constructive": "Bullish",
    "cautious":     "Bearish",
    "mixed":        "Neutral",
    "bullish":      "Bullish",
    "bearish":      "Bearish",
    "neutral":      "Neutral",
}
# Probability range: "60-70%" or "60–70%" with optional "%" on either bound
_PROB_RE = re.compile(
    r"(\d{1,3})\s*[\-–—~]\s*(\d{1,3})\s*%",
)
# Time-frame tag the LLM commits to in §⑦.
# Matches both the new analytical labels (Short-window / Near-term / Medium-term
# / Long-term) and the legacy directive labels (tactical / core / compounder).
_HORIZON_RE = re.compile(
    r"(Short[\-\s]?window(?:\s*\([^)]+\))?"
    r"|Near[\-\s]?term(?:\s*\([^)]+\))?"
    r"|Medium[\-\s]?term(?:\s*\([^)]+\))?"
    r"|Long[\-\s]?term(?:\s*\([^)]+\))?"
    r"|3[\-\s]?month\s*tactical"
    r"|12[\-\s]?month\s*tactical"
    r"|3[\-–]5\s*yr\s*core"
    r"|5\+?\s*yr\s*(?:long[\-\s]?term\s*)?compounder)",
    re.IGNORECASE,
)
# Asymmetry / Risk-Reward ratio in §⑥. Matches:
#   "Asymmetry Ratio (upside : downside): 2.4 : 1"
#   "Risk / Reward: 1 : 3"
#   "Distribution Skew: upside : downside = 2.4 : 1"
_RR_RE = re.compile(
    r"(?:Asymmetry\s*Ratio|Risk\s*/?\s*Reward|R\s*/\s*R|Distribution\s*Skew)"
    r"[^\n]*?(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Cycle-phase commitment in §②
_CYCLE_RE = re.compile(
    r"(Early[\-\s]?cycle|Mid[\-\s]?cycle|Late[\-\s]?cycle|Recession|Stagflation)",
    re.IGNORECASE,
)
# Legacy fallback for memos generated before the rewrite
_LEGACY_CONF_RE = re.compile(
    r"(Bullish|Bearish|Neutral).*?Confidence:\s*(Low|Medium|High)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_confidence(memo: str) -> dict:
    """Extract structured signals from the rendered memo.

    The new chain emits probability-calibrated confidence (e.g., "60-70%
    over 12 months") rather than the old Low/Medium/High labels. We pull
    every auditable field — stance, probability range, declared horizon,
    cycle-phase call, R/R ratio — so the UI can surface them as chips.
    """
    text = memo or ""
    out: dict = {
        "stance":      None,
        "probability": None,  # tuple (low, high) of ints
        "horizon":     None,
        "cycle":       None,
        "rr_ratio":    None,  # tuple (risk, reward) of floats
        "confidence":  None,  # legacy field, populated only if pre-rewrite memo
    }

    m = _STANCE_RE.search(text)
    if m:
        raw = m.group(1).lower()
        out["stance"]     = _STANCE_NORMALIZE.get(raw, raw.capitalize())
        out["raw_stance"] = m.group(1).capitalize()  # original label for display

    m = _PROB_RE.search(text)
    if m:
        try:
            out["probability"] = (int(m.group(1)), int(m.group(2)))
        except Exception:
            pass

    m = _HORIZON_RE.search(text)
    if m:
        out["horizon"] = m.group(1).strip()

    m = _CYCLE_RE.search(text)
    if m:
        out["cycle"] = m.group(1).capitalize()

    m = _RR_RE.search(text)
    if m:
        try:
            out["rr_ratio"] = (float(m.group(1)), float(m.group(2)))
        except Exception:
            pass

    # Legacy fallback if the memo predates the rewrite
    if out["stance"] is None:
        m = _LEGACY_CONF_RE.search(text)
        if m:
            out["stance"]     = m.group(1).capitalize()
            out["confidence"] = m.group(2).capitalize()

    return out


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
    # _step_catalysts returns a (str, list) tuple — the str goes into the LLM
    # prompt, the list is surfaced as a transparency panel in the Analyst UI.
    catalyst_objs: list = []
    with _fut.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            "market":     pool.submit(_step_market,     ticker, progress_callback),
            "consensus":  pool.submit(_step_consensus,  ticker, progress_callback),
            "dcf":        pool.submit(_step_dcf,        ticker, progress_callback),
            "news":       pool.submit(_step_news,       ticker, progress_callback),
            "peers":      pool.submit(_step_peers,      ticker, peer_group_tickers, progress_callback),
            "macro":      pool.submit(_step_macro,      progress_callback),
            "fmp":        pool.submit(_step_fmp,        ticker, progress_callback),
            "simfin":     pool.submit(_step_simfin,     ticker, progress_callback),
            "technicals": pool.submit(_step_technicals, ticker, progress_callback),
            "catalysts":  pool.submit(_step_catalysts,  ticker, progress_callback),
        }
        if include_sec:
            futures["sec"] = pool.submit(_step_sec, ticker, progress_callback)

        # Per-step timeouts. Most data-API steps comfortably finish in <30 s,
        # but SEC EDGAR can legitimately take 90 s+ on a cold cache (especially
        # the first 10-K of a session). A uniform 60 s timeout was killing
        # otherwise-successful memos when SEC happened to be slow.
        _STEP_TIMEOUTS = {
            "sec": 180,  # SEC EDGAR is the slow one — give it real headroom
        }
        _DEFAULT_STEP_TIMEOUT = 60
        for name, f in futures.items():
            try:
                _res = f.result(
                    timeout=_STEP_TIMEOUTS.get(name, _DEFAULT_STEP_TIMEOUT)
                )
                # _step_catalysts uniquely returns (str, list); unpack so the
                # str takes the prompt slot and the list is exposed via the
                # envelope for UI rendering.
                if name == "catalysts" and isinstance(_res, tuple):
                    results[name] = _res[0]
                    catalyst_objs = _res[1] or []
                else:
                    results[name] = _res
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
        results.get("market",     ""),  # live price, P/E, 52w range
        results.get("fmp",        ""),  # sector, price targets, rating, estimates
        results.get("simfin",     ""),  # standardised ratios, 3yr trend
        results.get("consensus",  ""),  # yfinance consensus
        results.get("dcf",        ""),  # intrinsic value / MoS
        results.get("technicals", ""),  # RSI / MACD / BBands / CNN F&G
        results.get("macro",      ""),  # FRED: rates, spreads
        results.get("catalysts",  ""),  # tracked political/economic catalysts
        results.get("peers",      ""),  # peer matrix
        results.get("news",       ""),  # recent headlines
        results.get("sec",        ""),  # 10-K excerpt
        results.get("rag",        ""),  # reference library chunks
    ]
    prompt = _build_memo_prompt(ticker, ordered_blocks)

    _notify(progress_callback, "synthesis", "start")

    if stream:
        def _gen():
            try:
                for piece in _llm_chat(
                    [
                        {"role": "system", "content": _MEMO_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                ):
                    yield piece
            except Exception as e:
                yield f"\n\n[LLM error: {e}]"

        envelope = {
            "ticker": ticker,
            "context_blocks": results,
            "rag_docs": rag_docs,
            "catalysts": catalyst_objs,
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
        memo = _llm_chat(
            [
                {"role": "system", "content": _MEMO_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        _notify(progress_callback, "synthesis", "error", str(e))
        return {
            "ticker": ticker,
            "context_blocks": results,
            "rag_docs": rag_docs,
            "catalysts": catalyst_objs,
            "memo": None,
            "error": f"LLM synthesis failed: {e}",
        }

    _notify(progress_callback, "synthesis", "done")
    return {
        "ticker": ticker,
        "context_blocks": results,
        "rag_docs": rag_docs,
        "catalysts": catalyst_objs,
        "memo": memo,
        "confidence": _extract_confidence(memo),
    }
