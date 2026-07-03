import os

import streamlit as st

from agent import run_analyze_ticker as _run_analyze_ticker
from ui.common import _render_catalyst_card
from utils import sanitize_ticker


def _render_analyst_banner(conf: dict) -> None:
    if not conf:
        return
    stance = conf.get("stance")
    if not stance:
        return

    banner_fn = {
        "Bullish": st.success,
        "Bearish": st.error,
        "Neutral": st.info,
    }.get(stance, st.info)

    display_label = conf.get("raw_stance") or stance
    parts = [f"**Posture:** {display_label}"]

    prob = conf.get("probability")
    if prob and len(prob) == 2:
        parts.append(f"**Probability:** {prob[0]}–{prob[1]}%")

    horizon = conf.get("horizon")
    if horizon:
        parts.append(f"**Horizon:** {horizon}")

    cycle = conf.get("cycle")
    if cycle:
        parts.append(f"**Cycle:** {cycle}")

    rr = conf.get("rr_ratio")
    if rr and len(rr) == 2:
        try:
            ratio = rr[1] / rr[0] if rr[0] else None
            if ratio is not None:
                rr_label = (
                    "asymmetric ↑" if ratio >= 1.5 else
                    "asymmetric ↓" if ratio <= 0.67 else
                    "symmetric"
                )
                parts.append(f"**R/R:** {rr[0]:g}:{rr[1]:g}  _({rr_label})_")
        except Exception:
            pass

    legacy = conf.get("confidence")
    if legacy and not prob:
        parts.append(f"**Confidence:** {legacy}")

    banner_fn("  ·  ".join(parts))


def render(app_data: dict) -> None:
    peer_groups = app_data.get("peer_groups", {})

    st.header("🧠 Analyze Ticker — Agentic Workflow")
    st.markdown(
        "Produces an institutional-grade investment memo via a 7-layer reasoning chain: "
        "**Micro → Macro → Valuation → Probability-Weighted Scenarios → Counter-Thesis "
        "→ Asymmetric Payoff → Declared Horizon**. Every quantitative claim is "
        "source-tagged ([SimFin], [FMP], [10-K], [chunk_N]…), confidence is "
        "probability-calibrated (e.g. 60–70% over 12 months), and the chain explicitly "
        "steelmans the opposite view to counter confirmation bias. Data is gathered "
        "concurrently from yfinance, FMP, SimFin, Alpha Vantage, FRED, SEC EDGAR, "
        "and your reference library — expect ~30–60 s on a cold cache."
    )

    a_col1, a_col2 = st.columns([2, 3])
    with a_col1:
        analyst_ticker = sanitize_ticker(
            st.text_input("Target Ticker", value="AAPL", key="analyst_ticker").upper()
        )
    with a_col2:
        peer_options = ["(none)"] + list(peer_groups.keys())
        analyst_peer_group = st.selectbox(
            "Inject peer cohort (optional)", peer_options, key="analyst_peer_group"
        )

    ac1, ac2 = st.columns([1, 1])
    with ac1:
        include_sec_10k = st.checkbox(
            "Include SEC 10-K excerpt", value=True,
            help="Pulls the most recent annual filing. Adds ~10-15s on cold "
                 "cache. Turn off if you only want fast market-layer analysis.",
        )
    with ac2:
        stream_memo = st.checkbox(
            "Stream memo as it generates", value=True,
            help="Off = faster for short memos; On = see the Oracle think.",
        )

    run_analyst = st.button(
        "🚀 Run Full Analysis", type="primary", use_container_width=True, key="analyst_run"
    )

    if "analyst_result" not in st.session_state:
        st.session_state["analyst_result"] = None

    if run_analyst:
        if not analyst_ticker:
            st.warning("Enter a ticker to analyze.")
        else:
            _step_labels = {
                "market":     "Live market snapshot (yfinance / Alpaca)",
                "fmp":        "FMP — analyst targets & ratings",
                "simfin":     "SimFin — standardised financials & ratios",
                "technicals": "Alpha Vantage — RSI / MACD / BBands + CNN F&G",
                "consensus":  "Wall Street consensus (yfinance)",
                "dcf":        "Quick DCF (default assumptions)",
                "news":       "Recent news (Yahoo RSS)",
                "peers":      "Peer metrics matrix",
                "macro":      "Macro & credit (FRED)",
                "catalysts":  "🎯 Tracked political / economic catalysts",
                "sec":        "SEC 10-K excerpt",
                "rag":        "Reference library retrieval",
                "synthesis":  "LLM synthesis",
            }
            _status_icons = {"start": "⏳", "done": "✅", "skip": "⚪", "error": "❌"}
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

                _render_analyst_banner(envelope.get("confidence") or {})
                st.session_state["analyst_result"] = envelope
            elif envelope and envelope.get("error"):
                st.error(envelope["error"])

    _cached = st.session_state.get("analyst_result")
    if _cached and not run_analyst:
        st.divider()
        st.subheader(f"📋 Last Memo — {_cached.get('ticker', '?')}")
        st.markdown(_cached.get("memo") or "_(no memo)_")
        _render_analyst_banner(_cached.get("confidence") or {})

    if _cached:
        _cached_catalysts = _cached.get("catalysts") or []
        if _cached_catalysts:
            with st.expander(
                f"🎯 Tracked Catalysts in this Memo ({len(_cached_catalysts)})",
                expanded=False,
            ):
                st.caption(
                    "These are the catalysts pulled from your **🎯 Catalyst "
                    "Calendar** that the Analyst LLM was given as context. "
                    "Click any card to inspect or edit."
                )
                for _c in _cached_catalysts:
                    _render_catalyst_card(_c)

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
