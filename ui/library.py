import concurrent.futures as _fut
import os

import streamlit as st

from api import (
    fetch_av_indicators,
    fetch_fear_greed,
    fetch_financial_news,
    fetch_fmp_analyst_estimates,
    fetch_fmp_profile,
    fetch_fmp_ratings,
    fetch_fmp_price_targets,
    fetch_live_prices,
    fetch_macro_data,
    fetch_peer_metrics,
    fetch_sec_filing,
    fetch_simfin_statements,
    fetch_stock_details,
    fred,
    has_alpha_vantage,
    has_fmp,
    has_simfin,
)
from data_store import (
    add_institution,
    link_institution_doc,
    list_institutional_coverage,
    remove_institution,
    set_primary_institution_doc,
    unlink_doc_from_all_catalysts,
    unlink_doc_from_all_institutions,
    unlink_institution_doc,
)
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import (
    CATEGORIES as _CATEGORIES,
    TOPICS as _TOPICS,
    TOPIC_LABELS as _TOPIC_LABELS,
    already_ingested as _already_ingested,
    decompose_query as _decompose_query,
    delete_document as _delete_document,
    format_chunks_for_citation as _format_chunks_for_citation,
    ingest_chunks as _ingest_chunks,
    list_documents as _list_documents,
    retrieve,
    retrieve_multi as _retrieve_multi,
    route_query as _route_query,
    set_category as _set_category,
    set_topics as _set_topics,
    verify_citations as _verify_citations,
)
from ui.common import DB_DIR, UPLOAD_DIR
from utils import format_large_number, sanitize_ticker


def render(app_data: dict) -> None:
    peer_groups = app_data.get("peer_groups", {})

    st.header("The Library (Local RAG Database)")
    st.markdown(
        "Inject financial PDFs, Annual 10-Ks, and Quarterly 8-K Earnings data directly "
        "into Noodle Bot's permanent memory."
    )

    # ── Institutional Research Coverage Board ──────────────────────────────────
    st.subheader("🏦 Institutional Research Coverage")
    st.caption(
        "Track which major institutions have a market report in the library. "
        "🟢 = covered · ⚪ = missing. Each institution can have **multiple** "
        "reports attached; the ⭐ marks the *primary* (default-displayed) one."
    )

    _inst_rows = list_institutional_coverage()
    _lib_docs  = _list_documents()
    _mr_docs   = [d for d in _lib_docs if d["category"] == "market_report"]
    _mr_by_id  = {d["doc_id"]: d for d in _mr_docs}
    _doc_labels = {d["doc_id"]: d["source"] for d in _mr_docs}

    if "_inst_update_mode" not in st.session_state:
        st.session_state._inst_update_mode = set()

    def _doc_label_short(doc_id: str) -> str:
        raw = _doc_labels.get(doc_id, doc_id)
        raw = raw.replace("./temp_pdfs/", "").replace("PDF: ", "")
        if raw.lower().endswith(".pdf"):
            raw = raw[:-4]
        return raw

    if not _inst_rows:
        st.info("No institutions tracked yet — add one below.")
    else:
        with st.container(border=True):
            _hc1, _hc2, _hc3 = st.columns([2, 3, 2])
            _hc1.markdown("**Institution**")
            _hc2.markdown("**Attached Reports**")
            _hc3.markdown("**Update Portal**")

        for _inst in _inst_rows:
            _name       = _inst["institution"]
            _primary_id = _inst.get("primary_doc_id")
            _doc_ids    = _inst.get("doc_ids") or []
            _editing    = _name in st.session_state._inst_update_mode
            _has_any    = bool(_doc_ids)

            with st.container(border=True):
                _ic1, _ic2, _ic3 = st.columns([2, 3, 2])
                _ic1.markdown(f"{'🟢' if _has_any else '⚪'}  {_name}")

                if _editing:
                    with _ic2:
                        _new_pdfs = st.file_uploader(
                            f"Drop new PDFs for {_name}", type="pdf",
                            accept_multiple_files=True,
                            key=f"inst_upload_{_name}",
                            label_visibility="collapsed",
                        )
                        _available_ids = [d["doc_id"] for d in _mr_docs
                                          if d["doc_id"] not in _doc_ids]
                        _sel_existing = st.multiselect(
                            "Or pick from existing market reports",
                            options=_available_ids,
                            format_func=_doc_label_short,
                            key=f"inst_link_{_name}",
                            label_visibility="collapsed",
                            placeholder="— pick existing report(s) to attach —",
                        )

                        if _doc_ids:
                            st.markdown("**Currently attached:**")
                            for _doc_id in _doc_ids:
                                _is_primary = _doc_id == _primary_id
                                _l1, _l2, _l3 = st.columns([8, 1, 1])
                                _star = "⭐ " if _is_primary else "☆ "
                                _l1.markdown(f"{_star}_{_doc_label_short(_doc_id)}_")
                                if not _is_primary:
                                    if _l2.button("⭐", key=f"inst_setpri_{_name}_{_doc_id}",
                                                  help="Mark as primary"):
                                        set_primary_institution_doc(_name, _doc_id)
                                        st.rerun()
                                else:
                                    _l2.markdown("&nbsp;", unsafe_allow_html=True)
                                if _l3.button("✖", key=f"inst_unlink_{_name}_{_doc_id}",
                                              help="Detach from this institution"):
                                    unlink_institution_doc(_name, _doc_id)
                                    st.rerun()

                    with _ic3:
                        _sb1, _sb2, _sb3 = st.columns(3)
                        if _sb1.button("💾", key=f"inst_save_{_name}", help="Save"):
                            _added  = 0
                            _errors: list[str] = []
                            for _pdf in (_new_pdfs or []):
                                _safe_name  = os.path.basename(_pdf.name)
                                _file_path  = os.path.join(UPLOAD_DIR, _safe_name)
                                _new_doc_id = f"pdf::{_safe_name}"
                                try:
                                    with open(_file_path, "wb") as _f:
                                        _f.write(_pdf.getbuffer())
                                    if not _already_ingested(_new_doc_id):
                                        with st.spinner(f"Ingesting {_safe_name}…"):
                                            _loader     = PyMuPDFLoader(_file_path)
                                            _pages      = _loader.load()
                                            _splitter   = RecursiveCharacterTextSplitter(
                                                chunk_size=1000, chunk_overlap=200
                                            )
                                            _doc_chunks = _splitter.split_documents(_pages)
                                            _ingest_chunks(
                                                _doc_chunks, _new_doc_id,
                                                f"PDF: {_safe_name}", category="market_report",
                                            )
                                    link_institution_doc(_name, _new_doc_id)
                                    _added += 1
                                except Exception as _e:
                                    _errors.append(f"{_safe_name}: {_e}")
                            for _doc_id in (_sel_existing or []):
                                link_institution_doc(_name, _doc_id)
                                _added += 1
                            if _errors:
                                for _err in _errors:
                                    st.error(f"Failed: {_err}")
                            if _added:
                                st.session_state._inst_update_mode.discard(_name)
                                st.toast(
                                    f"📄 Linked {_added} report{'s' if _added != 1 else ''} → {_name}",
                                    icon="🔗",
                                )
                                st.rerun()
                            elif not _errors:
                                st.session_state._inst_update_mode.discard(_name)
                                st.rerun()

                        if _sb2.button("✖", key=f"inst_cancel_{_name}", help="Cancel"):
                            st.session_state._inst_update_mode.discard(_name)
                            st.rerun()
                        if _sb3.button("🗑️", key=f"inst_del_{_name}",
                                       help=f"Remove {_name} from watchlist"):
                            remove_institution(_name)
                            st.session_state._inst_update_mode.discard(_name)
                            st.rerun()

                else:
                    with _ic2:
                        if not _doc_ids:
                            st.markdown("_— not yet linked —_")
                        else:
                            for _doc_id in _doc_ids:
                                _star = "⭐ " if _doc_id == _primary_id else "•  "
                                st.markdown(f"{_star}_{_doc_label_short(_doc_id)}_")
                    if _ic3.button("Update here", key=f"inst_edit_{_name}",
                                   use_container_width=True):
                        st.session_state._inst_update_mode.add(_name)
                        st.rerun()

    st.divider()

    with st.form("inst_add_form", clear_on_submit=True):
        _inst_col1, _inst_col2 = st.columns([4, 1])
        _new_inst = _inst_col1.text_input(
            "Institution name",
            placeholder="e.g. Goldman Sachs, BlackRock, JPMorgan…",
            label_visibility="collapsed",
        )
        _inst_submitted = _inst_col2.form_submit_button("➕ Add", use_container_width=True)
    if _inst_submitted:
        if _new_inst.strip():
            if add_institution(_new_inst.strip()):
                st.toast(f"Added '{_new_inst.strip()}' to coverage watchlist", icon="🏦")
                st.rerun()
            else:
                st.warning(f"'{_new_inst.strip()}' is already in the list.")

    st.divider()

    # ── Ingestion module ───────────────────────────────────────────────────────
    col_pdf, col_sec = st.columns(2)

    with col_pdf:
        with st.expander("📚 Upload Local PDF", expanded=False):
            uploaded_file    = st.file_uploader("Upload Financial Document", type="pdf")
            _upload_cat_opts = ["textbook", "market_report", "sec_filing"]
            upload_category  = st.selectbox(
                "Categorize this PDF",
                options=_upload_cat_opts,
                format_func=lambda k: _CATEGORIES[k],
                key="upload_pdf_category",
            )
            upload_topics = st.multiselect(
                "Topics (multi-select)", options=_TOPICS, default=[],
                format_func=lambda k: _TOPIC_LABELS[k], key="upload_pdf_topics",
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
                            loader    = PyMuPDFLoader(file_path)
                            pages     = loader.load()
                            splitter  = RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=200
                            )
                            doc_chunks = splitter.split_documents(pages)

                        with st.spinner(f"Translating {len(doc_chunks)} chunks to vector coordinates..."):
                            try:
                                n      = _ingest_chunks(
                                    doc_chunks, doc_id, f"PDF: {safe_name}",
                                    category=upload_category, topics=upload_topics,
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
            sec_ticker    = sanitize_ticker(
                st.text_input("Enter Ticker (e.g., TSLA)").upper()
            )
            sec_form_type = st.radio(
                "Select Document Type",
                ["10-K (Annual Report)", "8-K (Latest Earnings/Material Events)"],
            )
            if st.button("Fetch & Inject SEC Data", type="primary",
                         use_container_width=True) and sec_ticker:
                target_form = "10-K" if "10-K" in sec_form_type else "8-K"

                with st.spinner(f"Locating {sec_ticker} {target_form} via SEC EDGAR…"):
                    raw_text, source_url = fetch_sec_filing(sec_ticker, form_type=target_form)

                if raw_text is None:
                    st.error(source_url)
                else:
                    doc_id = f"sec::{sec_ticker}::{target_form}"
                    if _already_ingested(doc_id):
                        st.info(
                            f"{sec_ticker}'s {target_form} is already in the library "
                            "— skipping re-embed."
                        )
                    else:
                        file_name = f"{sec_ticker}_{target_form}.txt"
                        file_path = os.path.join(UPLOAD_DIR, file_name)
                        with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
                            f.write(raw_text)

                        with st.spinner(f"{target_form} Downloaded. Chunking text..."):
                            loader     = TextLoader(file_path, encoding="utf-8")
                            pages      = loader.load()
                            splitter   = RecursiveCharacterTextSplitter(
                                chunk_size=1500, chunk_overlap=300
                            )
                            doc_chunks = splitter.split_documents(pages)
                            for chunk in doc_chunks:
                                chunk.metadata["ticker"] = sec_ticker
                                chunk.metadata["form"]   = target_form

                        with st.spinner(f"Translating {len(doc_chunks)} chunks to vector coordinates..."):
                            try:
                                n = _ingest_chunks(
                                    doc_chunks, doc_id,
                                    f"SEC EDGAR {target_form}: {sec_ticker}",
                                    category="sec_filing",
                                )
                                st.success(
                                    f"✅ Successfully injected {sec_ticker}'s "
                                    f"{target_form} ({n} chunks)!"
                                )
                            except Exception as e:
                                st.error(f"Failed to embed {target_form}. Error: {e}")

    st.divider()

    # ── Library Manager ───────────────────────────────────────────────────────
    _library_docs = _lib_docs
    _uncat_count  = sum(1 for d in _library_docs if d["category"] == "uncategorized")
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
                    st.write("")
                    if st.button(
                        f"Apply to {_uncat_count} document(s)", key="bulk_cat_apply",
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
                    _topics_str = ", ".join(_cur_topics) if _cur_topics else "no topics"
                    st.caption(
                        f"`{d['doc_id']}` · {d['chunks']} chunks · "
                        f"temporal: {d.get('temporal_validity', 'unknown')} · "
                        f"topics: {_topics_str}"
                    )
                with c2:
                    current = d["category"] if d["category"] in _cat_keys else "uncategorized"
                    new_cat = st.selectbox(
                        "Category", options=_cat_keys,
                        index=_cat_keys.index(current),
                        format_func=lambda k: _CATEGORIES[k],
                        key=f"cat_{d['doc_id']}",
                        label_visibility="collapsed",
                    )
                    new_topics = st.multiselect(
                        "Topics", options=_TOPICS,
                        default=[t for t in _cur_topics if t in _TOPICS],
                        format_func=lambda k: _TOPIC_LABELS[k],
                        key=f"top_{d['doc_id']}",
                        label_visibility="collapsed",
                        placeholder="Topics (optional, multi-select)",
                    )
                    _cat_changed  = new_cat != current
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
                    if st.button("🗑️", key=f"del_{d['doc_id']}",
                                 help="Remove from library"):
                        _delete_document(d["doc_id"])
                        unlink_doc_from_all_institutions(d["doc_id"])
                        unlink_doc_from_all_catalysts(d["doc_id"])
                        st.rerun()

    st.divider()

    # ── Oracle ─────────────────────────────────────────────────────────────────
    st.subheader("💬 Ask The Oracle")
    st.markdown(
        "Query your uploaded documents. Noodle Bot will synthesize an answer based on "
        "your database, current macro conditions, live news, real-time market data, "
        "and forward-looking Wall Street consensus."
    )

    if "oracle_answer" not in st.session_state:
        st.session_state["oracle_answer"]  = None
        st.session_state["oracle_sources"] = None

    user_query = st.text_area(
        "What would you like to know about your documents?",
        placeholder="e.g., Does management's 10-K outlook align with Wall Street's growth estimates?",
    )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        context_ticker = sanitize_ticker(
            st.text_input("Target Ticker (Injects News, Price & Consensus)").upper()
        )
    with col_q2:
        peer_group_options = ["None"] + list(peer_groups.keys())
        context_group      = st.selectbox("Inject Peer Group Matrix", peer_group_options)

    _filter_cat_keys = list(_CATEGORIES.keys())

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
            _sc  = ", ".join(_sug.get("categories") or []) or "(all)"
            _st_ = ", ".join(_sug.get("topics") or [])     or "(none)"
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
        "Retrieve only from categories", options=_filter_cat_keys, default=_default_cats,
        format_func=lambda k: _CATEGORIES[k], key="oracle_cat_filter",
    )
    selected_topics = st.multiselect(
        "Retrieve only from topics (OR — any selected topic matches)",
        options=_TOPICS, default=_default_tops,
        format_func=lambda k: _TOPIC_LABELS[k], key="oracle_topic_filter",
    )

    _opt_col1, _opt_col2, _opt_col3 = st.columns(3)
    with _opt_col1:
        use_mmr = st.checkbox("MMR diversification", value=True,
                              help="Max Marginal Relevance — reduces near-duplicate chunks.")
    with _opt_col2:
        use_multiquery = st.checkbox("Multi-query decomposition", value=True,
                                     help="Break compound questions into sub-queries.")
    with _opt_col3:
        use_citations = st.checkbox("Citation grounding", value=True,
                                    help="Force the Oracle to cite [chunk_N].")

    trigger_oracle = st.button("🔮 Consult The Oracle", type="primary", use_container_width=True)

    if trigger_oracle:
        if not user_query:
            st.warning("Please enter a question for the Oracle.")
        elif not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            st.warning("Your library is empty. Please upload a PDF or rip an SEC filing first.")
        else:
            with st.spinner("Initializing Omni-Context Engine…"):

                def _fetch_macro_block() -> str:
                    if not fred:
                        return ""
                    try:
                        fed_df = fetch_macro_data("FEDFUNDS")
                        hy_df  = fetch_macro_data("BAMLH0A0HYM2")
                        rate_val = (f"{fed_df['Value'].iloc[-1]:.2f}%"
                                    if (fed_df is not None and not fed_df.empty) else "Unknown")
                        hy_val   = (f"{hy_df['Value'].iloc[-1]:.2f}%"
                                    if (hy_df  is not None and not hy_df.empty)  else "Unknown")
                        return (
                            f"\nLIVE MACRO & CREDIT ENVIRONMENT:\n"
                            f"- Current Federal Funds Rate: {rate_val}\n"
                            f"- High Yield Credit Spread (Corporate Stress): {hy_val}\n"
                        )
                    except Exception:
                        return ""

                def _fetch_market_and_forward() -> tuple[str, str]:
                    if not context_ticker:
                        return "", ""
                    try:
                        live_p_data = fetch_live_prices([context_ticker])
                        p_info      = live_p_data.get(context_ticker, {})
                        curr_price  = p_info.get("price", "N/A")
                        day_change  = p_info.get("change", "N/A")
                        hist, info  = fetch_stock_details(context_ticker, "1M")
                        mkt_cap     = format_large_number(info.get("marketCap"))
                        pe          = info.get("trailingPE", "N/A")
                        fwd_pe      = info.get("forwardPE", "N/A")
                        target_price = info.get("targetMeanPrice", "N/A")
                        rec         = info.get("recommendationKey", "N/A").upper()
                        rev_growth  = info.get("revenueGrowth", 0)
                        earn_growth = info.get("earningsGrowth", 0)
                        rev_str     = f"{rev_growth * 100:.1f}%"  if rev_growth  else "N/A"
                        earn_str    = f"{earn_growth * 100:.1f}%" if earn_growth else "N/A"
                        trend_str   = "N/A"
                        if not hist.empty and len(hist) > 20:
                            trend_pct = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-21])
                                         / hist["Close"].iloc[-21]) * 100
                            trend_str = f"{trend_pct:.2f}%"
                        mkt = (
                            f"\nLIVE MARKET VALUATION FOR {context_ticker}:\n"
                            f"- Current Price: ${curr_price} (Day Change: {day_change}%)\n"
                            f"- 1-Month Trend: {trend_str}\n"
                            f"- Market Cap: {mkt_cap}\n"
                            f"- P/E Ratio (Trailing): {pe} | P/E Ratio (Forward): {fwd_pe}\n"
                        )
                        fwd = (
                            f"\nWALL STREET CONSENSUS & FORWARD EXPECTATIONS FOR {context_ticker}:\n"
                            f"- Mean Target Price: ${target_price}\n"
                            f"- Analyst Consensus: {rec}\n"
                            f"- Est. Forward Revenue Growth: {rev_str}\n"
                            f"- Est. Forward Earnings Growth: {earn_str}\n"
                        )
                        return mkt, fwd
                    except Exception:
                        return (
                            f"\nLIVE MARKET VALUATION FOR {context_ticker}: Temporarily Unavailable.\n",
                            "",
                        )

                def _fetch_news_block() -> str:
                    if not context_ticker:
                        return ""
                    try:
                        recent_news = fetch_financial_news(context_ticker)
                        if not recent_news:
                            return (
                                f"\nLIVE NEWS ENVIRONMENT FOR {context_ticker}:\n"
                                f"- No major institutional headlines in the last 24 hours.\n"
                            )
                        body = f"\nLIVE NEWS ENVIRONMENT FOR {context_ticker}:\n"
                        for article in recent_news:
                            body += f"- {article['title']} ({article['time']})\n"
                        return body
                    except Exception:
                        return ""

                def _fetch_fmp_block() -> str:
                    if not context_ticker or not has_fmp():
                        return ""
                    try:
                        _fmp_p  = fetch_fmp_profile(context_ticker)
                        _fmp_t  = fetch_fmp_price_targets(context_ticker)
                        _fmp_r  = fetch_fmp_ratings(context_ticker)
                        _fmp_e  = fetch_fmp_analyst_estimates(context_ticker)
                        _flines = [f"\nFMP ANALYST DATA FOR {context_ticker}:"]
                        if _fmp_p:
                            _flines.append(
                                f"- Sector: {_fmp_p.get('sector','N/A')} | "
                                f"Industry: {_fmp_p.get('industry','N/A')} | "
                                f"CEO: {_fmp_p.get('ceo','N/A')}"
                            )
                        if _fmp_t:
                            _flines.append(
                                f"- Price Targets: Low ${_fmp_t.get('priceTargetLow','N/A')} | "
                                f"Avg ${_fmp_t.get('priceTargetAverage') or _fmp_t.get('priceTarget','N/A')} | "
                                f"High ${_fmp_t.get('priceTargetHigh','N/A')}"
                            )
                        if _fmp_r:
                            _flines.append(
                                f"- FMP Rating: {_fmp_r.get('rating','N/A')} "
                                f"({_fmp_r.get('ratingRecommendation','N/A')})"
                            )
                        if _fmp_e is not None and not _fmp_e.empty:
                            _er = _fmp_e.iloc[0]
                            _rev = _er.get("Est. Revenue")
                            _rev_s = (
                                f"${float(_rev)/1e9:.2f}B" if _rev and float(_rev) > 1e9 else
                                f"${float(_rev)/1e6:.0f}M" if _rev else "N/A"
                            )
                            _flines.append(
                                f"- Next Period ({_er.get('Period','')}): "
                                f"Est EPS ${_er.get('Est. EPS','N/A')}, "
                                f"Est Rev {_rev_s}"
                            )
                        return "\n".join(_flines) + "\n"
                    except Exception:
                        return ""

                def _fetch_simfin_block() -> str:
                    if not context_ticker or not has_simfin():
                        return ""
                    try:
                        _sf = fetch_simfin_statements(context_ticker, period="annual")
                        if not _sf:
                            return ""
                        _sflines = [f"\nSIMFIN STANDARDISED FINANCIALS FOR {context_ticker}:"]
                        _der = _sf.get("derived")
                        if _der is not None and not _der.empty:
                            _d = _der.iloc[0]
                            def _rv(col, fmt="{:.2f}"):
                                try: return fmt.format(float(_d[col]))
                                except Exception: return "N/A"
                            _sflines.append(
                                f"- ROE: {_rv('Return on Equity','{:.1%}')} | "
                                f"ROA: {_rv('Return on Assets','{:.1%}')} | "
                                f"Debt/Equity: {_rv('Debt to Equity Ratio','{:.2f}')}"
                            )
                            _sflines.append(
                                f"- P/E: {_rv('Price to Earnings Ratio (EPS Diluted)','{:.1f}x')} | "
                                f"P/B: {_rv('Price to Book Value','{:.1f}x')} | "
                                f"FCF Yield: {_rv('Free Cash Flow Yield','{:.1%}')}"
                            )
                        _inc = _sf.get("income")
                        if _inc is not None and not _inc.empty and "Revenue" in _inc.columns:
                            _tparts = []
                            for _, _row in _inc.head(3).iterrows():
                                _yr = _row.get("Fiscal Year", "?")
                                try:
                                    _rv2 = float(_row["Revenue"])
                                    _rv_s = f"${_rv2/1e9:.2f}B" if _rv2 > 1e9 else f"${_rv2/1e6:.0f}M"
                                except Exception:
                                    _rv_s = "N/A"
                                _tparts.append(f"{_yr}: {_rv_s}")
                            _sflines.append("- Revenue trend: " + " → ".join(_tparts))
                        return "\n".join(_sflines) + "\n"
                    except Exception:
                        return ""

                def _fetch_technicals_block() -> str:
                    if not context_ticker or not has_alpha_vantage():
                        return ""
                    try:
                        _av  = fetch_av_indicators(context_ticker)
                        _avl = _av.get("latest", {})
                        if not _avl:
                            return ""
                        _avlines = [f"\nTECHNICAL SIGNALS FOR {context_ticker}:"]
                        _rsi = _avl.get("rsi")
                        if _rsi is not None:
                            _rsi_lbl = ("Overbought" if _rsi > 70 else
                                        "Oversold" if _rsi < 30 else "Neutral")
                            _avlines.append(f"- RSI(14): {_rsi:.1f} — {_rsi_lbl}")
                        _mhist = _avl.get("macd_hist")
                        _mval  = _avl.get("macd")
                        if _mval is not None:
                            _mdir = "Bullish" if (_mhist or 0) > 0 else "Bearish"
                            _avlines.append(
                                f"- MACD Histogram: {_mhist:+.4f} → {_mdir} momentum"
                            )
                        _bbu = _avl.get("bb_upper")
                        _bbl = _avl.get("bb_lower")
                        _bbm = _avl.get("bb_middle")
                        if _bbu and _bbl:
                            _avlines.append(
                                f"- Bollinger Bands: ${_bbl:.2f} — ${_bbm:.2f} — ${_bbu:.2f}"
                            )
                        return "\n".join(_avlines) + "\n"
                    except Exception:
                        return ""

                def _fetch_sentiment_block() -> str:
                    try:
                        _fg = fetch_fear_greed()
                        if _fg and _fg.get("score") is not None:
                            return (
                                f"\nMARKET SENTIMENT (CNN Fear & Greed): "
                                f"{_fg['score']:.0f}/100 — {_fg.get('rating','')}\n"
                            )
                    except Exception:
                        pass
                    return ""

                def _fetch_peer_block() -> str:
                    if not context_group or context_group == "None":
                        return ""
                    try:
                        g_tickers = peer_groups[context_group]
                        if not g_tickers:
                            return ""
                        p_df = fetch_peer_metrics(g_tickers)
                        if p_df.empty:
                            return ""
                        body = f"\nLIVE PEER GROUP VALUATION MATRIX ({context_group}):\n"
                        for _, r in p_df.iterrows():
                            body += (
                                f"- {r['Ticker']}: Price: ${r['Price']} | "
                                f"Trailing P/E: {r['P/E (Trailing)']} | "
                                f"EV/EBITDA: {r['EV/EBITDA']} | "
                                f"ROE: {r['ROE (%)']}% | "
                                f"D/E: {r['Debt/Equity']}\n"
                            )
                        return body
                    except Exception:
                        return (
                            f"\nLIVE PEER GROUP VALUATION MATRIX ({context_group}): "
                            "Temporarily Unavailable.\n"
                        )

                with _fut.ThreadPoolExecutor(max_workers=8) as _pool:
                    _f_macro      = _pool.submit(_fetch_macro_block)
                    _f_market_fwd = _pool.submit(_fetch_market_and_forward)
                    _f_news       = _pool.submit(_fetch_news_block)
                    _f_fmp        = _pool.submit(_fetch_fmp_block)
                    _f_simfin     = _pool.submit(_fetch_simfin_block)
                    _f_tech       = _pool.submit(_fetch_technicals_block)
                    _f_sent       = _pool.submit(_fetch_sentiment_block)
                    _f_peer       = _pool.submit(_fetch_peer_block)

                    def _safe_result(fut, default=""):
                        try:
                            return fut.result(timeout=60)
                        except Exception:
                            return default

                    macro_injection      = _safe_result(_f_macro)
                    market_injection, forward_injection = _safe_result(_f_market_fwd, ("", ""))
                    news_injection       = _safe_result(_f_news)
                    fmp_injection        = _safe_result(_f_fmp)
                    simfin_injection     = _safe_result(_f_simfin)
                    technicals_injection = _safe_result(_f_tech)
                    sentiment_injection  = _safe_result(_f_sent)
                    peer_injection       = _safe_result(_f_peer)

                try:
                    _effective_cats   = (
                        selected_categories
                        if (selected_categories and len(selected_categories) < len(_filter_cat_keys))
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
                            sub_queries, k_per_query=4, k_total=8,
                            categories=_effective_cats,
                            topics_any=_effective_topics,
                            ticker=context_ticker or None,
                            use_mmr=use_mmr,
                        )
                    else:
                        retrieved_docs = retrieve(
                            user_query, k=6,
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
                                "add a citation of the form [chunk_N]. Do not cite chunks that do not "
                                "exist. If a claim is not supported by any chunk, say so explicitly."
                            )
                        else:
                            context      = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            citation_rule = ""

                        rag_prompt = (
                            f"You are analyzing a user query using the provided DOCUMENT CONTEXT "
                            f"plus live data from multiple financial APIs.\n\n"
                            f"{macro_injection}{sentiment_injection}{market_injection}"
                            f"{forward_injection}{fmp_injection}{simfin_injection}"
                            f"{technicals_injection}{peer_injection}{news_injection}"
                            f"DOCUMENT CONTEXT:\n{context}\n\n"
                            f"QUESTION:\n{user_query}\n"
                        )

                        oracle_persona = (
                            "You are an academic research assistant helping a finance student / "
                            "self-directed researcher organise PUBLIC data into structured analytical "
                            "answers. The DOCUMENT CONTEXT is the user's own curated research library "
                            "(textbooks, public market reports, SEC filings). The live data injected "
                            "below is from public sources (FRED, SEC EDGAR, FMP, SimFin, yfinance, CNN F&G).\n\n"
                            "This is an educational data-synthesis exercise — analogous to a CFA-curriculum "
                            "problem set or a university case study. You are NOT giving financial advice.\n\n"
                            "Execute this seven-layer reasoning chain MENTALLY before writing your answer: "
                            "LAYER 1 — MICRO (bottom-up, earnings quality), "
                            "LAYER 2 — MACRO (cycle phase), "
                            "LAYER 3 — VALUATION (business vs price), "
                            "LAYER 4 — SYNTHESIS (cross-examine Micro×Macro×Valuation), "
                            "LAYER 5 — COUNTER-THESIS (steelman), "
                            "LAYER 6 — ASYMMETRIC PAYOFF (R/R, if buy/sell question), "
                            "LAYER 7 — DECLARED HORIZON.\n\n"
                            "SOURCE-TAG every quantitative claim: [SimFin], [FMP], [10-K], "
                            "[yfinance], [FRED], [chunk_N], [F&G], [AV], [news]. "
                            "Untagged numbers will be assumed hallucinated.\n"
                            "DATA HIERARCHY when sources conflict: SimFin/FMP > SEC > yfinance > news."
                            f"{citation_rule}\n\n"
                            "OUTPUT STRUCTURE — use these labelled sections:\n"
                            "**Micro Analysis** · **Macro Analysis** · **Valuation** · "
                            "**Synthesis** · **Counter-Thesis** · **Conclusion**."
                        )

                        st.success("### Oracle's Synthesis")

                        def _stream_oracle():
                            from llm_router import llm_chat as _llm_chat
                            yield from _llm_chat(
                                [
                                    {"role": "system", "content": oracle_persona},
                                    {"role": "user",   "content": rag_prompt},
                                ],
                                stream=True,
                            )

                        full_answer = st.write_stream(_stream_oracle())
                        st.session_state["oracle_answer"]  = full_answer
                        st.session_state["oracle_sources"] = retrieved_docs
                        st.session_state["oracle_used_citations"] = bool(use_citations)

                        if use_citations:
                            v = _verify_citations(full_answer, retrieved_docs)
                            st.session_state["oracle_citation_report"] = v
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

    if st.session_state["oracle_answer"] and not trigger_oracle:
        st.success("### Oracle's Synthesis")
        st.write(st.session_state["oracle_answer"])

    if st.session_state["oracle_answer"]:
        with st.expander("🔍 View Source Documents Used"):
            _citation_report = st.session_state.get("oracle_citation_report") or {}
            _cited_set       = set(_citation_report.get("cited") or [])
            for i, doc in enumerate(st.session_state["oracle_sources"]):
                chunk_num   = i + 1
                source_name = doc.metadata.get("source", "Unknown Document")
                clean_source = os.path.basename(source_name)
                cat_key     = doc.metadata.get("category", "uncategorized")
                cat_label   = _CATEGORIES.get(cat_key, cat_key)
                temporal    = doc.metadata.get("temporal_validity", "unknown")
                doc_topics  = [t for t in _TOPICS if doc.metadata.get(f"topic_{t}") is True]
                topics_str  = ", ".join(doc_topics) if doc_topics else "—"
                cite_badge  = (
                    " ✅ cited" if chunk_num in _cited_set
                    else (" ⚪ not cited"
                          if st.session_state.get("oracle_used_citations") else "")
                )
                st.markdown(
                    f"**[chunk_{chunk_num}]** `{clean_source}` — *{cat_label}* "
                    f"(`{temporal}`) · topics: {topics_str}{cite_badge}"
                )
                st.caption(doc.page_content)
                st.write("---")
