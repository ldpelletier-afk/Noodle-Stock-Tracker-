import time

import pandas as pd
import streamlit as st

from api import (
    fetch_all_news,
    fetch_financial_highlights,
    fetch_financial_statements,
    fetch_live_prices,
    fetch_recent_sec_filings,
    fetch_sec_filing,
    fetch_url_metadata,
)
from data_store import (
    add_favorite,
    delete_saved_article,
    list_favorites,
    list_saved_articles,
    remove_favorite,
    save_article,
    update_favorite,
    update_saved_article_note,
)
from utils import format_large_number, sanitize_ticker


def render(app_data: dict) -> None:
    st.header("⭐ Favorite Stocks")
    st.caption(
        "A curated, high-attention list — track goals, notes, financials, SEC filings, "
        "and noteworthy articles per stock. Articles you save are tied to the specific "
        "ticker you saved them under."
    )

    favorites = list_favorites()
    fav_tickers = list(favorites.keys())

    _fav_ready = "favorites" in st.session_state._lazy_loaded

    if fav_tickers:
        if not _fav_ready:
            fav_prices = {}
            st.caption(
                "📡 Live prices not loaded yet — click **⚡ Load all live data** on "
                "the 🏠 Dashboard tab to populate this summary."
            )
        else:
            with st.spinner("Fetching live data for your favorites..."):
                fav_prices = fetch_live_prices(fav_tickers)

        summary_rows = []
        for t in fav_tickers:
            fav = favorites[t]
            live = fav_prices.get(t, {}) or {}
            price = live.get("price")
            change = live.get("change")
            goal = fav.get("goal_price")
            if price is not None and goal:
                progress = ((price - goal) / goal) * 100
            else:
                progress = None
            summary_rows.append({
                "Ticker": t,
                "Price": price,
                "Day %": change,
                "Goal": goal,
                "vs Goal %": progress,
                "Notes": (fav.get("notes") or "").splitlines()[0][:80] if fav.get("notes") else "",
            })

        summary_df = pd.DataFrame(summary_rows)
        st.subheader("At-a-glance")
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Day %": st.column_config.NumberColumn(format="%.2f%%"),
                "Goal": st.column_config.NumberColumn(format="$%.2f"),
                "vs Goal %": st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )
    else:
        st.info("No favorites yet. Add one below to start tracking.")

    st.divider()

    if fav_tickers:
        focus_ticker = st.selectbox("Focus stock", fav_tickers, key="fav_focus_select")
        fav = favorites[focus_ticker]

        live = fav_prices.get(focus_ticker, {}) or {}
        price = live.get("price")
        change = live.get("change")
        goal = fav.get("goal_price")

        hdr1, hdr2, hdr3, hdr4 = st.columns(4)
        hdr1.metric(
            "Live Price",
            f"${price:.2f}" if price is not None else "N/A",
            delta=f"{change:+.2f}%" if change is not None else None,
        )
        hdr2.metric("Goal Price", f"${goal:.2f}" if goal else "— not set —")
        if price is not None and goal:
            delta = price - goal
            pct = (delta / goal) * 100
            hdr3.metric("Δ vs Goal", f"${delta:+.2f}", delta=f"{pct:+.2f}%")
        else:
            hdr3.metric("Δ vs Goal", "—")
        hdr4.metric(
            "Added",
            time.strftime("%Y-%m-%d", time.localtime(fav.get("added_at") or 0))
            if fav.get("added_at") else "—",
        )

        with st.expander("📝 Notes, goal, and position", expanded=True):
            with st.form(key=f"fav_meta_form_{focus_ticker}", clear_on_submit=False):
                col_n, col_g = st.columns([3, 1])
                with col_n:
                    new_notes = st.text_area(
                        "Notes",
                        value=fav.get("notes") or "",
                        height=140,
                        key=f"fav_notes_{focus_ticker}",
                        help="Your thesis, watch criteria, or anything else worth remembering.",
                    )
                with col_g:
                    new_goal = st.number_input(
                        "Goal price ($)",
                        min_value=0.0,
                        value=float(fav.get("goal_price") or 0.0),
                        step=0.01,
                        format="%.2f",
                        key=f"fav_goal_{focus_ticker}",
                        help=(
                            "Accepts dollars and cents (e.g. 247.85). "
                            "Set to any value > 0 to save a goal. Leave at 0 to skip. "
                            "Tick 'Clear goal' to explicitly remove an existing goal."
                        ),
                    )
                    clear_goal = st.checkbox(
                        "Clear goal", value=False, key=f"fav_clear_goal_{focus_ticker}"
                    )
                new_position = st.text_input(
                    "Position note (e.g., '10 shares @ $150, added 2024-11-01')",
                    value=fav.get("position_note") or "",
                    key=f"fav_pos_{focus_ticker}",
                )
                submitted = st.form_submit_button("💾 Save changes")

            if submitted:
                kwargs = {"notes": new_notes, "position_note": new_position}
                if clear_goal:
                    kwargs["clear_goal"] = True
                elif new_goal > 0:
                    kwargs["goal_price"] = float(new_goal)
                update_favorite(focus_ticker, **kwargs)
                saved_bits = ["notes", "position"]
                if clear_goal:
                    saved_bits.append("goal cleared")
                elif new_goal > 0:
                    saved_bits.append(f"goal=${new_goal:.2f}")
                st.toast(f"Saved {focus_ticker}: " + ", ".join(saved_bits), icon="💾")
                st.rerun()

        st.subheader("💰 Financial Highlights")
        with st.spinner(f"Loading financials for {focus_ticker}..."):
            try:
                highlights = fetch_financial_highlights(focus_ticker)
            except Exception as e:
                highlights = {}
                st.warning(f"Could not fetch financial highlights: {e}")

        if highlights.get("long_name"):
            st.caption(
                f"**{highlights['long_name']}** · {highlights.get('sector') or '—'} "
                f"/ {highlights.get('industry') or '—'} · {highlights.get('currency') or ''}"
            )

        fh1, fh2, fh3, fh4 = st.columns(4)
        fh1.metric("Revenue (TTM/FY)", format_large_number(highlights.get("revenue")))
        fh2.metric("Net Income", format_large_number(highlights.get("net_income")))
        fh3.metric("Free Cash Flow", format_large_number(highlights.get("fcf")))
        fh4.metric(
            "Profit Margin",
            f"{highlights['profit_margin']:.2f}%"
            if highlights.get("profit_margin") is not None else "N/A",
        )

        fh5, fh6, fh7, fh8 = st.columns(4)
        fh5.metric("Total Debt", format_large_number(highlights.get("total_debt")))
        fh6.metric("Cash", format_large_number(highlights.get("cash")))
        fh7.metric("Net Debt", format_large_number(highlights.get("net_debt")))
        fh8.metric("Market Cap", format_large_number(highlights.get("market_cap")))

        fh9, fh10, fh11, fh12 = st.columns(4)
        fh9.metric("P/E (Trailing)", f"{highlights['pe_trailing']:.2f}" if highlights.get("pe_trailing") else "N/A")
        fh10.metric("P/E (Forward)", f"{highlights['pe_forward']:.2f}" if highlights.get("pe_forward") else "N/A")
        fh11.metric("EPS (TTM)", f"${highlights['eps_trailing']:.2f}" if highlights.get("eps_trailing") else "N/A")
        fh12.metric("Div Yield", f"{highlights['dividend_yield']:.2f}%" if highlights.get("dividend_yield") else "N/A")

        with st.expander("📄 Full Statements (Income / Balance / Cash Flow)"):
            try:
                statements = fetch_financial_statements(focus_ticker)
            except Exception as e:
                statements = {"income": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame()}
                st.warning(f"Could not fetch statements: {e}")

            stmt_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            for stmt_tab, key, label in zip(
                stmt_tabs,
                ["income", "balance_sheet", "cashflow"],
                ["Income Statement", "Balance Sheet", "Cash Flow"],
            ):
                with stmt_tab:
                    df_stmt = statements.get(key, pd.DataFrame())
                    if df_stmt is None or df_stmt.empty:
                        st.info(f"No {label.lower()} available from yfinance.")
                    else:
                        try:
                            df_show = df_stmt.copy()
                            df_show.columns = [
                                c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                                for c in df_show.columns
                            ]
                        except Exception:
                            df_show = df_stmt
                        st.dataframe(df_show, use_container_width=True)

        st.subheader("📋 Recent SEC Filings")
        with st.spinner(f"Loading SEC filings for {focus_ticker}..."):
            filings = fetch_recent_sec_filings(focus_ticker, n=10)
        if not filings:
            st.info(
                "No recent SEC filings found (or ticker not in SEC EDGAR — non-US "
                "listings and ETFs typically won't appear)."
            )
        else:
            filings_df = pd.DataFrame([
                {"Form": f["form"], "Date": f["date"], "Link": f["url"]} for f in filings
            ])
            st.dataframe(
                filings_df,
                use_container_width=True,
                hide_index=True,
                column_config={"Link": st.column_config.LinkColumn("Link", display_text="📄 Open")},
            )
            with st.expander("🔎 Fetch text of a specific filing type"):
                form_options = sorted({f["form"] for f in filings})
                if form_options:
                    pick_form = st.selectbox("Form type", form_options, key=f"fav_form_pick_{focus_ticker}")
                    if st.button("Fetch most recent text", key=f"fav_fetch_filing_{focus_ticker}"):
                        with st.spinner(f"Fetching {pick_form}..."):
                            text, url_or_err = fetch_sec_filing(focus_ticker, form_type=pick_form)
                        if text:
                            st.success(f"Fetched {pick_form} — {url_or_err}")
                            st.text_area(
                                f"{pick_form} text (first 10k chars)",
                                value=text[:10000],
                                height=300,
                                key=f"fav_filing_text_{focus_ticker}_{pick_form}",
                            )
                        else:
                            st.warning(url_or_err)

        st.subheader("📰 News")
        news_col1, news_col2 = st.tabs(["🌐 Auto feed (multi-source)", "⭐ Saved articles"])

        with news_col1:
            refresh_news = st.button("🔄 Refresh feed", key=f"fav_news_refresh_{focus_ticker}")
            if refresh_news:
                fetch_all_news.clear()
            with st.spinner("Fetching news from all configured sources..."):
                try:
                    articles = fetch_all_news(focus_ticker)
                except Exception as e:
                    articles = []
                    st.warning(f"News fetch error: {e}")

            if not articles:
                st.info("No news articles found for this ticker.")
            else:
                st.caption(
                    f"{len(articles)} article(s), newest first. Sources include Yahoo, "
                    "Google News (incl. WSJ/FT/Bloomberg/Reuters site filters), Seeking Alpha, "
                    "and any configured Finnhub/NewsAPI/MarketAux keys."
                )
                for art in articles[:40]:
                    when = (
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(art["published_ts"]))
                        if art.get("published_ts") else art.get("time") or "Recent"
                    )
                    c_art, c_btn = st.columns([5, 1])
                    with c_art:
                        st.markdown(
                            f"**[{art['title']}]({art['url']})**  \n"
                            f"*{art.get('source') or 'Source'}* · {when}"
                        )
                        if art.get("summary"):
                            st.caption(art["summary"][:280] + ("…" if len(art["summary"]) > 280 else ""))
                    with c_btn:
                        if st.button(
                            "⭐ Save",
                            key=f"fav_save_feed_{focus_ticker}_{hash(art['url'])}",
                            help="Save this article to this ticker's noteworthy list",
                        ):
                            new_id = save_article(
                                ticker=focus_ticker,
                                url=art["url"],
                                title=art.get("title"),
                                source=art.get("source"),
                                note="",
                                published_at=art.get("published_ts") or None,
                            )
                            if new_id:
                                st.toast(f"Saved to {focus_ticker}'s noteworthy list", icon="⭐")
                            else:
                                st.toast("Already saved for this ticker", icon="ℹ️")

        with news_col2:
            st.caption(
                f"Articles below are saved **specifically to {focus_ticker}**. "
                "Paste any URL (WSJ+, FT, blog, analyst report, etc.) — we'll pull "
                "the title/source from the page's public Open Graph metadata "
                "(works even for paywalled articles, since OG tags are public)."
            )
            with st.form(key=f"fav_url_form_{focus_ticker}", clear_on_submit=True):
                paste_url = st.text_input(
                    "URL to save",
                    placeholder="https://www.wsj.com/articles/...",
                    key=f"fav_paste_url_{focus_ticker}",
                )
                paste_note = st.text_area(
                    "Optional note (why is this noteworthy?)",
                    key=f"fav_paste_note_{focus_ticker}",
                    height=80,
                )
                submit_url = st.form_submit_button("💾 Save URL to this stock")
                if submit_url and paste_url.strip():
                    with st.spinner("Fetching article metadata..."):
                        meta = fetch_url_metadata(paste_url.strip())
                    new_id = save_article(
                        ticker=focus_ticker,
                        url=paste_url.strip(),
                        title=meta.get("title") or paste_url.strip(),
                        source=meta.get("source") or "",
                        note=paste_note or "",
                        published_at=meta.get("published_ts") or None,
                    )
                    if new_id:
                        st.success(f"Saved to {focus_ticker}: {meta.get('title') or paste_url.strip()}")
                    else:
                        st.info(f"This URL is already saved for {focus_ticker}.")
                    st.rerun()

            saved = list_saved_articles(focus_ticker)
            st.write(f"**{len(saved)} saved article(s)** for `{focus_ticker}`")
            for art in saved:
                pub_ts = art.get("published_at") or art.get("saved_at")
                when = time.strftime("%Y-%m-%d", time.localtime(pub_ts)) if pub_ts else "—"
                with st.container(border=True):
                    c_info, c_note, c_act = st.columns([4, 3, 1])
                    with c_info:
                        st.markdown(
                            f"**[{art.get('title') or art['url']}]({art['url']})**  \n"
                            f"*{art.get('source') or 'Source'}* · {when} · "
                            f"saved {time.strftime('%Y-%m-%d', time.localtime(art['saved_at']))}"
                        )
                    with c_note:
                        new_note = st.text_input(
                            "Note",
                            value=art.get("note") or "",
                            key=f"fav_saved_note_{art['id']}",
                            label_visibility="collapsed",
                            placeholder="Your note...",
                        )
                        if new_note != (art.get("note") or ""):
                            if st.button("Save note", key=f"fav_saved_note_btn_{art['id']}"):
                                update_saved_article_note(art["id"], new_note)
                                st.toast("Note updated", icon="✏️")
                                st.rerun()
                    with c_act:
                        if st.button("🗑️", key=f"fav_saved_del_{art['id']}", help="Remove"):
                            delete_saved_article(art["id"])
                            st.toast("Removed", icon="🗑️")
                            st.rerun()

    st.divider()

    col_add, col_del = st.columns(2)
    with col_add:
        st.subheader("⭐ Add a favorite")
        add_input = st.text_input("Ticker (e.g. AAPL, RY.TO, SHOP)", key="fav_add_input")
        if st.button("Add to favorites", key="fav_add_btn", use_container_width=True):
            sym = sanitize_ticker(add_input)
            if sym:
                if add_favorite(sym):
                    st.toast(f"{sym} added to favorites", icon="⭐")
                    st.rerun()
                else:
                    st.warning(f"{sym} is already in your favorites.")

    with col_del:
        st.subheader("🗑️ Remove a favorite")
        if fav_tickers:
            rm_choice = st.selectbox("Select favorite to remove", fav_tickers, key="fav_rm_select")
            if st.button(
                "Remove (also deletes its saved articles)",
                type="primary",
                key="fav_rm_btn",
                use_container_width=True,
            ):
                remove_favorite(rm_choice)
                st.toast(f"{rm_choice} removed", icon="🗑️")
                st.rerun()
        else:
            st.caption("No favorites to remove.")
