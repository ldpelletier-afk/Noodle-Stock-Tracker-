import datetime as _dt

import streamlit as st

from api import get_catalyst_news, _CATALYST_KEYWORD_MAP
from data_store import list_catalysts


def render(app_data: dict) -> None:
    st.header("📰 Catalyst News")
    st.caption(
        "Keyword-filtered news stream — headlines scored for policy / monetary / "
        "contract / court / earnings relevance and auto-tagged to upcoming "
        "catalysts whose tickers overlap."
    )

    _cn_c1, _cn_c2, _cn_c3 = st.columns([2, 2, 1])

    _all_upcoming_cats = list_catalysts(status="upcoming") + list_catalysts(status="live")
    _cat_tickers: list[str] = sorted(
        {t.upper() for c in _all_upcoming_cats for t in (c.get("tickers") or [])}
    )

    _cn_ticker_input = _cn_c1.text_input(
        "Tickers to scan (comma-separated)",
        value=", ".join(_cat_tickers[:15]),
        key="cn_ticker_input",
        help="Leave blank to scan all catalyst tickers. Add more with commas.",
    )
    _cn_tickers_raw = [t.strip().upper() for t in _cn_ticker_input.split(",") if t.strip()]
    _cn_tickers: tuple[str, ...] = tuple(_cn_tickers_raw or _cat_tickers[:20])

    _cn_all_cats  = ["(all)"] + list(_CATALYST_KEYWORD_MAP.keys())
    _cn_cat_filter = _cn_c2.selectbox(
        "Filter by category", options=_cn_all_cats, index=0, key="cn_cat_filter",
        format_func=lambda x: x.replace("_", " ").title() if x != "(all)" else "All categories",
    )

    _cn_min_score = _cn_c3.number_input(
        "Min score", min_value=1, max_value=10, value=1, step=1,
        key="cn_min_score",
        help="Minimum keyword-hit count to display a headline.",
    )

    _cn_col_r, _cn_col_i = st.columns([1, 5])
    _cn_refresh = _cn_col_r.button("🔄 Refresh", key="cn_refresh_btn")
    if _cn_refresh:
        get_catalyst_news.clear()

    if not _cn_tickers:
        st.info(
            "Add at least one ticker above, or import catalysts on the "
            "**🎯 Catalyst Calendar** / **🏛️ Monetary Policy** / "
            "**🏗️ Federal Contracts** / **⚖️ Court Docket** tabs first."
        )
    else:
        with st.spinner(f"Scanning news for {len(_cn_tickers)} ticker(s)…"):
            _cn_articles = get_catalyst_news(_cn_tickers, min_score=int(_cn_min_score))

        if _cn_cat_filter != "(all)":
            _cn_articles = [
                a for a in _cn_articles if _cn_cat_filter in a.get("scores", {})
            ]

        _cn_linked   = [a for a in _cn_articles if a.get("matched_catalysts")]
        _cn_unlinked = [a for a in _cn_articles if not a.get("matched_catalysts")]
        _s1, _s2, _s3 = st.columns(3)
        _s1.metric("Headlines found", len(_cn_articles))
        _s2.metric("Linked to a catalyst", len(_cn_linked))
        _s3.metric("Tickers scanned", len(_cn_tickers))
        st.divider()

        if not _cn_articles:
            st.info(
                "No catalyst-relevant headlines found for the selected tickers "
                "and score threshold. Try lowering **Min score** or adding more tickers."
            )
        else:
            _cn_cat_counts: dict[str, int] = {}
            for _a in _cn_articles:
                for _cat in _a.get("scores", {}):
                    _cn_cat_counts[_cat] = _cn_cat_counts.get(_cat, 0) + 1
            if _cn_cat_counts:
                _badge_parts = " · ".join(
                    f"**{k.title()}** {v}"
                    for k, v in sorted(_cn_cat_counts.items(), key=lambda x: -x[1])
                )
                st.caption(f"Category hits: {_badge_parts}")

            if _cn_linked:
                st.subheader(
                    f"🔗 Catalyst-linked headlines ({len(_cn_linked)})",
                    help="These headlines mention tickers that appear in at least one "
                         "upcoming/live catalyst.",
                )
                for _art in _cn_linked[:40]:
                    _art_ts   = _art.get("published_ts") or 0
                    _art_date = (
                        _dt.datetime.fromtimestamp(_art_ts).strftime("%b %d") if _art_ts else ""
                    )
                    _art_score_cats = " · ".join(
                        f"`{k.title()}` ×{v}"
                        for k, v in sorted(_art.get("scores", {}).items(), key=lambda x: -x[1])
                    )
                    _art_tickers_str = " ".join(
                        f"`{t}`" for t in _art.get("matched_tickers", [])
                    )
                    _art_cats_str = ", ".join(
                        c.get("title", "") for c in _art.get("matched_catalysts", [])[:3]
                    )
                    if len(_art.get("matched_catalysts", [])) > 3:
                        _art_cats_str += f" +{len(_art['matched_catalysts']) - 3} more"

                    with st.expander(
                        f"🔗 {_art.get('title', '(no title)')} "
                        f"— {_art.get('source', '')}  {_art_date}",
                        expanded=False,
                    ):
                        _exp_c1, _exp_c2 = st.columns([3, 2])
                        with _exp_c1:
                            if _art.get("summary"):
                                st.caption(_art["summary"][:280])
                            st.markdown(f"[📎 Open article]({_art.get('url', '#')})")
                        with _exp_c2:
                            st.caption(
                                f"**Source:** {_art.get('source', '—')}  \n"
                                f"**Published:** {_art.get('time', '—')}  \n"
                                f"**Tickers:** {_art_tickers_str or '—'}  \n"
                                f"**Score:** {_art.get('total_score', 0)} — {_art_score_cats}"
                            )
                            if _art_cats_str:
                                st.caption(f"**Catalysts:** {_art_cats_str}")
                            for _mc in _art.get("matched_catalysts", [])[:3]:
                                _mc_date = (
                                    _dt.datetime.fromtimestamp(_mc["event_date"]).strftime("%b %d, %Y")
                                    if _mc.get("event_date") else "—"
                                )
                                st.caption(f"🎯 **{_mc.get('title', '')}** · {_mc_date}")

            if _cn_unlinked:
                with st.expander(
                    f"📋 Other catalyst-relevant headlines ({len(_cn_unlinked)}) "
                    "— no direct ticker match",
                    expanded=False,
                ):
                    for _art in _cn_unlinked[:30]:
                        _art_ts   = _art.get("published_ts") or 0
                        _art_date = (
                            _dt.datetime.fromtimestamp(_art_ts).strftime("%b %d") if _art_ts else ""
                        )
                        _art_score_cats = " · ".join(
                            f"`{k.title()}` ×{v}"
                            for k, v in sorted(
                                _art.get("scores", {}).items(), key=lambda x: -x[1]
                            )
                        )
                        st.markdown(
                            f"- [{_art.get('title', '(no title)')}]({_art.get('url', '#')})  "
                            f"*{_art.get('source', '')} · {_art_date} · "
                            f"score {_art.get('total_score', 0)} — {_art_score_cats}*"
                        )
