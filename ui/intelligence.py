import streamlit as st

from api import fetch_financial_news
from utils import sanitize_ticker


def render(app_data: dict) -> None:
    st.header("Market Intelligence")
    st.markdown("Live, unfiltered news feeds analyzed entirely offline by Llama 3.2.")

    news_ticker = sanitize_ticker(
        st.text_input("Target Asset for Reconnaissance", value="AAPL", key="intel_search").upper()
    )
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
                            ai_prompt = (
                                f"Analyze this financial news headline for the stock {news_ticker}: '{title}'. "
                                "Respond strictly in this exact format: "
                                "[BULLISH/BEARISH/NEUTRAL] - [One concise sentence explanation of why]."
                            )
                            oracle_persona = (
                                "You are 'The True Oracle', an elite financial AI. You must strictly obey the following rules:\n"
                                "1. The Logic-First Filter: Perform a Logical Audit defining the Domain of Discourse and isolating atomic propositions.\n"
                                "2. Probabilistic Calibration: Reject binary True/False. Treat new info as Evidence updating a Prior Belief."
                            )
                            from llm_router import llm_chat as _llm_chat
                            ai_analysis = _llm_chat([
                                {'role': 'system', 'content': oracle_persona},
                                {'role': 'user', 'content': ai_prompt},
                            ]).strip()

                            if "BULLISH" in ai_analysis.upper():
                                st.success(f"**AI Sentiment:** {ai_analysis}")
                            elif "BEARISH" in ai_analysis.upper():
                                st.error(f"**AI Sentiment:** {ai_analysis}")
                            else:
                                st.info(f"**AI Sentiment:** {ai_analysis}")
                        except Exception:
                            st.warning(
                                "AI Engine offline. Ensure the Ollama Mac app is running in your menu bar."
                            )
                    st.write("---")
