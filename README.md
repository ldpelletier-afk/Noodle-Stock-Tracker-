# Noodle Stock Tracker

"The True Oracle" — a personal stock-portfolio tracker and research assistant.
Streamlit UI for tracking holdings/watchlists, an LLM-backed research agent
(RAG over ingested articles/filings), a price watcher that can page you on
Telegram, and a grab-bag of optional third-party data sources for deeper
fundamentals, macro, and news.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally, with a model pulled — this is
  the default LLM backend and needs no API key:
  ```sh
  ollama pull llama3.2
  ```
  (Anthropic/OpenAI can be used instead via `LLM_BACKEND` / the sidebar
  dropdown, but that requires your own key for that provider.)
- Playwright's browser binaries, for JS-rendered page scraping:
  ```sh
  playwright install chromium
  ```

## Setup

```sh
git clone https://github.com/ldpelletier-afk/Noodle-Stock-Tracker-.git
cd Noodle-Stock-Tracker-

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium

cp .env.example .env   # fill in whatever optional keys you want, see below
```

## Run

```sh
.venv/bin/streamlit run "Stock Tracker.py"
```

Opens at http://localhost:8501.

### Optional: price watcher (Telegram alerts)

```sh
.venv/bin/python watcher.py
```

Polls your watch targets and pings a Telegram bot when a price crosses a
buy target. Needs `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `.env` —
without them it just logs a warning and does nothing.

### Optional: Docker

```sh
docker compose up
```

Runs the Streamlit app and the watcher as two containers. The watcher
container reads `.env` for its Telegram tokens.

## Configuration

Nothing above is required to get a working app — core price/fundamentals
data comes from `yfinance` (free, no key). Everything in `.env.example` is an
**optional** enrichment source with its own fallback; the app degrades
gracefully (fewer data sources, no alerts) if a given key is missing. See
`.env.example` for the full list and what each one unlocks.

## Data storage

`portfolio.db` (SQLite), `chroma_db/` (vector store for the RAG agent), and
`temp_pdfs/` are created locally on first run and are gitignored.

## Stack

Streamlit, yfinance, pandas/plotly, LangChain + Chroma (RAG over ingested
articles/filings), litellm (unified Anthropic/OpenAI/Ollama router), FastAPI-free
— everything runs as a single Streamlit process plus an optional watcher.
