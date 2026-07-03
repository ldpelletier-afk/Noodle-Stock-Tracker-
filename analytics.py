"""Analytical query layer — DuckDB on top of the SQLite portfolio store.

DuckDB is an embedded OLAP engine that reads SQLite files directly via the
``sqlite_scanner`` extension. This module exposes high-level analytical
queries (catalyst exposure by ticker, density-by-month, top sectors, etc.)
that would be slow or awkward to express in vanilla SQLite — DuckDB has
real window functions, fast aggregations, and zero per-query setup cost
when run in :memory: against a small DB.

Why DuckDB and not just pandas?
- The catalyst table can be filtered, joined, and aggregated in one SQL
  pass without round-tripping through Python. Once we start saving news
  history (next phase), DuckDB scales to millions of rows where pandas
  starts thrashing.
- Same query language as Postgres / BigQuery — easy to graduate to a
  hosted warehouse later if the corpus outgrows local files.

Public API
----------
- ``catalyst_density_by_month()``      → DataFrame (month, count, type breakdown)
- ``ticker_exposure()``                → DataFrame (ticker, n_catalysts, types, ...)
- ``sector_exposure()``                → DataFrame (sector, n_catalysts, ...)
- ``imminent_catalysts(window_days)``  → DataFrame of the next N days' events
- ``run_query(sql)``                   → arbitrary read-only SQL (escape hatch)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

# DB path — same convention as data_store.py.
_DB_FILE = os.getenv("NOODLE_DB_FILE", "portfolio.db")


@contextmanager
def _conn() -> Iterator["duckdb.DuckDBPyConnection"]:  # type: ignore  # noqa: F821
    """Open a fresh DuckDB connection, attach the SQLite portfolio DB, yield.

    A new connection per call keeps the lifecycle predictable across
    Streamlit's re-run model (no stale handles after a hot reload). The
    sqlite_scanner extension is installed on first use.
    """
    import duckdb  # local import keeps cold-start cheap
    con = duckdb.connect(database=":memory:")
    try:
        # Install + load the SQLite reader once per connection.
        con.execute("INSTALL sqlite; LOAD sqlite;")
        # ``ATTACH`` makes the SQLite tables queryable as ``portfolio.<table>``.
        # READ_ONLY=true avoids any chance of corrupting the live DB.
        con.execute(
            f"ATTACH '{_DB_FILE}' AS portfolio (TYPE sqlite, READ_ONLY)"
        )
        yield con
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Pre-canned analytical queries
# ---------------------------------------------------------------------------

def catalyst_density_by_month(months_ahead: int = 12) -> pd.DataFrame:
    """Count of upcoming catalysts grouped by event month + type.

    Returns long-form: ``(month, catalyst_type, count)``. Pivot in the UI
    layer if you want a wide table for plotting.
    """
    sql = f"""
        WITH events AS (
            SELECT
                date_trunc(
                    'month',
                    to_timestamp(event_date)
                )                                    AS month,
                catalyst_type
            FROM portfolio.political_catalysts
            WHERE status IN ('upcoming', 'live')
              AND event_date BETWEEN
                    extract(epoch FROM now())
                AND extract(epoch FROM now() + INTERVAL '{months_ahead}' MONTH)
        )
        SELECT
            strftime(month, '%Y-%m')   AS month,
            catalyst_type,
            count(*)                   AS count
        FROM events
        GROUP BY month, catalyst_type
        ORDER BY month, catalyst_type
    """
    with _conn() as con:
        return con.execute(sql).df()


def ticker_exposure(top_n: int = 25) -> pd.DataFrame:
    """For every ticker that appears in any upcoming catalyst, list:
    catalyst count, distinct types touched, and earliest/latest event date.

    Tickers are stored comma-separated in SQLite, so we explode them with
    DuckDB's ``string_split`` + ``unnest``.
    """
    sql = f"""
        WITH exploded AS (
            SELECT
                trim(t) AS ticker,
                catalyst_type,
                event_date,
                title
            FROM portfolio.political_catalysts,
                 unnest(string_split(coalesce(tickers, ''), ',')) AS u(t)
            WHERE status IN ('upcoming', 'live')
              AND length(trim(t)) > 0
        )
        SELECT
            upper(ticker)                     AS ticker,
            count(*)                          AS n_catalysts,
            count(DISTINCT catalyst_type)     AS n_types,
            string_agg(DISTINCT catalyst_type, ', ')
                                              AS types,
            strftime(
                to_timestamp(min(event_date)),
                '%Y-%m-%d'
            )                                 AS earliest,
            strftime(
                to_timestamp(max(event_date)),
                '%Y-%m-%d'
            )                                 AS latest
        FROM exploded
        GROUP BY upper(ticker)
        ORDER BY n_catalysts DESC, ticker
        LIMIT {top_n}
    """
    with _conn() as con:
        return con.execute(sql).df()


def sector_exposure(top_n: int = 20) -> pd.DataFrame:
    """Same shape as ``ticker_exposure`` but rolled up by sector tag."""
    sql = f"""
        WITH exploded AS (
            SELECT
                trim(s) AS sector,
                catalyst_type,
                event_date
            FROM portfolio.political_catalysts,
                 unnest(string_split(coalesce(sectors, ''), ',')) AS u(s)
            WHERE status IN ('upcoming', 'live')
              AND length(trim(s)) > 0
        )
        SELECT
            sector,
            count(*)                          AS n_catalysts,
            count(DISTINCT catalyst_type)     AS n_types,
            string_agg(DISTINCT catalyst_type, ', ')
                                              AS types
        FROM exploded
        GROUP BY sector
        ORDER BY n_catalysts DESC, sector
        LIMIT {top_n}
    """
    with _conn() as con:
        return con.execute(sql).df()


def imminent_catalysts(window_days: int = 30) -> pd.DataFrame:
    """All catalysts firing in the next ``window_days``, sorted by date.

    Useful as a single-query 'what's about to hit?' view that the Catalyst
    News tab can join against headlines.
    """
    sql = f"""
        SELECT
            id,
            title,
            catalyst_type,
            category,
            tickers,
            sectors,
            strftime(to_timestamp(event_date), '%Y-%m-%d') AS event_date,
            event_date                                      AS event_ts,
            (event_date - extract(epoch FROM now())) / 86400.0
                                                             AS days_until
        FROM portfolio.political_catalysts
        WHERE status IN ('upcoming', 'live')
          AND event_date BETWEEN
                extract(epoch FROM now())
            AND extract(epoch FROM now() + INTERVAL '{window_days}' DAY)
        ORDER BY event_date ASC
    """
    with _conn() as con:
        return con.execute(sql).df()


def run_query(sql: str) -> pd.DataFrame:
    """Escape hatch for ad-hoc analytical SQL.

    The portfolio DB is attached as ``portfolio`` — qualify table names
    accordingly, e.g. ``SELECT * FROM portfolio.political_catalysts LIMIT 10``.
    The connection is opened READ_ONLY so this is safe to expose in a UI.
    """
    with _conn() as con:
        return con.execute(sql).df()
