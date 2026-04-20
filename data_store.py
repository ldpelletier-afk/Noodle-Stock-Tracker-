"""SQLite-backed persistence for NoodleStockTracker.

Drop-in replacement for the old portfolio.json layer:
- `load_data()` / `save_data(data)` preserve the original dict shape, so the
  Streamlit UI can keep working without structural changes.
- Adds concurrency-safe storage (WAL mode) so the watcher service and the UI
  can read/write simultaneously without corrupting state.
- Adds a transaction log and an alert-history table that the legacy JSON
  layer couldn't support.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

DB_FILE = os.getenv("NOODLE_DB_FILE", "portfolio.db")
LEGACY_JSON = os.getenv("NOODLE_LEGACY_JSON", "portfolio.json")

_DEFAULT_PEER_GROUPS = {
    "Tech Titans": ["AAPL", "MSFT", "GOOG", "META", "NVDA"],
    "Automakers": ["TSLA", "F", "GM", "TM"],
}
_DEFAULT_WATCHLIST = {"AAPL": 250.0, "GOOG": 250.0, "TSLA": 0.0}

_init_lock = threading.Lock()
_initialized = False


def _sanitize_ticker(ticker: str) -> str:
    if not ticker:
        return ticker
    ticker = ticker.upper().strip()
    if ticker.endswith(":CA"):
        return ticker.replace(":CA", ".TO")
    return ticker


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_FILE, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    finally:
        conn.close()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS portfolios (
            name TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS holdings (
            portfolio_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL,
            average_cost REAL NOT NULL,
            PRIMARY KEY (portfolio_name, ticker),
            FOREIGN KEY (portfolio_name) REFERENCES portfolios(name) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS watch_list (
            ticker TEXT PRIMARY KEY,
            target_price REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS peer_groups (
            name TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS peer_group_members (
            group_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            PRIMARY KEY (group_name, ticker),
            FOREIGN KEY (group_name) REFERENCES peer_groups(name) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            portfolio_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_transactions_portfolio
            ON transactions(portfolio_name, ts);

        CREATE TABLE IF NOT EXISTS alert_history (
            ticker TEXT NOT NULL,
            kind TEXT NOT NULL,
            target REAL NOT NULL,
            last_fired_at INTEGER NOT NULL,
            last_price REAL,
            PRIMARY KEY (ticker, kind, target)
        );
        """
    )


def _maybe_seed_defaults(conn: sqlite3.Connection) -> None:
    has_any = conn.execute(
        "SELECT 1 FROM portfolios LIMIT 1"
    ).fetchone() or conn.execute(
        "SELECT 1 FROM watch_list LIMIT 1"
    ).fetchone()
    if has_any:
        return

    conn.execute("INSERT OR IGNORE INTO portfolios(name) VALUES (?)", ("My First Portfolio",))
    for ticker, target in _DEFAULT_WATCHLIST.items():
        conn.execute(
            "INSERT OR IGNORE INTO watch_list(ticker, target_price) VALUES (?, ?)",
            (ticker, target),
        )
    for group, tickers in _DEFAULT_PEER_GROUPS.items():
        conn.execute("INSERT OR IGNORE INTO peer_groups(name) VALUES (?)", (group,))
        for t in tickers:
            conn.execute(
                "INSERT OR IGNORE INTO peer_group_members(group_name, ticker) VALUES (?, ?)",
                (group, t),
            )


def _migrate_from_json(conn: sqlite3.Connection) -> None:
    """One-shot import of the legacy portfolio.json if present."""
    if not os.path.exists(LEGACY_JSON):
        return
    # Don't re-migrate if there's already data.
    if conn.execute("SELECT 1 FROM portfolios LIMIT 1").fetchone():
        return

    try:
        with open(LEGACY_JSON, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    # Handle the oldest shape: flat {ticker: target}
    if data and isinstance(list(data.values())[0], (int, float)):
        data = {
            "portfolios": {"My First Portfolio": {}},
            "watch_list_targets": data,
            "peer_groups": {},
        }

    _write_dict(conn, data)

    backup = LEGACY_JSON + ".migrated"
    try:
        os.replace(LEGACY_JSON, backup)
    except OSError:
        pass


def init_db() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        with _connect() as conn:
            _create_schema(conn)
            _migrate_from_json(conn)
            _maybe_seed_defaults(conn)
        _initialized = True


# ---------- Dict-shape load/save (drop-in for the old JSON API) ----------

def load_data() -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        portfolios: dict[str, dict[str, dict[str, float]]] = {}
        for row in conn.execute("SELECT name FROM portfolios"):
            portfolios[row["name"]] = {}
        for row in conn.execute(
            "SELECT portfolio_name, ticker, quantity, average_cost FROM holdings"
        ):
            portfolios.setdefault(row["portfolio_name"], {})[row["ticker"]] = {
                "quantity": row["quantity"],
                "average_cost": row["average_cost"],
            }

        watch_list = {
            row["ticker"]: row["target_price"]
            for row in conn.execute("SELECT ticker, target_price FROM watch_list")
        }

        peer_groups: dict[str, list[str]] = {
            row["name"]: []
            for row in conn.execute("SELECT name FROM peer_groups")
        }
        for row in conn.execute(
            "SELECT group_name, ticker FROM peer_group_members"
        ):
            peer_groups.setdefault(row["group_name"], []).append(row["ticker"])

    return {
        "portfolios": portfolios,
        "watch_list_targets": watch_list,
        "peer_groups": peer_groups,
    }


def _write_dict(conn: sqlite3.Connection, data: dict[str, Any]) -> None:
    portfolios = data.get("portfolios", {}) or {}
    watch_list = data.get("watch_list_targets", {}) or {}
    peer_groups = data.get("peer_groups", {}) or {}

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM holdings")
        conn.execute("DELETE FROM portfolios")
        conn.execute("DELETE FROM watch_list")
        conn.execute("DELETE FROM peer_group_members")
        conn.execute("DELETE FROM peer_groups")

        for name, assets in portfolios.items():
            conn.execute("INSERT OR IGNORE INTO portfolios(name) VALUES (?)", (name,))
            for ticker, pos in (assets or {}).items():
                ticker = _sanitize_ticker(ticker)
                conn.execute(
                    """INSERT INTO holdings(portfolio_name, ticker, quantity, average_cost)
                       VALUES (?, ?, ?, ?)""",
                    (
                        name,
                        ticker,
                        float(pos.get("quantity", 0.0)),
                        float(pos.get("average_cost", 0.0)),
                    ),
                )

        for ticker, target in watch_list.items():
            conn.execute(
                "INSERT INTO watch_list(ticker, target_price) VALUES (?, ?)",
                (_sanitize_ticker(ticker), float(target or 0.0)),
            )

        for group, tickers in peer_groups.items():
            conn.execute("INSERT OR IGNORE INTO peer_groups(name) VALUES (?)", (group,))
            for t in tickers or []:
                conn.execute(
                    "INSERT OR IGNORE INTO peer_group_members(group_name, ticker) VALUES (?, ?)",
                    (group, _sanitize_ticker(t)),
                )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def save_data(data: dict[str, Any]) -> None:
    init_db()
    with _connect() as conn:
        _write_dict(conn, data)


# ---------- Transaction log ----------

def log_transaction(
    portfolio_name: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    ts: int | None = None,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO transactions(ts, portfolio_name, ticker, action, quantity, price)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                int(ts if ts is not None else time.time()),
                portfolio_name,
                _sanitize_ticker(ticker),
                action.upper(),
                float(quantity),
                float(price),
            ),
        )


def fetch_transactions(portfolio_name: str | None = None) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        if portfolio_name:
            rows = conn.execute(
                """SELECT id, ts, portfolio_name, ticker, action, quantity, price
                   FROM transactions WHERE portfolio_name = ? ORDER BY ts DESC""",
                (portfolio_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, ts, portfolio_name, ticker, action, quantity, price
                   FROM transactions ORDER BY ts DESC"""
            ).fetchall()
    return [dict(r) for r in rows]


# ---------- Watcher alert dedup ----------

def should_fire_alert(
    ticker: str,
    kind: str,
    target: float,
    cooldown_seconds: int = 6 * 60 * 60,
) -> bool:
    """Return True if an alert for (ticker, kind, target) hasn't fired recently."""
    init_db()
    now = int(time.time())
    with _connect() as conn:
        row = conn.execute(
            """SELECT last_fired_at FROM alert_history
               WHERE ticker = ? AND kind = ? AND target = ?""",
            (ticker, kind, float(target)),
        ).fetchone()
    if row is None:
        return True
    return (now - int(row["last_fired_at"])) >= cooldown_seconds


def record_alert(ticker: str, kind: str, target: float, price: float) -> None:
    init_db()
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """INSERT INTO alert_history(ticker, kind, target, last_fired_at, last_price)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(ticker, kind, target) DO UPDATE SET
                 last_fired_at = excluded.last_fired_at,
                 last_price = excluded.last_price""",
            (ticker, kind, float(target), now, float(price)),
        )


def get_watch_targets() -> dict[str, float]:
    """Convenience accessor for the watcher service."""
    init_db()
    with _connect() as conn:
        return {
            row["ticker"]: row["target_price"]
            for row in conn.execute(
                "SELECT ticker, target_price FROM watch_list WHERE target_price > 0"
            )
        }
