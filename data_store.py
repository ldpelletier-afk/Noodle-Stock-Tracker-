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


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "transactions", "cost_basis"):
        conn.execute("ALTER TABLE transactions ADD COLUMN cost_basis REAL")

    # One-shot: migrate the old flat watch_list into the new named-list structure.
    # Only runs when watch_lists is empty (first upgrade from a pre-multi-list DB).
    if not conn.execute("SELECT 1 FROM watch_lists LIMIT 1").fetchone():
        old_items = conn.execute(
            "SELECT ticker, target_price FROM watch_list"
        ).fetchall()
        if old_items:
            now = int(time.time())
            conn.execute(
                "INSERT OR IGNORE INTO watch_lists(name, sort_order, created_at)"
                " VALUES (?,0,?)",
                ("General", now),
            )
            for row in old_items:
                conn.execute(
                    "INSERT OR IGNORE INTO watch_list_items"
                    "(list_name, ticker, target_price) VALUES (?,?,?)",
                    ("General", row["ticker"], row["target_price"]),
                )


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

        -- Named, user-created watchlists (replaces the single flat watch_list)
        CREATE TABLE IF NOT EXISTS watch_lists (
            name TEXT PRIMARY KEY,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS watch_list_items (
            list_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            target_price REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (list_name, ticker),
            FOREIGN KEY (list_name)
                REFERENCES watch_lists(name) ON DELETE CASCADE ON UPDATE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_wli_list
            ON watch_list_items(list_name);

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
            price REAL NOT NULL,
            cost_basis REAL
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

        CREATE TABLE IF NOT EXISTS favorite_stocks (
            ticker TEXT PRIMARY KEY,
            notes TEXT NOT NULL DEFAULT '',
            goal_price REAL,
            position_note TEXT NOT NULL DEFAULT '',
            added_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS saved_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            source TEXT,
            note TEXT NOT NULL DEFAULT '',
            saved_at INTEGER NOT NULL,
            published_at INTEGER,
            UNIQUE(ticker, url)
        );
        CREATE INDEX IF NOT EXISTS idx_saved_articles_ticker
            ON saved_articles(ticker, saved_at DESC);
        """
    )


def _maybe_seed_defaults(conn: sqlite3.Connection) -> None:
    has_any = (
        conn.execute("SELECT 1 FROM portfolios LIMIT 1").fetchone()
        or conn.execute("SELECT 1 FROM watch_lists LIMIT 1").fetchone()
    )
    if has_any:
        return

    now = int(time.time())
    conn.execute("INSERT OR IGNORE INTO portfolios(name) VALUES (?)", ("My First Portfolio",))

    # Seed a default "General" watchlist in the new multi-list tables.
    conn.execute(
        "INSERT OR IGNORE INTO watch_lists(name, sort_order, created_at) VALUES (?,0,?)",
        ("General", now),
    )
    for ticker, target in _DEFAULT_WATCHLIST.items():
        conn.execute(
            "INSERT OR IGNORE INTO watch_list_items(list_name, ticker, target_price)"
            " VALUES (?,?,?)",
            ("General", ticker, target),
        )
        # Also seed the legacy flat table so the watcher still works on first run.
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
            _apply_migrations(conn)
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

        # Named watchlists (new multi-list structure)
        watchlists: dict[str, dict[str, float]] = {}
        for row in conn.execute(
            "SELECT name FROM watch_lists ORDER BY sort_order, name"
        ):
            watchlists[row["name"]] = {}
        for row in conn.execute(
            "SELECT list_name, ticker, target_price FROM watch_list_items"
        ):
            watchlists.setdefault(row["list_name"], {})[row["ticker"]] = row["target_price"]

        # Flat aggregated view kept for backward compat (watcher, old tab code)
        watch_list = {
            t: p for items in watchlists.values() for t, p in items.items()
        }

        peer_groups: dict[str, list[str]] = {
            row["name"]: []
            for row in conn.execute("SELECT name FROM peer_groups")
        }
        for row in conn.execute(
            "SELECT group_name, ticker FROM peer_group_members"
        ):
            peer_groups.setdefault(row["group_name"], []).append(row["ticker"])

        favorite_stocks: dict[str, dict[str, Any]] = {}
        for row in conn.execute(
            "SELECT ticker, notes, goal_price, position_note, added_at "
            "FROM favorite_stocks ORDER BY added_at ASC"
        ):
            favorite_stocks[row["ticker"]] = {
                "notes": row["notes"] or "",
                "goal_price": row["goal_price"],
                "position_note": row["position_note"] or "",
                "added_at": row["added_at"],
            }

    return {
        "portfolios": portfolios,
        "watchlists": watchlists,
        "watch_list_targets": watch_list,   # flat aggregated view, for compat
        "peer_groups": peer_groups,
        "favorite_stocks": favorite_stocks,
    }


def _write_dict(conn: sqlite3.Connection, data: dict[str, Any]) -> None:
    portfolios = data.get("portfolios", {}) or {}
    peer_groups = data.get("peer_groups", {}) or {}
    favorite_stocks = data.get("favorite_stocks", None)  # None = don't touch
    # Watchlists are now managed exclusively via dedicated helpers; _write_dict
    # no longer touches watch_list or watch_list_items.

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM holdings")
        conn.execute("DELETE FROM portfolios")
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

        for group, tickers in peer_groups.items():
            conn.execute("INSERT OR IGNORE INTO peer_groups(name) VALUES (?)", (group,))
            for t in tickers or []:
                conn.execute(
                    "INSERT OR IGNORE INTO peer_group_members(group_name, ticker) VALUES (?, ?)",
                    (group, _sanitize_ticker(t)),
                )

        # Only rewrite favorites if the caller explicitly passed them; `None`
        # means preserve existing rows (favorites are usually mutated via
        # dedicated helpers, not via full-dict save_data).
        if favorite_stocks is not None:
            conn.execute("DELETE FROM favorite_stocks")
            for ticker, fav in (favorite_stocks or {}).items():
                conn.execute(
                    """INSERT INTO favorite_stocks
                         (ticker, notes, goal_price, position_note, added_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        _sanitize_ticker(ticker),
                        str(fav.get("notes") or ""),
                        float(fav["goal_price"]) if fav.get("goal_price") is not None else None,
                        str(fav.get("position_note") or ""),
                        int(fav.get("added_at") or time.time()),
                    ),
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
    cost_basis: float | None = None,
    ts: int | None = None,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO transactions(ts, portfolio_name, ticker, action, quantity, price, cost_basis)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                int(ts if ts is not None else time.time()),
                portfolio_name,
                _sanitize_ticker(ticker),
                action.upper(),
                float(quantity),
                float(price),
                float(cost_basis) if cost_basis is not None else None,
            ),
        )


def import_transactions(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Bulk-insert transactions. Rows must have keys:
    ts (int epoch), portfolio_name, ticker, action (BUY|SELL), quantity, price.
    Optional: cost_basis.

    Returns {"added": int, "skipped": int, "errors": int}. Duplicates (same
    ts, portfolio, ticker, action, quantity, price) are skipped.
    """
    init_db()
    added = skipped = errors = 0
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for r in rows:
                try:
                    ts = int(r["ts"])
                    portfolio = str(r["portfolio_name"])
                    ticker = _sanitize_ticker(str(r["ticker"]))
                    action = str(r["action"]).upper()
                    qty = float(r["quantity"])
                    price = float(r["price"])
                    cb = r.get("cost_basis")
                    cb = float(cb) if cb not in (None, "", "nan") else None
                except Exception:
                    errors += 1
                    continue

                if action not in ("BUY", "SELL"):
                    errors += 1
                    continue

                dup = conn.execute(
                    """SELECT 1 FROM transactions
                       WHERE ts=? AND portfolio_name=? AND ticker=?
                         AND action=? AND quantity=? AND price=? LIMIT 1""",
                    (ts, portfolio, ticker, action, qty, price),
                ).fetchone()
                if dup:
                    skipped += 1
                    continue

                conn.execute(
                    """INSERT INTO transactions
                         (ts, portfolio_name, ticker, action, quantity, price, cost_basis)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ts, portfolio, ticker, action, qty, price, cb),
                )
                added += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {"added": added, "skipped": skipped, "errors": errors}


def fetch_transactions(portfolio_name: str | None = None) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        if portfolio_name:
            rows = conn.execute(
                """SELECT id, ts, portfolio_name, ticker, action, quantity, price, cost_basis
                   FROM transactions WHERE portfolio_name = ? ORDER BY ts DESC""",
                (portfolio_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, ts, portfolio_name, ticker, action, quantity, price, cost_basis
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
    """Convenience accessor for the watcher service.
    Aggregates across all named watchlists; uses the highest target when the
    same ticker appears in multiple lists."""
    init_db()
    with _connect() as conn:
        return {
            row["ticker"]: row["target_price"]
            for row in conn.execute(
                """SELECT ticker, MAX(target_price) AS target_price
                   FROM watch_list_items
                   WHERE target_price > 0
                   GROUP BY ticker"""
            )
        }


# ---------- Named watchlist helpers ----------

def list_watchlists() -> list[str]:
    """Return watchlist names in display order."""
    init_db()
    with _connect() as conn:
        return [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM watch_lists ORDER BY sort_order, name"
            )
        ]


def create_watchlist(name: str) -> bool:
    """Create a new named watchlist. Returns False if the name already exists."""
    init_db()
    name = name.strip()
    if not name:
        return False
    with _connect() as conn:
        max_order = (
            conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM watch_lists"
            ).fetchone()[0]
        )
        cur = conn.execute(
            "INSERT OR IGNORE INTO watch_lists(name, sort_order, created_at)"
            " VALUES (?,?,?)",
            (name, max_order, int(time.time())),
        )
        return cur.rowcount > 0


def rename_watchlist(old_name: str, new_name: str) -> bool:
    """Rename a watchlist. Returns False if new_name already exists."""
    init_db()
    new_name = new_name.strip()
    if not new_name or old_name == new_name:
        return False
    with _connect() as conn:
        if conn.execute(
            "SELECT 1 FROM watch_lists WHERE name = ?", (new_name,)
        ).fetchone():
            return False
        # ON UPDATE CASCADE propagates the rename to watch_list_items automatically.
        conn.execute(
            "UPDATE watch_lists SET name = ? WHERE name = ?", (new_name, old_name)
        )
        return True


def delete_watchlist(name: str) -> None:
    """Delete a watchlist and all its tickers (CASCADE handles items)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM watch_lists WHERE name = ?", (name,))


def add_to_watchlist(list_name: str, ticker: str, target: float = 0.0) -> bool:
    """Add a ticker to a watchlist. Returns False if already present."""
    init_db()
    ticker = _sanitize_ticker(ticker)
    if not ticker:
        return False
    with _connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO watch_list_items(list_name, ticker, target_price)"
            " VALUES (?,?,?)",
            (list_name, ticker, float(target)),
        )
        return cur.rowcount > 0


def remove_from_watchlist(list_name: str, ticker: str) -> None:
    """Remove a ticker from a watchlist."""
    init_db()
    ticker = _sanitize_ticker(ticker)
    with _connect() as conn:
        conn.execute(
            "DELETE FROM watch_list_items WHERE list_name = ? AND ticker = ?",
            (list_name, ticker),
        )


def set_target_in_watchlist(list_name: str, ticker: str, target: float) -> None:
    """Update the target price for a ticker in a specific watchlist."""
    init_db()
    ticker = _sanitize_ticker(ticker)
    with _connect() as conn:
        conn.execute(
            "UPDATE watch_list_items SET target_price = ?"
            " WHERE list_name = ? AND ticker = ?",
            (float(target), list_name, ticker),
        )


# ---------- Favorite stocks ----------

def list_favorites() -> dict[str, dict[str, Any]]:
    init_db()
    with _connect() as conn:
        return {
            row["ticker"]: {
                "notes": row["notes"] or "",
                "goal_price": row["goal_price"],
                "position_note": row["position_note"] or "",
                "added_at": row["added_at"],
            }
            for row in conn.execute(
                "SELECT ticker, notes, goal_price, position_note, added_at "
                "FROM favorite_stocks ORDER BY added_at ASC"
            )
        }


def add_favorite(ticker: str) -> bool:
    """Insert a new favorite. Returns True if inserted, False if already existed."""
    init_db()
    ticker = _sanitize_ticker(ticker)
    if not ticker:
        return False
    with _connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO favorite_stocks(ticker, added_at) VALUES (?, ?)",
            (ticker, int(time.time())),
        )
        return cur.rowcount > 0


def remove_favorite(ticker: str) -> None:
    init_db()
    ticker = _sanitize_ticker(ticker)
    with _connect() as conn:
        conn.execute("DELETE FROM favorite_stocks WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM saved_articles WHERE ticker = ?", (ticker,))


def update_favorite(
    ticker: str,
    notes: str | None = None,
    goal_price: float | None = None,
    position_note: str | None = None,
    clear_goal: bool = False,
) -> None:
    """Patch a favorite. Pass only the fields you want to change.
    Use `clear_goal=True` to explicitly null-out goal_price (since passing None
    means 'don't change'). Same pattern for notes/position_note: pass the new
    string to overwrite, or None to leave alone."""
    init_db()
    ticker = _sanitize_ticker(ticker)
    updates = []
    params: list[Any] = []
    if notes is not None:
        updates.append("notes = ?")
        params.append(str(notes))
    if goal_price is not None or clear_goal:
        updates.append("goal_price = ?")
        params.append(None if clear_goal else float(goal_price))
    if position_note is not None:
        updates.append("position_note = ?")
        params.append(str(position_note))
    if not updates:
        return
    params.append(ticker)
    with _connect() as conn:
        conn.execute(
            f"UPDATE favorite_stocks SET {', '.join(updates)} WHERE ticker = ?",
            params,
        )


# ---------- Saved articles (manual "noteworthy" URL capture) ----------

def save_article(
    ticker: str,
    url: str,
    title: str | None = None,
    source: str | None = None,
    note: str = "",
    published_at: int | None = None,
) -> int | None:
    """Insert a saved article. Returns the new row id, or None if it was a
    duplicate (same ticker + url)."""
    init_db()
    ticker = _sanitize_ticker(ticker)
    with _connect() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO saved_articles
                 (ticker, url, title, source, note, saved_at, published_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                url.strip(),
                (title or "").strip() or None,
                (source or "").strip() or None,
                note or "",
                int(time.time()),
                int(published_at) if published_at else None,
            ),
        )
        return cur.lastrowid if cur.rowcount > 0 else None


def list_saved_articles(ticker: str) -> list[dict[str, Any]]:
    init_db()
    ticker = _sanitize_ticker(ticker)
    with _connect() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT id, ticker, url, title, source, note, saved_at, published_at
                   FROM saved_articles
                   WHERE ticker = ?
                   ORDER BY COALESCE(published_at, saved_at) DESC""",
                (ticker,),
            )
        ]


def delete_saved_article(article_id: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM saved_articles WHERE id = ?", (int(article_id),))


def update_saved_article_note(article_id: int, note: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE saved_articles SET note = ? WHERE id = ?",
            (note or "", int(article_id)),
        )
