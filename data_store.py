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

        -- Tiny key/value store for persistent UI preferences (e.g. which
        -- watchlists were left expanded). Survives app relaunches.
        CREATE TABLE IF NOT EXISTS ui_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        -- Tracks which major institutions we expect a market report from.
        -- `doc_id` here holds the *primary* (default-displayed) report.
        -- NULL means no report yet linked. The full set of attached reports
        -- lives in institutional_coverage_docs (one row per (institution, doc_id) pair).
        CREATE TABLE IF NOT EXISTS institutional_coverage (
            institution TEXT PRIMARY KEY,
            doc_id      TEXT,
            added_at    INTEGER NOT NULL DEFAULT 0
        );

        -- Multi-doc join table: an institution can have many attached reports.
        -- `doc_id` is the ChromaDB document id. The "primary" report for each
        -- institution is mirrored on the parent table's `doc_id` column so the
        -- legacy single-doc UI / API path keeps working.
        -- ON DELETE CASCADE here is enforced by Python (see remove_institution
        -- and unlink_doc_from_all_institutions) because SQLite needs
        -- PRAGMA foreign_keys=ON per-connection to enforce it natively.
        CREATE TABLE IF NOT EXISTS institutional_coverage_docs (
            institution TEXT NOT NULL,
            doc_id      TEXT NOT NULL,
            linked_at   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (institution, doc_id)
        );

        CREATE INDEX IF NOT EXISTS idx_inst_cov_docs_doc
            ON institutional_coverage_docs(doc_id);

        -- Political / geopolitical investment catalysts.
        -- A catalyst is a date-bound event with a defined investment thesis:
        --   - monetary  : Fed/Treasury/regulatory rulings affecting earnings power
        --   - contract  : federal procurement awards, GAO protests, recompetes
        --   - court     : SCOTUS rulings, antitrust verdicts, SEC enforcement,
        --                 patent decisions
        -- `doc_ids` is a comma-separated list of RAG library doc_ids attached
        -- as background reading. `tickers` / `sectors` are comma-separated.
        CREATE TABLE IF NOT EXISTS political_catalysts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date    INTEGER NOT NULL,                  -- unix ts
            event_end     INTEGER,                            -- optional range end
            title         TEXT NOT NULL,
            catalyst_type TEXT NOT NULL,                      -- monetary|contract|court
            category      TEXT,                               -- finer subcategory
            stakes        TEXT,                               -- markdown free-text
            tickers       TEXT,                               -- comma-separated
            sectors       TEXT,                               -- comma-separated
            probability   TEXT,                               -- subjective
            status        TEXT NOT NULL DEFAULT 'upcoming',   -- upcoming|live|resolved
            outcome_notes TEXT,
            source        TEXT NOT NULL DEFAULT 'manual',
            doc_ids       TEXT,                               -- comma-separated
            created_at    INTEGER NOT NULL,
            updated_at    INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_catalysts_date
            ON political_catalysts(event_date);
        CREATE INDEX IF NOT EXISTS idx_catalysts_type
            ON political_catalysts(catalyst_type);
        CREATE INDEX IF NOT EXISTS idx_catalysts_status
            ON political_catalysts(status);
        """
    )

    # One-shot migration: copy any existing single-doc link from the parent
    # table into the join table so legacy data shows up in the new multi-doc UI.
    now_ts = int(time.time())
    conn.executescript(
        "BEGIN;"
        " INSERT OR IGNORE INTO institutional_coverage_docs (institution, doc_id, linked_at)"
        "   SELECT institution, doc_id, COALESCE(added_at, " + str(now_ts) + ")"
        "     FROM institutional_coverage"
        "    WHERE doc_id IS NOT NULL;"
        " COMMIT;"
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


# ---------- Persistent UI preferences ----------

def _ensure_ui_state(conn: sqlite3.Connection) -> None:
    """Create the ui_state table if it doesn't exist yet. Safe to call from
    every helper — this lets the helpers work even if the running server hadn't
    re-run _create_schema after a code update."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ui_state ("
        " key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def ui_state_get(key: str, default: str = "") -> str:
    """Read a persisted UI preference, or return `default` if absent."""
    init_db()
    with _connect() as conn:
        _ensure_ui_state(conn)
        row = conn.execute(
            "SELECT value FROM ui_state WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def ui_state_set(key: str, value: str) -> None:
    """Persist a UI preference."""
    init_db()
    with _connect() as conn:
        _ensure_ui_state(conn)
        conn.execute(
            "INSERT INTO ui_state(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


def ui_state_delete_prefix(prefix: str) -> None:
    """Delete every UI preference whose key starts with `prefix`. Used to clean
    up rows for renamed/deleted watchlists."""
    init_db()
    with _connect() as conn:
        _ensure_ui_state(conn)
        conn.execute(
            "DELETE FROM ui_state WHERE key LIKE ?", (prefix + "%",)
        )


# ---------------------------------------------------------------------------
# Institutional coverage watchlist
# ---------------------------------------------------------------------------

def list_institutional_coverage() -> list[dict]:
    """Return all tracked institutions, alphabetical.

    Each entry:
      {
        "institution":     str,
        "doc_id":          str | None,    # the *primary* doc (back-compat)
        "primary_doc_id":  str | None,    # explicit alias of doc_id
        "doc_ids":         list[str],     # ALL attached reports, newest-first
        "added_at":        int,
      }
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT institution, doc_id, added_at"
            " FROM institutional_coverage"
            " ORDER BY institution ASC"
        ).fetchall()
        # Pull every link in a single query, then group in Python — avoids
        # an N+1 over institutions.
        link_rows = conn.execute(
            "SELECT institution, doc_id, linked_at"
            " FROM institutional_coverage_docs"
            " ORDER BY linked_at DESC, doc_id ASC"
        ).fetchall()

    by_inst: dict[str, list[str]] = {}
    for lr in link_rows:
        by_inst.setdefault(lr["institution"], []).append(lr["doc_id"])

    out: list[dict] = []
    for r in rows:
        d = dict(r)
        d["primary_doc_id"] = d["doc_id"]
        d["doc_ids"]        = by_inst.get(d["institution"], [])
        # Self-heal: if the parent's primary points at a doc that's no longer
        # in the join table (rare but possible after manual db edits), promote
        # the most-recent linked doc, or clear if the institution has none.
        if d["primary_doc_id"] and d["primary_doc_id"] not in d["doc_ids"]:
            d["primary_doc_id"] = d["doc_ids"][0] if d["doc_ids"] else None
            d["doc_id"]         = d["primary_doc_id"]
        out.append(d)
    return out


def list_institution_docs(institution: str) -> list[dict]:
    """All docs attached to an institution, newest-first, with primary flag."""
    init_db()
    with _connect() as conn:
        primary_row = conn.execute(
            "SELECT doc_id FROM institutional_coverage WHERE institution = ?",
            (institution,),
        ).fetchone()
        primary = primary_row["doc_id"] if primary_row else None
        rows = conn.execute(
            "SELECT doc_id, linked_at"
            " FROM institutional_coverage_docs"
            " WHERE institution = ?"
            " ORDER BY linked_at DESC, doc_id ASC",
            (institution,),
        ).fetchall()
    return [
        {"doc_id": r["doc_id"], "linked_at": r["linked_at"],
         "is_primary": r["doc_id"] == primary}
        for r in rows
    ]


def add_institution(institution: str) -> bool:
    """Add an institution to the coverage watchlist.

    Returns True if inserted, False if it already existed.
    """
    institution = institution.strip()
    if not institution:
        return False
    init_db()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT 1 FROM institutional_coverage WHERE institution = ?",
            (institution,),
        ).fetchone()
        if existing:
            return False
        conn.execute(
            "INSERT INTO institutional_coverage(institution, doc_id, added_at)"
            " VALUES (?, NULL, ?)",
            (institution, int(time.time())),
        )
    return True


def link_institution_doc(institution: str, doc_id: str | None) -> None:
    """Attach a doc to an institution.

    Multi-doc semantics:
      * doc_id = a real id  → APPEND to the join table. If the institution had
        no primary set, this becomes the primary. If it already had one, the
        primary is unchanged (call set_primary_institution_doc to change it).
      * doc_id = None       → unlink ALL docs (full reset). Same effect as the
        legacy single-doc API used to have.

    The function is idempotent: linking the same (institution, doc_id) twice
    is a no-op (no duplicate row, no error).
    """
    init_db()
    with _connect() as conn:
        if doc_id is None:
            conn.execute(
                "DELETE FROM institutional_coverage_docs WHERE institution = ?",
                (institution,),
            )
            conn.execute(
                "UPDATE institutional_coverage SET doc_id = NULL WHERE institution = ?",
                (institution,),
            )
            return

        conn.execute(
            "INSERT OR IGNORE INTO institutional_coverage_docs"
            " (institution, doc_id, linked_at) VALUES (?, ?, ?)",
            (institution, doc_id, int(time.time())),
        )
        # Promote to primary only if no primary is set yet.
        existing = conn.execute(
            "SELECT doc_id FROM institutional_coverage WHERE institution = ?",
            (institution,),
        ).fetchone()
        if existing is None or existing["doc_id"] is None:
            conn.execute(
                "UPDATE institutional_coverage SET doc_id = ? WHERE institution = ?",
                (doc_id, institution),
            )


def unlink_institution_doc(institution: str, doc_id: str) -> None:
    """Detach a single doc from an institution. If the unlinked doc was the
    primary, the next-most-recent linked doc (if any) is promoted."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM institutional_coverage_docs"
            " WHERE institution = ? AND doc_id = ?",
            (institution, doc_id),
        )
        # Was it primary?
        row = conn.execute(
            "SELECT doc_id FROM institutional_coverage WHERE institution = ?",
            (institution,),
        ).fetchone()
        if row and row["doc_id"] == doc_id:
            replacement = conn.execute(
                "SELECT doc_id FROM institutional_coverage_docs"
                " WHERE institution = ?"
                " ORDER BY linked_at DESC LIMIT 1",
                (institution,),
            ).fetchone()
            new_primary = replacement["doc_id"] if replacement else None
            conn.execute(
                "UPDATE institutional_coverage SET doc_id = ? WHERE institution = ?",
                (new_primary, institution),
            )


def set_primary_institution_doc(institution: str, doc_id: str) -> bool:
    """Mark `doc_id` as the primary report for `institution`.

    Returns False if `doc_id` isn't currently linked to that institution
    (caller should link it first). Otherwise updates the parent table.
    """
    init_db()
    with _connect() as conn:
        link = conn.execute(
            "SELECT 1 FROM institutional_coverage_docs"
            " WHERE institution = ? AND doc_id = ?",
            (institution, doc_id),
        ).fetchone()
        if not link:
            return False
        conn.execute(
            "UPDATE institutional_coverage SET doc_id = ? WHERE institution = ?",
            (doc_id, institution),
        )
    return True


def unlink_doc_from_all_institutions(doc_id: str) -> int:
    """Cascade-cleanup: when a library doc is deleted, sever every institutional
    link that pointed at it. Returns the number of links removed.

    Also clears the primary pointer on any institution whose primary was this
    doc, promoting the next-most-recent linked doc as new primary (or NULL).
    """
    init_db()
    with _connect() as conn:
        removed = conn.execute(
            "DELETE FROM institutional_coverage_docs WHERE doc_id = ?",
            (doc_id,),
        ).rowcount or 0
        # Anyone who had this as primary needs a replacement (or NULL).
        affected = conn.execute(
            "SELECT institution FROM institutional_coverage WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
        for row in affected:
            inst = row["institution"]
            replacement = conn.execute(
                "SELECT doc_id FROM institutional_coverage_docs"
                " WHERE institution = ?"
                " ORDER BY linked_at DESC LIMIT 1",
                (inst,),
            ).fetchone()
            conn.execute(
                "UPDATE institutional_coverage SET doc_id = ? WHERE institution = ?",
                (replacement["doc_id"] if replacement else None, inst),
            )
    return removed


def remove_institution(institution: str) -> None:
    """Remove an institution from the watchlist (and all of its doc links)."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM institutional_coverage_docs WHERE institution = ?",
            (institution,),
        )
        conn.execute(
            "DELETE FROM institutional_coverage WHERE institution = ?",
            (institution,),
        )


# ---------------------------------------------------------------------------
# Political / geopolitical catalysts
# ---------------------------------------------------------------------------

CATALYST_TYPES    = ("monetary", "contract", "court")
CATALYST_STATUSES = ("upcoming", "live", "resolved")

# Display labels for the UI — kept here so the data layer is the single
# source of truth and the UI doesn't drift.
CATALYST_TYPE_LABELS = {
    "monetary": "🏛️ Monetary",
    "contract": "🏗️ Contract",
    "court":    "⚖️ Court",
}
CATALYST_STATUS_LABELS = {
    "upcoming": "🟡 Upcoming",
    "live":     "🟢 Live",
    "resolved": "✅ Resolved",
}


def _normalize_csv(value) -> str:
    """Accept list[str] or str, return canonical comma-separated string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _decode_catalyst_row(row) -> dict:
    """Convert a sqlite Row → dict, expanding CSV fields into lists."""
    d = dict(row)
    for k in ("tickers", "sectors", "doc_ids"):
        raw = d.get(k) or ""
        d[k] = [x.strip() for x in raw.split(",") if x.strip()]
    return d


def add_catalyst(
    *,
    event_date: int,
    title: str,
    catalyst_type: str,
    event_end: int | None = None,
    category: str = "",
    stakes: str = "",
    tickers=None,
    sectors=None,
    probability: str = "",
    status: str = "upcoming",
    outcome_notes: str = "",
    source: str = "manual",
    doc_ids=None,
) -> int:
    """Insert a new catalyst, returning its row id."""
    if catalyst_type not in CATALYST_TYPES:
        raise ValueError(f"Unknown catalyst_type: {catalyst_type}")
    if status not in CATALYST_STATUSES:
        raise ValueError(f"Unknown status: {status}")
    title = (title or "").strip()
    if not title:
        raise ValueError("Catalyst title is required.")

    init_db()
    now_ts = int(time.time())
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO political_catalysts
               (event_date, event_end, title, catalyst_type, category, stakes,
                tickers, sectors, probability, status, outcome_notes, source,
                doc_ids, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(event_date), event_end and int(event_end),
                title, catalyst_type, (category or "").strip(), stakes or "",
                _normalize_csv(tickers), _normalize_csv(sectors),
                (probability or "").strip(), status, outcome_notes or "",
                source, _normalize_csv(doc_ids), now_ts, now_ts,
            ),
        )
        return cur.lastrowid


def list_catalysts(
    *,
    catalyst_types: list[str] | None = None,
    ticker: str | None = None,
    status: str | None = None,
    source: str | None = None,
    days_ahead: int | None = None,
    days_behind: int | None = None,
    order: str = "ASC",
) -> list[dict]:
    """Return catalysts matching the filters, chronological by default.

    `ticker` filtering happens in Python (after CSV decode) so callers can
    match case-insensitively. The other filters compile to SQL clauses.
    `source` is exposed to support import-dedup queries (find catalysts
    previously imported from the same external feed).
    """
    init_db()
    clauses: list[str] = []
    params: list = []
    if catalyst_types:
        valid = [t for t in catalyst_types if t in CATALYST_TYPES]
        if valid:
            ph = ",".join("?" * len(valid))
            clauses.append(f"catalyst_type IN ({ph})")
            params.extend(valid)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if source:
        clauses.append("source = ?")
        params.append(source)
    now_ts = int(time.time())
    if days_ahead is not None:
        clauses.append("event_date <= ?")
        params.append(now_ts + days_ahead * 86400)
    if days_behind is not None:
        clauses.append("event_date >= ?")
        params.append(now_ts - days_behind * 86400)
    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_sql = "DESC" if order.upper() == "DESC" else "ASC"
    sql = f"SELECT * FROM political_catalysts {where_sql} ORDER BY event_date {order_sql}"

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    decoded = [_decode_catalyst_row(r) for r in rows]
    if ticker:
        tk = ticker.upper().strip()
        decoded = [c for c in decoded if tk in [t.upper() for t in c["tickers"]]]
    return decoded


def get_catalyst(catalyst_id: int) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM political_catalysts WHERE id = ?",
            (int(catalyst_id),),
        ).fetchone()
    return _decode_catalyst_row(row) if row else None


def update_catalyst(catalyst_id: int, **fields) -> bool:
    """Patch any subset of fields on a catalyst. Lists for tickers/sectors/
    doc_ids are auto-converted to CSV strings. Returns True if a row was updated.
    """
    if not fields:
        return False
    if "catalyst_type" in fields and fields["catalyst_type"] not in CATALYST_TYPES:
        raise ValueError(f"Unknown catalyst_type: {fields['catalyst_type']}")
    if "status" in fields and fields["status"] not in CATALYST_STATUSES:
        raise ValueError(f"Unknown status: {fields['status']}")
    for k in ("tickers", "sectors", "doc_ids"):
        if k in fields:
            fields[k] = _normalize_csv(fields[k])
    if "title" in fields:
        fields["title"] = (fields["title"] or "").strip()
    fields["updated_at"] = int(time.time())

    set_sql = ", ".join(f"{k} = ?" for k in fields)
    params  = list(fields.values()) + [int(catalyst_id)]
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE political_catalysts SET {set_sql} WHERE id = ?",
            params,
        )
        return (cur.rowcount or 0) > 0


def delete_catalyst(catalyst_id: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM political_catalysts WHERE id = ?",
            (int(catalyst_id),),
        )


def link_catalyst_doc(catalyst_id: int, doc_id: str) -> None:
    """Idempotent: append a library doc_id to a catalyst's reading list."""
    cat = get_catalyst(catalyst_id)
    if not cat:
        return
    docs = list(cat["doc_ids"])
    if doc_id and doc_id not in docs:
        docs.append(doc_id)
        update_catalyst(catalyst_id, doc_ids=docs)


def unlink_catalyst_doc(catalyst_id: int, doc_id: str) -> None:
    cat = get_catalyst(catalyst_id)
    if not cat:
        return
    docs = [d for d in cat["doc_ids"] if d != doc_id]
    update_catalyst(catalyst_id, doc_ids=docs)


def unlink_doc_from_all_catalysts(doc_id: str) -> int:
    """Cascade-cleanup: when a library doc is deleted, sever every catalyst
    reading-list reference to it. Returns the number of catalysts updated."""
    if not doc_id:
        return 0
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, doc_ids FROM political_catalysts WHERE doc_ids LIKE ?",
            (f"%{doc_id}%",),
        ).fetchall()
        n = 0
        now_ts = int(time.time())
        for row in rows:
            ids = [x.strip() for x in (row["doc_ids"] or "").split(",") if x.strip()]
            if doc_id in ids:
                ids.remove(doc_id)
                conn.execute(
                    "UPDATE political_catalysts SET doc_ids = ?, updated_at = ?"
                    " WHERE id = ?",
                    (",".join(ids), now_ts, row["id"]),
                )
                n += 1
        return n


# ---------------------------------------------------------------------------
# Portfolio CRUD
# ---------------------------------------------------------------------------

def list_portfolio_names() -> list[str]:
    """Return all portfolio names in insertion order."""
    init_db()
    with _connect() as conn:
        return [row["name"] for row in conn.execute("SELECT name FROM portfolios")]


def create_portfolio(name: str) -> bool:
    """Create an empty portfolio. Returns False if the name already exists."""
    name = (name or "").strip()
    if not name:
        return False
    init_db()
    with _connect() as conn:
        cur = conn.execute("INSERT OR IGNORE INTO portfolios(name) VALUES (?)", (name,))
        return cur.rowcount > 0


def delete_portfolio(name: str) -> None:
    """Delete a portfolio and all its holdings (ON DELETE CASCADE)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM portfolios WHERE name = ?", (name,))


def get_portfolio_holdings(name: str) -> dict[str, dict[str, float]]:
    """Return {ticker: {quantity, average_cost}} for every holding in a portfolio."""
    init_db()
    with _connect() as conn:
        return {
            row["ticker"]: {
                "quantity": row["quantity"],
                "average_cost": row["average_cost"],
            }
            for row in conn.execute(
                "SELECT ticker, quantity, average_cost FROM holdings WHERE portfolio_name = ?",
                (name,),
            )
        }


def upsert_holding(portfolio: str, ticker: str, quantity: float, avg_cost: float) -> None:
    """Insert or update a single holding in a portfolio.

    Creates the parent portfolio row if it does not exist yet.
    """
    ticker = _sanitize_ticker(ticker)
    if not ticker:
        return
    init_db()
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO portfolios(name) VALUES (?)", (portfolio,))
        conn.execute(
            """INSERT INTO holdings(portfolio_name, ticker, quantity, average_cost)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(portfolio_name, ticker) DO UPDATE SET
                 quantity     = excluded.quantity,
                 average_cost = excluded.average_cost""",
            (portfolio, ticker, float(quantity), float(avg_cost)),
        )


def remove_holding(portfolio: str, ticker: str) -> None:
    """Delete a single holding row from a portfolio."""
    ticker = _sanitize_ticker(ticker)
    init_db()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM holdings WHERE portfolio_name = ? AND ticker = ?",
            (portfolio, ticker),
        )


def replace_portfolio_holdings(
    portfolio: str, holdings: dict[str, dict[str, float]]
) -> None:
    """Atomically replace all holdings for a portfolio (used by the data editor).

    Any tickers absent from `holdings` are deleted; tickers present are upserted.
    The portfolio row itself is preserved.
    """
    init_db()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM holdings WHERE portfolio_name = ?", (portfolio,))
            for ticker, pos in (holdings or {}).items():
                ticker = _sanitize_ticker(ticker)
                if not ticker:
                    continue
                conn.execute(
                    """INSERT INTO holdings(portfolio_name, ticker, quantity, average_cost)
                       VALUES (?, ?, ?, ?)""",
                    (
                        portfolio, ticker,
                        float(pos.get("quantity", 0.0)),
                        float(pos.get("average_cost", 0.0)),
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


# ---------------------------------------------------------------------------
# Peer group CRUD
# ---------------------------------------------------------------------------

def list_peer_groups() -> dict[str, list[str]]:
    """Return {group_name: [tickers]} for all peer groups."""
    init_db()
    with _connect() as conn:
        groups: dict[str, list[str]] = {
            row["name"]: []
            for row in conn.execute("SELECT name FROM peer_groups")
        }
        for row in conn.execute("SELECT group_name, ticker FROM peer_group_members"):
            groups.setdefault(row["group_name"], []).append(row["ticker"])
    return groups


def create_peer_group(name: str) -> bool:
    """Create an empty peer group. Returns False if the name already exists."""
    name = (name or "").strip()
    if not name:
        return False
    init_db()
    with _connect() as conn:
        cur = conn.execute("INSERT OR IGNORE INTO peer_groups(name) VALUES (?)", (name,))
        return cur.rowcount > 0


def delete_peer_group(name: str) -> None:
    """Delete a peer group and all its members (ON DELETE CASCADE)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM peer_groups WHERE name = ?", (name,))


def add_to_peer_group(group: str, ticker: str) -> bool:
    """Add a ticker to a peer group. Returns False if already present."""
    ticker = _sanitize_ticker(ticker)
    if not ticker:
        return False
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO peer_group_members(group_name, ticker) VALUES (?, ?)",
            (group, ticker),
        )
        return cur.rowcount > 0


def remove_from_peer_group(group: str, ticker: str) -> None:
    """Remove a ticker from a peer group."""
    ticker = _sanitize_ticker(ticker)
    init_db()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM peer_group_members WHERE group_name = ? AND ticker = ?",
            (group, ticker),
        )


# ---------------------------------------------------------------------------
# Eager init at module load
# ---------------------------------------------------------------------------
# Public functions all call init_db() defensively, but the body of init_db()
# uses double-checked locking so repeat calls are near-free (a single global
# var read). Triggering it once here moves the schema-creation + migration
# cost to import time, so the *first* user-facing call doesn't pay it.
# Wrapped in try/except so a corrupt DB doesn't prevent the module from
# importing — the deferred init_db() inside each function will raise the
# same error at the actual call site, where Streamlit can surface it cleanly.
try:
    init_db()
except Exception:
    pass
