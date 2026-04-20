"""Small pure helpers shared across the app."""
from __future__ import annotations

import pandas as pd


def sanitize_ticker(ticker: str) -> str:
    if not ticker:
        return ticker
    ticker = ticker.upper().strip()
    if ticker.endswith(":CA"):
        return ticker.replace(":CA", ".TO")
    return ticker


def format_large_number(num) -> str:
    if num is None:
        return "N/A"
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    return f"{num:.2f}"


def highlight_buy_zone(row):
    """Row-level pandas Styler callback used by the Market Watch table."""
    live = row.get("Live Price (from API)")
    target = row.get("Target Price (Self-set)")
    change = row.get("Day Change (%)")
    styles = [""] * len(row)
    if pd.notna(live) and pd.notna(target) and target > 0.0 and live <= target:
        styles = ["background-color: #28a745; color: white;"] * len(row)
    if pd.notna(change) and change <= -5.0:
        change_idx = list(row.index).index("Day Change (%)")
        ticker_idx = list(row.index).index("Ticker")
        styles[change_idx] += "color: #ff4b4b; font-weight: bold;"
        styles[ticker_idx] += "color: #ff4b4b; font-weight: bold;"
    return styles
