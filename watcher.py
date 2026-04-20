import os
import time

import requests
import yfinance as yf
from dotenv import load_dotenv

from data_store import get_watch_targets, should_fire_alert, record_alert

# --- SECURITY: Load environment variables ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CHECK_INTERVAL = int(os.getenv("WATCHER_INTERVAL_SECONDS", "300"))
ALERT_COOLDOWN = int(os.getenv("WATCHER_ALERT_COOLDOWN_SECONDS", str(6 * 60 * 60)))


def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Telegram tokens not found in .env file.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send alert: {e}")
        return False


def check_prices() -> None:
    print("Watcher service active. Checking prices...")
    targets = get_watch_targets()
    if not targets:
        return

    tickers = list(targets.keys())
    try:
        data = yf.Tickers(" ".join(tickers))
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    for ticker in tickers:
        target = targets[ticker]
        try:
            live_price = data.tickers[ticker].fast_info["last_price"]
        except (AttributeError, KeyError):
            live_price = data.tickers[ticker].info.get("currentPrice")
        except Exception as e:
            print(f"Price lookup failed for {ticker}: {e}")
            continue

        if live_price is None or live_price > target:
            continue

        if not should_fire_alert(ticker, "BUY_TARGET", target, cooldown_seconds=ALERT_COOLDOWN):
            continue

        msg = f"🚨 BUY ALERT: {ticker}\nPrice: ${live_price:.2f}\nTarget: ${target:.2f}"
        if send_telegram_message(msg):
            record_alert(ticker, "BUY_TARGET", target, live_price)
            print(msg)


if __name__ == "__main__":
    send_telegram_message("Stock Watcher Service Started 📈")
    while True:
        try:
            check_prices()
        except Exception as e:
            print(f"Unhandled error in check loop: {e}")
        time.sleep(CHECK_INTERVAL)
