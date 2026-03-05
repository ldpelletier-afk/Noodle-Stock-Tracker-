import yfinance as yf
import json
import time
import requests
import os
from dotenv import load_dotenv

# --- SECURITY: Load environment variables ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- CONFIGURATION ---
DATA_FILE = "portfolio.json"
CHECK_INTERVAL = 300  # Check every 5 minutes

def send_telegram_message(message):
    """Sends a message to your phone via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Telegram tokens not found in .env file.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send alert: {e}")

def check_prices():
    print("Watcher service active. Checking prices...")
    
    if not os.path.exists(DATA_FILE):
        return

    try:
        with open(DATA_FILE, 'r') as f:
            portfolio = json.load(f)
    except (json.JSONDecodeError, IOError):
        return 

    # Look for targets in the watch_list_targets section (matching your new UI structure)
    targets = portfolio.get("watch_list_targets", {})
    tickers = [t for t in targets.keys() if targets[t] > 0]
    
    if not tickers:
        return

    try:
        tickers_str = " ".join(tickers)
        data = yf.Tickers(tickers_str)
        
        for ticker in tickers:
            target = targets[ticker]
            try:
                live_price = data.tickers[ticker].fast_info['last_price']
            except AttributeError:
                 live_price = data.tickers[ticker].info.get('currentPrice')

            if live_price is not None and live_price <= target:
                msg = f"🚨 BUY ALERT: {ticker}\nPrice: ${live_price:.2f}\nTarget: ${target:.2f}"
                send_telegram_message(msg)
                print(msg) 

    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    send_telegram_message("Stock Watcher Service Started 📈")
    while True:
        check_prices()
        time.sleep(CHECK_INTERVAL)