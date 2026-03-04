import yfinance as yf
import json
import time
import requests
import os

# --- CONFIGURATION ---
# Replace these with your actual details
BOT_TOKEN = "8715551315:AAFh_EhKkAO5LMBZqhDlA3fkvRLLJhSbUzA"
CHAT_ID = "8676760709"
DATA_FILE = "portfolio.json"
CHECK_INTERVAL = 300  # Check every 5 minutes

def send_telegram_message(message):
    """Sends a message to your phone via Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send alert: {e}")

def check_prices():
    print("Watcher service active. Checking prices...")
    
    # Reload portfolio on every loop to catch updates from the UI
    if not os.path.exists(DATA_FILE):
        return

    try:
        with open(DATA_FILE, 'r') as f:
            portfolio = json.load(f)
    except (json.JSONDecodeError, IOError):
        return # Skip this cycle if file is being written to

    tickers = [t for t in portfolio.keys() if portfolio[t] > 0]
    if not tickers:
        return

    # Fetch live prices (Batch fetch for efficiency)
    try:
        # yfinance allows fetching multiple tickers at once string-separated
        tickers_str = " ".join(tickers)
        data = yf.Tickers(tickers_str)
        
        for ticker in tickers:
            target = portfolio[ticker]
            # specific handling for single vs multiple result structures
            try:
                live_price = data.tickers[ticker].fast_info['last_price']
            except AttributeError:
                # Fallback if structure differs for single ticker
                 live_price = data.tickers[ticker].info['currentPrice']

            if live_price is not None and live_price <= target:
                msg = f"🚨 BUY ALERT: {ticker}\nPrice: ${live_price:.2f}\nTarget: ${target:.2f}"
                send_telegram_message(msg)
                print(msg) # Log to Docker console

    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    send_telegram_message("Stock Watcher Service Started 📈")
    while True:
        check_prices()
        time.sleep(CHECK_INTERVAL)
