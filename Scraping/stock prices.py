import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# === SETTINGS ===
TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT", "NVDA", "TSLA", "META", "NFLX"]
OUTPUT_FILE = "stock_data_summary.csv"
TODAY = datetime.today().date()

# === 1. FETCH STOCK DATA (ROBUST VERSION) ===
def fetch_stock_summary():
    print("📊 Fetching stock summary data...")
    summary_data = []
    
    for ticker in TICKERS:
        try:
            # Get ticker object with timeout
            stock = yf.Ticker(ticker)
            
            # Fetch key statistics (with fallbacks for missing data)
            info = stock.info
            history = stock.history(period="2d")  # Fallback for bid/ask
            
            summary_data.append({
                "ticker": ticker,
                "date": TODAY.strftime("%Y-%m-%d"),
                "open": history.iloc[-1]["Open"] if not history.empty else None,
                "close": info.get("currentPrice", info.get("regularMarketPrice")),
                "bid": info.get("bid", history.iloc[-1]["Close"] if not history.empty else None),
                "ask": info.get("ask", history.iloc[-1]["Close"] if not history.empty else None),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "volume": info.get("volume", history.iloc[-1]["Volume"] if not history.empty else None),
                "avg_volume": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "eps": info.get("trailingEps"),
                "currency": info.get("currency"),
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            print(f"🚨 Error fetching {ticker}: {str(e)}")
            continue
    
    return pd.DataFrame(summary_data)

# === 2. SAVE DATA (DUPLICATE-PROOF) ===
def save_summary_data(df):
    # Load existing data if file exists
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        # Remove today's data to avoid duplicates
        existing_df = existing_df[existing_df["date"] != TODAY.strftime("%Y-%m-%d")]
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved data for {len(df[df['date'] == TODAY.strftime('%Y-%m-%d')])} tickers")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    new_data = fetch_stock_summary()
    if not new_data.empty:
        save_summary_data(new_data)
        print("\n📋 Sample of Today's Data:")
        print(new_data.head())
    else:
        print("❌ No data fetched. Check API/ticker symbols.")
