import pandas as pd
import feedparser
from datetime import datetime, timedelta
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === SETTINGS ===
TICKERS = {
    "AAPL": "Apple",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "NFLX": "Netflix",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla"
}

DAYS_TO_KEEP = 7
TODAY = datetime.today().date()
CUTOFF_DATE = TODAY - timedelta(days=DAYS_TO_KEEP)

# === FINANCE-SPECIFIC SENTIMENT ===
analyzer = SentimentIntensityAnalyzer()
finance_lexicon = {
    "bullish": 1.5, "bearish": -1.5, "rally": 1.3,
    "plummet": -1.7, "dividend": 0.5, "bankrupt": -2.0,
    "breakout": 1.2, "downgrade": -1.3, "upgrade": 1.3,
    "short squeeze": 1.4
}
analyzer.lexicon.update(finance_lexicon)

# === CORE FUNCTIONS ===
def fetch_articles():
    """Scrape news from multiple RSS feeds"""
    all_articles = []
    
    for ticker, name in TICKERS.items():
        # Google News RSS
        try:
            g_feed = feedparser.parse(f"https://news.google.com/rss/search?q={name}+stock&hl=en-US&gl=US&ceid=US:en")
            all_articles.extend(process_entries(g_feed.entries, ticker, "Google RSS"))
        except Exception as e:
            print(f"Google RSS failed for {ticker}: {str(e)}")
        
        # Bing News RSS
        try:
            b_feed = feedparser.parse(f"https://www.bing.com/news/search?q={name}+stock&format=rss")
            all_articles.extend(process_entries(b_feed.entries, ticker, "Bing RSS", default_date=TODAY))
        except Exception as e:
            print(f"Bing RSS failed for {ticker}: {str(e)}")
        
        # Yahoo Finance RSS
        try:
            y_feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
            all_articles.extend(process_entries(y_feed.entries, ticker, "Yahoo RSS", default_date=TODAY))
        except Exception as e:
            print(f"Yahoo RSS failed for {ticker}: {str(e)}")
    
    return pd.DataFrame(all_articles)

def process_entries(entries, ticker, source, default_date=None):
    """Process RSS feed entries with error handling"""
    processed = []
    for entry in entries:
        try:
            pub_date = (datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z").date() 
                       if hasattr(entry, 'published') 
                       else default_date or TODAY)
            
            if pub_date >= CUTOFF_DATE:
                processed.append({
                    "date": pub_date,
                    "headline": clean_text(entry.title),
                    "ticker": ticker,
                    "source": source
                })
        except Exception as e:
            print(f"Error processing entry: {str(e)}")
            continue
    return processed

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    return (text.replace('\n', ' ')
               .replace('\t', ' ')
               .strip())

def analyze_sentiment(df):
    """Perform sentiment analysis on headlines"""
    # Sentiment scoring
    sentiment_data = df["headline"].apply(
        lambda x: analyzer.polarity_scores(str(x))
    ).apply(pd.Series)
    
    df = pd.concat([df, sentiment_data], axis=1)
    
    # Labeling
    df['sentiment_label'] = df['compound'].apply(
        lambda x: 'positive' if x >= 0.1 else ('negative' if x <= -0.1 else 'neutral')
    )
    
    return df

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"Starting scraping at {datetime.now()}")
    
    # 1. Fetch and combine articles
    new_df = fetch_articles()
    combined_df = new_df.drop_duplicates(subset=["headline", "ticker"])
    
    # 2. Analyze sentiment
    if not combined_df.empty:
        analyzed_df = analyze_sentiment(combined_df)
        
        # 3. Save with date-stamped filename
        today_str = TODAY.strftime("%Y%m%d")
        os.makedirs("data/raw_news", exist_ok=True)
        output_path = f"data/raw_news/news_{today_str}.csv"
        analyzed_df.to_csv(output_path, index=False)
        print(f"Saved {len(analyzed_df)} articles to {output_path}")
        
        # 4. Print summary
        print("\n=== Sentiment Summary ===")
        print(analyzed_df['sentiment_label'].value_counts())
    else:
        print("No new articles found today.")
