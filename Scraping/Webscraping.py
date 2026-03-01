"""
Author: Phanidhar Kasuba
Description: Automated MLOps Sentiment Terminal (Tier-1 Upgraded)
Copyright (c) 2026. All rights reserved.
"""
import pandas as pd
import feedparser
import requests # <-- NEW: Needed for the human disguise
from datetime import datetime, timedelta
import os
import glob
import joblib 
from transformers import pipeline

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("Loading Stage 1: (Text Cleaner)...")
def clean(doc): 
    doc = str(doc).replace("</br>", " ") 
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    tokens = nltk.word_tokenize(doc)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# === SETTINGS & DISGUISES ===
TICKERS = {
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "META": "Meta",
    "NFLX": "Netflix"
}

# The "Human" Disguise to bypass bot-blockers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# The new Tier-1 Institutional Feeds
MACRO_FEEDS = {
    "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "MarketWatch Top": "http://feeds.marketwatch.com/marketwatch/topstories",
    "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Yahoo Global": "https://finance.yahoo.com/news/rss"
}

print("Loading Stage 2: The Bouncer (Custom NLP Filter)...")
try:
    vectorizer = joblib.load('Scraping/tfidf_vectorizer.pkl')
    classifier = joblib.load('Scraping/financial_news_classifier.pkl')
    bouncer_loaded = True
except Exception as e:
    print(f"⚠️ Could not load custom models: {e}")
    bouncer_loaded = False

print("Loading Stage 3: The Expert (FinBERT)...")
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

DAYS_TO_KEEP = 7
TODAY = datetime.today().date()
CUTOFF_DATE = TODAY - timedelta(days=DAYS_TO_KEEP)

# === CORE FUNCTIONS ===
def fetch_articles():
    """Scrape news from multiple RSS feeds"""
    all_articles = []
    
    # 1. Ticker-Specific Feeds (Upgraded with Headers)
    for ticker, name in TICKERS.items():
        try:
            res = requests.get(f"https://news.google.com/rss/search?q={name}+stock&hl=en-US&gl=US&ceid=US:en", headers=HEADERS, timeout=10)
            g_feed = feedparser.parse(res.content)
            all_articles.extend(process_entries(g_feed.entries, ticker, "Google RSS"))
        except Exception as e:
            print(f"Google RSS failed for {ticker}: {str(e)}")
        
        try:
            res = requests.get(f"https://www.bing.com/news/search?q={name}+stock&format=rss", headers=HEADERS, timeout=10)
            b_feed = feedparser.parse(res.content)
            all_articles.extend(process_entries(b_feed.entries, ticker, "Bing RSS", default_date=TODAY))
        except Exception as e:
            print(f"Bing RSS failed for {ticker}: {str(e)}")
        
        try:
            res = requests.get(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US", headers=HEADERS, timeout=10)
            y_feed = feedparser.parse(res.content)
            all_articles.extend(process_entries(y_feed.entries, ticker, "Yahoo RSS", default_date=TODAY))
        except Exception as e:
            print(f"Yahoo RSS failed for {ticker}: {str(e)}")

    # 2. Institutional Macro Feeds (The New Pipeline)
    for source_name, url in MACRO_FEEDS.items():
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            m_feed = feedparser.parse(res.content)
            
            for entry in m_feed.entries:
                headline = clean_text(entry.title)
                # Check if any tracked company is mentioned in the headline
                for ticker, name in TICKERS.items():
                    if name.lower() in headline.lower() or ticker in headline:
                        all_articles.extend(process_entries([entry], ticker, source_name))
        except Exception as e:
            print(f"{source_name} feed failed: {str(e)}")
    
    return pd.DataFrame(all_articles)

def process_entries(entries, ticker, source, default_date=None):
    """Process RSS feed entries and capture exact timestamps"""
    processed = []
    cutoff_datetime = pd.to_datetime(CUTOFF_DATE)
    
    for entry in entries:
        try:
            if hasattr(entry, 'published'):
                pub_date = pd.to_datetime(entry.published)
                if pub_date.tzinfo is not None:
                    pub_date = pub_date.tz_localize(None)
            else:
                pub_date = pd.to_datetime(datetime.now())
            
            if pub_date >= cutoff_datetime:
                processed.append({
                    "date": pub_date.strftime('%Y-%m-%d %H:%M:%S'), 
                    "headline": clean_text(entry.title),
                    "ticker": ticker,
                    "source": source
                })
        except Exception as e:
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
    if df.empty:
        return df
        
    print(f"Total articles scraped this session: {len(df)}")
    
    if bouncer_loaded:
        print("Filtering irrelevant news...")
        X_daily = vectorizer.transform(df['headline'])
        df['is_relevant'] = classifier.predict(X_daily)
        df_filtered = df[df['is_relevant'] == 1].copy()
        print(f"🗑️ Filtered out {len(df) - len(df_filtered)} junk articles. {len(df_filtered)} passed the Bouncer.")
    else:
        df_filtered = df.copy()

    print("Scoring sentiment with FinBERT...")
    sentiments = []
    scores = []
    
    for headline in df_filtered["headline"]:
        try:
            result = sentiment_pipeline(str(headline))[0]
            label = result['label'] 
            score = result['score'] 
            
            sentiments.append(label)
            
            if label == 'negative':
                scores.append(-score)
            elif label == 'positive':
                scores.append(score)
            else:
                scores.append(0.0)
        except Exception as e:
            sentiments.append('neutral')
            scores.append(0.0)
            
    df_filtered['sentiment_label'] = sentiments
    df_filtered['compound'] = scores
    
    if 'is_relevant' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['is_relevant'])
        
    return df_filtered

def cleanup_old_news(days_to_keep=30):
    print(f"🧹 Janitor: Checking for news files older than {days_to_keep} days...")
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    news_files = glob.glob("data/raw_news/news_*.csv")
    
    for file_path in news_files:
        try:
            filename = os.path.basename(file_path)
            date_str = filename.split('_')[1].split('.')[0]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            
            if file_date < cutoff_date:
                os.remove(file_path)
                print(f"🗑️ Deleted old file: {filename}")
        except Exception as e:
            print(f"⚠️ Could not parse date for {file_path}: {e}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"Starting scraping at {datetime.now()}")
    
    new_df = fetch_articles()
    
    if not new_df.empty:
        analyzed_df = analyze_sentiment(new_df)
        today_str = TODAY.strftime("%Y%m%d")
        os.makedirs("data/raw_news", exist_ok=True)
        output_path = f"data/raw_news/news_{today_str}.csv"
        
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            final_df = pd.concat([existing_df, analyzed_df], ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["headline", "ticker"], keep="first")
        else:
            final_df = analyzed_df.drop_duplicates(subset=["headline", "ticker"])
            
        final_df.to_csv(output_path, index=False)
        print(f"Saved {len(final_df)} unique articles to {output_path}")
    else:
        print("No new articles found this hour.")
    
    cleanup_old_news(days_to_keep=30)