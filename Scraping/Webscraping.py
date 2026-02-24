import pandas as pd
import feedparser
from datetime import datetime, timedelta
import os
import glob
import joblib # For loading the vectorizer and classifier
from transformers import pipeline

import nltk
# Tell the robot to download the dictionaries quietly in the background
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)


# --- NEW IMPORTS NEEDED FOR CLEAN FUNCTION ---
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer once, outside the function, to save memory
lemmatizer = WordNetLemmatizer()
# ---------------------------------------------

print("Loading Stage 1: (Text Cleaner)...")
def clean(doc): # doc is a string of text
    # This text contains a lot of <br/> tags.
    doc = str(doc).replace("</br>", " ") # Added str() just to be perfectly safe
    
    # Remove punctuation and numbers
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    
    # Tokenization
    tokens = nltk.word_tokenize(doc)

    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    
    # Join and return
    return " ".join(filtered_tokens)

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
    """Process RSS feed entries and capture exact timestamps"""
    processed = []
    
    # Ensure CUTOFF_DATE is a full datetime so we can compare it safely
    cutoff_datetime = pd.to_datetime(CUTOFF_DATE)
    
    for entry in entries:
        try:
            # 1. Grab the exact publish time
            if hasattr(entry, 'published'):
                pub_date = pd.to_datetime(entry.published)
                # Strip timezone info so Pandas doesn't crash during comparison
                if pub_date.tzinfo is not None:
                    pub_date = pub_date.tz_localize(None)
            else:
                # If the RSS feed is broken, use the exact current time (not midnight!)
                pub_date = pd.to_datetime(datetime.now())
            
            # 2. Check if it's recent enough
            if pub_date >= cutoff_datetime:
                processed.append({
                    # THIS IS THE FIX: Save as YYYY-MM-DD HH:MM:SS
                    "date": pub_date.strftime('%Y-%m-%d %H:%M:%S'), 
                    "headline": clean_text(entry.title),
                    "ticker": ticker,
                    "source": source
                })
        except Exception as e:
            # Silently skip broken entries to keep logs clean
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
        
    print(f"Total articles scraped: {len(df)}")
    
    # === STAGE 1: THE BOUNCER ===
    if bouncer_loaded:
        print("Filtering irrelevant news...")
        # Translate text to numbers
        X_daily = vectorizer.transform(df['headline'])
        # Predict if relevant (Assuming your notebook used '1' for relevant. If you used a word like 'Relevant', change the 1 below to 'Relevant')
        df['is_relevant'] = classifier.predict(X_daily)
        
        # Keep only the good articles
        df_filtered = df[df['is_relevant'] == 1].copy()
        print(f"🗑️ Filtered out {len(df) - len(df_filtered)} junk articles. {len(df_filtered)} passed the Bouncer.")
    else:
        df_filtered = df.copy()

    # === STAGE 2: THE EXPERT ===
    print("Scoring sentiment with FinBERT...")
    sentiments = []
    scores = []
    
    for headline in df_filtered["headline"]:
        try:
            # FinBERT reads the context
            result = sentiment_pipeline(str(headline))[0]
            label = result['label'] # 'positive', 'negative', or 'neutral'
            score = result['score'] # Confidence score
            
            sentiments.append(label)
            
            # Convert to a -1.0 to 1.0 scale for your dashboard graphs
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
    
    # Drop the temporary 'is_relevant' column to keep the CSV clean
    if 'is_relevant' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['is_relevant'])
        
    return df_filtered


def cleanup_old_news(days_to_keep=30):
    """Deletes news CSV files older than X days."""
    print(f"🧹 Janitor: Checking for news files older than {days_to_keep} days...")
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    # Find all the news CSV files in the folder
    news_files = glob.glob("data/raw_news/news_*.csv")
    
    for file_path in news_files:
        try:
            # Extract the date from the filename (e.g., news_20260222.csv -> 20260222)
            filename = os.path.basename(file_path)
            date_str = filename.split('_')[1].split('.')[0]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            
            # If the file's date is older than the cutoff, delete it
            if file_date < cutoff_date:
                os.remove(file_path)
                print(f"🗑️ Deleted old file: {filename}")
        except Exception as e:
            print(f"⚠️ Could not parse date for {file_path}: {e}")

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
    
    # Janitorial cleanup of old news files (keep the last 30 days)
    cleanup_old_news(days_to_keep=30)
