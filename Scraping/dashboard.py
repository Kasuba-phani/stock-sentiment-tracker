import streamlit as st
import pandas as pd
from glob import glob
import os

# Make the dashboard span the full width of the screen
st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide", page_icon="📈")

st.title("📈 AI Stock Sentiment & Price Tracker")
st.markdown("Automated NLP pipeline analyzing daily financial news against market movements.")

# === 1. LOAD THE DATA SAFELY ===
st.sidebar.header("Data Status")
try:
    # Grab the newest CSV file from the raw_news folder
    latest_news_file = sorted(glob("data/raw_news/*.csv"))[-1]
    df_news = pd.read_csv(latest_news_file)
    st.sidebar.success(f"News Data Loaded: {os.path.basename(latest_news_file)}")
except IndexError:
    st.error("⚠️ Waiting for GitHub Actions to scrape news data...")
    st.stop()

try:
    # Load the stock prices (it saves to the main folder based on your Code 3)
    df_prices = pd.read_csv("stock_data_summary.csv")
    # Get only the latest date's data to avoid mixing old records
    latest_date = df_prices['date'].max()
    df_prices = df_prices[df_prices['date'] == latest_date]
    st.sidebar.success(f"Stock Data Loaded: {latest_date}")
except FileNotFoundError:
    st.error("⚠️ Stock price data not found.")
    st.stop()

# === 2. MERGE AND ANALYZE ===
# Calculate the average sentiment (compound score) for each ticker
avg_sentiment = df_news.groupby('ticker')['compound'].mean().reset_index()

# Merge the sentiment with the stock prices
merged = pd.merge(avg_sentiment, df_prices, on='ticker')

# === 3. BUILD THE VISUALS ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Today's Sentiment vs. Stock Price")
    # A scatter plot helps visualize if highly positive news correlates with higher prices
    st.scatter_chart(merged, x='compound', y='close', color='ticker')

with col2:
    st.subheader("Market Summary")
    st.dataframe(merged[['ticker', 'close', 'compound']].sort_values('compound', ascending=False), hide_index=True)

st.divider()

st.subheader("📰 Latest Processed Headlines")
# Filter down to the most important columns for the user
display_news = df_news[['date', 'ticker', 'source', 'headline', 'sentiment_label', 'compound']]
st.dataframe(display_news.sort_values('compound', ascending=False), use_container_width=True)
