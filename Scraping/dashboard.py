import streamlit as st
import pandas as pd
from glob import glob

st.title("Stock Sentiment Dashboard")

# Load latest data
latest_news = sorted(glob("data/raw_news/*.csv"))[-1]
latest_prices = sorted(glob("data/stock_prices/*.csv"))[-1]

df_news = pd.read_csv(latest_news)
df_prices = pd.read_csv(latest_prices)

# Merge data
merged = pd.merge(
    df_news.groupby('ticker')['compound'].mean().reset_index(),
    df_prices,
    on='ticker'
)

# Show results
st.write("## Today's Sentiment vs Prices")
st.scatter_chart(merged, x='compound', y='close', color='ticker')

st.write("## Latest Headlines")
st.dataframe(df_news[['ticker','headline','compound']].sort_values('compound'))
