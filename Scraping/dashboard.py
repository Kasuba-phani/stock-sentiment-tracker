import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px
import plotly.graph_objects as go

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="AI Stock Sentiment Tracker", layout="wide", page_icon="📈")

# Custom CSS to make it look premium
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI Stock Sentiment & Price Tracker")
st.markdown("Automated NLP pipeline analyzing daily financial news against market movements using **FinBERT** and **TF-IDF**.")

# === 1. LOAD THE DATA (DYNAMICALLY) ===
@st.cache_data
def load_data():
    """Loads and stitches all historical data together for trend analysis"""
    # 1. Load News Data
    all_news_files = glob.glob("data/raw_news/*.csv")
    if not all_news_files:
        return pd.DataFrame(), pd.DataFrame()
    
    df_news = pd.concat([pd.read_csv(f) for f in all_news_files], ignore_index=True)
    df_news['date'] = pd.to_datetime(df_news['date']).dt.date
    
    # 2. Load Price Data
    try:
        df_prices = pd.read_csv("data/stock_data_summary.csv")
        df_prices['date'] = pd.to_datetime(df_prices['date']).dt.date
    except FileNotFoundError:
        df_prices = pd.DataFrame()
        
    return df_news, df_prices

df_news, df_prices = load_data()

if df_news.empty:
    st.error("⚠️ Waiting for GitHub Actions to scrape news data...")
    st.stop()

# === 2. INTERACTIVE SIDEBAR FILTERS ===
st.sidebar.header("🔍 Filter Data")

# Get a unique list of tickers available in the data
available_tickers = df_news['ticker'].unique().tolist()
selected_ticker = st.sidebar.selectbox("Select a Ticker to Analyze:", ["All Market"] + available_tickers)

# Filter the data based on the user's selection
if selected_ticker == "All Market":
    filtered_news = df_news
    if not df_prices.empty:
        filtered_prices = df_prices
else:
    filtered_news = df_news[df_news['ticker'] == selected_ticker]
    if not df_prices.empty:
        filtered_prices = df_prices[df_prices['ticker'] == selected_ticker]

# === 3. TOP LEVEL KPIs (Metrics) ===
st.subheader(f"📊 {selected_ticker} Overview")

# Calculate overall sentiment for the selected filter
avg_compound = filtered_news['compound'].mean()
total_articles = len(filtered_news)
pos_articles = len(filtered_news[filtered_news['sentiment_label'] == 'positive'])
neg_articles = len(filtered_news[filtered_news['sentiment_label'] == 'negative'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Articles Analyzed", total_articles)
col2.metric("Average Sentiment Score", f"{avg_compound:.2f}", 
            delta="Bullish" if avg_compound > 0.05 else ("Bearish" if avg_compound < -0.05 else "Neutral"),
            delta_color="normal")
col3.metric("Positive News", pos_articles)
col4.metric("Negative News", neg_articles)

st.divider()

# === 4. INTERACTIVE PLOTLY VISUALS ===
st.subheader("📈 Sentiment Trends Over Time")

if not filtered_news.empty:
    # Group sentiment by date to show a trendline
    daily_sentiment = filtered_news.groupby('date')['compound'].mean().reset_index()
    
    # Create a dynamic interactive line chart
    fig_trend = px.line(daily_sentiment, x='date', y='compound', markers=True, 
                        title="Average Daily Sentiment Score",
                        labels={'compound': 'Sentiment Score', 'date': 'Date'})
    
    # Add a horizontal line at 0 (Neutral)
    fig_trend.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    
    # Customize line color based on overall sentiment
    line_color = "green" if avg_compound > 0 else "red"
    fig_trend.update_traces(line_color=line_color)
    
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# === 5. THE NEWS FEED ===
st.subheader(f"📰 Processed Headlines for {selected_ticker}")

# Create visual tags for sentiment in the dataframe
def sentiment_color(val):
    color = 'green' if val == 'positive' else ('red' if val == 'negative' else 'gray')
    return f'color: {color}'

display_cols = ['date', 'ticker', 'source', 'headline', 'sentiment_label', 'compound']
st.dataframe(
    filtered_news[display_cols].sort_values(by=['date', 'compound'], ascending=[False, False])
    .style.map(sentiment_color, subset=['sentiment_label']),
    use_container_width=True,
    hide_index=True
)