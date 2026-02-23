import streamlit as st
import pandas as pd
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Executive AI Sentiment Terminal", layout="wide", page_icon="📈")

# === PREMIUM CSS STYLING ===
st.markdown("""
    <style>
    /* Beautiful KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #1e1e2f;
        border: 1px solid #33334d;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #00ff9d; /* Neon green accent */
    }
    
    /* Change the KPI label color */
    div[data-testid="stMetricLabel"] {
        color: #a0a0b5 !important;
        font-weight: 600;
    }
    
    /* Hide the default Streamlit footer */
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Executive AI Sentiment Terminal")
st.markdown("Real-time NLP correlation analysis between financial news and market movements.")

# === 1. LOAD AND CLEAN THE DATA ===
@st.cache_data
def load_data():
    all_news_files = glob.glob("data/raw_news/*.csv")
    df_list = []
    
    for f in all_news_files:
        try:
            df_list.append(pd.read_csv(f, encoding='utf-8'))
        except UnicodeDecodeError:
            df_list.append(pd.read_csv(f, encoding='latin-1'))
        except Exception:
            pass

    if not df_list:
        return pd.DataFrame(), pd.DataFrame()

    df_news = pd.concat(df_list, ignore_index=True)
    
    # FIX 1: Drop NaN values to clean up the dashboard
    df_news = df_news.dropna(subset=['headline', 'compound'])
    
    # FIX 2: Parse datetime properly to keep HH:MM if available
    if 'date' in df_news.columns:
        df_news['datetime'] = pd.to_datetime(df_news['date'], errors='coerce')
        # Create a display column formatted as YYYY-MM-DD HH:MM
        df_news['display_date'] = df_news['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        # Keep a strict date column for merging with daily stock prices
        df_news['date_only'] = df_news['datetime'].dt.date
    
    try:
        df_prices = pd.read_csv("data/stock_data_summary.csv")
        df_prices = df_prices.dropna(subset=['close'])
        if 'date' in df_prices.columns:
            df_prices['date_only'] = pd.to_datetime(df_prices['date'], errors='coerce').dt.date
    except FileNotFoundError:
        df_prices = pd.DataFrame()
        
    return df_news, df_prices

df_news, df_prices = load_data()

if df_news.empty:
    st.error("⚠️ Waiting for data to populate...")
    st.stop()

# === 2. SIDEBAR FILTERS ===
st.sidebar.header("🔍 Terminal Controls")

tickers = sorted(df_news['ticker'].dropna().unique().tolist())
selected_ticker = st.sidebar.selectbox("Select Asset:", ["All Market"] + tickers)

if selected_ticker == "All Market":
    filtered_news = df_news
    filtered_prices = df_prices
else:
    filtered_news = df_news[df_news['ticker'] == selected_ticker]
    if not df_prices.empty:
        filtered_prices = df_prices[df_prices['ticker'] == selected_ticker]

# === 3. PREMIUM KPI CARDS ===
avg_compound = filtered_news['compound'].mean()
total_articles = len(filtered_news)
pos_articles = len(filtered_news[filtered_news['sentiment_label'] == 'positive'])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Intel Analyzed", f"{total_articles:,}")
with col2:
    st.metric("Avg Sentiment Score", f"{avg_compound:.3f}", 
              delta="Bullish" if avg_compound > 0.05 else ("Bearish" if avg_compound < -0.05 else "Neutral"))
with col3:
    st.metric("Bullish Signals", pos_articles)
with col4:
    # If price data exists, show the latest close. Otherwise, show Bearish Signals.
    if not filtered_prices.empty and selected_ticker != "All Market":
        latest_price = filtered_prices.sort_values('date_only').iloc[-1]['close']
        st.metric("Latest Close Price", f"${latest_price:.2f}")
    else:
        st.metric("Bearish Signals", len(filtered_news[filtered_news['sentiment_label'] == 'negative']))

st.markdown("<br>", unsafe_allow_html=True)

# === 4. DUAL Y-AXIS CHART (PRICE VS SENTIMENT) ===
st.subheader(f"📊 {selected_ticker} Correlation Engine: Price vs. Sentiment")

if not filtered_news.empty and not filtered_prices.empty and selected_ticker != "All Market":
    # Group sentiment by day to match daily stock prices
    daily_sentiment = filtered_news.groupby('date_only')['compound'].mean().reset_index()
    
    # Merge price and sentiment on the exact date
    merged_chart_data = pd.merge(daily_sentiment, filtered_prices, on='date_only', how='inner').sort_values('date_only')
    
    if not merged_chart_data.empty:
        # Create the dual-axis Plotly chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. Sentiment as a Bar Chart (Left Y-Axis)
        colors = ['rgba(0, 255, 157, 0.6)' if val > 0 else 'rgba(255, 77, 77, 0.6)' for val in merged_chart_data['compound']]
        fig.add_trace(
            go.Bar(x=merged_chart_data['date_only'], y=merged_chart_data['compound'], name="Sentiment", marker_color=colors),
            secondary_y=False,
        )
        
        # 2. Stock Price as a Line Chart (Right Y-Axis)
        fig.add_trace(
            go.Scatter(x=merged_chart_data['date_only'], y=merged_chart_data['close'], name="Stock Price", mode='lines+markers', line=dict(color='#ffffff', width=3)),
            secondary_y=True,
        )
        
        # Format the chart to look professional
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False, gridcolor='#33334d')
        fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
        fig.update_xaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough overlapping dates to plot Price vs. Sentiment yet. Check back tomorrow!")
else:
    st.info("Select a specific asset from the sidebar to view the Price vs. Sentiment overlay.")

st.divider()

# === 5. THE RAW INTEL FEED ===
st.subheader("📰 Live Intel Feed")

# Format dataframe to highlight rows based on sentiment
def sentiment_color(val):
    if val == 'positive': return 'background-color: rgba(0, 255, 157, 0.1); color: #00ff9d;'
    elif val == 'negative': return 'background-color: rgba(255, 77, 77, 0.1); color: #ff4d4d;'
    return 'color: gray;'

display_cols = ['display_date', 'ticker', 'source', 'headline', 'sentiment_label', 'compound']
clean_display_df = filtered_news[display_cols].sort_values(by=['display_date', 'compound'], ascending=[False, False])

# Rename columns for the UI
clean_display_df.columns = ['Time (HH:MM)', 'Asset', 'Source', 'Headline', 'Signal', 'Score']

st.dataframe(
    clean_display_df.style.map(sentiment_color, subset=['Signal']),
    use_container_width=True, hide_index=True, height=400
)