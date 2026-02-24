import streamlit as st
import pandas as pd
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

st.set_page_config(page_title="Executive AI Sentiment Terminal", layout="wide", page_icon="📈")

# === PREMIUM CSS STYLING ===
st.markdown("""
    <style>
    /* Make the title massive */
    .big-title {
        font-size: 3.5rem !important;
        font-weight: 700;
        margin-bottom: 0px;
        color: #ffffff;
    }
    /* Hide default header to make it cleaner */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-title">📈 Executive AI Sentiment Terminal</p>', unsafe_allow_html=True)
st.markdown("Real-time NLP correlation analysis between financial news and market movements.")

# === CUSTOM KPI CARD BUILDER ===
def create_kpi_card(title, value, delta_text="", delta_type=""):
    """Generates a custom HTML KPI card to bypass Streamlit's stubborn default CSS"""
    if delta_type == "bull":
        color = "#00ff9d" # Bright neon green for positive
        icon = "▲"
    elif delta_type == "bear":
        color = "#ff4d4d" # Bright red for negative
        icon = "▼"
    else:
        color = "#ffffff"
        icon = ""

    delta_html = ""
    if delta_text:
        # Pinned exactly to the top right!
        delta_html = f'<div style="position: absolute; top: 15px; right: 20px; color: {color}; font-weight: 700; font-size: 1.1rem;">{icon} {delta_text}</div>'

    html = f"""
    <div style="
        background-color: #1e1e2f;
        border: 1px solid #33334d;
        border-left: 5px solid #4A90E2;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        position: relative;
        height: 110px;
        margin-bottom: 1rem;
    ">
        <p style="color: #ffffff; font-weight: 600; font-size: 1.1rem; margin: 0; padding: 0;">{title}</p>
        <h2 style="color: #ffffff; font-weight: 700; margin: 10px 0 0 0; padding: 0; font-size: 2.2rem;">{value}</h2>
        {delta_html}
    </div>
    """
    return html

# === 1. LOAD THE DATA ===
@st.cache_data
def load_news_data():
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
        return pd.DataFrame()

    df_news = pd.concat(df_list, ignore_index=True)
    df_news = df_news.dropna(subset=['headline', 'compound'])
    
    if 'date' in df_news.columns:
        df_news['datetime'] = pd.to_datetime(df_news['date'], errors='coerce')
        df_news['display_date'] = df_news['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        df_news['date_only'] = df_news['datetime'].dt.date
        
    return df_news

@st.cache_data(ttl=3600)
def get_historical_prices(ticker, days=30):
    if ticker == "All Market": return pd.DataFrame()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")
        hist = hist.reset_index()
        hist['date_only'] = hist['Date'].dt.date
        return hist[['date_only', 'Close']]
    except:
        return pd.DataFrame()

df_news = load_news_data()

if df_news.empty:
    st.error("⚠️ Waiting for data to populate...")
    st.stop()

# === 2. SIDEBAR FILTERS & MARKET PULSE ===
st.sidebar.header("🔍 Terminal Controls")

tickers = sorted(df_news['ticker'].dropna().unique().tolist())
selected_ticker = st.sidebar.selectbox("Select Asset:", ["All Market"] + tickers)

st.sidebar.divider()

st.sidebar.subheader("📡 Today's Market Pulse")
if not df_news.empty:
    for t in tickers[:5]:
        try:
            pulse_price = yf.Ticker(t).fast_info.last_price
            st.sidebar.write(f"**{t}**: ${pulse_price:.2f}")
        except:
            pass

if selected_ticker == "All Market":
    filtered_news = df_news
else:
    filtered_news = df_news[df_news['ticker'] == selected_ticker]

df_prices = get_historical_prices(selected_ticker, days=30)

# === 3. PREMIUM KPI CARDS ===
avg_compound = filtered_news['compound'].mean()
total_articles = len(filtered_news)
pos_articles = len(filtered_news[filtered_news['sentiment_label'] == 'positive'])
neg_articles = len(filtered_news[filtered_news['sentiment_label'] == 'negative'])

# Determine Delta logic for the average sentiment
if avg_compound > 0.05:
    sent_text, sent_type = "Bullish", "bull"
elif avg_compound < -0.05:
    sent_text, sent_type = "Bearish", "bear"
else:
    sent_text, sent_type = "Neutral", "neutral"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(create_kpi_card("Total Intel", f"{total_articles:,}"), unsafe_allow_html=True)
with col2:
    st.markdown(create_kpi_card("Avg Sentiment", f"{avg_compound:.3f}", sent_text, sent_type), unsafe_allow_html=True)
with col3:
    st.markdown(create_kpi_card("Bullish Signals", f"{pos_articles:,}"), unsafe_allow_html=True)
with col4:
    if not df_prices.empty and selected_ticker != "All Market":
        latest_price = df_prices.iloc[-1]['Close']
        st.markdown(create_kpi_card("Latest Close", f"${latest_price:.2f}"), unsafe_allow_html=True)
    else:
        st.markdown(create_kpi_card("Bearish Signals", f"{neg_articles:,}"), unsafe_allow_html=True)

# === 4. DUAL Y-AXIS LINE CHART ===
st.subheader(f"📊 {selected_ticker} Correlation: Price vs. Sentiment")

if not filtered_news.empty and not df_prices.empty and selected_ticker != "All Market":
    daily_sentiment = filtered_news.groupby('date_only')['compound'].mean().reset_index()
    merged_chart_data = pd.merge(daily_sentiment, df_prices, on='date_only', how='inner').sort_values('date_only')
    
    if not merged_chart_data.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. Sentiment Line (Tech Blue)
        fig.add_trace(
            go.Scatter(x=merged_chart_data['date_only'], y=merged_chart_data['compound'], 
                       name="Sentiment", mode='lines+markers', 
                       line=dict(color='#4A90E2', width=3)),
            secondary_y=False,
        )
        
        # 2. Stock Price Line (White, Dotted)
        fig.add_trace(
            go.Scatter(x=merged_chart_data['date_only'], y=merged_chart_data['Close'], 
                       name="Stock Price", mode='lines+markers', 
                       line=dict(color='#ffffff', width=2, dash='dot')),
            secondary_y=True,
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False, gridcolor='#33334d')
        fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
        fig.update_xaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Accumulating dates to plot Price vs. Sentiment overlay...")
else:
    st.info("Select a specific asset from the sidebar to view the overlay.")

st.divider()

# === 5. THE RAW INTEL FEED ===
st.subheader("📰 Live Intel Feed")

def sentiment_color(val):
    if val == 'positive': return 'background-color: rgba(74, 144, 226, 0.1); color: #4A90E2;' 
    elif val == 'negative': return 'background-color: rgba(255, 77, 77, 0.1); color: #ff4d4d;' 
    return 'color: #ffffff;'

display_cols = ['display_date', 'ticker', 'source', 'headline', 'sentiment_label', 'compound']
clean_display_df = filtered_news[display_cols].sort_values(by=['display_date', 'compound'], ascending=[False, False])
clean_display_df.columns = ['Time (HH:MM)', 'Asset', 'Source', 'Headline', 'Signal', 'Score']

st.dataframe(
    clean_display_df.style.map(sentiment_color, subset=['Signal']),
    use_container_width=True, hide_index=True, height=400
)
# === 6. THE OVERNIGHT PREDICTION ENGINE ===
st.divider()
st.subheader(f"🔮 AI Predictive Engine: {selected_ticker} (Overnight Gap Analysis)")

if not filtered_news.empty and not df_prices.empty and selected_ticker != "All Market":
    import datetime
    
    # 1. Get the latest available date in our data
    latest_date = pd.to_datetime(filtered_news['datetime']).dt.date.max()
    yesterday = latest_date - datetime.timedelta(days=1)
    
    # 2. Filter for "Overnight News" (4:00 PM yesterday to 9:30 AM today)
    # Note: Streamlit uses UTC, so you might need to adjust hours based on your timezone later
    overnight_news = filtered_news[
        ((pd.to_datetime(filtered_news['datetime']).dt.date == yesterday) & (pd.to_datetime(filtered_news['datetime']).dt.hour >= 16)) |
        ((pd.to_datetime(filtered_news['datetime']).dt.date == latest_date) & (pd.to_datetime(filtered_news['datetime']).dt.hour < 9))
    ]
    
    if not overnight_news.empty:
        # 3. Calculate Overnight Sentiment
        overnight_sentiment = overnight_news['compound'].mean()
        overnight_count = len(overnight_news)
        
        # 4. Get the last known close price and recent volatility
        last_close = df_prices.iloc[-1]['Close']
        volatility = df_prices['Close'].pct_change().std() # How jumpy the stock has been lately
        
        # 5. The Prediction Math (Simplified Alpha Model)
        # We assume the price will move by a fraction of its normal volatility, guided by sentiment
        predicted_move_pct = overnight_sentiment * (volatility * 100) 
        predicted_price = last_close * (1 + (predicted_move_pct / 100))
        
        # 6. Build the UI
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            st.info(f"**Overnight Articles:** {overnight_count}")
            st.write(f"News captured since previous market close.")
            
        with pred_col2:
            st.info(f"**Overnight Sentiment:** {overnight_sentiment:.3f}")
            if overnight_sentiment > 0.1: st.success("Strong Bullish Momentum")
            elif overnight_sentiment < -0.1: st.error("Strong Bearish Momentum")
            else: st.warning("Neutral / Mixed Signals")
            
        with pred_col3:
            st.info(f"**Predicted Next Open:** ${predicted_price:.2f}")
            move_dollar = predicted_price - last_close
            st.write(f"Projected Gap: {'+' if move_dollar > 0 else ''}${move_dollar:.2f}")
            
    else:
        st.write("Not enough overnight news data collected yet to run the prediction engine.")
else:
    st.write("Select a specific asset from the sidebar to activate the AI Prediction Engine.")