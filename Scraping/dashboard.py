import streamlit as st
import pandas as pd
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime

st.set_page_config(page_title="Executive AI Sentiment Terminal", layout="wide")

# === PREMIUM CSS STYLING ===
st.markdown("""
    <style>
    /* Force compact layout to minimize scrolling */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 98% !important;
    }
    
    /* We removed the 'header' hiding rule so your sidebar toggle button stays visible! */
    footer {visibility: hidden;}
    
    /* Optional: Hides the "Deploy" button in the top right so it looks like a real app */
    .stAppDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h3 style="color: white; margin-bottom: 0px; margin-top: -20px;">Executive AI Sentiment Terminal</h3>', unsafe_allow_html=True)

# === CUSTOM KPI CARD BUILDER (Now with custom border colors!) ===
def create_kpi_card(title, value, delta_text="", delta_type="", border_color="#4A90E2"):
    if delta_type == "bull": color, icon = "#00ff9d", "▲"
    elif delta_type == "bear": color, icon = "#ff4d4d", "▼"
    else: color, icon = "#ffffff", ""

    delta_html = ""
    if delta_text:
        delta_html = f'<div style="position: absolute; top: 10px; right: 15px; color: {color}; font-weight: 700; font-size: 0.9rem;">{icon} {delta_text}</div>'

    html = f"""
    <div style="
        background-color: #1e1e2f;
        border: 1px solid #33334d;
        border-left: 5px solid {border_color};
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        position: relative;
        height: 85px;
        margin-bottom: 5px;
    ">
        <p style="color: #a0a0b5; font-weight: 600; font-size: 0.95rem; margin: 0; padding: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{title}</p>
        <h2 style="color: #ffffff; font-weight: 700; margin: 2px 0 0 0; padding: 0; font-size: 1.8rem;">{value}</h2>
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
        try: df_list.append(pd.read_csv(f, encoding='utf-8'))
        except UnicodeDecodeError: df_list.append(pd.read_csv(f, encoding='latin-1'))
        except Exception: pass

    if not df_list: return pd.DataFrame()

    df_news = pd.concat(df_list, ignore_index=True).dropna(subset=['headline', 'compound'])
    if 'date' in df_news.columns:
        df_news['datetime'] = pd.to_datetime(df_news['date'], errors='coerce')
        df_news['display_date'] = df_news['datetime'].dt.strftime('%m-%d %H:%M')
        df_news['date_only'] = df_news['datetime'].dt.date
        
    return df_news

@st.cache_data(ttl=3600)
def get_historical_prices(ticker, days=30):
    if ticker == "All Market": return pd.DataFrame()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d").reset_index()
        hist['date_only'] = hist['Date'].dt.date
        return hist[['date_only', 'Close']]
    except: return pd.DataFrame()

df_news = load_news_data()
if df_news.empty:
    st.error("⚠️ Waiting for data to populate...")
    st.stop()

# === 2. SIDEBAR BUTTONS ===
st.sidebar.markdown('<p style="font-size:1.3rem; font-weight:700; color:white;">Terminal Controls</p>', unsafe_allow_html=True)
tickers = sorted(df_news['ticker'].dropna().unique().tolist())
selected_ticker = st.sidebar.radio("Select Asset:", ["All Market"] + tickers)

st.sidebar.divider()
st.sidebar.markdown('<p style="font-size:1.1rem; font-weight:600; color:white;">Today\'s Market Pulse</p>', unsafe_allow_html=True)
if not df_news.empty:
    for t in tickers[:5]:
        try:
            pulse_price = yf.Ticker(t).fast_info.last_price
            st.sidebar.write(f"**{t}**: ${pulse_price:.2f}")
        except: pass

filtered_news = df_news if selected_ticker == "All Market" else df_news[df_news['ticker'] == selected_ticker]
df_prices = get_historical_prices(selected_ticker, days=30)

# === 3. TOP ROW: PRIMARY KPIs ===
avg_compound = filtered_news['compound'].mean()
pos_articles = len(filtered_news[filtered_news['sentiment_label'] == 'positive'])
neg_articles = len(filtered_news[filtered_news['sentiment_label'] == 'negative'])

if avg_compound > 0.05: sent_text, sent_type = "Bullish", "bull"
elif avg_compound < -0.05: sent_text, sent_type = "Bearish", "bear"
else: sent_text, sent_type = "Neutral", "neutral"

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(create_kpi_card("Total Intel", f"{len(filtered_news):,}"), unsafe_allow_html=True)
with col2: st.markdown(create_kpi_card("Avg Sentiment", f"{avg_compound:.3f}", sent_text, sent_type), unsafe_allow_html=True)
with col3: st.markdown(create_kpi_card("Bullish Signals", f"{pos_articles:,}"), unsafe_allow_html=True)
with col4:
    if not df_prices.empty and selected_ticker != "All Market":
        latest_price = df_prices.iloc[-1]['Close']
        st.markdown(create_kpi_card("Latest Close", f"${latest_price:.2f}"), unsafe_allow_html=True)
    else:
        st.markdown(create_kpi_card("Bearish Signals", f"{neg_articles:,}"), unsafe_allow_html=True)

st.markdown('<hr style="margin: 10px 0px 10px 0px; border-color: #33334d;">', unsafe_allow_html=True)

# === 4. MIDDLE ROW: GRAPH (65%) & PREDICTIONS (35%) ===
col_graph, col_pred = st.columns([6.5, 3.5])

with col_graph:
    st.markdown(f'<p style="color:white; font-size:1.05rem; font-weight:600; margin-bottom:0px;">📊 {selected_ticker} Correlation Overlay</p>', unsafe_allow_html=True)
    if not filtered_news.empty and not df_prices.empty and selected_ticker != "All Market":
        daily_sentiment = filtered_news.groupby('date_only')['compound'].mean().reset_index()
        merged_chart_data = pd.merge(daily_sentiment, df_prices, on='date_only', how='inner').sort_values('date_only')
        
        if not merged_chart_data.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=merged_chart_data['date_only'], y=merged_chart_data['compound'], name="Sentiment", mode='lines+markers', line=dict(color='#4A90E2', width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=merged_chart_data['date_only'], y=merged_chart_data['Close'], name="Stock Price", mode='lines+markers', line=dict(color='#ffffff', width=2, dash='dot')), secondary_y=True)
            
            # Graph height set to ~280px to perfectly match the 3 stacked prediction boxes
            fig.update_layout(
                height=285, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified"
            )
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=False, gridcolor='#33334d')
            fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Accumulating dates to plot Price vs. Sentiment overlay...")
    else: st.info("Select a specific asset from the sidebar to view the overlay.")

with col_pred:
    st.markdown('<p style="color:white; font-size:1.05rem; font-weight:600; margin-bottom:0px;">🔮 AI Overnight Gap Prediction</p>', unsafe_allow_html=True)
    if not filtered_news.empty and not df_prices.empty and selected_ticker != "All Market":
        clean_dates = pd.to_datetime(filtered_news['datetime']).dropna().dt.date
        if clean_dates.empty:
            st.warning("No valid timestamps found.")
        else:
            latest_date = clean_dates.max()
            yesterday = latest_date - datetime.timedelta(days=1)
            
            overnight_news = filtered_news[
                ((pd.to_datetime(filtered_news['datetime']).dt.date == yesterday) & (pd.to_datetime(filtered_news['datetime']).dt.hour >= 16)) |
                ((pd.to_datetime(filtered_news['datetime']).dt.date == latest_date) & (pd.to_datetime(filtered_news['datetime']).dt.hour < 9))
            ]
        
            if not overnight_news.empty:
                overnight_sentiment = overnight_news['compound'].mean()
                last_close = df_prices.iloc[-1]['Close']
                volatility = df_prices['Close'].pct_change().std() 
                
                predicted_move_pct = overnight_sentiment * (volatility * 100) 
                predicted_price = last_close * (1 + (predicted_move_pct / 100))
                move_dollar = predicted_price - last_close
                
                if overnight_sentiment > 0.05: p_text, p_type = "Bullish", "bull"
                elif overnight_sentiment < -0.05: p_text, p_type = "Bearish", "bear"
                else: p_text, p_type = "Neutral", "neutral"
                
                # Stack 3 AI-specific KPI cards (Notice the purple border color!)
                st.markdown(create_kpi_card("Overnight Volume", f"{len(overnight_news)}", "Articles", "", border_color="#b829ff"), unsafe_allow_html=True)
                st.markdown(create_kpi_card("Overnight Sentiment", f"{overnight_sentiment:.3f}", p_text, p_type, border_color="#b829ff"), unsafe_allow_html=True)
                st.markdown(create_kpi_card("Projected Open", f"${predicted_price:.2f}", f"{'+' if move_dollar > 0 else ''}${move_dollar:.2f}", p_type, border_color="#b829ff"), unsafe_allow_html=True)
            else:
                st.markdown("<br><p style='color:#a0a0b5; text-align:center;'>Waiting for overnight news (4 PM - 9:30 AM) to run prediction...</p>", unsafe_allow_html=True)
    else:
        st.markdown("<br><p style='color:#a0a0b5; text-align:center;'>Select a specific asset to activate prediction.</p>", unsafe_allow_html=True)

# === 5. BOTTOM ROW: FULL WIDTH NEWS FEED ===
st.markdown('<p style="color:white; font-size:1.05rem; font-weight:600; margin-bottom:5px; margin-top:10px;">📰 High-Impact Intel Feed</p>', unsafe_allow_html=True)

def sentiment_color(val):
    if val == 'positive': return 'background-color: rgba(74, 144, 226, 0.1); color: #4A90E2;' 
    elif val == 'negative': return 'background-color: rgba(255, 77, 77, 0.1); color: #ff4d4d;' 
    return 'color: #ffffff;'

display_cols = ['display_date', 'ticker', 'source', 'headline', 'sentiment_label', 'compound']
clean_display_df = filtered_news[display_cols].copy()

# Advanced sorting: Rank by absolute magnitude (highest impact first)
clean_display_df['impact'] = clean_display_df['compound'].abs()
clean_display_df = clean_display_df.sort_values(by=['impact', 'display_date'], ascending=[False, False])
clean_display_df = clean_display_df.drop(columns=['impact'])

clean_display_df.columns = ['Time', 'Asset', 'Source', 'Headline', 'Signal', 'Score']

# Full width table at the bottom, height adjusted to roughly 250px to fit without major scrolling
st.dataframe(clean_display_df.style.map(sentiment_color, subset=['Signal']), use_container_width=True, hide_index=True, height=250)