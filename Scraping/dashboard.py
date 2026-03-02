"""
Author: Phanidhar Kasuba
Description: Automated MLOps Sentiment Terminal
Copyright (c) 2026. All rights reserved.
"""
import streamlit as st
import pandas as pd
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime
import os

st.set_page_config(page_title="S.E.N.S.E. Terminal", layout="wide")

# === PREMIUM RESPONSIVE CSS ===
st.markdown("""
    <style>
    .block-container {
        padding-top: 3.8rem !important; 
        padding-bottom: 0rem !important;
        max-width: 98% !important; 
    }
    footer {visibility: hidden;} 
    </style>
    """, unsafe_allow_html=True)

# === BRANDING: STABLE LOGO & TITLE WITH TRADEMARK ===
st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 0px;">
        <img src="https://raw.githubusercontent.com/Kasuba-phani/stock-sentiment-tracker/main/Scraping/sense_logo.jpeg" style="width: 45px; height: 45px; object-fit: contain; margin-right: 15px; border-radius: 6px;">
        <h1 style="color: var(--text-color); font-size: 2.2rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: 0.5px;">S.E.N.S.E.™ Terminal</h1>
    </div>
""", unsafe_allow_html=True)

# === THE UI WATERMARK ===
st.markdown("<p style='color: gray; font-size: 0.85rem; margin-top: 2px; margin-bottom: 15px;'>Sentiment Evaluation & News Scoring Engine | Engineered by Phanidhar Kasuba | M.S. Data Analytics</p>", unsafe_allow_html=True)

# === TICKER TO LOGO/NAME DICTIONARY ===
TICKER_MAP = {
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "META": "Meta",
    "NFLX": "Netflix"
}
REV_TICKER_MAP = {v: k for k, v in TICKER_MAP.items()}

LOGO_MAP = {
    "AAPL": "https://companiesmarketcap.com/img/company-logos/64/AAPL.webp",
    "GOOGL": "https://companiesmarketcap.com/img/company-logos/64/GOOG.webp",
    "AMZN": "https://companiesmarketcap.com/img/company-logos/64/AMZN.webp",
    "MSFT": "https://companiesmarketcap.com/img/company-logos/64/MSFT.webp",
    "NVDA": "https://companiesmarketcap.com/img/company-logos/64/NVDA.webp",
    "TSLA": "https://companiesmarketcap.com/img/company-logos/64/TSLA.webp",
    "META": "https://companiesmarketcap.com/img/company-logos/64/META.webp",
    "NFLX": "https://companiesmarketcap.com/img/company-logos/64/NFLX.webp"
}

# === CUSTOM KPI CARD BUILDER ===
def create_kpi_card(title, value, delta_text="", delta_type="", border_color="#4A90E2", context=""):
    if delta_type == "bull": color, icon = "#00b36b", "▲" 
    elif delta_type == "bear": color, icon = "#ff4d4d", "▼"
    else: color, icon = "gray", ""

    delta_html = ""
    if delta_text:
        delta_html = f'<div style="position: absolute; bottom: 10px; right: 15px; color: {color}; font-weight: 700; font-size: 0.9rem;">{icon} {delta_text}</div>'

    context_html = f"<span style='font-size: 0.7rem; font-weight: 400; color: gray; margin-left: 6px;'>{context}</span>" if context else ""

    html = f"""
    <div style="
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 5px solid {border_color};
        padding: 5px 15px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        position: relative;
        height: 75px; 
        margin-bottom: 0px;
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
        align-items: center; 
        text-align: center;
    ">
        <p style="color: gray; font-weight: 700; font-size: 0.95rem; margin: 0; padding: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{title}{context_html}</p>
        <h2 style="color: var(--text-color); font-weight: 700; margin: 0px 0 0 0; padding: 0; font-size: 1.6rem;">{value}</h2>
        {delta_html}
    </div>
    """
    return html

def format_large_number(num):
    try:
        num = float(num)
        if pd.isna(num): return "N/A"
        if num >= 1e12: return f"${num/1e12:.2f}T"
        if num >= 1e9: return f"${num/1e9:.2f}B"
        if num >= 1e6: return f"{num/1e6:.2f}M"
        return f"{num:,.0f}"
    except: return "N/A"

# === 1. LOAD THE DATA ===
@st.cache_data(ttl=3600)
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

# --- UPDATED: Bulletproof Fundamentals Loader ---
@st.cache_data(ttl=3600)
def load_fundamentals():
    if os.path.exists("data/stock_data_summary.csv"):
        try:
            df = pd.read_csv("data/stock_data_summary.csv")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date') 
            df = df.drop_duplicates(subset=['ticker'], keep='last')
            return df
        except: return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_historical_prices(ticker, days=30):
    if ticker == "All Market": return pd.DataFrame()
    try:
        hist = yf.Ticker(ticker).history(period=f"{days}d").reset_index()
        hist['date_only'] = hist['Date'].dt.date
        return hist[['date_only', 'Close']]
    except: return pd.DataFrame()

df_news = load_news_data()
df_fundamentals = load_fundamentals()

if df_news.empty:
    st.error("⚠️ Waiting for data to populate...")
    st.stop()

# === EXTRACT LATEST DATES FOR CONTEXT ===
latest_dt = pd.to_datetime(df_news['datetime']).max()
latest_date_str = latest_dt.strftime('%b %d') if pd.notnull(latest_dt) else "N/A"

# === 2. SIDEBAR WITH REAL LOGOS & TIMESTAMPS ===
st.sidebar.markdown('<p style="font-size:1.3rem; font-weight:700; color:var(--text-color);">Terminal Controls</p>', unsafe_allow_html=True)
available_tickers = sorted(df_news['ticker'].dropna().unique().tolist())

display_options = ["🌍 All Market"] + [TICKER_MAP.get(t, t) for t in available_tickers]
selected_display = st.sidebar.radio("Select Asset:", display_options)

if selected_display == "🌍 All Market":
    selected_ticker = "All Market"
else:
    selected_ticker = REV_TICKER_MAP.get(selected_display, selected_display)

st.sidebar.divider()

if selected_ticker == "All Market":
    st.sidebar.markdown('<p style="font-size:1.1rem; font-weight:600; color:var(--text-color);">Market Pulse <span style="font-size:0.75rem; color:gray; font-weight:400; margin-left: 5px;">(Live)</span></p>', unsafe_allow_html=True)
    for t in available_tickers[:5]:
        try:
            pulse_price = yf.Ticker(t).fast_info.last_price
            st.sidebar.write(f"**{TICKER_MAP.get(t, t)}**: ${pulse_price:.2f}")
        except: pass
else:
    logo_url = LOGO_MAP.get(selected_ticker, "")
    
    st.sidebar.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <img src="{logo_url}" width="32" style="border-radius: 4px; margin-right: 12px; background-color: white; padding: 2px;" onerror="this.style.display='none'">
            <p style="font-size:1.1rem; font-weight:600; color:var(--text-color); margin:0;">{TICKER_MAP.get(selected_ticker, selected_ticker)} Profile </p>
        </div>
    """, unsafe_allow_html=True)
    if not df_fundamentals.empty and selected_ticker in df_fundamentals['ticker'].values:
        asset_funds = df_fundamentals[df_fundamentals['ticker'] == selected_ticker].iloc[-1]
        st.sidebar.markdown(f"""
        <div style="background-color: var(--secondary-background-color); padding: 15px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.2);">
            <p style="margin: 0px 0px 5px 0px; color: gray; font-size: 0.85rem;">Market Cap</p>
            <p style="margin: 0px 0px 10px 0px; color: var(--text-color); font-weight: bold; font-size: 1.05rem;">{format_large_number(asset_funds.get('market_cap'))}</p>
            <p style="margin: 0px 0px 5px 0px; color: gray; font-size: 0.85rem;">P/E Ratio</p>
            <p style="margin: 0px 0px 10px 0px; color: var(--text-color); font-weight: bold; font-size: 1.05rem;">{asset_funds.get('pe_ratio', 'N/A')}</p>
            <p style="margin: 0px 0px 5px 0px; color: gray; font-size: 0.85rem;">Daily Volume</p>
            <p style="margin: 0px 0px 10px 0px; color: var(--text-color); font-weight: bold; font-size: 1.05rem;">{format_large_number(asset_funds.get('volume'))}</p>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p style="margin: 0px 0px 2px 0px; color: gray; font-size: 0.75rem;">52W High</p>
                    <p style="margin: 0px; color: #00b36b; font-weight: bold;">${asset_funds.get('52_week_high', 'N/A')}</p>
                </div>
                <div>
                    <p style="margin: 0px 0px 2px 0px; color: gray; font-size: 0.75rem;">52W Low</p>
                    <p style="margin: 0px; color: #ff4d4d; font-weight: bold;">${asset_funds.get('52_week_low', 'N/A')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.info("Awaiting fundamental data sync from warehouse...")

filtered_news = df_news if selected_ticker == "All Market" else df_news[df_news['ticker'] == selected_ticker]
df_prices = get_historical_prices(selected_ticker, days=30)

# === 3. TOP ROW: PRIMARY KPIs ===
avg_compound = filtered_news['compound'].mean()
pos_articles = len(filtered_news[filtered_news['sentiment_label'] == 'positive'])
neg_articles = len(filtered_news[filtered_news['sentiment_label'] == 'negative'])

if avg_compound > 0.05: sent_text, sent_type = "Bullish", "bull"
elif avg_compound < -0.05: sent_text, sent_type = "Bearish", "bear"
else: sent_text, sent_type = "Neutral", "neutral"

st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(create_kpi_card("Total Intel", f"{len(filtered_news):,}", "", "", "#4A90E2", f"(Up to {latest_date_str})"), unsafe_allow_html=True)
with col2: st.markdown(create_kpi_card("Avg Sentiment", f"{avg_compound:.3f}", sent_text, sent_type, "#4A90E2", "(30-Day Avg)"), unsafe_allow_html=True)
with col3: st.markdown(create_kpi_card("Bullish Signals", f"{pos_articles:,}", "", "", "#4A90E2", f"(Up to {latest_date_str})"), unsafe_allow_html=True)

# --- UPDATED: Live Price Fetcher with historical fallback ---
with col4:
    if not df_prices.empty and selected_ticker != "All Market":
        try:
            live_price = yf.Ticker(selected_ticker).fast_info.last_price
            st.markdown(create_kpi_card("Live Price", f"${live_price:.2f}", "Active", "bull", "#00ff9d", "(Real-Time)"), unsafe_allow_html=True)
        except:
            latest_price = df_prices.iloc[-1]['Close']
            latest_price_date = df_prices.iloc[-1]['date_only'].strftime('%b %d')
            st.markdown(create_kpi_card("Latest Close", f"${latest_price:.2f}", "", "", "#4A90E2", f"(On {latest_price_date})"), unsafe_allow_html=True)
    else:
        st.markdown(create_kpi_card("Bearish Signals", f"{neg_articles:,}", "", "", "#4A90E2", f"(Up to {latest_date_str})"), unsafe_allow_html=True)

st.markdown('<hr style="margin: 8px 0px; border-color: rgba(128,128,128,0.2);">', unsafe_allow_html=True)

# === 4. MIDDLE ROW: GRAPH (65%) & PREDICTIONS (35%) ===
col_graph, col_pred = st.columns([6.5, 3.5])

with col_graph:
    if selected_ticker == "All Market":
        st.markdown(f'<p style="color:var(--text-color); font-size:0.95rem; font-weight:600; margin-bottom:0px;">📊 Correlation Overlay</p>', unsafe_allow_html=True)
        st.info("Select a specific asset from the sidebar to view the overlay.")
        st.markdown("<p style='color: gray; font-size: 0.85rem; text-align: center; margin-top: 15px;'>⚠️ <b>Disclaimer:</b> This dashboard is for educational purposes only. AI predictions and sentiment correlations are heuristic models and may be faulty or inaccurate. We are not responsible for any financial losses or trading decisions made based on this tool.</p>", unsafe_allow_html=True)
        
    elif not filtered_news.empty and not df_prices.empty:
        logo_url = LOGO_MAP.get(selected_ticker, "")
        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <img src="{logo_url}" width="20" style="border-radius: 3px; margin-right: 8px; background-color: white; padding: 1px;" onerror="this.style.display='none'">
                <p style="color:var(--text-color); font-size:0.95rem; font-weight:600; margin:0;">{TICKER_MAP.get(selected_ticker, selected_ticker)} Correlation Overlay</p>
            </div>
        """, unsafe_allow_html=True)
        
        daily_sentiment = filtered_news.groupby('date_only')['compound'].mean().reset_index()
        merged_chart_data = pd.merge(daily_sentiment, df_prices, on='date_only', how='inner').sort_values('date_only')
        
        if not merged_chart_data.empty:
            first_price = merged_chart_data.iloc[0]['Close']
            last_price = merged_chart_data.iloc[-1]['Close']
            dynamic_price_color = '#00ff9d' if last_price >= first_price else '#ff4d4d'

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=merged_chart_data['date_only'], y=merged_chart_data['compound'], name="Sentiment", mode='lines+markers', line=dict(color='#4A90E2', width=3)), secondary_y=False)
            fig.add_trace(go.Scatter(x=merged_chart_data['date_only'], y=merged_chart_data['Close'], name="Stock Price", mode='lines+markers', line=dict(color=dynamic_price_color, width=2, dash='dot')), secondary_y=True)
            
            fig.update_layout(
                height=260, margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified"
            )
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=False, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Accumulating dates to plot...")

with col_pred:
    st.markdown('<p style="color:var(--text-color); font-size:0.95rem; font-weight:600; margin-bottom:0px;">🔮 AI Overnight Gap Prediction</p>', unsafe_allow_html=True)
    if not filtered_news.empty and not df_prices.empty and selected_ticker != "All Market":
        clean_dates = pd.to_datetime(filtered_news['datetime']).dropna().dt.date
        if clean_dates.empty:
            st.warning("No valid timestamps found.")
        else:
            latest_date = clean_dates.max()
            yesterday = latest_date - datetime.timedelta(days=1)
            
            yest_str = yesterday.strftime('%b %d')
            today_str = latest_date.strftime('%b %d')
            overnight_context = f"({yest_str} 4PM - {today_str} 9:30AM)"
            
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
                
                st.markdown(create_kpi_card("Gap Volume", f"{len(overnight_news)}", "Articles", "", "#b829ff", overnight_context), unsafe_allow_html=True)
                st.markdown(create_kpi_card("Gap Sentiment", f"{overnight_sentiment:.3f}", p_text, p_type, "#b829ff", overnight_context), unsafe_allow_html=True)
                st.markdown(create_kpi_card("Projected Open", f"${predicted_price:.2f}", f"{'+' if move_dollar > 0 else ''}${move_dollar:.2f}", p_type, "#b829ff", f"(For {today_str})"), unsafe_allow_html=True)
            else:
                st.markdown("<br><p style='color:gray; text-align:center;'>Waiting for overnight news (4 PM - 9:30 AM)...</p>", unsafe_allow_html=True)
    else:
        st.markdown("<br><p style='color:gray; text-align:center;'>Select a specific asset to activate prediction.</p>", unsafe_allow_html=True)

# === 5. BOTTOM ROW: FULL WIDTH NEWS FEED ===
st.markdown('<p style="color:var(--text-color); font-size:0.95rem; font-weight:600; margin-bottom:5px; margin-top:0px;">📰 High-Impact Intel Feed</p>', unsafe_allow_html=True)

def sentiment_color(val):
    if val == 'positive': return 'background-color: rgba(74, 144, 226, 0.1); color: #4A90E2;' 
    elif val == 'negative': return 'background-color: rgba(255, 77, 77, 0.1); color: #ff4d4d;' 
    return 'color: gray;'

display_cols = ['display_date', 'ticker', 'source', 'headline', 'sentiment_label', 'compound']
clean_display_df = filtered_news[display_cols].copy()

clean_display_df['impact'] = clean_display_df['compound'].abs()
clean_display_df = clean_display_df.sort_values(by=['impact', 'display_date'], ascending=[False, False])
clean_display_df = clean_display_df.drop(columns=['impact'])

clean_display_df.columns = ['Time', 'Asset', 'Source', 'Headline', 'Signal', 'Score']

st.dataframe(clean_display_df.style.map(sentiment_color, subset=['Signal']), use_container_width=True, hide_index=True, height=220)