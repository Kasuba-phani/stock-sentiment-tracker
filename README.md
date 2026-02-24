# 📈 Executive AI Sentiment Terminal (Automated MLOps Pipeline)

**Live Dashboard:** [https://stock-sentiment-tracker-bp9fsi9nsrbnww6li8kaze.streamlit.app/]

## 🎯 Project Overview
An automated, cloud-hosted Machine Learning pipeline that tracks financial news, performs Natural Language Processing (NLP) sentiment analysis, and correlates breaking news with stock market movements in real-time. 

Instead of relying on static, historical datasets, this project operates as a continuous CI/CD pipeline, scraping live data, processing it through an AI model, and updating a public-facing executive dashboard completely autonomously.

## ⚙️ The Architecture (How it Works)
1. **Automated Ingestion (GitHub Actions):** A Python-based web scraper wakes up every hour via a Cron job to pull the latest financial RSS feeds.
2. **AI Processing (NLP):** Headlines are cleaned and fed through a FinBERT sentiment analysis model to extract compound polarity scores.
3. **Data Partitioning:** Processed data is partitioned by date, deduplicated, and pushed automatically to the warehouse.
4. **Live UI (Streamlit):** The dashboard continuously listens for repository updates and dynamically stitches historical CSVs together alongside 30-day trailing stock prices (via `yfinance`).
5. **Heuristic Forecasting:** An overnight gap prediction engine isolates off-hours news (4:00 PM - 9:30 AM) to forecast the next day's opening price movement based on sentiment and historical volatility.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Data Engineering:** Pandas, Glob, yfinance
* **Machine Learning / NLP:** Transformers (HuggingFace), FinBERT, Scikit-Learn
* **Automation / MLOps:** GitHub Actions (CI/CD)
* **Frontend / Visualization:** Streamlit, Plotly (Interactive Dual-Axis Charts)

## 🚀 Future Roadmap (Active Development)
This pipeline is currently in active development. Immediate next steps include:
* **Time Series Forecasting:** Transitioning from the current heuristic math model to a rigorous **ARIMAX** or **Facebook Prophet** time-series model.
* **Exogenous Variables:** Using the accumulated daily sentiment scores as an exogenous variable to train the ARIMAX model for highly accurate, multi-day price forecasting.

---
*Developed by Phanidhar Kasuba*
