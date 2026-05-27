<div align="center">

<h1>GO Quant </>

<br/>

**Ingest Twitter, Reddit & News in real time. Score with FinBERT. Generate BUY / SELL / HOLD signals.**

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/FinBERT-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## 🧠 What Is This?

**GoQuant** is a real-time trading sentiment engine that aggregates financial text from three live sources, scores it with **FinBERT** (a finance-domain BERT model), and converts the results into a **Fear / Neutral / Greed index** and actionable **BUY / SELL / HOLD** signals — all visualised in an interactive Streamlit dashboard.

| Layer | What it does |
|---|---|
| **Data Ingestion** | Streams tweets, Reddit posts, and news articles in real time |
| **NLP (FinBERT)** | Scores each text as Negative / Neutral / Positive with finance-tuned BERT |
| **Aggregation** | Rolling-window Fear/Neutral/Greed index per source |
| **Signal Engine** | Weighted multi-source composite → BUY / SELL / HOLD |
| **Dashboard** | Live Streamlit UI with charts, signal tables, and performance metrics |

---

## ✨ Features

- 📡 **3 live data sources** — Twitter/X (Tweepy v2), Reddit (PRAW), News (NewsAPI)
- 🤖 **FinBERT NLP** — finance-domain transformer, outperforms general-purpose sentiment models on market text
- 📊 **Rolling aggregation** — configurable window-size keeps the index responsive without overreacting to single posts
- ⚖️ **Weighted signal fusion** — each source carries a configurable weight; tune from the sidebar, no code edits needed
- 🎛️ **Adjustable thresholds** — BUY/SELL trigger thresholds exposed as sliders
- 📈 **Backtesting CSV output** — every signal written to CSV for strategy evaluation
- 🔐 **Secure by design** — all API keys loaded from environment variables, never hardcoded; `.env.example` provided
- 📐 **Fully portable** — no hardcoded paths; runs on Windows, macOS, and Linux

---

## 🗂️ Project Structure

```
GoQuant/
│
├── data_ingestion/
│   ├── twitter.py          # Tweet streaming, sentiment scoring, signal CSV
│   ├── reddit.py           # Reddit hot-post ingestion
│   ├── news.py             # NewsAPI article ingestion
│   └── finance.py          # Yahoo Finance OHLCV data (yfinance)
│
├── nlp/
│   ├── sentiment.py        # FinBERT wrapper (predict → negative/neutral/positive)
│   ├── sentiment_aggregator.py  # Shared rolling-window aggregator
│   └── entity_tagging.py   # Named entity recognition (ticker extraction)
│
├── aggregation/
│   └── sentiment_aggregator.py  # Legacy alias (see nlp/)
│
├── signals/
│   └── trade_signal.py     # Single + weighted multi-source signal generation
│
├── dashboard/
│   └── app.py              # Streamlit dashboard (4 tabs)
│
├── utils/
│   └── helpers.py          # Shared JSON/CSV helpers, .env loader, logging
│
├── .env.example            # API key template (copy → .env)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GoQuant Pipeline                              │
│                                                                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                        │
│  │ Twitter   │  │  Reddit   │  │   News    │  ← Live data sources   │
│  │ (Tweepy)  │  │  (PRAW)   │  │ (NewsAPI) │                        │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                        │
│        │              │              │                               │
│        └──────────────┴──────────────┘                               │
│                        │                                             │
│                        ▼                                             │
│               ┌─────────────────┐                                    │
│               │  FinBERT NLP    │  ← finance-tuned BERT              │
│               │  (HuggingFace)  │    negative/neutral/positive       │
│               └────────┬────────┘                                    │
│                        │                                             │
│                        ▼                                             │
│           ┌────────────────────────┐                                 │
│           │  Rolling Aggregator    │  ← configurable window size     │
│           │  Fear / Neutral / Greed│                                 │
│           └────────────┬───────────┘                                 │
│                        │                                             │
│                        ▼                                             │
│        ┌───────────────────────────────┐                             │
│        │  Weighted Signal Engine       │  ← per-source weights       │
│        │  BUY  /  SELL  /  HOLD        │    + configurable thresh.   │
│        └───────────────┬───────────────┘                             │
│                        │                                             │
│          ┌─────────────┴──────────────┐                              │
│          ▼                            ▼                              │
│   ┌─────────────┐           ┌──────────────────┐                    │
│   │  Dashboard  │           │  Signal CSV       │                   │
│   │  (Streamlit)│           │  (backtesting)    │                   │
│   └─────────────┘           └──────────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup

### 1 — Clone

```bash
git clone https://github.com/<your-username>/GoQuant.git
cd GoQuant
```

### 2 — Virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU inference (recommended for FinBERT speed):**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4 — Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your Twitter, Reddit, and NewsAPI credentials
```

### 5 — Run individual ingestion scripts

```bash
python data_ingestion/twitter.py   # streams tweets → saves JSON + CSV
python data_ingestion/reddit.py    # fetches Reddit posts → saves JSON
python data_ingestion/news.py      # fetches news articles → saves JSON
```

### 6 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501**.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| NLP Model | [FinBERT](https://huggingface.co/ProsusAI/finbert) via HuggingFace Transformers |
| Twitter ingestion | [Tweepy](https://www.tweepy.org/) (API v2) |
| Reddit ingestion | [PRAW](https://praw.readthedocs.io/) |
| News ingestion | [NewsAPI Python](https://newsapi.org/docs/client-libraries/python) |
| Price data | [yfinance](https://github.com/ranaroussi/yfinance) |
| Dashboard | [Streamlit](https://streamlit.io) |
| Data processing | Pandas, NumPy |
| Deep learning | PyTorch |
| Language | Python 3.10+ |

---

## 🔐 API Keys Required

| Service | Where to get it |
|---|---|
| Twitter/X | [developer.twitter.com](https://developer.twitter.com) |
| Reddit | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| NewsAPI | [newsapi.org](https://newsapi.org) |

Yahoo Finance (yfinance) requires no API key.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with 📈 and FinBERT
</div>
