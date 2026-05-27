"""
data_ingestion/twitter.py
~~~~~~~~~~~~~~~~~~~~~~~~~
GoQuant — Fear & Greed Sentiment Engine

Fetches recent tweets, scores them with FinBERT, aggregates into a
Fear/Neutral/Greed index, generates BUY/SELL/HOLD signals, and persists
results to JSON + CSV for the dashboard and backtesting pipeline.
"""

import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import tweepy

# Allow running as a standalone script from any working directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "nlp"))
from sentiment import FinBertSentiment
from sentiment_aggregator import SentimentAggregator   # shared aggregation module

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output paths (relative to project root) ───────────────────────────────────
DASHBOARD_DIR      = Path(__file__).resolve().parent.parent / "dashboard"
INDEX_PATH         = DASHBOARD_DIR / "twitter_index.json"
HISTORY_CSV_PATH   = DASHBOARD_DIR / "twitter_sentiment_history.csv"
SIGNALS_CSV_PATH   = DASHBOARD_DIR / "twitter_signals.csv"

# ── Default query — override via CLI or env var TWITTER_QUERY ────────────────
DEFAULT_QUERY = os.getenv(
    "TWITTER_QUERY",
    "AAPL OR TSLA OR bitcoin OR crypto lang:en -is:retweet",
)


# ── API client ────────────────────────────────────────────────────────────────

def get_twitter_client() -> tweepy.Client:
    """
    Build an authenticated Tweepy client from environment variables.

    Returns:
        tweepy.Client ready for API v2 requests.

    Raises:
        ValueError: TWITTER_BEARER_TOKEN is not set.
    """
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        raise ValueError(
            "TWITTER_BEARER_TOKEN is not set. "
            "Export it as an environment variable before running."
        )
    return tweepy.Client(bearer_token=token, wait_on_rate_limit=True)


# ── Data ingestion ────────────────────────────────────────────────────────────

def stream_tweets(query: str = DEFAULT_QUERY, max_results: int = 100):
    """
    Paginate through recent tweets matching *query* and yield English ones.

    Args:
        query:       Twitter search query string (API v2 format).
        max_results: Maximum number of tweets to yield in total.

    Yields:
        dict with keys: id, text, created_at, author_id, public_metrics.
    """
    client   = get_twitter_client()
    yielded  = 0

    for tweet in tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["created_at", "author_id", "lang", "public_metrics"],
        max_results=min(max_results, 100),   # API cap per page is 100
    ).flatten(limit=max_results):
        if tweet.lang != "en":
            continue
        yield {
            "id":             tweet.id,
            "text":           tweet.text,
            "created_at":     tweet.created_at,
            "author_id":      tweet.author_id,
            "public_metrics": tweet.public_metrics,
        }
        yielded += 1

    logger.info("Streamed %d English tweet(s) for query '%s'.", yielded, query)


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signal(sentiment_row: dict, buy_threshold: float = 0.6,
                    sell_threshold: float = 0.6) -> str:
    """
    Convert a Fear/Neutral/Greed row into a trading signal.

    Args:
        sentiment_row:   Dict with float keys 'greed' and 'fear'.
        buy_threshold:   Greed score above which a BUY signal is issued.
        sell_threshold:  Fear score above which a SELL signal is issued.

    Returns:
        'BUY', 'SELL', or 'HOLD'.
    """
    greed = float(sentiment_row.get("greed", 0))
    fear  = float(sentiment_row.get("fear",  0))

    if greed > buy_threshold:
        return "BUY"
    if fear > sell_threshold:
        return "SELL"
    return "HOLD"


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_index_to_json(index: dict, path: Path = INDEX_PATH) -> None:
    """Persist the current sentiment index to *path* as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info("Sentiment index saved → %s", path)


def append_sentiment_to_csv(index: dict, path: Path = HISTORY_CSV_PATH) -> None:
    """
    Append one row (timestamp + fear/neutral/greed) to the history CSV.
    Creates the file with a header row if it does not yet exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()
    fieldnames  = ["timestamp", "fear", "neutral", "greed"]

    with open(path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fear":      index["fear"],
            "neutral":   index["neutral"],
            "greed":     index["greed"],
        })
    logger.info("Appended sentiment row → %s", path)


def generate_signals_csv(
    input_path:  Path = HISTORY_CSV_PATH,
    output_path: Path = SIGNALS_CSV_PATH,
) -> None:
    """
    Read the full history CSV, attach a BUY/SELL/HOLD signal to every row,
    and write the result to *output_path*.
    """
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Sentiment history not found at '{input_path}'. "
            "Run the main block first to generate it."
        )

    with open(input_path, newline="") as infile, \
         open(output_path, "w", newline="") as outfile:

        reader     = csv.DictReader(infile)
        fieldnames = (reader.fieldnames or []) + ["signal"]
        writer     = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row["signal"] = generate_signal(row)
            writer.writerow(row)

    logger.info("Signals CSV saved → %s", output_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer   = FinBertSentiment()
    aggregator = SentimentAggregator(window_size=20)

    for tweet in stream_tweets(max_results=20):
        sentiment = analyzer.predict(tweet["text"])
        aggregator.add_score(sentiment)
        logger.info("Tweet scored: %s → %s", tweet["text"][:60], sentiment)

    index = aggregator.get_index()
    logger.info("Final Twitter index: %s", index)
    logger.info("Signal: %s", generate_signal(index))

    save_index_to_json(index)
    append_sentiment_to_csv(index)
    generate_signals_csv()
