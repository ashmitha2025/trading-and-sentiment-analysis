"""
data_ingestion/reddit.py
~~~~~~~~~~~~~~~~~~~~~~~~
GoQuant — Fear & Greed Sentiment Engine

Fetches hot posts from a subreddit, scores them with FinBERT, aggregates
into a Fear/Neutral/Greed index, and saves the result for the dashboard.
"""

import json
import logging
import os
import sys
from pathlib import Path

import praw

# Allow running as a standalone script from any working directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "nlp"))
from sentiment import FinBertSentiment
from sentiment_aggregator import SentimentAggregator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output path ───────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
INDEX_PATH    = DASHBOARD_DIR / "reddit_index.json"

# ── Default subreddit — override via env var REDDIT_SUBREDDIT ────────────────
DEFAULT_SUBREDDIT = os.getenv("REDDIT_SUBREDDIT", "stocks")


# ── API client ────────────────────────────────────────────────────────────────

def get_reddit_client() -> praw.Reddit:
    """
    Build a read-only PRAW Reddit client from environment variables.

    Required env vars:
        REDDIT_CLIENT_ID      — app client ID
        REDDIT_CLIENT_SECRET  — app client secret
        REDDIT_USER_AGENT     — user-agent string (optional, has a default)

    Returns:
        praw.Reddit instance configured for read-only access.
    """
    return praw.Reddit(
        client_id     = os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent    = os.getenv("REDDIT_USER_AGENT", "GoQuantBot/0.1"),
    )


# ── Data ingestion ────────────────────────────────────────────────────────────

def fetch_reddit_posts(subreddit: str = DEFAULT_SUBREDDIT, limit: int = 20):
    """
    Yield hot posts from *subreddit*.

    Args:
        subreddit: Subreddit name (without the r/ prefix).
        limit:     Maximum number of posts to fetch.

    Yields:
        dict with keys: id, title, selftext, created_utc,
        score, num_comments, url.
    """
    reddit  = get_reddit_client()
    fetched = 0

    for submission in reddit.subreddit(subreddit).hot(limit=limit):
        yield {
            "id":           submission.id,
            "title":        submission.title,
            "selftext":     submission.selftext or "",
            "created_utc":  submission.created_utc,
            "score":        submission.score,
            "num_comments": submission.num_comments,
            "url":          submission.url,
        }
        fetched += 1

    logger.info("Fetched %d post(s) from r/%s.", fetched, subreddit)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index_to_json(index: dict, path: Path = INDEX_PATH) -> None:
    """Persist the current sentiment index to *path* as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info("Reddit index saved → %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer   = FinBertSentiment()
    aggregator = SentimentAggregator(window_size=20)

    for post in fetch_reddit_posts(limit=20):
        text      = post["title"] + "\n" + post["selftext"]
        sentiment = analyzer.predict(text)
        post["sentiment"] = sentiment
        aggregator.add_score(sentiment)
        logger.info("Post scored: '%s…' → %s", post["title"][:50], sentiment)

    index = aggregator.get_index()
    logger.info("Final Reddit index: %s", index)

    save_index_to_json(index)
