"""
data_ingestion/news.py
~~~~~~~~~~~~~~~~~~~~~~
GoQuant — Fear & Greed Sentiment Engine

Fetches news articles from NewsAPI, scores them with FinBERT, aggregates
into a Fear/Neutral/Greed index, and saves the result for the dashboard.
"""

import json
import logging
import os
import sys
from pathlib import Path

from newsapi import NewsApiClient

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
INDEX_PATH    = DASHBOARD_DIR / "news_index.json"

# ── Defaults — override via environment variables ─────────────────────────────
DEFAULT_QUERY     = os.getenv("NEWS_QUERY",    "bitcoin OR crypto OR stocks")
DEFAULT_LANGUAGE  = os.getenv("NEWS_LANGUAGE", "en")
DEFAULT_PAGE_SIZE = int(os.getenv("NEWS_PAGE_SIZE", "10"))


# ── API client ────────────────────────────────────────────────────────────────

def get_news_client() -> NewsApiClient:
    """
    Build a NewsAPI client from environment variables.

    Returns:
        NewsApiClient ready for requests.

    Raises:
        ValueError: NEWSAPI_KEY is not set.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError(
            "NEWSAPI_KEY is not set. "
            "Export it as an environment variable before running."
        )
    return NewsApiClient(api_key=api_key)


# ── Data ingestion ────────────────────────────────────────────────────────────

def fetch_news(
    query:     str = DEFAULT_QUERY,
    language:  str = DEFAULT_LANGUAGE,
    page_size: int = DEFAULT_PAGE_SIZE,
):
    """
    Yield recent news articles matching *query*.

    Args:
        query:     NewsAPI search query.
        language:  Two-letter language code (default: 'en').
        page_size: Number of articles to retrieve (max 100 on free tier).

    Yields:
        dict with keys: title, description, publishedAt, source, url.
    """
    client   = get_news_client()
    response = client.get_everything(q=query, language=language, page_size=page_size)
    articles = response.get("articles", [])

    for article in articles:
        yield {
            "title":       article.get("title")       or "",
            "description": article.get("description") or "",
            "publishedAt": article.get("publishedAt") or "",
            "source":      article.get("source", {}).get("name", "unknown"),
            "url":         article.get("url")         or "",
        }

    logger.info("Fetched %d article(s) for query '%s'.", len(articles), query)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index_to_json(index: dict, path: Path = INDEX_PATH) -> None:
    """Persist the current sentiment index to *path* as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info("News index saved → %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer   = FinBertSentiment()
    aggregator = SentimentAggregator(window_size=20)

    for article in fetch_news():
        text      = article["title"] + "\n" + article["description"]
        sentiment = analyzer.predict(text)
        article["sentiment"] = sentiment
        aggregator.add_score(sentiment)
        logger.info("Article scored: '%s…' → %s", article["title"][:50], sentiment)

    index = aggregator.get_index()
    logger.info("Final News index: %s", index)

    save_index_to_json(index)
