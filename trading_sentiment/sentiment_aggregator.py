"""
nlp/sentiment_aggregator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
GoQuant — Fear & Greed Sentiment Engine

Shared rolling-window aggregator used by all data ingestion modules.

The original codebase duplicated this class inside twitter.py, reddit.py,
and news.py. Extracting it here eliminates the duplication and ensures
all sources use identical aggregation logic.
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Aggregate per-text FinBERT sentiment scores into a rolling
    Fear / Neutral / Greed index.

    Uses a fixed-size deque so old scores fall off automatically —
    no manual ``pop(0)`` needed.

    Args:
        window_size: Maximum number of scores to keep in the rolling window.

    Example::

        aggregator = SentimentAggregator(window_size=20)
        for text_score in scores:
            aggregator.add_score(text_score)
        index = aggregator.get_index()
        # → {"fear": 0.12, "neutral": 0.45, "greed": 0.43}
    """

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self._scores: deque[dict] = deque(maxlen=window_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_score(self, sentiment: dict) -> None:
        """
        Add one FinBERT prediction to the rolling window.

        Args:
            sentiment: Dict with float keys 'negative', 'neutral', 'positive'
                       as returned by ``FinBertSentiment.predict()``.
        """
        self._scores.append(sentiment)

    def get_index(self) -> dict[str, float]:
        """
        Compute the current Fear/Neutral/Greed index from the rolling window.

        Returns:
            Dict with keys 'fear', 'neutral', 'greed' (floats 0–1 that sum to 1).
            Returns all-zero dict if no scores have been added.
        """
        if not self._scores:
            logger.debug("Aggregator is empty — returning zero index.")
            return {"fear": 0.0, "neutral": 0.0, "greed": 0.0}

        arr = np.array(
            [[s["negative"], s["neutral"], s["positive"]] for s in self._scores],
            dtype=float,
        )
        avg = arr.mean(axis=0)
        return {
            "fear":    float(avg[0]),
            "neutral": float(avg[1]),
            "greed":   float(avg[2]),
        }

    def reset(self) -> None:
        """Clear all scores from the rolling window."""
        self._scores.clear()
        logger.debug("Aggregator reset.")

    def __len__(self) -> int:
        return len(self._scores)

    def __repr__(self) -> str:
        return (
            f"SentimentAggregator("
            f"window_size={self.window_size}, "
            f"current_size={len(self._scores)})"
        )
