"""
signals/trade_signal.py
~~~~~~~~~~~~~~~~~~~~~~~
GoQuant — Fear & Greed Sentiment Engine

Converts aggregated Fear/Neutral/Greed indices into BUY / SELL / HOLD
trading signals. Supports multi-source weighted scoring so Twitter,
Reddit, and News can carry different confidence weights.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Signal = Literal["BUY", "SELL", "HOLD"]

# ── Default thresholds — tweak via env or pass explicitly ─────────────────────
DEFAULT_BUY_THRESHOLD  = 0.60
DEFAULT_SELL_THRESHOLD = 0.60

# ── Default source weights (must sum to 1.0) ──────────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "twitter": 0.40,
    "reddit":  0.35,
    "news":    0.25,
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Full output of a signal generation call."""
    signal:         Signal
    composite_fear:    float
    composite_neutral: float
    composite_greed:   float
    source_weights: dict[str, float] = field(default_factory=dict)
    reasoning:      str = ""

    def __str__(self) -> str:
        return (
            f"Signal: {self.signal} | "
            f"Fear={self.composite_fear:.3f}  "
            f"Neutral={self.composite_neutral:.3f}  "
            f"Greed={self.composite_greed:.3f} | "
            f"{self.reasoning}"
        )


# ── Core signal logic ─────────────────────────────────────────────────────────

def generate_signal(
    sentiment_index:  dict,
    buy_threshold:    float = DEFAULT_BUY_THRESHOLD,
    sell_threshold:   float = DEFAULT_SELL_THRESHOLD,
) -> Signal:
    """
    Derive a trading signal from a single Fear/Neutral/Greed index.

    Args:
        sentiment_index: Dict with float keys 'fear', 'neutral', 'greed'.
        buy_threshold:   Greed score that triggers a BUY.
        sell_threshold:  Fear score that triggers a SELL.

    Returns:
        'BUY', 'SELL', or 'HOLD'.
    """
    greed = float(sentiment_index.get("greed", 0))
    fear  = float(sentiment_index.get("fear",  0))

    if greed > buy_threshold:
        return "BUY"
    if fear > sell_threshold:
        return "SELL"
    return "HOLD"


def generate_weighted_signal(
    indices:       dict[str, dict],
    weights:       dict[str, float] | None = None,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
) -> SignalResult:
    """
    Combine multiple source indices into one weighted composite signal.

    Args:
        indices:  Mapping of source name → sentiment index dict.
                  e.g. {"twitter": {...}, "reddit": {...}, "news": {...}}
        weights:  Optional per-source weights (must sum to 1.0).
                  Defaults to DEFAULT_WEIGHTS. Unknown sources receive
                  equal share of remaining weight.
        buy_threshold:   Composite greed score that triggers a BUY.
        sell_threshold:  Composite fear score that triggers a SELL.

    Returns:
        SignalResult with composite scores, final signal, and reasoning.

    Raises:
        ValueError: *indices* is empty.
    """
    if not indices:
        raise ValueError("At least one sentiment index is required.")

    w = weights or DEFAULT_WEIGHTS

    # Normalise weights to only the sources that were actually provided
    active_sources = {k: v for k, v in w.items() if k in indices}
    if not active_sources:
        # Fallback: equal weighting for all provided sources
        active_sources = {k: 1 / len(indices) for k in indices}
    else:
        total = sum(active_sources.values())
        active_sources = {k: v / total for k, v in active_sources.items()}

    composite_fear    = 0.0
    composite_neutral = 0.0
    composite_greed   = 0.0

    for source, idx in indices.items():
        w_src = active_sources.get(source, 0.0)
        composite_fear    += float(idx.get("fear",    0)) * w_src
        composite_neutral += float(idx.get("neutral", 0)) * w_src
        composite_greed   += float(idx.get("greed",   0)) * w_src

    composite = {
        "fear":    composite_fear,
        "neutral": composite_neutral,
        "greed":   composite_greed,
    }
    signal = generate_signal(composite, buy_threshold, sell_threshold)

    reasoning = (
        f"Composite greed={composite_greed:.3f} "
        f"(threshold={buy_threshold}), "
        f"fear={composite_fear:.3f} "
        f"(threshold={sell_threshold})"
    )
    logger.info("Signal generated: %s | %s", signal, reasoning)

    return SignalResult(
        signal=signal,
        composite_fear=composite_fear,
        composite_neutral=composite_neutral,
        composite_greed=composite_greed,
        source_weights=active_sources,
        reasoning=reasoning,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke-test with dummy indices
    sample_indices = {
        "twitter": {"fear": 0.10, "neutral": 0.25, "greed": 0.65},
        "reddit":  {"fear": 0.20, "neutral": 0.40, "greed": 0.40},
        "news":    {"fear": 0.15, "neutral": 0.50, "greed": 0.35},
    }
    result = generate_weighted_signal(sample_indices)
    print(result)
