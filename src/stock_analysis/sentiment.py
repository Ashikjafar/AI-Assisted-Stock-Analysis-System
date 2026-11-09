"""Lightweight, explainable news-headline sentiment (no model download needed).

A word-list scorer is deliberately simple: every score can be traced back to
the exact words that caused it, which suits a debate system that must explain
itself. It can be swapped for FinBERT later without changing the interface.
"""

import re
from typing import Iterable, List, Tuple

POSITIVE = {
    "surge", "surges", "soar", "soars", "rally", "rallies", "gain", "gains",
    "rise", "rises", "jump", "jumps", "beat", "beats", "record", "growth",
    "profit", "profits", "upgrade", "upgrades", "outperform", "strong",
    "bullish", "boost", "boosts", "win", "wins", "order", "orders", "expands",
    "expansion", "dividend", "buyback", "recovery", "optimistic", "improves",
    "positive", "high", "approval", "approved",
}
NEGATIVE = {
    "fall", "falls", "drop", "drops", "slump", "slumps", "plunge", "plunges",
    "loss", "losses", "miss", "misses", "weak", "downgrade", "downgrades",
    "underperform", "bearish", "cut", "cuts", "fraud", "probe", "lawsuit",
    "penalty", "fine", "decline", "declines", "layoff", "layoffs", "warning",
    "risk", "risks", "slowdown", "default", "recall", "negative", "low",
    "concern", "concerns", "resigns", "sell-off", "selloff",
}
NEGATIONS = {"not", "no", "never", "without", "fails", "failed", "unlikely"}

_TOKEN = re.compile(r"[a-z][a-z\-']*")


def score_headline(text: str) -> Tuple[float, List[str]]:
    """Score one headline in [-1, 1] and return the words that drove it."""
    tokens = _TOKEN.findall(text.lower())
    total, drivers = 0, []
    for i, tok in enumerate(tokens):
        sign = 1 if tok in POSITIVE else -1 if tok in NEGATIVE else 0
        if sign == 0:
            continue
        # A negation flips sentiment words up to three tokens after it.
        if any(t in NEGATIONS for t in tokens[max(0, i - 3):i]):
            sign = -sign
        total += sign
        drivers.append(("+" if sign > 0 else "-") + tok)
    if total == 0:
        return 0.0, drivers
    # Squash so a headline with many hits does not dominate the average.
    return max(-1.0, min(1.0, total / 2.0)), drivers


def aggregate(headlines: Iterable[str]) -> dict:
    """Average sentiment over headlines. Empty input gives a neutral result."""
    scored = [(h, *score_headline(h)) for h in headlines if h and h.strip()]
    if not scored:
        return {"score": 0.0, "count": 0, "positive": 0, "negative": 0, "details": []}
    scores = [s for _, s, _ in scored]
    return {
        "score": sum(scores) / len(scores),
        "count": len(scores),
        "positive": sum(1 for s in scores if s > 0),
        "negative": sum(1 for s in scores if s < 0),
        "details": [{"headline": h, "score": s, "drivers": d} for h, s, d in scored],
    }
