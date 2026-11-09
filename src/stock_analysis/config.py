"""Central configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "predictions.db"

# Candidate watchlist (NSE tickers, Yahoo Finance format).
# IMPORTANT: Shariah compliance changes over time. Treat this list as a
# starting point and verify every symbol with a screener such as Zoya
# before acting on it.
WATCHLIST = [
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "HINDUNILVR.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
]

# Prediction horizons in calendar days. Each call is scored at each horizon.
HORIZONS = (1, 7, 14)

# A WAIT call counts as correct if the price moved less than this (percent).
WAIT_BAND_PCT = 1.0

# Judge thresholds.
MIN_NET_SCORE = 1.5      # |bull - bear| needed to take a side
MIN_CONFIDENCE = 0.30    # |bull - bear| / (bull + bear) needed to take a side

DISCLAIMER = (
    "Educational project. Output is decision support, not financial advice. "
    "Markets are hard to predict; always check the tracked accuracy."
)
