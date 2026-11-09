"""Price and news data access.

Live data comes from yfinance (optional dependency). A deterministic synthetic
generator is provided so the whole system can be demonstrated and tested
offline, e.g. during a college presentation with no internet.
"""

import hashlib
from typing import List

import numpy as np
import pandas as pd


class DataError(RuntimeError):
    """Raised when market data cannot be loaded."""


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        raise DataError(f"Missing columns, got {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"]).sort_index()


def synthetic_prices(symbol: str, days: int = 500, seed: int = 0) -> pd.DataFrame:
    """Deterministic fake OHLCV (regime-switching random walk) for demos/tests."""
    digest = hashlib.sha256(f"{symbol}:{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    drift = rng.choice([-0.0006, 0.0002, 0.0008], size=days // 60 + 1)
    mu = np.repeat(drift, 60)[:days]
    rets = rng.normal(mu, 0.012)
    close = 1000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, days)))
    volume = rng.integers(500_000, 2_000_000, days) * (1 + 3 * np.abs(rets))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def load_prices(symbol: str, period: str = "2y", demo: bool = False) -> pd.DataFrame:
    if demo:
        return synthetic_prices(symbol)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError(
            "yfinance is not installed. Run `pip install -r requirements.txt` "
            "or use --demo for offline synthetic data."
        ) from exc
    try:
        raw = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    except Exception as exc:  # network errors vary by library version
        raise DataError(f"Download failed for {symbol}: {exc}") from exc
    if raw is None or raw.empty:
        raise DataError(f"No price data returned for {symbol}")
    return _normalise(raw)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalise(df)


DEMO_HEADLINES = [
    "Company wins large order, analysts upgrade outlook",
    "Quarterly profit beats estimates on strong growth",
    "Regulator probe raises concerns for sector",
    "Shares rally as management announces buyback",
]


def fetch_headlines(symbol: str, demo: bool = False, limit: int = 15) -> List[str]:
    """Recent headlines for a symbol. Returns [] if unavailable (never raises)."""
    if demo:
        return list(DEMO_HEADLINES)
    try:
        import yfinance as yf

        items = yf.Ticker(symbol).news or []
    except Exception:
        return []
    titles = []
    for item in items[:limit]:
        title = item.get("title") or (item.get("content") or {}).get("title")
        if title:
            titles.append(title)
    return titles


def close_on_or_after(symbol: str, date: pd.Timestamp, demo: bool = False):
    """First available close on or after `date`, or None if not yet available."""
    df = load_prices(symbol, period="3mo", demo=demo)
    later = df.loc[df.index >= pd.Timestamp(date).normalize(), "close"]
    return float(later.iloc[0]) if len(later) else None
