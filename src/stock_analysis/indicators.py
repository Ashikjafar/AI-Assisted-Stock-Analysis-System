"""Technical indicators.

Every indicator here is causal: the value on day t only uses data up to and
including day t. That property is what makes the walk-forward backtest honest.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, bounded to [0, 100]."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses at all in the window means RSI is 100 (once it is defined).
    out = out.where(~((avg_loss == 0.0) & avg_gain.notna()), 100.0)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(series: pd.Series, window: int = 20, k: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    return mid + k * std, mid, mid - k * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with indicator columns added."""
    out = df.copy()
    close = out["close"]
    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["rsi14"] = rsi(close, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(close)
    width = (out["bb_upper"] - out["bb_lower"]).replace(0.0, np.nan)
    # 0 = at the lower band, 1 = at the upper band.
    out["bb_pos"] = (close - out["bb_lower"]) / width
    if "volume" in out:
        out["vol_ratio"] = out["volume"] / out["volume"].rolling(20, min_periods=20).mean()
    else:
        out["vol_ratio"] = np.nan
    out["ret_1d"] = close.pct_change()
    return out
