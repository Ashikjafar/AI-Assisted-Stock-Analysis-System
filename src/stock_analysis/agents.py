"""Bull / Bear / Judge debate.

Bull and Bear each build the strongest case for their side from the same
evidence. The Judge weighs both, and only takes a side when the gap is large
and consistent enough; otherwise the call is WAIT. Everything is rule-based so
each decision is fully explainable.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from . import config


@dataclass
class Argument:
    side: str
    points: List[Tuple[str, float]] = field(default_factory=list)

    def add(self, text: str, weight: float) -> None:
        self.points.append((text, weight))

    @property
    def score(self) -> float:
        return sum(w for _, w in self.points)


@dataclass
class Verdict:
    call: str            # BUY | AVOID | WAIT
    confidence: float    # 0..1
    bull: Argument
    bear: Argument
    reasoning: str


def _ok(*values) -> bool:
    return all(pd.notna(v) for v in values)


def bull_case(row: pd.Series, prev: pd.Series, sentiment: float) -> Argument:
    a = Argument("bull")
    if _ok(row.close, row.sma50) and row.close > row.sma50:
        a.add("Price is above the 50-day average (uptrend)", 2.0)
    if _ok(row.sma20, row.sma50) and row.sma20 > row.sma50:
        a.add("20-day average is above the 50-day average", 1.5)
    if _ok(row.macd_hist, prev.macd_hist) and row.macd_hist > 0 and row.macd_hist > prev.macd_hist:
        a.add("MACD histogram is positive and rising (momentum building)", 1.5)
    if _ok(row.rsi14):
        if row.rsi14 < 35:
            a.add(f"RSI {row.rsi14:.0f} is oversold, bounce is plausible", 1.5)
        elif 50 <= row.rsi14 <= 65:
            a.add(f"RSI {row.rsi14:.0f} shows healthy bullish momentum", 1.0)
    if _ok(row.bb_pos) and row.bb_pos < 0.2:
        a.add("Price is near the lower Bollinger band", 1.0)
    if _ok(row.vol_ratio, row.ret_1d) and row.vol_ratio > 1.5 and row.ret_1d > 0:
        a.add("Volume surge on an up day (buyers active)", 1.0)
    if sentiment > 0.15:
        a.add(f"News sentiment is positive ({sentiment:+.2f})", 1.5)
    return a


def bear_case(row: pd.Series, prev: pd.Series, sentiment: float) -> Argument:
    a = Argument("bear")
    if _ok(row.close, row.sma50) and row.close < row.sma50:
        a.add("Price is below the 50-day average (downtrend)", 2.0)
    if _ok(row.sma20, row.sma50) and row.sma20 < row.sma50:
        a.add("20-day average is below the 50-day average", 1.5)
    if _ok(row.macd_hist, prev.macd_hist) and row.macd_hist < 0 and row.macd_hist < prev.macd_hist:
        a.add("MACD histogram is negative and falling (momentum fading)", 1.5)
    if _ok(row.rsi14) and row.rsi14 > 70:
        a.add(f"RSI {row.rsi14:.0f} is overbought, pullback is plausible", 2.0)
    if _ok(row.bb_pos) and row.bb_pos > 0.8:
        a.add("Price is near the upper Bollinger band", 1.0)
    if _ok(row.vol_ratio, row.ret_1d) and row.vol_ratio > 1.5 and row.ret_1d < 0:
        a.add("Volume surge on a down day (sellers active)", 1.0)
    if sentiment < -0.15:
        a.add(f"News sentiment is negative ({sentiment:+.2f})", 1.5)
    return a


def judge(bull: Argument, bear: Argument) -> Verdict:
    net = bull.score - bear.score
    total = bull.score + bear.score
    confidence = abs(net) / total if total > 0 else 0.0
    if total == 0:
        call, why = "WAIT", "No usable evidence on either side."
    elif net >= config.MIN_NET_SCORE and confidence >= config.MIN_CONFIDENCE:
        call, why = "BUY", f"Bull case outweighs bear by {net:.1f} points."
    elif net <= -config.MIN_NET_SCORE and confidence >= config.MIN_CONFIDENCE:
        call, why = "AVOID", f"Bear case outweighs bull by {-net:.1f} points."
    else:
        call = "WAIT"
        why = (f"Evidence is mixed (bull {bull.score:.1f} vs bear {bear.score:.1f}); "
               "the gap is too small to act on.")
    return Verdict(call, round(confidence, 3), bull, bear, why)


def debate(df: pd.DataFrame, sentiment: float = 0.0, at: int = -1) -> Verdict:
    """Run the debate on row `at` of an indicator frame (default: latest)."""
    if len(df) < 2:
        raise ValueError("Need at least two rows of data")
    row = df.iloc[at]
    prev = df.iloc[at - 1]
    return judge(bull_case(row, prev, sentiment), bear_case(row, prev, sentiment))
