"""Walk-forward evaluation against simple baselines.

For each historical day the debate sees only data up to that day (indicators
are causal), then its call is scored against the real return `horizon` trading
days later. This gives an immediate, honest accuracy estimate without waiting
for live predictions. News sentiment is not available historically, so it is
held neutral here; the report says so.
"""

import pandas as pd

from . import agents
from .indicators import add_indicators
from .tracker import score_call


def walk_forward(df: pd.DataFrame, horizon: int = 5, step: int = 1, warmup: int = 60) -> pd.DataFrame:
    ind = add_indicators(df)
    close = ind["close"].to_numpy()
    rows = []
    for i in range(warmup, len(ind) - horizon, step):
        verdict = agents.debate(ind, sentiment=0.0, at=i)
        ret = (close[i + horizon] / close[i] - 1.0) * 100.0
        rsi_v = ind["rsi14"].iloc[i]
        rsi_call = "BUY" if rsi_v < 30 else "AVOID" if rsi_v > 70 else "WAIT"
        rows.append({
            "date": ind.index[i],
            "debate_call": verdict.call,
            "confidence": verdict.confidence,
            "rsi_call": rsi_call,
            "return_pct": ret,
        })
    return pd.DataFrame(rows)


def _acc(calls: pd.Series, rets: pd.Series, only_actionable: bool):
    mask = calls != "WAIT" if only_actionable else pd.Series(True, index=calls.index)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "accuracy": None}
    hits = sum(score_call(c, r) for c, r in zip(calls[mask], rets[mask]))
    return {"n": n, "accuracy": hits / n}


def summarise(results: pd.DataFrame) -> dict:
    """Compare the debate system with always-buy and a plain RSI rule."""
    if results.empty:
        return {}
    rets = results["return_pct"]
    always_buy = pd.Series("BUY", index=results.index)
    return {
        "samples": len(results),
        "debate (BUY/AVOID calls only)": _acc(results["debate_call"], rets, True),
        "rsi rule (BUY/AVOID calls only)": _acc(results["rsi_call"], rets, True),
        "always buy": _acc(always_buy, rets, False),
        "debate share of days it acted": float((results["debate_call"] != "WAIT").mean()),
    }
