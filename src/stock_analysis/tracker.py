"""Prediction log and accuracy tracker (SQLite).

Every call is stored with the price at the time. Once a call's horizon has
passed, `evaluate_due` fills in the real outcome and marks it correct or not.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    call TEXT NOT NULL,
    confidence REAL NOT NULL,
    bull_score REAL NOT NULL,
    bear_score REAL NOT NULL,
    sentiment REAL NOT NULL,
    price REAL NOT NULL,
    horizon_days INTEGER NOT NULL,
    due_date TEXT NOT NULL,
    outcome_price REAL,
    return_pct REAL,
    correct INTEGER,
    evaluated_at TEXT
);
"""


def score_call(call: str, return_pct: float) -> bool:
    """BUY wins if price rose, AVOID wins if it fell, WAIT wins if it stayed flat."""
    if call == "BUY":
        return return_pct > 0
    if call == "AVOID":
        return return_pct < 0
    return abs(return_pct) < config.WAIT_BAND_PCT


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_prediction(conn, symbol, verdict, sentiment, price,
                   horizons=config.HORIZONS, now: Optional[datetime] = None) -> int:
    """Store one row per horizon. Returns number of rows written."""
    now = now or datetime.now()
    for h in horizons:
        due = (now + timedelta(days=h)).date().isoformat()
        conn.execute(
            "INSERT INTO predictions (created_at, symbol, call, confidence, bull_score,"
            " bear_score, sentiment, price, horizon_days, due_date)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), symbol, verdict.call, verdict.confidence,
             verdict.bull.score, verdict.bear.score, sentiment, float(price), h, due),
        )
    conn.commit()
    return len(horizons)


def evaluate_due(conn, price_fn: Callable[[str, str], Optional[float]],
                 today: Optional[datetime] = None) -> int:
    """Score every pending prediction whose due date has arrived.

    `price_fn(symbol, due_date_iso)` returns the close on/after that date, or
    None if it is not available yet (those rows stay pending).
    """
    today = today or datetime.now()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE correct IS NULL AND due_date <= ?",
        (today.date().isoformat(),),
    ).fetchall()
    done = 0
    for r in rows:
        outcome = price_fn(r["symbol"], r["due_date"])
        if outcome is None:
            continue
        ret = (outcome / r["price"] - 1.0) * 100.0
        conn.execute(
            "UPDATE predictions SET outcome_price=?, return_pct=?, correct=?, evaluated_at=?"
            " WHERE id=?",
            (float(outcome), ret, int(score_call(r["call"], ret)),
             today.isoformat(timespec="seconds"), r["id"]),
        )
        done += 1
    conn.commit()
    return done


def accuracy_summary(conn) -> dict:
    """Overall, per-call and per-horizon accuracy over evaluated predictions."""
    def stats(where: str = "", args=()):
        row = conn.execute(
            f"SELECT COUNT(*) n, COALESCE(SUM(correct),0) hits, AVG(return_pct) avg_ret"
            f" FROM predictions WHERE correct IS NOT NULL {where}", args).fetchone()
        n = row["n"]
        return {"n": n, "hits": row["hits"],
                "accuracy": (row["hits"] / n) if n else None,
                "avg_return_pct": row["avg_ret"]}

    pending = conn.execute(
        "SELECT COUNT(*) c FROM predictions WHERE correct IS NULL").fetchone()["c"]
    return {
        "overall": stats(),
        "pending": pending,
        "by_call": {c: stats("AND call=?", (c,)) for c in ("BUY", "AVOID", "WAIT")},
        "by_horizon": {h: stats("AND horizon_days=?", (h,)) for h in config.HORIZONS},
    }
