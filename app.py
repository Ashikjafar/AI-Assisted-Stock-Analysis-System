"""Streamlit dashboard.   Run:  streamlit run app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st

from stock_analysis import agents, backtest, config, data, sentiment, tracker
from stock_analysis.indicators import add_indicators

st.set_page_config(page_title="AI-Assisted Stock Analysis", layout="wide")
st.title("AI-Assisted Stock Analysis System")
st.caption(config.DISCLAIMER)

with st.sidebar:
    demo = st.toggle("Offline demo data", value=False,
                     help="Synthetic prices and headlines, no internet needed")
    symbol = st.selectbox("Stock", config.WATCHLIST)
    custom = st.text_input("...or type a symbol (e.g. RELIANCE.NS)")
    symbol = custom.strip().upper() or symbol
    log_it = st.checkbox("Log this call for accuracy tracking", value=False)

tab_analyze, tab_track, tab_backtest = st.tabs(["Analyze", "Live accuracy", "Backtest"])

try:
    prices = data.load_prices(symbol, demo=demo)
except data.DataError as exc:
    st.error(str(exc))
    st.stop()

ind = add_indicators(prices)
headlines = data.fetch_headlines(symbol, demo=demo)
sent = sentiment.aggregate(headlines)
verdict = agents.debate(ind, sent["score"])

with tab_analyze:
    price = float(ind["close"].iloc[-1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Last close", f"{price:,.2f}")
    c2.metric("Judge call", verdict.call)
    c3.metric("Confidence", f"{verdict.confidence:.0%}")
    st.info(verdict.reasoning)

    st.subheader("Price and moving averages")
    st.line_chart(ind[["close", "sma20", "sma50"]].tail(180))
    st.subheader("RSI (14)")
    st.line_chart(ind[["rsi14"]].tail(180))

    bull_col, bear_col = st.columns(2)
    for col, arg, title in ((bull_col, verdict.bull, "Bull case"),
                            (bear_col, verdict.bear, "Bear case")):
        with col:
            st.subheader(f"{title} ({arg.score:.1f})")
            if not arg.points:
                st.write("Nothing convincing.")
            for text, w in arg.points:
                st.write(f"- {text} **[{w:+.1f}]**")

    st.subheader(f"News sentiment: {sent['score']:+.2f} ({sent['count']} headlines)")
    if sent["details"]:
        st.dataframe(pd.DataFrame(sent["details"]), use_container_width=True)

    if log_it and not demo:
        conn = tracker.connect()
        n = tracker.log_prediction(conn, symbol, verdict, sent["score"], price)
        st.success(f"Logged {n} prediction(s).")

with tab_track:
    conn = tracker.connect()
    if st.button("Score predictions that are due"):
        done = tracker.evaluate_due(
            conn, lambda s, d: data.close_on_or_after(s, pd.Timestamp(d)))
        st.write(f"Scored {done} prediction(s).")
    summ = tracker.accuracy_summary(conn)
    o = summ["overall"]
    st.metric("Overall accuracy",
              "n/a" if o["accuracy"] is None else f"{o['accuracy']:.0%}",
              help=f"{o['n']} scored, {summ['pending']} pending")
    rows = [{"group": f"call: {c}", **v} for c, v in summ["by_call"].items()]
    rows += [{"group": f"horizon: {h}d", **v} for h, v in summ["by_horizon"].items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    history = pd.read_sql_query(
        "SELECT created_at, symbol, call, confidence, price, horizon_days,"
        " return_pct, correct FROM predictions ORDER BY id DESC LIMIT 200", conn)
    st.dataframe(history, use_container_width=True)

with tab_backtest:
    horizon = st.slider("Horizon (trading days)", 1, 20, 5)
    st.caption("Walk-forward: each call only sees data up to that day. "
               "News sentiment is neutral in this test.")
    res = backtest.walk_forward(prices, horizon=horizon)
    summ = backtest.summarise(res)
    table = [{"strategy": k, "n": v["n"], "accuracy": v["accuracy"]}
             for k, v in summ.items() if isinstance(v, dict)]
    st.dataframe(pd.DataFrame(table), use_container_width=True)
