"""Command line interface.

    python -m stock_analysis analyze TCS.NS INFY.NS
    python -m stock_analysis analyze --demo TCS.NS
    python -m stock_analysis evaluate
    python -m stock_analysis backtest --demo TCS.NS
    python -m stock_analysis report
"""

import argparse
import sys

from . import agents, backtest, config, data, sentiment, tracker
from .indicators import add_indicators


def _print_verdict(symbol, price, sent, v):
    print(f"\n=== {symbol}  (close {price:.2f}) ===")
    print(f"Sentiment: {sent['score']:+.2f} from {sent['count']} headline(s)")
    for arg in (v.bull, v.bear):
        print(f"\n{arg.side.upper()} case (score {arg.score:.1f}):")
        if not arg.points:
            print("  - nothing convincing")
        for text, w in arg.points:
            print(f"  - {text} [{w:+.1f}]")
    print(f"\nJUDGE: {v.call}  (confidence {v.confidence:.0%})")
    print(f"  {v.reasoning}")


def cmd_analyze(args):
    conn = None if (args.no_log or args.demo) else tracker.connect()
    status = 0
    for symbol in args.symbols or config.WATCHLIST:
        try:
            prices = data.load_prices(symbol, demo=args.demo)
        except data.DataError as exc:
            print(f"{symbol}: {exc}", file=sys.stderr)
            status = 1
            continue
        sent = sentiment.aggregate(data.fetch_headlines(symbol, demo=args.demo))
        ind = add_indicators(prices)
        verdict = agents.debate(ind, sent["score"])
        price = float(ind["close"].iloc[-1])
        _print_verdict(symbol, price, sent, verdict)
        if conn is not None:
            n = tracker.log_prediction(conn, symbol, verdict, sent["score"], price)
            print(f"  logged {n} prediction(s) for later scoring")
    print(f"\n{config.DISCLAIMER}")
    return status


def cmd_evaluate(args):
    conn = tracker.connect()

    def price_fn(symbol, due):
        import pandas as pd
        return data.close_on_or_after(symbol, pd.Timestamp(due))

    try:
        n = tracker.evaluate_due(conn, price_fn)
    except data.DataError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Evaluated {n} prediction(s).")
    return cmd_report(args)


def cmd_report(args):
    conn = tracker.connect()
    s = tracker.accuracy_summary(conn)

    def line(label, st):
        acc = "n/a" if st["accuracy"] is None else f"{st['accuracy']:.0%}"
        avg = "n/a" if st["avg_return_pct"] is None else f"{st['avg_return_pct']:+.2f}%"
        return f"  {label:<10} n={st['n']:<4} accuracy={acc:<5} avg return={avg}"

    print("Live prediction accuracy")
    print(line("overall", s["overall"]))
    for c, st in s["by_call"].items():
        print(line(c, st))
    for h, st in s["by_horizon"].items():
        print(line(f"{h}d", st))
    print(f"  pending predictions: {s['pending']}")
    return 0


def cmd_backtest(args):
    symbol = args.symbols[0] if args.symbols else config.WATCHLIST[0]
    try:
        prices = data.load_prices(symbol, period="5y", demo=args.demo)
    except data.DataError as exc:
        print(exc, file=sys.stderr)
        return 1
    res = backtest.walk_forward(prices, horizon=args.horizon)
    print(f"Walk-forward backtest: {symbol}, horizon {args.horizon} trading days")
    print("(news sentiment held neutral: no historical headlines)\n")
    for k, v in backtest.summarise(res).items():
        if isinstance(v, dict):
            acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.1%}"
            print(f"  {k:<34} n={v['n']:<5} accuracy={acc}")
        else:
            print(f"  {k:<34} {v if not isinstance(v, float) else format(v, '.1%')}")
    print(f"\n{config.DISCLAIMER}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="stock_analysis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("analyze", cmd_analyze), ("backtest", cmd_backtest)):
        sp = sub.add_parser(name)
        sp.add_argument("symbols", nargs="*")
        sp.add_argument("--demo", action="store_true", help="use offline synthetic data")
        if name == "analyze":
            sp.add_argument("--no-log", action="store_true")
        else:
            sp.add_argument("--horizon", type=int, default=5)
        sp.set_defaults(fn=fn)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
