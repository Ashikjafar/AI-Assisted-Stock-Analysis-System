import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from stock_analysis import agents, backtest, data, sentiment, tracker
from stock_analysis.indicators import add_indicators, rsi, sma


class IndicatorTests(unittest.TestCase):
    def test_sma_known_values(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(sma(s, 3).iloc[-1], 4.0)
        self.assertTrue(np.isnan(sma(s, 3).iloc[1]))

    def test_rsi_bounds_and_extremes(self):
        up = pd.Series(np.arange(1.0, 60.0))
        down = pd.Series(np.arange(60.0, 1.0, -1.0))
        self.assertEqual(rsi(up).iloc[-1], 100.0)
        self.assertLess(rsi(down).iloc[-1], 1.0)
        r = rsi(data.synthetic_prices("X")["close"]).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_indicators_are_causal(self):
        """Changing the future must not change past indicator values."""
        df = data.synthetic_prices("CAUSAL", days=200)
        a = add_indicators(df)
        df2 = df.copy()
        df2.iloc[150:, df2.columns.get_loc("close")] *= 2
        b = add_indicators(df2)
        pd.testing.assert_series_equal(a["rsi14"].iloc[:150], b["rsi14"].iloc[:150])
        pd.testing.assert_series_equal(a["macd_hist"].iloc[:150], b["macd_hist"].iloc[:150])


class SentimentTests(unittest.TestCase):
    def test_polarity(self):
        self.assertGreater(sentiment.score_headline("Profit beats estimates, shares surge")[0], 0)
        self.assertLess(sentiment.score_headline("Regulator probe, shares plunge")[0], 0)

    def test_negation_flips(self):
        self.assertLess(sentiment.score_headline("Company does not gain approval")[0], 0)

    def test_empty_is_neutral(self):
        self.assertEqual(sentiment.aggregate([])["score"], 0.0)


class AgentTests(unittest.TestCase):
    def _frame(self, **last):
        base = dict(close=110, sma20=105, sma50=100, rsi14=55, macd_hist=0.5,
                    bb_pos=0.5, vol_ratio=1.0, ret_1d=0.01)
        prev = dict(base, macd_hist=0.2)
        base.update(last)
        return pd.DataFrame([prev, base])

    def test_bullish_setup_is_buy(self):
        v = agents.debate(self._frame(), sentiment=0.4)
        self.assertEqual(v.call, "BUY")

    def test_bearish_setup_is_avoid(self):
        df = self._frame(close=90, sma20=95, sma50=100, rsi14=75, macd_hist=-0.5)
        df.loc[0, "macd_hist"] = -0.2
        v = agents.debate(df, sentiment=-0.4)
        self.assertEqual(v.call, "AVOID")

    def test_mixed_is_wait(self):
        v = agents.judge(agents.Argument("bull", [("a", 2.0)]),
                         agents.Argument("bear", [("b", 1.5)]))
        self.assertEqual(v.call, "WAIT")

    def test_no_evidence_is_wait(self):
        v = agents.judge(agents.Argument("bull"), agents.Argument("bear"))
        self.assertEqual((v.call, v.confidence), ("WAIT", 0.0))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.conn = tracker.connect(":memory:")
        self.verdict = agents.judge(agents.Argument("bull", [("a", 4.0)]),
                                    agents.Argument("bear"))

    def test_score_rules(self):
        self.assertTrue(tracker.score_call("BUY", 1.0))
        self.assertFalse(tracker.score_call("BUY", -1.0))
        self.assertTrue(tracker.score_call("AVOID", -0.5))
        self.assertTrue(tracker.score_call("WAIT", 0.5))
        self.assertFalse(tracker.score_call("WAIT", 3.0))

    def test_log_evaluate_and_accuracy(self):
        past = datetime.now() - timedelta(days=30)
        n = tracker.log_prediction(self.conn, "TCS.NS", self.verdict, 0.2, 100.0, now=past)
        self.assertEqual(n, 3)
        done = tracker.evaluate_due(self.conn, lambda s, d: 110.0)
        self.assertEqual(done, 3)
        s = tracker.accuracy_summary(self.conn)
        self.assertEqual(s["overall"]["n"], 3)
        self.assertEqual(s["overall"]["accuracy"], 1.0)
        self.assertEqual(s["pending"], 0)

    def test_future_predictions_stay_pending(self):
        tracker.log_prediction(self.conn, "TCS.NS", self.verdict, 0.0, 100.0)
        self.assertEqual(tracker.evaluate_due(self.conn, lambda s, d: 110.0,
                                              today=datetime.now()), 0)
        self.assertEqual(tracker.accuracy_summary(self.conn)["pending"], 3)

    def test_missing_price_stays_pending(self):
        past = datetime.now() - timedelta(days=30)
        tracker.log_prediction(self.conn, "TCS.NS", self.verdict, 0.0, 100.0, now=past)
        self.assertEqual(tracker.evaluate_due(self.conn, lambda s, d: None), 0)


class BacktestTests(unittest.TestCase):
    def test_walk_forward_runs_and_summarises(self):
        df = data.synthetic_prices("TCS.NS", days=400)
        res = backtest.walk_forward(df, horizon=5)
        self.assertFalse(res.empty)
        self.assertTrue(set(res["debate_call"]) <= {"BUY", "AVOID", "WAIT"})
        summ = backtest.summarise(res)
        self.assertIn("always buy", summ)
        acc = summ["always buy"]["accuracy"]
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_synthetic_data_is_deterministic(self):
        a = data.synthetic_prices("ABC")
        b = data.synthetic_prices("ABC")
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
