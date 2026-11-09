# Presentation Guide

Suggested length: 8 to 10 minutes plus a 2-minute live demo.

## Slide outline

1. **Title:** AI-Assisted Stock Analysis System (name, register number, guide).
2. **Problem:** retail investors get tips with no reasoning and no track record. Many student "predictors" show inflated accuracy on back-tested data.
3. **Objective:** build an explainable decision-support system and measure its accuracy honestly.
4. **Existing approaches and gap:** black-box models, no explanation, no live accuracy tracking.
5. **Architecture:** use the diagram from the README.
6. **Data and indicators:** price data, SMA, RSI, MACD, Bollinger, volume, and why each is used.
7. **Sentiment module:** headline scoring, negation, explainability.
8. **Debate engine:** Bull vs Bear vs Judge, with one worked example from the demo output.
9. **Prediction tracker:** log each call, score it later, show scoring rules.
10. **Evaluation:** walk-forward backtest vs always-buy and RSI baselines, and the live accuracy table with sample sizes.
11. **Halal filter:** how the watchlist is screened and why compliance must be re-verified.
12. **Limitations and future work:** be upfront (see README).
13. **Conclusion and demo.**

## Live demo script (2 minutes)

1. `python -m stock_analysis analyze --demo TCS.NS` shows the debate and the Judge call.
2. `python -m stock_analysis backtest --demo TCS.NS` shows the baseline comparison.
3. `streamlit run app.py` shows the dashboard. Use the offline demo toggle if the venue has no internet.

Demo mode uses synthetic data, so say so out loud. Never present demo numbers as real market results.

## Before the presentation

- Run `analyze` on live data for 2 to 3 weeks before the date, then `evaluate`, so real predictions exist.
- Report accuracy together with the number of predictions (for example "55% over 40 scored calls"), never alone.
- Run `backtest` on live data for 3 or 4 stocks and note the results.

## Likely viva questions

**Why rule-based and not machine learning?**
It is explainable and needs no training data. The weights can be learned later; the tracker already collects the data needed for that.

**How do you know the accuracy is not luck?**
Compare with baselines and report the sample size. A small sample cannot prove skill, and the report says so.

**What is look-ahead bias, and how do you avoid it?**
Using future data when making a past decision. All indicators are causal, and a unit test checks that changing future prices does not change past indicator values.

**Why is WAIT a valid call?**
Acting only when the evidence is strong reduces bad trades. WAIT is scored as correct only when the price stays within 1%.

**Does sentiment help?**
It is only used in live calls because historical headlines are not available. Measuring its effect needs weeks of tracked live data.

**Is this guaranteed to make money?**
No. It is decision support. Accuracy above 50% does not imply profit once costs, slippage and position sizing are included.

**What would you improve?**
Learned weights, a transformer sentiment model, risk management, and a larger tracked sample.
