

# ML-Based Alpha Factor Research

This project explores the use of machine learning to identify predictive alpha factors for short-term equity price movements.

---

## Motivation

Quantitative trading relies on identifying alpha signals from noisy market data. This project attempts to combine traditional financial features with machine learning models to generate actionable predictions.

---

## Methods

- Historical equity data fetched using `yfinance`
- Engineered alpha factors: 
  - Momentum
  - Volatility
  - Z-score
  - RSI, MACD
  - Bollinger Bands
- Binary classification: predict whether return over the next 5 days is positive
- Models:
  - Logistic Regression (baseline)
  - XGBoost or CNN (TBD)
- Walk-forward time-based validation
- Backtest signal-based positions and evaluate:
  - Sharpe Ratio
  - Cumulative Returns
  - Hit Rate (accuracy)

---

## Installation

```bash
pip install -r requirements.txt
