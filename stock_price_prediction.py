"""
Stock Price Prediction (Linear Regression on Moving Averages) — cleaned from a Colab notebook export.

What this script does:
- Downloads OHLCV data from Yahoo Finance via yfinance
- Builds simple technical features (SMA_10, SMA_30)
- Trains a Linear Regression model
- Evaluates with MSE, RMSE, and R^2
- Produces basic plots

Usage:
  python stock_price_prediction.py --ticker LMT --start 2016-01-01 --end 2026-01-22 --plot

Notes:
- For time series, we set shuffle=False to avoid leakage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import yfinance as yf


@dataclass
class Results:
    mse: float
    rmse: float
    r2: float


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker}. Check ticker symbol and dates.")
    df = df.dropna()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA_10"] = out["Close"].rolling(window=10).mean()
    out["SMA_30"] = out["Close"].rolling(window=30).mean()
    out = out.dropna()
    return out


def train_and_evaluate(df: pd.DataFrame) -> tuple[LinearRegression, pd.Series, np.ndarray, Results]:
    X = df[["SMA_10", "SMA_30"]]
    y = df["Close"]

    # keep chronological order to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    return model, y_test, y_pred, Results(mse=mse, rmse=rmse, r2=r2)


def plot_results(y_test: pd.Series, y_pred: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual", alpha=0.8)
    plt.plot(y_test.index, y_pred, label="Predicted", alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted (Test Set)")
    plt.grid(True)
    plt.show()

    errors = y_test.to_numpy() - y_pred
    plt.figure(figsize=(10, 4))
    plt.hist(errors, bins=25, density=True, alpha=0.7)
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock price prediction with linear regression.")
    parser.add_argument("--ticker", default="LMT", help="Ticker symbol (e.g., LMT, AAPL, MSFT).")
    parser.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-01-22", help="End date YYYY-MM-DD.")
    parser.add_argument("--plot", action="store_true", help="Show plots.")
    args = parser.parse_args()

    raw = download_data(args.ticker, args.start, args.end)
    df = add_features(raw)
    _, y_test, y_pred, res = train_and_evaluate(df)

    print(f"Ticker: {args.ticker}")
    print(f"MSE : {res.mse:.4f}")
    print(f"RMSE: {res.rmse:.4f}")
    print(f"R^2 : {res.r2:.4f}")

    if args.plot:
        plot_results(y_test, y_pred, f"{args.ticker} — Actual vs Predicted Close (Test Set)")


if __name__ == "__main__":
    main()
