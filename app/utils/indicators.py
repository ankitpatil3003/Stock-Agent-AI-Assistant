# app/utils/indicators.py
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic TA indicators on a daily OHLCV DataFrame.

    Required columns: Date, Open, High, Low, Close, Volume
    Returns the same frame with columns:
      RSI_14, MACD, MACD_SIGNAL, SMA_50, SMA_200, VOL_SMA_20
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Ensure sorted by Date and clean column names
    if "Date" in out.columns:
        out = out.sort_values("Date").reset_index(drop=True)

    # Simple moving averages
    out["SMA_50"] = out["Close"].rolling(50, min_periods=1).mean()
    out["SMA_200"] = out["Close"].rolling(200, min_periods=1).mean()

    # RSI (14)
    rsi = ta.rsi(out["Close"], length=14)
    out["RSI_14"] = rsi

    # MACD
    macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        # macd frame has columns like MACD_12_26_9 / MACDs_12_26_9
        out["MACD"] = macd.iloc[:, 0]
        out["MACD_SIGNAL"] = macd.iloc[:, 1]
    else:
        out["MACD"] = pd.NA
        out["MACD_SIGNAL"] = pd.NA

    # Volume SMA (20)
    out["VOL_SMA_20"] = out["Volume"].rolling(20, min_periods=1).mean()

    # Replace infs, then forward/back fill a bit to avoid fresh-NaN edges
    out = out.replace([float("inf"), -float("inf")], pd.NA)
    out = out.ffill().bfill()

    # We must have at least Close to proceed
    out = out.dropna(subset=["Close"])

    return out
