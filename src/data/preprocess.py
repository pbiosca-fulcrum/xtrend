#!/usr/bin/env python3
import os
import sys

# Ensure the parent 'src' directory is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from config import RAW_DIR, PROC_DIR, CTX_LEN, REGIMES_FILE
from data.utils import save_memmap, get_macd, ewma_vol
from ruptures import Pelt

def segment_regimes(close, pen=5):
    """Return list of (start,end) indices via PELT."""
    algo = Pelt(model="rbf").fit(close.values)
    bkps = algo.predict(pen=pen)
    segs = []
    prev = 0
    for b in bkps:
        if b - prev >= 5:
            segs.append((prev, b))
        prev = b
    return segs

def compute_features(df):
    """Given a daily-ohlc df, produce features matrix [T x D] and 1-day returns."""
    # drop rows with missing prices
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    close = df["Close"]

    # 1) multi-horizon returns
    r1   = close.pct_change().fillna(0)
    r21  = close.pct_change(21).fillna(0)
    r63  = close.pct_change(63).fillna(0)
    r126 = close.pct_change(126).fillna(0)
    r252 = close.pct_change(252).fillna(0)
    # 2) MACD factor
    macd = get_macd(close, span_short=12, span_long=26)
    # 3) EWMA volatility (60 days)
    vol = ewma_vol(r1, span=60).clip(lower=1e-4)

    feats = np.vstack([r1, r21, r63, r126, r252, macd, vol]).T
    # clean any NaN or infinite
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    rets = r1.values
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    return feats, rets

def load_and_resample(ticker):
    """Load raw intraday CSV, resample to daily OHLCV."""
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df = (pd.read_csv(path, parse_dates=["Date"])  # may contain NaNs
            .sort_values("Date"))
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Date"] = df["Date"].dt.floor("D")

    daily = (df.groupby("Date")
               .agg({
                   "Open":      "first",
                   "High":      "max",
                   "Low":       "min",
                   "Close":     "last",
                   "Volume":    "sum",
                   "Tick Count":"sum"
               })
               .reset_index())
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"])
    return daily

def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    all_regs = {}

    for fn in os.listdir(RAW_DIR):
        if not fn.endswith(".csv"): continue
        ticker = fn[:-4]
        print(f"Processing {ticker}...")

        daily = load_and_resample(ticker)
        if len(daily) < CTX_LEN + 1:
            print(f"  → skipping {ticker}, insufficient data")
            continue

        feats, rets = compute_features(daily)

        feat_path = os.path.join(PROC_DIR, f"{ticker}_feat.npy")
        ret_path  = os.path.join(PROC_DIR, f"{ticker}_ret.npy")
        save_memmap(feat_path, feats.astype(np.float32))
        save_memmap(ret_path,  rets.astype(np.float32))

        regs = segment_regimes(daily["Close"])
        clipped = [(s, min(e, s + CTX_LEN)) for s,e in regs]
        all_regs[ticker] = clipped
        print(f"  → {len(clipped)} regimes found")

    with open(REGIMES_FILE, "w") as f:
        json.dump(all_regs, f, indent=2)
    print("All regimes saved to", REGIMES_FILE)

if __name__ == "__main__":
    main()
