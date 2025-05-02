import numpy as np
import os
import pandas as pd

def save_memmap(path, arr):
    """Save a numpy array as a .npy file (with header)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def ensure_memmap(path, mode='r'):
    """
    Open a numpy .npy file as a memmap if mode='r', else load it normally.
    Returns an array-like object with the correct shape.
    """
    if mode == 'r':
        return np.load(path, mmap_mode='r')
    else:
        return np.load(path)

def ewma_vol(returns, span=60):
    """
    Exponentially-weighted standard deviation of a pandas Series.
    Returns a pandas Series of vol with NO NaNs.
    """
    # compute rolling EWM std; the very first entry will be NaN
    vol = pd.Series(returns).ewm(span=span, adjust=False).std()
    # fill any NaNs (especially at the start) with a small positive floor
    return vol.fillna(1e-4)

def get_macd(close, span_short=12, span_long=26):
    """
    Compute MACD factor:
      - ema_short = EWMA(close, span_short)
      - ema_long  = EWMA(close, span_long)
      - diff      = ema_short - ema_long
      - macd      = diff / rolling_std(diff over 252 days)
    """
    s = pd.Series(close)
    ema_s = s.ewm(span=span_short, adjust=False).mean()
    ema_l = s.ewm(span=span_long,  adjust=False).mean()
    diff  = ema_s - ema_l
    denom = diff.rolling(window=252, min_periods=1).std().replace(0, 1e-4)
    macd  = (diff / denom).fillna(0)
    return macd.values
