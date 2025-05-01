# src/data/utils.py
import numpy as np
import os

def save_memmap(path, arr):
    """Save numpy array as a read/write memmap (then close)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    m[:] = arr[:]
    del m

def ensure_memmap(path, mode='r'):
    """Open a numpy memmap in given mode."""
    return np.memmap(path, mode=mode)

def ewma_vol(returns, span=60):
    """
    Exponentially-weighted standard deviation of a pandas Series.
    Returns a pandas Series of vol.
    """
    # pandas EWM std
    return returns.ewm(span=span, adjust=False).std()

def get_macd(close, span_short=12, span_long=26):
    """
    Compute MACD factor: 
      m_t = EMA_short(close) - EMA_long(close)
      macd  = m_t / rolling_std(m over 252 days)
    """
    ema_s = close.ewm(span=span_short, adjust=False).mean()
    ema_l = close.ewm(span=span_long,  adjust=False).mean()
    m     = ema_s - ema_l
    norm  = m.rolling(window=252, min_periods=1).std().replace(0, 1e-4)
    macd  = (m / norm).fillna(0)
    return macd.values
