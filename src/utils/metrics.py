import numpy as np

def sharpe(returns, freq=252):
    """Annualized Sharpe ratio."""
    r = np.array(returns)
    return np.sqrt(freq) * r.mean() / (r.std(ddof=1)+1e-9)

def max_drawdown(equity):
    """Max drawdown of PnL series."""
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    return dd.min()
