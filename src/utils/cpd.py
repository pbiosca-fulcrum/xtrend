import ruptures as rpt

def detect_change_points(series, model="rbf", pen=5):
    """Return breakpoints via PELT on 1D array."""
    algo = rpt.Pelt(model=model).fit(series)
    return algo.predict(pen=pen)
