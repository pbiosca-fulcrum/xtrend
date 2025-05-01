import os
import glob
import numpy as np
import pandas as pd
from .utils import ensure_memmap

RAW_DIR  = os.path.expanduser("data/raw")
PROC_DIR = os.path.expanduser("data/processed")

class AssetDataset:
    """
    Loads preprocessed features & returns for one asset via memmaps.
    """
    def __init__(self, ticker):
        feat_path = os.path.join(PROC_DIR, f"{ticker}_feat.npy")
        ret_path  = os.path.join(PROC_DIR, f"{ticker}_ret.npy")
        self.features = ensure_memmap(feat_path, mode='r')
        self.returns  = ensure_memmap(ret_path,  mode='r')
        self.length   = len(self.returns)

def all_tickers():
    # infer tickers by scanning processed folder
    files = glob.glob(os.path.join(PROC_DIR, "*_feat.npy"))
    return [os.path.basename(f).split("_")[0] for f in files]
