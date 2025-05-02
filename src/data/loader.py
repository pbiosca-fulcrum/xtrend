import os
import glob
import numpy as np
from .utils import ensure_memmap

RAW_DIR  = os.path.expanduser("data/raw")
PROC_DIR = os.path.expanduser("data/processed")

class AssetDataset:
    """
    Loads preprocessed features & returns for one asset via memmaps,
    and ensures they are float32.
    """
    def __init__(self, ticker):
        feat_path = os.path.join(PROC_DIR, f"{ticker}_feat.npy")
        ret_path  = os.path.join(PROC_DIR, f"{ticker}_ret.npy")

        # load and cast to float32
        self.features = ensure_memmap(feat_path, mode='r').astype(np.float32)
        self.returns  = ensure_memmap(ret_path,  mode='r').astype(np.float32)
        self.length   = len(self.returns)

def all_tickers():
    # infer tickers by scanning processed folder
    files = glob.glob(os.path.join(PROC_DIR, "*_feat.npy"))
    return [os.path.basename(f).split("_")[0] for f in files]
