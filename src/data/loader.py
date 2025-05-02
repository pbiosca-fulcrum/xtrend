import os
import glob
import numpy as np
from .utils import ensure_memmap

RAW_DIR  = os.path.expanduser("data/raw")
PROC_DIR = os.path.expanduser("data/processed")

class AssetDataset:
    """
    Loads preprocessed features & returns via memmaps, cleans NaNs/Infs.
    """
    def __init__(self, ticker):
        feat_path = os.path.join(PROC_DIR, f"{ticker}_feat.npy")
        ret_path  = os.path.join(PROC_DIR, f"{ticker}_ret.npy")

        # load memmaps
        feats = ensure_memmap(feat_path, mode='r').astype(np.float32)
        rets  = ensure_memmap(ret_path,  mode='r').astype(np.float32)

        # clean any NaN or infinite
        self.features = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        self.returns  = np.nan_to_num(rets,  nan=0.0, posinf=0.0, neginf=0.0)
        self.length   = len(self.returns)


def all_tickers():
    files = glob.glob(os.path.join(PROC_DIR, "*_feat.npy"))
    return [os.path.basename(f).split("_")[0] for f in files]