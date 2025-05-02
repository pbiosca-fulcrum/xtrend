# src/config.py
import os
import torch

# data paths
RAW_DIR      = os.path.expanduser("data/2025-05-01")  # Updated to your specific directory
PROC_DIR     = os.path.expanduser("data/processed")
REGIMES_FILE = os.path.join(PROC_DIR, "regimes.json")

# model & training hyperparameters
HID_DIM      = 64      # hidden dimension
CTX_SIZE     = 20      # number of context regimes
CTX_LEN      = 21      # days per regime segment
LT           = 126     # target lookback days
LS           = 63      # warm-up days for Sharpe loss
LR           = 1e-3
BATCH_SIZE   = 32
EPOCHS       = 20
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
