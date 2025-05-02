#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sys

# Ensure the top-level 'src' folder is on Python’s import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from data.loader import AssetDataset, all_tickers
from models.xtrend import XTrend
from models.dmn import DMN
from utils.metrics import sharpe, max_drawdown

def backtest(model, setting="fewshot"):
    model.eval()
    pnl, eq = [], [1.0]

    for tk in all_tickers():
        ds   = AssetDataset(tk)
        regs = json.load(open(REGIMES_FILE))[tk]

        for (s, e) in regs:
            # 1) skip any regime that ends before we have LT days of lookback
            if e < LT:  
                continue
            # 2) also ensure there's a next-day return
            if e + 1 >= ds.length:
                continue

            # — prepare the input windows
            x_np = ds.features[e - LT:e]           # shape [LT, feat_dim]
            # context regimes (reuse first CTX_SIZE slots)
            ctxs = []
            for (s0, e0) in regs[:CTX_SIZE]:
                if e0 > e:
                    continue
                arr = ds.features[s0: min(e0, s0 + CTX_LEN)]
                if len(arr) < CTX_LEN:
                    pad = np.zeros((CTX_LEN - len(arr), arr.shape[1]), dtype=np.float32)
                    arr = np.vstack([pad, arr])
                else:
                    arr = arr[:CTX_LEN]
                ctxs.append(arr)
            # if not enough contexts, pad with last window
            while len(ctxs) < CTX_SIZE:
                ctxs.append(x_np[-CTX_LEN:])
            ctx_np = np.stack(ctxs, axis=0)        # [CTX_SIZE, CTX_LEN, feat_dim]

            # — convert to float32 tensors
            x_tgt = torch.tensor(x_np,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_ctx = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if setting == "baseline":
                    pos = DMN(input_dim=7, hidden_dim=HID_DIM)(x_tgt).item()
                else:
                    mu, sd, pos = model(x_tgt, x_ctx)
                    pos = pos.item()

            ret = ds.returns[e]  # float32
            pnl.append(pos * ret / (sd.cpu().item() + 1e-6))
            eq.append(eq[-1] * (1 + pnl[-1]))

    return pnl, eq

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--setting",   type=str, required=True,
                   choices=["baseline","fewshot","zeroshot"])
    p.add_argument("--ckpt_path", type=str, required=True)
    args = p.parse_args()

    # load model
    if args.setting == "baseline":
        model = DMN(input_dim=7, hidden_dim=HID_DIM)
    else:
        model = XTrend(feat_dim=7, hid_dim=HID_DIM, ctx_size=CTX_SIZE)
    model.load_state_dict(
        torch.load(args.ckpt_path, map_location=DEVICE),
        strict=True
    )
    model.to(DEVICE)

    pnl, eq = backtest(model, setting=args.setting)
    print(f"Sharpe:       {sharpe(pnl):.4f}")
    print(f"Max Drawdown: {max_drawdown(eq):.4f}")

if __name__ == "__main__":
    main()
