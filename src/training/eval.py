#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# ensure top-level src/ on Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from data.loader import AssetDataset, all_tickers
from data.preprocess import load_and_resample
from models.dmn import DMN
from models.xtrend import XTrend
from utils.metrics import sharpe, max_drawdown

def backtest_model(model, setting="fewshot"):
    model.eval()
    pnl, eq = [], [1.0]

    for tk in all_tickers():
        ds   = AssetDataset(tk)
        regs = json.load(open(REGIMES_FILE))[tk]

        for (s, e) in regs:
            if e < LT or e + 1 >= ds.length:
                continue

            # prepare windows
            x_np = ds.features[e - LT : e]
            # build contexts
            ctxs = []
            for (s0, e0) in regs[:CTX_SIZE]:
                if e0 > e: continue
                arr = ds.features[s0 : min(e0, s0 + CTX_LEN)]
                if len(arr) < CTX_LEN:
                    pad = np.zeros((CTX_LEN - len(arr), arr.shape[1]), dtype=np.float32)
                    arr = np.vstack([pad, arr])
                else:
                    arr = arr[:CTX_LEN]
                ctxs.append(arr)
            while len(ctxs) < CTX_SIZE:
                ctxs.append(x_np[-CTX_LEN:])
            ctx_np = np.stack(ctxs, axis=0)

            # tensorize
            x_tgt = torch.tensor(x_np,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_ctx = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # forward
            with torch.no_grad():
                if setting == "baseline":
                    pos = model(x_tgt).item()
                else:
                    mu, sd, pos = model(x_tgt, x_ctx)
                    pos = pos.item()

            ret = ds.returns[e]
            if setting == "baseline":
                pnl.append(pos * ret)
            else:
                pnl.append(pos * ret / (sd + 1e-6))
            eq.append(eq[-1] * (1 + pnl[-1]))

    return np.array(pnl), np.array(eq)


def backtest_ewma():
    """Traditional EWMA(9)−EWMA(65) signal backtest, per‐ticker."""
    eq_dict = {}
    all_pnls = []

    for tk in all_tickers():
        df = load_and_resample(tk)[["Date", "Close"]]
        df.set_index("Date", inplace=True)

        # compute EWMAs
        short = df["Close"].ewm(span=EWMA_SHORT, adjust=False).mean()
        long_ = df["Close"].ewm(span=EWMA_LONG,  adjust=False).mean()

        # signal = sign(diff) shifted one day
        sig = np.sign(short - long_).shift(1).fillna(0)

        # daily returns
        ret = df["Close"].pct_change().fillna(0)

        pnl = sig * ret
        eq  = (1 + pnl).cumprod()

        eq_dict[tk] = eq.values
        all_pnls.append(pnl.values)

    # flatten all pnls for overall stats
    flat_pnl = np.concatenate(all_pnls)
    return eq_dict, flat_pnl


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--setting",   type=str, required=True,
                   choices=["baseline","fewshot","zeroshot","ewma"])
    p.add_argument("--ckpt_path", type=str, default=None)
    args = p.parse_args()

    if args.setting == "ewma":
        # run the traditional EWMA strategy
        equity_dict, flat_pnl = backtest_ewma()

        # overall metrics
        print(f"EWMA Strategy (span={EWMA_SHORT},{EWMA_LONG})")
        print(f"Overall Sharpe:       {sharpe(flat_pnl):.4f}")
        print()

        # 3×3 grid of equity curves for first 9 tickers
        tickers = list(equity_dict.keys())[:9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
        for ax, tk in zip(axes.flat, tickers):
            ax.plot(equity_dict[tk], linewidth=1)
            ax.set_title(tk)
            ax.tick_params(labelsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Days")
        for ax in axes[:,0]:
            ax.set_ylabel("Equity")
        plt.suptitle(f"EWMA({EWMA_SHORT})−EWMA({EWMA_LONG}) Strategy — 3×3 Equity Curves", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.96])
        out_png = f"ewma_{EWMA_SHORT}_{EWMA_LONG}_3x3.png"
        plt.savefig(out_png)
        print(f"3×3 equity grid saved to {out_png}")
        return

    # otherwise load a neural model
    # instantiate model
    if args.setting == "baseline":
        model = DMN(input_dim=7, hidden_dim=HID_DIM)
    else:
        model = XTrend(feat_dim=7, hid_dim=HID_DIM, ctx_size=CTX_SIZE)

    # load weights
    state = torch.load(args.ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)

    pnl, eq = backtest_model(model, setting=args.setting)

    # overall metrics
    print(f"Sharpe:       {sharpe(pnl):.4f}")
    print(f"Max Drawdown: {max_drawdown(eq):.4f}\n")

    # per-year breakdown
    trades_per_year = 252
    n_years = len(pnl) // trades_per_year
    for i in range(n_years):
        start, end = i*trades_per_year, (i+1)*trades_per_year
        yr = pnl[start:end]
        print(f"Year {i+1:2d} Ret {np.prod(1+yr)-1:.1%}, Sharpe {sharpe(yr):.4f}")
    rem = len(pnl) % trades_per_year
    if rem:
        yr = pnl[-rem:]
        print(f"Partial Year Ret {np.prod(1+yr)-1:.1%}, Sharpe {sharpe(yr):.4f}")

    # equity curve
    plt.figure(figsize=(10,6))
    plt.plot(eq, linewidth=1)
    plt.title(f"Equity Curve — {args.setting.capitalize()}")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    fn = f"equity_curve_{args.setting}.png"
    plt.tight_layout()
    plt.savefig(fn)
    print(f"\nEquity curve saved to {fn}")


if __name__ == "__main__":
    main()
