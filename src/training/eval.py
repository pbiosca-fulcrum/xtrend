#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# ensure top-level src/ on Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from data.loader import AssetDataset, all_tickers
from models.dmn import DMN
from models.xtrend import XTrend
from utils.metrics import sharpe, max_drawdown

def backtest(model, setting="fewshot"):
    model.eval()
    pnl, eq = [], [1.0]
    mus, sds, poses, rets = [], [], [], []

    for tk in all_tickers():
        ds   = AssetDataset(tk)
        regs = json.load(open(REGIMES_FILE))[tk]

        for s,e in regs:
            if e<LT or e+1>=ds.length: continue
            x_np = ds.features[e-LT:e]
            # contexts (as before)...
            ctxs = []
            for s0,e0 in regs[:CTX_SIZE]:
                if e0>e: continue
                arr = ds.features[s0:min(e0,s0+CTX_LEN)]
                if len(arr)<CTX_LEN:
                    pad=np.zeros((CTX_LEN-len(arr),arr.shape[1]),dtype=np.float32)
                    arr=np.vstack([pad,arr])
                else:
                    arr=arr[:CTX_LEN]
                ctxs.append(arr)
            while len(ctxs)<CTX_SIZE:
                ctxs.append(x_np[-CTX_LEN:])
            ctx_np = np.stack(ctxs)

            x_tgt = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_ctx = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if setting=="baseline":
                    pos = model(x_tgt).item(); mu, sd = None, None
                else:
                    mu, sd, pos = model(x_tgt, x_ctx)
                    mu, sd, pos = mu.item(), sd.item(), pos.item()

            ret = float(ds.returns[e])
            pnl.append(pos * ret / (sd + 1e-6) if sd else pos*ret)
            eq.append(eq[-1] * (1 + pnl[-1]))

            mus.append(mu if mu is not None else 0.0)
            sds.append(sd if sd is not None else 0.0)
            poses.append(pos)
            rets.append(ret)

    return np.array(pnl), np.array(eq), np.array(mus), np.array(sds), np.array(poses), np.array(rets)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--setting",   type=str, required=True,
                   choices=["baseline","fewshot","zeroshot"])
    p.add_argument("--ckpt_path", type=str, required=True)
    args = p.parse_args()

    model = DMN(input_dim=7, hidden_dim=HID_DIM) if args.setting=="baseline" \
            else XTrend(feat_dim=7, hid_dim=HID_DIM, ctx_size=CTX_SIZE)
    state = torch.load(args.ckpt_path, map_location=DEVICE)
    model.load_state_dict(state); model.to(DEVICE)

    pnl, eq, mus, sds, poses, rets = backtest(model, args.setting)

    # Core performance
    print(f"Overall Sharpe:       {sharpe(pnl):.4f}")
    print(f"Overall Max Drawdown: {max_drawdown(eq):.4f}\n")

    # Distributional stats
    print("Forecast distribution:")
    print(f"  μ    mean/std: {mus.mean():.4f} ± {mus.std():.4f}")
    print(f"  σ    mean/std: {sds.mean():.4f} ± {sds.std():.4f}")
    print(f"  pos  mean/std: {poses.mean():.4f} ± {poses.std():.4f}")
    print(f"  ret  mean/std: {rets.mean():.4f} ± {rets.std():.4f}")
    print(f"  corr(mu,ret): {np.corrcoef(mus, rets)[0,1]:.4f}")
    print(f"  corr(pos,ret): {np.corrcoef(poses, rets)[0,1]:.4f}\n")

    # Annual breakdown
    trades_per_year = 252
    years = len(pnl)//trades_per_year
    for i in range(years):
        s,e = i*trades_per_year, (i+1)*trades_per_year
        yr = pnl[s:e]
        print(f"Year {i+1:2d} Ret {np.prod(1+yr)-1:.1%}, Sharpe {sharpe(yr):.4f}")
    if len(pnl)%trades_per_year:
        s = years*trades_per_year
        yr = pnl[s:]
        print(f"Partial Year Ret {(np.prod(1+yr)-1):.1%}, Sharpe {sharpe(yr):.4f}")

    # Equity curve
    plt.figure(figsize=(10,6))
    plt.plot(eq)
    plt.title(f"Equity Curve — {args.setting}")
    plt.xlabel("Trade#"); plt.ylabel("Equity")
    fn = f"equity_curve_{args.setting}.png"
    plt.savefig(fn)
    print(f"\nEquity curve saved to {fn}")

if __name__=="__main__":
    main()
