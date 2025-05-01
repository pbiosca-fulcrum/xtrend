import os
import json
import torch
import numpy as np
from tqdm import tqdm

from config import *
from data.loader import AssetDataset, all_tickers
from models.xtrend import XTrend
from models.dmn import DMN
from utils.metrics import sharpe, max_drawdown

def backtest(model, setting="fewshot"):
    model.eval()
    pnl, eq = [], [1.0]
    for tk in all_tickers():
        ds = AssetDataset(tk)
        regs = json.load(open(REGIMES_FILE))[tk]
        for (s,e) in regs:
            if e+1 >= ds.length: continue
            x_tgt = torch.tensor(ds.features[e-LT:e]).unsqueeze(0).to(DEVICE)
            # build arbitrary random contexts before e
            # here reuse first CTX_SIZE regs
            ctxs = []
            for (s0,e0) in regs[:CTX_SIZE]:
                arr = ds.features[s0: min(e0, s0+CTX_LEN)]
                if len(arr)<CTX_LEN:
                    pad = np.zeros((CTX_LEN-len(arr), arr.shape[1]))
                    arr = np.vstack([pad, arr])
                ctxs.append(arr)
            x_ctx = torch.tensor(np.stack(ctxs)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                if setting=="baseline":
                    pos = model(x_tgt).cpu().item()
                else:
                    mu, sd, pos = model(x_tgt, x_ctx)
                    pos = pos.cpu().item()
            ret = ds.returns[e]
            pnl.append(pos * ret / (sd.cpu().item()+1e-6))
            eq.append(eq[-1] * (1 + pnl[-1]))
    return pnl, eq

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--setting", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    args = p.parse_args()

    # load model
    if args.setting=="baseline":
        model = DMN(input_dim=7, hidden_dim=HID_DIM)
    else:
        model = XTrend(feat_dim=7, hid_dim=HID_DIM, ctx_size=CTX_SIZE)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=DEVICE))
    model.to(DEVICE)

    # run backtest
    pnl, eq = backtest(model, setting=args.setting)
    print(f"Sharpe: {sharpe(pnl):.4f}")
    print(f"Max Drawdown: {max_drawdown(eq):.4f}")

if __name__=="__main__":
    main()
