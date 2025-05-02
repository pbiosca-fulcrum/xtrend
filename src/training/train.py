#!/usr/bin/env python3
import os
import sys
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ensure top-level src/ on Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from data.loader import AssetDataset, all_tickers
from utils.metrics import sharpe
from models.dmn import DMN
from models.xtrend import XTrend

class EpisodicDataset(Dataset):
    def __init__(self, tickers, regimes, setting):
        self.tickers = tickers
        self.regimes = regimes
        self.setting = setting

    def __len__(self):
        return len(self.tickers) * 100

    def __getitem__(self, idx):
        tk = random.choice(self.tickers)
        ds = AssetDataset(tk)
        t_end = random.randint(LT, ds.length - 1)
        tgt = ds.features[t_end - LT : t_end]
        regs = [(s, e) for (s, e) in self.regimes[tk] if e < t_end]
        picks = random.choices(regs, k=CTX_SIZE) if regs else [(t_end-CTX_LEN, t_end)]*CTX_SIZE
        ctxs = []
        for s, e in picks:
            arr = ds.features[s : min(e, s+CTX_LEN)]
            if len(arr)<CTX_LEN:
                pad = np.zeros((CTX_LEN - len(arr), arr.shape[1]),dtype=np.float32)
                arr = np.vstack([pad, arr])
            else:
                arr = arr[:CTX_LEN]
            ctxs.append(arr)
        return (
            tgt.astype(np.float32),
            np.stack(ctxs).astype(np.float32),
            np.float32(ds.returns[t_end])
        )

def collate_fn(batch):
    tgts, ctxs, rets = zip(*batch)
    return (
        torch.from_numpy(np.stack(tgts)),
        torch.from_numpy(np.stack(ctxs)),
        torch.from_numpy(np.stack(rets))
    )

def main(args):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    with open(REGIMES_FILE) as f:
        regimes = json.load(f)

    all_tks = all_tickers()
    valid = [tk for tk in all_tks if AssetDataset(tk).length > LT]
    random.shuffle(valid)
    split = int(len(valid)*(1-VAL_SPLIT))
    train_tks, val_tks = valid[:split], valid[split:]
    print(f"→ {len(train_tks)} train tickers, {len(val_tks)} val tickers")

    ds_tr = EpisodicDataset(train_tks, regimes, args.setting)
    ds_val= EpisodicDataset(val_tks,   regimes, args.setting)
    dl_tr = DataLoader(ds_tr,  batch_size=args.batch_size, shuffle=True,
                      collate_fn=collate_fn, drop_last=True)
    dl_val= DataLoader(ds_val, batch_size=VAL_BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn, drop_last=False)

    model = DMN(input_dim=7, hidden_dim=HID_DIM) if args.setting=="baseline" \
            else XTrend(feat_dim=7, hid_dim=HID_DIM, ctx_size=CTX_SIZE)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_sh = -float("inf")
    for ep in range(1, args.epochs+1):
        start_time = time.time()
        model.train()

        # accumulators
        total_losses = []
        nll_losses   = []
        sharpe_losses= []
        pnl_values   = []
        pos_means    = []
        grad_norms   = []

        for tgt, ctx, ret in tqdm(dl_tr, desc=f"Epoch {ep} [Train]"):
            tgt, ctx, ret = tgt.to(DEVICE), ctx.to(DEVICE), ret.to(DEVICE)
            optimizer.zero_grad()

            if args.setting=="baseline":
                pos = model(tgt)
                pnl = pos * ret
                loss = -pnl.mean() / (pnl.std()+1e-6)
                nll, sl = torch.tensor(0.), loss.detach()
            else:
                mu, sd, pos = model(tgt, ctx)
                # NLL
                nll = 0.5*((ret-mu)**2)/(sd**2+1e-6) + torch.log(sd+1e-6)
                nll = nll.mean()
                # Sharpe loss
                pnl = pos * ret / (sd+1e-6)
                pnl_wu = pnl[LS:] if len(pnl)>LS else pnl
                sl = -pnl_wu.mean() / (pnl_wu.std()+1e-6)
                loss = nll + sl

            loss.backward()
            # gradient norm
            total_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))
            grad_norms.append(total_norm.item())
            optimizer.step()

            total_losses.append(loss.item())
            nll_losses.append(nll.item())
            sharpe_losses.append(sl.item())
            pnl_values.extend(pnl.detach().cpu().tolist())
            pos_means.append(pos.detach().cpu().mean().item())

        # epoch summaries
        tr_sh = sharpe(pnl_values)
        print(f"\nEpoch {ep} summary:")
        print(f"  Duration:           {time.time()-start_time:.1f}s")
        print(f"  Total loss:         {np.mean(total_losses):.4f} ± {np.std(total_losses):.4f}")
        print(f"  NLL loss:           {np.mean(nll_losses):.4f} ± {np.std(nll_losses):.4f}")
        print(f"  Sharpe loss:        {np.mean(sharpe_losses):.4f} ± {np.std(sharpe_losses):.4f}")
        print(f"  PnL Sharpe:         {tr_sh:.4f}")
        print(f"  Position mean/std:  {np.mean(pos_means):.4f} ± {np.std(pos_means):.4f}")
        print(f"  Grad norm:          {np.mean(grad_norms):.4f} ± {np.std(grad_norms):.4f}")

        # validation
        model.eval()
        val_pnls = []
        with torch.no_grad():
            for tgt, ctx, ret in tqdm(dl_val, desc=f"Epoch {ep} [Val]"):
                tgt, ctx, ret = tgt.to(DEVICE), ctx.to(DEVICE), ret.to(DEVICE)
                if args.setting=="baseline":
                    pos = model(tgt); pnl = pos*ret
                else:
                    mu, sd, pos = model(tgt, ctx)
                    pnl = pos*ret/(sd+1e-6)
                val_pnls.extend(pnl.cpu().tolist())

        val_sh = sharpe(val_pnls)
        print(f"  Validation Sharpe:  {val_sh:.4f}\n")

        # checkpoint
        ckpt = f"checkpoints/{args.setting}.pt"
        if val_sh > best_val_sh:
            best_val_sh = val_sh
            torch.save(model.state_dict(), ckpt)
            print(f"  → New best model saved to {ckpt}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--setting",   type=str, required=True,
                   choices=["baseline","fewshot","zeroshot"])
    p.add_argument("--lr",        type=float, default=LR)
    p.add_argument("--batch_size",type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",    type=int,   default=EPOCHS)
    args = p.parse_args()

    main(args)
    
    