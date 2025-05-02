#!/usr/bin/env python3
import os
import sys

# Ensure the top-level 'src' folder is on Python’s import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from config import *
from data.loader import AssetDataset, all_tickers
from utils.metrics import sharpe
from utils.cpd import detect_change_points
from models.xtrend import XTrend
from models.dmn import DMN

class EpisodicDataset(Dataset):
    def __init__(self, setting="fewshot"):
        # only keep tickers with enough history for lookback
        all_tks = all_tickers()
        valid_tks = []
        for tk in all_tks:
            ds = AssetDataset(tk)
            if ds.length > LT:
                valid_tks.append(tk)
        self.tickers = valid_tks
        with open(REGIMES_FILE, "r") as f:
            self.regimes = json.load(f)
        self.setting = setting

    def __len__(self):
        return len(self.tickers) * 100  # arbitrary episodes per ticker

    def __getitem__(self, idx):
        # pick random ticker
        tk = random.choice(self.tickers)
        ds = AssetDataset(tk)
        T = ds.length

        # sample a valid endpoint for the target window
        t_end = random.randint(LT, T - 1)

        # target feature window [t_end - LT, t_end)
        tgt_feat = ds.features[t_end - LT : t_end]

        # filter regimes that end before t_end
        valid_regs = [(s, e) for (s, e) in self.regimes[tk] if e < t_end]
        if not valid_regs:
            # fallback: last CTX_LEN days
            picks = [(t_end - CTX_LEN, t_end)] * CTX_SIZE
        else:
            # sample contexts with replacement
            picks = random.choices(valid_regs, k=CTX_SIZE)

        # build context windows
        ctx_samples = []
        for (s, e) in picks:
            ctx = ds.features[s:e]
            # pad or truncate to CTX_LEN
            if len(ctx) < CTX_LEN:
                pad = np.zeros((CTX_LEN - len(ctx), ctx.shape[1]), dtype=np.float32)
                ctx = np.vstack([pad, ctx])
            else:
                ctx = ctx[:CTX_LEN]
            ctx_samples.append(ctx)

        # stack into [CTX_SIZE, CTX_LEN, feat_dim]
        ctx_arr = np.stack(ctx_samples, axis=0)
        ret = ds.returns[t_end]
        return (
            tgt_feat.astype(np.float32),
            ctx_arr.astype(np.float32),
            np.float32(ret),
        )


def collate_fn(batch):
    tgts, ctxs, rets = zip(*batch)
    return (
        torch.from_numpy(np.stack(tgts, axis=0)),
        torch.from_numpy(np.stack(ctxs, axis=0)),
        torch.from_numpy(np.stack(rets, axis=0)),
    )


def main(args):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # prepare data
    ds = EpisodicDataset(setting=args.setting)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0
    )

    # instantiate model
    if args.setting == "baseline":
        model = DMN(input_dim=7, hidden_dim=HID_DIM)
    else:
        model = XTrend(feat_dim=7, hid_dim=HID_DIM, ctx_size=CTX_SIZE)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for ep in range(1, args.epochs + 1):
        model.train()
        losses, pnl_vals = [], []
        for tgt_feat, ctx_arr, ret in tqdm(dl, desc=f"Epoch {ep}"):
            tgt_feat = tgt_feat.to(DEVICE)
            ctx_arr  = ctx_arr.to(DEVICE)
            ret      = ret.to(DEVICE)

            optimizer.zero_grad()
            if args.setting == "baseline":
                pos = model(tgt_feat)
                pnl = pos * ret
            else:
                mu, sd, pos = model(tgt_feat, ctx_arr)
                pnl = pos * ret / (sd + 1e-6)

            loss = -torch.mean(pnl) / (torch.std(pnl) + 1e-6)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pnl_vals.extend(pnl.detach().cpu().tolist())

        print(f"Epoch {ep} — Loss: {np.mean(losses):.4f}, Sharpe: {sharpe(pnl_vals):.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = f"checkpoints/xtrend_{args.setting}.pt"
        torch.save(model.state_dict(), ckpt)

    print("Training complete. Last checkpoint:", ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting",    type=str,   default="fewshot", choices=["fewshot","zeroshot","baseline"])
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    args = parser.parse_args()
    main(args)
