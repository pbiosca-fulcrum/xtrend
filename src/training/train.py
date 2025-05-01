import os
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
        self.tickers = all_tickers()
        with open(REGIMES_FILE,"r") as f:
            self.regimes = json.load(f)
        self.setting = setting

    def __len__(self):
        return len(self.tickers) * 100  # arbitrary

    def __getitem__(self, idx):
        # pick random ticker
        tk = random.choice(self.tickers)
        ds = AssetDataset(tk)
        T = ds.length
        # sample target end t in [LT .. T-1]
        t_end = random.randint(LT, T-1)
        # get target window
        tgt_feat = ds.features[t_end-LT:t_end]
        # pick contexts
        regs = self.regimes[tk]
        ctx_samples = []
        for _ in range(CTX_SIZE):
            # choose random (s,e) with e < t_end
            cand = random.choice(regs)
            if cand[1] >= t_end: continue
            ctx = ds.features[cand[0]:cand[1]]
            # pad/truncate to CTX_LEN
            if len(ctx) >= CTX_LEN:
                ctx = ctx[:CTX_LEN]
            else:
                pad = np.zeros((CTX_LEN-len(ctx), ctx.shape[1]))
                ctx = np.vstack([pad, ctx])
            ctx_samples.append(ctx)
        ctx_arr = np.stack(ctx_samples, axis=0)  # [C, Lc, D]
        return (tgt_feat.astype(np.float32),
                ctx_arr.astype(np.float32),
                ds.returns[t_end])  # next-day return

def collate_fn(batch):
    tgts, ctxs, rets = zip(*batch)
    return (torch.tensor(np.stack(tgts)),
            torch.tensor(np.stack(ctxs)),
            torch.tensor(rets))

def main(args):
    torch.manual_seed(SEED)
    # dataset & loader
    ds = EpisodicDataset(setting=args.setting)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, collate_fn=collate_fn,
                    num_workers=0, drop_last=True)

    # model
    if args.setting=="baseline":
        model = DMN(input_dim=7, hidden_dim=HID_DIM)
    else:
        model = XTrend(feat_dim=7, hid_dim=HID_DIM, ctx_size=CTX_SIZE)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        losses = []
        pnl = []
        for x_tgt, x_ctx, ret in tqdm(dl, desc=f"Epoch {ep}"):
            x_tgt, x_ctx, ret = x_tgt.to(DEVICE), x_ctx.to(DEVICE), ret.to(DEVICE)
            opt.zero_grad()

            if args.setting=="baseline":
                pos = model(x_tgt)
                mu  = None; sd=None
            else:
                mu, sd, pos = model(x_tgt, x_ctx)

            # compute Sharpe loss: −mean( pos * ret / sd ) / std(...)
            pnl_batch = (pos * ret / (sd+1e-6)).detach().cpu().numpy()
            loss = - torch.mean( (pos*ret/sd) ) / (torch.std(pos*ret/sd)+1e-6)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pnl.extend(pnl_batch.tolist())

        print(f"Epoch {ep} — loss {np.mean(losses):.4f}  Sharpe {sharpe(pnl):.4f}")

        # save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/xtrend_{args.setting}.pt")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setting", type=str, default="fewshot",
                   choices=["fewshot","zeroshot","baseline"])
    p.add_argument("--epochs",    type=int, default=EPOCHS)
    p.add_argument("--batch_size",type=int, default=BATCH_SIZE)
    p.add_argument("--lr",        type=float, default=LR)
    args = p.parse_args()
    main(args)
