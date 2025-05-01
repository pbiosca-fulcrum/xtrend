import torch
import torch.nn as nn

class XTrend(nn.Module):
    """
    Cross-Attentive Time-Series Trend Network (Gaussian variant).
    """
    def __init__(self,
                 feat_dim: int,
                 hid_dim:  int,
                 ctx_size: int,
                 n_heads:  int=4):
        super().__init__()
        self.hid_dim  = hid_dim
        self.ctx_size = ctx_size

        # encoder LSTM for target & contexts
        self.enc_lstm = nn.LSTM(feat_dim, hid_dim, batch_first=True)
        # self-attention over contexts
        self.self_attn = nn.MultiheadAttention(hid_dim, n_heads, batch_first=True)
        # projections for cross-attention
        self.q_proj = nn.Linear(hid_dim, hid_dim)
        self.k_proj = nn.Linear(hid_dim, hid_dim)
        self.v_proj = nn.Linear(hid_dim, hid_dim)
        self.cross_attn = nn.MultiheadAttention(hid_dim, n_heads, batch_first=True)

        # decoder LSTM: takes [feat + context_summary]
        self.dec_lstm = nn.LSTM(feat_dim + hid_dim, hid_dim, batch_first=True)

        # Gaussian heads
        self.mu_head   = nn.Linear(hid_dim, 1)
        self.logsd_head= nn.Linear(hid_dim, 1)

        # PTP: map (mu,sd)→ position
        self.ptp = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1),
            nn.Tanh()
        )

    def encode(self, x):
        # x: [B, T, D] → [B, T, H]
        h, _ = self.enc_lstm(x)
        return h

    def forward(self, x_tgt, x_ctx):
        """
        x_tgt: [B, Lt, D]
        x_ctx: [B, C, Lc, D]
        """
        B, Lt, _ = x_tgt.shape
        _, C, Lc, _ = x_ctx.shape

        # 1) encode contexts
        x_ctx = x_ctx.view(B*C, Lc, -1)
        h_ctx = self.encode(x_ctx)                    # [B*C, Lc, H]
        h_ctx,_ = self.self_attn(h_ctx, h_ctx, h_ctx) # [B*C, Lc, H]
        h_ctx = h_ctx[:, -1]                          # [B*C, H]
        h_ctx = h_ctx.view(B, C, -1)                  # [B, C, H]

        # 2) encode target for cross-attn query
        h_tgt = self.encode(x_tgt)                    # [B, Lt, H]
        q     = self.q_proj(h_tgt)                    # [B, Lt, H]
        k     = self.k_proj(h_ctx)                    # [B,  C, H]
        v     = self.v_proj(h_ctx)                    # [B,  C, H]
        # cross-attend for each time-step:
        # MultiheadAttention expects shape [B, S, H]
        ctx_sum, _ = self.cross_attn(q, k, v)         # [B, Lt, H]

        # 3) decode
        dec_in = torch.cat([x_tgt, ctx_sum], dim=-1)  # [B, Lt, D+H]
        h_dec, _ = self.dec_lstm(dec_in)              # [B, Lt, H]
        h_last   = h_dec[:, -1]                       # [B, H]

        # 4) predictive Gaussian
        mu    = self.mu_head(h_last).squeeze(-1)      # [B]
        logsd = self.logsd_head(h_last).squeeze(-1)   # [B]
        sd    = torch.exp(logsd)

        # 5) PTP
        inp   = torch.stack([mu, sd], dim=1)          # [B,2]
        pos   = self.ptp(inp).squeeze(-1)             # [B]

        return mu, sd, pos
