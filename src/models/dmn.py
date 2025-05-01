import torch
import torch.nn as nn

class DMN(nn.Module):
    """
    Baseline Deep Momentum Network:
    simple LSTM → tanh position head, Sharpe loss outside.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [B, T, D]
        h, _ = self.lstm(x)         # [B, T, H]
        z    = self.head(h[:, -1])  # [B,1]
        return z.squeeze(-1)        # [B]
