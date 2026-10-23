import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, channel, emb, hidden_dim,mlp_hidden_dim: int = None,dropout = 0.0):
        super().__init__()

        input_dim = channel * emb
        mlp_hidden_dim = mlp_hidden_dim or input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x: [..., channel, emb]
        x = x.reshape(*x.shape[:-2], -1)  # flatten channel * emb
        x = self.encoder(x)
        return x
