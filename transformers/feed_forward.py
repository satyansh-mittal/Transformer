import torch.nn as nn
from utils import load_config

config = load_config()
d_model = config["d_model"]
heads = config["heads"]
dropout = config["dropout"]


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_dim=None, dropout=0.1):
        super().__init__()
        if inner_dim is None:
            inner_dim = 4 * d_model
        self.d_model = d_model
        self.inner_dim = inner_dim
        self.l1 = nn.Linear(d_model, inner_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.l2(self.relu(self.l1(x))))