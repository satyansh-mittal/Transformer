import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from utils import load_config

config = load_config()
d_model = config["d_model"]
heads = config["heads"]
dropout = config["dropout"]


class Encoder(nn.Module):
    def __init__(self, d_model, heads, inner_dim, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.heads = heads
        self.inner_dim = inner_dim

        self.attention = MultiHeadAttention(d_model=self.d_model, heads=self.heads)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.ffn = FeedForward(self.d_model, inner_dim=4*self.d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        y1 = self.attention(x, x, x, mask=src_mask)
        x = x + y1
        x = self.layer_norm1(x)
        
        y2 = self.ffn(x)
        x = x + y2
        encoder_output = self.layer_norm2(x)
        
        return encoder_output
