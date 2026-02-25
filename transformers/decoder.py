import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from utils import load_config

config = load_config()
d_model = config["d_model"]
heads = config["heads"]
dropout = config["dropout"]


class Decoder(nn.Module):
    def __init__(self, d_model, heads, inner_dim=None, dropout=0.1):
        super().__init__()
        if inner_dim is None:
            inner_dim = 4 * d_model
        self.d_model = d_model
        self.heads = heads
        self.inner_dim = inner_dim
        
        self.masked_attention = MultiHeadAttention(d_model=self.d_model, heads=self.heads)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.cross_attention = MultiHeadAttention(self.d_model, self.heads)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.ffn = FeedForward(d_model=self.d_model)
        self.layer_norm3 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        y1 = self.masked_attention(x, x, x, mask=tgt_mask)
        x = x + y1
        x = self.layer_norm1(x)
        
        y2 = self.cross_attention(x, encoder_output, encoder_output, mask=src_mask)
        x = x + y2
        x = self.layer_norm2(x)
        
        y3 = self.ffn(x)
        x = x + y3
        decoder_output = self.layer_norm3(x)
        
        return decoder_output
