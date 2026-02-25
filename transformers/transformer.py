import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from encoder import Encoder
from decoder import Decoder
from utils import load_config

config = load_config()
d_model = config["d_model"]
heads = config["heads"]
dropout = config["dropout"]


class Transformer(nn.Module):
    def __init__(self, d_model, heads, dropout, vocab_size, num_layers, embedding=None):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.inner_dim = 4*self.d_model
        self.num_layers = num_layers
        
        self.encoder_layers = nn.ModuleList([
            Encoder(self.d_model, self.heads, self.inner_dim, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(self.d_model, self.heads, self.inner_dim, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
        self.linear = nn.Linear(self.d_model, self.vocab_size, bias=True)
        
        if embedding is not None:  # tying weights of linear with embedding
            assert tuple(embedding.weight.shape) == (self.vocab_size, self.d_model), \
                f"Embedding weight shape {tuple(embedding.weight.shape)} doesn't match (vocab_size, d_model)"
            
            self.linear.weight = embedding.weight
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
            
    def forward(self, x_encoder, x_decoder, tgt_mask=None, src_mask=None):
        
        # Encoder stack
        x = x_encoder
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        encoder_output = x
        
        # decoder stack
        y = x_decoder
        for layer in self.decoder_layers:
            y = layer(y, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
        decoder_output = y
        
        logits = self.linear(decoder_output)
        
        return logits
