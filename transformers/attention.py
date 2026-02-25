import torch
import torch.nn as nn
import math
from utils import load_config

config = load_config()
d_model = config["d_model"]
heads = config["heads"]
dropout = config["dropout"]

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, d_k):
#         super().__init__()
# W_Q = torch.randn(x.shape[2], x.shape[2])   # (512, 512)
# W_K = torch.randn(x.shape[2], x.shape[2])
# W_V = torch.randn(x.shape[2], x.shape[2])

# Q = x @ W_Q  # (1, 3, 512)
# K = x @ W_K
# V = x @ W_V

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        attention = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if (mask is not None):
            attention = attention.masked_fill(~mask, -1e9)
        
        attention_weights = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads    # 512//8 = 64 because they are concatenated later
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)
        self.z = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # Linear
        Q = self.W_Q(q)   # (1, 3, 512)
        K = self.W_K(k)
        V = self.W_V(v)
        
        batch_size, tgt_len, _ = Q.shape
        _, src_len, _ = K.shape
        # heads
        Q = Q.view(batch_size, tgt_len, self.heads, self.d_k)   # (1, 3, 8, 64)
        K = K.view(batch_size, src_len, self.heads, self.d_k)
        V = V.view(batch_size, src_len, self.heads, self.d_k)
        
        Q = Q.transpose(1,2)    # (1, 8, 3, 64)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        # attention
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)    # (1, 1, tgt_len, src_len) = (1, 1, 3, 3)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)     # (1, 1, 3, 3)
            elif mask.dim() == 4:
                pass
            mask = mask.to(torch.bool)
                
                
        output, attention_weights = self.attention(Q, K, V, mask=mask)    # (1, 8, 3, 64) ; (1, 8, 3, 3)
        # concat
        concat_output = output.transpose(1,2)     # (1, 3, 8, 64)
        concat_output = concat_output.contiguous().view(batch_size, tgt_len, self.d_model)   # (1, 3, 512)
        # linear
        output = self.z(concat_output)
        
        output = self.dropout(output)
        
        return output