import json
import torch
import math
import os

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

config = load_config()
SPECIAL_TOKENS = config["SPECIAL_TOKENS"]
d_model = config["d_model"]

from collections import Counter

def build_vocab(sentences, max_size=30000):
    vocab = dict(SPECIAL_TOKENS)
    idx = len(vocab)
    
    counter = Counter()
    for s in sentences:
        tokens = s.lower().split()
        counter.update(tokens)
        
    for token, _ in counter.most_common(max_size - len(SPECIAL_TOKENS)):
        if token not in vocab:
            vocab[token] = idx
            idx += 1
            
    return vocab

def tokenize(sentence, vocab):
    tokens = sentence.lower().split()
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    input_ids = torch.tensor(ids).unsqueeze(0)  # (3,) -> (1,3)
    return input_ids

def positionalEncoding(input_embeddings, d_model=512):
    seq_length = input_embeddings.shape[1] if input_embeddings.dim() == 3 else input_embeddings.shape[0]   # 3
    positional_encoding = torch.zeros(seq_length, d_model, device=input_embeddings.device)
    
    for pos in range(seq_length):
        for i in range(0, d_model, 2):  # [sin, cos, sin, cos , ...,sin, cos] for 512 dimensions
            PE_sin = math.sin(pos / 10000**(2*i/d_model))
            PE_cos = math.cos(pos / 10000**(2*i/d_model))
            positional_encoding[pos, i] = PE_sin
            positional_encoding[pos, i+1] = PE_cos

    return positional_encoding.unsqueeze(0)

def positionalEncoding2(input_embeddings, d_model=512):
    """
    x: (B, seq_len, d_model)
    Returns: (1, seq_len, d_model) on same device as x
    """
    seq_len = input_embeddings.shape[1]
    device = input_embeddings.device
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                         (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)  # (1, seq_len, d_model)

def make_padding_mask(input_ids, pad_idx=0):
    # (b, l) -> (b, 1, 1, l)
    return (input_ids != pad_idx).unsqueeze(1).unsqueeze(1)

def make_causal_mask(tgt_len, device):  # (1, tgt_len, tgt_len)
    return torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device)).unsqueeze(0)

def combine_padding_and_causal(tgt_pad_mask, causal_mask):    # tgt_pad_mask: (B, 1, 1, L) -> convert to (B, L) valid positions
    batch_size = tgt_pad_mask.shape[0]
    L = tgt_pad_mask.shape[-1]
    valid = tgt_pad_mask.squeeze(1).squeeze(1)   # (batch_size, L)
    causal_b = causal_mask.expand(batch_size, -1, -1)   # (batch_size, L, L)
    valid_src = valid.unsqueeze(1).expand(-1, L, -1)   # (batch_size, L, L)
    combined = causal_b & valid_src
    return combined.unsqueeze(1)  # (batch, 1, L, L)

def shift_right(target_ids, vocab):
    batch_size = target_ids.shape[0]
    sos = vocab['<sos>']
    sos_tensor = torch.full((batch_size, 1), sos, device=target_ids.device, dtype=torch.long)
    return torch.cat([sos_tensor, target_ids[:, :-1]], dim=1)