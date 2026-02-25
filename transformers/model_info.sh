"""
model_info.py  –  Print a detailed summary of the Transformer model.

Run from inside the transformers/ directory:
    python model_info.sh

If a checkpoint.pt exists it is loaded so the summary reflects the
exact vocab/config that was used during training. Otherwise a dummy
model is built straight from config.json.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torchinfo import summary

from utils import load_config, build_vocab
from transformer import Transformer

# ── Config ──────────────────────────────────────────────────────────────
config     = load_config()
d_model    = config["d_model"]
heads      = config["heads"]
dropout    = config["dropout"]
max_len    = config["max_len"]

CKPT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load or build model ──────────────────────────────────────────────────
if os.path.exists(CKPT_PATH):
    print(f"Loading checkpoint: {CKPT_PATH}\n")
    ckpt       = torch.load(CKPT_PATH, map_location=device)
    vocab      = ckpt["vocab"]
    vocab_size = ckpt["vocab_size"]
    d_model    = ckpt["d_model"]
    heads      = ckpt["heads"]
    dropout    = ckpt["dropout"]
    num_layers = ckpt["num_layers"]
    embedding  = nn.Embedding(vocab_size, d_model)
    model      = Transformer(d_model, heads, dropout,
                             vocab_size=vocab_size,
                             num_layers=num_layers,
                             embedding=embedding)
    model.load_state_dict(ckpt["model_state"])
    embedding.load_state_dict(ckpt["embedding_state"])
    source     = "checkpoint"
else:
    print(f"No checkpoint found at {CKPT_PATH}.")
    print("Building a dummy model from config.json ...\n")
    vocab_size = 10_000          
    num_layers = 6
    embedding  = nn.Embedding(vocab_size, d_model)
    model      = Transformer(d_model, heads, dropout,
                             vocab_size=vocab_size,
                             num_layers=num_layers,
                             embedding=embedding)
    source     = "config.json (dummy)"

model     = model.to(device)
embedding = embedding.to(device)
model.eval()

# ── Manual stats ─────────────────────────────────────────────────────────
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
emb_params       = embedding.weight.numel()

print("=" * 60)
print("  TRANSFORMER MODEL INFORMATION")
print("=" * 60)
print(f"  Source          : {source}")
print(f"  Device          : {device}")
print("-" * 60)
print(f"  d_model         : {d_model}")
print(f"  Heads           : {heads}")
print(f"  d_k (per head)  : {d_model // heads}")
print(f"  FFN inner dim   : {4 * d_model}")
print(f"  Encoder layers  : {num_layers}")
print(f"  Decoder layers  : {num_layers}")
print(f"  Dropout         : {dropout}")
print(f"  Max seq length  : {max_len}")
print(f"  Vocab size      : {vocab_size:,}")
print("-" * 60)
print(f"  Embedding params: {emb_params:>12,}")
print(f"  Total params    : {total_params:>12,}  ({total_params/1e6:.2f} M)")
print(f"  Trainable params: {trainable_params:>12,}  ({trainable_params/1e6:.2f} M)")
print("=" * 60)
print()

# ── torchinfo summary ─────────────────────────────────────────────────────
# Transformer.forward(x_encoder, x_decoder, tgt_mask=None, src_mask=None)
# We pass pre-embedded tensors: (batch=1, seq_len=10, d_model)
BATCH, SEQ = 1, 10

x_enc = torch.randn(BATCH, SEQ, d_model, device=device)
x_dec = torch.randn(BATCH, SEQ, d_model, device=device)

print("torchinfo layer-by-layer summary")
print(f"  Input shape  — x_encoder : {list(x_enc.shape)}")
print(f"                 x_decoder : {list(x_dec.shape)}")
print()

summary(
    model,
    input_data=(x_enc, x_dec),
    col_names=("input_size", "output_size", "num_params", "trainable"),
    col_width=20,
    depth=4,
    device=device,
    verbose=1,
)