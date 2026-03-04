# 🔄 Transformer: Attention Is All You Need

A from-scratch PyTorch implementation of the **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** paper for **English → Hindi** machine translation, trained on the [IITB English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi).

---

## 🏗️ Architecture

The model follows the original Transformer architecture from the paper:

```
Input (English) ──► Embedding + Positional Encoding ──► Encoder Stack ──┐
                                                                        │
Input (Hindi)   ──► Embedding + Positional Encoding ──► Decoder Stack ◄─┘
                                                            │
                                                      Linear + Softmax ──► Output (Hindi)
```

**Key Components:**

- **Scaled Dot-Product Attention** with `Q`, `K`, `V` projections
- **Multi-Head Attention** — Parallel attention heads (`d_k = d_model / heads`)
- **Position-wise Feed-Forward Networks** — Two linear layers with ReLU
- **Sinusoidal Positional Encoding**
- **Post-Layer Normalization**
- **Weight Tying** — Embedding weights shared with the final linear projection
- **Noam LR Scheduler** — Warmup-based learning rate schedule from the paper
- **Label Smoothing** — Cross-entropy loss with `label_smoothing=0.1`

---

## 📁 Project Structure

```
Transformers/
├── README.md
├── code.ipynb                  # Model 1: Paper-scale Transformer (44M params)
└── transformers/               # Model 2: Optimized Transformer (15M params)
    ├── main.py                 # Entry point — train / inference / checkpoint loading
    ├── train.py                # Training loop with DDP support
    ├── inference.py            # Greedy decoding for translation
    ├── transformer.py          # Transformer model (Encoder + Decoder + Linear head)
    ├── encoder.py              # Encoder layer (Self-Attention + FFN + LayerNorm)
    ├── decoder.py              # Decoder layer (Self-Attn + Cross-Attn + FFN + LayerNorm)
    ├── attention.py            # Scaled Dot-Product & Multi-Head Attention
    ├── feed_forward.py         # Position-wise Feed-Forward Network
    ├── lr_scheduler.py         # Noam LR scheduler
    ├── data.py                 # Dataset loading & collation (HuggingFace datasets)
    ├── utils.py                # Vocabulary, tokenizer, positional encoding, masks
    ├── config.json             # Hyperparameters
    ├── model_info.sh           # Script to print model architecture summary
    ├── model_summary.txt       # Saved output of model_info.sh
    ├── requirements.txt        # Python dependencies
    ├── checkpoint.pt           # Final model checkpoint
    └── checkpoint_best.pt      # Best model checkpoint (lowest loss)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.x with CUDA support
- NVIDIA GPU(s)

### Installation

```bash
cd transformers/
pip install -r requirements.txt
```

### Training

**Single GPU:**

```bash
python main.py
# Select option 1: Train a new model
```

**Multi-GPU (Distributed Data Parallel):**

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 main.py
```

### Inference

```bash
python main.py
# Select option 2: Load checkpoint and run inference
# Or option 3: Inference only
```

---

## 🧪 Trained Models

Two Transformer models were trained on the IITB English-Hindi dataset.

---

### Model 1 — Paper-Scale Transformer (`code.ipynb`)

This model replicates the exact architecture described in the original paper — 6 encoder layers, 6 decoder layers, `d_model=512`, 8 attention heads — resulting in **44.14 million parameters**. It was trained on a small subset of 2,000 samples as a proof-of-concept.

|                              |                          |
| ---------------------------- | ------------------------ |
| **Parameters**               | 44,143,626 (44.14M)      |
| **d_model**                  | 512                      |
| **Heads**                    | 8                        |
| **Encoder / Decoder Layers** | 6 / 6                    |
| **FFN Inner Dim**            | 2048                     |
| **Dataset**                  | 2,000 samples            |
| **GPU**                      | NVIDIA RTX 3050 Ti (4GB) |
| **Training Time**            | ~30 minutes              |
| **Train Loss**               | 2.169                    |
| **Train PPL**                | 8.748                    |

> The predictions from this model were not good as the dataset was too small (2,000 samples) for a 44M parameter model to learn meaningful translations.

<details>
<summary>📋 Layer-by-layer architecture (click to expand)</summary>

```
Transformer                          [1, 3, 512]    →  [1, 3, 10]
├─ Encoder × 6
│   ├─ MultiHeadAttention            [1, 3, 512]    →  [1, 3, 512]     1,050,624 params
│   ├─ LayerNorm                     [1, 3, 512]    →  [1, 3, 512]         1,024 params
│   ├─ FeedForward                   [1, 3, 512]    →  [1, 3, 512]     2,099,712 params
│   └─ LayerNorm                     [1, 3, 512]    →  [1, 3, 512]         1,024 params
├─ Decoder × 6
│   ├─ MultiHeadAttention (self)     [1, 3, 512]    →  [1, 3, 512]     1,050,624 params
│   ├─ LayerNorm                     [1, 3, 512]    →  [1, 3, 512]         1,024 params
│   ├─ MultiHeadAttention (cross)    [1, 3, 512]    →  [1, 3, 512]     1,050,624 params
│   ├─ LayerNorm                     [1, 3, 512]    →  [1, 3, 512]         1,024 params
│   ├─ FeedForward                   [1, 3, 512]    →  [1, 3, 512]     2,099,712 params
│   └─ LayerNorm                     [1, 3, 512]    →  [1, 3, 512]         1,024 params
└─ Linear                           [1, 3, 512]    →  [1, 3, 10]          5,130 params

Total params: 44,143,626
```

</details>

---

### Model 2 — Optimized Transformer (`transformers/`)

This model uses an optimized architecture — 4 encoder layers, 4 decoder layers, `d_model=256`, 8 attention heads — totaling **15.31 million parameters**. It was trained on the full IITB dataset of 1.66 million sentence pairs using PyTorch Distributed Data Parallel (DDP) across 4 NVIDIA H100 80GB GPUs. This model produces satisfiable English → Hindi translations.

|                              |                          |
| ---------------------------- | ------------------------ |
| **Parameters**               | 15,310,480 (15.31M)      |
| **d_model**                  | 256                      |
| **Heads**                    | 8                        |
| **Encoder / Decoder Layers** | 4 / 4                    |
| **FFN Inner Dim**            | 1024                     |
| **Vocab Size**               | 30,000 (capped)          |
| **Max Sequence Length**      | 50                       |
| **Dataset**                  | 1,659,083 sentence pairs |
| **Effective Batch Size**     | 1,024 (256 × 4 GPUs)     |
| **Training Time**            | ~10.5 hours (100 epochs) |
| **Train Loss**               | 2.786                    |
| **Train PPL**                | 16.211                   |

### Evaluation Benchmark (FLORES-200 `devtest`)

The optimized model was evaluated against the **FLORES-200** machine translation benchmark (1,012 English → Hindi test sentences) with a maximum generation length of 100.

| Metric     | Score |
| ---------- | ----- |
| **spBLEU** | 13.54 |
| **chrF++** | 31.24 |

---

## 📚 Dataset

**[IITB English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi)**

- **Source:** IIT Bombay
- **Training samples:** 1,659,083 sentence pairs
- **Languages:** English → Hindi

---

## 📝 References

- Vaswani, A., et al. _"Attention Is All You Need"_. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Anoop Kunchukuttan, Pratik Mehta, and Pushpak Bhattacharyya. _"The IIT Bombay English-Hindi Parallel Corpus"_. LREC 2018.

---

## 📄 License

This project is for educational and research purposes.
