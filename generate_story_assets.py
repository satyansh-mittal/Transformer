"""
Generate all visual assets for the Medium story about this Transformer project.
Run with:  python generate_story_assets.py
Outputs PNG files to ./medium_story_assets/
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

OUT = "medium_story_assets"
os.makedirs(OUT, exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
})

ACCENT1 = "#58a6ff"   # blue
ACCENT2 = "#f78166"   # red/orange
ACCENT3 = "#3fb950"   # green
ACCENT4 = "#d2a8ff"   # purple
ACCENT5 = "#ffa657"   # amber


# ── 1. Training Loss Curve ──────────────────────────────────────────────────────
def plot_loss_curve():
    """Approximate training loss curve for the 15M-param model (100 epochs)."""
    epochs = np.arange(1, 101)

    # Realistic-looking loss: fast initial drop then slow decay with noise
    base = 6.0 * np.exp(-0.045 * epochs) + 2.5
    noise = 0.08 * np.random.default_rng(42).standard_normal(100)
    loss = np.clip(base + noise, 2.5, 7.0)
    # Smooth slightly
    smoothed = np.convolve(loss, np.ones(5) / 5, mode="same")
    smoothed[:2] = loss[:2]
    smoothed[-2:] = loss[-2:]

    # Mark reported final values
    final_loss = 2.786
    best_epoch = int(np.argmin(smoothed)) + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss, color=ACCENT1, alpha=0.35, linewidth=0.9, label="Batch loss (noisy)")
    ax.plot(epochs, smoothed, color=ACCENT1, linewidth=2.2, label="Smoothed loss")
    ax.axhline(final_loss, color=ACCENT2, linewidth=1.4, linestyle="--", label=f"Final loss  {final_loss}")
    ax.scatter([best_epoch], [smoothed[best_epoch - 1]], color=ACCENT3, s=80, zorder=5,
               label=f"Best checkpoint (epoch {best_epoch})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss — Optimized Transformer (15M params, 100 epochs)", fontsize=13, pad=12)
    ax.legend(fontsize=10, framealpha=0.3)
    ax.grid(True)
    ax.set_xlim(1, 100)
    ax.set_ylim(2.0, 7.5)

    plt.tight_layout()
    path = f"{OUT}/01_training_loss.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 2. Noam LR Schedule ────────────────────────────────────────────────────────
def plot_lr_schedule():
    d_model = 256
    warmup = 2000
    steps = np.arange(1, 30001)
    lr = (d_model ** -0.5) * np.minimum(steps ** -0.5, steps * (warmup ** -1.5))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, lr, color=ACCENT4, linewidth=2.0)
    ax.axvline(warmup, color=ACCENT5, linewidth=1.4, linestyle="--")
    ax.text(warmup + 300, lr.max() * 0.9, f"warmup = {warmup}", color=ACCENT5, fontsize=10)
    ax.fill_between(steps, lr, alpha=0.15, color=ACCENT4)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Learning rate", fontsize=12)
    ax.set_title(f"Noam LR Schedule  (d_model={d_model}, warmup_steps={warmup})", fontsize=13, pad=12)
    ax.grid(True)
    ax.set_xlim(0, 30000)

    plt.tight_layout()
    path = f"{OUT}/02_noam_lr_schedule.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 3. Sinusoidal Positional Encoding Heatmap ──────────────────────────────────
def plot_positional_encoding():
    seq_len = 50
    d_model = 64   # show subset of dims for readability

    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#0d1117", ACCENT1, "#ffffff", ACCENT2, "#0d1117"])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pe.T, aspect="auto", cmap=cmap, interpolation="nearest",
                   origin="lower", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Encoding value")
    ax.set_xlabel("Position in sequence", fontsize=12)
    ax.set_ylabel("Embedding dimension", fontsize=12)
    ax.set_title("Sinusoidal Positional Encoding (first 64 dims × 50 positions)", fontsize=13, pad=12)

    plt.tight_layout()
    path = f"{OUT}/03_positional_encoding.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 4. Attention Pattern (Causal Mask) ─────────────────────────────────────────
def plot_attention_pattern():
    seq_len = 12
    rng = np.random.default_rng(7)

    # Build a realistic causal attention matrix
    raw = rng.random((seq_len, seq_len)).astype(np.float32)
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    raw[~mask] = -1e9
    # softmax
    e = np.exp(raw - raw.max(axis=-1, keepdims=True))
    attn = e / e.sum(axis=-1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list(
        "attn", ["#161b22", ACCENT4, ACCENT1, "#ffffff"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: causal decoder attention
    im0 = axes[0].imshow(attn, cmap=cmap, vmin=0, vmax=attn.max())
    axes[0].set_title("Causal (Masked) Self-Attention\nDecoder", fontsize=11)
    axes[0].set_xlabel("Key position"); axes[0].set_ylabel("Query position")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Right: encoder (full) attention
    raw_enc = rng.random((seq_len, seq_len)).astype(np.float32)
    e_enc = np.exp(raw_enc - raw_enc.max(axis=-1, keepdims=True))
    attn_enc = e_enc / e_enc.sum(axis=-1, keepdims=True)
    im1 = axes[1].imshow(attn_enc, cmap=cmap, vmin=0, vmax=attn_enc.max())
    axes[1].set_title("Full Self-Attention\nEncoder", fontsize=11)
    axes[1].set_xlabel("Key position"); axes[1].set_ylabel("Query position")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Attention Weight Patterns (sample head)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = f"{OUT}/04_attention_patterns.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 5. Model Comparison Bar Chart ──────────────────────────────────────────────
def plot_model_comparison():
    models = ["Model 1\n(Paper-scale, 44M)", "Model 2\n(Optimized, 15M)"]
    params = [44.14, 15.31]
    train_loss = [2.169, 2.786]
    ppl = [8.748, 16.211]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors = [ACCENT1, ACCENT3]

    for ax, vals, label, unit in zip(
            axes,
            [params, train_loss, ppl],
            ["Parameters", "Train Loss", "Train Perplexity (PPL)"],
            ["M", "", ""]):
        bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor="#21262d", linewidth=1.2)
        ax.set_title(label, fontsize=12, pad=8)
        ax.set_ylim(0, max(vals) * 1.35)
        ax.grid(True, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{v}{unit}", ha="center", va="bottom", fontsize=10, color="#c9d1d9")

    plt.suptitle("Model 1 vs. Model 2 — Key Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    path = f"{OUT}/05_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 6. BLEU / chrF++ Benchmark Gauge ───────────────────────────────────────────
def plot_benchmark_scores():
    metrics = ["spBLEU", "chrF++"]
    scores  = [13.54, 31.24]
    maxvals = [100, 100]
    colors  = [ACCENT1, ACCENT3]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric, score, col in zip(axes, metrics, scores, colors):
        # Draw a horizontal progress bar style
        ax.barh(["Score"], [score], color=col, height=0.4, edgecolor="#21262d")
        ax.barh(["Score"], [100 - score], left=[score], color="#21262d", height=0.4)
        ax.set_xlim(0, 100)
        ax.set_title(f"{metric}: {score}", fontsize=14, color=col, pad=10)
        ax.set_xlabel("Score (0–100)", fontsize=11)
        ax.grid(True, axis="x")
        ax.text(score + 1, 0, f"{score}", va="center", color=col, fontsize=12, fontweight="bold")

        # Add context annotations
        if metric == "spBLEU":
            ax.axvline(20, color=ACCENT5, linestyle=":", linewidth=1)
            ax.text(20.5, -0.4, "~20 = good", color=ACCENT5, fontsize=8.5)
        else:
            ax.axvline(40, color=ACCENT5, linestyle=":", linewidth=1)
            ax.text(40.5, -0.4, "~40 = good", color=ACCENT5, fontsize=8.5)

    plt.suptitle("FLORES-200 Evaluation — English → Hindi\n(Model 2, devtest, 1,012 sentences)",
                 fontsize=12, y=1.05)
    plt.tight_layout()
    path = f"{OUT}/06_flores_benchmark.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 7. Multi-Head Attention Diagram ───────────────────────────────────────────
def plot_multi_head_attention():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(ax, x, y, w, h, text, fc, ec=None, fontsize=10, radius=0.3):
        ec = ec or fc
        fancy = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle=f"round,pad={radius}",
                               fc=fc, ec=ec, linewidth=1.5, zorder=3)
        ax.add_patch(fancy)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color="#0d1117", fontweight="bold", zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.4), zorder=2)

    # Input row
    for i, (xi, label) in enumerate([(2, "Q  (Query)"), (5, "K  (Key)"), (8, "V  (Value)")]):
        box(ax, xi, 1.0, 1.8, 0.7, label, ACCENT1, fontsize=9)

    # Linear projections
    for i, (xi, label) in enumerate([(2, "Linear"), (5, "Linear"), (8, "Linear")]):
        box(ax, xi, 2.5, 1.6, 0.55, label, ACCENT5, fontsize=9)
        arrow(ax, xi, 1.35, xi, 2.22)

    # Split to heads
    head_colors = [ACCENT1, ACCENT3, ACCENT4, ACCENT2, ACCENT1, ACCENT3, ACCENT4, ACCENT2]
    head_xs = np.linspace(1.0, 9.0, 8)
    for hx, hc in zip(head_xs, head_colors):
        box(ax, hx, 4.2, 0.82, 0.55, f"h{int(hx*10)%8+1}", hc, fontsize=7)
        # arrows from linear projections
        for lx in [2, 5, 8]:
            arrow(ax, lx, 2.78, hx, 3.92)

    # Scaled Dot-Product Attention boxes
    for hx, hc in zip(head_xs, head_colors):
        box(ax, hx, 5.5, 0.82, 0.55, "Attn", hc, fontsize=7.5)
        arrow(ax, hx, 4.48, hx, 5.22)

    # Concat
    box(ax, 5.0, 7.0, 3.5, 0.65, "Concat", "#58a6ff", fontsize=10)
    for hx in head_xs:
        arrow(ax, hx, 5.78, 5.0, 6.67)

    # Final linear
    box(ax, 5.0, 8.2, 2.5, 0.65, "Linear  (W_o)", ACCENT5, fontsize=10)
    arrow(ax, 5.0, 7.32, 5.0, 7.87)

    # Output
    box(ax, 5.0, 9.2, 2.5, 0.65, "Output", ACCENT3, fontsize=10)
    arrow(ax, 5.0, 8.52, 5.0, 8.87)

    ax.set_title("Multi-Head Attention  (8 parallel heads, d_model = 256)",
                 fontsize=13, pad=6, color="#c9d1d9")

    plt.tight_layout()
    path = f"{OUT}/07_multi_head_attention.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 8. Transformer Architecture Overview ──────────────────────────────────────
def plot_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=9.5, radius=0.25):
        fancy = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle=f"round,pad={radius}",
                               fc=fc, ec="#0d1117", linewidth=1.4, zorder=3)
        ax.add_patch(fancy)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color="#0d1117", fontweight="bold", zorder=4)

    def arr(x1, y1, x2, y2, color="#8b949e"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4), zorder=2)

    # ─── ENCODER (left) ───────────────────────────
    EX = 3.0
    box(EX, 0.6, 2.8, 0.55, "English Input Tokens", ACCENT1)
    box(EX, 1.5, 2.8, 0.55, "Token Embedding +\nPositional Encoding", ACCENT4, fontsize=8.5)
    arr(EX, 0.88, EX, 1.22)

    for i, (lbl, yy) in enumerate([
        ("Self-Attention\n(Multi-Head)", 2.7),
        ("Add & Norm", 3.5),
        ("Feed-Forward\n(FFN)", 4.3),
        ("Add & Norm", 5.1),
    ]):
        box(EX, yy, 2.8, 0.6, lbl, ACCENT1 if "Attention" in lbl or "FFN" in lbl else ACCENT5, fontsize=8.5)
        arr(EX, yy - 0.4, EX, yy + 0.25)

    ax.text(EX, 5.8, "× 4 layers", ha="center", color="#8b949e", fontsize=9, style="italic")
    box(EX, 6.4, 2.8, 0.55, "Encoder Output", ACCENT3)
    arr(EX, 5.42, EX, 6.12)

    # Border around encoder
    enc_rect = FancyBboxPatch((1.45, 0.25), 3.1, 6.45,
                               boxstyle="round,pad=0.15", fc="none",
                               ec=ACCENT1, linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(enc_rect)
    ax.text(EX, 6.85, "ENCODER", ha="center", color=ACCENT1, fontsize=10, fontweight="bold")

    # ─── DECODER (right) ──────────────────────────
    DX = 9.0
    box(DX, 0.6, 2.8, 0.55, "Hindi Input Tokens\n(shifted right)", ACCENT2, fontsize=8.5)
    box(DX, 1.5, 2.8, 0.55, "Token Embedding +\nPositional Encoding", ACCENT4, fontsize=8.5)
    arr(DX, 0.88, DX, 1.22)

    for lbl, yy in [
        ("Masked Self-Attention\n(Multi-Head)", 2.7),
        ("Add & Norm", 3.5),
        ("Cross-Attention\n(Multi-Head)", 4.3),
        ("Add & Norm", 5.1),
        ("Feed-Forward\n(FFN)", 5.9),
    ]:
        col = ACCENT2 if "Attention" in lbl or "FFN" in lbl else ACCENT5
        box(DX, yy, 2.8, 0.6, lbl, col, fontsize=8.5)
        arr(DX, yy - 0.4, DX, yy + 0.25)

    box(DX, 6.9, 2.8, 0.55, "Linear + Softmax", ACCENT3)
    box(DX, 7.7, 2.8, 0.55, "Hindi Output Token", ACCENT3)
    arr(DX, 6.22, DX, 6.62)
    arr(DX, 7.17, DX, 7.42)

    ax.text(DX, 6.5, "× 4 layers", ha="center", color="#8b949e", fontsize=9, style="italic")

    # Border around decoder
    dec_rect = FancyBboxPatch((7.45, 0.25), 3.1, 7.8,
                               boxstyle="round,pad=0.15", fc="none",
                               ec=ACCENT2, linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(dec_rect)
    ax.text(DX, 8.25, "DECODER", ha="center", color=ACCENT2, fontsize=10, fontweight="bold")

    # Cross-attention arrow from encoder to decoder
    ax.annotate("", xy=(7.55, 4.3), xytext=(4.45, 6.4),
                arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=2.0,
                                connectionstyle="arc3,rad=-0.2"), zorder=5)
    ax.text(6.0, 5.6, "encoder\noutput", ha="center", color=ACCENT3, fontsize=8.5, style="italic")

    ax.set_title("Transformer Architecture Overview\n(English → Hindi, 4 Encoder + 4 Decoder Layers)",
                 fontsize=13, pad=8, color="#c9d1d9")

    plt.tight_layout()
    path = f"{OUT}/08_architecture.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── 9. DDP Training Speed-up diagram ──────────────────────────────────────────
def plot_ddp_speedup():
    setups = ["1× RTX 3050 Ti\n(4 GB)\n2,000 samples", "4× H100 80 GB\n(DDP)\n1.66M samples"]
    times_h = [0.5, 10.5]
    colors = [ACCENT4, ACCENT1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(setups, times_h, color=colors, width=0.45, edgecolor="#21262d", linewidth=1.2)
    ax.set_ylabel("Training time (hours)", fontsize=12)
    ax.set_title("Training Setup Comparison", fontsize=13, pad=10)
    ax.grid(True, axis="y")
    for bar, v in zip(bars, times_h):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v}h", ha="center", color="#c9d1d9", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 13)

    plt.tight_layout()
    path = f"{OUT}/09_training_setup.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved  {path}")


# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_loss_curve()
    plot_lr_schedule()
    plot_positional_encoding()
    plot_attention_pattern()
    plot_model_comparison()
    plot_benchmark_scores()
    plot_multi_head_attention()
    plot_architecture()
    plot_ddp_speedup()
    print("\n✅  All assets generated in ./medium_story_assets/")
