"""
FLORES-200 Benchmark Script for your English → Hindi Transformer
- Uses facebook/flores (eng_Latn → hin_Deva)
- Reuses your exact translate() and load_checkpoint()
- Computes spBLEU, chrF++, and COMET-22 (optional but recommended)
- Saves full report to flores_benchmark.txt
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
import sacrebleu
import torch

# Import directly from your existing code
from main import load_checkpoint
from inference import translate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt",
                        help="Path to your checkpoint.pt")
    parser.add_argument("--split", type=str, default="devtest", choices=["dev", "devtest"],
                        help="FLORES split to use (devtest is the standard one)")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Max generation length")
    parser.add_argument("--output", type=str, default="flores_benchmark.txt")
    args = parser.parse_args()

    # ==================== LOAD MODEL ====================
    print("Loading model from checkpoint...")
    model, embedding, vocab, id2word, device = load_checkpoint(args.checkpoint)
    model.eval()
    print(f"Model loaded! (vocab size: {len(vocab)}, d_model: {model.d_model}, layers: {model.num_layers})")

    # ==================== LOAD FLORES-200 ====================
    print(f"\nLoading FLORES-200 {args.split} (English → Hindi)...")
    
    import urllib.request
    import tarfile
    tar_path = "flores200_dataset.tar.gz"
    if not os.path.exists(tar_path):
        print("Downloading FLORES-200 dataset...")
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", tar_path)
    
    with tarfile.open(tar_path, "r:gz") as tar:
        eng_file = tar.extractfile(f"./flores200_dataset/{args.split}/eng_Latn.{args.split}")
        hin_file = tar.extractfile(f"./flores200_dataset/{args.split}/hin_Deva.{args.split}")
        if eng_file is None or hin_file is None:
            raise RuntimeError("Could not find required language files in the tarball.")
        sources = [line.decode("utf-8").strip() for line in eng_file]
        references = [line.decode("utf-8").strip() for line in hin_file]

    print(f"Translating {len(sources):,} sentences...\n")

    # ==================== INFERENCE (reuse your translate) ====================
    predictions = []
    for src in tqdm(sources, desc="Translating"):
        pred = translate(
            src_sentence=src,
            model=model,
            embedding=embedding,
            vocab=vocab,
            id2word=id2word,
            device=device,
            max_len=args.max_len
        )
        predictions.append(pred)

    # ==================== METRICS ====================
    print("\nComputing metrics...")

    # spBLEU (the official FLORES metric)
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="spm")
    # chrF++
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    print(f"spBLEU  : {bleu.score:.2f}")
    print(f"chrF++  : {chrf.score:.2f}")

    # COMET-22 (highest correlation with human judgment)
    comet_score = None
    try:
        from comet import download_model, load_from_checkpoint
        print("Downloading COMET-22 model (one-time)...")
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)

        comet_data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(sources, predictions, references)
        ]
        result = comet_model.predict(comet_data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
        # Handle both older object-based and newer dict-based returns
        comet_score = result.mean_score if hasattr(result, "mean_score") else result["mean_score"]
        print(f"COMET-22: {comet_score:.4f}")
    except Exception as e:
        print(f"COMET skipped (optional): {e}")

    # ==================== SAVE REPORT ====================
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("=== FLORES-200 BENCHMARK (English → Hindi) ===\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint       : {os.path.abspath(args.checkpoint)}\n")
        f.write(f"Split            : {args.split}\n")
        f.write(f"Sentences        : {len(sources):,}\n")
        f.write(f"d_model          : {model.d_model}\n")
        f.write(f"Layers           : {model.num_layers}\n")
        f.write(f"Vocab size       : {len(vocab):,}\n\n")

        f.write(f"spBLEU           : {bleu.score:.2f}\n")
        f.write(f"chrF++           : {chrf.score:.2f}\n")
        if comet_score is not None:
            f.write(f"COMET-22         : {comet_score:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("SAMPLE TRANSLATIONS (first 10)\n")
        f.write("=" * 60 + "\n\n")

        for i in range(min(10, len(sources))):
            f.write(f"English   : {sources[i]}\n")
            f.write(f"Reference : {references[i]}\n")
            f.write(f"Model     : {predictions[i]}\n")
            f.write("-" * 80 + "\n")

    print(f"\n✅ Benchmark complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()