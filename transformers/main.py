
import sys
import os
import torch
import torch.distributed as dist

# Ensure the script directory is on the path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import json

from utils import build_vocab, load_config
from transformer import Transformer
from inference import translate


def load_checkpoint(checkpoint_path):
    """Load a saved checkpoint and reconstruct model + embedding + vocab."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab      = checkpoint["vocab"]
    vocab_size = checkpoint["vocab_size"]
    d_model    = checkpoint["d_model"]
    heads      = checkpoint["heads"]
    dropout_p  = checkpoint["dropout"]
    num_layers = checkpoint["num_layers"]
    id2word    = {idx: word for word, idx in vocab.items()}

    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    model = Transformer(d_model, heads, dropout_p, vocab_size=vocab_size,
                        num_layers=num_layers, embedding=embedding)

    model.load_state_dict(checkpoint["model_state"])
    embedding.load_state_dict(checkpoint["embedding_state"])

    model     = model.to(device)
    embedding = embedding.to(device)
    model.eval()

    return model, embedding, vocab, id2word, device


def main():
    # DDP setup
    ddp = False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        ddp = True
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    is_main_process = (local_rank == 0)

    if is_main_process:
        print("=" * 50)
        print("  Transformer (English → Hindi) ")
        print("=" * 50)
        print()
        print("Select an option:")
        print("  1. Train a new model")
        print("  2. Load an existing checkpoint and run inference")
        print("  3. Inference only (enter text)")
        print()
        choice = input("Enter choice (1/2/3): ").strip()
    else:
        choice = "1"  # non-main processes always train
    if ddp:
        dist.barrier()


    if choice == "1":
        #  Train
        from train import train
        if is_main_process:
            print("\nStarting training ...\n")
        model, embedding, vocab, id2word = train(ddp=ddp, local_rank=local_rank)
        if is_main_process:
            print("\nTraining complete.")
            # After training, offer inference
            run_inference = input("\nRun inference now? (y/n): ").strip().lower()
            if run_inference == "y":
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                _inference_loop(model, embedding, vocab, id2word, device)

    elif choice == "2" and is_main_process:
        #   Load checkpoint 
        default_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")
        print(f"\nDefault checkpoint path: {default_ckpt}")
        ckpt_input = input("Enter checkpoint path (or press Enter to use default): ").strip()
        checkpoint_path = ckpt_input if ckpt_input else default_ckpt

        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        print(f"\nLoading checkpoint from: {checkpoint_path}")
        model, embedding, vocab, id2word, device = load_checkpoint(checkpoint_path)
        print("Checkpoint loaded successfully.\n")

        _inference_loop(model, embedding, vocab, id2word, device)

    elif choice == "3" and is_main_process:
        #   Inference only (needs a checkpoint) 
        default_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")
        print(f"\nDefault checkpoint path: {default_ckpt}")
        ckpt_input = input("Enter checkpoint path (or press Enter to use default): ").strip()
        checkpoint_path = ckpt_input if ckpt_input else default_ckpt

        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        print(f"\nLoading checkpoint from: {checkpoint_path}")
        model, embedding, vocab, id2word, device = load_checkpoint(checkpoint_path)
        print("Checkpoint loaded successfully.\n")

        _inference_loop(model, embedding, vocab, id2word, device)

    elif is_main_process:
        print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")
        sys.exit(1)


def _inference_loop(model, embedding, vocab, id2word, device):
    print("\nEnter an English sentence to translate to Hindi.")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        src_sentence = input("English: ").strip()
        if src_sentence.lower() in ("quit", "exit", "q"):
            print("Exiting inference. Goodbye!")
            break
        if not src_sentence:
            continue
        output = translate(src_sentence, model, embedding, vocab, id2word, device)
        print(f"Hindi : {output}\n")


if __name__ == "__main__":
    main()