import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
import math
import os

from utils import positionalEncoding, positionalEncoding2, make_padding_mask, make_causal_mask, combine_padding_and_causal, shift_right, build_vocab, load_config
from data import SentencePairDataset, sentence_collate, download_data, download_data2
from transformer import Transformer
from lr_scheduler import NoamScheduler

config = load_config()
d_model = config["d_model"]
heads = config["heads"]
dropout = config["dropout"]
N_EPOCHS = config["N_EPOCHS"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(model, iterator, embedding, loss_fn, optimizer, scheduler, d_model, dropout, vocab, device, local_rank=0):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in iterator:
        src_input_ids, tgt_input_ids = batch
    
        src_input_ids = src_input_ids.to(device)
        tgt_input_ids = tgt_input_ids.to(device)
        
        tgt_labels = tgt_input_ids.clone()
        
        src_input_embeddings = embedding(src_input_ids) * math.sqrt(d_model)
        src_positional_encoding = positionalEncoding2(src_input_embeddings, d_model)
        x_encoder = src_input_embeddings + src_positional_encoding
        x_encoder = F.dropout(x_encoder, p=dropout, training=model.training)
        
        tgt_input_ids = shift_right(tgt_input_ids, vocab)
        tgt_input_embeddings = embedding(tgt_input_ids) * math.sqrt(d_model)
        tgt_positional_encoding = positionalEncoding2(tgt_input_embeddings, d_model)
        x_decoder = tgt_input_embeddings + tgt_positional_encoding
        x_decoder = F.dropout(x_decoder, p=dropout, training=model.training)
        
        src_mask = make_padding_mask(src_input_ids, pad_idx=vocab["<pad>"])
        tgt_pad_mask = make_padding_mask(tgt_input_ids, pad_idx=vocab["<pad>"])
        causal = make_causal_mask(tgt_input_ids.shape[1], device=device)
        tgt_mask = combine_padding_and_causal(tgt_pad_mask, causal)
        
        optimizer.zero_grad()
        logits = model(x_encoder, x_decoder, tgt_mask=tgt_mask, src_mask=src_mask)
        
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        if num_batches % 100 == 0 and local_rank == 0:
            print(f"  [Batch {num_batches}] Loss: {loss.item():.4f}", flush=True)
        
    return epoch_loss / num_batches


def save_checkpoint(path, model, embedding, vocab, vocab_size, d_model, heads, dropout, num_layers, epoch, loss):
    torch.save({
        "model_state":     model.state_dict(),
        "embedding_state": embedding.state_dict(),
        "vocab":           vocab,
        "vocab_size":      vocab_size,
        "d_model":         d_model,
        "heads":           heads,
        "dropout":         dropout,
        "num_layers":      num_layers,
        "epoch":           epoch,
        "best_loss":       loss,
    }, path)


def train(ddp=False, local_rank=0):
    torch.autograd.set_detect_anomaly(True)
    import torch.distributed as dist
    if ddp:
        device = f"cuda:{local_rank}"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if local_rank == 0 or not ddp:
        print("Downloading / loading dataset...")
    use_more_data = True 
    if use_more_data:
        if local_rank == 0 or not ddp:
            print("Using larger dataset (download_data2)...")
        source_sentences, target_sentences = download_data2()
    else:
        if local_rank == 0 or not ddp:
            print("Using smaller dataset (download_data)...")
        source_sentences, target_sentences = download_data()

    if local_rank == 0 or not ddp:
        print("Building vocabulary...")
    sentences = source_sentences + target_sentences
    vocab = build_vocab(sentences)
    vocab_size = len(vocab)
    id2word = {idx: word for word, idx in vocab.items()}

    collate_fn = partial(sentence_collate, vocab=vocab)

    dataset = SentencePairDataset(source_sentences, target_sentences)
    sampler = None
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
    batch_size = config.get("batch_size", 256)
    iterator = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    _dropout = config.get("dropout", 0.1)
    _d_model = config.get("d_model", 256)
    _heads = config.get("heads", 8)
    num_layers = config.get("num_layers", 4)

    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=_d_model).to(device)
    model = Transformer(_d_model, _heads, _dropout, vocab_size=vocab_size, num_layers=num_layers, embedding=embedding).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    if embedding.weight.dim() > 1:
        nn.init.normal_(embedding.weight, mean=0, std=0.02)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'], label_smoothing=0.1)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedding.parameters()),
        lr=0.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=_d_model, warmup_steps=2000)

    _N_EPOCHS  = N_EPOCHS
    best_loss  = float('inf')
    ckpt_dir   = os.path.dirname(os.path.abspath(__file__))
    best_path  = os.path.join(ckpt_dir, "checkpoint_best.pt")
    final_path = os.path.join(ckpt_dir, "checkpoint.pt")

    for epoch in range(_N_EPOCHS):
        if ddp:
            iterator.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, iterator, embedding,
            loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
            d_model=_d_model, dropout=_dropout,
            vocab=vocab, device=device, local_rank=local_rank)

        improved = train_loss < best_loss
        if improved and (local_rank == 0 or not ddp):
            best_loss = train_loss
            save_checkpoint(best_path, model.module if ddp else model, embedding, vocab, vocab_size,
                            _d_model, _heads, _dropout, num_layers,
                            epoch=epoch + 1, loss=best_loss)

        tag = "  *** best ***" if improved else ""
        if local_rank == 0 or not ddp:
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | '
                  f'Train PPL: {math.exp(train_loss):7.3f}{tag}')

    if local_rank == 0 or not ddp:
        save_checkpoint(final_path, model.module if ddp else model, embedding, vocab, vocab_size,
                        _d_model, _heads, _dropout, num_layers,
                        epoch=_N_EPOCHS, loss=train_loss)
        print(f"\nFinal model  saved to : {final_path}")
        print(f"Best model   saved to : {best_path}  (loss={best_loss:.4f})")
    return model, embedding, vocab, id2word