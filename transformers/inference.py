import torch
import math

from utils import tokenize, positionalEncoding2, make_padding_mask, make_causal_mask, combine_padding_and_causal, load_config

config = load_config()
d_model = config["d_model"]
heads = config["heads"]
dropout = config["dropout"]


def translate(src_sentence, model, embedding, vocab, id2word, device, max_len=50):
    
    model.eval()
    with torch.no_grad():
        # 1. Encode source (once)
        src_ids = tokenize(src_sentence, vocab).to(device)
        src_emb = embedding(src_ids) * math.sqrt(d_model)
        src_pe = positionalEncoding2(src_emb, d_model).to(device)
        x_encoder = src_emb + src_pe
        src_mask = make_padding_mask(src_ids, pad_idx=vocab["<pad>"])

        # 2. Start decoding with <sos>
        tgt_ids = torch.tensor([[vocab['<sos>']]], device=device)

        for step in range(max_len):
            # Decoder input
            tgt_emb = embedding(tgt_ids) * math.sqrt(d_model)
            tgt_pe = positionalEncoding2(tgt_emb, d_model).to(device)
            x_decoder = tgt_emb + tgt_pe

            # Masks (using your exact functions)
            tgt_pad_mask = make_padding_mask(tgt_ids, pad_idx=vocab["<pad>"])
            causal = make_causal_mask(tgt_ids.shape[1], device=device)
            tgt_mask = combine_padding_and_causal(tgt_pad_mask, causal)

            # Forward
            logits = model(x_encoder, x_decoder, tgt_mask=tgt_mask, src_mask=src_mask)

            # Softmax + argmax (exactly as in the paper diagram)
            probs = torch.softmax(logits[0, -1, :], dim=-1)      # (vocab_size,)
            next_id = torch.argmax(probs).item()
            next_token = torch.tensor([[next_id]], device=device)

            # Append
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            if next_id == vocab['<eos>']:
                break

        # Convert to sentence
        translated_tokens = [id2word[tok.item()] for tok in tgt_ids[0][1:]]  # skip <sos>
        translated = ' '.join(translated_tokens).replace('<eos>', '').strip()

        return translated
