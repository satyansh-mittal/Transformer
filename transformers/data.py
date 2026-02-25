from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from utils import tokenize


def download_data():
    dataset = load_dataset("cfilt/iitb-english-hindi")          # or: load_dataset("cfilt/iitb-english-hindi", "default")
    # Take first 2000 clean pairs
    small_data = dataset["train"].select(range(2000))
    source_sentences = [ex["translation"]["en"] for ex in small_data]
    target_sentences = [ex["translation"]["hi"] + " <eos>" for ex in small_data]
    return source_sentences, target_sentences

def download_data2():
    dataset = load_dataset("cfilt/iitb-english-hindi")
    # Full dataset for a large GPU like H100
    large_data = dataset["train"]
    source_sentences = [ex["translation"]["en"] for ex in large_data]
    target_sentences = [ex["translation"]["hi"] + " <eos>" for ex in large_data]
    return source_sentences, target_sentences

class SentencePairDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences):
        assert len(src_sentences) == len(tgt_sentences)
        self.pairs = list(zip(src_sentences, tgt_sentences))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


from utils import load_config
config = load_config()
MAX_LEN = config.get("max_len", 50)

def sentence_collate(batch, vocab):
    src_list, tgt_list = zip(*batch)
    
    # Process src
    src_ids_list = [tokenize(s, vocab)[0][:MAX_LEN] for s in src_list]
    max_src = max((len(ids) for ids in src_ids_list), default=1)
    src_padded = torch.full((len(batch), max_src), vocab['<pad>'], dtype=torch.long)
    for i, ids in enumerate(src_ids_list):
        src_padded[i, :len(ids)] = ids
    
    # Process tgt
    tgt_ids_list = []
    eos_token = vocab['<eos>']
    for t in tgt_list:
        ids = tokenize(t, vocab)[0]
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
            ids[-1] = eos_token  # ensure <eos> is cleanly at the end
        tgt_ids_list.append(ids)
        
    max_tgt = max((len(ids) for ids in tgt_ids_list), default=1)
    tgt_padded = torch.full((len(batch), max_tgt), vocab['<pad>'], dtype=torch.long)
    for i, ids in enumerate(tgt_ids_list):
        tgt_padded[i, :len(ids)] = ids
        
    return src_padded, tgt_padded