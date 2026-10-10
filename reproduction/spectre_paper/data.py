"""WikiText-2 → 1024-token blocks (train/validation/test).

Concatenates all documents (the standard LM corpus treatment), tokenizes with
GPT-2's BPE, and cuts into non-overlapping 1024-token blocks.
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

BLOCK = 1024


def _token_stream(split: str) -> torch.Tensor:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    text = "\n".join(str(t) for t in ds[split]["text"])
    tok = AutoTokenizer.from_pretrained("gpt2")
    return tok(text, return_tensors="pt", truncation=False).input_ids.flatten()


def blocks(split: str) -> torch.Tensor:
    """(n_blocks, 1024) int64 tensor of non-overlapping token blocks."""
    ids = _token_stream(split)
    n_blocks = ids.numel() // BLOCK
    return ids[: n_blocks * BLOCK].view(n_blocks, BLOCK)


def loaders(batch_size: int, split: str):
    """Simple deterministic loader: yields (B, 1024) batches, no shuffle."""
    data = blocks(split)
    for i in range(0, data.shape[0], batch_size):
        yield data[i : i + batch_size]