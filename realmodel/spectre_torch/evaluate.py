"""Standard batched WikiText-2 perplexity: exp(mean NLL over predicted tokens).

Each 1024-token block contributes 1023 predictions (positions 1..1022 are
supervised; the last token has no successor within the block).
"""
import math

import torch


@torch.no_grad()
def perplexity(model, loader, device: str) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).logits  # (B, 1024, vocab)
        # shift: predict token t+1 from position t
        preds = logits[:, :-1, :]
        targets = batch[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            preds.reshape(-1, preds.shape[-1]), targets.reshape(-1), reduction="sum"
        )
        total_nll += nll.item()
        total_tokens += targets.numel()
    return math.exp(total_nll / total_tokens)