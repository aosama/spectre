"""R3: GPT-2 surgery — replace every GPT2Attention with a SPECTRE layer.

Warm init (plan D6): W_q/W_v sliced from c_attn's [Q|K|V] rows, W_o copied
from c_proj, gate all-pass (g == 1) so the swapped model starts as close to
the original as the architecture allows. K rows are discarded (SPECTRE has no
key projection). Everything non-SPECTRE is frozen.
"""
import torch
import torch.nn as nn

from .layer import SpectreLayer

N_FFT = 1024
GATE_HIDDEN = 64
D_HEAD = 64
N_HEADS = 12
D_MODEL = 768


class SpectreAttention(nn.Module):
    """Drop-in wrapper: keeps the block's residual/LN wiring, swaps the math.

    forward returns just the output tensor (no attn weights) — GPT-2's block
    code accepts a tuple, so we wrap it. attention_mask/past_key_values are
    accepted and ignored: SPECTRE mixing is causal by construction and needs
    no KV cache.
    """

    def __init__(self, layer: SpectreLayer):
        super().__init__()
        self.spectre = layer

    def forward(self, hidden_states, *args, **kwargs):
        out = self.spectre(hidden_states)
        return (out, None)


def _slice_head(conv_weight: torch.Tensor, conv_bias: torch.Tensor, h: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """c_attn Conv1D weight is (in, 3*out); the [offset : offset+64] columns of
    the transpose are head h's projection rows for Q (offset=0), K (768), V (1536)."""
    w_t = conv_weight.t()  # (2304, 768)
    wq = w_t[offset + D_HEAD * h : offset + D_HEAD * (h + 1), :].t()  # (768, 64)
    bq = conv_bias[offset + D_HEAD * h : offset + D_HEAD * (h + 1)]
    return wq, bq


def swap_gpt2_attention(model, causal_chunks: int = 0) -> "GPT2LMHeadModel":
    """Replace each block's GPT2Attention with a warm-initialized SpectreAttention.

    causal_chunks >= 2 (R10, Issue #3) makes the gate strictly causal: one
    kernel per chunk from strictly earlier queries instead of the whole-window
    query mean.
    """
    for block in model.transformer.h:
        attn = block.attn
        layer = SpectreLayer(D_MODEL, N_HEADS, N_FFT, GATE_HIDDEN, causal_chunks=causal_chunks)
        with torch.no_grad():
            for h, head in enumerate(layer.heads):
                wq, bq = _slice_head(attn.c_attn.weight, attn.c_attn.bias, h, 0)
                wv, bv = _slice_head(attn.c_attn.weight, attn.c_attn.bias, h, 1536)
                head.wq.copy_(wq)
                head.bq.copy_(bq)
                head.wv.copy_(wv)
                head.bv.copy_(bv)
                # gate stays at its all-pass init (l2.weight=0, bias=(1,0))
            layer.wo.weight.copy_(attn.c_proj.weight.t())
            layer.wo.bias.copy_(attn.c_proj.bias)
        block.attn = SpectreAttention(layer)
    return model


def freeze_backbone(model) -> "GPT2LMHeadModel":
    """All non-SPECTRE params requires_grad_(False)."""
    model.requires_grad_(False)
    for block in model.transformer.h:
        block.attn.requires_grad_(True)
    return model


def param_report(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}