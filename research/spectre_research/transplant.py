"""R3: GPT-2 transplant of our v1 SPECTRE layer (the research variant).

Warm init (plan D6): W_q/W_v sliced from c_attn's [Q|K|V] rows, W_o copied
from c_proj, gate all-pass (g == 1) so the swapped model starts as close to
the original as the architecture allows. K rows are discarded (SPECTRE has no
key projection). Everything non-SPECTRE is frozen via
spectre_paper.surgery.freeze_backbone.

The official-math counterpart lives in the reproduction package
(spectre_paper.official.swap_gpt2_attention_official).
"""
import torch
import torch.nn as nn

from spectre_paper.surgery import (
    D_HEAD,
    D_MODEL,
    GATE_HIDDEN,
    N_FFT,
    N_HEADS,
    slice_head_projection,
)

from .v1_spectre import SpectreLayer


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


def swap_gpt2_attention(
    model,
    causal_chunks: int = 0,
    chunk0_init: bool = False,
) -> "GPT2LMHeadModel":
    """Replace each block's GPT2Attention with a warm-initialized SpectreAttention.

    causal_chunks >= 2 (R10, Issue #3) makes the gate strictly causal: one
    kernel per chunk from strictly earlier queries instead of the whole-window
    query mean. chunk0_init (R11) copies the empty-prefix gate output to all
    chunks at initialization while leaving the gate trainable.
    """
    for block in model.transformer.h:
        attn = block.attn
        layer = SpectreLayer(
            D_MODEL,
            N_HEADS,
            N_FFT,
            GATE_HIDDEN,
            causal_chunks=causal_chunks,
            chunk0_init=chunk0_init,
        )
        with torch.no_grad():
            for h, head in enumerate(layer.heads):
                wq, bq = slice_head_projection(attn.c_attn.weight, attn.c_attn.bias, h, 0)
                wv, bv = slice_head_projection(attn.c_attn.weight, attn.c_attn.bias, h, 1536)
                head.wq.copy_(wq)
                head.bq.copy_(bq)
                head.wv.copy_(wv)
                head.bv.copy_(bv)
                # gate stays at its all-pass init (l2.weight=0, bias=(1,0))
            layer.wo.weight.copy_(attn.c_proj.weight.t())
            layer.wo.bias.copy_(attn.c_proj.bias)
        block.attn = SpectreAttention(layer)
    return model
