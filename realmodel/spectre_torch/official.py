"""R8: the official SPECTRE math, imported directly from the vendored author code.

Rather than re-transcribing the author's implementation (audit in
docs/audit-vs-official.md), we load implementation_from_whitepaper_author/spectre.py
verbatim and build the GPT-2 transplant on top of it. A SHA-256 pin refuses to
load anything other than the audited copy, so "our math matches Jacob's" holds
by construction, not by transcription discipline.

Differences vs our v1 layer (realmodel/spectre_torch/layer.py), all inherited
from the official code:
  - grouped gate: G=4 groups, B=max(4, sqrt(F_half)) anchors, cubic-interpolated
  - DCT pooling of Q (first 64 DCT components), not mean pooling
  - smooth modReLU (sqrt(|z|^2+eps^2) denominator), bias init -0.1
  - exact erf GELU in the gate MLP, hidden dim d_gate=256
  - per-head W_q/W_v project only that head's 64-dim input slice, bias-free
  - mixing is rfft(V, n=n_fft) * gate -> irfft with n_fft == block length:
    circular-window semantics, no causal mask (the audited official behavior)
  - out_proj bias-free

n_fft = 1024 = block length (circular window over the full block, matching the
decode cache's window semantics). Wavelet refinement is disabled (on_rate=0):
the paper's Table 1 reports SPECTRE and SPECTRE+Wavelet separately, and we are
reproducing plain SPECTRE.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from .surgery import D_HEAD, D_MODEL, N_HEADS

_REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_PATH = _REPO_ROOT / "implementation_from_whitepaper_author" / "spectre.py"
PINNED_SHA256 = "ae63a56a3ad549d561dee67eb65f2267cadc9e5052fabcf4b45dc74965dcfbe2"

N_FFT_OFFICIAL = 1024
D_GATE_OFFICIAL = 256
NUM_GROUPS_OFFICIAL = 4
WAVELET_ON_RATE = 0.0

_OFFICIAL_MODULE = None


def load_official_module():
    """Import the vendored spectre.py, refusing any copy that fails the hash pin."""
    digest = hashlib.sha256(VENDORED_PATH.read_bytes()).hexdigest()
    if digest != PINNED_SHA256:
        raise RuntimeError(
            f"{VENDORED_PATH} does not match the audited SHA-256 pin.\n"
            f"  expected {PINNED_SHA256}\n"
            f"  got      {digest}\n"
            "The vendored copy must stay byte-identical to jacobfa/fft@spectre.py "
            "(see implementation_from_whitepaper_author/README.md)."
        )
    spec = importlib.util.spec_from_file_location("spectre_official_vendored", VENDORED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official():
    global _OFFICIAL_MODULE
    if _OFFICIAL_MODULE is None:
        _OFFICIAL_MODULE = load_official_module()
    return _OFFICIAL_MODULE


class SpectreAttentionOfficial(nn.Module):
    """GPT-2 attention replacement wrapping the vendored SpectreMultiHead.

    The wrapped module is exposed as `.spectre` so train.py's existing
    ".attn.spectre." param-group filter keeps working unchanged.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.spectre = layer

    def forward(self, hidden_states, *args, **kwargs):
        return (self.spectre(hidden_states), None)


def _build_layer():
    return official().SpectreMultiHead(
        embed_dim=D_MODEL,
        num_heads=N_HEADS,
        n_fft=N_FFT_OFFICIAL,
        d_gate=D_GATE_OFFICIAL,
        pooling_type="dct",
        num_groups=NUM_GROUPS_OFFICIAL,
        wavelet_on_rate=WAVELET_ON_RATE,
    )


def swap_gpt2_attention_official(model) -> "GPT2LMHeadModel":
    """Replace each block's GPT2Attention with the vendored SpectreMultiHead.

    Warm start, adapted to the official head geometry: the official W_q/W_v
    are head_dim x head_dim (each head only sees its own input slice), so each
    head is seeded with the block-diagonal slice of GPT-2's Q/V projection for
    that head. out_proj takes c_proj's weight (the official out_proj is
    bias-free, so c_proj's bias is dropped). Gate MLP, q_norm and modrelu keep
    the official default init — the official code has no gate warm-start.
    """
    for block in model.transformer.h:
        attn = block.attn
        layer = _build_layer()
        with torch.no_grad():
            # c_attn Conv1D weight (768, 2304); transpose -> (2304, 768) rows [Q | K | V]
            w_t = attn.c_attn.weight.t()
            for h, head in enumerate(layer.heads):
                rows_q = slice(D_HEAD * h, D_HEAD * (h + 1))
                rows_v = slice(1536 + D_HEAD * h, 1536 + D_HEAD * (h + 1))
                cols = slice(D_HEAD * h, D_HEAD * (h + 1))
                head.W_q.weight.copy_(w_t[rows_q, cols])
                head.W_v.weight.copy_(w_t[rows_v, cols])
            layer.out_proj.weight.copy_(attn.c_proj.weight.t())
        block.attn = SpectreAttentionOfficial(layer)
    return model
