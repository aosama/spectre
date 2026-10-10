"""GPT-2 geometry constants and transplant helpers shared by both packages.

These are properties of GPT-2 small, not of any SPECTRE variant, so both the
reproduction (spectre_paper.official) and the research transplants
(research.spectre_research.transplant / causal_hybrid) build on them.
Variants warm-init their layers with slice_head_projection and then freeze
everything else with freeze_backbone.
"""

N_FFT = 1024
GATE_HIDDEN = 64
D_HEAD = 64
N_HEADS = 12
D_MODEL = 768


def slice_head_projection(conv_weight, conv_bias, h: int, offset: int) -> tuple:
    """c_attn Conv1D weight is (in, 3*out); the [offset : offset+64] columns of
    the transpose are head h's projection rows for Q (offset=0), K (768), V (1536)."""
    w_t = conv_weight.t()  # (2304, 768)
    wq = w_t[offset + D_HEAD * h : offset + D_HEAD * (h + 1), :].t()  # (768, 64)
    bq = conv_bias[offset + D_HEAD * h : offset + D_HEAD * (h + 1)]
    return wq, bq


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