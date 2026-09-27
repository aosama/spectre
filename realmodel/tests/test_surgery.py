"""R3 tests: warm-init surgery, frozen backbone, param accounting.

Run: cd realmodel && uv run python -m pytest tests/ -q
"""
import hashlib

import torch
from transformers import GPT2LMHeadModel

from spectre_torch.surgery import (
    D_HEAD,
    N_HEADS,
    freeze_backbone,
    param_report,
    swap_gpt2_attention,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# Plan said 31,482,144 but that arithmetic omits the per-head modrelu_bias
# (513 params x 12 heads x 12 layers = 73,872). Correct count:
# per layer = 12 heads x (98,432 qv + 71,491 gate) + 590,592 wo = 2,629,668.
EXPECTED_TRAINABLE = 31_556_016


def _hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def test_warm_init_slices_match():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    attn = model.transformer.h[0].attn
    w_t = attn.c_attn.weight.t()  # (2304, 768): rows [Q | K | V]
    expected = {
        "wq0": w_t[0:64, :].clone(),
        "wv0": w_t[1536:1600, :].clone(),
        "bq0": attn.c_attn.bias[0:64].clone(),
        "bv0": attn.c_attn.bias[1536:1600].clone(),
        "wo": attn.c_proj.weight.clone(),
        "bo": attn.c_proj.bias.clone(),
    }
    swap_gpt2_attention(model)
    head = model.transformer.h[0].attn.spectre.heads[0]
    assert torch.equal(head.wq, expected["wq0"].t())
    assert torch.equal(head.wv, expected["wv0"].t())
    assert torch.equal(head.bq, expected["bq0"])
    assert torch.equal(head.bv, expected["bv0"])
    assert torch.equal(model.transformer.h[0].attn.spectre.wo.weight, expected["wo"].t())
    assert torch.equal(model.transformer.h[0].attn.spectre.wo.bias, expected["bo"])


def test_forward_runs_on_mps():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention(model)
    model.to(DEVICE).eval()
    x = torch.randint(0, 50257, (2, 1024), device=DEVICE)
    with torch.no_grad():
        logits = model(x).logits
    assert logits.shape == (2, 1024, 50257)


def test_frozen_params_unchanged():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention(model)
    freeze_backbone(model)
    model.to(DEVICE)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    sample = frozen[::17]  # ~every 17th frozen tensor
    before = [_hash(p) for p in sample]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = model(torch.randint(0, 50257, (2, 64), device=DEVICE)).logits.sum()
    loss.backward()
    opt.step()
    after = [_hash(p) for p in sample]
    assert before == after, "frozen params changed after optimizer step"


def test_param_report():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention(model)
    freeze_backbone(model)
    report = param_report(model)
    print(f"param report: {report}")
    assert report["trainable"] == EXPECTED_TRAINABLE, (
        f"trainable={report['trainable']}, expected {EXPECTED_TRAINABLE}"
    )
    # Post-surgery total exceeds stock GPT-2 (124,439,808): the gate MLPs add
    # more params than the discarded K rows removed. Assert consistency and
    # the exact frozen count instead.
    assert report["total"] == report["trainable"] + report["frozen"]
    assert report["frozen"] == 96_091_392