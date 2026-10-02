"""R2 cross-validation: the PyTorch SPECTRE layer must reproduce the verified
Rust layer (all-pass gate) and the Rust rfft, before any model result is
trusted. Run: cd realmodel && uv run python -m spectre_torch.xval
"""
import json
import os

import torch

from .v1_spectre import SpectreLayer

_LAYER_THRESHOLD = 1e-4
_RFFT_THRESHOLD = 1e-5


def _load_layer(dump: dict) -> SpectreLayer:
    cfg = dump["cfg"]
    layer = SpectreLayer(cfg["d_model"], cfg["n_heads"], cfg["n_fft"], cfg["gate_hidden"])
    f32 = torch.float32
    with torch.no_grad():
        for i, hd in enumerate(dump["heads"]):
            head = layer.heads[i]
            head.wq.copy_(torch.tensor(hd["wq"], dtype=f32))
            head.wv.copy_(torch.tensor(hd["wv"], dtype=f32))
            # The Rust PoC head has no q/v bias; keep these at zero so the two
            # implementations are identical for this comparison.
            head.bq.zero_()
            head.bv.zero_()
            g = hd["gate"]
            gate = head.v1_gate
            gate.ln.weight.copy_(torch.tensor(g["ln_gamma"], dtype=f32))
            gate.ln.bias.copy_(torch.tensor(g["ln_beta"], dtype=f32))
            # Rust Linear stores (in x out) and computes w^T x; PyTorch
            # nn.Linear.weight is (out x in), so transpose on load. wq/wv use
            # raw matmul in both and are loaded directly above.
            gate.l1.weight.copy_(torch.tensor(g["l1_w"], dtype=f32).T)
            gate.l1.bias.copy_(torch.tensor(g["l1_b"], dtype=f32))
            gate.l2.weight.copy_(torch.tensor(g["l2_w"], dtype=f32).T)
            gate.l2.bias.copy_(torch.tensor(g["l2_b"], dtype=f32))
            gate.modrelu_bias.copy_(torch.tensor(g["modrelu_bias"], dtype=f32))
        layer.wo.weight.copy_(torch.tensor(dump["wo_w"], dtype=f32).T)
        layer.wo.bias.copy_(torch.tensor(dump["wo_b"], dtype=f32))
    return layer


def _max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).abs().max().item()
    return diff / max(b.abs().max().item(), 1e-12)


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "xval-dump", "out.json")) as f:
        dump = json.load(f)

    layer = _load_layer(dump)
    x = torch.tensor(dump["x"], dtype=torch.float32).unsqueeze(0)  # (1, n, d_model)
    y_ref = torch.tensor(dump["y"], dtype=torch.float32).squeeze(0)
    layer_err = _max_rel_err(layer(x).squeeze(0), y_ref)

    rfft_in = torch.tensor(dump["rfft_input"], dtype=torch.float32)
    rfft_ref = torch.tensor(dump["rfft_output"], dtype=torch.float32)
    rfft_ref_c = torch.complex(rfft_ref[:, 0], rfft_ref[:, 1])
    rfft_err = _max_rel_err(torch.fft.rfft(rfft_in, n=rfft_in.numel()), rfft_ref_c)

    print(
        f"XVAL layer max rel err: {layer_err:.3e} "
        f"(threshold {_LAYER_THRESHOLD}) {'PASS' if layer_err < _LAYER_THRESHOLD else 'FAIL'}"
    )
    print(
        f"XVAL rfft max rel err: {rfft_err:.3e} "
        f"(threshold {_RFFT_THRESHOLD}) {'PASS' if rfft_err < _RFFT_THRESHOLD else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
