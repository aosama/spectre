"""R10 tests: strictly causal gate granularity (Issue #3).

The R10 layers compute one spectral kernel per chunk of the window, from a
query descriptor pooled over strictly earlier positions only — so the kernel
cannot see future tokens even when the gate MLP is non-constant (the gap that
made the R5/R9 causality tests vacuous; see Issue #3).

Acceptance criteria under test:
  - future-input invariance with a randomized (non-constant) gate, both variants
  - no gradient path from future queries to earlier outputs (kernel-source)
  - chunked convolution with per-chunk kernels matches a direct time-domain oracle
  - causal query pools match their definition (pool_0 empty, pool_c = mean of prefix)
  - near-identity warm start survives chunking (official variant)
  - full R10-surgered GPT-2 is future-invariant end to end

Run: cd realmodel && uv run python -m pytest tests/ -q
"""
import math

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from spectre_torch.chunked_causal import causal_query_pools, chunk_starts, chunked_causal_conv
from spectre_torch.layer import SpectreLayer
from spectre_torch.official_causal import (
    CausalSpectreMultiHead,
    interp_complex_1d_cubic,
    swap_gpt2_attention_official_causal,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_CHUNKS = 4


def _randomize_gate(last_linear: torch.nn.Linear) -> None:
    """Make the gate non-constant — the state any trained model is in, and the
    state in which the R5/R9 causality tests were vacuous (Issue #3)."""
    torch.nn.init.normal_(last_linear.weight, 0.0, 0.5)
    torch.nn.init.normal_(last_linear.bias, 0.0, 0.5)


def test_chunk_starts_partition_the_window():
    assert chunk_starts(1024, 4) == [0, 256, 512, 768]
    assert chunk_starts(48, 4) == [0, 12, 24, 36]
    assert chunk_starts(10, 4) == [0, 2, 5, 7]


def test_causal_query_pools_match_definition():
    torch.manual_seed(0)
    q = torch.randn(2, 48, 8)
    pools = causal_query_pools(q, 4)
    starts = chunk_starts(48, 4)
    assert pools.shape == (2, 4, 8)
    for c, start in enumerate(starts):
        expected = q[:, :start, :].mean(dim=1) if start > 0 else torch.zeros(2, 8)
        assert torch.allclose(pools[:, c], expected, atol=1e-6), f"pool {c} mismatch"


def test_chunked_causal_conv_matches_direct_oracle():
    """Per-chunk kernels with carry must equal the direct time-domain operator
    y[m] = sum_{s<=m} h_{chunk(m)}[m-s] * v[s]."""
    rng = np.random.default_rng(1)
    S, n, d, N, C = 3, 64, 5, 64, 4
    v = rng.standard_normal((S, n, d))
    kernels = rng.standard_normal((S, C, N))
    out = chunked_causal_conv(torch.from_numpy(v), torch.from_numpy(kernels), C)
    starts = chunk_starts(n, C)
    expected = np.zeros((S, n, d))
    for s_idx in range(S):
        for m in range(n):
            chunk = max(c for c, start in enumerate(starts) if start <= m)
            for lag in range(min(N, m + 1)):
                expected[s_idx, m, :] += kernels[s_idx, chunk, lag] * v[s_idx, m - lag, :]
    assert np.abs(out.numpy() - expected).max() < 1e-4


def test_chunked_causal_conv_is_bit_stable_under_future_perturbation():
    """Past outputs must not depend on perturbed future values.

    The bound is 1e-5 rather than exact equality because each chunk is a real
    FFT round trip; float32 round-off leaves ~1e-6 of noise in a signal whose
    magnitude is ~10. Exact-zero here would test the FFT, not the causality.
    """
    torch.manual_seed(2)
    S, n, d, N, C = 2, 64, 4, 64, 4
    v = torch.randn(S, n, d)
    kernels = torch.randn(S, C, N)
    v2 = v.clone()
    v2[:, 40:, :] += torch.randn(S, n - 40, d)
    y1 = chunked_causal_conv(v, kernels, C)
    y2 = chunked_causal_conv(v2, kernels, C)
    delta = (y1[:, :40] - y2[:, :40]).abs().max().item()
    assert delta <= 1e-5, f"past outputs moved under future perturbation: {delta}"


def test_r10_official_future_invariance_with_randomized_gate():
    """Issue #3 criterion: past outputs invariant to future inputs under a
    non-constant gate (the probe that measured 0.49 on the R9 layer)."""
    torch.manual_seed(3)
    layer = CausalSpectreMultiHead(embed_dim=32, num_heads=4, n_fft=64, causal_chunks=N_CHUNKS)
    for head in layer.heads:
        _randomize_gate(head.gate_mlp[-1])
    layer = layer.to(DEVICE).eval()
    x = torch.randn(2, 64, 32, device=DEVICE)
    with torch.no_grad():
        y1 = layer(x)
        x2 = x.clone()
        x2[:, 40:, :] = torch.randn(2, 24, 32, device=DEVICE)
        y2 = layer(x2)
    delta = (y1[:, :40] - y2[:, :40]).abs().max().item()
    assert delta <= 1e-5, f"R10 official layer leaks the future: max |delta| {delta}"


def test_r10_v1_layer_future_invariance_with_randomized_gate():
    """Same criterion for the v1 variant, which previously had no end-to-end
    causality test at all (only the fixed-kernel conv probe)."""
    torch.manual_seed(4)
    layer = SpectreLayer(32, 4, 64, 16, causal_chunks=N_CHUNKS)
    with torch.no_grad():
        for head in layer.heads:
            _randomize_gate(head.gate.l2)
    layer = layer.eval()
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        y1 = layer(x)
        x2 = x.clone()
        x2[:, 40:, :] = torch.randn(2, 24, 32)
        y2 = layer(x2)
    delta = (y1[:, :40] - y2[:, :40]).abs().max().item()
    assert delta <= 1e-5, f"R10 v1 layer leaks the future: max |delta| {delta}"


def test_r10_no_gradient_from_future_queries():
    """Issue #3 criterion: chunk c's kernel is a function of chunk c's *output*
    gradient through the gate descriptor alone, and that descriptor's gradient
    reaches only queries strictly before chunk c.

    We differentiate through the gate MLP fed by the causal pool — the only
    route by which a query can influence a kernel — and require that no
    gradient lands on queries at or after the chunk's own start. The value
    path (V -> convolution) is deliberately outside this test: position t
    legitimately depends on V_t.
    """
    torch.manual_seed(5)
    head = CausalSpectreMultiHead(embed_dim=32, num_heads=4, n_fft=64, causal_chunks=N_CHUNKS).heads[0]
    _randomize_gate(head.gate_mlp[-1])
    n, C = 64, N_CHUNKS
    q = torch.randn(1, n, head.d, requires_grad=True)
    starts = chunk_starts(n, C)
    chunk_index = 2
    start_c = starts[chunk_index]

    pools = head.q_norm(causal_query_pools(q, C))
    anchors = head.gate_mlp(pools.reshape(C, head.d)).view(C, head.G, head.B, 2)
    gate_half = torch.view_as_complex(anchors)
    gate_half = interp_complex_1d_cubic(gate_half, size=head.F_half)
    gate_half = head.modrelu(gate_half.reshape(C, -1)).view_as(gate_half)
    kernels = torch.fft.irfft(gate_half, n=head.n_fft, dim=-1)
    kernels[chunk_index].abs().sum().backward()

    self_and_future_grad = q.grad[:, start_c:, :].abs().max().item()
    assert self_and_future_grad == 0.0, (
        f"chunk-{chunk_index} kernel gradient reaches queries at/after its start: {self_and_future_grad}"
    )
    past_grad = q.grad[:, :start_c, :].abs().max().item()
    assert past_grad > 0.0, "kernel received no gradient from its own causal pool"


def test_r10_official_near_identity_init_preserved():
    """Warm start semantics survive chunking: zeroed gate MLP -> every chunk's
    kernel is 0.9*delta -> head output 0.9 * (x @ W_v.T) at every position."""
    torch.manual_seed(6)
    from spectre_torch.official_causal import CausalSpectreHead, make_gate_near_identity

    head = CausalSpectreHead(16, fft_size=64, d_gate=32, num_groups=4, causal_chunks=N_CHUNKS)
    make_gate_near_identity(head)
    head.eval()
    x = torch.randn(2, 64, 16)
    with torch.no_grad():
        y = head(x)
        expected = 0.9 * (x @ head.W_v.weight.t())
    assert torch.allclose(y, expected, atol=1e-4)


def test_r10_full_model_smoke_and_causality():
    """The R10-surgered GPT-2 must run and be future-invariant end to end with
    a non-constant gate path active (randomized gate MLPs)."""
    torch.manual_seed(7)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention_official_causal(model, causal_chunks=N_CHUNKS)
    for block in model.transformer.h:
        for head in block.attn.spectre.heads:
            _randomize_gate(head.gate_mlp[-1])
    model.to(DEVICE).eval()
    ids = torch.randint(0, 50256, (1, 64), device=DEVICE)
    with torch.no_grad():
        logits_orig = model(ids).logits
        ids_mod = ids.clone()
        ids_mod[:, 40:] = torch.randint(0, 50256, (1, 24), device=DEVICE)
        logits_mod = model(ids_mod).logits
    delta = (logits_orig[:, :40] - logits_mod[:, :40]).abs().max().item()
    assert delta <= 1e-3, f"full-model R10 future leakage: max |delta| {delta}"


def test_r10_v1_batched_layer_equals_per_head_loop():
    """The batched R10 v1 path must equal the per-head reference with a
    non-trivial gate — chunked pooling and conv included."""
    torch.manual_seed(8)
    d_model, n_heads, n_fft, hidden = 32, 4, 64, 16
    layer = SpectreLayer(d_model, n_heads, n_fft, hidden, causal_chunks=N_CHUNKS)
    with torch.no_grad():
        for head in layer.heads:
            head.gate.l2.weight.normal_(0, 0.5)
            head.gate.l2.bias.normal_(0, 0.1)
            head.gate.modrelu_bias.normal_(0, 0.05)
    x = torch.randn(3, 48, d_model)
    outs = [head(x) for head in layer.heads]
    expected = layer.wo(torch.cat(outs, dim=-1))
    actual = layer(x)
    assert (actual - expected).abs().max() < 1e-5, (
        f"batched R10 path diverges from per-head loop: {(actual - expected).abs().max()}"
    )
