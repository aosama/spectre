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
  - absolute chunk boundaries (Issue #6): prefix consistency at every length, the
    n == n_fft layout identical to the previous implementation, n_fft % C != 0 rejected
  - near-identity warm start survives chunking (official variant)
  - full R10-surgered GPT-2 is future-invariant end to end

Run: uv run pytest -q  (from the repo root)
"""
import math

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel

from spectre_research.r10_causal import (
    causal_query_pools,
    chunk_length_for,
    chunk_starts,
    chunked_causal_conv,
    effective_chunk_length,
)
from spectre_research.transplant import swap_gpt2_attention
from spectre_research.v1_spectre import SpectreHead, SpectreLayer
from spectre_research.causal_hybrid import (
    CausalSpectreHead,
    CausalSpectreMultiHead,
    interp_complex_1d_cubic,
    swap_gpt2_attention_official_causal,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_CHUNKS = 4
N_FFT_SMALL = 64
CHUNK_LENGTH_SMALL = chunk_length_for(N_FFT_SMALL, N_CHUNKS)  # 16


def _randomize_gate(last_linear: torch.nn.Linear) -> None:
    """Make the gate non-constant — the state any trained model is in, and the
    state in which the R5/R9 causality tests were vacuous (Issue #3)."""
    torch.nn.init.normal_(last_linear.weight, 0.0, 0.5)
    torch.nn.init.normal_(last_linear.bias, 0.0, 0.5)


def test_should_place_boundaries_at_multiples_of_chunk_length():
    assert chunk_starts(1024, 256) == [0, 256, 512, 768]  # the layout training has always used
    assert chunk_starts(48, 16) == [0, 16, 32]  # a shorter window has fewer chunks, never empty ones
    assert chunk_starts(10, 4) == [0, 4, 8]  # last chunk partial
    assert chunk_starts(100, 16) == [0, 16, 32, 48, 64, 80, 96]  # n > n_fft: more than C chunks


def test_should_keep_a_single_cold_start_chunk_for_short_sequences():
    assert chunk_starts(1, 16) == [0]
    assert chunk_starts(2, 16) == [0]  # n < C no longer raises: a 2-token prompt works with 4 chunks
    assert chunk_starts(16, 16) == [0]
    assert chunk_starts(17, 16) == [0, 16]


def test_should_not_move_boundaries_when_tokens_are_appended():
    for n in (1, 15, 16, 17, 300, 1024):
        shorter = chunk_starts(n, 16)
        longer = chunk_starts(n + 37, 16)
        assert longer[: len(shorter)] == shorter, f"boundaries moved when extending n={n}"


def test_should_reject_invalid_chunk_layouts():
    for bad_n, bad_length in [(0, 16), (-3, 16), (10, 0), (10, -1)]:
        with pytest.raises(ValueError):
            chunk_starts(bad_n, bad_length)


def test_should_derive_chunk_length_from_n_fft_and_chunk_count():
    assert chunk_length_for(1024, 4) == 256
    assert chunk_length_for(64, 2) == 32
    assert chunk_length_for(64, 64) == 1  # boundary: one chunk per token


def test_should_reject_n_fft_not_divisible_by_chunk_count():
    for n_fft, causal_chunks in [(64, 3), (1024, 6), (10, 4)]:
        with pytest.raises(ValueError, match="divisible"):
            chunk_length_for(n_fft, causal_chunks)


def test_should_reject_fewer_than_two_chunks_for_chunked_layouts():
    for causal_chunks in (1, 0, -2):
        with pytest.raises(ValueError, match=">= 2"):
            chunk_length_for(64, causal_chunks)


def test_should_treat_legacy_as_one_window_wide_chunk():
    assert effective_chunk_length(48, 64, 0) == 48
    assert effective_chunk_length(1, 64, 0) == 1
    assert effective_chunk_length(48, 64, 4) == 16


def test_should_reject_layers_whose_n_fft_is_not_divisible_by_chunk_count():
    with pytest.raises(ValueError, match="divisible"):
        SpectreHead(32, 8, 64, 16, causal_chunks=3)
    with pytest.raises(ValueError, match="divisible"):
        SpectreLayer(32, 4, 64, 16, causal_chunks=3)
    with pytest.raises(ValueError, match="divisible"):
        CausalSpectreHead(16, fft_size=64, d_gate=32, num_groups=4, causal_chunks=3)


def test_causal_query_pools_match_definition():
    torch.manual_seed(0)
    q = torch.randn(2, 48, 8)
    pools = causal_query_pools(q, CHUNK_LENGTH_SMALL)
    starts = chunk_starts(48, CHUNK_LENGTH_SMALL)
    assert pools.shape == (2, 3, 8)  # n=48 with 16-token chunks: K=3, not C=4
    for c, start in enumerate(starts):
        expected = q[:, :start, :].mean(dim=1) if start > 0 else torch.zeros(2, 8)
        assert torch.allclose(pools[:, c], expected, atol=1e-6), f"pool {c} mismatch"


def test_should_give_a_zero_pool_to_the_only_chunk_of_a_short_sequence():
    q = torch.randn(2, 5, 8)  # n < chunk_length: K == 1, and it must still be the cold-start chunk
    pools = causal_query_pools(q, CHUNK_LENGTH_SMALL)
    assert pools.shape == (2, 1, 8)
    assert torch.count_nonzero(pools) == 0


def test_should_keep_pools_of_a_prefix_equal_to_pools_of_the_full_sequence():
    torch.manual_seed(20)
    q = torch.randn(2, 100, 8)
    full_pools = causal_query_pools(q, CHUNK_LENGTH_SMALL)
    for prefix_length in (1, 15, 16, 17, 48, 64, 99):
        prefix_pools = causal_query_pools(q[:, :prefix_length], CHUNK_LENGTH_SMALL)
        assert torch.allclose(prefix_pools, full_pools[:, : prefix_pools.shape[1]], atol=1e-6), prefix_length


def _direct_chunked_oracle(v, kernels, chunk_length):
    S, n, d = v.shape
    N = kernels.shape[2]
    starts = chunk_starts(n, chunk_length)
    expected = np.zeros((S, n, d))
    for s_idx in range(S):
        for m in range(n):
            chunk = max(c for c, start in enumerate(starts) if start <= m)
            for lag in range(min(N, m + 1)):
                expected[s_idx, m, :] += kernels[s_idx, chunk, lag] * v[s_idx, m - lag, :]
    return expected


def test_chunked_causal_conv_matches_direct_oracle():
    """Per-chunk kernels with carry must equal the direct time-domain operator
    y[m] = sum_{s<=m} h_{chunk(m)}[m-s] * v[s]. Covers a full window, a partial
    last chunk, and n > n_fft (more chunks than causal_chunks)."""
    rng = np.random.default_rng(1)
    S, d, N = 3, 5, 64
    for n in (64, 50, 100):
        chunk_count = len(chunk_starts(n, CHUNK_LENGTH_SMALL))
        v = rng.standard_normal((S, n, d))
        kernels = rng.standard_normal((S, chunk_count, N))
        out = chunked_causal_conv(torch.from_numpy(v), torch.from_numpy(kernels), CHUNK_LENGTH_SMALL)
        expected = _direct_chunked_oracle(v, kernels, CHUNK_LENGTH_SMALL)
        assert np.abs(out.numpy() - expected).max() < 1e-4, f"n={n}"


def test_should_reject_a_kernel_count_that_does_not_match_the_chunk_count():
    v = torch.randn(1, 50, 3)
    wrong_kernels = torch.randn(1, 2, 64)  # n=50 needs 4 chunks of 16
    with pytest.raises(ValueError, match="chunks"):
        chunked_causal_conv(v, wrong_kernels, CHUNK_LENGTH_SMALL)


def test_should_make_chunked_conv_of_a_prefix_bit_identical_at_chunk_boundaries():
    """Cutting at a chunk boundary leaves every earlier chunk's FFT inputs
    unchanged, so those outputs are bit-for-bit equal (Issue #6)."""
    torch.manual_seed(21)
    S, d, N, n = 2, 4, 64, 100
    v = torch.randn(S, n, d)
    kernels = torch.randn(S, len(chunk_starts(n, CHUNK_LENGTH_SMALL)), N)
    full_output = chunked_causal_conv(v, kernels, CHUNK_LENGTH_SMALL)
    for prefix_length in (16, 32, 64, 96):
        prefix_chunks = len(chunk_starts(prefix_length, CHUNK_LENGTH_SMALL))
        prefix_output = chunked_causal_conv(v[:, :prefix_length], kernels[:, :prefix_chunks], CHUNK_LENGTH_SMALL)
        assert torch.equal(prefix_output, full_output[:, :prefix_length]), prefix_length


def test_chunked_causal_conv_is_bit_stable_under_future_perturbation():
    """Past outputs must not depend on perturbed future values.

    The bound is 1e-5 rather than exact equality because each chunk is a real
    FFT round trip; float32 round-off leaves ~1e-6 of noise in a signal whose
    magnitude is ~10. Exact-zero here would test the FFT, not the causality.
    """
    torch.manual_seed(2)
    S, n, d, N = 2, 64, 4, 64
    v = torch.randn(S, n, d)
    kernels = torch.randn(S, len(chunk_starts(n, CHUNK_LENGTH_SMALL)), N)
    v2 = v.clone()
    v2[:, 40:, :] += torch.randn(S, n - 40, d)
    y1 = chunked_causal_conv(v, kernels, CHUNK_LENGTH_SMALL)
    y2 = chunked_causal_conv(v2, kernels, CHUNK_LENGTH_SMALL)
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
            _randomize_gate(head.v1_gate.l2)
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
    n = 64
    starts = chunk_starts(n, CHUNK_LENGTH_SMALL)
    C = len(starts)
    q = torch.randn(1, n, head.d, requires_grad=True)
    chunk_index = 2
    start_c = starts[chunk_index]

    pools = head.q_norm(causal_query_pools(q, CHUNK_LENGTH_SMALL))
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


def test_r10_v1_no_gradient_from_future_queries():
    """Issue #3 criterion for the v1 layer: its per-chunk kernel cannot carry a
    gradient from the chunk's own-or-future queries, so the kernel is a pure
    function of strictly earlier positions only.

    Mirrors the official-layer gradient test but over the v1 gate path
    (LayerNorm -> Linear -> GELU(tanh) -> Linear -> modReLU -> iRFFT). The gate is forced
    non-constant, which is how the pre-R10 v1 tests were vacuous (the kernel
    was pooled over the whole window and moved whenever any future token moved).
    """
    torch.manual_seed(51)
    head = SpectreHead(32, 8, 64, 16, causal_chunks=N_CHUNKS)
    with torch.no_grad():
        _randomize_gate(head.v1_gate.l2)
        head.v1_gate.modrelu_bias.normal_(0.0, 0.05)
    n = 64
    starts = chunk_starts(n, CHUNK_LENGTH_SMALL)
    C = len(starts)
    q = torch.randn(1, n, head.d_head, requires_grad=True)  # per-head projected queries
    chunk_index = 2
    start_c = starts[chunk_index]

    # Build one length-N kernel per chunk from its own causal pool; the only
    # route by which a query can influence a kernel is through this gate.
    pools = causal_query_pools(q, CHUNK_LENGTH_SMALL)
    kernels = torch.fft.irfft(
        torch.stack([head.v1_gate(pools[:, c]) for c in range(C)], dim=0),
        n=head.n_fft, dim=-1,
    )
    kernels[chunk_index].abs().sum().backward()

    self_and_future_grad = q.grad[:, start_c:, :].abs().max().item()
    assert self_and_future_grad == 0.0, (
        f"v1 chunk-{chunk_index} kernel gradient reaches queries at/after its start: {self_and_future_grad}"
    )
    past_grad = q.grad[:, :start_c, :].abs().max().item()
    assert past_grad > 0.0, "v1 kernel received no gradient from its causal pool"


def test_r10_official_near_identity_init_preserved():
    """Warm start semantics survive chunking: zeroed gate MLP -> every chunk's
    kernel is 0.9*delta -> head output 0.9 * (x @ W_v.T) at every position."""
    torch.manual_seed(6)
    from spectre_research.causal_hybrid import CausalSpectreHead, make_gate_near_identity

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
            head.v1_gate.l2.weight.normal_(0, 0.5)
            head.v1_gate.l2.bias.normal_(0, 0.1)
            head.v1_gate.modrelu_bias.normal_(0, 0.05)
    x = torch.randn(3, 48, d_model)
    outs = [head(x) for head in layer.heads]
    expected = layer.wo(torch.cat(outs, dim=-1))
    actual = layer(x)
    assert (actual - expected).abs().max() < 1e-5, (
        f"batched R10 path diverges from per-head loop: {(actual - expected).abs().max()}"
    )


# --- Issue #6: absolute chunk boundaries ---------------------------------------------------

def _previous_chunk_starts(n: int, n_chunks: int) -> list[int]:
    """Frozen copy of the pre-#6 length-relative layout, kept as the oracle for
    "outputs at n == n_fft are unchanged" so trained R10/R11 checkpoints stay valid."""
    return [n * c // n_chunks for c in range(n_chunks)]


def _previous_causal_query_pools(q: torch.Tensor, n_chunks: int) -> torch.Tensor:
    B, n, d = q.shape
    starts = _previous_chunk_starts(n, n_chunks)
    pools = torch.zeros(B, n_chunks, d, dtype=q.dtype, device=q.device)
    cumsum = q.cumsum(dim=1)
    for c in range(1, n_chunks):
        pools[:, c] = cumsum[:, starts[c] - 1] / starts[c]
    return pools


def _previous_chunked_causal_conv(v, kernels, n_chunks):
    S, n, d = v.shape
    N = kernels.shape[2]
    starts = _previous_chunk_starts(n, n_chunks)
    out = torch.empty_like(v)
    for c in range(n_chunks):
        s0 = starts[c]
        s1 = starts[c + 1] if c + 1 < n_chunks else n
        a = max(0, s0 - (N - 1))
        m = s1 - a
        L = 1 << max(m + N - 1 - 1, 0).bit_length()
        v_pad = torch.zeros(S, L, d, device=v.device, dtype=v.dtype)
        v_pad[:, :m] = v[:, a:s1]
        h_pad = torch.zeros(S, L, device=v.device, dtype=v.dtype)
        h_pad[:, :N] = kernels[:, c]
        y = torch.fft.irfft(
            torch.fft.rfft(v_pad, n=L, dim=1) * torch.fft.rfft(h_pad, n=L, dim=1).unsqueeze(-1), n=L, dim=1
        )
        out[:, s0:s1] = y[:, m - (s1 - s0):m]
    return out


def test_should_match_previous_layout_at_full_length():
    torch.manual_seed(22)
    for n_fft, causal_chunks in [(64, 2), (64, 4), (128, 8), (1024, 4)]:
        chunk_length = chunk_length_for(n_fft, causal_chunks)
        assert chunk_starts(n_fft, chunk_length) == _previous_chunk_starts(n_fft, causal_chunks)
        q = torch.randn(2, n_fft, 6)
        assert torch.equal(causal_query_pools(q, chunk_length), _previous_causal_query_pools(q, causal_chunks))
        v = torch.randn(2, n_fft, 3)
        kernels = torch.randn(2, causal_chunks, n_fft)
        assert torch.equal(
            chunked_causal_conv(v, kernels, chunk_length), _previous_chunked_causal_conv(v, kernels, causal_chunks)
        ), (n_fft, causal_chunks)


def _randomized_v1_layer(chunk0_init: bool) -> torch.nn.Module:
    layer = SpectreLayer(32, 4, N_FFT_SMALL, 16, causal_chunks=N_CHUNKS, chunk0_init=chunk0_init)
    with torch.no_grad():
        for head in layer.heads:
            head.v1_gate.l2.weight.normal_(0, 0.5)
            head.v1_gate.l2.bias.normal_(0, 0.5)
            head.v1_gate.modrelu_bias.normal_(0, 0.05)
    return layer.eval()


def _randomized_official_layer() -> torch.nn.Module:
    layer = CausalSpectreMultiHead(embed_dim=32, num_heads=4, n_fft=N_FFT_SMALL, causal_chunks=N_CHUNKS)
    for head in layer.heads:
        _randomize_gate(head.gate_mlp[-1])
    return layer.eval()


def _layer_variants() -> dict:
    torch.manual_seed(23)
    return {
        "v1-r10": _randomized_v1_layer(chunk0_init=False),
        "v1-r11": _randomized_v1_layer(chunk0_init=True),
        "official-r10": _randomized_official_layer(),
    }


# Lengths around every boundary of a 16-token chunk, below one chunk, and beyond n_fft (K > C).
PREFIX_LENGTHS = [1, 2, 15, 16, 17, 40, 48, 63, 64, 65, 100]


def test_should_produce_prefix_consistent_output_for_every_length():
    """Issue #6: layer(x[:, :k]) == layer(x)[:, :k] under a randomized gate.
    The old length-relative layout measured max |delta| of 2.0-2.4 here. CPU
    only: MPS reductions are not deterministic enough for a 1e-5 bound."""
    torch.manual_seed(24)
    x = torch.randn(2, 130, 32)  # 130 > n_fft: the full run has more chunks than causal_chunks
    for variant_name, layer in _layer_variants().items():
        with torch.no_grad():
            full_output = layer(x)
            for prefix_length in PREFIX_LENGTHS:
                prefix_output = layer(x[:, :prefix_length])
                delta = (prefix_output - full_output[:, :prefix_length]).abs().max().item()
                assert delta <= 1e-5, f"{variant_name}: prefix {prefix_length} moved by {delta}"


def test_should_handle_a_single_token_and_fewer_tokens_than_chunks():
    torch.manual_seed(25)
    for variant_name, layer in _layer_variants().items():
        for token_count in (1, 2, 3):  # n < causal_chunks used to raise
            with torch.no_grad():
                output = layer(torch.randn(2, token_count, 32))
            assert output.shape == (2, token_count, 32), variant_name
            assert torch.isfinite(output).all(), f"{variant_name}: non-finite output for n={token_count}"


def test_should_stay_future_invariant_beyond_n_fft():
    """More chunks than causal_chunks (n > n_fft) must not reopen a future leak."""
    torch.manual_seed(26)
    x = torch.randn(2, 130, 32)
    perturbed = x.clone()
    perturbed[:, 100:] = torch.randn(2, 30, 32)
    for variant_name, layer in _layer_variants().items():
        with torch.no_grad():
            delta = (layer(x)[:, :100] - layer(perturbed)[:, :100]).abs().max().item()
        assert delta <= 1e-5, f"{variant_name}: leaks the future beyond n_fft: {delta}"


def test_should_keep_legacy_output_length_relative_by_design():
    """causal_chunks=0 is the leaking R5/R9 gate kept for historical numbers: its
    window-wide pool changes with n, and #6 must not alter that."""
    torch.manual_seed(27)
    layer = SpectreLayer(32, 4, N_FFT_SMALL, 16, causal_chunks=0)
    with torch.no_grad():
        for head in layer.heads:
            head.v1_gate.l2.weight.normal_(0, 0.5)
    layer = layer.eval()
    x = torch.randn(2, 48, 32)
    with torch.no_grad():
        delta = (layer(x[:, :20]) - layer(x)[:, :20]).abs().max().item()
    assert delta > 1e-3, "legacy gate unexpectedly became prefix-consistent"


def test_should_keep_full_gpt2_prefix_consistent_across_chunk_boundaries():
    """End to end (Issue #6): real GPT-2 with R10 surgery and a randomized gate
    returns the same logits for a prefix as for the full window, with cuts
    inside the 256-token chunks. CPU only, so the 1e-3 bound is deterministic."""
    torch.manual_seed(28)
    for variant_name, swap in [
        ("v1-r10", lambda model: swap_gpt2_attention(model, causal_chunks=N_CHUNKS)),
        ("official-r10", lambda model: swap_gpt2_attention_official_causal(model, causal_chunks=N_CHUNKS)),
    ]:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        swap(model)
        for block in model.transformer.h:
            for head in block.attn.spectre.heads:
                last_gate_linear = head.v1_gate.l2 if hasattr(head, "v1_gate") else head.gate_mlp[-1]
                _randomize_gate(last_gate_linear)
        model.eval()
        ids = torch.randint(0, 50256, (1, 700))
        with torch.no_grad():
            full_logits = model(ids).logits
            for prefix_length in (1, 130, 300, 520):
                prefix_logits = model(ids[:, :prefix_length]).logits
                delta = (prefix_logits - full_logits[:, :prefix_length]).abs().max().item()
                assert delta <= 1e-3, f"{variant_name}: prefix {prefix_length} moved logits by {delta}"
