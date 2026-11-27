# native — SPECTRE research in C++20 on MLX

All new SPECTRE research code lives here (Issue #5). Python under `research/` is a frozen reference, and `tools/` holds the Python exporters that feed this folder.

## Requirements

- macOS on Apple silicon, Xcode command line tools with the Metal toolchain
- CMake ≥ 3.27 and Ninja (`brew install cmake ninja`)
- **MLX ≥ 0.32.5** (decision D2). yaml-cpp and GoogleTest are fetched automatically at pinned tags.

### Installing MLX

Homebrew first, if it already ships 0.32.5 or newer (check with `brew info mlx`):

```sh
brew install mlx
```

If Homebrew still ships an older MLX, build the tagged release from source into a prefix of your choice:

```sh
git clone --depth 1 --branch v0.32.5 https://github.com/ml-explore/mlx.git
cmake -S mlx -B mlx/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$HOME/.local/mlx-0.32.5" \
      -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
      -DMLX_BUILD_PYTHON_BINDINGS=OFF -DBUILD_SHARED_LIBS=ON
cmake --build mlx/build --target install
```

Then tell CMake where it is **without environment variables**, using a personal, git-ignored `native/CMakeUserPresets.json`:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "dev-local",
      "inherits": "dev",
      "cacheVariables": { "CMAKE_PREFIX_PATH": "/Users/you/.local/mlx-0.32.5" }
    }
  ],
  "buildPresets": [{ "name": "dev-local", "configurePreset": "dev-local" }],
  "testPresets": [{ "name": "dev-local", "inherits": "dev", "configurePreset": "dev-local" }]
}
```

and use `dev-local` wherever the commands below say `dev`.

If the linker warns that `libmlx.dylib` was built for a newer macOS than the target, add `"CMAKE_OSX_DEPLOYMENT_TARGET"` to `cacheVariables` with the version named in the warning (a source-built MLX records the macOS version of the machine that built it).

### Gotchas

- CMake also searches `mlx*` directories under any prefix derived from `PATH` (for example `~/.local`), so an install at `~/.local/mlx-<version>` is found even without `CMAKE_PREFIX_PATH`. To test the "MLX too old or missing" error, add `-DCMAKE_IGNORE_PREFIX_PATH=$HOME/.local`.
- MLX C++: `mx::fft::rfft/irfft` take `(a, n, axis, FFTNorm, StreamOrDevice)` with the norm before the device, and `item<T>()` on a `const` unevaluated array throws, so call `mx::eval` first (also recorded in `docs/deviations.md`).

## Build and test

Run from this folder (`native/`):

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

| Preset | Build type | Tests run |
|---|---|---|
| `dev` | Debug | everything except labels `acceptance` and `performance*`, on CPU and GPU |
| `release` | Release | same selection as `dev` |
| `ci` | Release | labels `unit`, `component`, `integration`; GPU variants (`*/gpu`) excluded. Unused while CI is off; it is what a hosted CI job should run |

Tests carry CTest labels: `unit`, `component`, `integration`, `acceptance`, `performance-measurement` (reports numbers) and `performance` (asserts a threshold). Run one category with `ctest --preset dev -L unit`. Tests that touch MLX run on both devices and end in `/cpu` or `/gpu`.

One-shot (configure, build, test): `cmake --workflow --preset dev`.

## Pre-commit hooks (instead of CI)

While the research is unstable there is no GitHub Actions CI; a local gate replaces it. Install it once per clone (needs [pre-commit](https://pre-commit.com), e.g. `brew install pre-commit`):

```sh
pre-commit install --allow-missing-config
```

On each commit, only the hooks relevant to the staged files run: `native/**` runs `cmake --workflow --preset dev` (build plus every test on CPU and GPU); Python under `research/` or `reproduction/` runs `uv run pytest -q`; every file is checked for developer home paths, merge-conflict markers, oversized files and valid YAML/JSON. Run everything by hand with `pre-commit run --all-files`. If MLX is only found through a `CMakeUserPresets.json` prefix, run `cmake --workflow --preset dev-local` yourself and commit with `git commit --no-verify`, or install MLX where CMake finds it by default.

### Bringing GitHub Actions back

The T0.2 workflow (macOS arm64 runner, MLX v0.32.5 built from source and cached, then `cmake --preset ci`, build, `ctest --preset ci`) is kept in git history: `git show 1b8a759:.github/workflows/native.yml` (commit from PR #12; the PR ref also keeps it). Re-enable by restoring that file under `.github/workflows/`; the `ci` presets above are what it runs.

## Configuration

Executables take exactly one `--config <file.yaml>` argument; there are no behavior-changing environment variables. A config file is a set of top-level sections, and each section is parsed by its own loader, which rejects unknown keys and invalid values. The `spectre_layer` section:

```yaml
spectre_layer:
  d_model: 768
  head_count: 12
  n_fft: 1024               # power of two
  gate_hidden_width: 256
  causal_chunk_count: 4     # >= 2 and divides n_fft; 0 only with whole_window_legacy
  gate_pooling: causal_prefix_mean   # default; whole_window_legacy reproduces the leaking R5 gate
```

## Layout

See §5 of Issue #5. Implemented so far: `src/config/` (config types and the YAML loader) and `tests/unit/`. Generated artifacts (`build/`, `checkpoints/`, `data/wikitext2-gpt2-tokens.safetensors`) are git-ignored.
