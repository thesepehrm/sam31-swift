# Contributing

This package is a port. Its job is to match mlx-vlm's SAM 3.1 output, so every change is judged
against the Python reference first and against taste second.

## Setup

- macOS 14+ on Apple silicon, Xcode 26 (Swift 6.2 or later). Swift 6.1 cannot build mlx-swift 0.31.4.
- Weights, only for parity tests and the CLI:
  `hf download mlx-community/sam3.1-bf16 --local-dir weights/sam3.1-bf16` (3.3 GB).
- Python, only for the parity fixtures:
  `uv venv .venv --python 3.12 && uv pip install -r parity/requirements.txt`.

`weights/`, `parity/fixtures/` and `.venv/` are gitignored. Never commit them.

`Package.resolved` is committed on purpose. It pins mlx-swift to the version the parity suite was
run against, and CI builds that exact version. Update it in its own commit, after a full parity run.

## Build and test

Use xcodebuild. A bare `swift test` builds without MLX's Metal shaders and fails with "Failed to
load the default metallib" ([mlx-swift#488](https://github.com/ml-explore/mlx-swift/issues/488)).

```bash
xcodebuild test -scheme sam31-swift-Package -destination 'platform=macOS' -derivedDataPath .build/xcode -quiet
```

Run one suite with `-only-testing:SAM31Tests/<SuiteName>`.

Lint before committing. CI runs the same command:

```bash
swift format lint --recursive --strict Sources Tests
```

## Parity tests

The suites in `Tests/SAM31Tests/Parity/` read `SAM31_WEIGHTS` and `SAM31_FIXTURES`. When either is
unset they are disabled, not failed. xcodebuild only forwards variables with the `TEST_RUNNER_`
prefix:

```bash
TEST_RUNNER_SAM31_WEIGHTS=$PWD/weights/sam3.1-bf16 \
TEST_RUNNER_SAM31_FIXTURES=$PWD/parity/fixtures \
TEST_RUNNER_SAM31_FULL_MODEL=1 \
  xcodebuild test -scheme sam31-swift-Package -destination 'platform=macOS' -derivedDataPath .build/xcode -quiet
```

- `SAM31_FULL_MODEL=1` adds the full weight-key coverage check.
- `SAM31_BF16_GATE=1` runs the bf16 end-to-end gate (it currently fails on tracking; see
  [docs/perf.md](docs/perf.md)).

Generate the fixtures with `parity/dump.py`. [parity/README.md](parity/README.md) lists every key
and the exact tracking scenario the tests replay.

The `Parity` workflow in `.github/workflows/parity.yml` runs the whole thing on a self-hosted Apple
silicon runner, on demand.

## Running the CLI

```bash
scripts/run-cli.sh <command> ...
```

The script builds the MLX metallib once with xcodebuild and copies it next to the SwiftPM binary,
for the same reason as above. It rebuilds the metallib when `Package.resolved` changes.

## Every new op needs a parity assert

If you port a module, add or change math, or change an op's order, add an assert that compares its
output with the mlx-vlm tensor at the same point. Dump the tensor in `parity/dump.py` if it is not
there yet. A change that only passes the end-to-end check is not done: small per-layer drift adds up
over 32 ViT blocks and a 16-frame memory.

Tolerances:

- fp32 per layer: `rtol 1e-3, atol 1e-4`. Anything looser needs a comment giving the reason.
- End to end: mask IoU ≥ 0.99, boxes within 1 px in 1008 space, scores within 0.01.

## Porting conventions

These keep parameter keys identical to Python, so checkpoints load without a key map.

| Python (mlx)                                                 | Swift (mlx-swift)                                                                                        |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `class X(nn.Module)` + `__call__`                            | `final class X: Module` + `func callAsFunction(...)` (internal access)                                   |
| child module `self.q_proj = nn.Linear(...)`                  | `@ModuleInfo(key: "q_proj") var qProj: Linear`                                                           |
| list of modules `self.layers = [...]`                        | `@ModuleInfo(key: "layers") var layers: [VitBlock]`                                                      |
| raw array param `self.foo = mx.zeros(...)`                   | `@ParameterInfo(key: "foo") var foo: MLXArray`                                                           |
| non-parameter buffer or cache                                | a plain property that is not an `MLXArray`, or a private one whose key starts with `_`                   |
| `nn.Linear(i, o, bias=False)`                                | `Linear(i, o, bias: false)`                                                                              |
| `nn.Conv2d(i, o, k, stride, padding)` (NHWC)                 | `Conv2d(inputChannels: i, outputChannels: o, kernelSize: .init(k), stride: .init(s), padding: .init(p))` |
| `nn.ConvTranspose2d`                                         | `ConvTransposed2d(...)`                                                                                  |
| `nn.LayerNorm(d, eps=e)`                                     | `LayerNorm(dimensions: d, eps: e)`                                                                       |
| `nn.GroupNorm(g, d, pytorch_compatible=True)`                | `GroupNorm(groupCount: g, dimensions: d, pytorchCompatible: true)`, with `eps`/`affine` copied exactly   |
| `nn.Embedding(n, d)`                                         | `Embedding(embeddingCount: n, dimensions: d)`                                                            |
| `nn.gelu` / `nn.gelu_approx`                                 | `gelu` / `geluApproximate` (match exactly)                                                               |
| `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask)` | `MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask:)`                                     |
| `x.reshape(B, N, H, D).transpose(0, 2, 1, 3)`                | `x.reshaped(B, N, H, D).transposed(0, 2, 1, 3)`                                                          |
| `mx.concatenate([a, b], axis=1)`                             | `concatenated([a, b], axis: 1)`                                                                          |
| `x[..., 0]`                                                  | `x[.ellipsis, 0]`                                                                                        |
| `x[:, None]`                                                 | `expandedDimensions(x, axis: 1)` (see the `newAxis` bug below)                                           |
| `mx.eval(x)`                                                 | `eval(x)`                                                                                                |

Rules:

1. Each Swift file's header names its source, for example
   `// Port of mlx_vlm/models/sam3/text_encoder.py (mlx-vlm 0.7.3)`.
2. Keep Python's order of ops, dtypes and epsilon values exactly. Do not tidy the math: reordering
   adds float drift.
3. Configs are `Codable` structs with snake_case `CodingKeys`. Every field Python reads defaults to
   Python's dataclass default, and any `__post_init__` override is ported.
4. Python `dict` outputs become small structs with named fields.
5. Weight key names live only in `@ModuleInfo` and `@ParameterInfo`. Never repeat them as strings.
6. Where Python calls mlx's additive causal mask helper, reuse `CLIPTextModel.additiveCausalMask(_:)`.
   MLXNN's version uses -1e9; Python uses the float32 minimum.

### The `newAxis` bug in mlx-swift 0.31.4

A multi-operand subscript that contains `.newAxis` silently drops the new axis unless it also has an
ellipsis or integer index and more operands than the array's rank. For example, both 2-D
`x[0..., .newAxis]` and 3-D `y[0..., .newAxis]` return the input unchanged. Use
`expandedDimensions(x, axis:)` for new axes in multi-operand subscripts. A single-operand
`x[.newAxis]` works.

## CI

`.github/workflows/ci.yml` lints, builds the CLI in release, and runs the unit tests on a
GitHub-hosted macOS runner. Those runners are virtual machines, and the unit tests that run MLX on
the GPU may fail there if the VM exposes no usable Metal device. If that happens, the failure is the
runner's, not the change's; confirm locally before merging.

## Commits

Conventional Commits (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `chore:`). Keep perf changes in
separate commits, each with before and after numbers and a passing parity run.
