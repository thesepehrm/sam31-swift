# Performance: Swift vs Python mlx-vlm

Both sides run the same fp32 checkpoint (`weights/sam3.1-bf16/model.safetensors` holds fp32
tensors) on the same 1080p clip, 120 frames:

- Python: `.venv/bin/python reference/bench_sam31_python.py 120` (mlx 0.32.2, mlx-vlm 0.7.3)
- Swift: `scripts/run-cli.sh bench --weights weights/sam3.1-bf16 --video clip.mp4 --frames 120`
  (mlx-swift 0.31.4, release build)

## Machine

- MacBook Pro, Apple M4 Pro, 24 GB, macOS 26.1.
- Runs were sequential and alternating: Python 1, Swift 1, Python 2, Swift 2 (2026-09-28, about
  10:00-10:25). No other benchmark or build was running.
- The machine was not idle. Desktop apps were open (one used about 1.4 cores), and about 12 GB of swap
  was in use from earlier work. Absolute times drift by 10-20% between sessions on this machine. An
  earlier, quieter session measured the backbone at about 1215 ms on both sides. Compare the two sides
  within this table, not with numbers from other sessions.

## Final table (fp32)

Two runs per side; lower is better. The Swift code is `44d12aa` (the three commits below).

| metric                                | Python run 1 / 2               | Swift run 1 / 2                      | verdict                         |
| ------------------------------------- | ------------------------------ | ------------------------------------ | ------------------------------- |
| load model                            | 648 / 657 ms                   | 657 / 668 ms                         | tie, within noise (Swift +1.5%) |
| text 'person' cold (encode + detect)  | 2275 / 2087 ms                 | 1574 / 1590 ms                       | Swift faster                    |
| text 'person' warm (encode + detect)  | 1899 / 1746 ms                 | 1622 / 1664 ms                       | Swift faster                    |
| text 'city skyline' (encode + detect) | 1905 / 2144 ms                 | 1855 / 2003 ms                       | Swift faster                    |
| backbone                              | 1517 / 1622 ms (no preprocess) | 1590 / 1563 ms (includes preprocess) | tie, within noise               |
| tracker frame features + click 1-cold | 108.5 + 14.9 / 111.2 + 11.8 ms | 64.4 / 64.9 ms                       | Swift faster                    |
| click 1-warm                          | 7.1 / 14.8 ms                  | 6.7 / 6.2 ms                         | tie / Swift                     |
| click 2-pos-neg                       | 6.3 / 7.9 ms                   | 6.5 / 6.5 ms                         | tie                             |
| track median ms/frame (119 frames)    | 2878 / 3287                    | 2460 / 2493                          | Swift 15-24% faster             |
| track p90 ms/frame                    | 3363 / 4117                    | 2625 / 2691                          | Swift faster                    |
| peak MLX memory                       | 11.77 / 11.77 GB               | 5.89 / 5.89 GB                       | Swift uses half                 |
| detections 'person' / 'city skyline'  | 1 [0.98] / 4                   | 1 [0.98] / 2                         | Swift applies NMS               |
| click coverages                       | 0.048 / 0.048 / 0.023          | 0.048 / 0.048 / 0.023                | identical                       |

How the rows line up:

- **Backbone.** Swift's `encode` includes preprocessing (BGRA to RGB, Pillow-compatible resize,
  normalize). Python preprocesses before starting its timer. An earlier instrumented run measured
  preprocessing at about 18 ms in Swift and 25 ms in Python. The ViT itself is the same MLX graph on
  both sides, so the row is a tie within run-to-run noise (±50 ms).
- **Click 1-cold.** Swift computes the interactive neck lazily on a frame's first prompt, so its
  cold click includes the work of Python's separate `tracker frame features` row.
- **Box-guided detect.** Python only. The public `detect()` has no box prompt. Not compared.
- **Load.** Swift reads, sanitizes and evaluates all 1961 tensors. It is within 10 ms of Python in
  both pairs, and a later Swift run measured 650 ms. This is below this machine's noise floor. No
  slowdown was found to fix.
- **Track.** Swift's per-frame time covers preprocess, backbone, propagation neck and track step,
  the same work as Python's loop. Swift also pays for its memory encoder on the click frame outside
  the loop (`add click 2 to session`, 95-97 ms). Python pays for it inside track frame 1.

## Optimizations

Each change was checked against the full fp32 parity suite with fixtures and `SAM31_FULL_MODEL=1`.

| #   | change                                                                                                        | before → after                                                                                                                                                                                                                                                          | decision            |
| --- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| 1   | Zero-pad memory-attention heads (8 × 32) to 64 so SDPA takes MLX's fused kernel (`fusedAttention`, `f9fa5a3`) | track median 2738 → 1959 ms/frame (40 frames); peak 11.70 → 5.89 GB                                                                                                                                                                                                     | **kept**            |
| 2   | Same padding for the DETR encoder self-attention (5184 queries, 8 × 32) (`9d8564c`)                           | detect after encode about 245 → 185 ms                                                                                                                                                                                                                                  | **kept**            |
| 3   | Port mlx-vlm's `separable_interpolate` Metal kernel for resizing (`44d12aa`)                                  | memory encoder about 55 → 35 ms/frame; SAM heads 8.6 → 7.6 ms                                                                                                                                                                                                           | **kept**            |
| 4   | `compile` each ViT block, with an uncompiled exact GELU inside (the WIP diff)                                 | first measured at 1216 → 1182-1190 ms backbone (about 2.5%) on a quieter machine. In today's interleaved A/B (3 × 40 frames each): track median 2095 / 2304 / 2341 ms eager vs 2168 / 2285 / 3027 ms compiled, backbone row 1519 / 1561 / 1617 vs 1418 / 1566 / 1555 ms | **dropped**         |
| 5   | Propagation-only neck while tracking                                                                          | already in the API (`trackerFeatures(interactive: false, propagation: true)`)                                                                                                                                                                                           | verified, no change |
| 6   | Reuse backbone features for frames already encoded                                                            | already in the API (`FrameInput.features`, cached neck heads on `FrameFeatures`)                                                                                                                                                                                        | verified, no change |
| 7   | `eval` only at stage boundaries; no `.item()` in hot loops                                                    | The remaining `eval` calls are per-layer ones that mirror mlx-vlm (they free attention intermediates) or stage boundaries. The only `.item()` is one per prompt encode                                                                                                  | verified, no change |
| 8   | Profile `add click 2 to session` (634 ms in the Task 16 bench)                                                | 82-110 ms on a quiet machine. The 634 ms came from contention                                                                                                                                                                                                           | no change needed    |
| 9   | Per-frame `Mask` host copy                                                                                    | 288 × 288 floats (330 KB) per object per frame, under 1 ms                                                                                                                                                                                                              | no change           |

Why change 4 was dropped:

- The A/B gain was smaller than the noise. Its outlier (3027 ms) lands on the compiled side, and one
  clean run per side cannot confirm the 2.5% seen earlier.
- It passed parity, but it is not bit-identical to eager: max abs diff was 9.4e-3 on values up to 230,
  about 4e-5 relative.
- The deadlock it works around is real. In mlx-swift 0.31.4, `CompiledFunction.call` takes the
  function's own `NSLock` and then the global `evalLock`, an `NSRecursiveLock`, and traces while
  holding both. Suppose a traced block calls MLXNN's `gelu`, which is itself `compile(shapeless:
true)`. It then takes `gelu`'s lock while holding `evalLock`. A second thread calling `gelu`
  directly takes `gelu`'s lock and then `evalLock`. That lock-order inversion deadlocked the
  parallel test run. The patch avoided it by using an uncompiled `geluErf` inside the block and by
  capturing the weights instead of passing them as compile inputs.

## Stage profile per track frame

From an earlier instrumented run on a quieter machine; this instrumentation is not committed.

| stage            | Swift                               | Python        |
| ---------------- | ----------------------------------- | ------------- |
| preprocess       | 18 ms                               | 25 ms         |
| backbone (ViT)   | 1215 ms                             | 1215-1330 ms  |
| propagation neck | 44 ms                               | 45 ms         |
| memory attention | about 520 ms (1060 before change 1) | about 1060 ms |
| SAM heads        | 7.6 ms                              | 7.1 ms        |
| memory encoder   | 35 ms (55 before change 3)          | 37 ms         |

The ViT backbone is about half of every frame on both sides. It is matmul and attention bound, and
the two sides run the same MLX graph for it.

## bf16 gate

`Tests/SAM31Tests/Parity/BF16GateTests.swift` runs a `.bfloat16` model through the public API
against the fp32 Python fixtures, on the end-to-end thresholds only. Enable it with
`TEST_RUNNER_SAM31_BF16_GATE=1`.

**Result: fail. The default stays `.float32`.**

Detect scenario (imageA, 'person'): **passes.**

```
det 0 | IoU 0.99970 | box Δ 0.069 px | score Δ 0.0000
```

Tracking scenario: **fails.** Object 1 is the mask-prompted object. Its IoU drops below 0.99 on 5 of
10 frames. Object scores stay within 0.003.

```
frame | IoU obj0 | IoU obj1 | max |Δ score|
f1 | 0.99860 | 0.99750 | 0.0007
f2 | 0.99434 | 0.98714 | 0.0018
f3 | 0.99503 | 0.98685 | 0.0019
f4 | 0.99673 | 0.98992 | 0.0025
f5.pre_refine | 0.99713 | 0.98947 | 0.0024
f5 refine obj0 | 0.99707 | -
f6 | 0.99857 | 0.99191 | 0.0023
f7 | 0.99675 | 0.99115 | 0.0026
f8 | 0.99599 | 0.98645 | 0.0015
f9 | 0.99808 | 0.99065 | 0.0028
```

`.bfloat16` casts only the weights. The inputs (pixels, prompts, masks) stay fp32, so most
activations promote to fp32. Even so, weight rounding alone fails the tracking gate. Casting the
activations to bf16 as well could only add error, so it was not tried. No bf16 speed numbers are
reported, because bf16 is not a shipped mode.

## Raw bench outputs

### Python, run 1

```
load model                                      647.6 ms
video 1920x1080, 120 frames
text 'person' (cold)                           2274.6 ms
text 'person' (warm)                           1898.5 ms
  detections: 1 [0.98]
  text mask coverage: 0.048
text 'city skyline' (warm backbone? no cache)   1904.6 ms
  detections: 4
box-guided (text 'object' + box)               1792.1 ms
  box detections: 0
  pixel_values (1, 1008, 1008, 3)
backbone (frame analysis)                      1516.5 ms
tracker frame features                          108.5 ms
click 1-cold                                     14.9 ms   coverage 0.048
click 1-warm                                      7.1 ms   coverage 0.048
click 2-pos-neg                                   6.3 ms   coverage 0.023
track 119 frames: total 354.2 s, median 2878 ms/frame, p90 3363 ms, = 0.3 fps (realtime would be 30)
peak MLX memory GB: 11.77
```

### Swift, run 1

```
load model                                      657.1 ms
video 1920x1080, 120 frames
text 'person' (cold) [encode+detect]           1574.1 ms
text 'person' (warm) [encode+detect]           1621.8 ms
  detections: 1 [0.98]
  text mask coverage: 0.048
text 'city skyline' (warm backbone? no cache) [encode+detect]   1854.5 ms
  detections: 2
box-guided (text 'object' + box)                  n/a
  skipped: the public detect() takes no box prompt
  pixel_values [1, 1008, 1008, 3]
backbone (frame analysis) [+preprocess]        1589.9 ms
tracker frame features                            n/a
  fused into click 1-cold (computed on a frame's first prompt)
  click point (1220.7, 752.8), negative (576.0, 324.0)
click 1-cold [+interactive neck]                 64.4 ms   coverage 0.048
click 1-warm                                      6.7 ms   coverage 0.048
click 2-pos-neg                                   6.5 ms   coverage 0.023
  (extra) add click 2 to session [+prop neck, mem enc]     96.9 ms
track 119 frames: total 292.2 s, median 2460 ms/frame, p90 2625 ms, = 0.4 fps (realtime would be 30)
peak MLX memory GB: 5.89
```

### Python, run 2

```
load model                                      656.7 ms
video 1920x1080, 120 frames
text 'person' (cold)                           2087.3 ms
text 'person' (warm)                           1746.1 ms
  detections: 1 [0.98]
  text mask coverage: 0.048
text 'city skyline' (warm backbone? no cache)   2143.8 ms
  detections: 4
box-guided (text 'object' + box)               1929.3 ms
  box detections: 0
  pixel_values (1, 1008, 1008, 3)
backbone (frame analysis)                      1621.9 ms
tracker frame features                          111.2 ms
click 1-cold                                     11.8 ms   coverage 0.048
click 1-warm                                     14.8 ms   coverage 0.048
click 2-pos-neg                                   7.9 ms   coverage 0.023
track 119 frames: total 406.7 s, median 3287 ms/frame, p90 4117 ms, = 0.3 fps (realtime would be 30)
peak MLX memory GB: 11.77
```

### Swift, run 2

```
load model                                      668.0 ms
video 1920x1080, 120 frames
text 'person' (cold) [encode+detect]           1590.3 ms
text 'person' (warm) [encode+detect]           1663.5 ms
  detections: 1 [0.98]
  text mask coverage: 0.048
text 'city skyline' (warm backbone? no cache) [encode+detect]   2003.2 ms
  detections: 2
box-guided (text 'object' + box)                  n/a
  skipped: the public detect() takes no box prompt
  pixel_values [1, 1008, 1008, 3]
backbone (frame analysis) [+preprocess]        1563.1 ms
tracker frame features                            n/a
  fused into click 1-cold (computed on a frame's first prompt)
  click point (1220.7, 752.8), negative (576.0, 324.0)
click 1-cold [+interactive neck]                 64.9 ms   coverage 0.048
click 1-warm                                      6.2 ms   coverage 0.048
click 2-pos-neg                                   6.5 ms   coverage 0.023
  (extra) add click 2 to session [+prop neck, mem enc]     94.7 ms
track 119 frames: total 298.3 s, median 2493 ms/frame, p90 2691 ms, = 0.4 fps (realtime would be 30)
peak MLX memory GB: 5.89
```
