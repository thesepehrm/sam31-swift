# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-28

First public release: a Swift/MLX port of Meta's SAM 3.1, verified layer by layer against
mlx-vlm 0.7.3.

### Added

- `SAM31Model` actor: load the SAM 3.1 checkpoint, encode a `CGImage` or `CVPixelBuffer`
  (`32BGRA`, `420f`, `420v`) once, and prompt on the cached `FrameFeatures`.
  `encode(_: FrameInput)` accepts a pixel buffer wrapped in `FrameInput`, which Swift 6 callers
  need because `CVPixelBuffer` is not `Sendable`.
- Click, box, and box-plus-click segmentation, with iterative refinement from the previous result.
- Text detection (`detect(_:text:scoreThreshold:)`) with boxes, scores, masks, and non-maximum
  suppression. Text embeddings for the 16 most recent phrases are cached.
- `TrackingSession`: multi-object video tracking with SAM 3.1's multiplex memory tracker (up to 16
  objects per session). Objects can be added from clicks, boxes, or masks, refined on a tracked
  frame, and removed. `propagate` is a pull-based `AsyncThrowingStream` that stops on task
  cancellation and can resume from any frame.
- `PointMapper` to convert between source pixels and the 1008×1008 model space.
- `Mask`/`BinaryMask` results with bilinear upsampling to source size and `CGImage` export.
  `Mask.values` exposes the raw logits, and `Mask(logits:width:height:)` builds a mask from your own
  logits (for example for `TrackPrompt.mask`).
- `WeightDownloader`: resumable Hugging Face download with progress, cancellation, and `HF_TOKEN`
  support. The token is not forwarded when the Hub redirects to another host.
- Depends on mlx-swift 0.31.x from 0.31.4 (`.upToNextMinor(from: "0.31.4")`).
- `sam31-cli` with `download`, `segment`, `detect`, `track`, and `bench` commands, plus
  `scripts/run-cli.sh` to supply the MLX metallib for SwiftPM builds.
- Pure-Swift CLIP BPE tokenizer with the vocabulary bundled as a package resource.
- Pillow-compatible bilinear preprocessing, byte-identical to Pillow 12.3.0.
- Parity suites against mlx-vlm fixtures (`parity/dump.py`), a self-hosted parity workflow, and
  opt-in bf16 gate tests.

### Performance

Measured against mlx-vlm 0.7.3 on an M4 Pro (24 GB), fp32, 1080p clip, 120 frames:

- Tracking: 2460-2493 ms per frame median, against 2878-3287 ms in Python (15-24% faster).
- Peak MLX memory: 5.89 GB, against 11.77 GB in Python.
- Text detection, encode plus detect: about 1.6 s cold, against 2.1-2.3 s in Python.
- Backbone and model load: tied with Python, within noise.

### Known limits

- fp32 is the default. bf16 fails the tracking parity gate.
- Objects added or refined mid-video leave tracker memory after about 16 frames, as in mlx-vlm.
- No box-guided text detection in the public API.

[Unreleased]: https://github.com/thesepehrm/sam31-swift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thesepehrm/sam31-swift/releases/tag/v0.1.0
