# sam31-swift

Meta's SAM 3.1 in pure Swift on Apple silicon. Built on MLX, no Python.

```swift
import SAM31

let model = try await SAM31Model.load(from: URL(filePath: "weights/sam3.1-bf16"))
let frame = try await model.encode(image)  // CGImage, or .pixelBuffer(buffer)

// Click: coordinates are in 1008×1008 model space (PointMapper converts from pixels).
let clicked = try await model.segment(frame, prompt: .points([.init(x: 504, y: 380, label: .positive)]))

// Text: every "person" in the frame, with box, score and mask.
let people = try await model.detect(frame, text: "person")

// Track: follow them through the rest of the video.
let session = await model.makeTrackingSession()
for person in people {
    _ = try await session.addObject(frameIndex: 0, features: frame, prompt: .mask(person.mask))
}
for try await tracked in session.propagate(videoFrames, startingAt: 1, frameCount: 120) {
    print(tracked.frameIndex, tracked.objects.count)
}
```

Against Python mlx-vlm on the same checkpoint and 1080p clip (M4 Pro): tracking takes 2.5 s per
frame instead of 2.9-3.3 s, and peak memory is 5.9 GB instead of 11.8 GB.

```bash
swift package add-dependency https://github.com/thesepehrm/sam31-swift --from 0.1.0
hf download mlx-community/sam3.1-bf16 --local-dir weights/sam3.1-bf16
```

---

## Requirements

- macOS 14 or later on Apple silicon.
- Swift 6.2 or later (Xcode 26). mlx-swift 0.31.4 does not build with Swift 6.1.
- 16 GB of unified memory is comfortable. Tracking a 1080p clip peaks at 5.9 GB in fp32.
- 3.3 GB of disk for the weights.

## Install

Add the package to `Package.swift`:

```swift
.package(url: "https://github.com/thesepehrm/sam31-swift", from: "0.1.0"),
```

Then add `.product(name: "SAM31", package: "sam31-swift")` to your target's dependencies. In Xcode,
use File > Add Package Dependencies with the same URL.

The only runtime dependency is [mlx-swift](https://github.com/ml-explore/mlx-swift), 0.31.x from 0.31.4
(`.upToNextMinor(from: "0.31.4")`).

## Weights

The package does not ship weights. Download them once, with either command:

```bash
scripts/run-cli.sh download --dir weights/sam3.1-bf16
hf download mlx-community/sam3.1-bf16 --local-dir weights/sam3.1-bf16
```

Both fetch `config.json` and `model.safetensors` from
[mlx-community/sam3.1-bf16](https://huggingface.co/mlx-community/sam3.1-bf16). Despite the name, that
checkpoint stores fp32 tensors. The CLI download resumes after an interrupt and sends `HF_TOKEN` if
it is set.

From an app, call `WeightDownloader().download(to:progress:)` to do the same thing.

**License note.** Meta releases the SAM 3.1 weights under the
[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). You download them yourself
and accept that license. This repository does not redistribute them.

## API

All prompt coordinates are in the 1008×1008 space the model sees. `PointMapper(sourceSize:)` converts
points and boxes from source pixels, and `Detection.box` comes back in source pixels.

### Load and encode

```swift
let model = try await SAM31Model.load(from: weightsURL)  // dtype: .float32 by default
let frame = try await model.encode(cgImage)              // CGImage
let video = try await model.encode(.pixelBuffer(buffer)) // 32BGRA / 420f / 420v CVPixelBuffer
```

`CVPixelBuffer` is not `Sendable`, so under Swift 6 pass pixel buffers wrapped in `FrameInput`, as
above. Don't write to the buffer until `encode` returns.

`encode` runs the ViT backbone once. Keep the `FrameFeatures` while you prompt on that frame: clicks,
text and tracking all reuse it, and each neck head is computed on first use and cached.

### Click and box segmentation

```swift
let mapper = PointMapper(sourceSize: frame.sourceSize)
let p = mapper.toModel(CGPoint(x: 1220, y: 750))
let result = try await model.segment(frame, prompt: .points([
    PromptPoint(x: Float(p.x), y: Float(p.y), label: .positive)]))

// Refine: pass every click so far plus the last result.
let better = try await model.segment(frame, prompt: .points(allClicks), previous: result)

// Box, or box plus clicks.
let boxed = try await model.segment(frame, prompt: .box(mapper.toModel(sourceRect)))

let mask = result.mask.upsampled(to: frame.sourceSize)  // BinaryMask, 0 or 255 per pixel
let image = mask.cgImage()
```

`SegmentResult` carries the `mask`, a predicted IoU `score` and an `objectScore`. `Mask.values` holds
the raw logits at model resolution (`width × height`, row-major; 288×288 for a click or box), where a
logit above 0 is foreground. `Mask(logits:width:height:)` builds one from your own logits, for
example to seed `TrackPrompt.mask`.

### Text detection

```swift
let detections = try await model.detect(frame, text: "red car", scoreThreshold: 0.5)
for d in detections {
    print(d.box, d.score, d.mask.upsampled(to: frame.sourceSize).coverage)
}
```

Results are sorted by score, after non-maximum suppression. The model caches the text embeddings of
the 16 most recent phrases.

### Video tracking

```swift
let session = await model.makeTrackingSession()
let a = try await session.addObject(
    frameIndex: 0, features: frame0, prompt: .segment(.points([click])))
let b = try await session.addObject(
    frameIndex: 0, features: frame0, prompt: .mask(detections[0].mask))

// videoFrames: any AsyncSequence of FrameInput (.image, .pixelBuffer or .features).
let task = Task {
    for try await tracked in session.propagate(videoFrames, startingAt: 1, frameCount: 300) {
        for (id, object) in tracked.objects where object.isVisible {
            draw(object.mask.upsampled(to: size), for: id, at: tracked.frameIndex)
        }
    }
}

// Correct an object on a tracked frame, or drop it.
let fixed = try await session.refine(a, frameIndex: 40, features: frame40, prompt: .points(clicks))
try await session.remove(b)

task.cancel()  // stops before the next frame; the session stays usable
```

`propagate` pulls frames only as you consume results. Cancelling the consuming task ends the stream
with `CancellationError`, and you can resume later from any frame.

### Memory

`model.setCacheLimit(bytes:)` caps the buffer cache MLX keeps for reuse. Lower it to shrink the
process footprint at some cost in speed.

## CLI

The `sam31-cli` executable covers download, segment, detect, track and a benchmark. Run it through
the wrapper script:

```bash
scripts/run-cli.sh segment --weights weights/sam3.1-bf16 --image photo.jpg --point 1400,700 --out mask.png
scripts/run-cli.sh detect  --weights weights/sam3.1-bf16 --image photo.jpg --text person --out-dir out
scripts/run-cli.sh track   --weights weights/sam3.1-bf16 --video clip.mp4 --point 1220,750 --frames 60 --out-dir out
scripts/run-cli.sh bench   --weights weights/sam3.1-bf16 --video clip.mp4 --frames 120
```

CLI coordinates are source pixels. Add `,0` to a point for a negative click. `--help` lists every
option.

**Why the script exists.** SwiftPM does not compile mlx-swift's Metal shaders
([mlx-swift#488](https://github.com/ml-explore/mlx-swift/issues/488)), so a plain `swift run` binary
fails with "Failed to load the default metallib". The script builds the metallib once with
xcodebuild, copies it next to the SwiftPM binary as `mlx.metallib`, then runs the CLI. Apps built
with Xcode compile the shaders themselves and need none of this.

## Performance

MacBook Pro, M4 Pro, 24 GB, macOS 26.1. Both sides run the same fp32 checkpoint on the same 1920×1080
clip, 120 frames. Python is mlx 0.32.2 with mlx-vlm 0.7.3; Swift is mlx-swift 0.31.4, release build.
Two alternating runs per side. Lower is better.

| metric                                | Python run 1 / 2               | Swift run 1 / 2                      |
| ------------------------------------- | ------------------------------ | ------------------------------------ |
| load model                            | 648 / 657 ms                   | 657 / 668 ms                         |
| text 'person' cold (encode + detect)  | 2275 / 2087 ms                 | 1574 / 1590 ms                       |
| text 'person' warm (encode + detect)  | 1899 / 1746 ms                 | 1622 / 1664 ms                       |
| text 'city skyline' (encode + detect) | 1905 / 2144 ms                 | 1855 / 2003 ms                       |
| backbone                              | 1517 / 1622 ms (no preprocess) | 1590 / 1563 ms (includes preprocess) |
| tracker frame features + click 1-cold | 108.5 + 14.9 / 111.2 + 11.8 ms | 64.4 / 64.9 ms                       |
| click 1-warm                          | 7.1 / 14.8 ms                  | 6.7 / 6.2 ms                         |
| click 2-pos-neg                       | 6.3 / 7.9 ms                   | 6.5 / 6.5 ms                         |
| track median ms/frame (119 frames)    | 2878 / 3287                    | 2460 / 2493                          |
| track p90 ms/frame                    | 3363 / 4117                    | 2625 / 2691                          |
| peak MLX memory                       | 11.77 / 11.77 GB               | 5.89 / 5.89 GB                       |

The ViT backbone is about half of every frame on both sides, and it is the same MLX graph. Swift wins
on tracking mostly because memory attention runs on MLX's fused attention kernel (about 520 ms per
frame instead of 1060 ms), which also halves peak memory. The machine was not idle, so absolute times
drift 10-20% between sessions. [docs/perf.md](docs/perf.md) has the raw output, the optimizations
tried, and the stage profile.

## How parity is verified

Every layer is checked against mlx-vlm 0.7.3, the Python reference this port follows line by line.

1. `parity/dump.py` runs mlx-vlm on fixed inputs (a video frame, a synthetic image, a 10-frame clip)
   and saves inputs, intermediate tensors and outputs.
2. The Swift suites in `Tests/SAM31Tests/Parity/` load the same weights, feed the same inputs, and
   compare. They cover weight keys, the tokenizer, the text encoder, the vision backbone and neck,
   the detector, click and box prompts, a two-object tracking scenario with a mid-clip refine, and
   the public API.
3. Per-layer fp32 tolerance is `rtol 1e-3, atol 1e-4`. End to end, masks must reach IoU ≥ 0.99,
   boxes must land within 1 px in 1008 space, and scores within 0.01.

The suites skip themselves when the fixtures are missing, so ordinary test runs need no weights.
[parity/README.md](parity/README.md) explains how to generate the fixtures and run them.

## Known limits

- **fp32 by default.** `.bfloat16` loads and passes the detection gate, but tracking fails it (mask
  IoU drops to 0.986 on some frames), so it is not the default.
- **16 objects per tracking session.** The tracker runs a session's objects as one multiplexed
  bucket. Start another session for more.
- **Mid-video prompts age out.** An object added or refined on a propagated frame leaves memory after
  about 16 frames. mlx-vlm behaves the same way. Only the first prompted frame stays a conditioning
  frame.
- **No box-guided text detection.** `detect` takes text only.
- **BT.709 for 4:2:0 buffers.** `420f` and `420v` pixel buffers are decoded with the BT.709 matrix,
  whatever their color attachments say.
- **macOS 14+ on Apple silicon only.** No iOS, no Intel.

## Why it exists

Built for [Cask Studio](https://cask.studio), a macOS editor for AI-generated video, and released so
any Mac app can use it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: build and test with xcodebuild, follow the porting
conventions, and add a parity assert for every new op.

## License and acknowledgements

- The code is MIT, © 2026 Sepehr Mohammadi. See [LICENSE](LICENSE).
- The SAM 3.1 weights are Meta's, under the SAM License. You download them; this repository does not
  redistribute them.
- The bundled CLIP vocabulary and merges come from OpenAI's CLIP, under the MIT license.

This port follows [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)'s SAM 3.1 implementation and runs on
[mlx-swift](https://github.com/ml-explore/mlx-swift). SAM 3 and SAM 3.1 are the work of Meta FAIR.
The image resize reimplements Pillow's bilinear filter. [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) has
the full notices.
