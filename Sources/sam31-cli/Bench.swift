import CoreGraphics
import CoreVideo
import Foundation
import MLX
import SAM31

/// Mirrors `reference/bench_sam31_python.py` step for step, with the same labels, prompts, and
/// summary line, timing the public API call that does each step's work.
///
/// Where the public API fuses Python steps, the label says so:
/// - `encode` includes image preprocessing, which Python does before its backbone timer.
/// - The tracker frame features are computed lazily: the interactive neck on a frame's first click
///   (part of `click 1-cold`), the propagation neck when the object joins a tracking session.
/// - A click is `segment`: the SAM heads on a fresh one-object state. Python times
///   `track_step(is_init_cond_frame=True)` but evaluates only the masks, so MLX's lazy evaluation
///   skips its memory encoder there too; Python pays for it inside the first track step. Here it
///   is timed on its own (`add click 2 to session`), outside the track total.
/// - The public `detect` has no box prompt, so the box-guided row is skipped.
/// - The public `detect` applies NMS and sorts by score; Python keeps query order. The click point
///   is the centre of the highest-scoring "person" box and is printed so runs can be compared.
func runBench(_ options: Options) async throws {
    let url = try videoURL(options)
    let frameCount = try options.int("frames") ?? 120
    let outDir = try options.optional("out-dir").map(outputDirectory)

    let model = try await loadModel(options)

    var frames: [FrameInput] = []
    for try await frame in VideoFrames(url: url, limit: frameCount) { frames.append(frame) }
    guard let frame0 = frames.first, case .pixelBuffer(let first) = frame0 else {
        throw CLIError("\(url.path) has no frames")
    }
    let size = CGSize(width: CVPixelBufferGetWidth(first), height: CVPixelBufferGetHeight(first))
    let (W, H) = (Double(size.width), Double(size.height))
    print("video \(Int(W))x\(Int(H)), \(frames.count) frames")

    func save(_ mask: Mask, _ name: String) throws -> Double {
        let binary = mask.upsampled(to: size)
        if let outDir { try writePNG(binary, to: outDir.appendingPathComponent(name)) }
        return binary.coverage
    }

    // ---- text prompt (detector path) ----
    // Python's predict() preprocesses, runs the backbone, and detects on every call.
    _ = try await timed("text 'person' (cold) [encode+detect]") {
        try await model.detect(model.encode(frame0), text: "person", scoreThreshold: 0.3)
    }
    let people = try await timed("text 'person' (warm) [encode+detect]") {
        try await model.detect(model.encode(frame0), text: "person", scoreThreshold: 0.3)
    }
    print("  detections: \(people.count) \(round2(people.map(\.score)))")
    if let top = people.first {
        print("  text mask coverage: \(String(format: "%.3f", try save(top.mask, "out-text.png")))")
    }
    let skyline = try await timed("text 'city skyline' (warm backbone? no cache) [encode+detect]") {
        try await model.detect(model.encode(frame0), text: "city skyline", scoreThreshold: 0.3)
    }
    print("  detections: \(skyline.count)")

    // ---- box prompt ----
    if !people.isEmpty {
        print(
            "box-guided (text 'object' + box)".padding(toLength: 44, withPad: " ", startingAt: 0)
                + "      n/a")
        print("  skipped: the public detect() takes no box prompt")
    }

    // ---- click prompt (interactive tracker path) ----
    let side = model.configuration.imageSize
    print("  pixel_values [1, \(side), \(side), 3]")
    let features = try await timed("backbone (frame analysis) [+preprocess]") {
        try await model.encode(frame0)
    }
    print("tracker frame features".padding(toLength: 44, withPad: " ", startingAt: 0) + "      n/a")
    print("  fused into click 1-cold (computed on a frame's first prompt)")

    var (px, py) = (W * 0.73, H * 0.66)
    if let top = people.first { (px, py) = (Double(top.box.midX), Double(top.box.midY)) }
    print(String(format: "  click point (%.1f, %.1f), negative (%.1f, %.1f)", px, py, W * 0.3, H * 0.3))
    let mapper = PointMapper(sourceSize: size)
    func point(_ x: Double, _ y: Double, _ label: PromptPoint.Label) -> PromptPoint {
        let p = mapper.toModel(CGPoint(x: x, y: y))
        return PromptPoint(x: Float(p.x), y: Float(p.y), label: label)
    }

    func click(_ points: [PromptPoint], _ label: String, _ tag: String) async throws {
        let start = ContinuousClock.now
        let result = try await model.segment(features, prompt: .points(points))
        let ms = elapsedMS(since: start)
        let coverage = try save(result.mask, "out-click-\(tag).png")
        printTiming(label, ms, suffix: String(format: "   coverage %.3f", coverage))
    }
    let positive = point(px, py, .positive)
    let negative = point(W * 0.3, H * 0.3, .negative)
    try await click([positive], "click 1-cold [+interactive neck]", "1-cold")
    try await click([positive], "click 1-warm", "1-warm")
    try await click([positive, negative], "click 2-pos-neg", "2-pos-neg")
    let session = await model.makeTrackingSession()
    _ = try await timed("  (extra) add click 2 to session [+prop neck, mem enc]") {
        try await session.addObject(
            frameIndex: 0, features: features, prompt: .segment(.points([positive, negative])))
    }

    // ---- track the click through the clip (memory tracker) ----
    // Each frame's time covers preprocess + backbone + propagation neck + track step, as in Python.
    let rest = FrameList(frames: Array(frames.dropFirst()))
    var stream = session.propagate(rest, startingAt: 1, frameCount: frames.count).makeAsyncIterator()
    var perFrame: [Double] = []
    var last: TrackedFrame?
    let total = ContinuousClock.now
    while true {
        let start = ContinuousClock.now
        guard let tracked = try await stream.next() else { break }
        perFrame.append(elapsedMS(since: start))
        last = tracked
    }
    let totalSeconds = elapsedMS(since: total) / 1000
    if let object = last?.objects.values.first { _ = try save(object.mask, "out-track-last.png") }
    printTrackSummary(perFrame, totalSeconds: totalSeconds)
    print("peak MLX memory GB: \(String(format: "%.2f", Double(Memory.peakMemory) / 1e9))")
}

/// Times `Mask.upsampled(to:)` from a square logit grid to `width × height`, the call an app makes
/// once per click to draw the mask. Needs no weights: the logits are a seeded blob plus noise, so the
/// mask has real edges.
func runBenchUpsample(_ options: Options) throws {
    let side = try options.int("mask-size") ?? 288
    let width = try options.int("width") ?? 1920
    let height = try options.int("height") ?? 1080
    let runs = try options.int("runs") ?? 20

    var state: UInt64 = 0x5A31
    func noise() -> Float {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float(state >> 40) / Float(1 << 24) * 6 - 3
    }
    let center = Float(side) / 2
    let logits = (0..<(side * side)).map { i -> Float in
        let x = Float(i % side) - center
        let y = Float(i / side) - center
        return 40 - (x * x + y * y) / 200 + noise()
    }
    let mask = try Mask(logits: logits, width: side, height: side)
    let size = CGSize(width: width, height: height)

    let coverage = mask.upsampled(to: size).coverage  // also warms up
    var times: [Double] = []
    for _ in 0..<runs {
        let start = ContinuousClock.now
        let binary = mask.upsampled(to: size)
        times.append(elapsedMS(since: start))
        precondition(binary.bytes.count == width * height)
    }
    times.sort()
    print(
        "upsample \(side)x\(side) -> \(width)x\(height), \(runs) runs, coverage \(String(format: "%.3f", coverage))"
    )
    print(
        String(
            format: "  min %.2f ms, median %.2f ms, max %.2f ms", times[0], times[times.count / 2],
            times[times.count - 1]))
}
