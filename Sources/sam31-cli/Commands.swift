import CoreGraphics
import Foundation
import ImageIO
import MLX
import SAM31
import UniformTypeIdentifiers

// MARK: - Errors and usage

/// A user-facing CLI failure (bad input file, unreadable video). Printed without a stack trace.
struct CLIError: Error {
    let message: String
    init(_ message: String) { self.message = message }
}

/// Bad command-line arguments: printed with the usage text.
struct UsageError: Error {
    let message: String
    init(_ message: String) { self.message = message }
}

let usage = """
    usage: sam31-cli <command> [options]

    commands:
      download [--dir weights/sam3.1-bf16] [--repo mlx-community/sam3.1-bf16] [--revision main]
          Download config.json and model.safetensors from the Hugging Face Hub. Complete files are
          skipped, interrupted ones resume. Sends HF_TOKEN as a bearer token only when it is set.
      segment --weights D --image I (--point x,y[,0|1])... [--box x0,y0,x1,y1] --out mask.png
          Segment the object at the clicks (label 1 = object, 0 = background; default 1) and/or box.
          Coordinates are source-image pixels.
      detect --weights D --image I --text "the dancer" [--threshold 0.5] --out-dir O
          Find every object matching the text; writes one mask PNG per detection.
      track --weights D --video V (--point x,y[,0|1])... [--frames N] --out-dir O
          Click an object on frame 0 and track it through the next N-1 frames (default: whole video);
          writes one mask PNG per tracked frame.
      bench --weights D --video V [--frames 120] [--out-dir O]
          Mirror reference/bench_sam31_python.py step for step and print its timings.
      bench-upsample [--mask-size 288] [--width 1920] [--height 1080] [--runs 20]
          Time Mask.upsampled(to:) on random logits. Needs no weights.

    common options:
      --dtype float32|bfloat16   model precision (default float32)
      -h, --help                 show this help
    """

/// A readable message for errors the library throws.
func describe(_ error: SAM31Error) -> String {
    switch error {
    case .weightsNotFound(let url): "weights not found in \(url.path) (run `sam31-cli download`)"
    case .missingParameter(let key): "checkpoint is missing parameter \(key)"
    case .shapeMismatch(let key, let expected, let got):
        "checkpoint tensor \(key) has shape \(got), expected \(expected)"
    case .unusedParameters(let keys):
        "checkpoint has \(keys.count) unused tensors (first: \(keys.first ?? "-"))"
    case .invalidConfig(let message): "invalid config: \(message)"
    case .invalidPrompt(let message): "invalid prompt: \(message)"
    case .invalidImage(let message): "invalid image: \(message)"
    case .unknownObject(let id): "unknown object \(id)"
    case .download(let message): "download failed: \(message)"
    }
}

// MARK: - Argument parsing

/// `--name value` options. Every option takes exactly one value; options may repeat.
struct Options {
    private var values: [String: [String]] = [:]

    init(_ arguments: ArraySlice<String>, allowed: Set<String>) throws {
        var rest = arguments[...]
        while let name = rest.popFirst() {
            guard name.hasPrefix("--"), name.count > 2 else {
                throw UsageError("unexpected argument '\(name)'")
            }
            let key = String(name.dropFirst(2))
            guard allowed.contains(key) else { throw UsageError("unknown option '\(name)'") }
            guard let value = rest.popFirst() else { throw UsageError("\(name) needs a value") }
            values[key, default: []].append(value)
        }
    }

    func all(_ key: String) -> [String] { values[key] ?? [] }

    func optional(_ key: String) throws -> String? {
        let found = all(key)
        guard found.count <= 1 else { throw UsageError("--\(key) given more than once") }
        return found.first
    }

    func required(_ key: String) throws -> String {
        guard let value = try optional(key) else { throw UsageError("missing --\(key)") }
        return value
    }

    func int(_ key: String) throws -> Int? {
        guard let raw = try optional(key) else { return nil }
        guard let value = Int(raw), value > 0 else { throw UsageError("--\(key) must be a positive integer") }
        return value
    }

    func float(_ key: String) throws -> Float? {
        guard let raw = try optional(key) else { return nil }
        guard let value = Float(raw), value.isFinite else { throw UsageError("--\(key) must be a number") }
        return value
    }

    func dtype() throws -> SAM31DType {
        switch try optional("dtype") ?? "float32" {
        case "float32", "fp32": .float32
        case "bfloat16", "bf16": .bfloat16
        case let other: throw UsageError("--dtype must be float32 or bfloat16, not '\(other)'")
        }
    }
}

/// A click in source pixels: `x,y` or `x,y,label`.
func parsePoint(_ raw: String) throws -> (CGPoint, PromptPoint.Label) {
    let parts = raw.split(separator: ",", omittingEmptySubsequences: false).map {
        $0.trimmingCharacters(in: .whitespaces)
    }
    guard parts.count == 2 || parts.count == 3,
        let x = Double(parts[0]), let y = Double(parts[1]), x.isFinite, y.isFinite
    else { throw UsageError("--point must be x,y or x,y,label, not '\(raw)'") }
    var label = PromptPoint.Label.positive
    if parts.count == 3 {
        guard let raw = Int32(parts[2]), let parsed = PromptPoint.Label(rawValue: raw) else {
            throw UsageError("point label must be 0 (background) or 1 (object), not '\(parts[2])'")
        }
        label = parsed
    }
    return (CGPoint(x: x, y: y), label)
}

/// A box in source pixels: `x0,y0,x1,y1`.
func parseBox(_ raw: String) throws -> CGRect {
    let values = raw.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
    guard values.count == 4, values.allSatisfy(\.isFinite), values[2] > values[0], values[3] > values[1]
    else {
        throw UsageError("--box must be x0,y0,x1,y1 with x1 > x0 and y1 > y0, not '\(raw)'")
    }
    return CGRect(x: values[0], y: values[1], width: values[2] - values[0], height: values[3] - values[1])
}

// MARK: - Shared helpers

/// Prints `label` padded to 44 columns and the time in ms, as the Python benchmark does.
func printTiming(_ label: String, _ ms: Double, suffix: String = "") {
    let padded = label.count >= 44 ? label : label.padding(toLength: 44, withPad: " ", startingAt: 0)
    print(padded + " " + String(format: "%8.1f ms", ms) + suffix)
    fflush(stdout)
}

/// Milliseconds since `start`.
func elapsedMS(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
}

/// Runs `body`, prints its time under `label`, and returns its result.
func timed<T>(_ label: String, _ body: () async throws -> T) async rethrows -> T {
    let start = ContinuousClock.now
    let result = try await body()
    printTiming(label, elapsedMS(since: start))
    return result
}

func loadImage(_ path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: url.path) else { throw CLIError("no such image: \(path)") }
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { throw CLIError("cannot decode image: \(path)") }
    return image
}

func writePNG(_ mask: BinaryMask, to url: URL) throws {
    guard let image = mask.cgImage() else { throw CLIError("mask is empty; nothing to write to \(url.path)") }
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard
        let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw CLIError("cannot write \(url.path)") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw CLIError("cannot write \(url.path)") }
}

func outputDirectory(_ path: String) throws -> URL {
    let url = URL(fileURLWithPath: path, isDirectory: true)
    do {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    } catch {
        throw CLIError("cannot create \(path): \(error.localizedDescription)")
    }
    return url
}

func loadModel(_ options: Options) async throws -> SAM31Model {
    let weights = URL(fileURLWithPath: try options.required("weights"), isDirectory: true)
    let dtype = try options.dtype()
    return try await timed("load model") { try await SAM31Model.load(from: weights, dtype: dtype) }
}

func videoURL(_ options: Options) throws -> URL {
    let path = try options.required("video")
    guard FileManager.default.fileExists(atPath: path) else { throw CLIError("no such video: \(path)") }
    return URL(fileURLWithPath: path)
}

/// Source-pixel clicks in model-input space.
func modelPoints(_ raw: [String], mapper: PointMapper) throws -> [PromptPoint] {
    try raw.map(parsePoint).map { point, label in
        let p = mapper.toModel(point)
        return PromptPoint(x: Float(p.x), y: Float(p.y), label: label)
    }
}

func round2(_ values: [Float]) -> String {
    "[" + values.map { String(format: "%.2f", $0) }.joined(separator: " ") + "]"
}

// MARK: - Commands

func runDownload(_ options: Options) async throws {
    let directory = URL(
        fileURLWithPath: try options.optional("dir") ?? "weights/sam3.1-bf16", isDirectory: true)
    let downloader = WeightDownloader(
        repo: try options.optional("repo") ?? "mlx-community/sam3.1-bf16",
        revision: try options.optional("revision") ?? "main")
    let meter = ProgressMeter()
    // Ctrl-C cancels the download task; the .partial file stays for the next run to resume.
    let task = Task {
        try await downloader.download(to: directory) { meter.show($0) }
    }
    signal(SIGINT, SIG_IGN)
    let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    interrupt.setEventHandler { task.cancel() }
    interrupt.resume()
    defer { interrupt.cancel() }
    do {
        try await task.value
    } catch is CancellationError {
        FileHandle.standardError.write("\ncancelled; run the command again to resume\n".data(using: .utf8)!)
        throw CLIError("download cancelled")
    }
    FileHandle.standardError.write("\n".data(using: .utf8)!)
    print("weights in \(directory.path)")
}

/// Prints download progress to stderr at most once per 0.1 %.
private final class ProgressMeter: @unchecked Sendable {
    private let lock = NSLock()
    private var last = -1

    func show(_ fraction: Double) {
        let permille = Int(fraction * 1000)
        lock.lock()
        defer { lock.unlock() }
        guard permille != last else { return }
        last = permille
        FileHandle.standardError.write(
            String(format: "\rdownloading %5.1f%%", fraction * 100).data(using: .utf8)!)
    }
}

func runSegment(_ options: Options) async throws {
    let image = try loadImage(try options.required("image"))
    let out = URL(fileURLWithPath: try options.required("out"))
    let rawPoints = options.all("point")
    let rawBox = try options.optional("box")
    guard !rawPoints.isEmpty || rawBox != nil else { throw UsageError("segment needs --point or --box") }
    let size = CGSize(width: image.width, height: image.height)
    let mapper = PointMapper(sourceSize: size)
    let points = try modelPoints(rawPoints, mapper: mapper)
    let box = try rawBox.map { mapper.toModel(try parseBox($0)) }
    let prompt: SegmentPrompt =
        switch box {
        case .some(let box) where points.isEmpty: .box(box)
        case .some(let box): .boxAndPoints(box, points)
        case .none: .points(points)
        }

    let model = try await loadModel(options)
    let frame = try await timed("encode (preprocess + backbone)") { try await model.encode(image) }
    let result = try await timed("segment (+ interactive neck, cold)") {
        try await model.segment(frame, prompt: prompt)
    }
    let mask = result.mask.upsampled(to: size)
    try writePNG(mask, to: out)
    print(
        String(
            format: "score %.3f  object score %.3f  coverage %.3f", result.score, result.objectScore,
            mask.coverage))
    print("wrote \(out.path)")
}

func runDetect(_ options: Options) async throws {
    let image = try loadImage(try options.required("image"))
    let text = try options.required("text")
    let threshold = try options.float("threshold") ?? 0.5
    let outDir = try outputDirectory(try options.required("out-dir"))
    let size = CGSize(width: image.width, height: image.height)

    let model = try await loadModel(options)
    let frame = try await timed("encode (preprocess + backbone)") { try await model.encode(image) }
    let detections = try await timed("detect '\(text)' (+ det neck + text, cold)") {
        try await model.detect(frame, text: text, scoreThreshold: threshold)
    }
    _ = try await timed("detect '\(text)' (warm)") {
        try await model.detect(frame, text: text, scoreThreshold: threshold)
    }
    print("detections: \(detections.count) \(round2(detections.map(\.score)))")
    for (i, d) in detections.enumerated() {
        let mask = d.mask.upsampled(to: size)
        let url = outDir.appendingPathComponent(String(format: "detection-%02d.png", i))
        try writePNG(mask, to: url)
        let b = d.box
        print(
            String(
                format: "  #%d score %.3f  box [%.1f, %.1f, %.1f, %.1f]  coverage %.3f  %@", i, d.score,
                b.minX, b.minY, b.maxX, b.maxY, mask.coverage, url.lastPathComponent))
    }
    print("wrote \(detections.count) masks to \(outDir.path)")
}

func runTrack(_ options: Options) async throws {
    let url = try videoURL(options)
    let rawPoints = options.all("point")
    guard !rawPoints.isEmpty else { throw UsageError("track needs at least one --point") }
    let frameLimit = try options.int("frames")
    let outDir = try outputDirectory(try options.required("out-dir"))

    let model = try await loadModel(options)
    var first = VideoFrames(url: url, limit: 1).makeAsyncIterator()
    guard let input = try await first.next() else {
        throw CLIError("\(url.path) has no frames")
    }
    let frame0 = try await timed("encode frame 0") { try await model.encode(input) }
    let size = frame0.sourceSize
    let points = try modelPoints(rawPoints, mapper: PointMapper(sourceSize: size))
    let session = await model.makeTrackingSession()
    let id = try await timed("add object on frame 0 (+ tracker features)") {
        try await session.addObject(frameIndex: 0, features: frame0, prompt: .segment(.points(points)))
    }
    let click = try await model.segment(frame0, prompt: .points(points)).mask.upsampled(to: size)
    try writePNG(click, to: outDir.appendingPathComponent("frame-00000.png"))

    let rest = VideoFrames(url: url, start: 1, limit: frameLimit.map { $0 - 1 })
    var stream = session.propagate(rest, startingAt: 1, frameCount: frameLimit).makeAsyncIterator()
    var perFrame: [Double] = []
    let total = ContinuousClock.now
    while true {
        let start = ContinuousClock.now
        guard let tracked = try await stream.next() else { break }
        let ms = elapsedMS(since: start)
        perFrame.append(ms)
        guard let object = tracked.objects[id] else { continue }
        let mask = object.mask.upsampled(to: size)
        try writePNG(
            mask, to: outDir.appendingPathComponent(String(format: "frame-%05d.png", tracked.frameIndex)))
        printTiming(
            "frame \(tracked.frameIndex)", ms,
            suffix: String(
                format: "   %@ score %.3f coverage %.3f", object.isVisible ? "visible" : "hidden ",
                object.score,
                mask.coverage))
    }
    printTrackSummary(perFrame, totalSeconds: elapsedMS(since: total) / 1000)
    print("wrote \(perFrame.count + 1) masks to \(outDir.path)")
}

/// numpy `percentile` with linear interpolation.
func percentile(_ values: [Double], _ q: Double) -> Double {
    guard !values.isEmpty else { return .nan }
    let sorted = values.sorted()
    let rank = q / 100 * Double(sorted.count - 1)
    let lo = Int(rank.rounded(.down))
    let hi = min(lo + 1, sorted.count - 1)
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - Double(lo))
}

func printTrackSummary(_ perFrame: [Double], totalSeconds: Double) {
    guard !perFrame.isEmpty else {
        print("track 0 frames")
        return
    }
    print(
        String(
            format:
                "track %d frames: total %.1f s, median %.0f ms/frame, p90 %.0f ms, = %.1f fps (realtime would be 30)",
            perFrame.count, totalSeconds, percentile(perFrame, 50), percentile(perFrame, 90),
            Double(perFrame.count) / totalSeconds))
}
