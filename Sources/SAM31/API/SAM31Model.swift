import CoreGraphics
import CoreVideo
import Foundation
import MLX

/// Floating-point precision of the model weights and computation.
public enum SAM31DType: Sendable {
    /// 32-bit float: the reference precision.
    case float32
    /// bfloat16: half the memory, slightly less exact.
    case bfloat16

    var mlx: DType {
        switch self {
        case .float32: .float32
        case .bfloat16: .bfloat16
        }
    }
}

/// Fixed properties of a loaded model.
public struct ModelSummary: Sendable, Hashable {
    /// Side of the square model input, in pixels (1008). Prompt coordinates use this space.
    public let imageSize: Int
    /// Most objects one ``TrackingSession`` can hold (16).
    public let maxObjectsPerBucket: Int
    /// Precision the model was loaded with.
    public let dtype: SAM31DType
}

/// Meta SAM 3.1 on MLX: click and box segmentation, text detection, and video object tracking.
///
/// Load the model once, encode each frame once with `encode`, then prompt on the returned
/// ``FrameFeatures`` as often as needed. The actor serializes all GPU work.
///
/// ```swift
/// let model = try await SAM31Model.load(from: weightsDirectory)
/// let frame = try await model.encode(image)
/// let mapper = PointMapper(sourceSize: frame.sourceSize)
/// let click = mapper.toModel(CGPoint(x: 640, y: 360))
/// let result = try await model.segment(frame, prompt: .points([
///     PromptPoint(x: Float(click.x), y: Float(click.y), label: .positive)]))
/// let mask = result.mask.upsampled(to: frame.sourceSize)
/// ```
public actor SAM31Model {
    /// Fixed properties of this model.
    public nonisolated let configuration: ModelSummary

    let root: SAM31Root
    private let preprocessor = ImagePreprocessor()
    private var textCache = TextEmbeddingCache(capacity: 16)

    var detector: DetectorModel { root.detectorModel }
    var tracker: MultiplexTrackerModel { root.trackerModel }

    /// Loads `config.json` and `model.safetensors` from `directory`.
    ///
    /// - Throws: ``SAM31Error/weightsNotFound(_:)`` when a file is missing, or another
    ///   ``SAM31Error`` when the files do not match the SAM 3.1 architecture.
    public static func load(from directory: URL, dtype: SAM31DType = .float32) async throws -> SAM31Model {
        try SAM31Model(config: ModelConfig.load(from: directory), directory: directory, dtype: dtype)
    }

    init(config: ModelConfig, directory: URL, dtype: SAM31DType) throws {
        root = SAM31Root(config)
        try loadWeights(into: root, from: directory, dtype: dtype.mlx, prefix: nil)
        configuration = Self.summary(config, dtype)
    }

    /// A model built from an already-sanitized checkpoint (tests share one copy of the weights).
    init(config: ModelConfig, weights: [String: MLXArray], dtype: SAM31DType) throws {
        root = SAM31Root(config)
        try loadWeights(into: root, weights: weights, dtype: dtype.mlx, prefix: nil)
        configuration = Self.summary(config, dtype)
    }

    private static func summary(_ config: ModelConfig, _ dtype: SAM31DType) -> ModelSummary {
        ModelSummary(
            imageSize: config.trackerConfig.imageSize,
            maxObjectsPerBucket: config.trackerConfig.multiplexCount,
            dtype: dtype)
    }

    // MARK: - Encoding

    /// Runs the vision backbone on an RGB image.
    ///
    /// - Throws: ``SAM31Error/invalidImage(_:)`` when the image cannot be drawn as RGB.
    public func encode(_ image: CGImage) throws -> FrameFeatures {
        let pixels = try preprocessor.pixelValues(from: image)
        return features(pixels, sourceSize: CGSize(width: image.width, height: image.height))
    }

    /// Runs the vision backbone on a `32BGRA` or 4:2:0 bi-planar Y'CbCr (`420f`/`420v`) pixel buffer.
    ///
    /// - Throws: ``SAM31Error/invalidImage(_:)`` for other pixel formats.
    public func encode(_ pixelBuffer: CVPixelBuffer) throws -> FrameFeatures {
        let pixels = try preprocessor.pixelValues(from: pixelBuffer)
        let size = CGSize(
            width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        return features(pixels, sourceSize: size)
    }

    private func features(_ pixelValues: MLXArray, sourceSize: CGSize) -> FrameFeatures {
        let backbone = detector.visionEncoder.backbone(pixelValues)
        eval(backbone)
        return FrameFeatures(backbone: backbone, sourceSize: sourceSize)
    }

    /// Encodes a ``FrameInput``: runs the vision backbone on an image or pixel buffer, or returns
    /// already-encoded features unchanged.
    ///
    /// `CVPixelBuffer` is not `Sendable`, so Swift 6 callers outside the actor cannot pass one to the
    /// `CVPixelBuffer` overload of `encode` directly. Wrap it as `.pixelBuffer(buffer)` and call this
    /// instead. The buffer is read while the call is in flight; do not write to it until the call
    /// returns.
    ///
    /// - Throws: ``SAM31Error/invalidImage(_:)`` for an unsupported image or pixel format.
    public func encode(_ input: FrameInput) throws -> FrameFeatures {
        switch input {
        case .image(let image): try encode(image)
        case .pixelBuffer(let buffer): try encode(buffer)
        case .features(let features): features
        }
    }

    /// The tracker features of `frame`, computing (and caching) only the requested neck heads.
    func trackerFeatures(_ frame: FrameFeatures, interactive: Bool, propagation: Bool) -> TrackerFrameFeatures
    {
        var out = TrackerFrameFeatures()
        if interactive {
            out.interactive = frame.interactiveFeatures {
                let fpn = detector.visionEncoder.neck(
                    frame.backbone, needDet: false, needInteractive: true, needPropagation: false
                ).interactive
                let level = tracker.prepareFrameFeatures(interactiveFPN: fpn, propagationFPN: nil)
                    .interactive!
                eval(level.visionFeat, level.highRes)
                return level
            }
        }
        if propagation {
            out.propagation = frame.propagationFeatures {
                let fpn = detector.visionEncoder.neck(
                    frame.backbone, needDet: false, needInteractive: false, needPropagation: true
                ).propagation
                let level = tracker.prepareFrameFeatures(interactiveFPN: nil, propagationFPN: fpn)
                    .propagation!
                eval(level.visionFeat, level.visionPos!, level.highRes)
                return level
            }
        }
        return out
    }

    // MARK: - Interactive segmentation

    /// Segments the object that `prompt` points at.
    ///
    /// Coordinates are in 1008×1008 model-input space (see ``PointMapper``). To refine a mask with
    /// more clicks on the same frame, pass the last result as `previous` together with all clicks
    /// so far.
    ///
    /// - Throws: ``SAM31Error/invalidPrompt(_:)`` for an empty or non-finite prompt.
    public func segment(_ frame: FrameFeatures, prompt: SegmentPrompt, previous: SegmentResult? = nil) throws
        -> SegmentResult
    {
        let out = try interactiveHeads(frame, prompt: prompt, previousLogits: previous?.mask.logits)
        // ious: (1, K); with multimask the selected mask is the argmax, so its IoU is the max.
        let score = out.ious[0].max()
        let objectLogit = out.objectScoreLogits[0, 0]
        eval(out.lowResMasks, score, objectLogit)
        return SegmentResult(
            mask: Mask(logits: out.lowResMasks[0, 0]), score: score.item(Float.self),
            objectScore: sigmoidProbability(objectLogit.item(Float.self)))
    }

    /// The interactive SAM heads for one object on a throwaway single-object state: what
    /// `track_step(is_init_cond_frame=True, point_inputs=...)` runs for a click (the reference
    /// benchmark's click path), without the memory encoder and propagation head a one-off click does
    /// not need. `previousLogits` is the `(h, w)` mask prompt (`prev_sam_mask_logits`).
    func interactiveHeads(_ frame: FrameFeatures, prompt: SegmentPrompt, previousLogits: MLXArray?) throws
        -> SAMHeadOutput
    {
        let points = try prompt.pointInputs()
        let features = trackerFeatures(frame, interactive: true, propagation: false).interactive!
        let state = tracker.initState(numObjects: 1)
        let maskPrompt = previousLogits.map { expandedDimensions($0, axes: [0, 1]) }
        return tracker.forwardSAMHeads(
            backboneFeatures: tracker.getInteractivePixMem(features.visionFeat), pointInputs: points,
            maskInputs: maskPrompt, interactiveHighResFeatures: features.highRes,
            multimaskOutput: tracker.useMultimask(isInitCondFrame: true, pointInputs: points),
            multiplexState: state.multiplexState)
    }

    // MARK: - Text detection

    /// Finds every object matching a short noun phrase, such as "person" or "red car".
    ///
    /// Text embeddings of the 16 most recent phrases are cached.
    ///
    /// - Parameter scoreThreshold: minimum detection confidence (0...1).
    /// - Returns: detections in descending score order, after non-maximum suppression.
    /// - Throws: ``SAM31Error/invalidPrompt(_:)`` for blank text or a non-finite threshold.
    public func detect(_ frame: FrameFeatures, text: String, scoreThreshold: Float = 0.5) throws
        -> [Detection]
    {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SAM31Error.invalidPrompt("text prompt is empty")
        }
        guard scoreThreshold.isFinite else {
            throw SAM31Error.invalidPrompt("score threshold \(scoreThreshold) is not finite")
        }
        let (embeds, mask) = try textEmbeddings(text)
        let det = frame.detFeatures {
            let levels = detector.visionEncoder.neck(
                frame.backbone, needDet: true, needInteractive: false, needPropagation: false
            ).det
            eval(levels)
            return levels
        }
        let raw = detector.detect(detFeatures: det, textEmbeds: embeds, textMask: mask)
        let size = (width: Int(frame.sourceSize.width), height: Int(frame.sourceSize.height))
        return postprocessDetections(raw, imageSize: size, threshold: scoreThreshold).map { d in
            Detection(
                box: CGRect(
                    x: CGFloat(d.box[0]), y: CGFloat(d.box[1]), width: CGFloat(d.box[2] - d.box[0]),
                    height: CGFloat(d.box[3] - d.box[1])),
                score: d.score, mask: Mask(logits: d.maskLogits))
        }
    }

    private func textEmbeddings(_ text: String) throws -> (MLXArray, MLXArray) {
        if let hit = textCache.get(text) { return hit }
        let tokens = try CLIPTokenizer.bundled().encode(text)
        let ids = MLXArray(tokens.inputIds).reshaped(1, -1)
        let mask = MLXArray(tokens.attentionMask).reshaped(1, -1)
        let embeds = detector.textEmbeddings(ids, mask)
        eval(embeds)
        textCache.put(text, (embeds, mask))
        return (embeds, mask)
    }

    // MARK: - Tracking and memory

    /// Starts an empty video tracking session on this model.
    public func makeTrackingSession() -> TrackingSession {
        TrackingSession(model: self)
    }

    /// Caps the memory MLX keeps cached for reuse after arrays are freed (`Memory.cacheLimit`).
    /// Lower it to reduce the process footprint at some speed cost. Applies to the whole process.
    public func setCacheLimit(bytes: Int) {
        Memory.cacheLimit = max(0, bytes)
    }
}

/// Least-recently-used cache of text embeddings, keyed by prompt text.
struct TextEmbeddingCache {
    let capacity: Int
    /// Most recently used last.
    private var entries: [(key: String, value: (MLXArray, MLXArray))] = []

    init(capacity: Int) {
        self.capacity = capacity
    }

    mutating func get(_ key: String) -> (MLXArray, MLXArray)? {
        guard let i = entries.firstIndex(where: { $0.key == key }) else { return nil }
        let entry = entries.remove(at: i)
        entries.append(entry)
        return entry.value
    }

    mutating func put(_ key: String, _ value: (MLXArray, MLXArray)) {
        entries.removeAll { $0.key == key }
        entries.append((key, value))
        if entries.count > capacity { entries.removeFirst(entries.count - capacity) }
    }
}
