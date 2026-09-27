import CoreGraphics
import CoreVideo
import Foundation
import MLX

/// A segmentation mask as logits at model resolution (typically 288×288 for the full
/// 1008×1008 model input). Pixels with a logit above 0 belong to the object.
///
/// The logits are copied to host memory, so a `Mask` is a plain value that is safe to use from any
/// thread.
public struct Mask: Sendable {
    /// Width of the logit grid.
    public let width: Int
    /// Height of the logit grid.
    public let height: Int
    /// Row-major logits at model resolution: `width × height` values, row `y` starting at index
    /// `y * width`. A logit above 0 means the pixel belongs to the object.
    public let values: [Float]

    /// Creates a mask from row-major logits, for example to pass as ``TrackPrompt/mask(_:)``.
    ///
    /// Logits above 0 mark foreground. A binary mask works too: use a positive value (such as 10)
    /// for foreground and a negative one (such as -10) for background.
    ///
    /// - Throws: ``SAM31Error/invalidPrompt(_:)`` when `width` or `height` is not positive or
    ///   `logits.count` is not `width * height`.
    public init(logits: [Float], width: Int, height: Int) throws {
        guard width > 0, height > 0 else {
            throw SAM31Error.invalidPrompt("mask size \(width)×\(height) is not positive")
        }
        guard width.multipliedReportingOverflow(by: height) == (logits.count, false) else {
            throw SAM31Error.invalidPrompt(
                "mask has \(logits.count) logits, expected \(width)×\(height)")
        }
        self.width = width
        self.height = height
        self.values = logits
    }

    /// Copies `(h, w)` logits to the host (evaluates them).
    init(logits: MLXArray) {
        precondition(logits.ndim == 2, "mask logits must be (h, w), got \(logits.shape)")
        height = logits.dim(0)
        width = logits.dim(1)
        values = logits.asType(.float32).asArray(Float.self)
    }

    /// The `(height, width)` float32 logits.
    var logits: MLXArray { MLXArray(values, [height, width]) }

    /// The mask resized to `size` pixels (usually the source image size) and thresholded.
    ///
    /// Resizes the logits bilinearly (PyTorch `interpolate(mode: "bilinear")` taps, the same as
    /// the tracker's own 288→1008 upsample), then keeps pixels whose logit is above `threshold`.
    /// mlx-vlm's `_masks_to_detection` resizes the thresholded mask with nearest instead; bilinear on
    /// logits gives smoother edges, and the parity tests compare at model resolution, so the
    /// difference does not affect them. Runs on the CPU, so it never touches the model's GPU work.
    ///
    /// - Returns: the binary mask, or an empty (0×0) mask when `size` is not a positive finite size.
    public func upsampled(to size: CGSize, threshold: Float = 0) -> BinaryMask {
        guard size.width.isFinite, size.height.isFinite, size.width >= 1, size.height >= 1,
            size.width < 1e6, size.height < 1e6, width > 0, height > 0
        else { return BinaryMask(width: 0, height: 0, bytes: []) }
        let outW = Int(size.width.rounded())
        let outH = Int(size.height.rounded())
        let tx = atenBilinearTaps(inSize: width, outSize: outW, alignCorners: false, antialias: false)
        let ty = atenBilinearTaps(inSize: height, outSize: outH, alignCorners: false, antialias: false)

        // Width pass, then height pass, accumulating taps in order (as `separableInterpolate`).
        var rows = [Float](repeating: 0, count: height * outW)
        for y in 0..<height {
            let src = y * width
            let dst = y * outW
            for t in tx.indices.indices {
                let (idx, wt) = (tx.indices[t], tx.weights[t])
                for x in 0..<outW {
                    rows[dst + x] += values[src + Int(idx[x])] * wt[x]
                }
            }
        }
        var bytes = [UInt8](repeating: 0, count: outH * outW)
        var column = [Float](repeating: 0, count: outW)
        for y in 0..<outH {
            for x in 0..<outW { column[x] = 0 }
            for t in ty.indices.indices {
                let src = Int(ty.indices[t][y]) * outW
                let wt = ty.weights[t][y]
                for x in 0..<outW {
                    column[x] += rows[src + x] * wt
                }
            }
            let dst = y * outW
            for x in 0..<outW where column[x] > threshold {
                bytes[dst + x] = 255
            }
        }
        return BinaryMask(width: outW, height: outH, bytes: bytes)
    }
}

/// A thresholded mask: one byte per pixel, 255 inside the object and 0 outside.
public struct BinaryMask: Sendable {
    /// Width in pixels.
    public let width: Int
    /// Height in pixels.
    public let height: Int
    /// Row-major pixels, 0 or 255.
    public let bytes: [UInt8]

    /// The mask as an 8-bit grayscale image, or nil when it is empty.
    public func cgImage() -> CGImage? {
        guard width > 0, height > 0, bytes.count == width * height,
            let provider = CGDataProvider(data: Data(bytes) as CFData)
        else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Fraction of pixels inside the object (0...1); 0 for an empty mask.
    public var coverage: Double {
        guard !bytes.isEmpty else { return 0 }
        return Double(bytes.lazy.filter { $0 != 0 }.count) / Double(bytes.count)
    }
}

/// The result of ``SAM31Model/segment(_:prompt:previous:)``.
public struct SegmentResult: Sendable {
    /// The object's mask at model resolution.
    public let mask: Mask
    /// The model's predicted IoU of `mask` (roughly 0...1; higher is better).
    public let score: Float
    /// Probability (0...1) that the prompt hit an object at all.
    public let objectScore: Float
}

/// One object found by ``SAM31Model/detect(_:text:scoreThreshold:)``.
public struct Detection: Sendable {
    /// Bounding box in source-image pixels.
    public let box: CGRect
    /// Detection confidence (0...1).
    public let score: Float
    /// The object's mask at model resolution.
    public let mask: Mask
}

/// Identifies an object within one ``TrackingSession``.
public struct ObjectID: Hashable, Sendable {
    /// The session-unique number of the object, assigned in the order objects are added.
    public let rawValue: Int
}

/// One object's state on a tracked frame.
public struct TrackedObject: Sendable {
    /// The object's mask at model resolution.
    public let mask: Mask
    /// Probability (0...1) that the object is present on the frame.
    public let score: Float
    /// Whether the object is present on the frame (its object logit is above 0). When false the
    /// mask is empty.
    public let isVisible: Bool
}

/// The tracking result for one frame.
public struct TrackedFrame: Sendable {
    /// Index of the frame in the video.
    public let frameIndex: Int
    /// Every object in the session.
    public let objects: [ObjectID: TrackedObject]
}

/// A frame to track: an image, a pixel buffer, or features from ``SAM31Model`` `encode`.
///
/// Images and pixel buffers are encoded on the fly and only the propagation head is computed.
public enum FrameInput: @unchecked Sendable {
    /// An RGB image.
    case image(CGImage)
    /// A `32BGRA` or 4:2:0 bi-planar Y'CbCr pixel buffer.
    case pixelBuffer(CVPixelBuffer)
    /// A frame that was already encoded.
    case features(FrameFeatures)
}

/// Logistic sigmoid of a logit.
func sigmoidProbability(_ logit: Float) -> Float {
    1 / (1 + exp(-logit))
}
