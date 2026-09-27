import CoreGraphics
import MLX

/// A click prompt in the 1008×1008 model-input space.
///
/// Use ``PointMapper`` to convert from source-image pixels.
public struct PromptPoint: Sendable, Hashable {
    /// Whether the click marks the object (`positive`) or background (`negative`).
    public enum Label: Int32, Sendable {
        /// Background: the mask should not cover this point.
        case negative = 0
        /// Foreground: the mask should cover this point.
        case positive = 1
    }

    /// Horizontal position in model-input pixels (0...1008).
    public var x: Float
    /// Vertical position in model-input pixels (0...1008).
    public var y: Float
    /// Positive or negative click.
    public var label: Label

    /// Creates a click at (`x`, `y`) in model-input pixels.
    public init(x: Float, y: Float, label: Label) {
        self.x = x
        self.y = y
        self.label = label
    }
}

/// A prompt for interactive segmentation, in 1008×1008 model-input space.
public enum SegmentPrompt: Sendable {
    /// One or more clicks.
    case points([PromptPoint])
    /// A bounding box around the object.
    case box(CGRect)
    /// A bounding box refined with clicks.
    case boxAndPoints(CGRect, [PromptPoint])
}

/// How to add an object to a ``TrackingSession``.
public enum TrackPrompt: Sendable {
    /// Clicks and/or a box, segmented on the prompt frame.
    case segment(SegmentPrompt)
    /// A mask, for example from ``SAM31Model/detect(_:text:scoreThreshold:)``. Pixels with a logit
    /// above 0 belong to the object.
    case mask(Mask)
}

/// Converts between source-image pixels and the 1008×1008 model-input space.
///
/// The model sees every image stretched to a square, so x and y scale independently.
public struct PointMapper: Sendable {
    let sourceSize: CGSize
    private static let modelSide: CGFloat = 1008

    /// Creates a mapper for images of `sourceSize` pixels. Both sides must be positive.
    public init(sourceSize: CGSize) {
        self.sourceSize = sourceSize
    }

    /// A source-pixel point in model-input space.
    public func toModel(_ p: CGPoint) -> CGPoint {
        CGPoint(x: p.x * Self.modelSide / sourceSize.width, y: p.y * Self.modelSide / sourceSize.height)
    }

    /// A source-pixel rectangle in model-input space.
    public func toModel(_ r: CGRect) -> CGRect {
        let sx = Self.modelSide / sourceSize.width
        let sy = Self.modelSide / sourceSize.height
        return CGRect(x: r.minX * sx, y: r.minY * sy, width: r.width * sx, height: r.height * sy)
    }

    /// A model-input rectangle in source pixels.
    public func toSource(_ r: CGRect) -> CGRect {
        let sx = sourceSize.width / Self.modelSide
        let sy = sourceSize.height / Self.modelSide
        return CGRect(x: r.minX * sx, y: r.minY * sy, width: r.width * sx, height: r.height * sy)
    }
}

extension SegmentPrompt {
    /// Flat `[x0, y0, x1, y1, ...]` coordinates and labels. A box becomes its top-left and
    /// bottom-right corners with labels 2 and 3, placed before any clicks (SAM's box encoding).
    ///
    /// - Throws: ``SAM31Error/invalidPrompt(_:)`` for no points, non-finite coordinates, or an
    ///   empty box.
    func pointList() throws -> (coords: [Float], labels: [Int32]) {
        var coords: [Float] = []
        var labels: [Int32] = []
        func addBox(_ rect: CGRect) throws {
            let r = rect.standardized
            guard [r.minX, r.minY, r.width, r.height].allSatisfy(\.isFinite) else {
                throw SAM31Error.invalidPrompt("box has non-finite coordinates: \(rect)")
            }
            guard r.width > 0, r.height > 0 else {
                throw SAM31Error.invalidPrompt("box is empty: \(rect)")
            }
            coords += [Float(r.minX), Float(r.minY), Float(r.maxX), Float(r.maxY)]
            labels += [2, 3]
        }
        func addPoints(_ points: [PromptPoint]) throws {
            for p in points {
                guard p.x.isFinite, p.y.isFinite else {
                    throw SAM31Error.invalidPrompt("point has non-finite coordinates: (\(p.x), \(p.y))")
                }
                coords += [p.x, p.y]
                labels.append(p.label.rawValue)
            }
        }
        switch self {
        case .points(let points):
            guard !points.isEmpty else {
                throw SAM31Error.invalidPrompt("a point prompt needs at least one point")
            }
            try addPoints(points)
        case .box(let box):
            try addBox(box)
        case .boxAndPoints(let box, let points):
            try addBox(box)
            try addPoints(points)
        }
        return (coords, labels)
    }

    /// The prompt as tracker point inputs for one object: `(1, P, 2)` coords and `(1, P)` labels.
    func pointInputs() throws -> PointInputs {
        let (coords, labels) = try pointList()
        return PointInputs(
            coords: MLXArray(coords).reshaped(1, labels.count, 2),
            labels: MLXArray(labels).reshaped(1, labels.count))
    }
}
