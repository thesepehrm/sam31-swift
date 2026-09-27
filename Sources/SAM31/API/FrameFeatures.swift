import CoreGraphics
import Foundation
import MLX

/// The encoded form of one image or video frame: the ViT backbone output, computed once by
/// `SAM31Model.encode(_:)`.
///
/// Clicks, text detection and tracking on the same frame all reuse it. The neck head each one
/// needs (detection, interactive, propagation) is computed on first use and cached, so keep the
/// value for as long as you may prompt on the frame and drop it afterwards to free its memory.
/// Use it only with the model that created it.
public final class FrameFeatures: @unchecked Sendable {
    /// Size of the source image, in pixels.
    public let sourceSize: CGSize
    /// `(1, 72, 72, 1024)` evaluated backbone output. Immutable.
    let backbone: MLXArray

    // The lazily computed neck heads. `lock` guards them; they are only computed on the model actor.
    private let lock = NSLock()
    private var det: [MLXArray]?
    private var interactive: LevelFeatures?
    private var propagation: LevelFeatures?

    init(backbone: MLXArray, sourceSize: CGSize) {
        self.backbone = backbone
        self.sourceSize = sourceSize
    }

    /// The detection FPN levels, computed by `compute` on first use.
    func detFeatures(orCompute compute: () -> [MLXArray]) -> [MLXArray] {
        lock.withLock {
            if let det { return det }
            let value = compute()
            det = value
            return value
        }
    }

    /// The interactive tracker features, computed by `compute` on first use.
    func interactiveFeatures(orCompute compute: () -> LevelFeatures) -> LevelFeatures {
        lock.withLock {
            if let interactive { return interactive }
            let value = compute()
            interactive = value
            return value
        }
    }

    /// The propagation tracker features, computed by `compute` on first use.
    func propagationFeatures(orCompute compute: () -> LevelFeatures) -> LevelFeatures {
        lock.withLock {
            if let propagation { return propagation }
            let value = compute()
            propagation = value
            return value
        }
    }
}
