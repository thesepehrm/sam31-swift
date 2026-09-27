import MLX
import Testing

/// Largest amount by which |a-b| exceeds atol + rtol*|b|; <= 0 means within tolerance.
func maxRelativeViolation(_ a: MLXArray, _ b: MLXArray, rtol: Float, atol: Float) -> Float {
    let a32 = a.asType(.float32)
    let b32 = b.asType(.float32)
    return (abs(a32 - b32) - (atol + rtol * abs(b32))).max().item(Float.self)
}

func assertClose(
    _ a: MLXArray, _ b: MLXArray, rtol: Float = 1e-3, atol: Float = 1e-4, _ label: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(a.shape == b.shape, "\(label): shape \(a.shape) vs \(b.shape)", sourceLocation: sourceLocation)
    guard a.shape == b.shape else { return }
    let v = maxRelativeViolation(a, b, rtol: rtol, atol: atol)
    let maxAbs = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    #expect(
        v <= 0, "\(label): max|Δ|=\(maxAbs) exceeds rtol \(rtol) atol \(atol)",
        sourceLocation: sourceLocation)
}

/// IoU of (a > 0) vs (b > 0). Two empty masks count as identical (IoU 1).
func maskIoU(_ a: MLXArray, _ b: MLXArray) -> Float {
    let ma = a .> 0
    let mb = b .> 0
    let inter = logicalAnd(ma, mb).sum().item(Float.self)
    let union = logicalOr(ma, mb).sum().item(Float.self)
    return union == 0 ? 1 : inter / union
}
