import CoreGraphics
import MLX
import Testing

@testable import SAM31

@Suite struct MaskTests {
    @Test func upsampleThresholdsAndCounts() {
        var logits = [Float](repeating: -5, count: 16)
        logits[5] = 5
        logits[6] = 5
        logits[9] = 5
        logits[10] = 5
        let m = Mask(logits: MLXArray(logits).reshaped(4, 4))
        #expect(m.width == 4 && m.height == 4)
        let b = m.upsampled(to: CGSize(width: 8, height: 8))
        #expect(b.width == 8 && b.height == 8)
        #expect(b.coverage > 0.15 && b.coverage < 0.35)
        #expect(Set(b.bytes).isSubset(of: [0, 255]))
        #expect(b.cgImage() != nil)
    }

    /// The CPU upsample must match `resizeBilinearNHWC` on the logits (the tracker's own upsample).
    @Test func upsampleMatchesResizeBilinearNHWC() {
        let logits = MLXRandom.normal([12, 9], key: MLXRandom.key(7)) * 4
        let m = Mask(logits: logits)
        for (w, h) in [(40, 31), (5, 4), (9, 12)] {
            let ref = resizeBilinearNHWC(logits.reshaped(1, 12, 9, 1), h: h, w: w).reshaped(h, w)
            let expected = (ref .> 0.5).asType(.uint8).asArray(UInt8.self).map { $0 * 255 }
            let got = m.upsampled(to: CGSize(width: w, height: h), threshold: 0.5)
            #expect(got.bytes == expected, "\(w)x\(h)")
        }
    }

    @Test func degenerateSizesGiveAnEmptyMask() {
        let m = Mask(logits: MLXArray.ones([2, 2]))
        for size in [CGSize(width: 0, height: 4), CGSize(width: CGFloat.nan, height: 4)] {
            let b = m.upsampled(to: size)
            #expect(b.width == 0 && b.bytes.isEmpty && b.coverage == 0 && b.cgImage() == nil)
        }
    }
}
