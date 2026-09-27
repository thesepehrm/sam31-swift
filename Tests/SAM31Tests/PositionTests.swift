import Foundation
import MLX
import Testing

@testable import SAM31

struct PositionTests {
    @Test func rotatePairwiseSwapsAndNegatesPairs() {
        let x = MLXArray([1, 2, 3, 4] as [Float]).reshaped(1, 4)
        #expect(rotatePairwise(x).asArray(Float.self) == [-2, 1, -4, 3])
    }

    @Test func axialCISLayout() {
        // dim 8 -> freqs [1, 1/100]; grid 3x2 (endX 3), scale 0.5.
        let cis = computeAxialCIS(dim: 8, endX: 3, endY: 2, theta: 10000, scale: 0.5)
        #expect(cis.cos.shape == [6, 8])
        // Row 4 is x = 1, y = 1 -> angles (x*f0, x*f0, x*f1, x*f1, y*f0, y*f0, y*f1, y*f1) * 0.5.
        let expected: [Float] = [0.5, 0.5, 0.005, 0.005, 0.5, 0.5, 0.005, 0.005].map { Foundation.sin($0) }
        let row = cis.sin[4].asArray(Float.self)
        #expect(zip(row, expected).allSatisfy { abs($0 - $1) < 1e-6 })
    }

    @Test func init2DFreqsLayout() {
        // dim 8 -> half 4, freqs [1, 1/100]; y block then x block.
        let cis = init2DFreqs(dim: 8, featH: 2, featW: 3)
        #expect(cis.cos.shape == [6, 4])
        // Row 5 is y = 1, x = 2.
        let expected: [Float] = [1, 0.01, 2, 0.02].map { Foundation.cos($0) }
        let row = cis.cos[5].asArray(Float.self)
        #expect(zip(row, expected).allSatisfy { abs($0 - $1) < 1e-6 })
    }

    @Test func rotaryEnc1DRotatesPairsAndTilesKeys() {
        // One head, D = 4 -> two (real, imag) pairs; frequencies for 2 positions.
        let angles: [Float] = [0.5, 1.0, 2.0, 0.25]  // (N_f=2, D/2=2)
        let freqs = MLXArray(angles).reshaped(2, 2)
        let (fc, fs) = (MLX.cos(freqs), MLX.sin(freqs))
        let x = MLXArray([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1] as [Float]).reshaped(1, 3, 1, 4)
        let (q, k) = applyRotaryEnc1D(x[0..., ..<2], x, freqsCos: fc, freqsSin: fs, repeatFreqsK: true)
        // (1, 0) rotates to (cos a, sin a); (0, 1) rotates to (-sin a, cos a).
        func rotated(_ a: [Float]) -> [Float] {
            [Foundation.cos(a[0]), Foundation.sin(a[0]), -Foundation.sin(a[1]), Foundation.cos(a[1])]
        }
        let expectedQ = rotated([0.5, 1.0]) + rotated([2.0, 0.25])
        // Key position 2 wraps to frequency row 0.
        let expectedK = expectedQ + rotated([0.5, 1.0])
        #expect(zip(q.asArray(Float.self), expectedQ).allSatisfy { abs($0 - $1) < 1e-6 })
        #expect(zip(k.asArray(Float.self), expectedK).allSatisfy { abs($0 - $1) < 1e-6 })
        #expect(k.shape == [1, 3, 1, 4])
    }
}
