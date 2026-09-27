import MLX
import Testing

@Suite struct CompareTests {
    @Test func iouOfIdenticalMasksIsOne() {
        let a = MLXArray([1, -1, 2, -3] as [Float]).reshaped(2, 2)
        #expect(maskIoU(a, a) == 1)
    }

    @Test func iouOfDisjointMasksIsZero() {
        let a = MLXArray([1, -1, -1, -1] as [Float]).reshaped(2, 2)
        let b = MLXArray([-1, 1, -1, -1] as [Float]).reshaped(2, 2)
        #expect(maskIoU(a, b) == 0)
    }

    @Test func closeArraysPass() {
        let a = MLXArray([1.0, 2.0] as [Float])
        #expect(maxRelativeViolation(a, a + 1e-6, rtol: 1e-3, atol: 1e-4) <= 0)
    }

    @Test func farArraysFail() {
        let a = MLXArray([1.0, 2.0] as [Float])
        let b = MLXArray([1.0, 2.1] as [Float])
        // |Δ| = 0.1 vs allowed 1e-4 + 1e-3 * 2.1 = 0.0022 → violation ≈ 0.0978
        let v = maxRelativeViolation(a, b, rtol: 1e-3, atol: 1e-4)
        #expect(v > 0)
        #expect(abs(v - 0.0978) < 1e-3)
    }
}
