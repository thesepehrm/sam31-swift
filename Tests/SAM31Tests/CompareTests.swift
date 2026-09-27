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
}
