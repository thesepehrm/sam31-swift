import MLX
import Testing

@testable import SAM31

@Suite struct SmokeTests {
    @Test func versionIsSet() { #expect(SAM31.version.hasPrefix("0.1.0")) }

    @Test func mlxRunsOnGPU() {
        let a = MLXArray([1, 2, 3] as [Float])
        let b = (a * 2).sum()
        #expect(b.item(Float.self) == 12)
    }
}
