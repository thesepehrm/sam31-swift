import MLX
import Testing

@testable import SAM31

struct PromptEncoderTests {
    @Test func preparePointInputsPadsAndRescales() {
        let coords = MLXArray([504, 1008, 0, 14] as [Float]).reshaped(1, 2, 2)
        let labels = MLXArray([1, 0] as [Int32]).reshaped(1, 2)
        let (c, l) = preparePointInputs(coords: coords, labels: labels, embeddingSize: 72, imageSize: 1008)
        #expect(c.shape == [1, 3, 2])
        #expect(l.dtype == .int32)
        #expect(l.asArray(Int32.self) == [1, 0, -1])
        let expected: [Float] = [36, 72, 0, 1, 0, 0]
        #expect(zip(c.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 1e-4 })
    }

    @Test func paddingPointsUseNotAPointEmbedding() {
        let pe = SAMPromptEncoder(PromptEncoderConfig())
        let coords = MLXArray([10, 20, 0, 0] as [Float]).reshaped(1, 2, 2)
        let labels = MLXArray([1, -1] as [Int32]).reshaped(1, 2)
        let (sparse, dense) = pe(points: (coords, labels), boxes: nil, masks: nil)
        #expect(sparse.shape == [1, 2, 256])
        #expect(dense.shape == [1, 72 * 72, 256])
        #expect(allClose(sparse[0, 1], pe.notAPointEmbed.weight[0]).item(Bool.self))
    }
}
