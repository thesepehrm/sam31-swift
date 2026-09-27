import Foundation
import MLX
import MLXNN
import Testing

@testable import SAM31

@Suite(.enabled(if: Fixtures.available), .serialized) struct WeightKeyParityTests {
    /// Every checkpoint key under `prefix` is consumed by `module`, with matching shapes, and every
    /// parameter of `module` is set. Porting tasks call this for their subtree.
    static func assertSubtreeLoads(_ module: Module, prefix: String) throws {
        try loadWeights(into: module, from: Fixtures.weightsURL, dtype: .float32, prefix: prefix)
    }

    /// The real checkpoint is already in MLX layout: sanitize must only remap keys, never transpose.
    @Test func sanitizeKeepsMLXLayoutOfRealCheckpoint() throws {
        let raw = try loadArrays(url: Fixtures.weightsURL.appending(path: "model.safetensors"))
        let s = sanitize(raw)
        #expect(raw.count == 1961 && s.count == 1961)
        let patch = "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
        #expect(s[patch]?.shape == raw[patch]?.shape)
        #expect(s[patch]?.shape.last == 3)
        for (key, value) in raw where value.ndim == 4 {
            // Keys may be remapped, but no 4-D tensor may change shape.
            #expect(s[key].map { $0.shape == value.shape } ?? true, "\(key)")
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["SAM31_FULL_MODEL"] == "1"))
    func fullModelLoadsWithNoMissingOrUnusedKeys() throws {
        let config = try ModelConfig.load(from: Fixtures.weightsURL)
        let root = SAM31Root(config)
        try loadWeights(into: root, from: Fixtures.weightsURL, dtype: .float32, prefix: nil)
        #expect(root.parameters().flattened().count == 1961)
    }
}
