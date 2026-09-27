import Foundation
import MLX
import MLXNN
import Testing

@testable import SAM31

private final class Leaf: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "token") var token: MLXArray

    override init() {
        _proj.wrappedValue = Linear(4, 2)
        _token.wrappedValue = MLXArray.zeros([1, 4])
    }
}

private final class Tree: Module {
    @ModuleInfo(key: "leaf") var leaf: Leaf
    @ModuleInfo(key: "layers") var layers: [Linear]

    override init() {
        _leaf.wrappedValue = Leaf()
        _layers.wrappedValue = [Linear(2, 2), Linear(2, 2)]
    }
}

@Suite struct WeightLoaderTests {
    /// Writes `arrays` to `<tmp>/model.safetensors` and returns the directory.
    private func writeCheckpoint(_ arrays: [String: MLXArray]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try save(arrays: arrays, url: dir.appending(path: "model.safetensors"))
        return dir
    }

    private func treeWeights(prefix: String = "") -> [String: MLXArray] {
        [
            "\(prefix)leaf.proj.weight": MLXArray.ones([2, 4]),
            "\(prefix)leaf.proj.bias": MLXArray.ones([2]),
            "\(prefix)leaf.token": MLXArray.ones([1, 4]),
            "\(prefix)layers.0.weight": MLXArray.ones([2, 2]),
            "\(prefix)layers.0.bias": MLXArray.ones([2]),
            "\(prefix)layers.1.weight": MLXArray.ones([2, 2]),
            "\(prefix)layers.1.bias": MLXArray.ones([2]),
        ]
    }

    @Test func loadsAllParameters() throws {
        let dir = try writeCheckpoint(treeWeights())
        defer { try? FileManager.default.removeItem(at: dir) }
        let tree = Tree()
        try loadWeights(into: tree, from: dir, dtype: .float32, prefix: nil)
        #expect(tree.leaf.token.sum().item(Float.self) == 4)
        #expect(tree.layers[1].weight.sum().item(Float.self) == 4)
    }

    @Test func prefixSelectsAndStripsSubtree() throws {
        var w = treeWeights(prefix: "detector_model.sub.")
        w["tracker_model.other.weight"] = MLXArray.ones([3])
        let dir = try writeCheckpoint(w)
        defer { try? FileManager.default.removeItem(at: dir) }
        let tree = Tree()
        try loadWeights(into: tree, from: dir, dtype: .float32, prefix: "detector_model.sub.")
        #expect(tree.leaf.proj.weight.sum().item(Float.self) == 8)
    }

    @Test func castsOnlyFloatingTensors() throws {
        let dir = try writeCheckpoint(treeWeights())
        defer { try? FileManager.default.removeItem(at: dir) }
        let tree = Tree()
        try loadWeights(into: tree, from: dir, dtype: .bfloat16, prefix: nil)
        #expect(tree.leaf.proj.weight.dtype == .bfloat16)
        #expect(tree.leaf.token.dtype == .bfloat16)
    }

    @Test func missingFileThrowsWeightsNotFound() {
        let dir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        #expect(throws: SAM31Error.weightsNotFound(dir)) {
            try loadWeights(into: Tree(), from: dir, dtype: .float32, prefix: nil)
        }
    }

    @Test func missingParameterIsReported() throws {
        var w = treeWeights()
        w["leaf.token"] = nil
        let dir = try writeCheckpoint(w)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: SAM31Error.missingParameter("leaf.token")) {
            try loadWeights(into: Tree(), from: dir, dtype: .float32, prefix: nil)
        }
    }

    @Test func unusedParametersAreReportedWithFullKeys() throws {
        var w = treeWeights()
        w["leaf.extra"] = MLXArray.ones([1])
        w["stray.weight"] = MLXArray.ones([1])
        let dir = try writeCheckpoint(w)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: SAM31Error.unusedParameters(["leaf.extra", "stray.weight"])) {
            try loadWeights(into: Tree(), from: dir, dtype: .float32, prefix: nil)
        }
    }

    @Test func shapeMismatchIsReported() throws {
        var w = treeWeights()
        w["layers.1.weight"] = MLXArray.ones([3, 2])
        let dir = try writeCheckpoint(w)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: SAM31Error.shapeMismatch("layers.1.weight", expected: [2, 2], got: [3, 2])) {
            try loadWeights(into: Tree(), from: dir, dtype: .float32, prefix: nil)
        }
    }

    @Test func rootHasDetectorAndTrackerSubtrees() {
        let root = SAM31Root(ModelConfig())
        #expect(root.parameters().flattened().isEmpty)
        let names = Set(root.children().keys)
        #expect(names == ["detector_model", "tracker_model"])
    }
}
