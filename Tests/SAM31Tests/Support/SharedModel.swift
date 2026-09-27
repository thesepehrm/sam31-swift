import Foundation
import MLX

@testable import SAM31

/// One `SAM31Model` per test process, built on ``SharedWeights`` so the API suites don't each load
/// another 3.5 GB. Only touch it from suites gated on `Fixtures.available`.
enum SharedModel {
    private static let task = Task {
        try SAM31Model(
            config: ModelConfig.load(from: Fixtures.weightsURL), weights: SharedWeights.all, dtype: .float32)
    }

    static func get() async throws -> SAM31Model { try await task.value }
}
