import Foundation
import MLX

@testable import SAM31

/// The sanitized fp32 checkpoint, loaded once per test process so parity suites don't each reload
/// 3.5 GB. Only touch it from suites gated on `Fixtures.available`.
enum SharedWeights {
    nonisolated(unsafe) static let all: [String: MLXArray] = {
        let file = Fixtures.weightsURL.appending(path: "model.safetensors")
        let weights = sanitize(try! loadArrays(url: file))
        eval(Array(weights.values))
        return weights
    }()
}
