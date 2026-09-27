// Swift counterpart of mlx_vlm/utils.py::load_model's weight loading (mlx-vlm 0.7.3):
// load safetensors -> sanitize -> cast -> update(verify:).
import Foundation
import MLX
import MLXNN

/// Loads `<directory>/model.safetensors` into `module`.
///
/// Weights are sanitized (see ``sanitize(_:)``) and then applied as in
/// ``loadWeights(into:weights:dtype:prefix:)``.
func loadWeights(into module: Module, from directory: URL, dtype: DType, prefix: String?) throws {
    let file = directory.appending(path: "model.safetensors")
    guard FileManager.default.fileExists(atPath: file.path) else {
        throw SAM31Error.weightsNotFound(directory)
    }

    let raw: [String: MLXArray]
    do {
        raw = try loadArrays(url: file)
    } catch {
        throw SAM31Error.invalidConfig("\(file.lastPathComponent): \(error)")
    }

    try loadWeights(into: module, weights: sanitize(raw), dtype: dtype, prefix: prefix)
}

/// Loads already-sanitized `weights` into `module`.
///
/// Weights are optionally narrowed to one subtree, cast to `dtype`, and applied with full
/// verification: every parameter of `module` must be set, every tensor must be consumed, and shapes
/// must match.
///
/// - Parameters:
///   - weights: a checkpoint dictionary that has already been through ``sanitize(_:)``.
///   - prefix: when set, only keys starting with `prefix` are kept and the prefix is stripped, so a
///     subtree such as `"detector_model.vision_encoder.backbone."` loads into a standalone module.
///   - dtype: floating-point tensors are cast to this type; integer tensors are left unchanged.
func loadWeights(into module: Module, weights all: [String: MLXArray], dtype: DType, prefix: String?) throws {
    var weights: [String: MLXArray] = [:]
    for (key, value) in all {
        var key = key
        if let prefix {
            guard key.hasPrefix(prefix) else { continue }
            key = String(key.dropFirst(prefix.count))
        }
        weights[key] = value.dtype.isFloatingPoint && value.dtype != dtype ? value.asType(dtype) : value
    }

    // MLXNN reports unused keys one module at a time and relative to that module; check up front so
    // the error lists every unused key with its full path.
    let expected = Set(module.parameters().flattened().map(\.0))
    let unused = weights.keys.filter { !expected.contains($0) }.sorted()
    if !unused.isEmpty {
        throw SAM31Error.unusedParameters(unused)
    }

    do {
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch let error as UpdateError {
        throw SAM31Error(error)
    }
    eval(Array(weights.values))
}

extension SAM31Error {
    /// Maps an MLXNN update failure to the matching SAM31Error, with dotted parameter keys.
    fileprivate init(_ error: UpdateError) {
        switch error {
        case .keyNotFound(let path, _):
            self = .missingParameter(path.joined(separator: "."))
        case .mismatchedSize(let path, _, let expectedShape, let actualShape):
            self = .shapeMismatch(path.joined(separator: "."), expected: expectedShape, got: actualShape)
        case .unhandledKeys(let path, _, let keys):
            self = .unusedParameters(keys.map { (path + [$0]).joined(separator: ".") })
        default:
            self = .invalidConfig("weights do not match the module structure: \(error.localizedDescription)")
        }
    }
}
