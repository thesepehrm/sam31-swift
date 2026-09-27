import Foundation

/// Errors thrown by SAM31.
public enum SAM31Error: Error, Equatable {
    /// The weights directory, `config.json`, or `model.safetensors` is missing.
    case weightsNotFound(URL)
    /// A model parameter has no matching tensor in the checkpoint (dotted key).
    case missingParameter(String)
    /// A checkpoint tensor's shape differs from the model parameter's shape (dotted key).
    case shapeMismatch(String, expected: [Int], got: [Int])
    /// Checkpoint tensors that no model parameter consumes (dotted keys, sorted).
    case unusedParameters([String])
    /// `config.json` could not be decoded, or the weights do not fit the module structure.
    case invalidConfig(String)
    /// A prompt is malformed (for example an empty text prompt or out-of-range point).
    case invalidPrompt(String)
    /// An input image or pixel buffer cannot be read (for example an unsupported pixel format).
    case invalidImage(String)
    /// No tracked object has this ID.
    case unknownObject(Int)
    /// Downloading weights failed.
    case download(String)
}
