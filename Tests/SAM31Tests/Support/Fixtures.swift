import Foundation
import MLX

/// Parity fixtures written by parity/dump.py. Parity suites gate on `available`, so they are
/// disabled (not failed) when SAM31_FIXTURES or SAM31_WEIGHTS is unset.
enum Fixtures {
    static let env = ProcessInfo.processInfo.environment
    static let fixturesURL = env["SAM31_FIXTURES"].map { URL(fileURLWithPath: $0) }
    static let weightsDir = env["SAM31_WEIGHTS"].map { URL(fileURLWithPath: $0) }

    static var available: Bool {
        guard let f = fixturesURL, let w = weightsDir else { return false }
        let fm = FileManager.default
        return fm.fileExists(atPath: f.appending(path: "manifest.json").path)
            && fm.fileExists(atPath: w.appending(path: "model.safetensors").path)
    }
    static var weightsURL: URL { weightsDir! }

    static func load(_ file: String) throws -> [String: MLXArray] {
        try loadArrays(url: fixturesURL!.appending(path: file))
    }

    static var manifest: [String: Any] {
        let data = try! Data(contentsOf: fixturesURL!.appending(path: "manifest.json"))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }
}
