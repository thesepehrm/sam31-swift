import CoreGraphics
import Testing

@testable import SAM31

/// `Mask.upsampled` runs on vDSP. These tests hold it to the scalar loop it replaced, byte for byte.
@Suite struct MaskUpsampleParityTests {
    /// The 0.1.0 implementation, kept verbatim as the reference.
    static func reference(_ m: Mask, width outW: Int, height outH: Int, threshold: Float) -> [UInt8] {
        let (width, height, values) = (m.width, m.height, m.values)
        let tx = atenBilinearTaps(inSize: width, outSize: outW, alignCorners: false, antialias: false)
        let ty = atenBilinearTaps(inSize: height, outSize: outH, alignCorners: false, antialias: false)
        var rows = [Float](repeating: 0, count: height * outW)
        for y in 0..<height {
            let src = y * width
            let dst = y * outW
            for t in tx.indices.indices {
                let (idx, wt) = (tx.indices[t], tx.weights[t])
                for x in 0..<outW {
                    rows[dst + x] += values[src + Int(idx[x])] * wt[x]
                }
            }
        }
        var bytes = [UInt8](repeating: 0, count: outH * outW)
        var column = [Float](repeating: 0, count: outW)
        for y in 0..<outH {
            for x in 0..<outW { column[x] = 0 }
            for t in ty.indices.indices {
                let src = Int(ty.indices[t][y]) * outW
                let wt = ty.weights[t][y]
                for x in 0..<outW {
                    column[x] += rows[src + x] * wt
                }
            }
            let dst = y * outW
            for x in 0..<outW where column[x] > threshold {
                bytes[dst + x] = 255
            }
        }
        return bytes
    }

    struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    enum Kind: CaseIterable {
        /// A blob plus noise, like a model mask: long edges crossing 0 at arbitrary values.
        case blob
        /// Uniform noise in -8...8.
        case noise
        /// Small integers, so many resized values land exactly on 0 and on the tap weights' ties.
        case integers
    }

    static func logits(_ kind: Kind, width: Int, height: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(state: seed)
        return (0..<(width * height)).map { i in
            switch kind {
            case .blob:
                let x = Float(i % width) - Float(width) / 2
                let y = Float(i / width) - Float(height) / 3
                let r = Float(max(width, height))
                return 40 - (x * x + y * y) * 160 / (r * r) + Float.random(in: -3...3, using: &rng)
            case .noise:
                return Float.random(in: -8...8, using: &rng)
            case .integers:
                return Float(Int.random(in: -2...2, using: &rng))
            }
        }
    }

    static func expectParity(
        _ mask: Mask, _ w: Int, _ h: Int, threshold: Float = 0, _ label: String
    ) {
        let got = mask.upsampled(to: CGSize(width: w, height: h), threshold: threshold)
        let want = reference(mask, width: w, height: h, threshold: threshold)
        #expect(got.width == w && got.height == h)
        let differing = zip(got.bytes, want).filter { $0 != $1 }.count
        #expect(got.bytes.count == want.count && differing == 0, "\(label): \(differing) pixels differ")
    }

    static let cases: [(inW: Int, inH: Int, outW: Int, outH: Int)] = [
        (288, 288, 1920, 1080),  // the click mask to 1080p
        (288, 288, 1080, 1920),
        (288, 288, 1008, 1008),
        (288, 288, 333, 177),  // odd
        (288, 288, 97, 61),  // downsample
        (288, 288, 288, 288),  // same size
        (288, 288, 1, 1),
        (12, 9, 40, 31),
        (9, 12, 5, 4),
        (7, 5, 41, 23),
        (17, 13, 3, 2),
        (1, 1, 5, 3),
        (2, 3, 1, 1),
        (1, 7, 9, 1),
        (3, 1, 1, 11),
    ]

    @Test(arguments: Kind.allCases)
    func matchesTheScalarLoop(kind: Kind) throws {
        for (i, c) in Self.cases.enumerated() {
            let values = Self.logits(kind, width: c.inW, height: c.inH, seed: 0x5A31 &+ UInt64(i))
            let mask = try Mask(logits: values, width: c.inW, height: c.inH)
            Self.expectParity(mask, c.outW, c.outH, "\(kind) \(c.inW)x\(c.inH)->\(c.outW)x\(c.outH)")
        }
    }

    @Test func matchesAtOtherThresholds() throws {
        let values = Self.logits(.integers, width: 31, height: 29, seed: 11)
        let mask = try Mask(logits: values, width: 31, height: 29)
        for threshold: Float in [
            0.5, -0.5, -0.0, 1, 2, -2, .leastNonzeroMagnitude, -.leastNonzeroMagnitude,
            .greatestFiniteMagnitude, -.greatestFiniteMagnitude, .infinity, -.infinity, .nan,
        ] {
            Self.expectParity(mask, 101, 67, threshold: threshold, "threshold \(threshold)")
        }
    }

    @Test func nonFiniteLogitsMatch() throws {
        var values = Self.logits(.noise, width: 16, height: 16, seed: 3)
        values[17] = .nan
        values[40] = .infinity
        values[41] = .infinity
        values[90] = -.infinity
        values[200] = .greatestFiniteMagnitude
        let mask = try Mask(logits: values, width: 16, height: 16)
        Self.expectParity(mask, 53, 37, "non-finite")
        Self.expectParity(mask, 7, 5, "non-finite downsample")
    }

    @Test func allZeroLogitsStayEmpty() throws {
        let mask = try Mask(logits: [Float](repeating: 0, count: 64), width: 8, height: 8)
        let b = mask.upsampled(to: CGSize(width: 50, height: 30))
        #expect(b.bytes.allSatisfy { $0 == 0 })
        #expect(
            mask.upsampled(to: CGSize(width: 50, height: 30), threshold: -0.0).bytes.allSatisfy { $0 == 0 })
        #expect(
            mask.upsampled(to: CGSize(width: 50, height: 30), threshold: -1).bytes.allSatisfy { $0 == 255 })
    }
}
