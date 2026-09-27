// Port of Pillow 12.3.0 libImaging/Resample.c::{precompute_coeffs, bilinear_filter,
// normalize_coeffs_8bpc}
//
// Shared by the float mask resize (`pillowBilinearTaps` in MaskOps.swift) and the 8-bit RGB resize
// (`pillowResizeRGB` in ImagePreprocessor.swift).
import Foundation

/// Pillow's per-axis BILINEAR resampling coefficients for a full-image box (`box = (0, 0, in, out)`).
///
/// Output position `o` reads `count[o]` source samples starting at `xmin[o]` with the normalized
/// weights `weights[o * kSize ..< o * kSize + count[o]]`; the rest of each row is 0.
struct PillowCoefficients {
    /// Fractional bits of the fixed-point weights on the 8-bit path (`PRECISION_BITS`).
    static let precisionBits = 32 - 8 - 2

    let kSize: Int
    let xmin: [Int]
    let count: [Int]
    let weights: [Double]

    /// `precompute_coeffs` with `bilinear_filter` (support 1), widening the support by the scale
    /// factor when downsampling.
    init(bilinearFrom inSize: Int, to outSize: Int) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 1.0 * filterScale
        let kSize = Int(ceil(support)) * 2 + 1

        var xmins = [Int](repeating: 0, count: outSize)
        var counts = [Int](repeating: 0, count: outSize)
        var weights = [Double](repeating: 0, count: outSize * kSize)
        let ss = 1.0 / filterScale
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            // C `(int)` truncates toward zero; the clamp makes that equal to Pillow's result.
            let xmin = max(Int(center - support + 0.5), 0)
            let count = min(Int(center + support + 0.5), inSize) - xmin
            var ww = 0.0
            for x in 0..<count {
                let t = abs((Double(x + xmin) - center + 0.5) * ss)
                let w = t < 1.0 ? 1.0 - t : 0.0
                weights[xx * kSize + x] = w
                ww += w
            }
            if ww != 0 {
                for x in 0..<count { weights[xx * kSize + x] /= ww }
            }
            xmins[xx] = xmin
            counts[xx] = count
        }
        self.kSize = kSize
        self.xmin = xmins
        self.count = counts
        self.weights = weights
    }

    /// `normalize_coeffs_8bpc`: the weights as fixed point with `precisionBits` fractional bits,
    /// rounded half away from zero.
    func fixedPoint() -> [Int32] {
        let one = Double(1 << Self.precisionBits)
        return weights.map { Int32($0 < 0 ? -0.5 + $0 * one : 0.5 + $0 * one) }
    }
}
