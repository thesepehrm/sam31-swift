// Port of mlx_vlm/models/sam3/generate.py::_resize_masks and
// mlx_vlm/models/interpolate.py::{resize_bilinear_nhwc, _bilinear_weights_1d} (mlx-vlm 0.7.3)
//
// Both resizes are separable tap tables applied by `separableInterpolate`, which contracts the
// width taps first and then the height taps. That is the order of mlx-vlm's Metal
// `separable_interpolate` kernel and of Pillow's horizontal-then-vertical passes.
import Foundation
import MLX

/// Per-axis resampling taps: `indices[t][o]` and `weights[t][o]` are the source index and weight of
/// tap `t` for output position `o`. Unused taps have weight 0 and an in-range index.
struct ResampleTaps {
    var indices: [[Int32]]
    var weights: [[Float]]
}

/// Bilinear resize of `(N, h, w)` mask logits to `(N, size.h, size.w)`, matching Python's
/// `_resize_masks`: each mask goes through `PIL.Image.fromarray(float32).resize((W, H), BILINEAR)`.
///
/// Pillow's BILINEAR is a triangle filter on half-pixel centres whose support widens by the scale
/// factor when downsampling, with the weights renormalized per output pixel (so edges clamp). It is
/// not cv2's INTER_LINEAR and not mlx's `resize_bilinear_nhwc`. Pillow accumulates each pass in
/// double and stores float32 between passes; this port keeps float32 throughout and computes the
/// coefficients in double, so values differ from Pillow only by float32 rounding.
func resizeMasks(_ masks: MLXArray, to size: (h: Int, w: Int)) -> MLXArray {
    let (n, h, w) = (masks.dim(0), masks.dim(1), masks.dim(2))
    if n == 0 { return MLXArray.zeros([0, size.h, size.w], type: Float.self) }
    let ty = pillowBilinearTaps(inSize: h, outSize: size.h)
    let tx = pillowBilinearTaps(inSize: w, outSize: size.w)
    return separableInterpolate(masks.reshaped(n, h, w, 1), y: ty, x: tx).reshaped(n, size.h, size.w)
}

/// Channel-last bilinear resize of `(B, H, W, C)` to `(B, h, w, C)` matching PyTorch's
/// `F.interpolate(mode="bilinear")`, as ported by mlx-vlm's `resize_bilinear_nhwc`. `antialias` is
/// PyTorch's triangle filter for downscales. Returns float32.
func resizeBilinearNHWC(
    _ x: MLXArray, h: Int, w: Int, alignCorners: Bool = false, antialias: Bool = false
) -> MLXArray {
    let (inH, inW) = (x.dim(1), x.dim(2))
    if inH == h && inW == w { return x }
    let ty = atenBilinearTaps(inSize: inH, outSize: h, alignCorners: alignCorners, antialias: antialias)
    let tx = atenBilinearTaps(inSize: inW, outSize: w, alignCorners: alignCorners, antialias: antialias)
    return separableInterpolate(x, y: ty, x: tx)
}

/// Applies per-axis taps to `(B, H, W, C)`: width taps first, then height taps, in float32.
func separableInterpolate(_ x: MLXArray, y ty: ResampleTaps, x tx: ResampleTaps) -> MLXArray {
    let x = x.asType(.float32)
    let outH = ty.indices[0].count
    let outW = tx.indices[0].count

    var rows: MLXArray? = nil
    for (idx, wt) in zip(tx.indices, tx.weights) {
        let term = take(x, MLXArray(idx), axis: 2) * MLXArray(wt).reshaped(1, 1, outW, 1)
        rows = rows.map { $0 + term } ?? term
    }
    var out: MLXArray? = nil
    for (idx, wt) in zip(ty.indices, ty.weights) {
        let term = take(rows!, MLXArray(idx), axis: 1) * MLXArray(wt).reshaped(1, outH, 1, 1)
        out = out.map { $0 + term } ?? term
    }
    return out!
}

/// Port of Pillow's `precompute_coeffs` (libImaging/Resample.c) for the BILINEAR filter.
func pillowBilinearTaps(inSize: Int, outSize: Int) -> ResampleTaps {
    let scale = Double(inSize) / Double(outSize)
    let filterScale = max(scale, 1.0)
    let support = 1.0 * filterScale  // bilinear filter support is 1
    let kSize = Int(ceil(support)) * 2 + 1

    var indices = [[Int32]](repeating: [Int32](repeating: 0, count: outSize), count: kSize)
    var weights = [[Float]](repeating: [Float](repeating: 0, count: outSize), count: kSize)
    for xx in 0..<outSize {
        let center = (Double(xx) + 0.5) * scale
        let ss = 1.0 / filterScale
        // C `(int)` truncates toward zero; the clamp makes that equal to Pillow's result.
        let xmin = max(Int(center - support + 0.5), 0)
        let count = min(Int(center + support + 0.5), inSize) - xmin
        var k = [Double](repeating: 0, count: count)
        var ww = 0.0
        for x in 0..<count {
            let t = abs((Double(x + xmin) - center + 0.5) * ss)
            k[x] = t < 1.0 ? 1.0 - t : 0.0
            ww += k[x]
        }
        for x in 0..<count {
            indices[x][xx] = Int32(x + xmin)
            weights[x][xx] = Float(ww != 0 ? k[x] / ww : k[x])
        }
        for x in count..<kSize {
            indices[x][xx] = Int32(min(xmin, inSize - 1))
        }
    }
    return ResampleTaps(indices: indices, weights: weights)
}

/// Port of `interpolate._bilinear_weights_1d`: ATen `upsample_bilinear2d` taps, or the
/// `_upsample_bilinear2d_aa` triangle filter for antialiased downscales. Float32 math, as in Python.
func atenBilinearTaps(inSize: Int, outSize: Int, alignCorners: Bool, antialias: Bool) -> ResampleTaps {
    let scaleD: Double =
        alignCorners
        ? (outSize > 1 ? Double(inSize - 1) / Double(outSize - 1) : 0.0)
        : Double(inSize) / Double(outSize)
    let scale = Float(scaleD)
    let src: [Float] = (0..<outSize).map {
        alignCorners ? Float($0) * scale : (Float($0) + 0.5) * scale - 0.5
    }
    let last = Float(inSize - 1)

    if antialias && scaleD > 1.0 {
        let nTaps = Int(ceil(2 * scaleD))
        var indices = [[Int32]](repeating: [Int32](repeating: 0, count: outSize), count: nTaps)
        var weights = [[Float]](repeating: [Float](repeating: 0, count: outSize), count: nTaps)
        for o in 0..<outSize {
            let lo = max((src[o] - scale + 1).rounded(.down), 0)
            let hi = min((src[o] + scale).rounded(.down), last)
            var w = (0..<nTaps).map { t -> Float in
                let pos = lo + Float(t)
                return pos <= hi ? max(0, 1 - abs(pos - src[o]) / scale) : 0
            }
            let total = w.reduce(0, +)
            w = w.map { $0 / total }
            for t in 0..<nTaps {
                indices[t][o] = Int32(min(lo + Float(t), last))
                weights[t][o] = w[t]
            }
        }
        return ResampleTaps(indices: indices, weights: weights)
    }

    var indices = [[Int32]](repeating: [Int32](repeating: 0, count: outSize), count: 2)
    var weights = [[Float]](repeating: [Float](repeating: 0, count: outSize), count: 2)
    for o in 0..<outSize {
        let s = max(src[o], 0)
        let base = s.rounded(.down)
        let frac = s - base
        indices[0][o] = Int32(min(max(base, 0), last))
        indices[1][o] = Int32(min(max(base + 1, 0), last))
        weights[0][o] = 1 - frac
        weights[1][o] = frac
    }
    return ResampleTaps(indices: indices, weights: weights)
}
