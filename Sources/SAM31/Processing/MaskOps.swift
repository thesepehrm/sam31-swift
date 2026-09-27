// Port of mlx_vlm/models/sam3/generate.py::_resize_masks,
// mlx_vlm/models/interpolate.py::{resize_bilinear_nhwc, _bilinear_weights_1d} and
// mlx_vlm/models/kernels.py::separable_interpolate (mlx-vlm 0.7.3)
//
// Both resizes are separable tap tables applied by `separableInterpolate`, mlx-vlm's Metal
// `separable_interpolate` kernel, which contracts the width taps first and then the height taps.
// That is also the order of Pillow's horizontal-then-vertical passes.
import Foundation
import MLX
import MLXFast

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

/// Applies per-axis taps to `(B, H, W, C)` in float32: for each output pixel, the width taps of each
/// source row, then the height taps over those rows.
///
/// Runs mlx-vlm's `separable_interpolate` Metal kernel (`models/kernels.py`), one thread per output
/// element, so no `(B, H, W_out, C)` intermediate is written.
func separableInterpolate(_ x: MLXArray, y ty: ResampleTaps, x tx: ResampleTaps) -> MLXArray {
    let (b, c) = (x.dim(0), x.dim(3))
    let (outH, outW) = (ty.indices[0].count, tx.indices[0].count)
    // (out, taps) row-major tables, as the kernel indexes them.
    func table<T: HasDType>(_ rows: [[T]], _ out: Int) -> MLXArray {
        MLXArray(rows.flatMap { $0 }, [rows.count, out]).transposed().contiguous()
    }
    return separableInterpolateKernel(
        [
            x.asType(.float32), table(ty.indices, outH), table(ty.weights, outH), table(tx.indices, outW),
            table(tx.weights, outW),
        ],
        grid: (outW * c, outH, b), threadGroup: (min(256, outW * c), 1, 1),
        outputShapes: [[b, outH, outW, c]], outputDTypes: [.float32])[0]
}

/// mlx-vlm 0.7.3 `models/kernels.py::separable_interpolate`, verbatim.
private let separableInterpolateKernel = MLXFast.metalKernel(
    name: "separable_interpolate", inputNames: ["x", "iy", "wy", "ix", "wx"], outputNames: ["out"],
    source: """
            uint gx = thread_position_in_grid.x;
            uint y_out = thread_position_in_grid.y;
            uint b = thread_position_in_grid.z;

            int in_h = x_shape[1];
            int in_w = x_shape[2];
            int channels = x_shape[3];
            int out_h = iy_shape[0];
            int taps_y = iy_shape[1];
            int out_w = ix_shape[0];
            int taps_x = ix_shape[1];

            if (gx >= (uint)(out_w * channels) || y_out >= (uint)out_h)
                return;

            int x_out = gx / channels;
            size_t y_tap = (size_t)y_out * taps_y;
            size_t x_tap = (size_t)x_out * taps_x;
            size_t input_base = (size_t)b * in_h * in_w * channels + (gx % channels);

            // W taps are contracted inside the H loop to match ATen's summation order
            float result = 0.0f;
            for (int a = 0; a < taps_y; a++) {
                size_t row = input_base + (size_t)iy[y_tap + a] * in_w * channels;
                float row_result = 0.0f;
                for (int c = 0; c < taps_x; c++) {
                    row_result += x[row + (size_t)ix[x_tap + c] * channels] * wx[x_tap + c];
                }
                result += row_result * wy[y_tap + a];
            }

            out[((size_t)b * out_h + y_out) * out_w * channels + gx] = result;
        """)

/// Pillow's BILINEAR coefficients (`PillowCoefficients`) as float32 taps for `separableInterpolate`.
func pillowBilinearTaps(inSize: Int, outSize: Int) -> ResampleTaps {
    let c = PillowCoefficients(bilinearFrom: inSize, to: outSize)
    var indices = [[Int32]](repeating: [Int32](repeating: 0, count: outSize), count: c.kSize)
    var weights = [[Float]](repeating: [Float](repeating: 0, count: outSize), count: c.kSize)
    for xx in 0..<outSize {
        let (xmin, count) = (c.xmin[xx], c.count[xx])
        for x in 0..<count {
            indices[x][xx] = Int32(x + xmin)
            weights[x][xx] = Float(c.weights[xx * c.kSize + x])
        }
        for x in count..<c.kSize {
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
