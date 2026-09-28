// Host-side upsample for `Mask.upsampled(to:threshold:)`. Not a port: mlx-vlm resizes the thresholded
// mask with nearest instead (see `Mask.upsampled`).
//
// The math is `separableInterpolate` with `atenBilinearTaps` on the CPU: for each output pixel, the
// two width taps of each source row, then the two height taps over those rows, each pass summed as
// `(0 + v0·w0) + v1·w1` in float32. It runs on vDSP so that it is fast whatever the build: the
// equivalent scalar loop takes over a second per 1080p mask in a Debug build.
//
// Every product and sum is a separate vDSP multiply and add, never a multiply-add (`vDSP_vma`,
// `vDSP_vsma`), because a fused multiply-add rounds once instead of twice and can flip a pixel whose
// value sits within an ulp of the threshold. `MaskTests` checks the bytes against the scalar loop.
import Accelerate

/// Bilinear resize of `(inH, inW)` row-major logits to `(outH, outW)`, thresholded: 255 where the
/// resized value is above `threshold`, 0 elsewhere (including NaN).
func upsampleBinary(
    _ values: [Float], width inW: Int, height inH: Int, toWidth outW: Int, height outH: Int,
    threshold: Float
) -> [UInt8] {
    precondition(inW > 0 && inH > 0 && outW > 0 && outH > 0 && values.count == inW * inH)
    // `x > threshold` is `x >= threshold.nextUp` for every x except +inf when threshold is +inf.
    guard threshold != .infinity else { return [UInt8](repeating: 0, count: outW * outH) }

    let tx = TwoTaps(inSize: inW, outSize: outW)
    let ty = TwoTaps(inSize: inH, outSize: outH)
    let n = vDSP_Length(outW)

    // Width pass: each source row to `outW` values.
    var rows = [Float](repeating: 0, count: inH * outW)
    var scratch = [Float](repeating: 0, count: outW)
    values.withUnsafeBufferPointer { values in
        rows.withUnsafeMutableBufferPointer { rows in
            scratch.withUnsafeMutableBufferPointer { scratch in
                let tmp = scratch.baseAddress!
                for y in 0..<inH {
                    let src = values.baseAddress! + y * inW
                    let row = rows.baseAddress! + y * outW
                    vDSP_vindex(src, tx.index0, 1, row, 1, n)
                    vDSP_vmul(row, 1, tx.weight0, 1, row, 1, n)
                    vDSP_vindex(src, tx.index1, 1, tmp, 1, n)
                    vDSP_vmul(tmp, 1, tx.weight1, 1, tmp, 1, n)
                    vDSP_vadd(row, 1, tmp, 1, row, 1, n)
                }
            }
        }
    }

    // Height pass and threshold, one output row at a time.
    var bytes = [UInt8](repeating: 0, count: outW * outH)
    var column = [Float](repeating: 0, count: outW)
    var above = threshold.nextUp
    var high: Float = 255
    var zero: Float = 0
    rows.withUnsafeBufferPointer { rows in
        bytes.withUnsafeMutableBufferPointer { bytes in
            column.withUnsafeMutableBufferPointer { column in
                scratch.withUnsafeMutableBufferPointer { scratch in
                    let col = column.baseAddress!
                    let tmp = scratch.baseAddress!
                    for y in 0..<outH {
                        var w0 = ty.weight0[y]
                        var w1 = ty.weight1[y]
                        vDSP_vsmul(rows.baseAddress! + Int(ty.index0[y]) * outW, 1, &w0, col, 1, n)
                        vDSP_vsmul(rows.baseAddress! + Int(ty.index1[y]) * outW, 1, &w1, tmp, 1, n)
                        vDSP_vadd(col, 1, tmp, 1, col, 1, n)
                        // x >= threshold.nextUp ? 255 : -255 (NaN compares false), then clamp to 0.
                        vDSP_vlim(col, 1, &above, &high, col, 1, n)
                        vDSP_vthr(col, 1, &zero, col, 1, n)
                        vDSP_vfixu8(col, 1, bytes.baseAddress! + y * outW, 1, n)
                    }
                }
            }
        }
    }
    return bytes
}

/// `atenBilinearTaps` without antialias, as flat per-tap arrays. The indices are floats because
/// `vDSP_vindex` takes float indices; they are small integers, so the conversion is exact.
private struct TwoTaps {
    var index0: [Float]
    var index1: [Float]
    var weight0: [Float]
    var weight1: [Float]

    init(inSize: Int, outSize: Int) {
        let taps = atenBilinearTaps(inSize: inSize, outSize: outSize, alignCorners: false, antialias: false)
        precondition(taps.indices.count == 2)
        index0 = taps.indices[0].map(Float.init)
        index1 = taps.indices[1].map(Float.init)
        weight0 = taps.weights[0]
        weight1 = taps.weights[1]
    }
}
