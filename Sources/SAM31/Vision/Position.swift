// Port of mlx_vlm/models/sam3/position.py::{compute_axial_cis, rotate_pairwise, apply_rotary_enc,
// init_2d_freqs} (mlx-vlm 0.7.3). PositionEmbeddingSine follows in Task 5.
//
// Python keeps RoPE as real (cos, sin) arrays rather than complex numbers; so does this port.
import Foundation
import MLX

/// A rotary embedding table as paired real arrays: `cos` and `sin` share one shape.
struct RotaryCIS {
    let cos: MLXArray
    let sin: MLXArray
}

/// 2D axial rotary embeddings matching HF Sam3ViTRotaryEmbedding.
///
/// `scale` multiplies the grid coordinates. Global-attention blocks pass `windowSize / featSize`;
/// windowed blocks pass 1.
///
/// - Returns: `cos` and `sin`, each `(endX*endY, dim)`.
func computeAxialCIS(dim: Int, endX: Int, endY: Int, theta: Float = 10000.0, scale: Float = 1.0)
    -> RotaryCIS
{
    // Frequencies step by 4: dim splits into an x pair and a y pair.
    let freqs = 1.0 / pow(theta, MLXArray.arange(0, dim, step: 4).asType(.float32) / Float(dim))

    // Grid positions, row-major: y changes with row, x with column.
    let flatIdx = MLXArray.arange(endX * endY)
    let xPositions = (flatIdx % endX).asType(.float32) * scale
    let yPositions = floorDivide(flatIdx, endX).asType(.float32) * scale

    // Outer products: (N, dim/4) each.
    let freqsX = xPositions[0..., .newAxis] * freqs[.newAxis, 0...]
    let freqsY = yPositions[0..., .newAxis] * freqs[.newAxis, 0...]

    // (N, dim/2)
    var invFreq = concatenated([freqsX, freqsY], axis: -1)

    // repeat_interleave(2): [f0, f0, f1, f1, ...] -> (N, dim)
    invFreq = stacked([invFreq, invFreq], axis: -1).reshaped(invFreq.dim(0), -1)

    return RotaryCIS(cos: MLX.cos(invFreq), sin: MLX.sin(invFreq))
}

/// Pairwise rotation: `(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)`.
func rotatePairwise(_ x: MLXArray) -> MLXArray {
    let x = x.reshaped(Array(x.shape.dropLast()) + [-1, 2])
    let x1 = x[.ellipsis, 0]
    let x2 = x[.ellipsis, 1]
    let rotated = stacked([-x2, x1], axis: -1)
    return rotated.reshaped(Array(rotated.shape.dropLast(2)) + [-1])
}

/// Applies 2D rotary position encoding, matching HF: `out = x * cos + rotatePairwise(x) * sin`.
///
/// - Parameters:
///   - xq: `(B, H, N, D)` queries, already transposed for SDPA.
///   - xk: `(B, H, N, D)` keys.
///   - cos: `(N, D)` cosine embeddings.
///   - sin: `(N, D)` sine embeddings.
func applyRotaryEnc(_ xq: MLXArray, _ xk: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let xqOut = xq * cos + rotatePairwise(xq) * sin
    let xkOut = xk * cos + rotatePairwise(xk) * sin
    return (xqOut, xkOut)
}

/// 2D RoPE frequencies for memory attention.
///
/// - Returns: `cos` and `sin`, each `(featH*featW, dim/2)`.
func init2DFreqs(dim: Int, featH: Int, featW: Int, theta: Float = 10000.0) -> RotaryCIS {
    let half = dim / 2
    let freqs = 1.0 / pow(theta, MLXArray.arange(0, half, step: 2).asType(.float32) / Float(half))

    let tY = MLXArray.arange(featH).asType(.float32)
    let tX = MLXArray.arange(featW).asType(.float32)

    let grid = meshGrid([tY, tX], indexing: .ij)
    let gridY = grid[0].reshaped(-1)  // (H*W,)
    let gridX = grid[1].reshaped(-1)  // (H*W,)

    let freqsY = outer(gridY, freqs)  // (H*W, half/2)
    let freqsX = outer(gridX, freqs)  // (H*W, half/2)

    // Interleave y and x frequencies: (H*W, half)
    let freqsAll = concatenated([freqsY, freqsX], axis: -1)

    return RotaryCIS(cos: MLX.cos(freqsAll), sin: MLX.sin(freqsAll))
}
