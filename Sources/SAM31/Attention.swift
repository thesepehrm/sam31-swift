import MLX
import MLXFast

/// `MLXFast.scaledDotProductAttention` that stays on MLX's fused attention kernel for small heads.
///
/// For more than 8 queries, MLX 0.31 fuses attention only for head sizes 64, 80 and 128. Any other
/// size takes an unfused fallback that materializes the full `(B, H, N_q, N_k)` score matrix: 6 GB
/// and about 200 ms per call for the tracker's memory cross-attention (8 heads of 32, 5184 queries,
/// about 36k keys). This pads q, k and v with zeros up to the next fused size and slices the output
/// back. The zero lanes add exactly 0 to every q·k product and yield 0 output lanes, so the result
/// is the same attention; only the kernel's summation order (online softmax) differs, by about 1e-6.
///
/// - Parameters:
///   - queries: `(B, H, N_q, D)`.
///   - keys: `(B, H, N_k, D)`.
///   - values: `(B, H, N_k, D)`.
/// - Returns: `(B, H, N_q, D)`.
func fusedAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray? = nil
) -> MLXArray {
    let headDim = queries.dim(-1)
    guard queries.dim(2) > 8, keys.dim(-1) == headDim, values.dim(-1) == headDim,
        let padded = [64, 80, 128].first(where: { $0 >= headDim }), padded != headDim
    else {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
    let widths: [IntOrPair] = [0, 0, 0, .init((0, padded - headDim))]
    let out = MLXFast.scaledDotProductAttention(
        queries: MLX.padded(queries, widths: widths), keys: MLX.padded(keys, widths: widths),
        values: MLX.padded(values, widths: widths), scale: scale, mask: mask)
    return out[.ellipsis, ..<headDim]
}
