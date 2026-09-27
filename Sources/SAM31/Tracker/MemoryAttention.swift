// Port of mlx_vlm/models/sam3_1/sam_components.py::{SimpleRoPEAttention, DecoupledMemoryAttentionLayer,
// DecoupledMemoryAttention} (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXNN

/// RoPE attention without Q/K/V projections (the caller applies them): 2D rotary encoding plus
/// scaled dot-product attention.
///
/// Not a `Module`: Python's instance only holds the private `_freqs_cos`/`_freqs_sin` tables, which
/// are not parameters.
final class SimpleRoPEAttention {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let ropeKRepeat: Bool
    let freqs: RotaryCIS

    init(
        hiddenSize: Int, numHeads: Int, featSizes: [Int] = [72, 72], ropeTheta: Float = 10000.0,
        ropeKRepeat: Bool = false
    ) {
        self.numHeads = numHeads
        headDim = hiddenSize / numHeads
        scale = pow(Float(headDim), -0.5)
        self.ropeKRepeat = ropeKRepeat
        freqs = init2DFreqs(
            dim: hiddenSize / numHeads, featH: featSizes[0], featW: featSizes[1], theta: ropeTheta)
    }

    /// - Parameters:
    ///   - q: `(B, N_q, D)` pre-projected queries.
    ///   - k: `(B, N_k, D)` pre-projected keys.
    ///   - v: `(B, N_k, D)` pre-projected values.
    ///   - numKExcludeRope: the last `numKExcludeRope` keys (object pointers) skip RoPE.
    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, numKExcludeRope: Int = 0) -> MLXArray {
        let (b, nQ) = (q.dim(0), q.dim(1))
        let nK = k.dim(1)

        var q = q.reshaped(b, nQ, numHeads, headDim)
        var k = k.reshaped(b, nK, numHeads, headDim)
        var v = v.reshaped(b, nK, numHeads, headDim)

        // Apply RoPE (exclude the last numKExcludeRope keys)
        var kRope = k
        var kNoRope: MLXArray? = nil
        if numKExcludeRope > 0 {
            kRope = k[0..., ..<(nK - numKExcludeRope)]
            kNoRope = k[0..., (nK - numKExcludeRope)...]
        }

        (q, kRope) = applyRotaryEnc1D(
            q, kRope, freqsCos: freqs.cos, freqsSin: freqs.sin, repeatFreqsK: ropeKRepeat)

        if let kNoRope {
            k = concatenated([kRope, kNoRope], axis: 1)
        } else {
            k = kRope
        }

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Python calls mx.fast.scaled_dot_product_attention; see `fusedAttention` for the padding.
        let out = fusedAttention(queries: q, keys: k, values: v, scale: scale)
        return out.transposed(0, 2, 1, 3).reshaped(b, nQ, -1)
    }
}

/// Pre-norm decoupled transformer layer with separate self/cross-attention projections plus the
/// image cross-attention Q/K projections. Weight keys: `tracker_model.memory_attention.layers.*`.
final class DecoupledMemoryAttentionLayer: Module {
    @ModuleInfo(key: "self_attn_q_proj") var selfAttnQProj: Linear
    @ModuleInfo(key: "self_attn_k_proj") var selfAttnKProj: Linear
    @ModuleInfo(key: "self_attn_v_proj") var selfAttnVProj: Linear
    @ModuleInfo(key: "self_attn_out_proj") var selfAttnOutProj: Linear

    @ModuleInfo(key: "cross_attn_q_proj") var crossAttnQProj: Linear
    @ModuleInfo(key: "cross_attn_k_proj") var crossAttnKProj: Linear
    @ModuleInfo(key: "cross_attn_v_proj") var crossAttnVProj: Linear
    @ModuleInfo(key: "cross_attn_out_proj") var crossAttnOutProj: Linear

    @ModuleInfo(key: "image_cross_attn_q_proj") var imageCrossAttnQProj: Linear
    @ModuleInfo(key: "image_cross_attn_k_proj") var imageCrossAttnKProj: Linear

    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    let selfAttentionRope: SimpleRoPEAttention
    let crossAttentionRope: SimpleRoPEAttention

    init(_ config: TrackerConfig, selfAttnRope: SimpleRoPEAttention, crossAttnRope: SimpleRoPEAttention) {
        let d = config.memoryAttentionHiddenSize
        selfAttentionRope = selfAttnRope
        crossAttentionRope = crossAttnRope
        _selfAttnQProj.wrappedValue = Linear(d, d)
        _selfAttnKProj.wrappedValue = Linear(d, d)
        _selfAttnVProj.wrappedValue = Linear(d, d)
        _selfAttnOutProj.wrappedValue = Linear(d, d)
        _crossAttnQProj.wrappedValue = Linear(d, d)
        _crossAttnKProj.wrappedValue = Linear(d, d)
        _crossAttnVProj.wrappedValue = Linear(d, d)
        _crossAttnOutProj.wrappedValue = Linear(d, d)
        _imageCrossAttnQProj.wrappedValue = Linear(d, d)
        _imageCrossAttnKProj.wrappedValue = Linear(d, d)
        _linear1.wrappedValue = Linear(d, config.memoryAttentionFeedForwardHiddenSize)
        _linear2.wrappedValue = Linear(config.memoryAttentionFeedForwardHiddenSize, d)
        _norm1.wrappedValue = LayerNorm(dimensions: d)
        _norm2.wrappedValue = LayerNorm(dimensions: d)
        _norm3.wrappedValue = LayerNorm(dimensions: d)
    }

    /// Pre-norm decoupled layer (DecoupledTransformerDecoderLayerv2 port).
    ///
    /// - Parameters:
    ///   - image: `(1, HW, D)` raw current-frame image features (not normed).
    ///   - src: `(B, HW, D)` current-frame features (self-attention).
    ///   - memoryImage: `(1, N, D)` image features of the memory frames.
    ///   - memory: `(B, N, D)` mask-memory features (+ object pointers).
    ///   - memoryImagePos: `(1, N, D)` pos enc for the memory keys.
    ///   - numKExcludeRope: trailing keys excluded from RoPE (object pointers).
    func callAsFunction(
        image: MLXArray, src: MLXArray, memoryImage: MLXArray, memory: MLXArray, memoryImagePos: MLXArray?,
        numKExcludeRope: Int
    ) -> MLXArray {
        // 1. Self-attention with RoPE (pre-norm, no pos enc at attention)
        var residual = src
        var srcNormed = norm1(src)
        var q = selfAttnQProj(srcNormed)
        var k = selfAttnKProj(srcNormed)
        var v = selfAttnVProj(srcNormed)
        var src2 = selfAttentionRope(q, k, v)
        src2 = selfAttnOutProj(src2)
        var src = residual + src2

        // 2. Cross-attention to memory with RoPE (pre-norm); q/k get additional projections of the
        // (raw) image features
        residual = src
        srcNormed = norm2(src)
        q = imageCrossAttnQProj(image) + crossAttnQProj(srcNormed)
        k = imageCrossAttnKProj(memoryImage) + crossAttnKProj(memory)
        if let memoryImagePos {
            // pos enc at cross-attention keys only
            k = k + memoryImagePos
        }
        v = crossAttnVProj(memory)

        src2 = crossAttentionRope(q, k, v, numKExcludeRope: numKExcludeRope)
        src2 = crossAttnOutProj(src2)
        src = residual + src2

        // 3. FFN (pre-norm, gelu)
        residual = src
        src2 = linear2(gelu(linear1(norm3(src))))
        src = residual + src2

        eval(src)  // Free attention intermediates
        return src
    }
}

/// SAM 3.1 memory attention with decoupled projections.
/// Weight keys: `tracker_model.memory_attention.*`.
final class DecoupledMemoryAttention: Module {
    @ModuleInfo(key: "layers") var layers: [DecoupledMemoryAttentionLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: TrackerConfig) {
        let d = config.memoryAttentionHiddenSize
        let heads = config.memoryAttentionNumAttentionHeads
        let featSizes = config.memoryAttentionRopeFeatSizes
        let theta = config.memoryAttentionRopeTheta
        _layers.wrappedValue = (0..<config.memoryAttentionNumLayers).map { _ in
            DecoupledMemoryAttentionLayer(
                config,
                selfAttnRope: SimpleRoPEAttention(
                    hiddenSize: d, numHeads: heads, featSizes: featSizes, ropeTheta: theta),
                crossAttnRope: SimpleRoPEAttention(
                    hiddenSize: d, numHeads: heads, featSizes: featSizes, ropeTheta: theta, ropeKRepeat: true)
            )
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.memoryAttentionHiddenSize)
    }

    /// - Parameters:
    ///   - image: `(1, HW, D)` raw current-frame image features.
    ///   - src: `(B, HW, D)` current-frame features.
    ///   - memoryImage: `(1, N_img, D)` image features of the memory frames.
    ///   - memory: `(B, N_mem, D)` mask memories plus object-pointer tokens.
    ///   - srcPos: `(B, HW, D)` pos enc added to `src` at input (scaled by 0.1).
    ///   - memoryPos: `(B, N_mem, D)`; only its object-pointer tail is used.
    ///   - memoryImagePos: `(1, N_img, D)` pos enc for the memory image keys.
    ///   - numKExcludeRope: number of trailing object-pointer tokens.
    func callAsFunction(
        image: MLXArray, src: MLXArray, memoryImage: MLXArray, memory: MLXArray, srcPos: MLXArray?,
        memoryPos: MLXArray?, memoryImagePos: MLXArray?, numKExcludeRope: Int
    ) -> MLXArray {
        var src = src
        var memoryImage = memoryImage
        var memoryImagePos = memoryImagePos
        // pos enc at input (scaled by 0.1 as in the reference)
        if let srcPos {
            src = src + 0.1 * srcPos
        }

        // Pad the image memories with zeros for the object pointer tokens
        if memoryImage.dim(1) != memory.dim(1) {
            let pad = memory.dim(1) - memoryImage.dim(1)
            precondition(pad == numKExcludeRope)
            memoryImage = concatenated(
                [memoryImage, MLXArray.zeros([memoryImage.dim(0), pad, memoryImage.dim(2)])], axis: 1)
            if let pos = memoryImagePos, let memoryPos {
                let n = memoryPos.dim(1)
                memoryImagePos = concatenated([pos, memoryPos[0..<1, (n - pad)...]], axis: 1)
            }
        }

        for layer in layers {
            src = layer(
                image: image, src: src, memoryImage: memoryImage, memory: memory,
                memoryImagePos: memoryImagePos, numKExcludeRope: numKExcludeRope)
        }
        // use_image_in_output=False: norm the output only
        return layerNorm(src)
    }
}
