// Port of mlx_vlm/models/sam3/text_encoder.py (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXFast
import MLXNN

// Port of mlx_vlm/models/sam3/text_encoder.py::CLIPAttention (mlx-vlm 0.7.3)
final class CLIPAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: TextEncoderConfig) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (B, N, C) = (x.dim(0), x.dim(1), x.dim(2))
        let q = qProj(x).reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(B, N, C))
    }
}

final class CLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: TextEncoderConfig) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Python calls `nn.gelu` (exact erf) unconditionally; `hidden_act` is not consulted.
        fc2(gelu(fc1(x)))
    }
}

final class CLIPEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CLIPAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: CLIPMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: TextEncoderConfig) {
        _selfAttn.wrappedValue = CLIPAttention(config)
        _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = CLIPMLP(config)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var x = x + selfAttn(layerNorm1(x), mask: mask)
        x = x + mlp(layerNorm2(x))
        return x
    }
}

final class CLIPEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [CLIPEncoderLayer]

    init(_ config: TextEncoderConfig) {
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in CLIPEncoderLayer(config) }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var x = x
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return x
    }
}

final class CLIPTextEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(_ config: TextEncoderConfig) {
        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings, dimensions: config.hiddenSize)
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seqLen = inputIds.dim(1)
        let positionIds = MLXArray(0..<seqLen)
        return tokenEmbedding(inputIds) + positionEmbedding(positionIds)
    }
}

/// Weight keys: `detector_model.text_encoder.text_model.*`.
final class CLIPTextModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: CLIPTextEmbeddings
    @ModuleInfo(key: "encoder") var encoder: CLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: TextEncoderConfig) {
        _embeddings.wrappedValue = CLIPTextEmbeddings(config)
        _encoder.wrappedValue = CLIPEncoder(config)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    /// `inputIds` (B, seq_len) token ids, `attentionMask` (B, seq_len) binary mask.
    /// Returns hidden states (B, seq_len, hidden_size).
    func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let x = embeddings(inputIds)

        // Causal mask
        let seqLen = inputIds.dim(1)
        var causalMask = Self.additiveCausalMask(seqLen).asType(x.dtype)

        if let attentionMask {
            // Combine causal mask with padding mask
            let padMask = (1 - attentionMask[0..., .newAxis, .newAxis, 0...].asType(x.dtype)) * -1e9
            causalMask = causalMask + padMask
        }

        return finalLayerNorm(encoder(x, mask: causalMask))
    }

    /// Python's `nn.MultiHeadAttention.create_additive_causal_mask(n)`: a float32 mask whose masked
    /// entries are `mx.finfo(float32).min`. MLXNN's `createAdditiveCausalMask` uses -1e9 instead, so
    /// it is not used here.
    static func additiveCausalMask(_ n: Int) -> MLXArray {
        let indices = MLXArray(0..<n)
        let mask = indices[0..., .newAxis] .< indices[.newAxis]
        return mask.asType(.float32) * -Float.greatestFiniteMagnitude
    }
}

/// CLIP text model plus projection. Weight keys:
/// `detector_model.text_encoder.text_model.*` and `detector_model.text_encoder.text_projection.weight`.
final class TextEncoder: Module {
    @ModuleInfo(key: "text_model") var textModel: CLIPTextModel
    /// Projects CLIP hidden to `projection_dim` (1024 -> 512). Loaded but unused in forward, as in Python.
    @ModuleInfo(key: "text_projection") var textProjection: Linear

    init(_ config: TextEncoderConfig) {
        _textModel.wrappedValue = CLIPTextModel(config)
        _textProjection.wrappedValue = Linear(config.hiddenSize, config.projectionDim, bias: false)
    }

    /// Returns the raw hidden states (B, seq_len, hidden_size).
    func callAsFunction(_ ids: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        textModel(ids, attentionMask: attentionMask)
    }
}
