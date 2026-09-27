// Port of mlx_vlm/models/sam3/encoder.py::{MultiheadAttention, DETREncoderLayer, MLP, DETREncoder}
// (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXNN

/// Multi-head attention with separate q/k/v/o projections. Shared by the DETR encoder and decoder,
/// the geometry encoder and the segmentation head. Not MLXNN's `MultiHeadAttention`: the keys differ.
final class MultiheadAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    /// `dropout` is accepted for parity with Python's signature; inference never applies it.
    init(hiddenSize: Int, numHeads: Int, dropout: Float = 0, kvDim: Int? = nil) {
        self.numHeads = numHeads
        headDim = hiddenSize / numHeads
        scale = pow(Float(headDim), -0.5)
        let kvDim = kvDim ?? hiddenSize
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue = Linear(kvDim, hiddenSize)
        _vProj.wrappedValue = Linear(kvDim, hiddenSize)
        _oProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray? = nil) -> MLXArray
    {
        let (B, Nq) = (query.dim(0), query.dim(1))
        let nk = key.dim(1)
        let q = qProj(query).reshaped(B, Nq, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped(B, nk, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped(B, nk, numHeads, headDim).transposed(0, 2, 1, 3)
        // Python calls mx.fast.scaled_dot_product_attention; see `fusedAttention` for the padding.
        let out = fusedAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, Nq, -1))
    }
}

/// Two-layer feed-forward block: `fc1 -> act -> fc2`. `act` is `relu`, anything else is exact `gelu`.
final class MLP: Module {
    let act: String

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int, act: String = "relu") {
        self.act = act
        _fc1.wrappedValue = Linear(hiddenSize, intermediateSize)
        _fc2.wrappedValue = Linear(intermediateSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = fc1(x)
        x = act == "relu" ? relu(x) : gelu(x)
        return fc2(x)
    }
}

/// Pre-norm layer: image self-attention (pos on q/k only), text cross-attention, FFN.
final class DETREncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MultiheadAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiheadAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP

    init(_ config: DETREncoderConfig) {
        let d = config.hiddenSize
        _selfAttn.wrappedValue = MultiheadAttention(
            hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout)
        _crossAttn.wrappedValue = MultiheadAttention(
            hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout)
        _layerNorm1.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _layerNorm3.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _mlp.wrappedValue = MLP(
            hiddenSize: d, intermediateSize: config.intermediateSize, act: config.hiddenAct)
    }

    /// - Parameters:
    ///   - src: `(B, HW, D)` image features.
    ///   - pos: `(B, HW, D)` position encoding.
    ///   - textMemory: `(B, T, D)` text features.
    ///   - textMask: `(B, T)` padding mask, 1 = valid, 0 = pad (any dtype).
    func callAsFunction(_ src: MLXArray, pos: MLXArray, textMemory: MLXArray, textMask: MLXArray?) -> MLXArray
    {
        // 1. Self-attention with pre-norm; pos added to q/k only.
        var residual = src
        var hidden = layerNorm1(src)
        let hiddenWithPos = hidden + pos
        var src2 = selfAttn(hiddenWithPos, key: hiddenWithPos, value: hidden)
        var src = residual + src2

        // 2. Cross-attention to text with pre-norm.
        var crossMask: MLXArray? = nil
        if let textMask {
            crossMask = (1 - textMask[0..., .newAxis, .newAxis, 0...].asType(src.dtype)) * -1e9
        }

        residual = src
        hidden = layerNorm2(src)
        src2 = crossAttn(hidden, key: textMemory, value: textMemory, mask: crossMask)
        src = residual + src2

        // 3. FFN with pre-norm.
        residual = src
        hidden = layerNorm3(src)
        src2 = mlp(hidden)
        src = residual + src2

        return src
    }
}

/// DETR transformer encoder with text fusion. Weight keys: `detector_model.detr_encoder.layers.*`.
final class DETREncoder: Module {
    @ModuleInfo(key: "layers") var layers: [DETREncoderLayer]

    init(_ config: DETREncoderConfig) {
        _layers.wrappedValue = (0..<config.numLayers).map { _ in DETREncoderLayer(config) }
    }

    /// - Parameters:
    ///   - src: `(B, HW, D)` flattened image features (1x detection level).
    ///   - pos: `(B, HW, D)` position encoding.
    ///   - prompt: `(B, T, D)` projected text embeddings.
    ///   - promptMask: `(B, T)` text attention mask, 1 = valid.
    /// - Returns: `(B, HW, D)` encoded features.
    func callAsFunction(_ src: MLXArray, pos: MLXArray, prompt: MLXArray, promptMask: MLXArray?) -> MLXArray {
        var output = src
        for layer in layers {
            output = layer(output, pos: pos, textMemory: prompt, textMask: promptMask)
            eval(output)
        }
        return output
    }
}
