// Port of mlx_vlm/models/sam3/sam_components.py::{MLPBlock, SAMAttention, TwoWayAttentionBlock,
// TwoWayTransformer, OutputMLP} (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXFast
import MLXNN

/// Two-layer MLP with ReLU or GELU.
final class MLPBlock: Module {
    let act: String
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(inputDim: Int, hiddenDim: Int, act: String = "relu") {
        self.act = act
        _projIn.wrappedValue = Linear(inputDim, hiddenDim)
        _projOut.wrappedValue = Linear(hiddenDim, inputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = projIn(x)
        x = act == "gelu" ? gelu(x) : relu(x)
        return projOut(x)
    }
}

/// Multi-head attention with an optionally downsampled internal dimension.
final class SAMAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(hiddenSize: Int, numHeads: Int, downsampleRate: Int = 1) {
        self.numHeads = numHeads
        let internalDim = hiddenSize / downsampleRate
        headDim = internalDim / numHeads
        scale = Float(pow(Double(headDim), -0.5))  // Python: float64 head_dim**-0.5
        _qProj.wrappedValue = Linear(hiddenSize, internalDim)
        _kProj.wrappedValue = Linear(hiddenSize, internalDim)
        _vProj.wrappedValue = Linear(hiddenSize, internalDim)
        _oProj.wrappedValue = Linear(internalDim, hiddenSize)
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let (b, nQ) = (q.dim(0), q.dim(1))
        let nK = k.dim(1)
        let qh = qProj(q).reshaped(b, nQ, numHeads, headDim).transposed(0, 2, 1, 3)
        let kh = kProj(k).reshaped(b, nK, numHeads, headDim).transposed(0, 2, 1, 3)
        let vh = vProj(v).reshaped(b, nK, numHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale, mask: nil)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, nQ, -1))
    }
}

/// SAM two-way attention block: token self-attention, token-to-image, MLP, image-to-token.
final class TwoWayAttentionBlock: Module {
    let skipFirstLayerPE: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: SAMAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "cross_attn_token_to_image") var crossAttnTokenToImage: SAMAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLPBlock
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "cross_attn_image_to_token") var crossAttnImageToToken: SAMAttention
    @ModuleInfo(key: "layer_norm4") var layerNorm4: LayerNorm

    init(
        hiddenSize: Int, numHeads: Int, mlpDim: Int = 2048, attentionDownsampleRate: Int = 2,
        skipFirstLayerPE: Bool = false
    ) {
        self.skipFirstLayerPE = skipFirstLayerPE
        _selfAttn.wrappedValue = SAMAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        _layerNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _crossAttnTokenToImage.wrappedValue = SAMAttention(
            hiddenSize: hiddenSize, numHeads: numHeads, downsampleRate: attentionDownsampleRate)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _mlp.wrappedValue = MLPBlock(inputDim: hiddenSize, hiddenDim: mlpDim, act: "relu")
        _layerNorm3.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _crossAttnImageToToken.wrappedValue = SAMAttention(
            hiddenSize: hiddenSize, numHeads: numHeads, downsampleRate: attentionDownsampleRate)
        _layerNorm4.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, queryPE: MLXArray, keyPE: MLXArray)
        -> (MLXArray, MLXArray)
    {
        var queries = queries
        var keys = keys

        // Self-attention on the tokens. The first layer has no query PE and replaces the queries.
        if skipFirstLayerPE {
            queries = selfAttn(queries, queries, queries)
        } else {
            let q = queries + queryPE
            queries = queries + selfAttn(q, q, queries)
        }
        queries = layerNorm1(queries)

        // Tokens attend to the image.
        var q = queries + queryPE
        var k = keys + keyPE
        queries = queries + crossAttnTokenToImage(q, k, keys)
        queries = layerNorm2(queries)

        queries = queries + mlp(queries)
        queries = layerNorm3(queries)

        // The image attends to the tokens.
        q = keys + keyPE
        k = queries + queryPE
        keys = keys + crossAttnImageToToken(q, k, queries)
        keys = layerNorm4(keys)

        return (queries, keys)
    }
}

/// SAM two-way transformer used inside the mask decoder.
final class TwoWayTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [TwoWayAttentionBlock]
    @ModuleInfo(key: "final_attn_token_to_image") var finalAttnTokenToImage: SAMAttention
    @ModuleInfo(key: "layer_norm_final_attn") var layerNormFinalAttn: LayerNorm

    init(
        hiddenSize: Int = 256, numHeads: Int = 8, numLayers: Int = 2, mlpDim: Int = 2048,
        attentionDownsampleRate: Int = 2
    ) {
        _layers.wrappedValue = (0..<numLayers).map { i in
            TwoWayAttentionBlock(
                hiddenSize: hiddenSize, numHeads: numHeads, mlpDim: mlpDim,
                attentionDownsampleRate: attentionDownsampleRate, skipFirstLayerPE: i == 0)
        }
        _finalAttnTokenToImage.wrappedValue = SAMAttention(
            hiddenSize: hiddenSize, numHeads: numHeads, downsampleRate: attentionDownsampleRate)
        _layerNormFinalAttn.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    /// - Parameters:
    ///   - imageEmbedding: `(B, HW, D)`.
    ///   - imagePE: `(B, HW, D)`.
    ///   - pointEmbedding: `(B, N_tokens, D)`.
    /// - Returns: queries `(B, N_tokens, D)` and keys `(B, HW, D)`.
    func callAsFunction(_ imageEmbedding: MLXArray, _ imagePE: MLXArray, _ pointEmbedding: MLXArray)
        -> (MLXArray, MLXArray)
    {
        var queries = pointEmbedding
        var keys = imageEmbedding
        for layer in layers {
            (queries, keys) = layer(queries: queries, keys: keys, queryPE: pointEmbedding, keyPE: imagePE)
        }

        let q = queries + pointEmbedding
        let k = keys + imagePE
        queries = queries + finalAttnTokenToImage(q, k, keys)
        queries = layerNormFinalAttn(queries)
        return (queries, keys)
    }
}

/// Three-layer ReLU MLP for the decoder's output heads.
final class OutputMLP: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "layers") var layers: [Linear]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        _projIn.wrappedValue = Linear(inputDim, hiddenDim)
        _layers.wrappedValue = [Linear(hiddenDim, hiddenDim)]
        _projOut.wrappedValue = Linear(hiddenDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = relu(projIn(x))
        for layer in layers {
            x = relu(layer(x))
        }
        return projOut(x)
    }
}
