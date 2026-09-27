// Port of mlx_vlm/models/sam3/vision.py::{PatchProjection, PatchEmbeddings, VitAttention, VitMLP,
// VitBlock, _window_partition, _window_unpartition, ViTBackbone} (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXFast
import MLXNN

/// Inner projection layer, matching the weight key `patch_embeddings.projection`.
final class PatchProjection: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ config: ViTConfig) {
        _projection.wrappedValue = Conv2d(
            inputChannels: config.numChannels, outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize), stride: .init(config.patchSize), bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(x)
    }
}

/// Patch embedding with a Conv2d projection. The absolute position embeddings are applied by
/// ``ViTBackbone``, not here.
final class PatchEmbeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: PatchProjection
    /// (1, num_pretrain_patches, hidden_size)
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    init(_ config: ViTConfig) {
        _patchEmbeddings.wrappedValue = PatchProjection(config)
        let side = config.pretrainImageSize / config.patchSize
        let numPatches = side * side
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, numPatches, config.hiddenSize])
    }

    /// (B, H, W, C) channel-last image -> (B, num_patches, hidden_size)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = patchEmbeddings(x)  // (B, H', W', hidden_size)
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(B, H * W, C)  // (B, N, C)
    }
}

/// Multi-head attention with 2D RoPE.
final class VitAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let useRope: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: ViTConfig, useRope: Bool = true) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _oProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self.useRope = useRope
    }

    /// - Parameters:
    ///   - x: (B, N, C), or spatial (B, H, W, C).
    ///   - rope: `cos` and `sin`, each (N, D).
    /// - Returns: the same shape as `x`.
    func callAsFunction(_ x: MLXArray, rope: RotaryCIS?) -> MLXArray {
        let inputShape = x.shape
        let (B, N, C) =
            x.ndim == 4 ? (x.dim(0), x.dim(1) * x.dim(2), x.dim(3)) : (x.dim(0), x.dim(1), x.dim(2))
        let x = x.ndim == 4 ? x.reshaped(B, N, C) : x

        var q = qProj(x).reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)

        // RoPE after the transpose to (B, H, N, D), matching HF.
        if useRope, let rope {
            (q, k) = applyRotaryEnc(q, k, cos: rope.cos, sin: rope.sin)
        }

        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped(B, N, C)
        out = oProj(out)

        if inputShape.count == 4 {
            out = out.reshaped(inputShape)
        }
        return out
    }
}

final class VitMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: ViTConfig) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

/// ViT transformer block with optional windowed attention, on spatial (B, H, W, C) tensors
/// (HF Sam3ViTLayer).
final class VitBlock: Module {
    let windowSize: Int
    let isGlobal: Bool

    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: VitAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VitMLP

    init(_ config: ViTConfig, isGlobal: Bool = false) {
        _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attention.wrappedValue = VitAttention(config)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = VitMLP(config)
        windowSize = isGlobal ? 0 : config.windowSize
        self.isGlobal = isGlobal
    }

    /// - Parameters:
    ///   - x: (B, H, W, C) spatial features.
    ///   - rope: `cos` and `sin`, each (N, D).
    func callAsFunction(_ x: MLXArray, rope: RotaryCIS?) -> MLXArray {
        var residual = x
        var x = layerNorm1(x)

        if windowSize > 0 {
            let (H, W) = (x.dim(1), x.dim(2))
            let padHW: (Int, Int)
            (x, padHW) = windowPartition(x, windowSize)
            x = attention(x, rope: rope)
            x = windowUnpartition(x, windowSize, padHW, (H, W))
        } else {
            x = attention(x, rope: rope)
        }

        x = residual + x
        residual = x
        x = layerNorm2(x)
        x = mlp(x)
        x = residual + x
        return x
    }
}

/// Partitions (B, H, W, C) features into non-overlapping windows.
///
/// - Returns: windows (B*nH*nW, ws, ws, C) and the padded size (Hp, Wp).
func windowPartition(_ x: MLXArray, _ windowSize: Int) -> (MLXArray, (Int, Int)) {
    let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let ws = windowSize
    var x = x

    let padH = (ws - H % ws) % ws
    let padW = (ws - W % ws) % ws
    if padH > 0 || padW > 0 {
        x = padded(x, widths: [.init((0, 0)), .init((0, padH)), .init((0, padW)), .init((0, 0))])
    }
    let (Hp, Wp) = (H + padH, W + padW)

    let (nH, nW) = (Hp / ws, Wp / ws)
    x = x.reshaped(B, nH, ws, nW, ws, C)
    x = x.transposed(0, 1, 3, 2, 4, 5)  // (B, nH, nW, ws, ws, C)
    x = x.reshaped(B * nH * nW, ws, ws, C)
    return (x, (Hp, Wp))
}

/// Reverses ``windowPartition(_:_:)``: (B*nH*nW, ws, ws, C) -> (B, H, W, C).
func windowUnpartition(_ x: MLXArray, _ windowSize: Int, _ padHW: (Int, Int), _ originalHW: (Int, Int))
    -> MLXArray
{
    let ws = windowSize
    let (Hp, Wp) = padHW
    let (H, W) = originalHW
    let (nH, nW) = (Hp / ws, Wp / ws)
    let (B, C) = (x.dim(0) / (nH * nW), x.dim(-1))

    var x = x.reshaped(B, nH, nW, ws, ws, C)
    x = x.transposed(0, 1, 3, 2, 4, 5)  // (B, nH, ws, nW, ws, C)
    x = x.reshaped(B, Hp, Wp, C)

    if Hp > H || Wp > W {
        x = x[0..., ..<H, ..<W, 0...]
    }
    return x
}

/// Intermediates of one ``ViTBackbone`` forward pass, for parity debugging.
struct ViTCapture {
    /// ``PatchEmbeddings`` output (B, N, C), before the position-embedding add.
    var patchEmbed: MLXArray?
    /// Block outputs (B, H, W, C), keyed by layer index.
    var blocks: [Int: MLXArray] = [:]
}

/// Vision Transformer backbone with windowed and global attention and 2D RoPE.
final class ViTBackbone: Module {
    let config: ViTConfig
    let featSize: Int

    @ModuleInfo(key: "embeddings") var embeddings: PatchEmbeddings
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [VitBlock]

    // Precomputed RoPE for window and global attention. `RotaryCIS` is not an MLXArray, so MLXNN
    // does not treat these as parameters (Python's `_rope_*` attributes).
    let ropeWindow: RotaryCIS
    let ropeGlobal: RotaryCIS

    init(_ config: ViTConfig) {
        self.config = config
        _embeddings.wrappedValue = PatchEmbeddings(config)

        featSize = config.imageSize / config.patchSize

        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        let globalSet = Set(config.globalAttnIndexes)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map {
            VitBlock(config, isGlobal: globalSet.contains($0))
        }

        let headDim = config.hiddenSize / config.numAttentionHeads
        ropeWindow = computeAxialCIS(
            dim: headDim, endX: config.windowSize, endY: config.windowSize, theta: config.ropeTheta)
        ropeGlobal = computeAxialCIS(
            dim: headDim, endX: featSize, endY: featSize, theta: config.ropeTheta,
            scale: Float(Double(config.windowSize) / Double(featSize)))
    }

    /// (B, H, W, C) image, any resolution divisible by the patch size -> (B, feat_H, feat_W, hidden_size)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var capture: ViTCapture?
        return forward(x, capture: &capture)
    }

    /// Forward pass that also returns the intermediates named by `blocks`. Debugging and parity only.
    func callAsFunction(_ x: MLXArray, capture blocks: Set<Int>) -> (MLXArray, ViTCapture) {
        var capture: ViTCapture? = ViTCapture()
        let out = forward(x, capture: &capture, captureBlocks: blocks)
        return (out, capture!)
    }

    private func forward(_ x: MLXArray, capture: inout ViTCapture?, captureBlocks: Set<Int> = [])
        -> MLXArray
    {
        let (B, inputH, inputW) = (x.dim(0), x.dim(1), x.dim(2))
        let (H, W) = (inputH / config.patchSize, inputW / config.patchSize)

        var x = embeddings(x)  // (B, N, C)
        capture?.patchEmbed = x

        // Tile position embeddings to the actual feature size.
        let pos = tilePosEmbed(embeddings.positionEmbeddings, H, W)
        x = x + pos

        // Spatial (B, H, W, C), matching HF.
        x = x.reshaped(B, H, W, -1)
        x = layerNorm(x)

        // RoPE for the actual size when it differs from the default.
        let ropeGlobal: RotaryCIS
        if H != featSize || W != featSize {
            let headDim = config.hiddenSize / config.numAttentionHeads
            ropeGlobal = computeAxialCIS(
                dim: headDim, endX: H, endY: W, theta: config.ropeTheta,
                scale: Float(Double(config.windowSize) / Double(H)))
        } else {
            ropeGlobal = self.ropeGlobal
        }

        for (i, layer) in layers.enumerated() {
            if layer.isGlobal {
                x = layer(x, rope: ropeGlobal)
            } else {
                x = layer(x, rope: ropeWindow)
            }
            if captureBlocks.contains(i) {
                capture?.blocks[i] = x
            }
        }

        return x
    }

    /// Tiles (repeats, not interpolates, as HF SAM3 does) position embeddings
    /// (1, pretrain_size², hidden_size) to the target spatial size.
    private func tilePosEmbed(_ pos: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
        let pretrainSize = Int(Double(pos.dim(1)).squareRoot())
        let hiddenSize = pos.dim(-1)

        if pretrainSize == targetH && pretrainSize == targetW {
            return pos
        }

        var pos = pos.reshaped(1, pretrainSize, pretrainSize, hiddenSize)

        // Tile to cover the target size.
        let repeatH = targetH / pretrainSize + 1
        let repeatW = targetW / pretrainSize + 1

        pos = tiled(pos, repetitions: [1, repeatH, repeatW, 1])
        // Crop to the target size.
        pos = pos[0..., ..<targetH, ..<targetW, 0...]
        pos = pos.reshaped(1, targetH * targetW, hiddenSize)
        return pos
    }
}
