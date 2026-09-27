// Port of mlx_vlm/models/sam3/segmentation.py::{PixelDecoder, MaskEmbedder, MaskDecoder,
// TextScoringMLP, DotProductScoring} (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXNN

/// FPN-like upsampling decoder: from the coarsest level, nearest-upsample, add the next finer level,
/// then conv + GroupNorm + ReLU.
final class PixelDecoder: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [Conv2d]
    @ModuleInfo(key: "norms") var norms: [GroupNorm]

    init(hiddenSize: Int, numUpsamplingStages: Int = 3) {
        _convLayers.wrappedValue = (0..<numUpsamplingStages).map { _ in
            Conv2d(
                inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: .init(3), padding: .init(1)
            )
        }
        // Python uses mlx's default GroupNorm (eps 1e-5, affine, not pytorch_compatible).
        _norms.wrappedValue = (0..<numUpsamplingStages).map { _ in
            GroupNorm(groupCount: 8, dimensions: hiddenSize)
        }
    }

    /// - Parameter features: `(B, H_i, W_i, D)` levels, finest first and coarsest last.
    /// - Returns: `(B, H_finest, W_finest, D)`.
    func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        var x = features[features.count - 1]
        for (i, backboneFeat) in features.dropLast().reversed().enumerated() {
            x = Self.interpolateNearest(x, targetH: backboneFeat.dim(1), targetW: backboneFeat.dim(2))
            x = x + backboneFeat
            x = convLayers[i](x)
            x = norms[i](x)
            x = relu(x)
        }
        return x
    }

    /// Port of `nn.Upsample(scale_factor=(th/H, tw/W), mode="nearest")`. Integer factors repeat each
    /// pixel (mlx's broadcast path); other factors use mlx's `_nearest_indices`.
    static func interpolateNearest(_ x: MLXArray, targetH: Int, targetW: Int) -> MLXArray {
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        if H == targetH && W == targetW { return x }
        if targetH % H == 0 && targetW % W == 0 {
            let (sh, sw) = (targetH / H, targetW / W)
            return broadcast(x.reshaped(B, H, 1, W, 1, C), to: [B, H, sh, W, sw, C])
                .reshaped(B, H * sh, W * sw, C)
        }
        func indices(_ n: Int, _ scale: Double) -> MLXArray {
            let m = Int(scale * Double(n))
            var idx = MLXArray.arange(m).asType(.float32)
            if m > n {
                idx = round((idx + 0.5) * Float(Double(n) / Double(m)) - 0.5)
            } else {
                idx = idx * Float(Double(n) / Double(m))
            }
            return idx.asType(.uint32)
        }
        let rows = indices(H, Double(targetH) / Double(H))
        let cols = indices(W, Double(targetW) / Double(W))
        return take(take(x, rows, axis: 1), cols, axis: 2)
    }
}

/// 3-layer MLP (ReLU between layers) projecting queries to the mask embedding space.
final class MaskEmbedder: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(hiddenSize: Int) {
        _layers.wrappedValue = (0..<3).map { _ in Linear(hiddenSize, hiddenSize) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for (i, layer) in layers.enumerated() {
            x = layer(x)
            if i < layers.count - 1 {
                x = relu(x)
            }
        }
        return x
    }
}

/// Output of ``MaskDecoder``.
struct MaskDecoderOutput {
    /// `(B, Q, H, W)` per-query mask logits at the finest FPN resolution.
    let predMasks: MLXArray
    /// `(B, 1, H, W)` semantic segmentation logits.
    let semanticSeg: MLXArray
}

/// Detector segmentation head. Weight keys: `detector_model.mask_decoder.*`.
final class MaskDecoder: Module {
    @ModuleInfo(key: "pixel_decoder") var pixelDecoder: PixelDecoder
    @ModuleInfo(key: "mask_embedder") var maskEmbedder: MaskEmbedder
    @ModuleInfo(key: "prompt_cross_attn") var promptCrossAttn: MultiheadAttention
    @ModuleInfo(key: "prompt_cross_attn_norm") var promptCrossAttnNorm: LayerNorm
    @ModuleInfo(key: "semantic_projection") var semanticProjection: Conv2d
    @ModuleInfo(key: "instance_projection") var instanceProjection: Conv2d

    init(_ config: DetectorMaskDecoderConfig) {
        let d = config.hiddenSize
        _pixelDecoder.wrappedValue = PixelDecoder(
            hiddenSize: d, numUpsamplingStages: config.numUpsamplingStages)
        _maskEmbedder.wrappedValue = MaskEmbedder(hiddenSize: d)
        _promptCrossAttn.wrappedValue = MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads)
        _promptCrossAttnNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _semanticProjection.wrappedValue = Conv2d(inputChannels: d, outputChannels: 1, kernelSize: .init(1))
        _instanceProjection.wrappedValue = Conv2d(inputChannels: d, outputChannels: d, kernelSize: .init(1))
    }

    /// - Parameters:
    ///   - objQueries: `(B, Q, D)` last decoder layer's queries.
    ///   - backboneFeatures: `(B, H_i, W_i, D)` detection FPN levels, finest first.
    ///   - encoderHiddenStates: `(B, HW, D)` DETR encoder output for the coarsest level.
    ///   - promptFeatures: `(B, P, D)` text features for the cross-attention.
    ///   - promptMask: `(B, P)` 1 = valid, 0 = pad (any dtype).
    func callAsFunction(
        _ objQueries: MLXArray, backboneFeatures: [MLXArray], encoderHiddenStates: MLXArray? = nil,
        promptFeatures: MLXArray? = nil, promptMask: MLXArray? = nil
    ) -> MaskDecoderOutput {
        var encoderHiddenStates = encoderHiddenStates

        // Cross-attention: encoder states attend to the prompt (pre-norm).
        if let promptFeatures, let states = encoderHiddenStates {
            let residual = states
            let normed = promptCrossAttnNorm(states)
            var crossMask: MLXArray? = nil
            if let promptMask {
                crossMask = (1 - promptMask[0..., .newAxis, .newAxis, 0...].asType(.float32)) * -1e9
            }
            let attnOut = promptCrossAttn(normed, key: promptFeatures, value: promptFeatures, mask: crossMask)
            encoderHiddenStates = residual + attnOut
        }

        // Replace the coarsest backbone level with the encoder output.
        var featsForFPN = backboneFeatures
        if let states = encoderHiddenStates {
            let finest = featsForFPN[featsForFPN.count - 1]
            let (B, H, W, D) = (finest.dim(0), finest.dim(1), finest.dim(2), finest.dim(3))
            featsForFPN[featsForFPN.count - 1] = states[0..., ..<(H * W), 0...].reshaped(B, H, W, D)
        }

        let pixelEmbed = pixelDecoder(featsForFPN)
        let instanceEmbed = instanceProjection(pixelEmbed)

        // einsum("bqc,bhwc->bqhw") as a matmul.
        let maskEmbeddings = maskEmbedder(objQueries)
        let (B, H, W, D) = (
            instanceEmbed.dim(0), instanceEmbed.dim(1), instanceEmbed.dim(2), instanceEmbed.dim(3)
        )
        let instanceFlat = instanceEmbed.reshaped(B, H * W, D)
        let predMasks = matmul(maskEmbeddings, instanceFlat.transposed(0, 2, 1)).reshaped(B, -1, H, W)

        let semanticSeg = semanticProjection(pixelEmbed).transposed(0, 3, 1, 2)

        return MaskDecoderOutput(predMasks: predMasks, semanticSeg: semanticSeg)
    }
}

/// 2-layer MLP for text scoring; the residual is added by the caller.
final class TextScoringMLP: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(hiddenSize: Int) {
        _layer1.wrappedValue = Linear(hiddenSize, hiddenSize * 8)
        _layer2.wrappedValue = Linear(hiddenSize * 8, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer2(relu(layer1(x)))
    }
}

/// Dot-product classifier between decoder queries and pooled text. Weight keys:
/// `detector_model.dot_product_scoring.*`.
final class DotProductScoring: Module {
    let scale: Float
    let clampMaxVal: Float = 12.0

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "text_proj") var textProj: Linear
    @ModuleInfo(key: "text_mlp") var textMLP: TextScoringMLP
    @ModuleInfo(key: "text_mlp_out_norm") var textMLPOutNorm: LayerNorm

    init(hiddenSize: Int) {
        scale = Float(1.0 / pow(Double(hiddenSize), 0.5))
        _queryProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _textProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _textMLP.wrappedValue = TextScoringMLP(hiddenSize: hiddenSize)
        _textMLPOutNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    /// - Parameters:
    ///   - hs: `(L, B, Q, D)` decoder hidden states for every layer.
    ///   - inputsEmbeds: `(B, T, D)` text features.
    ///   - textMask: `(B, T)` 1 = valid, 0 = pad (any dtype).
    /// - Returns: `(L, B, Q, 1)` logits clamped to ±12.
    func callAsFunction(_ hs: MLXArray, inputsEmbeds: MLXArray, textMask: MLXArray? = nil) -> MLXArray {
        let origText = inputsEmbeds
        var textProcessed = textMLP(inputsEmbeds) + origText
        textProcessed = textMLPOutNorm(textProcessed)

        let pooledText: MLXArray
        if let textMask {
            let isValid = textMask[.ellipsis, .newAxis].asType(.float32)
            let numValid = maximum(isValid.sum(axis: 1), 1.0)
            pooledText = (textProcessed * isValid).sum(axis: 1) / numValid
        } else {
            pooledText = textProcessed.mean(axis: 1)
        }

        let projText = textProj(pooledText)  // (B, D)
        let projQueries = queryProj(hs)  // (L, B, Q, D)

        var scores = matmul(projQueries, projText[.newAxis, 0..., 0..., .newAxis])
        scores = scores * scale
        return clip(scores, min: -clampMaxVal, max: clampMaxVal)
    }
}
