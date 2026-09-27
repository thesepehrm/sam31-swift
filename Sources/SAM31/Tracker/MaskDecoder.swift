// Port of mlx_vlm/models/sam3_1/sam_components.py::MultiplexMaskDecoder (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXNN

/// Output of ``MultiplexMaskDecoder``, in the order of Python's return dict.
///
/// `M` is the multiplex count (1 for the interactive decoder) and `K` the number of selected masks.
struct MultiplexMaskDecoderOutput {
    /// `(B, M, K, H, W)` mask logits at 4x the image-embedding resolution.
    let masks: MLXArray
    /// `(B, M, K)` predicted mask IoU.
    let iouPred: MLXArray
    /// `(B, M, K', D)` mask-token outputs used for the object pointer.
    let samTokensOut: MLXArray
    /// `(B, M, 1)` object-existence logits.
    let objectScoreLogits: MLXArray
}

/// SAM mask decoder that decodes `multiplexCount` objects at once.
///
/// Token layout: `[obj_score(M), iou(M), mask(M * P), sparse...]`. With `multimaskOutputsOnly` there
/// is no single-mask token (the propagation decoder); otherwise `P = numMultimaskOutputs + 1`.
/// Weight keys: `tracker_model.sam_mask_decoder.*` / `tracker_model.interactive_sam_mask_decoder.*`.
final class MultiplexMaskDecoder: Module {
    let multiplexCount: Int
    let numMultimaskOutputs: Int
    let multimaskOutputsOnly: Bool
    let useMultimaskTokenForObjPtr: Bool
    let numMaskOutputPerObject: Int
    let numMaskTokens: Int
    let dynamicMultimaskViaStability: Bool
    let dynamicMultimaskStabilityDelta: Float
    let dynamicMultimaskStabilityThresh: Float

    @ModuleInfo(key: "iou_token") var iouToken: Embedding
    @ModuleInfo(key: "mask_tokens") var maskTokens: Embedding
    @ModuleInfo(key: "obj_score_token") var objScoreToken: Embedding
    @ModuleInfo(key: "transformer") var transformer: TwoWayTransformer
    @ModuleInfo(key: "output_hypernetworks_mlps") var outputHypernetworksMlps: [OutputMLP]
    @ModuleInfo(key: "iou_prediction_head") var iouPredictionHead: OutputMLP
    @ModuleInfo(key: "pred_obj_score_head") var predObjScoreHead: OutputMLP
    @ModuleInfo(key: "upscale_conv1") var upscaleConv1: ConvTransposed2d
    @ModuleInfo(key: "upscale_conv2") var upscaleConv2: ConvTransposed2d
    @ModuleInfo(key: "upscale_layer_norm") var upscaleLayerNorm: LayerNorm2d
    /// 1x1 projections of the high-res FPN levels, applied by the caller before decoding.
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d

    init(_ config: TrackerMaskDecoderConfig) {
        let d = config.hiddenSize
        multiplexCount = config.multiplexCount
        numMultimaskOutputs = config.numMultimaskOutputs
        multimaskOutputsOnly = config.multimaskOutputsOnly
        useMultimaskTokenForObjPtr = config.useMultimaskTokenForObjPtr
        // +1 for the single (best) mask token unless multimask-only.
        numMaskOutputPerObject = multimaskOutputsOnly ? numMultimaskOutputs : numMultimaskOutputs + 1
        numMaskTokens = multiplexCount * numMaskOutputPerObject
        dynamicMultimaskViaStability = config.dynamicMultimaskViaStability
        dynamicMultimaskStabilityDelta = config.dynamicMultimaskStabilityDelta
        dynamicMultimaskStabilityThresh = config.dynamicMultimaskStabilityThresh

        _iouToken.wrappedValue = Embedding(embeddingCount: multiplexCount, dimensions: d)
        _maskTokens.wrappedValue = Embedding(embeddingCount: numMaskTokens, dimensions: d)
        _objScoreToken.wrappedValue = Embedding(embeddingCount: multiplexCount, dimensions: d)
        _transformer.wrappedValue = TwoWayTransformer(
            hiddenSize: d, numHeads: config.numAttentionHeads, numLayers: config.numHiddenLayers,
            mlpDim: config.mlpDim, attentionDownsampleRate: config.attentionDownsampleRate)
        // One hypernetwork MLP per mask token of an object, shared across slots.
        _outputHypernetworksMlps.wrappedValue = (0..<numMaskOutputPerObject).map { _ in
            OutputMLP(inputDim: d, hiddenDim: d, outputDim: d / 8)
        }
        _iouPredictionHead.wrappedValue = OutputMLP(
            inputDim: d, hiddenDim: d, outputDim: numMaskOutputPerObject)
        _predObjScoreHead.wrappedValue = OutputMLP(inputDim: d, hiddenDim: d, outputDim: 1)
        _upscaleConv1.wrappedValue = ConvTransposed2d(
            inputChannels: d, outputChannels: d / 4, kernelSize: .init(2), stride: .init(2))
        _upscaleConv2.wrappedValue = ConvTransposed2d(
            inputChannels: d / 4, outputChannels: d / 8, kernelSize: .init(2), stride: .init(2))
        _upscaleLayerNorm.wrappedValue = LayerNorm2d(d / 4)
        _convS0.wrappedValue = Conv2d(
            inputChannels: d, outputChannels: d / 8, kernelSize: .init(1), bias: true)
        _convS1.wrappedValue = Conv2d(
            inputChannels: d, outputChannels: d / 4, kernelSize: .init(1), bias: true)
    }

    /// - Parameters:
    ///   - imageEmbeddings: `(B, HW, D)` image features.
    ///   - imagePE: `(1, HW, D)` dense positional encoding.
    ///   - multimaskOutput: return the multimask tokens, else the single/best token.
    ///   - highResFeatures: `[feat_s0, feat_s1]`, already projected by `convS0`/`convS1`.
    ///   - extraPerObjectEmbeddings: `(B, M, D)` added to every mask token.
    ///   - sparsePromptEmbeddings: `(B, N, D)` interactive prompts.
    ///   - densePromptEmbeddings: `(B, HW, D)` interactive prompts.
    func callAsFunction(
        imageEmbeddings: MLXArray,
        imagePE: MLXArray,
        multimaskOutput: Bool,
        highResFeatures: [MLXArray]? = nil,
        extraPerObjectEmbeddings: MLXArray? = nil,
        sparsePromptEmbeddings: MLXArray? = nil,
        densePromptEmbeddings: MLXArray? = nil
    ) -> MultiplexMaskDecoderOutput {
        if multimaskOutputsOnly {
            precondition(multimaskOutput, "multimask_output must be True with multimask_outputs_only")
        }

        let out = predictMasks(
            imageEmbeddings: imageEmbeddings, imagePE: imagePE, highResFeatures: highResFeatures,
            extraPerObjectEmbeddings: extraPerObjectEmbeddings,
            sparsePromptEmbeddings: sparsePromptEmbeddings, densePromptEmbeddings: densePromptEmbeddings)

        var masks = out.masks  // (B, M, P, H, W)
        var iouPred = out.iouPred  // (B, M, P)
        let maskTokensOut = out.maskTokensOut  // (B, M, P, D)

        if multimaskOutput {
            if !multimaskOutputsOnly {
                // Drop the single-mask token, keep the multimask tokens.
                masks = masks[0..., 0..., 1...]
                iouPred = iouPred[0..., 0..., 1...]
            }
        } else if dynamicMultimaskViaStability {
            (masks, iouPred) = dynamicMultimaskViaStability(masks, iouPred)
        } else {
            masks = masks[0..., 0..., 0..<1]
            iouPred = iouPred[0..., 0..., 0..<1]
        }

        let samTokensOut: MLXArray
        if multimaskOutput && useMultimaskTokenForObjPtr {
            samTokensOut = multimaskOutputsOnly ? maskTokensOut : maskTokensOut[0..., 0..., 1...]
        } else {
            // The object pointer always comes from the single-mask token.
            samTokensOut = maskTokensOut[0..., 0..., 0..<1]
        }

        return MultiplexMaskDecoderOutput(
            masks: masks, iouPred: iouPred, samTokensOut: samTokensOut,
            objectScoreLogits: out.objectScoreLogits)
    }

    private struct PredictedMasks {
        let masks: MLXArray
        let iouPred: MLXArray
        let maskTokensOut: MLXArray
        let objectScoreLogits: MLXArray
    }

    private func predictMasks(
        imageEmbeddings: MLXArray, imagePE: MLXArray, highResFeatures: [MLXArray]?,
        extraPerObjectEmbeddings: MLXArray?, sparsePromptEmbeddings: MLXArray?,
        densePromptEmbeddings: MLXArray?
    ) -> PredictedMasks {
        let (bImg, hw, d) = (imageEmbeddings.dim(0), imageEmbeddings.dim(1), imageEmbeddings.dim(2))
        let m = multiplexCount
        let p = numMaskOutputPerObject
        let b = sparsePromptEmbeddings?.dim(0) ?? bImg

        // Repeat the image embeddings to the token batch size if needed.
        var src: MLXArray
        if bImg != b {
            precondition(bImg == 1)
            src = broadcast(imageEmbeddings, to: [b, hw, d])
        } else {
            src = imageEmbeddings
        }
        if let densePromptEmbeddings {
            src = src + densePromptEmbeddings
        }

        var tokens = [
            broadcast(objScoreToken.weight[.newAxis], to: [b, m, d]),
            broadcast(iouToken.weight[.newAxis], to: [b, m, d]),
        ]
        var maskTok = maskTokens.weight.reshaped(1, m, p, d)
        if let extraPerObjectEmbeddings {
            maskTok = maskTok + extraPerObjectEmbeddings[0..., 0..., .newAxis, 0...]
        } else {
            maskTok = broadcast(maskTok, to: [b, m, p, d])
        }
        tokens.append(maskTok.reshaped(b, m * p, d))
        if let sparsePromptEmbeddings {
            tokens.append(sparsePromptEmbeddings)
        }
        let allTokens = concatenated(tokens, axis: 1)

        let pe = broadcast(imagePE, to: [b, hw, d])
        let (hs, srcOut) = transformer(src, pe, allTokens)

        let objScoreTokenOut = hs[0..., ..<m]
        let iouTokenOut = hs[0..., m..<(2 * m)]
        var maskTokensOut = hs[0..., (2 * m)..<(2 * m + m * p)]

        // Upscale image features (72 -> 144 -> 288) with high-res skip fusion.
        let h = Int(Double(hw).squareRoot())
        let w = h
        src = srcOut.reshaped(b, h, w, d)

        var upscaled = upscaleConv1(src)
        if let highResFeatures {
            upscaled = upscaled + highResFeatures[1]
        }
        upscaled = upscaleLayerNorm(upscaled)
        upscaled = gelu(upscaled)

        upscaled = upscaleConv2(upscaled)
        if let highResFeatures {
            upscaled = upscaled + highResFeatures[0]
        }
        upscaled = gelu(upscaled)

        let (hUp, wUp, cUp) = (upscaled.dim(1), upscaled.dim(2), upscaled.dim(3))
        let upscaledFlat = upscaled.reshaped(b, hUp * wUp, cUp)

        // Hypernetwork projections of the mask tokens: (B, M, P, C_up).
        maskTokensOut = maskTokensOut.reshaped(b, m, p, d)
        let hyperIn = stacked(
            (0..<p).map { i in outputHypernetworksMlps[i](maskTokensOut[0..., 0..., i]) }, axis: 2)

        // (B, M*P, C) @ (B, C, HW) -> (B, M, P, H, W)
        let masks = matmul(hyperIn.reshaped(b, m * p, cUp), upscaledFlat.transposed(0, 2, 1))
            .reshaped(b, m, p, hUp, wUp)

        let iouPred = iouPredictionHead(iouTokenOut)  // (B, M, P)
        let objectScoreLogits = predObjScoreHead(objScoreTokenOut)  // (B, M, 1)

        return PredictedMasks(
            masks: masks, iouPred: iouPred, maskTokensOut: maskTokensOut, objectScoreLogits: objectScoreLogits
        )
    }

    /// IoU between the upper- and lower-thresholded masks, per mask.
    private func stabilityScores(_ maskLogits: MLXArray) -> MLXArray {
        let flat = maskLogits.reshaped(Array(maskLogits.shape.dropLast(2)) + [-1])
        let delta = dynamicMultimaskStabilityDelta
        let areaI = (flat .> delta).sum(axis: -1).asType(.float32)
        let areaU = (flat .> -delta).sum(axis: -1).asType(.float32)
        return MLX.where(areaU .> 0, areaI / areaU, MLXArray(Float(1.0)))
    }

    /// Falls back to the best multimask output when the single-mask output is unstable.
    private func dynamicMultimaskViaStability(_ allMaskLogits: MLXArray, _ allIouScores: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let (b, m) = (allMaskLogits.dim(0), allMaskLogits.dim(1))
        let logits = allMaskLogits.reshaped([-1] + Array(allMaskLogits.shape.dropFirst(2)))
        let ious = allIouScores.reshaped(-1, allIouScores.dim(-1))

        // Best mask among the multimask output tokens (1..<P).
        let multimaskLogits = logits[0..., 1...]
        let multimaskIou = ious[0..., 1...]
        let bestInds = argMax(multimaskIou, axis: -1)  // (B*M,)
        let bestMultimaskLogits = takeAlong(
            multimaskLogits, bestInds[0..., .newAxis, .newAxis, .newAxis], axis: 1)
        let bestMultimaskIou = takeAlong(multimaskIou, bestInds[0..., .newAxis], axis: 1)

        // Single-mask output token 0 and its stability score.
        let singlemaskLogits = logits[0..., 0..<1]
        let singlemaskIou = ious[0..., 0..<1]
        let stability = stabilityScores(singlemaskLogits)[0..., 0..., .newAxis, .newAxis]
        let isStable = stability .>= dynamicMultimaskStabilityThresh

        let maskLogitsOut = MLX.where(isStable, singlemaskLogits, bestMultimaskLogits)
        let iouScoresOut = MLX.where(isStable[0..., 0..., 0, 0], singlemaskIou, bestMultimaskIou)

        return (
            maskLogitsOut.reshaped([b, m] + Array(maskLogitsOut.shape.dropFirst())),
            iouScoresOut.reshaped(b, m, -1)
        )
    }
}
