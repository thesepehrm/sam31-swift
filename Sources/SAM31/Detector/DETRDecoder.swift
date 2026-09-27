// Port of mlx_vlm/models/sam3/decoder.py::{DETRDecoderLayer, BoxHead, PresenceHead, RefPointHead,
// BoxRPBEmbed, SinePositionEmbeddingForBoxes, DETRDecoder, inverse_sigmoid} (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXFast
import MLXNN

/// Post-norm decoder layer: self-attention, text cross-attention, vision cross-attention (with the
/// box RPB bias as an additive mask), MLP. Each stage adds a residual and then applies LayerNorm.
final class DETRDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MultiheadAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "text_cross_attn") var textCrossAttn: MultiheadAttention
    @ModuleInfo(key: "text_cross_attn_layer_norm") var textCrossAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "vision_cross_attn") var visionCrossAttn: MultiheadAttention
    @ModuleInfo(key: "vision_cross_attn_layer_norm") var visionCrossAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "mlp_layer_norm") var mlpLayerNorm: LayerNorm

    init(_ config: DETRDecoderConfig) {
        let d = config.hiddenSize
        let heads = config.numAttentionHeads
        _selfAttn.wrappedValue = MultiheadAttention(hiddenSize: d, numHeads: heads)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _textCrossAttn.wrappedValue = MultiheadAttention(hiddenSize: d, numHeads: heads)
        _textCrossAttnLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _visionCrossAttn.wrappedValue = MultiheadAttention(hiddenSize: d, numHeads: heads)
        _visionCrossAttnLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _mlp.wrappedValue = MLP(
            hiddenSize: d, intermediateSize: config.intermediateSize, act: config.hiddenAct)
        _mlpLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
    }

    /// - Parameters:
    ///   - hiddenStates: `(B, Q+1, D)` presence token at index 0, then the queries.
    ///   - queryPos: `(B, Q+1, D)` query position (zero for the presence token).
    ///   - inputsEmbeds: `(B, T, D)` text features.
    ///   - visionFeatures: `(B, HW, D)`.
    ///   - visionPosEncoding: `(B, HW, D)`.
    ///   - textCrossAttnMask: `(B, 1, 1, T)` additive text mask.
    ///   - visionCrossAttnMask: `(B, H, Q+1, HW)` additive RPB bias.
    func callAsFunction(
        _ hiddenStates: MLXArray, queryPos: MLXArray, inputsEmbeds: MLXArray, visionFeatures: MLXArray,
        visionPosEncoding: MLXArray, textCrossAttnMask: MLXArray?, visionCrossAttnMask: MLXArray?
    ) -> MLXArray {
        // 1. Self-attention
        var residual = hiddenStates
        let qk = hiddenStates + queryPos
        var attnOut = selfAttn(qk, key: qk, value: hiddenStates)
        var hiddenStates = selfAttnLayerNorm(residual + attnOut)

        // 2. Text cross-attention
        residual = hiddenStates
        var qWithPos = hiddenStates + queryPos
        attnOut = textCrossAttn(qWithPos, key: inputsEmbeds, value: inputsEmbeds, mask: textCrossAttnMask)
        hiddenStates = textCrossAttnLayerNorm(residual + attnOut)

        // 3. Vision cross-attention (with RPB bias)
        residual = hiddenStates
        qWithPos = hiddenStates + queryPos
        let kWithPos = visionFeatures + visionPosEncoding
        attnOut = visionCrossAttn(qWithPos, key: kWithPos, value: visionFeatures, mask: visionCrossAttnMask)
        hiddenStates = visionCrossAttnLayerNorm(residual + attnOut)

        // 4. MLP
        residual = hiddenStates
        let mlpOut = mlp(hiddenStates)
        hiddenStates = mlpLayerNorm(residual + mlpOut)

        return hiddenStates
    }
}

/// 3-layer MLP for box regression.
final class BoxHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear

    init(hiddenSize: Int) {
        _layer1.wrappedValue = Linear(hiddenSize, hiddenSize)
        _layer2.wrappedValue = Linear(hiddenSize, hiddenSize)
        _layer3.wrappedValue = Linear(hiddenSize, 4)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = relu(layer1(x))
        x = relu(layer2(x))
        return layer3(x)
    }
}

/// 3-layer MLP for the presence score.
final class PresenceHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear

    init(hiddenSize: Int) {
        _layer1.wrappedValue = Linear(hiddenSize, hiddenSize)
        _layer2.wrappedValue = Linear(hiddenSize, hiddenSize)
        _layer3.wrappedValue = Linear(hiddenSize, 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = relu(layer1(x))
        x = relu(layer2(x))
        return layer3(x)
    }
}

/// Maps sine-encoded reference boxes (`2 * hiddenSize` features) to the query position encoding.
final class RefPointHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(hiddenSize: Int) {
        _layer1.wrappedValue = Linear(hiddenSize * 2, hiddenSize)
        _layer2.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer2(relu(layer1(x)))
    }
}

/// Box relative position bias: embeds 2 log-scaled deltas into one bias per head.
final class BoxRPBEmbed: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(numHeads: Int, hiddenSize: Int) {
        _layer1.wrappedValue = Linear(2, hiddenSize)
        _layer2.wrappedValue = Linear(hiddenSize, numHeads)
    }

    func callAsFunction(_ deltas: MLXArray) -> MLXArray {
        layer2(relu(layer1(deltas)))
    }
}

/// Sine position encoding for `(cx, cy, w, h)` boxes. Not a Module: it has no parameters.
struct SinePositionEmbeddingForBoxes {
    var numPosFeats = 128
    var temperature: Float = 10000.0
    var scale: Float = 2 * Float.pi

    /// - Parameter boxes: `(B, Q, 4)` in cxcywh.
    /// - Returns: `(B, Q, 4 * numPosFeats)`, coordinate order y, x, w, h; sin/cos interleaved.
    func encodeBoxes(_ boxes: MLXArray) -> MLXArray {
        var dimT = MLXArray.arange(numPosFeats).asType(.float32)
        dimT = pow(temperature, 2 * floorDivide(dimT, 2) / Float(numPosFeats))

        let x = boxes[.ellipsis, 0..<1] * scale
        let y = boxes[.ellipsis, 1..<2] * scale
        let w = boxes[.ellipsis, 2..<3] * scale
        let h = boxes[.ellipsis, 3..<4] * scale

        var encodings: [MLXArray] = []
        for coord in [y, x, w, h] {
            let pos = coord / dimT
            let sinEnc = MLX.sin(pos[.ellipsis, .stride(by: 2)])
            let cosEnc = MLX.cos(pos[.ellipsis, .stride(from: 1, by: 2)])
            let enc = stacked([sinEnc, cosEnc], axis: -1)
            encodings.append(enc.reshaped(Array(enc.shape.dropLast(2)) + [-1]))
        }
        return concatenated(encodings, axis: -1)
    }
}

/// `log(x / (1 - x))` after clipping `x` to `[eps, 1 - eps]`.
func inverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let x = clip(x, min: eps, max: 1 - eps)
    return log(x / (1 - x))
}

/// DETR transformer decoder. Weight keys: `detector_model.detr_decoder.*`.
final class DETRDecoder: Module {
    let numQueries: Int
    let hiddenSize: Int
    let clampPresenceLogitMaxVal: Float = 10.0
    let posEnc: SinePositionEmbeddingForBoxes

    @ModuleInfo(key: "layers") var layers: [DETRDecoderLayer]
    @ModuleInfo(key: "output_layer_norm") var outputLayerNorm: LayerNorm
    @ModuleInfo(key: "query_embed") var queryEmbed: Embedding
    @ModuleInfo(key: "reference_points") var referencePoints: Embedding
    @ModuleInfo(key: "presence_token") var presenceToken: Embedding
    @ModuleInfo(key: "presence_head") var presenceHead: PresenceHead
    @ModuleInfo(key: "presence_layer_norm") var presenceLayerNorm: LayerNorm
    @ModuleInfo(key: "box_head") var boxHead: BoxHead
    @ModuleInfo(key: "ref_point_head") var refPointHead: RefPointHead
    @ModuleInfo(key: "box_rpb_embed_x") var boxRpbEmbedX: BoxRPBEmbed
    @ModuleInfo(key: "box_rpb_embed_y") var boxRpbEmbedY: BoxRPBEmbed

    init(_ config: DETRDecoderConfig) {
        let d = config.hiddenSize
        numQueries = config.numQueries
        hiddenSize = d
        posEnc = SinePositionEmbeddingForBoxes(numPosFeats: d / 2)
        _layers.wrappedValue = (0..<config.numLayers).map { _ in DETRDecoderLayer(config) }
        _outputLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _queryEmbed.wrappedValue = Embedding(embeddingCount: config.numQueries, dimensions: d)
        _referencePoints.wrappedValue = Embedding(embeddingCount: config.numQueries, dimensions: 4)
        _presenceToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: d)
        _presenceHead.wrappedValue = PresenceHead(hiddenSize: d)
        _presenceLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _boxHead.wrappedValue = BoxHead(hiddenSize: d)
        _refPointHead.wrappedValue = RefPointHead(hiddenSize: d)
        _boxRpbEmbedX.wrappedValue = BoxRPBEmbed(numHeads: config.numAttentionHeads, hiddenSize: d)
        _boxRpbEmbedY.wrappedValue = BoxRPBEmbed(numHeads: config.numAttentionHeads, hiddenSize: d)
    }

    /// - Parameters:
    ///   - visionFeatures: `(B, HW, D)` DETR encoder output.
    ///   - inputsEmbeds: `(B, T, D)` text features.
    ///   - visionPosEncoding: `(B, HW, D)`.
    ///   - textMask: `(B, T)` 1 = valid, 0 = pad (any dtype).
    ///   - spatialShape: `(height, width)` of the vision grid; `nil` disables the RPB bias.
    /// - Returns: `hs (L, B, Q, D)` normed per layer, `refBoxes (L, B, Q, 4)` refined cxcywh boxes,
    ///   `presenceLogits (L, B, 1)` clamped to ±10.
    func callAsFunction(
        visionFeatures: MLXArray, inputsEmbeds: MLXArray, visionPosEncoding: MLXArray, textMask: MLXArray?,
        spatialShape: (Int, Int)?
    ) -> (hs: MLXArray, refBoxes: MLXArray, presenceLogits: MLXArray) {
        let (B, numQ, d) = (visionFeatures.dim(0), numQueries, hiddenSize)

        let queryEmbeds = broadcast(queryEmbed.weight[.newAxis], to: [B, numQ, d])
        var referenceBoxes = sigmoid(broadcast(referencePoints.weight[.newAxis], to: [B, numQ, 4]))
        let presence = broadcast(presenceToken.weight[.newAxis], to: [B, 1, d])

        // Presence token at position 0
        var hiddenStates = concatenated([presence, queryEmbeds], axis: 1)  // (B, Q+1, D)

        var textCrossMask: MLXArray? = nil
        if let textMask {
            textCrossMask = (1 - textMask[0..., .newAxis, .newAxis, 0...].asType(.float32)) * -1e9
        }

        var intermediateHs: [MLXArray] = []
        var intermediateBoxes: [MLXArray] = []
        var intermediatePresence: [MLXArray] = []

        for layer in layers {
            let querySineEmbed = posEnc.encodeBoxes(referenceBoxes)  // (B, Q, 2D)
            let queryPos = refPointHead(querySineEmbed)  // (B, Q, D)
            let queryPosPadded = concatenated([MLXArray.zeros([B, 1, d]), queryPos], axis: 1)

            var visionCrossMask: MLXArray? = nil
            if let spatialShape {
                let rpb = computeRPB(referenceBoxes, spatialShape: spatialShape)  // (B, H, Q, HW)
                visionCrossMask = concatenated(
                    [MLXArray.zeros([B, rpb.dim(1), 1, rpb.dim(3)]), rpb], axis: 2)
            }

            hiddenStates = layer(
                hiddenStates, queryPos: queryPosPadded, inputsEmbeds: inputsEmbeds,
                visionFeatures: visionFeatures, visionPosEncoding: visionPosEncoding,
                textCrossAttnMask: textCrossMask, visionCrossAttnMask: visionCrossMask)

            // Query hidden states skip the presence token at index 0.
            let queryHs = hiddenStates[0..., 1...]
            let queryHsNormed = outputLayerNorm(queryHs)

            // Box refinement
            let delta = boxHead(queryHsNormed)
            let newRef = sigmoid(inverseSigmoid(referenceBoxes) + delta)
            referenceBoxes = newRef

            intermediateHs.append(queryHsNormed)
            intermediateBoxes.append(newRef)

            // Presence logit from the presence token.
            let presHidden = hiddenStates[0..., ..<1]
            var presLogit = presenceHead(presenceLayerNorm(presHidden)).squeezed(axis: -1)
            presLogit = clip(presLogit, min: -clampPresenceLogitMaxVal, max: clampPresenceLogitMaxVal)
            intermediatePresence.append(presLogit)

            eval(hiddenStates, referenceBoxes)
        }

        return (stacked(intermediateHs), stacked(intermediateBoxes), stacked(intermediatePresence))
    }

    /// Box relative position bias.
    /// - Parameters:
    ///   - referenceBoxes: `(B, Q, 4)` cxcywh in sigmoid space.
    ///   - spatialShape: `(height, width)`.
    /// - Returns: `(B, numHeads, Q, HW)`.
    func computeRPB(_ referenceBoxes: MLXArray, spatialShape: (Int, Int)) -> MLXArray {
        let (height, width) = spatialShape
        let (B, Q) = (referenceBoxes.dim(0), referenceBoxes.dim(1))

        // cxcywh -> xyxy
        let cx = referenceBoxes[.ellipsis, 0]
        let cy = referenceBoxes[.ellipsis, 1]
        let w = referenceBoxes[.ellipsis, 2]
        let h = referenceBoxes[.ellipsis, 3]
        let x1 = cx - w / 2
        let y1 = cy - h / 2
        let x2 = cx + w / 2
        let y2 = cy + h / 2
        let boxesXyxy = stacked([x1, y1, x2, y2], axis: -1)  // (B, Q, 4)

        let coordsH = MLXArray.arange(height).asType(.float32) / Float(height)
        let coordsW = MLXArray.arange(width).asType(.float32) / Float(width)

        let yBounds = boxesXyxy[.ellipsis, .stride(from: 1, by: 2)].reshaped(B * Q, 1, 2)  // [y1, y2]
        var deltasY = coordsH.reshaped(1, -1, 1) - yBounds  // (B*Q, H, 2)
        deltasY = deltasY.reshaped(B, Q, height, 2)

        let xBounds = boxesXyxy[.ellipsis, .stride(by: 2)].reshaped(B * Q, 1, 2)  // [x1, x2]
        var deltasX = coordsW.reshaped(1, -1, 1) - xBounds  // (B*Q, W, 2)
        deltasX = deltasX.reshaped(B, Q, width, 2)

        // Log-scale encoding
        let log2of8 = Float(Foundation.log2(8.0))
        var deltasXLog = deltasX * 8
        deltasXLog = sign(deltasXLog) * log2(abs(deltasXLog) + 1.0) / log2of8
        var deltasYLog = deltasY * 8
        deltasYLog = sign(deltasYLog) * log2(abs(deltasYLog) + 1.0) / log2of8

        let rpbX = boxRpbEmbedX(deltasXLog)  // (B, Q, W, heads)
        let rpbY = boxRpbEmbedY(deltasYLog)  // (B, Q, H, heads)

        // (B, Q, H, W, heads) -> (B, Q, HW, heads) -> (B, heads, Q, HW)
        var rpb = rpbY[0..., 0..., 0..., .newAxis, 0...] + rpbX[0..., 0..., .newAxis, 0..., 0...]
        rpb = rpb.reshaped(B, Q, height * width, -1)
        return rpb.transposed(0, 3, 1, 2)
    }
}
