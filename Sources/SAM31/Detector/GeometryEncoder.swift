// Port of mlx_vlm/models/sam3/geometry.py::{_sine_encode_1d, _bilinear_sample_grid, roi_align,
// GeometryEncoderLayer, GeometryEncoder} and mlx_vlm/models/sam3_1/sam3_1.py::GeometryEncoder
// (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXNN

/// 1D sine/cosine position encoding matching HF `encode_1d_positions`.
/// `coords` is `(N,)`; returns `(N, numPosFeats)`.
func sineEncode1D(_ coords: MLXArray, numPosFeats: Int, temperature: Float = 10000) -> MLXArray {
    let scale = Float(2 * Double.pi)
    let embed = coords.asType(.float32) * scale
    var dimT = MLXArray.arange(numPosFeats).asType(.float32)
    dimT = pow(temperature, 2 * floorDivide(dimT, 2) / Float(numPosFeats))
    var pos = embed[0..., .newAxis] / dimT
    pos = stacked(
        [MLX.sin(pos[0..., .stride(by: 2)]), MLX.cos(pos[0..., .stride(from: 1, by: 2)])], axis: 2)
    return pos.reshaped(pos.dim(0), -1)
}

/// Bilinearly samples `(H, W, C)` features on the outer grid `(ys, xs)`, giving `(ys.count, xs.count, C)`.
///
/// Matches torchvision `roi_align` per-point interpolation (aligned=False): points outside
/// `[-1, H] x [-1, W]` contribute zero.
func bilinearSampleGrid(_ feat: MLXArray, ys: MLXArray, xs: MLXArray) -> MLXArray {
    let (H, W) = (feat.dim(0), feat.dim(1))

    let validY = logicalAnd(ys .>= -1.0, ys .<= Float(H))
    let validX = logicalAnd(xs .>= -1.0, xs .<= Float(W))

    let y = maximum(ys, 0.0)
    let x = maximum(xs, 0.0)

    let yf = floor(y).asType(.int32)
    let xf = floor(x).asType(.int32)

    let atEdgeY = yf .>= Int32(H - 1)
    let atEdgeX = xf .>= Int32(W - 1)

    let yLow = which(atEdgeY, MLXArray(Int32(H - 1)), yf)
    let xLow = which(atEdgeX, MLXArray(Int32(W - 1)), xf)
    let yHigh = which(atEdgeY, MLXArray(Int32(H - 1)), yf + 1)
    let xHigh = which(atEdgeX, MLXArray(Int32(W - 1)), xf + 1)

    let ly = which(atEdgeY, zeros(like: y), y - yf.asType(.float32))
    let lx = which(atEdgeX, zeros(like: x), x - xf.asType(.float32))
    let hy = 1.0 - ly
    let hx = 1.0 - lx

    let fLL = feat[yLow[0..., .newAxis], xLow[.newAxis, 0...]]
    let fLH = feat[yLow[0..., .newAxis], xHigh[.newAxis, 0...]]
    let fHL = feat[yHigh[0..., .newAxis], xLow[.newAxis, 0...]]
    let fHH = feat[yHigh[0..., .newAxis], xHigh[.newAxis, 0...]]

    let w1 = (hy[0..., .newAxis] * hx[.newAxis, 0...])[.ellipsis, .newAxis]
    let w2 = (hy[0..., .newAxis] * lx[.newAxis, 0...])[.ellipsis, .newAxis]
    let w3 = (ly[0..., .newAxis] * hx[.newAxis, 0...])[.ellipsis, .newAxis]
    let w4 = (ly[0..., .newAxis] * lx[.newAxis, 0...])[.ellipsis, .newAxis]

    let val = w1 * fLL + w2 * fLH + w3 * fHL + w4 * fHH
    let valid = logicalAnd(validY[0..., .newAxis], validX[.newAxis, 0...])[.ellipsis, .newAxis]
    return val * valid.asType(val.dtype)
}

/// torchvision-compatible ROI align (spatial_scale=1, aligned=False).
///
/// - Parameters:
///   - features: `(B, H, W, C)` channel-last feature maps.
///   - boxesXYXY: `(B, N, 4)` boxes in feature-map coordinates `(x1, y1, x2, y2)`.
///   - outputSize: pooled spatial size.
/// - Returns: `(B, N, outputSize, outputSize, C)`.
///
/// Like Python, the box coordinates are read back to the host and the per-box grid sizes are
/// computed in double precision.
func roiAlign(_ features: MLXArray, boxesXYXY: MLXArray, outputSize: Int) -> MLXArray {
    let (B, C) = (features.dim(0), features.dim(3))
    let numBoxes = boxesXYXY.dim(1)
    let (ph, pw) = (outputSize, outputSize)

    let boxes = boxesXYXY.asType(.float32).asArray(Float.self)

    var out: [MLXArray] = []
    for b in 0..<B {
        var row: [MLXArray] = []
        for i in 0..<numBoxes {
            let base = (b * numBoxes + i) * 4
            let (x1, y1, x2, y2) = (
                Double(boxes[base]), Double(boxes[base + 1]), Double(boxes[base + 2]), Double(boxes[base + 3])
            )
            let roiW = max(x2 - x1, 1.0)
            let roiH = max(y2 - y1, 1.0)
            let binW = roiW / Double(pw)
            let binH = roiH / Double(ph)
            let gridW = max(Int(ceil(binW)), 1)
            let gridH = max(Int(ceil(binH)), 1)

            let phArr = MLXArray.arange(ph).asType(.float32)
            let pwArr = MLXArray.arange(pw).asType(.float32)
            let iy = MLXArray.arange(gridH).asType(.float32)
            let ix = MLXArray.arange(gridW).asType(.float32)

            var ys =
                Float(y1) + phArr[0..., .newAxis] * Float(binH)
                + (iy[.newAxis, 0...] + 0.5) * Float(binH / Double(gridH))
            var xs =
                Float(x1) + pwArr[0..., .newAxis] * Float(binW)
                + (ix[.newAxis, 0...] + 0.5) * Float(binW / Double(gridW))
            ys = ys.reshaped(-1)
            xs = xs.reshaped(-1)

            var sampled = bilinearSampleGrid(features[b], ys: ys, xs: xs)
            sampled = sampled.reshaped(ph, gridH, pw, gridW, C)
            row.append(sampled.mean(axes: [1, 3]))
        }
        out.append(stacked(row, axis: 0))
    }
    return stacked(out, axis: 0)
}

/// Pre-norm layer: prompt self-attention, cross-attention to the image (pos on keys only), FFN.
final class GeometryEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: MultiheadAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiheadAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm

    init(_ config: GeometryEncoderConfig) {
        let d = config.hiddenSize
        _layerNorm1.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _selfAttn.wrappedValue = MultiheadAttention(
            hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout)
        _crossAttn.wrappedValue = MultiheadAttention(
            hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
        _mlp.wrappedValue = MLP(
            hiddenSize: d, intermediateSize: config.intermediateSize, act: config.hiddenAct)
        _layerNorm3.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)
    }

    func callAsFunction(
        _ promptFeats: MLXArray, visionFeats: MLXArray, visionPos: MLXArray, promptMask: MLXArray?
    ) -> MLXArray {
        var residual = promptFeats
        var hidden = layerNorm1(promptFeats)
        hidden = selfAttn(hidden, key: hidden, value: hidden, mask: promptMask)
        var promptFeats = residual + hidden

        residual = promptFeats
        hidden = layerNorm2(promptFeats)
        let key = visionFeats + visionPos
        hidden = crossAttn(hidden, key: key, value: visionFeats)
        promptFeats = residual + hidden

        residual = promptFeats
        hidden = layerNorm3(promptFeats)
        hidden = mlp(hidden)
        promptFeats = residual + hidden

        return promptFeats
    }
}

/// Encodes box prompts into prompt tokens. Weight keys: `detector_model.geometry_encoder.*`.
///
/// Includes the SAM 3.1 `points_*` projections, which mlx-vlm loads but never calls.
///
/// Parity: only the weight keys are checked (`DetectorParityTests.geometryEncoderKeysLoad`). The box
/// forward is ported line for line from `sam3/geometry.py` but has no fixture yet, because mlx-vlm's
/// text-only detect path never calls it.
final class GeometryEncoder: Module {
    let hiddenSize: Int
    let roiSize: Int
    let numPosFeats: Int

    @ModuleInfo(key: "label_embed") var labelEmbed: Embedding
    @ModuleInfo(key: "cls_embed") var clsEmbed: Embedding
    @ModuleInfo(key: "boxes_direct_project") var boxesDirectProject: Linear
    @ModuleInfo(key: "boxes_pool_project") var boxesPoolProject: Conv2d
    @ModuleInfo(key: "boxes_pos_enc_project") var boxesPosEncProject: Linear
    @ModuleInfo(key: "vision_layer_norm") var visionLayerNorm: LayerNorm
    @ModuleInfo(key: "final_proj") var finalProj: Linear
    @ModuleInfo(key: "prompt_layer_norm") var promptLayerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [GeometryEncoderLayer]
    @ModuleInfo(key: "output_layer_norm") var outputLayerNorm: LayerNorm
    // SAM 3.1 point prompt projections (unused in detection-only mode).
    @ModuleInfo(key: "points_direct_project") var pointsDirectProject: Linear
    @ModuleInfo(key: "points_pool_project") var pointsPoolProject: Linear
    @ModuleInfo(key: "points_pos_enc_project") var pointsPosEncProject: Linear

    init(_ config: GeometryEncoderConfig) {
        let d = config.hiddenSize
        hiddenSize = d
        roiSize = config.roiSize
        numPosFeats = d / 2

        _labelEmbed.wrappedValue = Embedding(embeddingCount: 2, dimensions: d)
        _clsEmbed.wrappedValue = Embedding(embeddingCount: 1, dimensions: d)

        _boxesDirectProject.wrappedValue = Linear(4, d)
        _boxesPoolProject.wrappedValue = Conv2d(
            inputChannels: d, outputChannels: d, kernelSize: .init(config.roiSize), bias: true)
        _boxesPosEncProject.wrappedValue = Linear(d + 2, d)

        _visionLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)

        _finalProj.wrappedValue = Linear(d, d)
        _promptLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)

        _layers.wrappedValue = (0..<config.numLayers).map { _ in GeometryEncoderLayer(config) }

        _outputLayerNorm.wrappedValue = LayerNorm(dimensions: d, eps: config.layerNormEps)

        _pointsDirectProject.wrappedValue = Linear(2, d)
        _pointsPoolProject.wrappedValue = Linear(d, d)
        _pointsPosEncProject.wrappedValue = Linear(d, d)
    }

    func encodeBoxCoordinates(cx: MLXArray, cy: MLXArray, w: MLXArray, h: MLXArray) -> MLXArray {
        let posX = sineEncode1D(cx, numPosFeats: numPosFeats)
        let posY = sineEncode1D(cy, numPosFeats: numPosFeats)
        return concatenated([posY, posX, h[0..., .newAxis], w[0..., .newAxis]], axis: 1)
    }

    func encodeBoxes(_ boxes: MLXArray, boxesLabels: MLXArray, visionFeatures: MLXArray) -> MLXArray {
        let (B, N) = (boxes.dim(0), boxes.dim(1))
        let (H, W) = (visionFeatures.dim(1), visionFeatures.dim(2))
        let dtype = visionFeatures.dtype
        let boxes = boxes.asType(dtype)

        var boxesEmbed = boxesDirectProject(boxes)

        let (cx, cy, bw, bh) = (
            boxes[.ellipsis, 0], boxes[.ellipsis, 1], boxes[.ellipsis, 2], boxes[.ellipsis, 3]
        )
        let (x1, y1) = (cx - 0.5 * bw, cy - 0.5 * bh)
        let (x2, y2) = (cx + 0.5 * bw, cy + 0.5 * bh)
        let boxesXYXY = stacked([x1 * W, y1 * H, x2 * W, y2 * H], axis: -1)
        var sampled = roiAlign(visionFeatures, boxesXYXY: boxesXYXY, outputSize: roiSize).asType(dtype)
        sampled = sampled.reshaped(B * N, roiSize, roiSize, hiddenSize)
        var pooled = boxesPoolProject(sampled)
        pooled = pooled.reshaped(B, N, hiddenSize)
        boxesEmbed = boxesEmbed + pooled

        var posEnc = encodeBoxCoordinates(
            cx: cx.reshaped(-1), cy: cy.reshaped(-1), w: bw.reshaped(-1), h: bh.reshaped(-1))
        posEnc = posEnc.reshaped(B, N, -1).asType(dtype)
        boxesEmbed = boxesEmbed + boxesPosEncProject(posEnc)

        let labelEmbed = labelEmbed(boxesLabels.asType(.int32))
        return labelEmbed + boxesEmbed
    }

    /// - Parameters:
    ///   - boxes: `(B, N, 4)` normalized cxcywh box prompts.
    ///   - boxLabels: `(B, N)` label ids (0 = negative, 1 = positive).
    ///   - boxMask: `(B, N)` validity mask (1 = valid).
    ///   - imgFeatMap: `(B, H, W, D)` image features (finest DETR level).
    ///   - imgPosMap: `(B, H, W, D)` image position encoding.
    /// - Returns: prompt tokens `(B, N+1, D)` and their mask `(B, N+1)`.
    func callAsFunction(
        boxes: MLXArray, boxLabels: MLXArray, boxMask: MLXArray, imgFeatMap: MLXArray, imgPosMap: MLXArray
    ) -> (promptFeats: MLXArray, promptMask: MLXArray) {
        let (B, H, W, D) = (imgFeatMap.dim(0), imgFeatMap.dim(1), imgFeatMap.dim(2), imgFeatMap.dim(3))
        let visionFeatsFlat = imgFeatMap.reshaped(B, H * W, D)
        let visionPosFlat = imgPosMap.reshaped(B, H * W, D)

        let normalizedImgFeats = visionLayerNorm(imgFeatMap)

        var promptEmbeds = encodeBoxes(boxes, boxesLabels: boxLabels, visionFeatures: normalizedImgFeats)

        let cls = broadcast(clsEmbed.weight[.newAxis], to: [B, 1, D])
        promptEmbeds = concatenated([promptEmbeds, cls], axis: 1)
        let clsMask = MLXArray.ones([B, 1], dtype: boxMask.dtype)
        let promptMask = concatenated([boxMask, clsMask], axis: 1)

        promptEmbeds = promptLayerNorm(finalProj(promptEmbeds))

        let valid = promptMask.asType(promptEmbeds.dtype)
        let selfMask = (1.0 - valid[0..., .newAxis, .newAxis, 0...]) * -1e9

        for layer in layers {
            promptEmbeds = layer(
                promptEmbeds, visionFeats: visionFeatsFlat, visionPos: visionPosFlat, promptMask: selfMask)
        }

        promptEmbeds = outputLayerNorm(promptEmbeds)
        return (promptEmbeds, promptMask)
    }
}
