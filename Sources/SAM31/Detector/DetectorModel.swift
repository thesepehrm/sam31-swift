// Port of mlx_vlm/models/sam3_1/sam3_1.py::DetectorModel and
// mlx_vlm/models/sam3_1/generate.py::{_get_det_features, _run_detr_encoder, _detect_with_backbone,
// _postprocess_mlx} (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXNN

/// Raw detector outputs for one prompt, in 1008-normalized model space.
struct DetectorRawOutput {
    /// `(B, Q)` class logits from the last decoder layer.
    let logits: MLXArray
    /// `(B, Q, 4)` boxes as normalized xyxy.
    let boxesXYXY: MLXArray
    /// `(B, Q, H, W)` mask logits at the finest FPN resolution (288×288).
    let masks: MLXArray
    /// `(B, 1)` presence logits from the last decoder layer.
    let presence: MLXArray
}

/// One detection after thresholding and NMS.
struct RawDetection {
    /// xyxy box in source-image pixels, clipped to `[0, max(width, height)]`.
    let box: SIMD4<Float>
    let score: Float
    /// `(h, w)` mask logits at model resolution; upsample with ``resizeMasks(_:to:)``.
    let maskLogits: MLXArray
}

/// SAM 3.1 detector: vision encoder, text encoder, DETR encoder/decoder, geometry encoder,
/// segmentation head and dot-product scoring. Weight keys: `detector_model.*`.
final class DetectorModel: Module {
    @ModuleInfo(key: "vision_encoder") var visionEncoder: VisionEncoder
    @ModuleInfo(key: "text_encoder") var textEncoder: TextEncoder
    /// Detector-level text projection (text hidden -> DETR hidden). Not `TextEncoder.text_projection`.
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "detr_encoder") var detrEncoder: DETREncoder
    @ModuleInfo(key: "detr_decoder") var detrDecoder: DETRDecoder
    @ModuleInfo(key: "geometry_encoder") var geometryEncoder: GeometryEncoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: MaskDecoder
    @ModuleInfo(key: "dot_product_scoring") var dotProductScoring: DotProductScoring

    /// Python's `_pos_enc`; parameterless, so not part of the weight tree.
    let posEnc: PositionEmbeddingSine

    init(_ config: DetectorConfig) {
        let hidden = config.detrEncoderConfig.hiddenSize
        _visionEncoder.wrappedValue = VisionEncoder(config.visionConfig)
        _textEncoder.wrappedValue = TextEncoder(config.textConfig)
        _textProjection.wrappedValue = Linear(config.textConfig.hiddenSize, hidden)
        _detrEncoder.wrappedValue = DETREncoder(config.detrEncoderConfig)
        _detrDecoder.wrappedValue = DETRDecoder(config.detrDecoderConfig)
        _geometryEncoder.wrappedValue = GeometryEncoder(config.geometryEncoderConfig)
        _maskDecoder.wrappedValue = MaskDecoder(config.maskDecoderConfig)
        _dotProductScoring.wrappedValue = DotProductScoring(hiddenSize: hidden)
        posEnc = PositionEmbeddingSine(numPosFeats: hidden / 2)
    }

    /// Python's `get_input_embeddings`: text encoder, then the detector's text projection.
    /// - Returns: `(B, T, D)` prompt features.
    func textEmbeddings(_ ids: MLXArray, _ mask: MLXArray) -> MLXArray {
        textProjection(textEncoder(ids, attentionMask: mask))
    }

    /// Runs the detection head on precomputed detection FPN features, as `_detect_with_backbone`
    /// does for one prompt (without its post-processing).
    ///
    /// - Parameters:
    ///   - detFeatures: `(B, H_i, W_i, D)` detection neck levels, finest first (288, 144, 72).
    ///   - textEmbeds: `(B, T, D)` from ``textEmbeddings(_:_:)``.
    ///   - textMask: `(B, T)` 1 = valid, 0 = pad (any dtype).
    func detect(detFeatures: [MLXArray], textEmbeds: MLXArray, textMask: MLXArray) -> DetectorRawOutput {
        // _get_det_features: flatten the 1x level and its position encoding for DETR.
        let encoderFeat = detFeatures[detFeatures.count - 1]
        let (B, hF, wF, D) = (encoderFeat.dim(0), encoderFeat.dim(1), encoderFeat.dim(2), encoderFeat.dim(3))
        let src = encoderFeat.reshaped(B, hF * wF, D)
        let posFlat = posEnc(encoderFeat).reshaped(B, hF * wF, D)

        let encoded = detrEncoder(src, pos: posFlat, prompt: textEmbeds, promptMask: textMask)

        let (hs, refBoxes, presenceLogits) = detrDecoder(
            visionFeatures: encoded, inputsEmbeds: textEmbeds, visionPosEncoding: posFlat,
            textMask: textMask, spatialShape: (hF, wF))

        // Box conversion cxcywh -> xyxy.
        let predBoxesCxcywh = refBoxes[-1]
        let cx = predBoxesCxcywh[.ellipsis, 0]
        let cy = predBoxesCxcywh[.ellipsis, 1]
        let w = predBoxesCxcywh[.ellipsis, 2]
        let h = predBoxesCxcywh[.ellipsis, 3]
        let predBoxesXYXY = stacked([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis: -1)

        let allLogits = dotProductScoring(hs, inputsEmbeds: textEmbeds, textMask: textMask)
        let predLogits = allLogits[-1].squeezed(axis: -1)
        let presence = presenceLogits[-1]

        let segOut = maskDecoder(
            hs[-1], backboneFeatures: detFeatures, encoderHiddenStates: encoded,
            promptFeatures: textEmbeds, promptMask: textMask)

        eval(predLogits, predBoxesXYXY, segOut.predMasks, presence)
        return DetectorRawOutput(
            logits: predLogits, boxesXYXY: predBoxesXYXY, masks: segOut.predMasks, presence: presence)
    }
}

/// Port of `_postprocess_mlx` followed by `nms`, for the first batch element.
///
/// Scores are `sigmoid(logits) * sigmoid(presence)` kept where `> threshold`; boxes are scaled by
/// `(W, H, W, H)` and clipped to `[0, max(W, H)]`. Masks stay as model-resolution logits: Python
/// resizes them to the image before NMS, but its NMS reads only boxes, so the kept set is the same.
///
/// - Returns: detections in descending score order.
func postprocessDetections(
    _ raw: DetectorRawOutput, imageSize: (width: Int, height: Int), threshold: Float, nmsIoU: Float = 0.5
) -> [RawDetection] {
    let (W, H) = (imageSize.width, imageSize.height)

    var scores = sigmoid(raw.logits[0].squeezed())
    scores = scores * sigmoid(raw.presence[0])

    var boxes =
        raw.boxesXYXY[0] * MLXArray([Float(W), Float(H), Float(W), Float(H)]).asType(raw.boxesXYXY.dtype)
    boxes = clip(boxes, min: Float(0), max: Float(max(H, W)))

    eval(scores, boxes)
    let scoresF = scores.asType(.float32).asArray(Float.self)
    let boxesF = boxes.asType(.float32).asArray(Float.self)

    let keep = scoresF.indices.filter { scoresF[$0] > threshold }
    let keptBoxes = keep.map {
        SIMD4<Float>(boxesF[4 * $0], boxesF[4 * $0 + 1], boxesF[4 * $0 + 2], boxesF[4 * $0 + 3])
    }
    let keptScores = keep.map { scoresF[$0] }

    return nmsIndices(boxes: keptBoxes, scores: keptScores, iouThreshold: nmsIoU).map { k in
        RawDetection(box: keptBoxes[k], score: keptScores[k], maskLogits: raw.masks[0, keep[k]])
    }
}
