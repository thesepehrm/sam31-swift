import MLX
import Testing

@testable import SAM31

@Suite(.enabled(if: Fixtures.available), .serialized) struct DetectorParityTests {
    static let cfg = try! ModelConfig.load(from: Fixtures.weightsURL).detectorConfig

    /// src/pos exactly as mlx-vlm's _get_det_features builds them from the 1x det level.
    static func srcPos() throws -> (MLXArray, MLXArray) {
        let fx = try Fixtures.load("vision.safetensors")
        let f = fx["neck.det.2"]!
        let p = fx["neck.det_pos.2"]!
        return (f.reshaped(1, -1, f.dim(3)), p.reshaped(1, -1, p.dim(3)))
    }

    @Test func detrEncoderMatchesPython() throws {
        let enc = DETREncoder(Self.cfg.detrEncoderConfig)
        try loadWeights(
            into: enc, weights: SharedWeights.all, dtype: .float32,
            prefix: "detector_model.detr_encoder.")
        let (src, pos) = try Self.srcPos()
        let tx = try Fixtures.load("text.safetensors")
        let out = enc(src, pos: pos, prompt: tx["person.embeds"]!, promptMask: tx["person.attention_mask"]!)
        assertClose(out, try Fixtures.load("detect.safetensors")["detr.enc"]!, "detr.enc")
    }

    @Test func geometryEncoderKeysLoad() throws {  // parity: keys only
        let g = GeometryEncoder(Self.cfg.geometryEncoderConfig)
        try loadWeights(
            into: g, weights: SharedWeights.all, dtype: .float32,
            prefix: "detector_model.geometry_encoder.")
    }

    @Test func detrDecoderMatchesPython() throws {
        let dec = DETRDecoder(Self.cfg.detrDecoderConfig)
        try loadWeights(
            into: dec, weights: SharedWeights.all, dtype: .float32,
            prefix: "detector_model.detr_decoder.")
        let (_, pos) = try Self.srcPos()
        let tx = try Fixtures.load("text.safetensors")
        let dx = try Fixtures.load("detect.safetensors")
        let (hs, ref, pres) = dec(
            visionFeatures: dx["detr.enc"]!, inputsEmbeds: tx["person.embeds"]!,
            visionPosEncoding: pos, textMask: tx["person.attention_mask"]!, spatialShape: (72, 72))
        assertClose(hs, dx["detr.dec.hs"]!, "detr.dec.hs")
        assertClose(ref, dx["detr.dec.ref_boxes"]!, "detr.dec.ref_boxes")
        assertClose(pres, dx["detr.dec.presence"]!, "detr.dec.presence")
    }

    @Test func dotProductScoringMatchesPython() throws {
        let scoring = DotProductScoring(hiddenSize: Self.cfg.detrEncoderConfig.hiddenSize)
        try loadWeights(
            into: scoring, weights: SharedWeights.all, dtype: .float32,
            prefix: "detector_model.dot_product_scoring.")
        let tx = try Fixtures.load("text.safetensors")
        let dx = try Fixtures.load("detect.safetensors")
        let out = scoring(
            dx["detr.dec.hs"]!, inputsEmbeds: tx["person.embeds"]!, textMask: tx["person.attention_mask"]!)
        assertClose(out, dx["det.scoring"]!, "det.scoring")
    }

    @Test func detectEndToEndMatchesPython() throws {
        let det = DetectorModel(Self.cfg)
        try loadWeights(into: det, weights: SharedWeights.all, dtype: .float32, prefix: "detector_model.")
        let vx = try Fixtures.load("vision.safetensors")
        let tx = try Fixtures.load("text.safetensors")
        let dx = try Fixtures.load("detect.safetensors")
        let ix = try Fixtures.load("inputs.safetensors")
        let detFeats = (0..<3).map { vx["neck.det.\($0)"]! }
        let raw = det.detect(
            detFeatures: detFeats, textEmbeds: tx["person.embeds"]!, textMask: tx["person.attention_mask"]!)
        // atol 1e-3 (brief value, not the 1e-4 per-layer default): mask logits are a 256-d dot product
        // after the full DETR stack and pixel decoder, so fp32 reordering drift reaches ~1e-3 near zero.
        assertClose(raw.masks, dx["det.mask_decoder.pred_masks"]!, rtol: 1e-3, atol: 1e-3, "det.pred_masks")
        let img = ix["imageA"]!
        let dets = postprocessDetections(raw, imageSize: (img.dim(1), img.dim(0)), threshold: 0.3)
        let pyScores = dx["final.scores"]!.asArray(Float.self)
        #expect(dets.count == pyScores.count)
        let pyBoxes = dx["final.boxes"]!
        let pyMasks = dx["final.masks"]!
        for (i, d) in dets.enumerated() where i < pyScores.count {
            #expect(abs(d.score - pyScores[i]) < 0.01)
            for j in 0..<4 {
                #expect(abs(d.box[j] - pyBoxes[i, j].item(Float.self)) < 1.0 * Float(img.dim(1)) / 1008)
            }
            let up = resizeMasks(d.maskLogits[.newAxis], to: (img.dim(0), img.dim(1)))[0]
            #expect(maskIoU(up, pyMasks[i].asType(.float32) - 0.5) >= 0.99)
        }
    }
}
