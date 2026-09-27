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
}
