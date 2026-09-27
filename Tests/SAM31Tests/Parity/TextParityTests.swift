import MLX
import MLXNN
import Testing

@testable import SAM31

@Suite(.enabled(if: Fixtures.available), .serialized) struct TextParityTests {
    @Test(arguments: ["person", "the_dancer", "city_skyline"])
    func textEncoderMatchesPython(_ key: String) throws {
        let cfg = try ModelConfig.load(from: Fixtures.weightsURL).detectorConfig
        let enc = TextEncoder(cfg.textConfig)
        try loadWeights(
            into: enc, weights: SharedWeights.all, dtype: .float32,
            prefix: "detector_model.text_encoder.")
        let proj = Linear(cfg.textConfig.hiddenSize, cfg.detrEncoderConfig.hiddenSize)
        try loadWeights(
            into: proj, weights: SharedWeights.all, dtype: .float32,
            prefix: "detector_model.text_projection.")
        let fx = try Fixtures.load("text.safetensors")
        let ids = fx["\(key).input_ids"]!
        let am = fx["\(key).attention_mask"]!
        let tok = try CLIPTokenizer.bundled().encode(key.replacingOccurrences(of: "_", with: " "))
        #expect(ids.asArray(Int32.self) == tok.inputIds)
        let hidden = enc(ids, attentionMask: am)
        assertClose(hidden, fx["\(key).hidden"]!, "\(key).hidden")
        assertClose(proj(hidden), fx["\(key).embeds"]!, "\(key).embeds")
    }
}
