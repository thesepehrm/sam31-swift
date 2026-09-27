import MLX
import Testing

@testable import SAM31

@Suite(.enabled(if: Fixtures.available), .serialized) struct VisionParityTests {
    @Test func backboneMatchesPython() throws {
        let cfg = try ModelConfig.load(from: Fixtures.weightsURL).detectorConfig.visionConfig.backboneConfig
        let vit = ViTBackbone(cfg)
        try loadWeights(
            into: vit, weights: SharedWeights.all, dtype: .float32,
            prefix: "detector_model.vision_encoder.backbone.")
        let pre = try Fixtures.load("preprocess.safetensors")
        let fx = try Fixtures.load("vision.safetensors")
        let (out, capture) = vit(pre["pixel_values_A"]!, capture: [0, 8, 16, 31])
        eval(out)

        // Per-layer asserts localize a regression to the first diverging layer.
        assertClose(capture.patchEmbed!, fx["vit.patch_embed"]!, "vit.patch_embed")
        for i in [0, 8, 16] {
            assertClose(capture.blocks[i]!, fx["vit.block.\(i)"]!, "vit.block.\(i)")
        }
        // atol 1e-3 for the last block: fp32 drift accumulates smoothly over 32 blocks with |x| up to
        // ~300 (max|Δ| 4.2e-3, relative to max|x| ~2e-5). The Python reference itself differs
        // by max|Δ| 9.0e-3 between its GPU and CPU runs at block 31, and would fail atol 1e-4 too.
        assertClose(capture.blocks[31]!, fx["vit.block.31"]!, atol: 1e-3, "vit.block.31")
        assertClose(out, fx["vit.out"]!, atol: 1e-3, "vit.out")
    }
}
