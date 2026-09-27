import MLX
import Testing

@testable import SAM31

@Suite struct SanitizeTests {
    let patchEmbed = "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
    let maskEmbedConv1 = "tracker_model.interactive_sam_prompt_encoder.mask_embed.conv1.weight"
    let finalConv = "tracker_model.memory_encoder.mask_downsampler.final_conv.weight"

    @Test func remapsMaskEmbedAndFuserAndDownsampler() {
        let w: [String: MLXArray] = [
            "tracker_model.interactive_sam_prompt_encoder.mask_embed.0.weight": zeros([4, 2, 2, 1]),
            "tracker_model.memory_encoder.memory_fuser.layers.0.norm.weight": zeros([256]),
            "tracker_model.memory_encoder.mask_downsampler.layers.4.conv.weight": zeros([256, 1, 1, 256]),
            // already-MLX marker: patch embed with channels-last
            patchEmbed: zeros([1024, 14, 14, 3]),
        ]
        let s = sanitize(w)
        #expect(s[maskEmbedConv1] != nil)
        #expect(s["tracker_model.memory_encoder.memory_fuser.layers.0.layer_norm.weight"] != nil)
        #expect(s[finalConv]?.shape == [256, 1, 1, 256])
        #expect(s[maskEmbedConv1]?.shape == [4, 2, 2, 1])
    }

    @Test func transposesTorchLayoutConvs() {
        let w: [String: MLXArray] = [
            patchEmbed: zeros([1024, 3, 14, 14]),
            "x.upscale_conv1.weight": zeros([256, 64, 2, 2]),
        ]
        let s = sanitize(w)
        #expect(s[patchEmbed]?.shape == [1024, 14, 14, 3])
        #expect(s["x.upscale_conv1.weight"]?.shape == [64, 2, 2, 256])
    }

    @Test func remapsAllMaskEmbedIndicesAndSkipsTemporalEncoding() {
        let w: [String: MLXArray] = [
            "a.patch_embeddings.projection.weight": zeros([8, 3, 2, 2]),
            "p.mask_embed.1.weight": zeros([4]),
            "p.mask_embed.3.weight": zeros([4, 1, 2, 2]),
            "p.mask_embed.4.bias": zeros([4]),
            "p.mask_embed.6.weight": zeros([4, 1, 1, 1]),
            "t.memory_temporal_positional_encoding": zeros([7, 1, 1, 64]),
        ]
        let s = sanitize(w)
        #expect(s["p.mask_embed.layer_norm1.weight"] != nil)
        #expect(s["p.mask_embed.conv2.weight"]?.shape == [4, 2, 2, 1])
        #expect(s["p.mask_embed.layer_norm2.bias"] != nil)
        #expect(s["p.mask_embed.conv3.weight"] != nil)
        #expect(s["t.memory_temporal_positional_encoding"]?.shape == [7, 1, 1, 64])
        #expect(s.count == w.count)
    }
}
