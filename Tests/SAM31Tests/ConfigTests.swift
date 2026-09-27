import Foundation
import Testing

@testable import SAM31

@Suite struct ConfigTests {
    let config: ModelConfig = {
        let url = Bundle.module.url(forResource: "Resources/config", withExtension: "json")!
        return try! JSONDecoder().decode(ModelConfig.self, from: Data(contentsOf: url))
    }()

    @Test func visionBackbone() {
        let vb = config.detectorConfig.visionConfig.backboneConfig
        #expect(vb.hiddenSize == 1024 && vb.numHiddenLayers == 32 && vb.numAttentionHeads == 16)
        #expect(vb.patchSize == 14 && vb.imageSize == 1008 && vb.windowSize == 24)
        #expect(vb.globalAttnIndexes == [7, 15, 23, 31])
        #expect(vb.intermediateSize == 4736)
        #expect(config.detectorConfig.visionConfig.scaleFactors == [4.0, 2.0, 1.0])
    }

    @Test func text() {
        let t = config.detectorConfig.textConfig
        #expect(t.hiddenSize == 1024 && t.numHiddenLayers == 24 && t.maxPositionEmbeddings == 32)
        #expect(t.vocabSize == 49408 && t.projectionDim == 512)
    }

    @Test func trackerPostInitOverrides() {
        let tc = config.trackerConfig
        #expect(tc.memoryAttentionNumAttentionHeads == 8)
        #expect(tc.sigmoidScaleForMemEnc == 2.0 && tc.sigmoidBiasForMemEnc == -1.0)
        #expect(tc.maskDecoderConfig.multimaskOutputsOnly)
        #expect(tc.multiplexCount == 16 && tc.numMaskmem == 7 && tc.imageSize == 1008)
    }

    @Test func nullAndMissingSubConfigsUseDefaults() throws {
        let json = #"{"detector_config": null, "tracker_config": {"mask_decoder_config": null}}"#
        let c = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
        #expect(c.lowResMaskSize == 288 && c.detNmsThresh == 0.1)
        #expect(c.detectorConfig.detrDecoderConfig.numQueries == 200)
        #expect(c.detectorConfig.visionConfig.backboneConfig.layerScaleInitValue == nil)
        // model_type defaults to "sam3.1_tracker_video", so the SAM 3 overrides do not apply,
        // but mask_decoder_config still becomes multimask-only.
        #expect(c.trackerConfig.sigmoidBiasForMemEnc == -1.0)
        #expect(c.trackerConfig.maskDecoderConfig.multimaskOutputsOnly)
    }

    @Test func loadReadsConfigJSONFromDirectory() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: SAM31Error.weightsNotFound(dir)) { try ModelConfig.load(from: dir) }
        let src = Bundle.module.url(forResource: "Resources/config", withExtension: "json")!
        try FileManager.default.copyItem(at: src, to: dir.appending(path: "config.json"))
        #expect(try ModelConfig.load(from: dir).trackerConfig.multiplexCount == 16)
    }
}
