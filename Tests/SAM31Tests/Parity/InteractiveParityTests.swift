import Foundation
import MLX
import Testing

@testable import SAM31

@Suite(.enabled(if: Fixtures.available), .serialized) struct InteractiveParityTests {
    static let tcfg = try! ModelConfig.load(from: Fixtures.weightsURL).trackerConfig
    static var prompts: [String: [Float]] {
        let raw = Fixtures.manifest["prompts"] as! [String: Any]
        return raw.compactMapValues { $0 as? [Double] }.mapValues { $0.map(Float.init) }
    }

    /// The three interactive cases of parity/dump.py: points in 1008 space and their labels.
    static var cases: [(tag: String, points: [Float], labels: [Int32])] {
        let p = prompts
        return [
            ("click1", p["pos"]!, [1]),
            ("click2", p["pos"]! + p["neg"]!, [1, 0]),
            ("box", p["box"]!, [2, 3]),
        ]
    }

    static func loadedPromptEncoder() throws -> SAMPromptEncoder {
        let pe = SAMPromptEncoder(tcfg.promptEncoderConfig)
        try loadWeights(
            into: pe, weights: SharedWeights.all, dtype: .float32,
            prefix: "tracker_model.interactive_sam_prompt_encoder.")
        return pe
    }

    static func encode(_ pe: SAMPromptEncoder, points: [Float], labels: [Int32]) -> (MLXArray, MLXArray) {
        let coords = MLXArray(points).reshaped(1, labels.count, 2)
        let (c, l) = preparePointInputs(
            coords: coords, labels: MLXArray(labels).reshaped(1, labels.count),
            embeddingSize: pe.imageEmbeddingSize.h, imageSize: tcfg.imageSize)
        return pe(points: (c, l), boxes: nil, masks: nil)
    }

    @Test func promptEncoderMatchesPython() throws {
        let pe = try Self.loadedPromptEncoder()
        let ix = try Fixtures.load("interactive.safetensors")
        for (tag, points, labels) in Self.cases {
            let (sparse, dense) = Self.encode(pe, points: points, labels: labels)
            assertClose(sparse, ix["\(tag).sam.prompt.sparse"]!, "\(tag).sparse")
            assertClose(dense, ix["\(tag).sam.prompt.dense"]!, "\(tag).dense")
        }
    }

    @Test func maskDecodersAndPromptEncoderKeysLoad() throws {  // parity: keys only
        _ = try Self.loadedPromptEncoder()
        let interactive = MultiplexMaskDecoder(Self.tcfg.interactiveMaskDecoderConfig)
        try loadWeights(
            into: interactive, weights: SharedWeights.all, dtype: .float32,
            prefix: "tracker_model.interactive_sam_mask_decoder.")
        #expect(interactive.outputHypernetworksMlps.count == 4)
        let propagation = MultiplexMaskDecoder(Self.tcfg.maskDecoderConfig)
        try loadWeights(
            into: propagation, weights: SharedWeights.all, dtype: .float32,
            prefix: "tracker_model.sam_mask_decoder.")
        #expect(propagation.outputHypernetworksMlps.count == 3)
    }

    /// Decoder-level check with the frame features rebuilt the way tracker.py does for an init
    /// conditioning frame: `prepare_frame_features` (conv_s0/conv_s1 on the interactive FPN) and
    /// `_get_interactive_pix_mem` (add `interactivity_no_mem_embed`).
    @Test func interactiveMaskDecoderMatchesPython() throws {
        let pe = try Self.loadedPromptEncoder()
        let dec = MultiplexMaskDecoder(Self.tcfg.interactiveMaskDecoderConfig)
        try loadWeights(
            into: dec, weights: SharedWeights.all, dtype: .float32,
            prefix: "tracker_model.interactive_sam_mask_decoder.")
        let vx = try Fixtures.load("vision.safetensors")
        let ix = try Fixtures.load("interactive.safetensors")
        let noMem = SharedWeights.all["tracker_model.interactivity_no_mem_embed"]!
        let feat = vx["neck.interactive.2"]!
        let pix = (feat + noMem).reshaped(1, feat.dim(1) * feat.dim(2), feat.dim(3))
        let highRes = [dec.convS0(vx["neck.interactive.0"]!), dec.convS1(vx["neck.interactive.1"]!)]
        for (tag, points, labels) in Self.cases {
            let (sparse, dense) = Self.encode(pe, points: points, labels: labels)
            // tracker.py::_use_multimask: multimask only for a single point on an init frame.
            let out = dec(
                imageEmbeddings: pix, imagePE: pe.getDensePE(), multimaskOutput: labels.count == 1,
                highResFeatures: highRes, sparsePromptEmbeddings: sparse, densePromptEmbeddings: dense)
            assertClose(out.masks, ix["\(tag).sam.decoder.masks"]!, "\(tag).masks")
            assertClose(out.iouPred, ix["\(tag).sam.decoder.iou_pred"]!, "\(tag).iou_pred")
            assertClose(out.samTokensOut, ix["\(tag).sam.decoder.sam_tokens_out"]!, "\(tag).sam_tokens_out")
            assertClose(
                out.objectScoreLogits, ix["\(tag).sam.decoder.object_score_logits"]!,
                "\(tag).object_score_logits")
        }
    }
}
