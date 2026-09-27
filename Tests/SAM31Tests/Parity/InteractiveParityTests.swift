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

    /// The whole tracker loaded from the checkpoint (verify `.all`), and the frame features that
    /// `Model.tracker_frame_features` builds for image A.
    static func trackerAndFeatures() throws -> (MultiplexTrackerModel, TrackerFrameFeatures) {
        let trk = MultiplexTrackerModel(tcfg)
        try loadWeights(into: trk, weights: SharedWeights.all, dtype: .float32, prefix: "tracker_model.")
        let vx = try Fixtures.load("vision.safetensors")
        let ff = trk.prepareFrameFeatures(
            interactiveFPN: (0..<3).map { vx["neck.interactive.\($0)"]! },
            propagationFPN: (0..<3).map { vx["neck.propagation.\($0)"]! })
        return (trk, ff)
    }

    /// Decoder-level check on the frame features of `prepareFrameFeatures` and
    /// `getInteractivePixMem`, as tracker.py builds them for an init conditioning frame.
    @Test func interactiveMaskDecoderMatchesPython() throws {
        let (trk, ff) = try Self.trackerAndFeatures()
        let pe = trk.interactiveSamPromptEncoder
        let ix = try Fixtures.load("interactive.safetensors")
        let inter = ff.interactive!
        let pix = trk.getInteractivePixMem(inter.visionFeat)
        for (tag, points, labels) in Self.cases {
            let (sparse, dense) = Self.encode(pe, points: points, labels: labels)
            // tracker.py::_use_multimask: multimask only for a single point on an init frame.
            let out = trk.interactiveSamMaskDecoder(
                imageEmbeddings: pix, imagePE: pe.getDensePE(), multimaskOutput: labels.count == 1,
                highResFeatures: inter.highRes, sparsePromptEmbeddings: sparse, densePromptEmbeddings: dense)
            assertClose(out.masks, ix["\(tag).sam.decoder.masks"]!, "\(tag).masks")
            assertClose(out.iouPred, ix["\(tag).sam.decoder.iou_pred"]!, "\(tag).iou_pred")
            assertClose(out.samTokensOut, ix["\(tag).sam.decoder.sam_tokens_out"]!, "\(tag).sam_tokens_out")
            assertClose(
                out.objectScoreLogits, ix["\(tag).sam.decoder.object_score_logits"]!,
                "\(tag).object_score_logits")
        }
    }

    /// `track_step` on a fresh 1-object state, init conditioning frame, points only (default path,
    /// memory encoder on, as in dump.py).
    @Test(arguments: ["click1", "click2", "box"])
    func interactiveStepMatchesPython(_ tag: String) throws {
        let (trk, ff) = try Self.trackerAndFeatures()
        let ix = try Fixtures.load("interactive.safetensors")
        let (_, points, labels) = Self.cases.first { $0.tag == tag }!
        let st = trk.initState(numObjects: 1, objectIDs: nil)
        let out = trk.trackStep(
            st, frameIndex: 0, isInitCondFrame: true, features: ff,
            pointInputs: PointInputs(
                coords: MLXArray(points).reshaped(1, -1, 2), labels: MLXArray(labels).reshaped(1, -1)),
            maskInputs: nil, numFrames: 10)
        // The brief allowed atol 1e-3 on pred_masks; the default fp32 tolerance holds (max |Δ| 2.1e-5).
        assertClose(out.predMasks, ix["\(tag).out.pred_masks"]!, "\(tag).pred_masks")
        assertClose(out.predMasksHighRes, ix["\(tag).out.pred_masks_high_res"]!, "\(tag).pred_masks_high_res")
        assertClose(
            out.objectScoreLogits, ix["\(tag).out.object_score_logits"]!, "\(tag).object_score_logits")
        assertClose(out.objPtr!, ix["\(tag).out.obj_ptr"]!, "\(tag).obj_ptr")
        #expect(maskIoU(out.predMasksHighRes, ix["\(tag).out.pred_masks_high_res"]!) >= 0.99)
        #expect(st.condFrameOutputs[0] === out)
        #expect(out.conditioningObjects == [0])
    }

    /// `track_step` with a mask prompt on a fresh 1-object state (mask-as-output mode), which also
    /// exercises the mask-prompt pre-step of `_forward_sam_heads`. Fixture: `mask_prompt.out.*`.
    /// `expandedDimensions` rather than `[.newAxis, .newAxis]`: mlx-swift 0.31.4 drops the new axes
    /// when the index count equals the array's rank.
    @Test func maskPromptStepMatchesPython() throws {
        let (trk, ff) = try Self.trackerAndFeatures()
        let tx = try Fixtures.load("track.safetensors")
        let st = trk.initState(numObjects: 1, objectIDs: nil)
        let out = trk.trackStep(
            st, frameIndex: 0, isInitCondFrame: true, features: ff, pointInputs: nil,
            maskInputs: expandedDimensions(tx["add_mask"]!, axes: [0, 1]), numFrames: 10)
        assertClose(out.predMasks, tx["mask_prompt.out.pred_masks"]!, "mask_prompt.pred_masks")
        assertClose(out.predMasksHighRes, tx["mask_prompt.out.pred_masks_high_res"]!, "mask_prompt.high_res")
        assertClose(
            out.objectScoreLogits, tx["mask_prompt.out.object_score_logits"]!,
            "mask_prompt.object_score_logits")
        assertClose(out.objPtr!, tx["mask_prompt.out.obj_ptr"]!, "mask_prompt.obj_ptr")
        #expect(out.conditioningObjects == [0])
    }
}
