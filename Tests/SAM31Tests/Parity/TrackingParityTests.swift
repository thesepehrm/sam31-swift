import Foundation
import MLX
import Testing

@testable import SAM31

/// Replays the "Tracking scenario" of parity/README.md: obj0 by click2 on f0, obj1 by
/// `add_mask_prompt` on f0, propagate f1-f4, propagate + refine f5, propagate f6-f9.
@Suite(.enabled(if: Fixtures.available), .serialized) struct TrackingParityTests {
    /// The raw `MultiplexMemoryEncoder` output of the first memory-encoder call (f0 click2
    /// `track_step`, 1 object), before `_encode_new_memory` adds `no_obj_embed_spatial`.
    @Test func firstMemoryEncoderCallMatchesPython() throws {
        let (trk, ff) = try InteractiveParityTests.trackerAndFeatures()
        let tx = try Fixtures.load("track.safetensors")
        let prompts = InteractiveParityTests.prompts
        let st = trk.initState(numObjects: 1, objectIDs: nil)
        let out = trk.trackStep(
            st, frameIndex: 0, isInitCondFrame: true, features: ff,
            pointInputs: PointInputs(
                coords: MLXArray(prompts["pos"]! + prompts["neg"]!).reshaped(1, 2, 2),
                labels: MLXArray([1, 0] as [Int32]).reshaped(1, 2)),
            maskInputs: nil, numFrames: 10)
        let masks = trk.memoryEncoderMaskInput(
            predMasksHighRes: out.predMasksHighRes, conditioningObjects: out.conditioningObjects,
            multiplexState: st.multiplexState)
        let (features, posEnc) = trk.memoryEncoder(ff.propagation!.visionFeat, masks: masks)
        assertClose(features, tx["mem.encoder.first.features"]!, "mem.encoder.first.features")
        let dF = abs(features - tx["mem.encoder.first.features"]!).max().item(Float.self)
        let dP = abs(posEnc - tx["mem.encoder.first.pos_enc"]!).max().item(Float.self)
        print("MEM-ENCODER-REPORT features max|Δ| \(dF), pos_enc max|Δ| \(dP)")
        assertClose(posEnc, tx["mem.encoder.first.pos_enc"]!, "mem.encoder.first.pos_enc")
        // The stored memory is the raw pos enc; the features gain only the no-object embedding.
        assertClose(out.maskmemPosEnc!, tx["mem.encoder.first.pos_enc"]!, "maskmem_pos_enc")
        #expect(out.maskmemFeatures!.shape == [1, 72, 72, 256])
    }

    @Test func tenFrameSessionMatchesPython() throws {
        let cfg = try ModelConfig.load(from: Fixtures.weightsURL)
        let root = SAM31Root(cfg)
        try loadWeights(into: root, weights: SharedWeights.all, dtype: .float32, prefix: nil)
        let trk = root.trackerModel
        let det = root.detectorModel
        let ix = try Fixtures.load("inputs.safetensors")
        let tx = try Fixtures.load("track.safetensors")
        let clip = ix["clip"]!  // (10,H,W,3) uint8
        let prompts = InteractiveParityTests.prompts
        let pre = ImagePreprocessor()

        func features(_ i: Int, interactive: Bool = true) -> TrackerFrameFeatures {
            let pv = pre.pixelValues(fromRGB: clip[i])
            let bb = det.visionEncoder.backbone(pv)
            let (_, inter, prop) = det.visionEncoder.neck(
                bb, needDet: false, needInteractive: interactive, needPropagation: true)
            let ff = trk.prepareFrameFeatures(interactiveFPN: interactive ? inter : nil, propagationFPN: prop)
            eval(ff.propagation!.visionFeat, ff.propagation!.highRes)
            return ff
        }

        let st = trk.initState(numObjects: 1, objectIDs: nil)
        let f0 = features(0)
        _ = trk.trackStep(
            st, frameIndex: 0, isInitCondFrame: true, features: f0,
            pointInputs: .init(
                coords: MLXArray(prompts["pos"]! + prompts["neg"]!).reshaped(1, 2, 2),
                labels: MLXArray([1, 0] as [Int32]).reshaped(1, 2)),
            maskInputs: nil, numFrames: 10)
        let out0 = trk.addMaskPrompt(
            st, frameIndex: 0, features: f0, masks: tx["add_mask"]![.newAxis], objectIDs: nil)
        var report = ["frame | obj0 IoU | obj1 IoU | obj-score max|Δ| | propagate ms"]
        func record(_ tag: String, _ out: FrameOutput, _ key: String, ms: Double?) {
            let py = tx["\(key).masks"]!
            #expect(out.predMasksHighRes.shape == py.shape, "\(tag) shape")
            guard out.predMasksHighRes.shape == py.shape else { return }
            var ious: [String] = []
            for o in 0..<py.dim(0) {
                let iou = maskIoU(out.predMasksHighRes[o], py[o])
                #expect(iou >= 0.99, "\(tag) object \(o) IoU \(iou)")
                ious.append(String(format: "%.5f", iou))
            }
            let scores = tx["\(key).object_score_logits"]!
            let d = abs(out.objectScoreLogits - scores).max().item(Float.self)
            assertClose(out.objectScoreLogits, scores, rtol: 1e-2, atol: 1e-2, "\(tag).obj_scores")
            report.append(
                "\(tag) | \(ious.joined(separator: " | ")) | \(String(format: "%.2e", d)) | "
                    + (ms.map { String(format: "%.0f", $0) } ?? "-"))
        }
        record("f0", out0, "f0", ms: nil)

        let clock = ContinuousClock()
        for i in 1..<10 {
            let fi = features(i, interactive: i == 5)
            let t0 = clock.now
            var out = trk.propagate(st, frameIndex: i, features: fi, numFrames: 10)  // R8: always 10
            eval(out.predMasksHighRes, out.objectScoreLogits, out.maskmemFeatures!)
            let dt = (clock.now - t0).components
            let ms = Double(dt.seconds) * 1e3 + Double(dt.attoseconds) / 1e15
            if i == 5 {
                record("f5.pre_refine", out, "f5.pre_refine", ms: ms)
                out = trk.trackStep(
                    st, frameIndex: 5, isInitCondFrame: false, features: fi,
                    pointInputs: .init(
                        coords: MLXArray([prompts["pos"]![0], prompts["pos"]![1] + 20]).reshaped(1, 1, 2),
                        labels: MLXArray([1] as [Int32]).reshaped(1, 1)),
                    maskInputs: nil, numFrames: 10,
                    objectsToInteract: [0])  // R6: no prevSAMMaskLogits (multi-object safe)
                record("f5", out, "f5", ms: nil)
            } else {
                record("f\(i)", out, "f\(i)", ms: ms)
            }
        }
        print("TRACKING-REPORT\n" + report.joined(separator: "\n"))
    }
}
