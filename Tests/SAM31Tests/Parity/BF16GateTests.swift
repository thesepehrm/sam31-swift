import Foundation
import MLX
import Testing

@testable import SAM31

/// The bf16 gate: a `.bfloat16` model through the public API against the fp32 Python fixtures, on
/// the end-to-end thresholds only (mask IoU ≥ 0.99, boxes within 1 px in 1008 space, scores within
/// 0.01). Runs the Task 10 detect scenario and the Task 13 tracking scenario.
///
/// Opt-in with `SAM31_BF16_GATE=1` (`TEST_RUNNER_SAM31_BF16_GATE=1` under xcodebuild). bf16 does not
/// pass yet: detection passes (IoU 0.9997), but the mask-prompted tracking object drops to IoU
/// 0.986-0.990 on 5 of 10 frames (docs/perf.md), so `.float32` stays the default.
@Suite(
    .enabled(if: Fixtures.available && ProcessInfo.processInfo.environment["SAM31_BF16_GATE"] == "1"),
    .serialized)
struct BF16GateTests {
    private static let model = Task {
        try SAM31Model(
            config: ModelConfig.load(from: Fixtures.weightsURL), weights: SharedWeights.all, dtype: .bfloat16)
    }

    @Test func detectScenarioPassesGate() async throws {
        let model = try await Self.model.value
        let frame = try await model.encode(APIParityTests.imageA())
        let dets = try await model.detect(frame, text: "person", scoreThreshold: 0.3)
        let dx = try Fixtures.load("detect.safetensors")
        let pyScores = dx["final.scores"]!.asArray(Float.self)
        let pyMasks = dx["final.masks"]!
        #expect(dets.count == pyScores.count)
        let (w, h) = (Int(frame.sourceSize.width), Int(frame.sourceSize.height))
        let px = Float(w) / 1008  // 1 px in 1008 space, in source pixels
        var report: [String] = []
        for (i, d) in dets.enumerated() where i < pyScores.count {
            let b = dx["final.boxes"]![i].asArray(Float.self)
            let got = [d.box.minX, d.box.minY, d.box.maxX, d.box.maxY].map { Float($0) }
            let boxErr = zip(got, b).map { abs($0 - $1) }.max()! / px
            let scoreErr = abs(d.score - pyScores[i])
            let up = resizeMasks(d.mask.logits[.newAxis], to: (h, w))[0]
            let iou = maskIoU(up, pyMasks[i].asType(.float32) - 0.5)
            #expect(scoreErr < 0.01, "detection \(i) score Δ \(scoreErr)")
            #expect(boxErr < 1, "detection \(i) box Δ \(boxErr) px (1008 space)")
            #expect(iou >= 0.99, "detection \(i) mask IoU \(iou)")
            report.append(
                String(format: "det %d | IoU %.5f | box Δ %.3f px | score Δ %.4f", i, iou, boxErr, scoreErr))
        }
        print(
            "BF16-GATE-DETECT \(dets.count) vs \(pyScores.count) detections\n"
                + report.joined(separator: "\n"))
    }

    @Test func trackingScenarioPassesGate() async throws {
        let rows = try await APIParityTests.trackingScenario(Self.model.value)
        for row in rows {
            for (o, iou) in row.ious.enumerated() {
                #expect(iou >= 0.99, "\(row.tag) object \(o) IoU \(iou)")
            }
            for (o, (s, py)) in zip(row.scores, row.pyScores).enumerated() {
                #expect(abs(s - py) < 0.01, "\(row.tag) object \(o) score \(s) vs \(py)")
            }
        }
        print("BF16-GATE-TRACK\n" + APIParityTests.TrackingRow.report(rows))
    }
}
