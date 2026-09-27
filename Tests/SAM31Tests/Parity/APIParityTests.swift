import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing

@testable import SAM31

/// The public API against the Python fixtures. All tests share one loaded model (``SharedModel``).
@Suite(.enabled(if: Fixtures.available), .serialized) struct APIParityTests {
    static func imageA() -> CGImage {
        let url = Fixtures.fixturesURL!.appending(path: "imageA.png")
        let src = CGImageSourceCreateWithURL(url as CFURL, nil)!
        return CGImageSourceCreateImageAtIndex(src, 0, nil)!
    }

    static func clipFrame(_ i: Int) throws -> CGImage {
        rgbToCGImage(try Fixtures.load("inputs.safetensors")["clip"]![i])
    }

    /// `(1, 1008, 1008)` logits of a model-resolution mask, upsampled as the tracker does.
    static func highRes(_ m: Mask) -> MLXArray {
        resizeBilinearNHWC(m.logits.reshaped(1, m.height, m.width, 1), h: 1008, w: 1008).reshaped(
            1, 1008, 1008)
    }

    @Test func clickDetectAndTrackThroughPublicAPI() async throws {
        let model = try await SharedModel.get()
        let frame = try await model.encode(Self.imageA())
        let p = InteractiveParityTests.prompts
        let seg = try await model.segment(
            frame,
            prompt: .points([
                .init(x: p["pos"]![0], y: p["pos"]![1], label: .positive),
                .init(x: p["neg"]![0], y: p["neg"]![1], label: .negative),
            ]))
        let ix = try Fixtures.load("interactive.safetensors")
        let clickIoU = maskIoU(seg.mask.logits, ix["click2.out.pred_masks"]![0, 0])
        #expect(clickIoU >= 0.99)
        #expect(seg.score > 0 && seg.score <= 1.5)
        #expect(seg.objectScore > 0.5)

        // A second click on the same frame, fed the first result (iterative refinement).
        let again = try await model.segment(
            frame, prompt: .points([.init(x: p["pos"]![0], y: p["pos"]![1], label: .positive)]),
            previous: seg)
        #expect(maskIoU(again.mask.logits, seg.mask.logits) > 0.5)

        let dets = try await model.detect(frame, text: "person", scoreThreshold: 0.3)
        let dx = try Fixtures.load("detect.safetensors")
        let pyScores = dx["final.scores"]!.asArray(Float.self)
        #expect(dets.count == pyScores.count)
        let sx = Float(frame.sourceSize.width) / 1008
        for (i, d) in dets.enumerated() where i < pyScores.count {
            #expect(abs(d.score - pyScores[i]) < 0.01)
            let b = dx["final.boxes"]![i].asArray(Float.self)
            let got = [d.box.minX, d.box.minY, d.box.maxX, d.box.maxY].map { Float($0) }
            for j in 0..<4 { #expect(abs(got[j] - b[j]) < sx) }
        }
        // Cached text embeddings give the same detections.
        let again2 = try await model.detect(frame, text: "person", scoreThreshold: 0.3)
        #expect(again2.map(\.score) == dets.map(\.score))

        let session = await model.makeTrackingSession()
        _ = try await session.addObject(
            frameIndex: 0, features: frame,
            prompt: .segment(.points([.init(x: p["pos"]![0], y: p["pos"]![1], label: .positive)])))
        let frames = AsyncStream<FrameInput> { c in
            for i in 1..<4 { c.yield(.image(try! Self.clipFrame(i))) }
            c.finish()
        }
        var n = 0
        for try await f in session.propagate(frames, startingAt: 1, frameCount: 10) {
            #expect(f.frameIndex == n + 1)
            #expect(f.objects.count == 1)
            n += 1
        }
        #expect(n == 3)
        print("API-CLICK-REPORT click2 IoU \(clickIoU), detections \(dets.count)")
    }

    /// The tracking scenario of parity/README.md, driven only through `TrackingSession`: obj0 by
    /// click2 on f0, obj1 by mask prompt on f0, propagate f1-f5, refine obj0 on f5, propagate f6-f9.
    @Test func trackingScenarioMatchesPython() async throws {
        let rows = try await Self.trackingScenario(await SharedModel.get())
        for row in rows {
            for (o, iou) in row.ious.enumerated() {
                #expect(iou >= 0.99, "\(row.tag) object \(o) IoU \(iou)")
            }
            for (o, visible) in row.visible.enumerated() {
                #expect(visible == row.pyVisible[o], "\(row.tag) object \(o) visibility")
            }
        }
        print("API-TRACKING-REPORT\n" + TrackingRow.report(rows))
    }

    /// One frame of ``trackingScenario(_:)`` against the Python fixture.
    struct TrackingRow {
        let tag: String
        /// Mask IoU per object at 1008.
        let ious: [Float]
        /// Object-score probability per object, Swift and Python.
        let scores: [Float]
        let pyScores: [Float]
        let visible: [Bool]
        let pyVisible: [Bool]

        static func report(_ rows: [TrackingRow]) -> String {
            (["frame | IoU per object | |Δ score| max"]
                + rows.map { r in
                    let d = zip(r.scores, r.pyScores).map { abs($0 - $1) }.max() ?? 0
                    return "\(r.tag) | "
                        + r.ious.map { String(format: "%.5f", $0) }.joined(separator: " | ")
                        + (r.scores.isEmpty ? " | -" : String(format: " | %.4f", d))
                }).joined(separator: "\n")
        }
    }

    /// Runs the parity/README.md tracking scenario on `model` and compares every frame with Python.
    static func trackingScenario(_ model: SAM31Model) async throws -> [TrackingRow] {
        let tx = try Fixtures.load("track.safetensors")
        let p = InteractiveParityTests.prompts
        let session = await model.makeTrackingSession()
        let f0 = try await model.encode(Self.clipFrame(0))
        let a = try await session.addObject(
            frameIndex: 0, features: f0,
            prompt: .segment(
                .points([
                    .init(x: p["pos"]![0], y: p["pos"]![1], label: .positive),
                    .init(x: p["neg"]![0], y: p["neg"]![1], label: .negative),
                ])))
        // add_mask is binary at 1008; as logits (±1) it upsamples to itself and thresholds exactly.
        let b = try await session.addObject(
            frameIndex: 0, features: f0, prompt: .mask(Mask(logits: tx["add_mask"]! * 2 - 1)))
        #expect(await session.objects == [a, b])

        var rows: [TrackingRow] = []
        func check(_ tag: String, _ frame: TrackedFrame, _ key: String) {
            let py = tx["\(key).masks"]!
            let logits = tx["\(key).object_score_logits"]!.asType(.float32)
            var ious: [Float] = []
            var scores: [Float] = []
            var pyScores: [Float] = []
            var visible: [Bool] = []
            var pyVisible: [Bool] = []
            for (o, id) in [a, b].enumerated() {
                guard let obj = frame.objects[id] else {
                    Issue.record("\(tag): object \(o) missing")
                    return
                }
                ious.append(maskIoU(Self.highRes(obj.mask), py[o]))
                let logit = logits[o, 0].item(Float.self)
                scores.append(obj.score)
                pyScores.append(1 / (1 + exp(-logit)))
                visible.append(obj.isVisible)
                pyVisible.append(logit > 0)
            }
            rows.append(
                TrackingRow(
                    tag: tag, ious: ious, scores: scores, pyScores: pyScores, visible: visible,
                    pyVisible: pyVisible))
        }

        func stream(_ range: Range<Int>) -> AsyncStream<FrameInput> {
            AsyncStream { c in
                for i in range { c.yield(.image(try! Self.clipFrame(i))) }
                c.finish()
            }
        }
        for try await f in session.propagate(stream(1..<5), startingAt: 1, frameCount: 10) {
            check("f\(f.frameIndex)", f, "f\(f.frameIndex)")
        }
        let f5 = try await model.encode(Self.clipFrame(5))
        let only5 = AsyncStream<FrameInput> { c in
            c.yield(.features(f5))
            c.finish()
        }
        for try await f in session.propagate(only5, startingAt: 5, frameCount: 10) {
            check("f5.pre_refine", f, "f5.pre_refine")
        }
        let refined = try await session.refine(
            a, frameIndex: 5, features: f5,
            prompt: .points([.init(x: p["pos"]![0], y: p["pos"]![1] + 20, label: .positive)]))
        rows.append(
            TrackingRow(
                tag: "f5 refine obj0", ious: [maskIoU(Self.highRes(refined), tx["f5.masks"]![0])], scores: [],
                pyScores: [], visible: [], pyVisible: []))
        for try await f in session.propagate(stream(6..<10), startingAt: 6, frameCount: 10) {
            check("f\(f.frameIndex)", f, "f\(f.frameIndex)")
        }
        return rows
    }

    /// Every user-reachable multiplex precondition is an error, not a trap.
    @Test func sessionRejectsInvalidUse() async throws {
        let model = try await SharedModel.get()
        let frame = try await model.encode(Self.imageA())
        let session = await model.makeTrackingSession()
        let one = AsyncStream<FrameInput> { c in
            c.yield(.features(frame))
            c.finish()
        }

        await #expect(throws: SAM31Error.self) {  // no objects yet
            for try await _ in session.propagate(one, startingAt: 0, frameCount: nil) {}
        }
        await #expect(throws: SAM31Error.unknownObject(42)) {
            try await session.remove(ObjectID(rawValue: 42))
        }
        await #expect(throws: SAM31Error.self) {
            try await session.addObject(frameIndex: -1, features: frame, prompt: .segment(.points([])))
        }

        // Fill the one bucket with 16 objects (mask prompts: 16 cheap merges on frame 0).
        let tx = try Fixtures.load("track.safetensors")
        let mask = Mask(logits: tx["add_mask"]! * 2 - 1)
        var ids: [ObjectID] = []
        for _ in 0..<16 {
            ids.append(try await session.addObject(frameIndex: 0, features: frame, prompt: .mask(mask)))
        }
        #expect(await session.objects.count == 16)
        await #expect(throws: SAM31Error.self) {
            try await session.addObject(frameIndex: 0, features: frame, prompt: .mask(mask))
        }
        // Removed objects keep their slot, so the bucket stays full.
        try await session.remove(ids[3])
        await #expect(throws: SAM31Error.unknownObject(ids[3].rawValue)) {
            _ = try await session.refine(
                ids[3], frameIndex: 0, features: frame,
                prompt: .points([
                    .init(x: 504, y: 504, label: .positive)
                ]))
        }
        await #expect(throws: SAM31Error.self) {
            try await session.addObject(frameIndex: 0, features: frame, prompt: .mask(mask))
        }
        // Refine on a frame the session never tracked.
        await #expect(throws: SAM31Error.self) {
            _ = try await session.refine(
                ids[0], frameIndex: 7, features: frame,
                prompt: .points([
                    .init(x: 504, y: 504, label: .positive)
                ]))
        }
        // Refine on a tracked frame works, and the removal took effect.
        let m = try await session.refine(
            ids[0], frameIndex: 0, features: frame, prompt: .points([.init(x: 504, y: 504, label: .positive)])
        )
        #expect(m.width == 288)
        let once = AsyncStream<FrameInput> { c in
            c.yield(.features(frame))
            c.finish()
        }
        for try await f in session.propagate(once, startingAt: 1, frameCount: nil) {
            #expect(f.objects.count == 15)
            #expect(f.objects[ids[3]] == nil)
        }

        // Removing every object resets the session; it can start again.
        for id in ids where id != ids[3] { try await session.remove(id) }
        #expect(await session.objects.isEmpty)
        let fresh = try await session.addObject(
            frameIndex: 2, features: frame,
            prompt: .segment(.box(CGRect(x: 300, y: 300, width: 400, height: 400))))
        #expect(await session.objects == [fresh])
        // A new object on a frame the session has not tracked yet is propagated there first.
        let other = try await session.addObject(
            frameIndex: 4, features: frame,
            prompt: .segment(.points([.init(x: 504, y: 504, label: .positive)])))
        #expect(await session.objects == [fresh, other])
    }

    @Test func propagationCancels() async throws {
        let model = try await SharedModel.get()
        let img = Self.imageA()
        let frame = try await model.encode(img)
        let session = await model.makeTrackingSession()
        _ = try await session.addObject(
            frameIndex: 0, features: frame,
            prompt: .segment(.points([.init(x: 504, y: 504, label: .positive)])))
        // Endless, pull-based: produces a frame only when the tracker asks for one.
        let endless = AsyncStream<FrameInput>(unfolding: { .image(img) })
        let task = Task {
            var n = 0
            for try await _ in session.propagate(endless, startingAt: 1, frameCount: nil) { n += 1 }
            return n
        }
        try await Task.sleep(for: .seconds(3))
        task.cancel()
        switch await task.result {
        case .failure(let e): #expect(e is CancellationError)
        case .success(let n): Issue.record("expected cancellation, got \(n) frames")
        }
        // The session stays valid: tracking resumes.
        let once = AsyncStream<FrameInput> { c in
            c.yield(.features(frame))
            c.finish()
        }
        var resumed = 0
        for try await f in session.propagate(once, startingAt: 50, frameCount: nil) {
            #expect(f.frameIndex == 50 && f.objects.count == 1)
            resumed += 1
        }
        #expect(resumed == 1)
    }
}
