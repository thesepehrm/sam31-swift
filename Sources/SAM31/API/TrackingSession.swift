import Foundation
import MLX

/// Tracks objects through a video with SAM 3.1's multiplex memory tracker.
///
/// Add objects on any frame with clicks, a box, or a mask (for example from
/// ``SAM31Model/detect(_:text:scoreThreshold:)``), then call
/// ``propagate(_:startingAt:frameCount:)`` to follow them frame by frame. Clicks on a tracked frame
/// correct an object with ``refine(_:frameIndex:features:prompt:)``.
///
/// A session holds at most 16 objects (``ModelSummary/maxObjectsPerBucket``): the tracker processes
/// all objects of a session as one multiplexed bucket. A removed object keeps its slot until every
/// object is removed, which resets the session. Adding an object beyond the limit throws
/// ``SAM31Error/invalidPrompt(_:)``; start another session for more objects.
///
/// The session keeps only the tracker's bounded memory (the prompted frames plus the most recent
/// frames), never past ``FrameFeatures``. All GPU work runs on the owning ``SAM31Model``.
public actor TrackingSession {
    let model: SAM31Model
    let state = SessionState()
    private var ids: [ObjectID] = []
    private var nextID = 0

    init(model: SAM31Model) {
        self.model = model
    }

    /// The objects in the session, in the order they were added.
    public var objects: [ObjectID] { ids }

    /// Adds an object prompted on frame `frameIndex`.
    ///
    /// On the session's first object the frame becomes a conditioning frame. On a frame the session
    /// has already tracked, the new object is merged into that frame's result; on any other frame the
    /// existing objects are first propagated to it.
    ///
    /// - Parameters:
    ///   - frameIndex: index of the frame in the video (0-based).
    ///   - features: the encoded frame.
    ///   - prompt: clicks/box in 1008×1008 model-input space, or a mask.
    /// - Returns: the new object's ID.
    /// - Throws: ``SAM31Error/invalidPrompt(_:)`` for a malformed prompt, a negative frame index, or
    ///   a full session.
    public func addObject(frameIndex: Int, features: FrameFeatures, prompt: TrackPrompt) async throws
        -> ObjectID
    {
        let id = ObjectID(rawValue: nextID)
        nextID += 1
        try await model.sessionAddObject(
            state, id: id.rawValue, frameIndex: frameIndex, features: features, prompt: prompt)
        ids.append(id)
        return id
    }

    /// Corrects an object on a tracked frame with more clicks or a box.
    ///
    /// The frame's other objects are re-propagated, and the refined object conditions on this frame
    /// from now on.
    ///
    /// - Returns: the object's corrected mask on this frame.
    /// - Throws: ``SAM31Error/unknownObject(_:)`` for an ID not in the session, and
    ///   ``SAM31Error/invalidPrompt(_:)`` for a malformed prompt or a frame the session has not
    ///   tracked (or no longer remembers).
    public func refine(_ id: ObjectID, frameIndex: Int, features: FrameFeatures, prompt: SegmentPrompt)
        async throws
        -> Mask
    {
        guard ids.contains(id) else { throw SAM31Error.unknownObject(id.rawValue) }
        return try await model.sessionRefine(
            state, id: id.rawValue, frameIndex: frameIndex, features: features, prompt: prompt)
    }

    /// Removes an object. It no longer appears in tracked frames. Removing the last object resets the
    /// session.
    ///
    /// - Throws: ``SAM31Error/unknownObject(_:)`` for an ID not in the session.
    public func remove(_ id: ObjectID) throws {
        guard let i = ids.firstIndex(of: id) else { throw SAM31Error.unknownObject(id.rawValue) }
        ids.remove(at: i)
        // Applied on the model actor before its next session call, so no GPU state changes here.
        state.requestRemoval(id.rawValue)
    }

    /// Tracks every object through `frames`, one result per frame.
    ///
    /// Frames are pulled from `frames` only as results are consumed. Frame `k` of the sequence is
    /// video frame `frameIndex + k`. Cancelling the consuming task stops tracking before the next
    /// frame and ends the stream with `CancellationError`; the session stays valid, so tracking can
    /// resume later from any frame.
    ///
    /// - Parameters:
    ///   - frameIndex: video index of the first frame of `frames`.
    ///   - frameCount: total number of video frames, if known; it bounds how far back the tracker
    ///     looks for object pointers. The session remembers it for later calls.
    /// - Returns: a stream that throws ``SAM31Error/invalidPrompt(_:)`` when the session has no objects
    ///   or the indices are negative, and ``SAM31Error/invalidImage(_:)`` for an unreadable frame.
    public nonisolated func propagate<S: AsyncSequence & Sendable>(
        _ frames: S, startingAt frameIndex: Int, frameCount: Int?
    ) -> AsyncThrowingStream<TrackedFrame, Error> where S.Element == FrameInput {
        let cursor = FrameCursor(frames.makeAsyncIterator(), firstIndex: frameIndex)
        let (model, state) = (model, state)
        // Pull-based: each frame is tracked inside the consumer's `next()`, so cancelling the consumer
        // cancels the work and surfaces as `CancellationError` rather than a quiet end of stream.
        return AsyncThrowingStream(unfolding: {
            try Task.checkCancellation()
            guard let input = try await cursor.next() else { return nil }
            try Task.checkCancellation()
            let frame = try await model.sessionPropagate(
                state, input: input, frameIndex: cursor.takeIndex(), frameCount: frameCount)
            try Task.checkCancellation()
            return frame
        })
    }
}

/// Walks a frame sequence for ``TrackingSession/propagate(_:startingAt:frameCount:)``.
///
/// `@unchecked Sendable`: `AsyncThrowingStream(unfolding:)` calls its producer for one element at a
/// time, so the iterator is never used concurrently.
final class FrameCursor<Iterator: AsyncIteratorProtocol>: @unchecked Sendable
where Iterator.Element == FrameInput {
    private var iterator: Iterator
    private var index: Int

    init(_ iterator: Iterator, firstIndex: Int) {
        self.iterator = iterator
        self.index = firstIndex
    }

    func next() async throws -> FrameInput? {
        var it = iterator
        let input = try await it.next()
        iterator = it
        return input
    }

    /// The video index of the frame just read; advances the index.
    func takeIndex() -> Int {
        defer { index += 1 }
        return index
    }
}

/// A session's tracker state.
///
/// `@unchecked Sendable`: `pendingRemovals` is guarded by `lock`; every other property is read and
/// written only on the owning ``SAM31Model`` actor (the `session*` methods), which serializes all
/// MLX work.
final class SessionState: @unchecked Sendable {
    private let lock = NSLock()
    private var pendingRemovals: [Int] = []

    /// nil until the first object is added, and again after every object is removed.
    var tracker: MultiplexTrackerState?
    /// Total video length, once a `propagate` call has given it.
    var frameCount: Int?

    func requestRemoval(_ id: Int) {
        lock.withLock { pendingRemovals.append(id) }
    }

    func takePendingRemovals() -> [Int] {
        lock.withLock {
            defer { pendingRemovals = [] }
            return pendingRemovals
        }
    }
}

// MARK: - Session operations (run on the model actor)

extension SAM31Model {
    /// Adds object `id` (the "Add/refine mechanics" of the design spec).
    func sessionAddObject(
        _ s: SessionState, id: Int, frameIndex: Int, features: FrameFeatures, prompt: TrackPrompt
    ) throws {
        applyPendingRemovals(s)
        guard frameIndex >= 0 else { throw SAM31Error.invalidPrompt("frame index \(frameIndex) is negative") }
        // Validate everything before touching the state.
        var segmentPrompt: SegmentPrompt?
        var points: PointInputs?
        var maskInput: MLXArray?
        switch prompt {
        case .segment(let p):
            segmentPrompt = p
            points = try p.pointInputs()
        case .mask(let m):
            maskInput = modelResolutionBinaryMask(m)
        }

        let ff = trackerFeatures(features, interactive: true, propagation: true)
        guard let st = s.tracker else {
            // First object: this frame becomes a conditioning frame.
            let st = tracker.initState(numObjects: 1, objectIDs: [id])
            let out: FrameOutput
            if let points {
                out = tracker.trackStep(
                    st, frameIndex: frameIndex, isInitCondFrame: true, features: ff, pointInputs: points,
                    maskInputs: nil, numFrames: numFrames(s, frameIndex))
            } else {
                out = tracker.addMaskPrompt(
                    st, frameIndex: frameIndex, features: ff, masks: maskInput!, objectIDs: [id])
            }
            evalOutput(out)
            s.tracker = st
            return
        }

        let mux = st.multiplexState
        guard mux.availableSlots >= 1 else {
            throw SAM31Error.invalidPrompt(
                "a tracking session holds at most \(configuration.maxObjectsPerBucket) objects, and removed "
                    + "objects keep their slot until the session is empty; start another session")
        }
        // The frame's output must cover every current object; otherwise track the frame first.
        let prevOutput: FrameOutput
        if let current = currentOutput(st, frameIndex: frameIndex) {
            prevOutput = current
        } else {
            prevOutput = tracker.propagate(
                st, frameIndex: frameIndex, features: ff, numFrames: numFrames(s, frameIndex))
        }

        let newMask: MLXArray
        if let segmentPrompt {
            let heads = try interactiveHeads(features, prompt: segmentPrompt, previousLogits: nil)
            newMask = (heads.highResMasks .> 0).asType(.float32)
        } else {
            newMask = maskInput!
        }
        let interactive = ff.interactive!
        tracker.addNewMasksToExistingState(
            interactivePixFeat: tracker.getInteractivePixMem(interactive.visionFeat),
            interactiveHighResFeatures: interactive.highRes,
            propagationVisionFeat: ff.propagation!.visionFeat,
            newMasks: newMask, objIdxsInMask: [mux.totalValidEntries], objIDsInMask: [id],
            prevOutput: prevOutput,
            state: st, areMasksFromPts: segmentPrompt != nil)
        evalOutput(prevOutput)
    }

    /// Refines object `id` on a tracked frame: propagation-and-interaction mode, without previous
    /// mask logits (ruling R6: mlx-vlm's interaction-only mode supports only single-object sessions).
    func sessionRefine(
        _ s: SessionState, id: Int, frameIndex: Int, features: FrameFeatures, prompt: SegmentPrompt
    ) throws -> Mask {
        applyPendingRemovals(s)
        guard let st = s.tracker, let idx = st.multiplexState.objectIDs?.firstIndex(of: id) else {
            throw SAM31Error.unknownObject(id)
        }
        let points = try prompt.pointInputs()
        guard st.condFrameOutputs[frameIndex] != nil || st.nonCondFrameOutputs[frameIndex] != nil else {
            throw SAM31Error.invalidPrompt(
                "frame \(frameIndex) has not been tracked (or is no longer in the tracker's memory); "
                    + "propagate to it first")
        }
        let ff = trackerFeatures(features, interactive: true, propagation: true)
        let out = tracker.trackStep(
            st, frameIndex: frameIndex, isInitCondFrame: false, features: ff, pointInputs: points,
            maskInputs: nil,
            numFrames: numFrames(s, frameIndex), objectsToInteract: [idx])
        evalOutput(out)
        return Mask(logits: out.predMasks[idx, 0])
    }

    /// Propagates every object to one frame.
    func sessionPropagate(_ s: SessionState, input: FrameInput, frameIndex: Int, frameCount: Int?) throws
        -> TrackedFrame
    {
        applyPendingRemovals(s)
        guard frameIndex >= 0 else { throw SAM31Error.invalidPrompt("frame index \(frameIndex) is negative") }
        if let frameCount {
            guard frameCount > 0 else {
                throw SAM31Error.invalidPrompt("frame count \(frameCount) is not positive")
            }
            s.frameCount = frameCount
        }
        guard let st = s.tracker else {
            throw SAM31Error.invalidPrompt("the session has no objects; add one before propagating")
        }
        let frame = try features(for: input)
        let ff = trackerFeatures(frame, interactive: false, propagation: true)
        let out = tracker.propagate(
            st, frameIndex: frameIndex, features: ff, numFrames: numFrames(s, frameIndex))
        evalOutput(out)

        let logits = out.objectScoreLogits.asType(.float32).asArray(Float.self)
        var objects: [ObjectID: TrackedObject] = [:]
        for (i, id) in (st.multiplexState.objectIDs ?? []).enumerated() {
            objects[ObjectID(rawValue: id)] = TrackedObject(
                mask: Mask(logits: out.predMasks[i, 0]), score: sigmoidProbability(logits[i]),
                isVisible: logits[i] > 0)
        }
        return TrackedFrame(frameIndex: frameIndex, objects: objects)
    }

    /// `num_frames` for the tracker (ruling R8): the video length when known, else `frameIndex + 1`.
    /// Never below `frameIndex + 1`, and never below 2: `_get_tpos_enc` divides by
    /// `min(num_frames, 16) - 1`, which is NaN for a 1-frame video.
    private func numFrames(_ s: SessionState, _ frameIndex: Int) -> Int {
        max(s.frameCount ?? frameIndex + 1, frameIndex + 1, 2)
    }

    /// The stored output of `frameIndex` if it has a row for every current object (conditioning
    /// output first, as `add_mask_prompt` looks it up).
    private func currentOutput(_ st: MultiplexTrackerState, frameIndex: Int) -> FrameOutput? {
        [st.condFrameOutputs[frameIndex], st.nonCondFrameOutputs[frameIndex]].compactMap { $0 }
            .first { $0.predMasks.dim(0) == st.numObjects }
    }

    /// A mask as the tracker's `(1, 1, 1008, 1008)` binary float mask input.
    private func modelResolutionBinaryMask(_ mask: Mask) -> MLXArray {
        let side = configuration.imageSize
        let logits = resizeBilinearNHWC(mask.logits.reshaped(1, mask.height, mask.width, 1), h: side, w: side)
        return (logits .> 0).asType(.float32).reshaped(1, 1, side, side)
    }

    private func evalOutput(_ out: FrameOutput) {
        let arrays =
            [out.predMasks, out.predMasksHighRes, out.objectScoreLogits]
            + [out.objPtr, out.maskmemFeatures, out.maskmemPosEnc].compactMap { $0 }
        eval(arrays)
    }

    /// Applies ``TrackingSession/remove(_:)`` requests.
    ///
    /// The multiplex state marks the objects removed and renumbers the rest; their slots stay taken,
    /// so stored memories (bucket and slot space) stay aligned. Stored per-object rows (data space) are
    /// dropped to match. Removing every object resets the session.
    private func applyPendingRemovals(_ s: SessionState) {
        let removed = s.takePendingRemovals()
        guard !removed.isEmpty, let st = s.tracker, let objectIDs = st.multiplexState.objectIDs else {
            return
        }
        let indices = Set(removed.compactMap { objectIDs.firstIndex(of: $0) })
        guard !indices.isEmpty else { return }
        if indices.count == objectIDs.count {
            s.tracker = nil
            return
        }
        let kept = objectIDs.indices.filter { !indices.contains($0) }
        let newIndex = Dictionary(uniqueKeysWithValues: kept.enumerated().map { ($1, $0) })
        st.multiplexState.removeObjects(indices.sorted())

        for out in Array(st.condFrameOutputs.values) + Array(st.nonCondFrameOutputs.values) {
            let rows = out.predMasks.dim(0)
            let keep = MultiplexTrackerModel.indexArray(kept.filter { $0 < rows })
            out.predMasks = take(out.predMasks, keep, axis: 0)
            out.predMasksHighRes = take(out.predMasksHighRes, keep, axis: 0)
            out.objectScoreLogits = take(out.objectScoreLogits, keep, axis: 0)
            out.conditioningObjects = Set(out.conditioningObjects.compactMap { newIndex[$0] })
            eval(out.predMasks, out.predMasksHighRes, out.objectScoreLogits)
        }
    }
}
