import MLX
import Testing

@testable import SAM31

/// Expected values come from running the same calls on mlx-vlm 0.7.3's `sam3_1/multiplex.py` in
/// `.venv` (the brief's values were guesses): `remove_objects` compacts the surviving indices, so
/// removing object 0 of {0, 1, 2} leaves {0, 1}, not {1, 2}.
struct MultiplexStateTests {
    @Test func allocatesSlotsAndRemoves() {
        let c = MultiplexController(multiplexCount: 16)
        let s = c.getState(numValidEntries: 2, random: false, objectIDs: [10, 11])
        #expect(s.totalValidEntries == 2)
        #expect(s.numBuckets == 1)
        #expect(s.availableSlots == 14)
        #expect(s.getAllValidObjectIdx() == [0, 1])
        let next = s.findNextBatchOfAvailableIndices(
            numObjects: 1, allowNewBuckets: false, preferNewBuckets: false)
        #expect(next == [2])
        s.addObjects(objectIndices: next, objectIDs: [12], allowNewBuckets: false, preferNewBuckets: false)
        #expect(Array(s.assignments[0].prefix(4)) == [0, 1, 2, -1])
        #expect(s.objectIDs == [10, 11, 12])
        #expect(s.removeObjects([0], strict: true) == [0])
        #expect(s.getAllValidObjectIdx() == [0, 1])
        #expect(Array(s.assignments[0].prefix(4)) == [-1116, 0, 1, -1])
        #expect(s.objectIDs == [11, 12])
        #expect(s.totalValidEntries == 2)
        #expect(s.totalNonPaddingEntries == 3)
        #expect(s.availableSlots == 13)
    }

    @Test func muxDemuxRoundTrip() {
        let s = MultiplexController(multiplexCount: 16).getState(numValidEntries: 3)
        _ = s.removeObjects([0])
        let x = MLXArray(0..<6).reshaped(2, 3).asType(.float32)
        let m = s.mux(x)
        #expect(m.shape == [1, 16, 3])
        #expect(m[0, 0..<4].asArray(Float.self) == [0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0])
        #expect(s.demux(m).asArray(Float.self) == [0, 1, 2, 3, 4, 5])
        #expect(s.getValidObjectMask()[0, 0..<4].asArray(Bool.self) == [false, true, true, false])
    }

    @Test func newBucketsAndBucketRemoval() {
        let c = MultiplexController(multiplexCount: 16)
        let s = c.getState(numValidEntries: 18)
        #expect(s.numBuckets == 2)
        #expect(Array(s.assignments[1].prefix(4)) == [16, 17, -1, -1])
        s.addObjects(objectIndices: [18], allowNewBuckets: true, preferNewBuckets: true)
        #expect(s.numBuckets == 3)
        #expect(Array(s.assignments[2].prefix(3)) == [18, -1, -1])
        #expect(s.removeObjects([16, 17]) == [0, 2])
        #expect(s.numBuckets == 2)
        #expect(Array(s.assignments[1].prefix(3)) == [16, -1, -1])
        #expect(s.totalValidEntries == 17)
    }

    @Test func trackerStateWrapsMultiplexState() {
        let st = MultiplexTrackerState(MultiplexController(multiplexCount: 16).getState(numValidEntries: 1))
        #expect(st.numObjects == 1)
        #expect(st.condFrameOutputs.isEmpty && st.nonCondFrameOutputs.isEmpty)
    }
}
