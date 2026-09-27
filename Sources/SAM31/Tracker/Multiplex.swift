// Port of mlx_vlm/models/sam3_1/multiplex.py::{MultiplexState, MultiplexController,
// MultiplexTrackerState} (mlx-vlm 0.7.3)
//
// A multiplex state maps objects (data space, batch dim) to slots in buckets (multiplex space,
// numBuckets x multiplexCount). The mask decoder runs on whole buckets at once; mux/demux convert
// between the two spaces with precomputed gather indices, so they are exact copies.
import MLX

/// Marks an empty slot in a bucket.
let multiplexPaddingNum = -1
/// Marks a slot whose object was removed.
let multiplexRemovedNum = -1116

/// Records the assignment of each object to a (bucket, slot) pair. A reference type, as in Python:
/// ``addObjects(objectIndices:objectIDs:allowNewBuckets:preferNewBuckets:)`` and
/// ``removeObjects(_:strict:)`` mutate it in place.
final class MultiplexState {
    let allowedBucketCapacity: Int
    /// One list per bucket: object indices `0..<totalValidEntries`, ``multiplexPaddingNum`` for
    /// empty slots, or ``multiplexRemovedNum`` for removed objects. Empty once every object is removed.
    private(set) var assignments: [[Int]] = []
    private(set) var numBuckets = 0
    private(set) var multiplexCount = 0
    private(set) var totalValidEntries = 0
    private(set) var totalNonPaddingEntries = 0
    /// Optional bookkeeping of global object IDs, one per valid entry.
    private(set) var objectIDs: [Int]?

    private var slotGatherIdx = MLXArray.zeros([0], type: Int32.self)
    private var slotIsValid = MLXArray.zeros([0], type: Bool.self)
    private var objToSlot = MLXArray.zeros([0], type: Int32.self)

    init(assignments: [[Int]], allowedBucketCapacity: Int, objectIDs: [Int]? = nil) {
        self.allowedBucketCapacity = allowedBucketCapacity
        initializeAssignments(assignments, objectIDs: objectIDs)
    }

    private func initializeAssignments(_ assignments: [[Int]], objectIDs: [Int]?) {
        self.assignments = assignments
        numBuckets = assignments.count
        precondition(numBuckets > 0, "No buckets found in the state")
        multiplexCount = assignments[0].count
        precondition(assignments.allSatisfy { $0.count == multiplexCount })

        let flat = assignments.flatMap { $0 }
        totalValidEntries = flat.filter { $0 >= 0 }.count
        totalNonPaddingEntries = flat.filter { $0 != multiplexPaddingNum }.count

        self.objectIDs = objectIDs
        if let objectIDs { precondition(objectIDs.count == totalValidEntries) }

        // Gather indices instead of permutation matrices, so mux/demux are exact copies.
        let slotToObj = flat.map { max($0, multiplexPaddingNum) }
        var objToSlot = [Int32](repeating: 0, count: totalValidEntries)
        for (slot, objIdx) in slotToObj.enumerated() where objIdx >= 0 {
            objToSlot[objIdx] = Int32(slot)
        }
        let slotToObjArray = MLXArray(slotToObj.map(Int32.init))
        slotGatherIdx = maximum(slotToObjArray, 0)
        slotIsValid = greaterEqual(slotToObjArray, 0)
        self.objToSlot = MLXArray(objToSlot)
    }

    var availableSlots: Int {
        numBuckets * allowedBucketCapacity - totalNonPaddingEntries
    }

    /// The next consecutive object indices available in the state.
    func findNextBatchOfAvailableIndices(
        numObjects: Int, allowNewBuckets: Bool = false, preferNewBuckets: Bool = false
    ) -> [Int] {
        precondition(numObjects > 0)
        if !allowNewBuckets {
            precondition(
                availableSlots >= numObjects, "not enough available slots \(availableSlots) < \(numObjects)")
        }
        return Array(totalValidEntries..<(totalValidEntries + numObjects))
    }

    /// Adds new objects, filling empty slots first, then new buckets.
    func addObjects(
        objectIndices: [Int], objectIDs newIDs: [Int]? = nil, allowNewBuckets: Bool = false,
        preferNewBuckets: Bool = false
    ) {
        let numNewObjects = objectIndices.count
        if numNewObjects == 0 { return }
        precondition(objectIndices == objectIndices.sorted())
        precondition((newIDs == nil) == (objectIDs == nil))
        if let newIDs { precondition(newIDs.count == numNewObjects) }
        if preferNewBuckets { precondition(allowNewBuckets) }

        var pending = Array(zip(objectIndices, newIDs ?? [Int](repeating: 0, count: numNewObjects)))
        var ids = objectIDs
        func place(_ bucket: inout [Int], _ i: Int) {
            let (objIdx, objID) = pending.removeFirst()
            bucket[i] = objIdx
            if newIDs != nil { ids?.append(objID) }
        }

        if !preferNewBuckets {
            // Fill empty slots in existing buckets first
            for b in assignments.indices {
                for i in 0..<allowedBucketCapacity
                where !pending.isEmpty && assignments[b][i] == multiplexPaddingNum {
                    place(&assignments[b], i)
                }
                if pending.isEmpty { break }
            }
        }

        precondition(
            pending.isEmpty || allowNewBuckets,
            "Cannot place objects \(pending.map(\.0)) without creating new buckets")

        // Create new buckets for remaining objects
        while !pending.isEmpty {
            var newBucket = [Int](repeating: multiplexPaddingNum, count: multiplexCount)
            for i in 0..<allowedBucketCapacity {
                if pending.isEmpty { break }
                place(&newBucket, i)
            }
            assignments.append(newBucket)
        }

        let originalNumEntries = totalValidEntries
        initializeAssignments(assignments, objectIDs: ids)
        precondition(totalValidEntries == originalNumEntries + numNewObjects)
    }

    /// Marks objects as removed and drops buckets that become empty; the surviving object indices are
    /// then renumbered to be sequential.
    ///
    /// - Returns: the indices of the buckets that are kept.
    @discardableResult
    func removeObjects(_ objectIndices: [Int], strict: Bool = true) -> [Int] {
        var remaining = Set(objectIndices)
        for b in assignments.indices {
            for slot in assignments[b].indices where remaining.contains(assignments[b][slot]) {
                remaining.remove(assignments[b][slot])
                assignments[b][slot] = multiplexRemovedNum
            }
        }
        if strict { precondition(remaining.isEmpty, "Failed to remove objects: \(remaining.sorted())") }

        // A bucket is dead once it holds no valid (non-negative) objects
        let bucketsToKeep = assignments.indices.filter { assignments[$0].contains { $0 >= 0 } }
        assignments = bucketsToKeep.map { assignments[$0] }

        if bucketsToKeep.isEmpty {
            // Python sets `assignments = None` and leaves the derived counts as they were.
            if objectIDs != nil { objectIDs = [] }
            return bucketsToKeep
        }

        // Remap remaining object indices to be sequential
        let allPositiveIDs = Set(assignments.flatMap { $0 }.filter { $0 >= 0 }).sorted()
        let idMapping = Dictionary(uniqueKeysWithValues: allPositiveIDs.enumerated().map { ($1, $0) })
        for b in assignments.indices {
            for i in assignments[b].indices where assignments[b][i] >= 0 {
                assignments[b][i] = idMapping[assignments[b][i]]!
            }
        }
        let ids = objectIDs.map { old in allPositiveIDs.map { old[$0] } }
        initializeAssignments(assignments, objectIDs: ids)
        return bucketsToKeep
    }

    /// `(totalValidEntries, ...)` -> `(numBuckets, multiplexCount, ...)`; padding slots are zero.
    func mux(_ x: MLXArray) -> MLXArray {
        let numValid = x.dim(0)
        precondition(
            numValid == totalValidEntries, "numValid=\(numValid) != totalValidEntries=\(totalValidEntries)")
        var result = take(x.reshaped(numValid, -1), slotGatherIdx, axis: 0)
        result = MLX.where(slotIsValid[0..., .newAxis], result, 0)
        return result.reshaped([numBuckets, multiplexCount] + x.shape.dropFirst())
    }

    /// `(numBuckets, multiplexCount, ...)` -> `(totalValidEntries, ...)`.
    func demux(_ x: MLXArray) -> MLXArray {
        let (nb, mc) = (x.dim(0), x.dim(1))
        precondition(nb == numBuckets && mc == multiplexCount)
        let result = take(x.reshaped(nb * mc, -1), objToSlot, axis: 0)
        return result.reshaped([totalValidEntries] + x.shape.dropFirst(2))
    }

    /// `(numBuckets, multiplexCount)` bool mask of valid (non-padding) slots.
    func getValidObjectMask() -> MLXArray {
        slotIsValid.reshaped(numBuckets, multiplexCount)
    }

    /// All valid internal object indices in the state.
    func getAllValidObjectIdx() -> Set<Int> {
        Set(assignments.flatMap { $0 }.filter { $0 >= 0 })
    }
}

/// Creates multiplex states by bucketing objects (inference: eval capacity).
struct MultiplexController {
    let multiplexCount: Int
    let evalMultiplexCount: Int

    init(multiplexCount: Int, evalMultiplexCount: Int = -1) {
        precondition(multiplexCount >= 1)
        self.multiplexCount = multiplexCount
        self.evalMultiplexCount = evalMultiplexCount < 0 ? multiplexCount : evalMultiplexCount
    }

    var allowedBucketCapacity: Int { evalMultiplexCount }

    /// Maps `numValidEntries` objects into buckets of size `multiplexCount`.
    ///
    /// `random` shuffles the order with Swift's generator rather than `mx.random.permutation`, so it
    /// does not reproduce Python's order; the tracker always passes `false`.
    func getState(numValidEntries: Int, random: Bool = false, objectIDs: [Int]? = nil) -> MultiplexState {
        let capacity = allowedBucketCapacity
        let numBuckets = (numValidEntries + capacity - 1) / capacity
        var ids = random ? Array(0..<numValidEntries).shuffled() : Array(0..<numValidEntries)
        ids += [Int](repeating: multiplexPaddingNum, count: numBuckets * capacity - ids.count)
        let assignments = (0..<numBuckets).map { i in
            Array(ids[(i * capacity)..<((i + 1) * capacity)])
                + [Int](repeating: multiplexPaddingNum, count: multiplexCount - capacity)
        }
        return MultiplexState(assignments: assignments, allowedBucketCapacity: capacity, objectIDs: objectIDs)
    }
}

/// Inference state of a multiplex video tracking session: the bucket assignment and the per-frame
/// outputs that serve as memory for later frames. A reference type, as in Python.
final class MultiplexTrackerState {
    let multiplexState: MultiplexState
    var condFrameOutputs: [Int: FrameOutput] = [:]
    var nonCondFrameOutputs: [Int: FrameOutput] = [:]

    init(_ multiplexState: MultiplexState) {
        self.multiplexState = multiplexState
    }

    var numObjects: Int { multiplexState.totalValidEntries }
}
