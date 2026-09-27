import MLX
import Testing

@testable import SAM31

/// Unit tests for the tracker's memory helpers. Expected values come from running
/// `sam3_1/tracker.py::{select_closest_cond_frames, get_1d_sine_pe}` in `.venv`.
struct TrackerMemoryTests {
    static func outputs(_ frames: [Int]) -> [Int: FrameOutput] {
        Dictionary(
            uniqueKeysWithValues: frames.map {
                (
                    $0,
                    FrameOutput(
                        conditioningObjects: [], pointInputs: nil, maskInputs: nil, predMasks: MLXArray(0),
                        predMasksHighRes: MLXArray(0), objectScoreLogits: MLXArray(0))
                )
            })
    }

    @Test(arguments: [
        (25, 4, [20, 30, 10, 40], [0, 50]),
        (60, 3, [50, 40, 30], [0, 10, 20]),
        (25, -1, [0, 10, 20, 30, 40, 50], []),
    ])
    func selectClosestCondFramesMatchesPython(frame: Int, maxNum: Int, selected: [Int], unselected: [Int]) {
        let (s, u) = selectClosestCondFrames(
            frameIndex: frame, condFrameOutputs: Self.outputs([0, 10, 20, 30, 40, 50]),
            maxCondFrameNum: maxNum)
        #expect(s.map(\.0) == selected)
        #expect(u.keys.sorted() == unselected)
    }

    @Test func oneDSinePEMatchesPython() {
        let pe = get1DSinePE(MLXArray([0, 1.0 / 9, 5.0 / 9] as [Float]), dim: 8)
        let expected: [Float] = [
            0, 0, 0, 0,
            1, 1, 1, 1,
            0.11088263, 0.11088263, 0.0011111108, 0.0011111108,
            0.99383348, 0.99383348, 0.99999946, 0.99999946,
            0.52741539, 0.52741539, 0.0055555273, 0.0055555273,
            0.84960753, 0.84960753, 0.99998456, 0.99998456,
        ]
        #expect(pe.shape == [3, 8])
        #expect(zip(pe.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 1e-6 })
    }
}
