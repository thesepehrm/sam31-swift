import Testing

@testable import SAM31

@Suite struct NMSTests {
    /// Python's `_box_iou` uses continuous coordinates (no +1 pixel convention), so the
    /// hand-computed values below are exactly what mlx-vlm produces.
    @Test func iou() {
        #expect(boxIoU(.init(0, 0, 10, 10), .init(0, 0, 10, 10)) == 1)
        #expect(boxIoU(.init(0, 0, 10, 10), .init(20, 20, 30, 30)) == 0)
        #expect(abs(boxIoU(.init(0, 0, 10, 10), .init(5, 0, 15, 10)) - 1.0 / 3.0) < 1e-6)
    }

    @Test func suppressesOverlapKeepsHighestScore() {
        let keep = nmsIndices(
            boxes: [.init(0, 0, 10, 10), .init(1, 0, 11, 10), .init(50, 50, 60, 60)],
            scores: [0.8, 0.9, 0.5], iouThreshold: 0.5)
        #expect(keep == [1, 2])
    }

    @Test func degenerateBoxesDoNotDivideByZero() {
        #expect(boxIoU(.init(3, 3, 3, 3), .init(3, 3, 3, 3)) == 0)
        #expect(nmsIndices(boxes: [], scores: [], iouThreshold: 0.5).isEmpty)
    }
}
