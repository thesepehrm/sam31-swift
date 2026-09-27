// Port of mlx_vlm/models/sam3/generate.py::{_box_iou, nms} (mlx-vlm 0.7.3)

/// IoU of two xyxy boxes, with continuous coordinates (no +1 pixel convention) and the union
/// floored at 1e-6, as in Python's `_box_iou`.
func boxIoU(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Float {
    let x1 = max(a[0], b[0])
    let y1 = max(a[1], b[1])
    let x2 = min(a[2], b[2])
    let y2 = min(a[3], b[3])
    let inter = max(0, x2 - x1) * max(0, y2 - y1)
    let a1 = (a[2] - a[0]) * (a[3] - a[1])
    let a2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(a1 + a2 - inter, 1e-6)
}

/// Greedy box NMS, as in Python's `nms`: visit boxes by descending score and drop any whose IoU
/// with an already-kept box is above `iouThreshold`. Python's `nms` works on boxes only (masks are
/// just carried along) and defaults to a 0.5 threshold.
///
/// Python orders with `np.argsort(-scores)`, which does not define the order of tied scores; this
/// port breaks ties by the lower index.
///
/// - Returns: indices into `boxes` of the kept detections, highest score first.
func nmsIndices(boxes: [SIMD4<Float>], scores: [Float], iouThreshold: Float = 0.5) -> [Int] {
    precondition(boxes.count == scores.count, "boxes and scores must have the same count")
    let order = scores.indices.sorted { scores[$0] > scores[$1] || (scores[$0] == scores[$1] && $0 < $1) }
    var keep: [Int] = []
    for i in order where !keep.contains(where: { boxIoU(boxes[i], boxes[$0]) > iouThreshold }) {
        keep.append(i)
    }
    return keep
}
