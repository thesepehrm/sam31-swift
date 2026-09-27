import CoreGraphics
import Foundation
import MLX
import Testing

@testable import SAM31

@Suite struct PointMapperTests {
    @Test func mapsSourceToModelSpace() {
        let m = PointMapper(sourceSize: CGSize(width: 1920, height: 1080))
        #expect(m.toModel(CGPoint(x: 960, y: 540)) == CGPoint(x: 504, y: 504))
        #expect(
            m.toSource(CGRect(x: 0, y: 0, width: 1008, height: 1008))
                == CGRect(x: 0, y: 0, width: 1920, height: 1080))
        #expect(
            m.toModel(CGRect(x: 0, y: 0, width: 1920, height: 1080))
                == CGRect(x: 0, y: 0, width: 1008, height: 1008))
    }
}

@Suite struct PromptValidationTests {
    @Test func boxBecomesCornerPointsBeforeClicks() throws {
        let p = SegmentPrompt.boxAndPoints(
            CGRect(x: 10, y: 20, width: 30, height: 40), [.init(x: 5, y: 6, label: .negative)])
        let (coords, labels) = try p.pointList()
        #expect(coords == [10, 20, 40, 60, 5, 6])
        #expect(labels == [2, 3, 0])
        let inputs = try p.pointInputs()
        #expect(inputs.coords.shape == [1, 3, 2])
        #expect(inputs.labels.shape == [1, 3])
    }

    @Test func rejectsMalformedPrompts() {
        let bad: [SegmentPrompt] = [
            .points([]),
            .points([.init(x: .nan, y: 1, label: .positive)]),
            .box(CGRect(x: 0, y: 0, width: 0, height: 10)),
            .box(CGRect(x: 0, y: 0, width: CGFloat.infinity, height: 10)),
        ]
        for prompt in bad {
            #expect(throws: SAM31Error.self) { try prompt.pointList() }
        }
    }

    @Test func loadThrowsForMissingWeights() async {
        let dir = URL(fileURLWithPath: "/nonexistent/sam31-weights")
        await #expect(throws: SAM31Error.weightsNotFound(dir)) { try await SAM31Model.load(from: dir) }
    }
}
