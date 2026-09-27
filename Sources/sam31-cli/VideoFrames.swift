import AVFoundation
import CoreVideo
import Foundation
import SAM31

/// Decoded frames of a video file as `FrameInput.pixelBuffer` (32BGRA), read with `AVAssetReader`.
///
/// Frames come in decode order, starting at frame `start`, and stop after `limit` frames when set.
/// The track's preferred transform (rotation) is not applied.
struct VideoFrames: AsyncSequence, Sendable {
    typealias Element = FrameInput

    let url: URL
    var start = 0
    var limit: Int?

    func makeAsyncIterator() -> Iterator {
        Iterator(url: url, skip: start, remaining: limit)
    }

    struct Iterator: AsyncIteratorProtocol {
        let url: URL
        var skip: Int
        var remaining: Int?
        private var reader: Reader?

        init(url: URL, skip: Int, remaining: Int?) {
            self.url = url
            self.skip = skip
            self.remaining = remaining
        }

        mutating func next() async throws -> FrameInput? {
            if let remaining, remaining <= 0 { return nil }
            if reader == nil { reader = try await Reader(url: url) }
            guard let reader else { return nil }
            while skip > 0 {
                guard reader.nextPixelBuffer() != nil else { return nil }
                skip -= 1
            }
            guard let buffer = reader.nextPixelBuffer() else {
                try reader.checkFailed()
                return nil
            }
            remaining = remaining.map { $0 - 1 }
            return .pixelBuffer(buffer)
        }
    }

    /// The asset reader of one pass over the video track.
    private final class Reader {
        let reader: AVAssetReader
        let output: AVAssetReaderTrackOutput
        let url: URL

        init(url: URL) async throws {
            self.url = url
            let asset = AVURLAsset(url: url)
            let tracks: [AVAssetTrack]
            do {
                tracks = try await asset.loadTracks(withMediaType: .video)
            } catch {
                throw CLIError("cannot open video \(url.path): \(error.localizedDescription)")
            }
            guard let track = tracks.first else { throw CLIError("\(url.path) has no video track") }
            do {
                reader = try AVAssetReader(asset: asset)
            } catch {
                throw CLIError("cannot read video \(url.path): \(error.localizedDescription)")
            }
            output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            reader.add(output)
            guard reader.startReading() else {
                throw CLIError(
                    "cannot read video \(url.path): \(reader.error?.localizedDescription ?? "unknown error")")
            }
        }

        func nextPixelBuffer() -> CVPixelBuffer? {
            // Skip samples without an image (for example empty edit markers).
            while let sample = output.copyNextSampleBuffer() {
                if let buffer = CMSampleBufferGetImageBuffer(sample) { return buffer }
            }
            return nil
        }

        func checkFailed() throws {
            if reader.status == .failed {
                throw CLIError(
                    "cannot decode \(url.path): \(reader.error?.localizedDescription ?? "unknown error")")
            }
        }
    }
}

/// Frames already in memory, as a sequence `TrackingSession.propagate` can consume.
struct FrameList: AsyncSequence, Sendable {
    typealias Element = FrameInput
    let frames: [FrameInput]

    func makeAsyncIterator() -> Iterator { Iterator(frames: frames) }

    struct Iterator: AsyncIteratorProtocol {
        let frames: [FrameInput]
        var index = 0

        mutating func next() async -> FrameInput? {
            guard index < frames.count else { return nil }
            defer { index += 1 }
            return frames[index]
        }
    }
}

extension SAM31Model {
    /// Encodes a decoded video frame.
    func encode(frame input: FrameInput) async throws -> FrameFeatures {
        guard case .pixelBuffer(let buffer) = input else { throw CLIError("expected a decoded video frame") }
        return try encode(buffer)
    }
}
