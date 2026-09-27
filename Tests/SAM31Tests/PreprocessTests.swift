import CoreGraphics
import CoreVideo
import CryptoKit
import Foundation
import MLX
import Testing

@testable import SAM31

@Suite struct PreprocessTests {
    /// One case of resize-golden.json, written by parity/make_resize_golden.py with Pillow 12.3.0.
    /// The input is `noise(count: width * height * 3, seed: seed)`; the expected Pillow output is
    /// stored raw (base64) when small and as the SHA-256 of its bytes otherwise.
    struct GoldenCase: Decodable, Sendable, CustomTestStringConvertible {
        let name: String
        let width: Int
        let height: Int
        let outWidth: Int
        let outHeight: Int
        let seed: UInt64
        let output: String?
        let sha256: String?
        var testDescription: String { name }
    }

    struct Golden: Decodable {
        let cases: [GoldenCase]
    }

    static let goldenCases: [GoldenCase] = {
        let url = Bundle.module.url(forResource: "Resources/resize-golden", withExtension: "json")!
        return try! JSONDecoder().decode(Golden.self, from: Data(contentsOf: url)).cases
    }()

    /// Covers 1008x1008 from 37x23, 1920x1080 and 640x360, plus 1xN, Nx1, the >100x-tall
    /// vertical-first rule, identity, upscale, non-square downscale and a width-only resize.
    @Test(arguments: goldenCases) func resizeMatchesPILExactly(_ golden: GoldenCase) throws {
        let input = Self.noise(count: golden.width * golden.height * 3, seed: golden.seed)
        let out = pillowResizeRGB(
            input, width: golden.width, height: golden.height,
            outWidth: golden.outWidth, outHeight: golden.outHeight)
        #expect(out.count == golden.outWidth * golden.outHeight * 3)
        if let raw = golden.output {
            let expected = [UInt8](try #require(Data(base64Encoded: raw)))
            #expect(out.count == expected.count)
            let diff = zip(out, expected).map { abs(Int($0) - Int($1)) }.max() ?? 0
            #expect(diff == 0, "max diff \(diff)")
        } else {
            let hash = SHA256.hash(data: Data(out)).map { String(format: "%02x", $0) }.joined()
            #expect(hash == golden.sha256)
        }
    }

    @Test func normalizesToMinusOneOne() {
        let pre = ImagePreprocessor()
        let white = MLXArray([UInt8](repeating: 255, count: 4 * 4 * 3)).reshaped(4, 4, 3)
        let pv = pre.pixelValues(fromRGB: white)
        #expect(pv.shape == [1, 1008, 1008, 3])
        #expect(abs(pv.max().item(Float.self) - 1) < 1e-6)
        let black = MLXArray([UInt8](repeating: 0, count: 4 * 4 * 3)).reshaped(4, 4, 3)
        #expect(abs(pre.pixelValues(fromRGB: black).min().item(Float.self) + 1) < 1e-6)
    }

    @Test func cgImageMatchesRawRGB() throws {
        let (w, h) = (40, 30)
        let rgb = Self.noise(count: w * h * 3)
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0..<(w * h) { for c in 0..<3 { rgba[i * 4 + c] = rgb[i * 3 + c] } }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let pre = ImagePreprocessor()
        let expected = pre.pixelValues(fromRGB: MLXArray(rgb, [h, w, 3]))
        #expect(arrayEqual(try pre.pixelValues(from: image), expected).item(Bool.self))
    }

    @Test func bgraPixelBufferMatchesRawRGB() throws {
        let (w, h) = (50, 34)
        let rgb = Self.noise(count: w * h * 3)
        let buffer = try Self.makeBuffer(w, h, kCVPixelFormatType_32BGRA) { plane, base, bytesPerRow in
            for y in 0..<h {
                for x in 0..<w {
                    let s = (y * w + x) * 3
                    let d = y * bytesPerRow + x * 4
                    base[d] = rgb[s + 2]
                    base[d + 1] = rgb[s + 1]
                    base[d + 2] = rgb[s]
                    base[d + 3] = 255
                }
            }
        }
        let pre = ImagePreprocessor()
        let expected = pre.pixelValues(fromRGB: MLXArray(rgb, [h, w, 3]))
        #expect(arrayEqual(try pre.pixelValues(from: buffer), expected).item(Bool.self))
    }

    /// A flat 4:2:0 frame whose Y'CbCr is the BT.709 encoding of RGB (200, 100, 50), quantized to 8
    /// bits. Quantizing Y'CbCr costs up to about 1.5 RGB levels and vImage rounds once more, so the
    /// decoded color must be within 3/255 per channel (6/255 in [-1, 1] units).
    @Test(arguments: [true, false]) func yuv420PixelBufferDecodesBT709(fullRange: Bool) throws {
        let (w, h) = (64, 48)
        let (r, g, b) = (200.0, 100.0, 50.0)
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        let (cb, cr) = ((b - y) / 1.8556, (r - y) / 1.5748)
        let (yCode, cbCode, crCode) =
            fullRange
            ? (y, 128 + cb, 128 + cr)
            : (16 + y * 219 / 255, 128 + cb * 224 / 255, 128 + cr * 224 / 255)
        let format =
            fullRange
            ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let buffer = try Self.makeBuffer(w, h, format) { plane, base, bytesPerRow in
            if plane == 0 {
                for row in 0..<h { for x in 0..<w { base[row * bytesPerRow + x] = UInt8(yCode.rounded()) } }
            } else {
                for row in 0..<(h / 2) {
                    for x in 0..<(w / 2) {
                        base[row * bytesPerRow + 2 * x] = UInt8(cbCode.rounded())
                        base[row * bytesPerRow + 2 * x + 1] = UInt8(crCode.rounded())
                    }
                }
            }
        }
        let pv = try ImagePreprocessor().pixelValues(from: buffer)
        #expect(pv.shape == [1, 1008, 1008, 3])
        let expected = MLXArray([r, g, b].map { Float($0 / 255 - 0.5) / 0.5 })
        let err = abs(pv - expected).max().item(Float.self)
        #expect(err <= 6 / 255, "max error \(err * 255 / 2) RGB levels")
    }

    @Test func unsupportedPixelFormatThrows() throws {
        let buffer = try Self.makeBuffer(8, 8, kCVPixelFormatType_32ARGB) { _, _, _ in }
        #expect(throws: SAM31Error.invalidImage("unsupported pixel format 32")) {
            try ImagePreprocessor().pixelValues(from: buffer)
        }
    }

    /// Deterministic bytes from a 64-bit LCG, mirrored by `noise` in parity/make_resize_golden.py.
    static func noise(count: Int, seed: UInt64 = 0) -> [UInt8] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15 ^ seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 56)
        }
    }

    /// Creates a pixel buffer and fills each plane (or the single plane) with `fill(plane, base, bytesPerRow)`.
    static func makeBuffer(
        _ w: Int, _ h: Int, _ format: OSType,
        fill: (Int, UnsafeMutablePointer<UInt8>, Int) -> Void
    ) throws -> CVPixelBuffer {
        var out: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, w, h, format, nil, &out)
        try #require(status == kCVReturnSuccess)
        let buffer = out!
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        if CVPixelBufferIsPlanar(buffer) {
            for plane in 0..<CVPixelBufferGetPlaneCount(buffer) {
                let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!.assumingMemoryBound(
                    to: UInt8.self)
                fill(plane, base, CVPixelBufferGetBytesPerRowOfPlane(buffer, plane))
            }
        } else {
            let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            fill(0, base, CVPixelBufferGetBytesPerRow(buffer))
        }
        return buffer
    }
}
