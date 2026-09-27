// Port of mlx_vlm/models/sam3/processing_sam3.py::Sam3Processor._process_single_image (mlx-vlm 0.7.3),
// with Pillow 12.3.0's `Image.resize(BILINEAR)` for RGB (PIL/Image.py::Image.resize and
// libImaging/Resample.c::{ImagingResampleInner, ImagingResampleHorizontal_8bpc,
// ImagingResampleVertical_8bpc}).
import Accelerate
import CoreGraphics
import CoreVideo
import Foundation
import MLX

/// Turns an RGB image into the `(1, 1008, 1008, 3)` float32 pixel values SAM 3.1 expects: Pillow
/// BILINEAR resize to 1008x1008 on uint8, then `x / 255`, then `(x - 0.5) / 0.5`, all in float32.
struct ImagePreprocessor: Sendable {
    let size = 1008

    /// Resizes packed `(height, width, 3)` RGB bytes to `size`x`size`, byte-identical to Pillow.
    func resizeRGB(_ rgb: [UInt8], width: Int, height: Int) -> [UInt8] {
        pillowResizeRGB(rgb, width: width, height: height, outWidth: size, outHeight: size)
    }

    /// `(H, W, 3)` uint8 RGB to `(1, 1008, 1008, 3)` float32 pixel values.
    func pixelValues(fromRGB rgb: MLXArray) -> MLXArray {
        precondition(rgb.ndim == 3 && rgb.dim(2) == 3, "expected (H, W, 3) RGB, got \(rgb.shape)")
        return pixelValues(rgb: rgb.asType(.uint8).asArray(UInt8.self), width: rgb.dim(1), height: rgb.dim(0))
    }

    /// Draws `image` into an 8-bit RGBA context and drops alpha.
    ///
    /// Pillow ignores embedded color profiles, so an 8-bit RGB image is drawn in its own color space
    /// (no conversion; the bytes pass through unchanged) and matches Python exactly. If no context
    /// can be made in that space, the image is drawn in sRGB instead. Gray, CMYK and other non-RGB
    /// images, and 16-bit or float images, go through CoreGraphics color and depth conversion and are
    /// not byte-identical to PIL's `convert("RGB")`. Partly transparent pixels are un-premultiplied.
    func pixelValues(from image: CGImage) throws -> MLXArray {
        let (w, h) = (image.width, image.height)
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        let own = image.colorSpace.flatMap { $0.model == .rgb && $0.supportsOutput ? $0 : nil }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { raw -> Bool in
            let make = { (space: CGColorSpace) in
                CGContext(
                    data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                    space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            }
            guard let ctx = own.flatMap(make) ?? make(srgb) else { return false }
            ctx.interpolationQuality = .none
            ctx.setBlendMode(.copy)
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw SAM31Error.invalidImage("cannot draw a \(w)x\(h) CGImage into RGBA8") }

        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            let a = Int(rgba[i * 4 + 3])
            for c in 0..<3 {
                let v = Int(rgba[i * 4 + c])
                rgb[i * 3 + c] = a == 255 || a == 0 ? UInt8(v) : UInt8(min(255, (v * 255 + a / 2) / a))
            }
        }
        return pixelValues(rgb: rgb, width: w, height: h)
    }

    /// Reads a `32BGRA` buffer directly, or converts a 4:2:0 bi-planar Y'CbCr buffer (`420f` full
    /// range or `420v` video range) to RGB. Y'CbCr is always decoded with the BT.709 matrix, whatever
    /// the buffer's color attachments say, which is right for HD video but not for BT.601 SD or BT.2020
    /// sources. Other pixel formats throw `SAM31Error.invalidImage`.
    func pixelValues(from buffer: CVPixelBuffer) throws -> MLXArray {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let (w, h) = (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        switch format {
        case kCVPixelFormatType_32BGRA:
            guard let base = CVPixelBufferGetBaseAddress(buffer) else {
                throw SAM31Error.invalidImage("pixel buffer has no base address")
            }
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            let src = base.assumingMemoryBound(to: UInt8.self)
            var rgb = [UInt8](repeating: 0, count: w * h * 3)
            for y in 0..<h {
                for x in 0..<w {
                    let s = y * rowBytes + x * 4
                    let d = (y * w + x) * 3
                    rgb[d] = src[s + 2]
                    rgb[d + 1] = src[s + 1]
                    rgb[d + 2] = src[s]
                }
            }
            return pixelValues(rgb: rgb, width: w, height: h)
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            let rgba = try Self.rgbaFromYpCbCr420(
                buffer, width: w, height: h,
                fullRange: format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
            var rgb = [UInt8](repeating: 0, count: w * h * 3)
            for i in 0..<(w * h) {
                rgb[i * 3] = rgba[i * 4]
                rgb[i * 3 + 1] = rgba[i * 4 + 1]
                rgb[i * 3 + 2] = rgba[i * 4 + 2]
            }
            return pixelValues(rgb: rgb, width: w, height: h)
        default:
            throw SAM31Error.invalidImage("unsupported pixel format \(Self.fourCC(format))")
        }
    }

    /// Resize, then `astype(float32) / 255.0` and `(x - 0.5) / 0.5` in float32, as numpy does.
    func pixelValues(rgb: [UInt8], width: Int, height: Int) -> MLXArray {
        let resized = resizeRGB(rgb, width: width, height: height)
        let lut: [Float] = (0..<256).map { (Float($0) / 255 - 0.5) / 0.5 }
        return MLXArray(resized.map { lut[Int($0)] }, [1, size, size, 3])
    }

    /// BT.709 conversion of a locked 4:2:0 bi-planar buffer to RGBA8 (alpha 255).
    private static func rgbaFromYpCbCr420(
        _ buffer: CVPixelBuffer, width w: Int, height h: Int, fullRange: Bool
    ) throws -> [UInt8] {
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
            let cBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)
        else { throw SAM31Error.invalidImage("pixel buffer has no plane base addresses") }

        var range =
            fullRange
            ? vImage_YpCbCrPixelRange(
                Yp_bias: 0, CbCr_bias: 128, YpRangeMax: 255, CbCrRangeMax: 255,
                YpMax: 255, YpMin: 0, CbCrMax: 255, CbCrMin: 0)
            : vImage_YpCbCrPixelRange(
                Yp_bias: 16, CbCr_bias: 128, YpRangeMax: 235, CbCrRangeMax: 240,
                YpMax: 235, YpMin: 16, CbCrMax: 240, CbCrMin: 16)
        // ITU-R BT.709 (Kr 0.2126, Kb 0.0722), the values of vImage's
        // `kvImage_YpCbCrToARGBMatrix_ITU_R_709_2`, which Swift 6 rejects as a mutable global.
        var matrix = vImage_YpCbCrToARGBMatrix(
            Yp: 1, Cr_R: 1.5748, Cr_G: -0.468_124_2, Cb_G: -0.187_324_3, Cb_B: 1.8556)
        var info = vImage_YpCbCrToARGB()
        var err = vImageConvert_YpCbCrToARGB_GenerateConversion(
            &matrix, &range, &info, kvImage420Yp8_CbCr8, kvImageARGB8888, vImage_Flags(kvImageNoFlags))
        guard err == kvImageNoError else {
            throw SAM31Error.invalidImage("vImage conversion setup failed (\(err))")
        }

        var yPlane = vImage_Buffer(
            data: yBase, height: vImagePixelCount(h), width: vImagePixelCount(w),
            rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0))
        var cPlane = vImage_Buffer(
            data: cBase, height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(buffer, 1)),
            width: vImagePixelCount(CVPixelBufferGetWidthOfPlane(buffer, 1)),
            rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 1))
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        err = rgba.withUnsafeMutableBytes { raw in
            var dest = vImage_Buffer(
                data: raw.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w),
                rowBytes: w * 4)
            let argbToRGBA: [UInt8] = [1, 2, 3, 0]
            return vImageConvert_420Yp8_CbCr8ToARGB8888(
                &yPlane, &cPlane, &dest, &info, argbToRGBA, 255, vImage_Flags(kvImageNoFlags))
        }
        guard err == kvImageNoError else {
            throw SAM31Error.invalidImage("vImage 4:2:0 conversion failed (\(err))")
        }
        return rgba
    }

    private static func fourCC(_ code: OSType) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
        let printable = bytes.allSatisfy { (0x20..<0x7F).contains($0) }
        return printable ? "'\(String(decoding: bytes, as: UTF8.self))'" : "\(code)"
    }
}

/// Pillow's `Image.resize((outWidth, outHeight), BILINEAR)` on packed RGB bytes.
///
/// Horizontal pass then vertical pass, each skipped when that axis keeps its size. The passes store
/// uint8, so the result is rounded twice exactly as Pillow rounds it. Like `Image.resize`, an image
/// more than 100 times taller than wide that shrinks vertically runs the vertical pass first.
func pillowResizeRGB(_ rgb: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int) -> [UInt8] {
    precondition(rgb.count == width * height * 3, "expected \(width)x\(height)x3 bytes, got \(rgb.count)")
    var (buf, w, h) = (rgb, width, height)
    if h > w * 100 && outHeight < h {
        buf = resampleVertical(buf, width: w, height: h, outHeight: outHeight)
        h = outHeight
    }
    if w != outWidth {
        buf = resampleHorizontal(buf, width: w, height: h, outWidth: outWidth)
        w = outWidth
    }
    if h != outHeight {
        buf = resampleVertical(buf, width: w, height: h, outHeight: outHeight)
    }
    return buf
}

/// `clip8`: drop the fixed-point fraction (arithmetic shift, so floor) and clamp to 0...255.
@inline(__always) private func clip8(_ v: Int) -> UInt8 {
    UInt8(clamping: v >> PillowCoefficients.precisionBits)
}

/// Rounding bias every accumulator starts from: 0.5 in fixed point.
private let fixedHalf = 1 << (PillowCoefficients.precisionBits - 1)

// The two passes below are deliberately exact scalar loops: about 1.6 s per 1080p frame in Debug
// builds, about 10 ms in Release.

/// `ImagingResampleHorizontal_8bpc` on packed RGB.
private func resampleHorizontal(_ src: [UInt8], width: Int, height: Int, outWidth: Int) -> [UInt8] {
    let c = PillowCoefficients(bilinearFrom: width, to: outWidth)
    let kk = c.fixedPoint()
    var out = [UInt8](repeating: 0, count: outWidth * height * 3)
    src.withUnsafeBufferPointer { s in
        kk.withUnsafeBufferPointer { k in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<height {
                    let row = y * width * 3
                    let outRow = y * outWidth * 3
                    for xx in 0..<outWidth {
                        let (xmin, n, kOff) = (c.xmin[xx], c.count[xx], xx * c.kSize)
                        var (ss0, ss1, ss2) = (fixedHalf, fixedHalf, fixedHalf)
                        for x in 0..<n {
                            let p = row + (xmin + x) * 3
                            let w = Int(k[kOff + x])
                            ss0 += Int(s[p]) * w
                            ss1 += Int(s[p + 1]) * w
                            ss2 += Int(s[p + 2]) * w
                        }
                        o[outRow + xx * 3] = clip8(ss0)
                        o[outRow + xx * 3 + 1] = clip8(ss1)
                        o[outRow + xx * 3 + 2] = clip8(ss2)
                    }
                }
            }
        }
    }
    return out
}

/// `ImagingResampleVertical_8bpc` on packed RGB.
private func resampleVertical(_ src: [UInt8], width: Int, height: Int, outHeight: Int) -> [UInt8] {
    let c = PillowCoefficients(bilinearFrom: height, to: outHeight)
    let kk = c.fixedPoint()
    let rowBytes = width * 3
    var out = [UInt8](repeating: 0, count: rowBytes * outHeight)
    src.withUnsafeBufferPointer { s in
        kk.withUnsafeBufferPointer { k in
            out.withUnsafeMutableBufferPointer { o in
                for yy in 0..<outHeight {
                    let (ymin, n, kOff) = (c.xmin[yy], c.count[yy], yy * c.kSize)
                    for i in 0..<rowBytes {
                        var ss = fixedHalf
                        for y in 0..<n { ss += Int(s[(ymin + y) * rowBytes + i]) * Int(k[kOff + y]) }
                        o[yy * rowBytes + i] = clip8(ss)
                    }
                }
            }
        }
    }
    return out
}
