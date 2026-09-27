import CoreGraphics
import Foundation
import MLX

/// `(H, W, 3)` uint8 RGB as an opaque RGBA8 sRGB `CGImage`.
func rgbToCGImage(_ rgb: MLXArray) -> CGImage {
    precondition(rgb.ndim == 3 && rgb.dim(2) == 3, "expected (H, W, 3), got \(rgb.shape)")
    let (h, w) = (rgb.dim(0), rgb.dim(1))
    let src = rgb.asType(.uint8).asArray(UInt8.self)
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0..<(w * h) {
        rgba[i * 4] = src[i * 3]
        rgba[i * 4 + 1] = src[i * 3 + 1]
        rgba[i * 4 + 2] = src[i * 3 + 2]
    }
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    return CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
        decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
}
