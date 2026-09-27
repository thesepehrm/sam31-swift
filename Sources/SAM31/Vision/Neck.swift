// Port of mlx_vlm/models/sam3/vision.py::FPNLayer and mlx_vlm/models/sam3_1/vision.py::{TriViTDetNeck,
// VisionEncoder} (mlx-vlm 0.7.3)
import MLX
import MLXNN

/// Single FPN scale: upscale, then 1x1 projection, then 3x3 refinement.
///
/// `scale_layers` keeps PyTorch's `nn.Sequential` numbering: for 4x it is `[ConvT, GELU, ConvT]`, so
/// the weights sit at indices 0 and 2 and the parameterless GELU holds index 1. For 2x it is
/// `[ConvT]`; for 1x it is empty.
final class FPNLayer: Module {
    let scaleFactor: Float
    let numUpscale: Int
    let hasScaleLayers: Bool
    let isDownsample: Bool

    @ModuleInfo(key: "scale_layers") var scaleLayers: [Module]
    @ModuleInfo(key: "proj1") var proj1: Conv2d
    @ModuleInfo(key: "proj2") var proj2: Conv2d

    init(inChannels: Int, outChannels: Int, scaleFactor: Float, fpnKernelSize: Int = 2, fpnStride: Int = 2) {
        self.scaleFactor = scaleFactor
        var currentChannels = inChannels
        var layers: [Module] = []
        if scaleFactor >= 4.0 {
            let mid = currentChannels / 2
            let mid2 = mid / 2
            layers = [
                ConvTransposed2d(
                    inputChannels: currentChannels, outputChannels: mid, kernelSize: .init(fpnKernelSize),
                    stride: .init(fpnStride)),
                GELU(),  // index 1: no weights
                ConvTransposed2d(
                    inputChannels: mid, outputChannels: mid2, kernelSize: .init(fpnKernelSize),
                    stride: .init(fpnStride)),
            ]
            currentChannels = mid2
            numUpscale = 2
        } else if scaleFactor >= 2.0 {
            let mid = currentChannels / 2
            layers = [
                ConvTransposed2d(
                    inputChannels: currentChannels, outputChannels: mid, kernelSize: .init(fpnKernelSize),
                    stride: .init(fpnStride))
            ]
            currentChannels = mid
            numUpscale = 1
        } else {
            numUpscale = 0
        }
        hasScaleLayers = numUpscale > 0
        isDownsample = scaleFactor <= 0.5

        _scaleLayers.wrappedValue = layers
        _proj1.wrappedValue = Conv2d(
            inputChannels: currentChannels, outputChannels: outChannels, kernelSize: 1, bias: true)
        _proj2.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1, bias: true)
    }

    /// - Parameter x: `(B, H, W, C)` feature map.
    /// - Returns: `(B, H', W', outChannels)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if hasScaleLayers {
            for layer in scaleLayers {
                // Every entry is a ConvTransposed2d or the GELU placeholder.
                x = (layer as! UnaryLayer)(x)
            }
        } else if isDownsample {
            // MaxPool2d equivalent: stride 2.
            let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            x = x.reshaped(B, H / 2, 2, W / 2, 2, C)
            x = x.max(axes: [2, 4])
        }

        x = proj1(x)
        x = proj2(x)
        return x
    }
}

/// FPN outputs of ``TriViTDetNeck``. Each list holds `(B, H_i, W_i, D)` maps at scales [4x, 2x, 1x],
/// or is empty when that head was not requested.
typealias TriNeckFeatures = (det: [MLXArray], interactive: [MLXArray], propagation: [MLXArray])

/// Triple-head FPN neck for SAM 3.1: detection, interactive, and propagation heads share one backbone
/// output. Only three scale factors, [4, 2, 1]; no 0.5x downsample.
final class TriViTDetNeck: Module {
    @ModuleInfo(key: "convs") var convs: [FPNLayer]
    @ModuleInfo(key: "interactive_convs") var interactiveConvs: [FPNLayer]
    @ModuleInfo(key: "propagation_convs") var propagationConvs: [FPNLayer]

    init(_ config: VisionEncoderConfig) {
        let inChannels = config.backboneConfig.hiddenSize
        func head() -> [FPNLayer] {
            config.scaleFactors.map {
                FPNLayer(
                    inChannels: inChannels, outChannels: config.fpnHiddenSize, scaleFactor: $0,
                    fpnKernelSize: config.fpnKernelSize, fpnStride: config.fpnStride)
            }
        }
        _convs.wrappedValue = head()
        _interactiveConvs.wrappedValue = head()
        _propagationConvs.wrappedValue = head()
    }

    /// - Parameter x: `(B, H, W, C)` backbone output.
    func callAsFunction(
        _ x: MLXArray, needDet: Bool = true, needInteractive: Bool = true, needPropagation: Bool = true
    ) -> TriNeckFeatures {
        var detFeatures: [MLXArray] = []
        var interactiveFeatures: [MLXArray] = []
        var propagationFeatures: [MLXArray] = []

        if needDet {
            for layer in convs {
                detFeatures.append(layer(x))
            }
        }

        if needInteractive {
            for layer in interactiveConvs {
                interactiveFeatures.append(layer(x))
            }
        }

        if needPropagation {
            for layer in propagationConvs {
                propagationFeatures.append(layer(x))
            }
        }

        return (detFeatures, interactiveFeatures, propagationFeatures)
    }
}

/// SAM 3.1 vision encoder: ViT backbone + ``TriViTDetNeck``.
final class VisionEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: ViTBackbone
    @ModuleInfo(key: "neck") var neck: TriViTDetNeck

    init(_ config: VisionEncoderConfig) {
        _backbone.wrappedValue = ViTBackbone(config.backboneConfig)
        _neck.wrappedValue = TriViTDetNeck(config)
    }

    func callAsFunction(
        _ x: MLXArray, needDet: Bool = true, needInteractive: Bool = true, needPropagation: Bool = true
    ) -> TriNeckFeatures {
        let features = backbone(x)
        return neck(
            features, needDet: needDet, needInteractive: needInteractive, needPropagation: needPropagation)
    }
}
