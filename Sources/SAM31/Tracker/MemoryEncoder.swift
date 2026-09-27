// Port of mlx_vlm/models/sam3_1/tracker.py::{MultiplexMaskDownSampler, MultiplexMemoryEncoder} and
// mlx_vlm/models/sam3/tracker.py::{DownsampleConvBlock, CXBlock, MemoryFuser} (mlx-vlm 0.7.3)
import MLX
import MLXNN

/// Single conv + layer norm + GELU block. Weight keys: `conv.*`, `layer_norm.*`.
final class DownsampleConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm2d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: .init(kernelSize),
            stride: .init(stride), padding: .init(padding))
        _layerNorm.wrappedValue = LayerNorm2d(outChannels)
    }

    /// `(B, H, W, C_in)` -> `(B, H', W', C_out)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = conv(x)
        x = layerNorm(x)
        x = gelu(x)
        return x
    }
}

/// ConvNeXt-style block with a depthwise conv.
/// Weight keys: `tracker_model.memory_encoder.memory_fuser.layers.*`.
final class CXBlock: Module {
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv2d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm2d
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Linear
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ config: TrackerConfig) {
        let dim = config.memoryFuserEmbedDim
        _depthwiseConv.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim, kernelSize: .init(config.memoryFuserKernelSize),
            padding: .init(config.memoryFuserPadding), groups: dim)
        _layerNorm.wrappedValue = LayerNorm2d(dim)
        _pointwiseConv1.wrappedValue = Linear(dim, config.memoryFuserIntermediateDim)
        _pointwiseConv2.wrappedValue = Linear(config.memoryFuserIntermediateDim, dim)
        _scale.wrappedValue = MLXArray.ones([dim]) * config.memoryFuserLayerScaleInitValue
    }

    /// `(B, H, W, C)` -> `(B, H, W, C)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = depthwiseConv(x)
        x = layerNorm(x)
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        x = x.reshaped(b * h * w, c)
        x = pointwiseConv1(x)
        x = gelu(x)
        x = pointwiseConv2(x)
        x = x.reshaped(b, h, w, c)
        x = scale * x
        return residual + x
    }
}

/// Stack of CXBlocks fusing mask and image features. Weight keys: `memory_fuser.*`.
final class MemoryFuser: Module {
    @ModuleInfo(key: "layers") var layers: [CXBlock]

    init(_ config: TrackerConfig) {
        _layers.wrappedValue = (0..<config.memoryFuserNumLayers).map { _ in CXBlock(config) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            x = layer(x)
        }
        return x
    }
}

/// Mask downsampler for multiplex: 32 input channels (16 objects x 2).
/// Weight keys: `tracker_model.memory_encoder.mask_downsampler.*`.
final class MultiplexMaskDownSampler: Module {
    let inputSize: Int

    @ModuleInfo(key: "layers") var layers: [DownsampleConvBlock]
    @ModuleInfo(key: "final_conv") var finalConv: Conv2d

    init(_ config: TrackerConfig) {
        let firstCh = config.maskDownsamplerFirstChannels
        inputSize = config.maskDownsamplerInputSize
        // Progressive: 32 -> 16 -> 64 -> 256 -> 1024 (from checkpoint)
        let channels = [firstCh * 2, firstCh, firstCh * 4, firstCh * 16, firstCh * 64]
        _layers.wrappedValue = zip(channels, channels.dropFirst()).map { inCh, outCh in
            DownsampleConvBlock(
                inChannels: inCh, outChannels: outCh, kernelSize: config.maskDownsamplerKernelSize,
                stride: config.maskDownsamplerStride, padding: config.maskDownsamplerPadding)
        }
        _finalConv.wrappedValue = Conv2d(
            inputChannels: channels[channels.count - 1], outputChannels: config.maskDownsamplerEmbedDim,
            kernelSize: .init(1), bias: true)
    }

    /// - Parameter masks: `(B, H, W, 2 * multiplexCount)` muxed mask channels.
    func callAsFunction(_ masks: MLXArray) -> MLXArray {
        var masks = masks
        if masks.dim(1) != inputSize || masks.dim(2) != inputSize {
            masks = resizeBilinearNHWC(masks, h: inputSize, w: inputSize)
        }
        for layer in layers {
            masks = layer(masks)
        }
        return finalConv(masks)
    }
}

/// Memory encoder for SAM 3.1 multiplex. Weight keys: `tracker_model.memory_encoder.*`.
/// SAM 3.1 uses dim = out_dim = 256, so there is no separate output projection.
final class MultiplexMemoryEncoder: Module {
    @ModuleInfo(key: "mask_downsampler") var maskDownsampler: MultiplexMaskDownSampler
    @ModuleInfo(key: "memory_fuser") var memoryFuser: MemoryFuser
    @ModuleInfo(key: "feature_projection") var featureProjection: Conv2d
    /// Parameter-free sinusoidal 2D pos enc (256-dim output).
    let positionEncoding: PositionEmbeddingSine

    init(_ config: TrackerConfig) {
        let dim = config.memoryEncoderHiddenSize
        _maskDownsampler.wrappedValue = MultiplexMaskDownSampler(config)
        _memoryFuser.wrappedValue = MemoryFuser(config)
        _featureProjection.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim, kernelSize: .init(1), bias: true)
        positionEncoding = PositionEmbeddingSine(numPosFeats: dim / 2)
    }

    /// - Parameters:
    ///   - features: `(1, H, W, D)` propagation vision features.
    ///   - masks: `(B, H_m, W_m, 2M)` muxed mask channels.
    /// - Returns: `(memory, posEnc)`, each `(B, H, W, D)`.
    func callAsFunction(_ features: MLXArray, masks: MLXArray) -> (memory: MLXArray, posEnc: MLXArray) {
        let fused = featureProjection(features) + maskDownsampler(masks)
        let memory = memoryFuser(fused)
        return (memory, positionEncoding(memory))
    }
}
