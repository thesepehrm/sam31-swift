// Port of mlx_vlm/models/sam3/sam_components.py::{LayerNorm2d, SAMPromptEncoder, MaskEmbedConvs,
// PositionalEmbedding} and the point pre-step of mlx_vlm/models/sam3_1/tracker.py::
// MultiplexTrackerModel._forward_sam_heads (mlx-vlm 0.7.3)
import Foundation
import MLX
import MLXNN

/// Channel-wise LayerNorm for channel-last spatial features `(B, H, W, C)`.
final class LayerNorm2d: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(_ numChannels: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([numChannels])
        _bias.wrappedValue = MLXArray.zeros([numChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let v = x.variance(axis: -1, keepDims: true)
        let y = (x - mean) / sqrt(v + eps)
        return y * weight + bias
    }
}

/// Random-Fourier spatial positional embedding.
/// Weight key: `shared_embedding.positional_embedding`.
final class PositionalEmbedding: Module {
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray

    init(numPosFeats: Int = 128) {
        _positionalEmbedding.wrappedValue = MLXArray.zeros([2, numPosFeats])
    }

    /// Positional encoding of an `H x W` grid.
    ///
    /// - Returns: `(H*W, D)`.
    func callAsFunction(_ size: (h: Int, w: Int)) -> MLXArray {
        let gridY = MLXArray.arange(size.h).asType(.float32) / Float(size.h)
        let gridX = MLXArray.arange(size.w).asType(.float32) / Float(size.w)
        let grid = meshGrid([gridY, gridX], indexing: .ij)
        let coords = stacked([grid[1].reshaped(-1), grid[0].reshaped(-1)], axis: -1)  // (H*W, 2)
        return forwardWithCoords(coords[.newAxis])[0]
    }

    /// - Parameter coords: `(B, N, 2)` coordinates in `[0, 1]`.
    /// - Returns: `(B, N, D)`.
    func forwardWithCoords(_ coords: MLXArray) -> MLXArray {
        var c = 2 * coords - 1
        c = matmul(c, positionalEmbedding)
        c = Float(2 * Double.pi) * c
        return concatenated([sin(c), cos(c)], axis: -1)
    }
}

/// Conv stack that embeds a mask prompt. Weight keys: `mask_embed.*`.
final class MaskEmbedConvs: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv3") var conv3: Conv2d
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm2d
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm2d

    init(embedDim: Int, maskInChans: Int) {
        _conv1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: maskInChans / 4, kernelSize: .init(2), stride: .init(2))
        _conv2.wrappedValue = Conv2d(
            inputChannels: maskInChans / 4, outputChannels: maskInChans, kernelSize: .init(2),
            stride: .init(2))
        _conv3.wrappedValue = Conv2d(
            inputChannels: maskInChans, outputChannels: embedDim, kernelSize: .init(1))
        _layerNorm1.wrappedValue = LayerNorm2d(maskInChans / 4)
        _layerNorm2.wrappedValue = LayerNorm2d(maskInChans)
    }

    /// - Parameter masks: `(B, H, W, 1)` channel-last mask input.
    /// - Returns: `(B, H/4 * W/4, embedDim)`.
    func callAsFunction(_ masks: MLXArray) -> MLXArray {
        var x = conv1(masks)
        x = layerNorm1(x)
        x = gelu(x)
        x = conv2(x)
        x = layerNorm2(x)
        x = gelu(x)
        x = conv3(x)
        return x.reshaped(x.dim(0), x.dim(1) * x.dim(2), x.dim(3))
    }
}

/// SAM prompt encoder for points, boxes, and masks.
///
/// Labels follow Python: -1 padding, 0 negative, 1 positive, 2/3 box corners. Point coordinates are
/// in embedding-grid units (see ``preparePointInputs(coords:labels:embeddingSize:imageSize:)``).
final class SAMPromptEncoder: Module {
    let embedDim: Int
    let imageEmbeddingSize: (h: Int, w: Int)

    @ModuleInfo(key: "point_embed") var pointEmbed: Embedding
    @ModuleInfo(key: "not_a_point_embed") var notAPointEmbed: Embedding
    @ModuleInfo(key: "mask_embed") var maskEmbed: MaskEmbedConvs
    @ModuleInfo(key: "no_mask_embed") var noMaskEmbed: Embedding
    @ModuleInfo(key: "shared_embedding") var sharedEmbedding: PositionalEmbedding

    init(_ config: PromptEncoderConfig) {
        let d = config.hiddenSize
        embedDim = d
        let side = config.imageSize / config.patchSize
        imageEmbeddingSize = (side, side)
        _pointEmbed.wrappedValue = Embedding(embeddingCount: config.numPointEmbeddings, dimensions: d)
        _notAPointEmbed.wrappedValue = Embedding(embeddingCount: 1, dimensions: d)
        _maskEmbed.wrappedValue = MaskEmbedConvs(embedDim: d, maskInChans: config.maskInputChannels)
        _noMaskEmbed.wrappedValue = Embedding(embeddingCount: 1, dimensions: d)
        _sharedEmbedding.wrappedValue = PositionalEmbedding(numPosFeats: d / 2)
    }

    /// Positional encoding for image-sized features: `(1, H*W, D)`.
    func getDensePE() -> MLXArray {
        sharedEmbedding(imageEmbeddingSize)[.newAxis]
    }

    /// - Parameters:
    ///   - points: coords `(B, N, 2)` and labels `(B, N)`.
    ///   - boxes: `(B, N_box, 4)`.
    ///   - masks: `(B, H, W, 1)` channel-last mask prompt.
    /// - Returns: sparse `(B, N_tokens, D)` and dense `(B, H*W, D)` embeddings.
    func callAsFunction(points: (coords: MLXArray, labels: MLXArray)?, boxes: MLXArray?, masks: MLXArray?)
        -> (sparse: MLXArray, dense: MLXArray)
    {
        var b = 1
        var sparse = MLXArray.zeros([b, 0, embedDim])

        if let points {
            b = points.coords.dim(0)
            let pointEmb = embedPoints(points.coords, points.labels)
            sparse = concatenated([broadcast(sparse, to: [b, 0, embedDim]), pointEmb], axis: 1)
        }

        if let boxes {
            b = boxes.dim(0)
            sparse = concatenated([sparse, embedBoxes(boxes)], axis: 1)
        }

        let dense: MLXArray
        if let masks {
            dense = maskEmbed(masks)
        } else {
            let (h, w) = imageEmbeddingSize
            dense = broadcast(noMaskEmbed.weight.reshaped(1, 1, embedDim), to: [b, h * w, embedDim])
        }
        return (sparse, dense)
    }

    private func embedPoints(_ coords: MLXArray, _ labels: MLXArray) -> MLXArray {
        var c = coords + 0.5  // shift to pixel centre
        c = c / MLXArray([Float(imageEmbeddingSize.w), Float(imageEmbeddingSize.h)])
        var pointEmb = sharedEmbedding.forwardWithCoords(c)

        // Label-specific embedding. A -1 label indexes the last row (mlx wraps negative gather
        // indices); padding points are overwritten below, as in Python.
        for i in 0..<labels.dim(-1) {
            let label = labels[0..., i..<(i + 1)].asType(.int32)
            pointEmb = pointEmb.at[0..., i..<(i + 1)].add(pointEmbed(label))
        }

        let paddingMask = labels .== -1
        if paddingMask.any().item(Bool.self) {
            pointEmb = MLX.where(paddingMask[.ellipsis, .newAxis], notAPointEmbed.weight, pointEmb)
        }
        return pointEmb
    }

    private func embedBoxes(_ boxes: MLXArray) -> MLXArray {
        let coords = boxes.reshaped(-1, 2, 2)
        var cornerEmb = sharedEmbedding.forwardWithCoords(coords)
        cornerEmb = cornerEmb.at[0..., 0..<1].add(pointEmbed(MLXArray([Int32(2)]).reshaped(1, 1)))
        cornerEmb = cornerEmb.at[0..., 1..<2].add(pointEmbed(MLXArray([Int32(3)]).reshaped(1, 1)))
        return cornerEmb
    }
}

/// The point pre-step of `tracker.py::_forward_sam_heads`, run before the interactive prompt encoder:
/// append one not-a-point pad (coords `(0, 0)`, label -1), since boxes are never passed in the
/// tracker, then rescale from image pixels to embedding-grid units (`embeddingSize / imageSize`).
///
/// - Parameters:
///   - coords: `(N, P, 2)` points in `imageSize` pixel space.
///   - labels: `(N, P)` labels (1 positive, 0 negative, 2/3 box corners, -1 padding).
/// - Returns: coords `(N, P+1, 2)` in grid units and int32 labels `(N, P+1)`.
func preparePointInputs(coords: MLXArray, labels: MLXArray, embeddingSize: Int, imageSize: Int)
    -> (coords: MLXArray, labels: MLXArray)
{
    let padCoord = MLXArray.zeros([coords.dim(0), 1, 2])
    let padLabel = -MLXArray.ones([labels.dim(0), 1], type: Int32.self)
    let c = concatenated([coords, padCoord], axis: 1)
    let l = concatenated([labels.asType(.int32), padLabel], axis: 1)
    // Python computes the scale as a float64 and mlx casts it to the array's float32.
    let gridScale = Float(Double(embeddingSize) / Double(imageSize))
    return (c * gridScale, l)
}
