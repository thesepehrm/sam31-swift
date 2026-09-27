// Port of mlx_vlm/models/sam3_1/tracker.py::{MultiplexTrackerModel, ObjectPointerMLP, _resize_masks}
// (mlx-vlm 0.7.3)
//
// Task 12 covers the module tree, frame features, the SAM heads, mask-as-output, and track_step for
// init conditioning frames. The memory side (`_prepare_memory_conditioned_features`,
// `_encode_new_memory`) and add/recondition land in Task 13.
import MLX
import MLXNN

/// A large negative value as a placeholder score for missing objects.
let noObjScore: Float = -1024.0

/// Projects SAM output tokens to object pointers. Weight keys: `tracker_model.(interactive_)obj_ptr_proj.*`.
final class ObjectPointerMLP: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(hiddenSize: Int) {
        _layers.wrappedValue = (0..<3).map { _ in Linear(hiddenSize, hiddenSize) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers.dropLast() {
            x = relu(layer(x))
        }
        return layers[layers.count - 1](x)
    }
}

/// Bilinear resize of the last two dims of `(N, K, H, W)` mask logits (`tracker.py::_resize_masks`).
/// Not the Pillow-based ``resizeMasks(_:to:)`` of the detector.
func resizeTrackerMasks(_ masks: MLXArray, h outH: Int, w outW: Int, antialias: Bool = false) -> MLXArray {
    let (n, k, h, w) = (masks.dim(0), masks.dim(1), masks.dim(2), masks.dim(3))
    if h == outH && w == outW { return masks }
    let out = resizeBilinearNHWC(masks.reshaped(n * k, h, w, 1), h: outH, w: outW, antialias: antialias)
    return out.reshaped(n, k, outH, outW)
}

/// Point prompts for ``MultiplexTrackerModel/trackStep(_:frameIndex:isInitCondFrame:features:pointInputs:maskInputs:numFrames:runMemEncoder:prevSAMMaskLogits:objectsToInteract:)``
/// (Python's `point_inputs` dict).
struct PointInputs {
    /// `(N, P, 2)` points in `imageSize` (1008) pixel space.
    let coords: MLXArray
    /// `(N, P)` int32 labels: 1 positive, 0 negative, 2/3 box corners, -1 padding.
    let labels: MLXArray
}

/// One FPN branch of the per-frame tracker features (Python's `frame_features[...]` dict).
struct LevelFeatures {
    /// `(1, 72, 72, D)` lowest-resolution FPN level.
    var visionFeat: MLXArray
    /// `(1, 72, 72, D)` sine position encoding of `visionFeat`; propagation branch only.
    var visionPos: MLXArray?
    /// `[s0 (1, 288, 288, D/8), s1 (1, 144, 144, D/4)]`, already projected by the decoder's
    /// `conv_s0`/`conv_s1`.
    var highRes: [MLXArray]
}

/// Per-frame tracker features, the output of `prepare_frame_features`.
struct TrackerFrameFeatures {
    var interactive: LevelFeatures?
    var propagation: LevelFeatures?
}

/// Output of `_forward_sam_heads` / `_use_mask_as_output`, in data space (one row per object).
struct SAMHeadOutput {
    /// `(N, K, h, w)` low-res mask logits for every candidate.
    var lowResMultimasks: MLXArray
    /// `(N, K, H_im, W_im)`.
    var highResMultimasks: MLXArray
    /// `(N, K)` predicted IoU.
    var ious: MLXArray
    /// `(N, 1, h, w)` selected low-res mask.
    var lowResMasks: MLXArray
    /// `(N, 1, H_im, W_im)` selected high-res mask.
    var highResMasks: MLXArray
    /// `(N, 1)`.
    var objectScoreLogits: MLXArray
    /// `(N, C)` object pointers.
    var objPtr: MLXArray
}

/// A frame's stored tracking output (Python's `current_out` dict). A reference type because Python
/// mutates it in place (`_merge_mask_output`, `_reencode_memory`) after storing it in the state.
final class FrameOutput {
    /// Object indices (data space) that condition on this frame.
    var conditioningObjects: Set<Int>
    var pointInputs: PointInputs?
    var maskInputs: MLXArray?
    /// `(N, 1, h, w)` low-res mask logits (288 from the SAM decoder, 252 for mask prompts).
    var predMasks: MLXArray
    /// `(N, 1, 1008, 1008)` mask logits.
    var predMasksHighRes: MLXArray
    /// `(N, 1)`.
    var objectScoreLogits: MLXArray
    /// `(numBuckets, multiplexCount, C)` object pointers in multiplex space; nil unless
    /// `use_obj_ptrs_in_encoder`.
    var objPtr: MLXArray?
    /// `(numBuckets, H, W, C)` spatial memory, set when the memory encoder runs.
    var maskmemFeatures: MLXArray?
    var maskmemPosEnc: MLXArray?
    /// `(1, H, W, C)` propagation features and their pos enc, when `save_image_features`.
    var imageFeatures: MLXArray?
    var imagePosEnc: MLXArray?

    init(
        conditioningObjects: Set<Int>, pointInputs: PointInputs?, maskInputs: MLXArray?,
        predMasks: MLXArray, predMasksHighRes: MLXArray, objectScoreLogits: MLXArray
    ) {
        self.conditioningObjects = conditioningObjects
        self.pointInputs = pointInputs
        self.maskInputs = maskInputs
        self.predMasks = predMasks
        self.predMasksHighRes = predMasksHighRes
        self.objectScoreLogits = objectScoreLogits
    }
}

/// Features `_track_step_aux` hands back for adding objects on the same frame (Python's `aux_output`).
struct TrackStepAuxOutput {
    let interactivePixFeat: MLXArray
    let interactiveHighResFeatures: [MLXArray]
    let propagationVisionFeat: MLXArray?
}

/// SAM 3.1 multiplex tracker (VideoTrackingMultiplex port). Weight keys: `tracker_model.*`.
final class MultiplexTrackerModel: Module {
    let config: TrackerConfig
    let multiplexCount: Int
    let hiddenDim: Int
    let multiplexController: MultiplexController

    /// Interactive SAM components (point/box prompts, single object slot).
    @ModuleInfo(key: "interactive_sam_prompt_encoder") var interactiveSamPromptEncoder: SAMPromptEncoder
    @ModuleInfo(key: "interactive_sam_mask_decoder") var interactiveSamMaskDecoder: MultiplexMaskDecoder
    /// Propagation SAM mask decoder (multiplex: 16 objects).
    @ModuleInfo(key: "sam_mask_decoder") var samMaskDecoder: MultiplexMaskDecoder

    @ModuleInfo(key: "memory_attention") var memoryAttention: DecoupledMemoryAttention
    @ModuleInfo(key: "memory_encoder") var memoryEncoder: MultiplexMemoryEncoder

    @ModuleInfo(key: "obj_ptr_proj") var objPtrProj: ObjectPointerMLP
    @ModuleInfo(key: "interactive_obj_ptr_proj") var interactiveObjPtrProj: ObjectPointerMLP

    @ParameterInfo(key: "memory_temporal_positional_encoding") var memoryTemporalPositionalEncoding: MLXArray
    @ModuleInfo(key: "temporal_positional_encoding_projection_layer")
    var temporalPositionalEncodingProjectionLayer: Linear

    // Multiplex-specific embeddings
    @ParameterInfo(key: "output_valid_embed") var outputValidEmbed: MLXArray
    @ParameterInfo(key: "output_invalid_embed") var outputInvalidEmbed: MLXArray
    @ParameterInfo(key: "no_obj_embed_spatial") var noObjEmbedSpatial: MLXArray
    @ModuleInfo(key: "no_obj_ptr_linear") var noObjPtrLinear: Linear
    @ParameterInfo(key: "interactivity_no_mem_embed") var interactivityNoMemEmbed: MLXArray

    /// Image positional encoding of the propagation decoder.
    @ModuleInfo(key: "image_pe_layer") var imagePELayer: PositionalEmbedding
    /// In converted checkpoints for weight-load compatibility; unused, as in Python.
    @ModuleInfo(key: "shared_image_embedding") var sharedImageEmbedding: PositionalEmbedding
    /// Python's `_pos_enc`: sinusoidal 2D pos enc for FPN features (parameter-free).
    let posEnc: PositionEmbeddingSine

    @ModuleInfo(key: "interactive_mask_downsample") var interactiveMaskDownsample: Conv2d

    init(_ config: TrackerConfig) {
        self.config = config
        let d = config.memoryAttentionHiddenSize
        let m = config.multiplexCount
        multiplexCount = m
        hiddenDim = d
        multiplexController = MultiplexController(multiplexCount: m)

        _interactiveSamPromptEncoder.wrappedValue = SAMPromptEncoder(config.promptEncoderConfig)
        _interactiveSamMaskDecoder.wrappedValue = MultiplexMaskDecoder(config.interactiveMaskDecoderConfig)
        _samMaskDecoder.wrappedValue = MultiplexMaskDecoder(config.maskDecoderConfig)

        _memoryAttention.wrappedValue = DecoupledMemoryAttention(config)
        _memoryEncoder.wrappedValue = MultiplexMemoryEncoder(config)

        _objPtrProj.wrappedValue = ObjectPointerMLP(hiddenSize: d)
        _interactiveObjPtrProj.wrappedValue = ObjectPointerMLP(hiddenSize: d)

        _memoryTemporalPositionalEncoding.wrappedValue = MLXArray.zeros([config.numMaskmem, 1, 1, d])
        _temporalPositionalEncodingProjectionLayer.wrappedValue = Linear(d, d)

        _outputValidEmbed.wrappedValue = MLXArray.zeros([m, d])
        _outputInvalidEmbed.wrappedValue = MLXArray.zeros([m, d])
        _noObjEmbedSpatial.wrappedValue = MLXArray.zeros([m, d])
        _noObjPtrLinear.wrappedValue = Linear(d, d)
        _interactivityNoMemEmbed.wrappedValue = MLXArray.zeros([1, 1, d])

        _imagePELayer.wrappedValue = PositionalEmbedding(numPosFeats: d / 2)
        _sharedImageEmbedding.wrappedValue = PositionalEmbedding(numPosFeats: d / 2)
        posEnc = PositionEmbeddingSine(numPosFeats: d / 2)

        _interactiveMaskDownsample.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: 1, kernelSize: .init(4), stride: .init(4), bias: true)
    }

    // MARK: - State management

    /// A fresh tracking session state for `numObjects` objects.
    func initState(numObjects: Int, objectIDs: [Int]? = nil) -> MultiplexTrackerState {
        let mux = multiplexController.getState(
            numValidEntries: numObjects, random: false, objectIDs: objectIDs)
        return MultiplexTrackerState(mux)
    }

    // MARK: - Frame feature preparation

    /// Per-frame backbone features (mirrors `forward_image`). Pre-applies the `conv_s0`/`conv_s1`
    /// high-res projections so they are not recomputed on every decoder call, and computes the sine
    /// pos enc of the propagation branch. Each FPN is `[f0 (288), f1 (144), f2 (72)]` or nil.
    func prepareFrameFeatures(interactiveFPN: [MLXArray]?, propagationFPN: [MLXArray]?)
        -> TrackerFrameFeatures
    {
        func prep(_ fpn: [MLXArray], _ decoder: MultiplexMaskDecoder) -> LevelFeatures {
            LevelFeatures(
                visionFeat: fpn[fpn.count - 1], visionPos: nil,
                highRes: [decoder.convS0(fpn[0]), decoder.convS1(fpn[1])])
        }

        var out = TrackerFrameFeatures()
        if let interactiveFPN {
            out.interactive = prep(interactiveFPN, interactiveSamMaskDecoder)
        }
        if let propagationFPN {
            out.propagation = prep(propagationFPN, samMaskDecoder)
            out.propagation!.visionPos = posEnc(propagationFPN[propagationFPN.count - 1])
        }
        return out
    }

    // MARK: - SAM heads

    /// Dense positional encoding for the propagation mask decoder: `(1, HW, D)`.
    ///
    /// Python caches this on the module, keyed on the embedding weight. It is a 72x72 grid, so this
    /// port recomputes it rather than keep a cache that weight reloads would have to invalidate.
    func getPropagationDensePE() -> MLXArray {
        let side = config.maskDownsamplerInputSize / 16  // 72
        return imagePELayer((side, side))[.newAxis]
    }

    /// `(1, H, W, D)` -> `(1, HW, D)` with the no-memory embedding added.
    func getInteractivePixMem(_ visionFeat: MLXArray) -> MLXArray {
        precondition(config.directlyAddNoMemEmbed)
        let (b, h, w, d) = (visionFeat.dim(0), visionFeat.dim(1), visionFeat.dim(2), visionFeat.dim(3))
        return (visionFeat + interactivityNoMemEmbed).reshaped(b, h * w, d)
    }

    /// Blends the learned no-object pointer into absent objects' pointers.
    func applyNoObjPtr(_ objPtr: MLXArray, isObjAppearing: MLXArray) -> MLXArray {
        guard config.predObjScores && config.useNoObjPtr else { return objPtr }
        let lam = isObjAppearing.asType(.float32)
        if config.useLinearNoObjPtr {
            return lam * objPtr + (1 - lam) * noObjPtrLinear(objPtr)
        }
        return config.fixedNoObjPtr ? lam * objPtr : objPtr
    }

    /// Forwards the SAM prompt encoder and mask heads: the interactive path when points or masks are
    /// given, else the multiplexed propagation path.
    ///
    /// - Parameters:
    ///   - backboneFeatures: `(B, HW, D)` image features for the decoder.
    ///   - pointInputs: `(N, P, 2)` points in image pixels and `(N, P)` labels.
    ///   - maskInputs: `(N, 1, H_im, W_im)` mask prompt.
    ///   - multimaskOutput: output multiple candidate masks and pick the best by predicted IoU.
    func forwardSAMHeads(
        backboneFeatures: MLXArray, pointInputs: PointInputs? = nil, maskInputs: MLXArray? = nil,
        interactiveHighResFeatures: [MLXArray]? = nil, propagationHighResFeatures: [MLXArray]? = nil,
        multimaskOutput: Bool = false, multiplexState: MultiplexState
    ) -> SAMHeadOutput {
        let isInteractive = pointInputs != nil || maskInputs != nil

        var lowResMultimasks: MLXArray
        let ious: MLXArray
        let samOutputTokens: MLXArray
        let objectScoreLogits: MLXArray

        if isInteractive {
            // Image-level, per-object interactive path
            precondition(interactiveHighResFeatures != nil)
            let pointCoords: MLXArray
            let pointLabels: MLXArray
            if let pointInputs {
                pointCoords = pointInputs.coords
                pointLabels = pointInputs.labels
            } else {
                // Pad with an empty point (label -1) when only masks are given
                let n = maskInputs!.dim(0)
                pointCoords = MLXArray.zeros([n, 1, 2])
                pointLabels = -MLXArray.ones([n, 1], type: Int32.self)
            }

            var samMaskPrompt: MLXArray? = nil
            if let maskInputs {
                precondition(maskInputs.ndim == 4)
                // Downsize into the prompt encoder's mask input size (4x72=288)
                let maskSize = 4 * interactiveSamPromptEncoder.imageEmbeddingSize.h
                samMaskPrompt = resizeTrackerMasks(maskInputs, h: maskSize, w: maskSize, antialias: true)
            }

            // Not-a-point pad (boxes are never used in the tracker) and 1008 -> 72 grid units.
            let points = preparePointInputs(
                coords: pointCoords, labels: pointLabels,
                embeddingSize: interactiveSamPromptEncoder.imageEmbeddingSize.h, imageSize: config.imageSize)
            let (sparse, dense) = interactiveSamPromptEncoder(
                points: points, boxes: nil, masks: samMaskPrompt?.transposed(0, 2, 3, 1))

            let out = interactiveSamMaskDecoder(
                imageEmbeddings: backboneFeatures, imagePE: interactiveSamPromptEncoder.getDensePE(),
                multimaskOutput: multimaskOutput, highResFeatures: interactiveHighResFeatures,
                sparsePromptEmbeddings: sparse, densePromptEmbeddings: dense)
            // Squeeze the singleton multiplex dim of the interactive decoder
            lowResMultimasks = out.masks[0..., 0]  // (N, K, h, w)
            ious = out.iouPred[0..., 0]  // (N, K)
            samOutputTokens = out.samTokensOut[0..., 0]  // (N, K', C)
            objectScoreLogits = out.objectScoreLogits[0..., 0]  // (N, 1)
        } else {
            // Multiplexed propagation path
            precondition(propagationHighResFeatures != nil)
            var outputMergedEmbed: MLXArray? = nil
            if config.addOutputSuppressionEmbeddings {
                // Inform the mask decoder which slots hold valid objects
                let validF = multiplexState.getValidObjectMask().asType(.float32)[.ellipsis, .newAxis]
                outputMergedEmbed =
                    validF * outputValidEmbed[.newAxis] + (1 - validF) * outputInvalidEmbed[.newAxis]
            }
            let out = samMaskDecoder(
                imageEmbeddings: backboneFeatures, imagePE: getPropagationDensePE(),
                multimaskOutput: multimaskOutput, highResFeatures: propagationHighResFeatures,
                extraPerObjectEmbeddings: outputMergedEmbed)
            lowResMultimasks = multiplexState.demux(out.masks)
            ious = multiplexState.demux(out.iouPred)
            objectScoreLogits = multiplexState.demux(out.objectScoreLogits)
            samOutputTokens = multiplexState.demux(out.samTokensOut)
        }

        // The interactive and propagation paths converge here
        var isObjAppearing: MLXArray? = nil
        if config.predObjScores {
            let appearing = objectScoreLogits .> config.objectScoreLogitThreshold
            isObjAppearing = appearing
            // Hard choice between obj and no-obj for the spatial memories
            lowResMultimasks = MLX.where(appearing[0..., .newAxis, .newAxis], lowResMultimasks, noObjScore)
        }

        let highResMultimasks = resizeTrackerMasks(lowResMultimasks, h: config.imageSize, w: config.imageSize)

        var samOutputToken = samOutputTokens[0..., 0]
        let lowResMasks: MLXArray
        let highResMasks: MLXArray
        if multimaskOutput {
            // Take the best mask prediction (highest estimated IoU)
            let bestIouInds = argMax(ious, axis: -1)  // (N,)
            let inds4 = bestIouInds[0..., .newAxis, .newAxis, .newAxis]
            lowResMasks = takeAlong(lowResMultimasks, inds4, axis: 1)
            highResMasks = takeAlong(highResMultimasks, inds4, axis: 1)
            if samOutputTokens.dim(1) > 1 {
                samOutputToken =
                    takeAlong(samOutputTokens, bestIouInds[0..., .newAxis, .newAxis], axis: 1)[0..., 0]
            }
        } else {
            lowResMasks = lowResMultimasks[0..., 0..<1]
            highResMasks = highResMultimasks[0..., 0..<1]
        }

        // Object pointer from the SAM output token
        let proj = isInteractive ? interactiveObjPtrProj : objPtrProj
        var objPtr = proj(samOutputToken)
        if let isObjAppearing {
            objPtr = applyNoObjPtr(objPtr, isObjAppearing: isObjAppearing)
        }

        return SAMHeadOutput(
            lowResMultimasks: lowResMultimasks, highResMultimasks: highResMultimasks, ious: ious,
            lowResMasks: lowResMasks, highResMasks: highResMasks, objectScoreLogits: objectScoreLogits,
            objPtr: objPtr)
    }

    /// Turns binary `maskInputs` `(N, 1, H_im, W_im)` directly into output mask logits (no SAM), and
    /// derives the object pointers from the SAM decoder run on the downsampled mask.
    func useMaskAsOutput(
        backboneFeatures: MLXArray, highResFeatures: [MLXArray], maskInputs: MLXArray,
        multiplexState: MultiplexState
    ) -> SAMHeadOutput {
        // -10/+10 logits for neg/pos pixels (~0/1 after sigmoid)
        let outScale: Float = 20.0
        let outBias: Float = -10.0
        let maskInputsFloat = maskInputs.asType(.float32)
        let highResMasks = maskInputsFloat * outScale + outBias
        let (h, w) = (highResMasks.dim(-2), highResMasks.dim(-1))
        let lowResMasks = resizeTrackerMasks(highResMasks, h: h / 4, w: w / 4, antialias: true)

        // Produce an object pointer using the SAM decoder from the mask input
        let maskPrompt = interactiveMaskDownsample(maskInputsFloat.transposed(0, 2, 3, 1))
            .transposed(0, 3, 1, 2)
        let samOutputs = forwardSAMHeads(
            backboneFeatures: backboneFeatures, maskInputs: maskPrompt,
            interactiveHighResFeatures: highResFeatures, multiplexState: multiplexState)

        // The mask input itself decides if the object appears
        let n = maskInputs.dim(0)
        let isObjAppearing = (maskInputs.reshaped(n, -1) .> 0.0).any(axis: 1)[0..., .newAxis].asType(.float32)
        let objPtr = applyNoObjPtr(samOutputs.objPtr, isObjAppearing: isObjAppearing)

        return SAMHeadOutput(
            lowResMultimasks: lowResMasks, highResMultimasks: highResMasks,
            // A dummy IoU prediction of all 1's under mask input
            ious: MLXArray.ones([n, 1]), lowResMasks: lowResMasks, highResMasks: highResMasks,
            objectScoreLogits: outScale * isObjAppearing + outBias, objPtr: objPtr)
    }

    // MARK: - Memory conditioning / encoding (Task 13)

    /// Fuses the current `(1, H, W, C)` features with past memories into `(numBuckets, H, W, C)`.
    func prepareMemoryConditionedFeatures(
        frameIndex: Int, currentVisionFeat: MLXArray, currentVisionPos: MLXArray,
        state: MultiplexTrackerState, numFrames: Int, trackInReverse: Bool
    ) -> MLXArray {
        fatalError("Task 13")
    }

    /// Encodes the frame's predictions into `(maskmemFeatures, maskmemPosEnc)`, each `(B, H, W, C)`.
    func encodeNewMemory(
        currentVisionFeat: MLXArray, predMasksHighRes: MLXArray, objectScoreLogits: MLXArray,
        conditioningObjects: Set<Int>, multiplexState: MultiplexState
    ) -> (features: MLXArray, posEnc: MLXArray) {
        fatalError("Task 13")
    }

    // MARK: - Track step

    /// Whether to use multimask output in the SAM head.
    func useMultimask(isInitCondFrame: Bool, pointInputs: PointInputs?) -> Bool {
        let numPts = pointInputs?.labels.dim(1) ?? 0
        return config.multimaskOutputInSam
            && (isInitCondFrame || config.multimaskOutputForTracking)
            && (config.multimaskMinPtNum <= numPts && numPts <= config.multimaskMaxPtNum)
            && config.numMultimaskOutputs > 0
    }

    /// Runs one tracking step on a frame and stores its output in `state`.
    ///
    /// The mode follows from the inputs: mask-as-output (`maskInputs`), propagation-only (no
    /// prompts), interaction-only (points on a conditioning frame, or refinement with
    /// `prevSAMMaskLogits`), and propagation-and-interaction (points on a non-conditioning frame).
    ///
    /// - Parameter numFrames: total video length (limits the object-pointer range); nil means
    ///   `frameIndex + 1`.
    @discardableResult
    func trackStep(
        _ state: MultiplexTrackerState, frameIndex: Int, isInitCondFrame: Bool,
        features: TrackerFrameFeatures,
        pointInputs: PointInputs?, maskInputs: MLXArray?, numFrames: Int?, trackInReverse: Bool = false,
        runMemEncoder: Bool = true, prevSAMMaskLogits: MLXArray? = nil, objectsToInteract: [Int]? = nil
    ) -> FrameOutput {
        let (currentOut, _) = trackStepAux(
            state, frameIndex: frameIndex, isInitCondFrame: isInitCondFrame, features: features,
            pointInputs: pointInputs, maskInputs: maskInputs, numFrames: numFrames ?? frameIndex + 1,
            trackInReverse: trackInReverse, runMemEncoder: runMemEncoder,
            prevSAMMaskLogits: prevSAMMaskLogits,
            objectsToInteract: objectsToInteract, needAuxOutput: false)

        if isInitCondFrame {
            state.condFrameOutputs[frameIndex] = currentOut
        } else {
            state.nonCondFrameOutputs[frameIndex] = currentOut
        }

        // Prune stale non-conditioning outputs that can no longer be referenced by the memory
        // attention (bounds session memory)
        let maxLookback = max(
            config.maxObjectPointersInEncoder,
            2 + (config.numMaskmem - 2) * config.memoryTemporalStrideForEval)
        let cutoff = frameIndex - maxLookback
        for t in state.nonCondFrameOutputs.keys where t < cutoff {
            state.nonCondFrameOutputs[t] = nil
        }
        return currentOut
    }

    private enum TrackingMode {
        case maskAsOutput, propagationOnly, interactionOnly, propagationAndInteraction
    }

    func trackStepAux(
        _ state: MultiplexTrackerState, frameIndex: Int, isInitCondFrame: Bool,
        features: TrackerFrameFeatures,
        pointInputs: PointInputs?, maskInputs: MLXArray?, numFrames: Int, trackInReverse: Bool,
        runMemEncoder: Bool, prevSAMMaskLogits: MLXArray?, objectsToInteract: [Int]?, needAuxOutput: Bool
    ) -> (FrameOutput, TrackStepAuxOutput?) {
        let multiplexState = state.multiplexState
        let interactive = features.interactive
        let propagation = features.propagation

        // Determine the tracking mode
        let mode: TrackingMode
        if maskInputs != nil {
            mode = .maskAsOutput
        } else if pointInputs == nil {
            mode = .propagationOnly
        } else if prevSAMMaskLogits != nil || isInitCondFrame {
            mode = .interactionOnly
        } else if objectsToInteract != nil {
            mode = .propagationAndInteraction
        } else {
            preconditionFailure(
                "Unable to determine tracking mode: pointInputs given, no prevSAMMaskLogits, "
                    + "objectsToInteract nil, isInitCondFrame false")
        }

        if mode == .interactionOnly || mode == .propagationAndInteraction {
            precondition(interactive != nil)
        }
        if mode == .propagationOnly || mode == .propagationAndInteraction {
            precondition(propagation != nil)
        }

        var conditioningObjects = Set<Int>()
        var interactivePixFeat: MLXArray? = nil
        var samOutputs: SAMHeadOutput
        if mode == .maskAsOutput {
            precondition(config.useMaskInputAsOutputWithoutSam)
            let interactive = interactive!
            let pix = getInteractivePixMem(interactive.visionFeat)
            interactivePixFeat = pix
            samOutputs = useMaskAsOutput(
                backboneFeatures: pix, highResFeatures: interactive.highRes, maskInputs: maskInputs!,
                multiplexState: multiplexState)
            conditioningObjects.formUnion(0..<maskInputs!.dim(0))
        } else {
            var propagationOut: SAMHeadOutput? = nil
            if mode == .propagationOnly || mode == .propagationAndInteraction {
                let propagation = propagation!
                let pixFeatWithMem = prepareMemoryConditionedFeatures(
                    frameIndex: frameIndex, currentVisionFeat: propagation.visionFeat,
                    currentVisionPos: propagation.visionPos!, state: state, numFrames: numFrames,
                    trackInReverse: trackInReverse)
                let (b, h, w, c) = (
                    pixFeatWithMem.dim(0), pixFeatWithMem.dim(1), pixFeatWithMem.dim(2), pixFeatWithMem.dim(3)
                )
                propagationOut = forwardSAMHeads(
                    backboneFeatures: pixFeatWithMem.reshaped(b, h * w, c),
                    propagationHighResFeatures: propagation.highRes,
                    multimaskOutput: useMultimask(isInitCondFrame: isInitCondFrame, pointInputs: nil),
                    multiplexState: multiplexState)
            }

            var interactionOut: SAMHeadOutput? = nil
            if mode == .interactionOnly || mode == .propagationAndInteraction {
                let interactive = interactive!
                let pix = getInteractivePixMem(interactive.visionFeat)
                interactivePixFeat = pix
                let pointInputs = pointInputs!
                var maskPrompt: MLXArray? = nil
                if let prevSAMMaskLogits {
                    precondition(objectsToInteract != nil && mode != .propagationAndInteraction)
                    maskPrompt = prevSAMMaskLogits[Self.indexArray(objectsToInteract!)]
                } else if mode == .propagationAndInteraction {
                    // Use the propagated masks as mask input
                    maskPrompt = propagationOut!.lowResMasks[Self.indexArray(objectsToInteract!)]
                }
                if let objectsToInteract {
                    precondition(
                        pointInputs.coords.dim(0) == pointInputs.labels.dim(0)
                            && pointInputs.labels.dim(0) == objectsToInteract.count)
                }
                interactionOut = forwardSAMHeads(
                    backboneFeatures: pix, pointInputs: pointInputs, maskInputs: maskPrompt,
                    interactiveHighResFeatures: interactive.highRes,
                    multimaskOutput: useMultimask(isInitCondFrame: isInitCondFrame, pointInputs: pointInputs),
                    multiplexState: multiplexState)
                conditioningObjects.formUnion(
                    objectsToInteract ?? Array(multiplexState.getAllValidObjectIdx()))
            }

            if let p = propagationOut, let i = interactionOut {
                // Merge: replace the interacted objects in the propagated output
                let idx = Self.indexArray(objectsToInteract!)
                let merged = p
                merged.lowResMultimasks[idx] = i.lowResMultimasks
                merged.highResMultimasks[idx] = i.highResMultimasks
                merged.lowResMasks[idx] = i.lowResMasks
                merged.highResMasks[idx] = i.highResMasks
                merged.ious[idx] = i.ious
                merged.objectScoreLogits[idx] = i.objectScoreLogits
                merged.objPtr[idx] = i.objPtr
                samOutputs = merged
            } else {
                samOutputs = (propagationOut ?? interactionOut)!
            }
        }

        let currentOut = FrameOutput(
            conditioningObjects: conditioningObjects, pointInputs: pointInputs, maskInputs: maskInputs,
            predMasks: samOutputs.lowResMasks, predMasksHighRes: samOutputs.highResMasks,
            objectScoreLogits: samOutputs.objectScoreLogits)
        if config.useObjPtrsInEncoder {
            // Object pointers are stored in the multiplex space
            currentOut.objPtr = multiplexState.mux(samOutputs.objPtr)
        }

        // Encode the predicted masks into a new memory for future frames
        if runMemEncoder && config.numMaskmem > 0 {
            let (features, posEnc) = encodeNewMemory(
                currentVisionFeat: propagation!.visionFeat, predMasksHighRes: currentOut.predMasksHighRes,
                objectScoreLogits: currentOut.objectScoreLogits,
                conditioningObjects: currentOut.conditioningObjects, multiplexState: multiplexState)
            currentOut.maskmemFeatures = features
            currentOut.maskmemPosEnc = posEnc
        }

        if config.saveImageFeatures {
            currentOut.imageFeatures = propagation?.visionFeat
            currentOut.imagePosEnc = propagation?.visionPos
        }

        var aux: TrackStepAuxOutput? = nil
        if needAuxOutput {
            let interactive = interactive!
            aux = TrackStepAuxOutput(
                interactivePixFeat: interactivePixFeat ?? getInteractivePixMem(interactive.visionFeat),
                interactiveHighResFeatures: interactive.highRes,
                propagationVisionFeat: propagation?.visionFeat)
        }
        return (currentOut, aux)
    }

    /// Object indices as an int32 gather index (Python's `x[list]`).
    private static func indexArray(_ indices: [Int]) -> MLXArray {
        MLXArray(indices.map(Int32.init))
    }
}
