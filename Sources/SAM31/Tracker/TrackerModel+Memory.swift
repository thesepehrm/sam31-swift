// Port of mlx_vlm/models/sam3_1/tracker.py::{get_1d_sine_pe, select_closest_cond_frames,
// MultiplexTrackerModel._get_tpos_enc, _prepare_memory_conditioned_features, _encode_new_memory}
// (mlx-vlm 0.7.3)
import MLX
import MLXNN

/// 1D sine positional embedding as in the original Transformer paper: `(..., dim)`.
func get1DSinePE(_ posInds: MLXArray, dim: Int, temperature: Float = 10000.0) -> MLXArray {
    let peDim = dim / 2
    var dimT = MLXArray.arange(peDim).asType(.float32)
    dimT = pow(temperature, 2 * floorDivide(dimT, 2) / Float(peDim))
    let posEmbed = expandedDimensions(posInds, axis: -1) / dimT
    return concatenated([MLX.sin(posEmbed), MLX.cos(posEmbed)], axis: -1)
}

/// Selects up to `maxCondFrameNum` temporally closest conditioning frames.
///
/// Python iterates dicts in insertion order; Swift dictionaries have none, so frames are taken in
/// ascending frame order where Python would use insertion order. The memory attention is invariant
/// to the order of whole memory frames (every frame spans exactly one RoPE period, and each object
/// pointer carries its own temporal encoding), so this only changes float summation order.
///
/// - Returns: the selected frames in Python's order, and the unselected ones.
func selectClosestCondFrames(
    frameIndex: Int, condFrameOutputs: [Int: FrameOutput], maxCondFrameNum: Int
) -> (selected: [(Int, FrameOutput)], unselected: [Int: FrameOutput]) {
    let keys = condFrameOutputs.keys.sorted()
    if maxCondFrameNum == -1 || condFrameOutputs.count <= maxCondFrameNum {
        return (keys.map { ($0, condFrameOutputs[$0]!) }, [:])
    }

    precondition(maxCondFrameNum >= 2)
    var selected: [Int] = []
    // closest conditioning frame before and after `frameIndex`
    if let before = keys.filter({ $0 < frameIndex }).max() {
        selected.append(before)
    }
    if let after = keys.filter({ $0 >= frameIndex }).min() {
        selected.append(after)
    }
    // fill up with the temporally closest remaining frames
    let numRemain = maxCondFrameNum - selected.count
    let remain = keys.filter { !selected.contains($0) }
        .enumerated()
        .sorted { (abs($0.element - frameIndex), $0.offset) < (abs($1.element - frameIndex), $1.offset) }
        .prefix(numRemain)
        .map(\.element)
    selected += remain
    let unselected = condFrameOutputs.filter { !selected.contains($0.key) }
    return (selected.map { ($0, condFrameOutputs[$0]!) }, unselected)
}

/// Python's `a // b` (floor division) for ints.
private func floorDiv(_ a: Int, _ b: Int) -> Int {
    let q = a / b
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
}

extension MultiplexTrackerModel {
    // MARK: - Memory conditioning / encoding

    /// Temporal positional encoding for object pointers: `(T, C)`.
    func getTposEnc(_ relPosList: [Int], maxAbsPos: Int) -> MLXArray {
        let pos = MLXArray(relPosList.map(Float.init)) / Float(maxAbsPos - 1)
        let posEnc = get1DSinePE(pos, dim: hiddenDim)
        return temporalPositionalEncodingProjectionLayer(posEnc)
    }

    /// Fuses the current `(1, H, W, C)` features and pos enc with past memories.
    ///
    /// - Returns: `(numBuckets, H, W, C)` memory-conditioned features.
    func prepareMemoryConditionedFeatures(
        frameIndex: Int, currentVisionFeat: MLXArray, currentVisionPos: MLXArray,
        state: MultiplexTrackerState, numFrames: Int, trackInReverse: Bool
    ) -> MLXArray {
        let b = state.multiplexState.numBuckets
        let (h, w, c) = (currentVisionFeat.dim(1), currentVisionFeat.dim(2), currentVisionFeat.dim(3))
        let hw = h * w
        let visionFeat = broadcast(currentVisionFeat.reshaped(1, hw, c), to: [b, hw, c])
        let srcPos = broadcast(currentVisionPos.reshaped(1, hw, c), to: [b, hw, c])

        // Gather spatial mask memories, their pos encs, and image features
        var toCatPrompt: [MLXArray] = []
        var toCatPromptPos: [MLXArray] = []
        var toCatImage: [MLXArray] = []
        var toCatImagePos: [MLXArray] = []

        let (selectedCond, unselectedCond) = selectClosestCondFrames(
            frameIndex: frameIndex, condFrameOutputs: state.condFrameOutputs,
            maxCondFrameNum: config.maxCondFrameNum)

        let tposSignMul = trackInReverse ? -1 : 1
        var tPosAndPrevs: [(tPos: Int, prev: FrameOutput?, isCond: Bool)] = selectedCond.map { t, out in
            ((frameIndex - t) * tposSignMul, out, true)
        }

        // Last (numMaskmem - 1) frames as non-conditioning memory
        let numMaskmem = config.numMaskmem
        let r = config.memoryTemporalStrideForEval
        for tPos in 1..<numMaskmem {
            let tRel = numMaskmem - tPos
            let prevFrameIndex: Int
            if tRel == 1 {
                // the frame immediately before/after this frame
                prevFrameIndex = trackInReverse ? frameIndex + tRel : frameIndex - tRel
            } else if !trackInReverse {
                prevFrameIndex = floorDiv(frameIndex - 2, r) * r - (tRel - 2) * r
            } else {
                prevFrameIndex = -floorDiv(-(frameIndex + 2), r) * r + (tRel - 2) * r
            }
            let out = state.nonCondFrameOutputs[prevFrameIndex] ?? unselectedCond[prevFrameIndex]
            tPosAndPrevs.append((tPos, out, false))
        }

        for (tPos, prev, isCond) in tPosAndPrevs {
            guard let prev, let maskmemFeatures = prev.maskmemFeatures else { continue }
            toCatPrompt.append(maskmemFeatures.reshaped(b, hw, c))
            let memPos = prev.maskmemPosEnc!.reshaped(b, hw, c)

            let tIdx: Int
            if config.useMaskmemTposV2 {
                // out-of-range tPos maps to the last ("out-of-range") slot
                tIdx = (0 < tPos && tPos < numMaskmem) ? numMaskmem - tPos - 1 : numMaskmem - 1
            } else {
                tIdx = numMaskmem - (isCond ? 0 : tPos) - 1
            }
            let tpos = memoryTemporalPositionalEncoding[tIdx]
            toCatPromptPos.append(memPos + tpos.reshaped(1, 1, c))

            if config.saveImageFeatures {
                toCatImage.append(prev.imageFeatures!.reshaped(1, hw, c))
                toCatImagePos.append(prev.imagePosEnc!.reshaped(1, hw, c) + tpos.reshaped(1, 1, c))
            }
        }

        // Object pointers from past frames
        var numObjPtrTokens = 0
        if config.useObjPtrsInEncoder {
            let maxObjPtrs = min(numFrames, config.maxObjectPointersInEncoder)
            var posAndOuts: [(Int, FrameOutput)] = selectedCond.map { t, out in (abs(frameIndex - t), out) }
            if maxObjPtrs > 1 {
                for tDiff in 1..<maxObjPtrs {
                    let t = trackInReverse ? frameIndex + tDiff : frameIndex - tDiff
                    if t < 0 || t >= numFrames { break }
                    if let out = state.nonCondFrameOutputs[t] ?? unselectedCond[t] {
                        posAndOuts.append((tDiff, out))
                    }
                }
            }

            let filtered = posAndOuts.filter { $0.1.objPtr != nil }
            if !filtered.isEmpty {
                // muxed ptrs per frame: (B, M, C) -> (B, T*M, C)
                let objPtrs = concatenated(filtered.map { $0.1.objPtr! }, axis: 1)
                var objPos = getTposEnc(filtered.map(\.0), maxAbsPos: maxObjPtrs)  // (T, C)
                // Each frame contributes multiplexCount pointers
                objPos = repeated(objPos, count: multiplexCount, axis: 0)
                objPos = broadcast(objPos[.newAxis], to: [b, objPos.dim(0), objPos.dim(1)])

                toCatPrompt.append(objPtrs)
                toCatPromptPos.append(objPos)
                numObjPtrTokens = objPtrs.dim(1)
            }
        }

        if toCatPrompt.isEmpty {
            // No available memories; propagate from current features only
            return visionFeat.reshaped(b, h, w, c)
        }

        let memory = concatenated(toCatPrompt, axis: 1)
        let memoryPos = concatenated(toCatPromptPos, axis: 1)

        let memoryImage: MLXArray
        let memoryImagePos: MLXArray
        if config.saveImageFeatures {
            if toCatImage.isEmpty {
                return visionFeat.reshaped(b, h, w, c)
            }
            memoryImage = concatenated(toCatImage, axis: 1)
            memoryImagePos = concatenated(toCatImagePos, axis: 1)
        } else {
            (memoryImage, memoryImagePos) = (memory, memoryPos)
        }

        let pixFeatWithMem = memoryAttention(
            image: currentVisionFeat.reshaped(1, hw, c), src: visionFeat, memoryImage: memoryImage,
            memory: memory, srcPos: srcPos, memoryPos: memoryPos, memoryImagePos: memoryImagePos,
            numKExcludeRope: numObjPtrTokens)
        return pixFeatWithMem.reshaped(b, h, w, c)
    }

    /// The memory encoder's mask input: `(numBuckets, H_im, W_im, 2 * multiplexCount)` channel-last,
    /// muxed mask probabilities plus (with `condition_as_mask_input`) the per-object conditioning
    /// channel. This is the first half of `_encode_new_memory`.
    func memoryEncoderMaskInput(
        predMasksHighRes: MLXArray, conditioningObjects: Set<Int>, multiplexState: MultiplexState
    ) -> MLXArray {
        var maskForMem = predMasksHighRes
        if config.applySigmoidToMaskLogitsForMemEnc {
            maskForMem =
                sigmoid(predMasksHighRes) * config.sigmoidScaleForMemEnc + config.sigmoidBiasForMemEnc
        }

        // (the reference also computes unconditioned objects here, only used by the
        // object-conditional embeddings that SAM 3.1 disables)
        let conditioning = conditioningObjects.sorted()

        // (N, 1, H, W) -> mux -> (B, M, H, W)
        let masks0 = maskForMem[0..., 0]
        var muxMask = multiplexState.mux(masks0)

        if config.conditionAsMaskInput {
            // Extra per-object channel marking conditioning objects
            let n = maskForMem.dim(0)
            let condValues = MLXArray.full([n], values: MLXArray(config.conditionAsMaskInputBg))
            if !conditioning.isEmpty {
                condValues[MLXArray(conditioning.map(Int32.init))] = MLXArray(config.conditionAsMaskInputFg)
            }
            let embedded = broadcast(condValues.reshaped(n, 1, 1), to: masks0.shape)
            muxMask = concatenated([muxMask, multiplexState.mux(embedded)], axis: 1)
        }

        // (B, 2M, H, W) -> (B, H, W, 2M) channel-last for the convs
        return muxMask.transposed(0, 2, 3, 1)
    }

    /// Encodes the frame's predictions into a memory.
    ///
    /// - Parameters:
    ///   - predMasksHighRes: `(N, 1, H_im, W_im)` mask logits.
    ///   - objectScoreLogits: `(N, 1)`.
    /// - Returns: `(maskmemFeatures, maskmemPosEnc)`, each `(numBuckets, H, W, C)`.
    func encodeNewMemory(
        currentVisionFeat: MLXArray, predMasksHighRes: MLXArray, objectScoreLogits: MLXArray,
        conditioningObjects: Set<Int>, multiplexState: MultiplexState
    ) -> (features: MLXArray, posEnc: MLXArray) {
        let muxMask = memoryEncoderMaskInput(
            predMasksHighRes: predMasksHighRes, conditioningObjects: conditioningObjects,
            multiplexState: multiplexState)
        var (maskmemFeatures, maskmemPosEnc) = memoryEncoder(currentVisionFeat, masks: muxMask)

        // Add a projected embedding for each empty object slot
        var objLogits = objectScoreLogits
        let numMissing = multiplexState.totalValidEntries - objLogits.dim(0)
        if numMissing > 0 {
            let pad = MLXArray.zeros([numMissing] + objLogits.shape.dropFirst())
            objLogits = concatenated([objLogits, pad], axis: 0)
        } else if numMissing < 0 {
            objLogits = objLogits[..<multiplexState.totalValidEntries]
        }
        let appearing = multiplexState.mux(objLogits)
        let isObjAppearing = (appearing .> config.objectScoreLogitThreshold).asType(.float32)  // (B, M, 1)
        let noObjEmbed = ((1 - isObjAppearing) * noObjEmbedSpatial[.newAxis]).sum(axis: 1)  // (B, C)
        maskmemFeatures = maskmemFeatures + noObjEmbed.reshaped(noObjEmbed.dim(0), 1, 1, noObjEmbed.dim(1))

        return (maskmemFeatures, maskmemPosEnc)
    }
}
