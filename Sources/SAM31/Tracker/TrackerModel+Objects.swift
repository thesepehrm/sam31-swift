// Port of mlx_vlm/models/sam3_1/tracker.py::MultiplexTrackerModel.{_merge_mask_output,
// _reencode_memory, add_new_masks_to_existing_state, recondition_masks_in_existing_state,
// add_mask_prompt, propagate} (mlx-vlm 0.7.3)
import MLX

extension MultiplexTrackerModel {
    // MARK: - Dynamic object management

    /// Merges mask-encoded objects into `prevOutput` in place.
    ///
    /// Appends the new rows, or replaces the `objIdxs` rows when given (reconditioning);
    /// `conditioned` objects condition on this frame.
    func mergeMaskOutput(
        _ prevOutput: FrameOutput, _ maskOutput: SAMHeadOutput, multiplexState: MultiplexState,
        conditioned: [Int], objIdxs: [Int]? = nil, existingPointers: MLXArray? = nil
    ) {
        let (h, w) = (prevOutput.predMasks.dim(-2), prevOutput.predMasks.dim(-1))
        let predMasks = resizeTrackerMasks(maskOutput.lowResMasks, h: h, w: w, antialias: true)
        if let objIdxs {
            // In-place scatter into the stored arrays, as Python's `prev_output[key][obj_idxs] = val`
            let idx = Self.indexArray(objIdxs)
            prevOutput.predMasks[idx] = predMasks
            prevOutput.predMasksHighRes[idx] = maskOutput.highResMasks
            prevOutput.objectScoreLogits[idx] = maskOutput.objectScoreLogits
        } else {
            prevOutput.predMasks = concatenated([prevOutput.predMasks, predMasks], axis: 0)
            prevOutput.predMasksHighRes = concatenated(
                [prevOutput.predMasksHighRes, maskOutput.highResMasks], axis: 0)
            prevOutput.objectScoreLogits = concatenated(
                [prevOutput.objectScoreLogits, maskOutput.objectScoreLogits], axis: 0)
        }

        if config.useObjPtrsInEncoder {
            let pointers: MLXArray
            if let objIdxs {
                pointers = multiplexState.demux(prevOutput.objPtr!)
                pointers[Self.indexArray(objIdxs)] = maskOutput.objPtr
            } else {
                pointers = concatenated([existingPointers!, maskOutput.objPtr], axis: 0)
            }
            prevOutput.objPtr = multiplexState.mux(pointers)
        }

        prevOutput.conditioningObjects.formUnion(conditioned)
    }

    /// Re-encodes the spatial memory of `prevOutput` from its merged predictions.
    func reencodeMemory(
        _ prevOutput: FrameOutput, propagationVisionFeat: MLXArray?, multiplexState: MultiplexState
    ) {
        precondition(propagationVisionFeat != nil, "re-encoding memory needs propagation features")
        let (features, posEnc) = encodeNewMemory(
            currentVisionFeat: propagationVisionFeat!, predMasksHighRes: prevOutput.predMasksHighRes,
            objectScoreLogits: prevOutput.objectScoreLogits,
            conditioningObjects: prevOutput.conditioningObjects, multiplexState: multiplexState)
        prevOutput.maskmemFeatures = features
        prevOutput.maskmemPosEnc = posEnc
    }

    /// Appends new objects to an existing frame output and multiplex state (in place).
    ///
    /// - Parameters:
    ///   - newMasks: `(K, 1, H_im, W_im)` binary masks of the new objects.
    ///   - objIdxsInMask: `K` indices; only their count is checked, as in Python.
    ///   - areMasksFromPts: accepted for signature parity; Python does not read it.
    func addNewMasksToExistingState(
        interactivePixFeat: MLXArray, interactiveHighResFeatures: [MLXArray],
        propagationVisionFeat: MLXArray?,
        newMasks: MLXArray, objIdxsInMask: [Int], objIDsInMask: [Int]?, prevOutput: FrameOutput,
        state: MultiplexTrackerState, addMaskToMemory: Bool = true, areMasksFromPts: Bool = false,
        allowNewBuckets: Bool = false, preferNewBuckets: Bool = false
    ) {
        precondition(config.useMaskInputAsOutputWithoutSam)
        let multiplexState = state.multiplexState
        let numNewObjects = newMasks.dim(0)
        precondition(numNewObjects == objIdxsInMask.count)

        // Python demuxes unconditionally (a missing obj_ptr raises there).
        let existingPointers = multiplexState.demux(prevOutput.objPtr!)

        // Step 1: extend the multiplex state
        let newObjectIdx = multiplexState.findNextBatchOfAvailableIndices(
            numObjects: numNewObjects, allowNewBuckets: allowNewBuckets, preferNewBuckets: preferNewBuckets)
        multiplexState.addObjects(
            objectIndices: newObjectIdx, objectIDs: objIDsInMask, allowNewBuckets: allowNewBuckets,
            preferNewBuckets: preferNewBuckets)

        // Step 2: encode the incoming masks
        let maskOutput = useMaskAsOutput(
            backboneFeatures: interactivePixFeat, highResFeatures: interactiveHighResFeatures,
            maskInputs: newMasks, multiplexState: multiplexState)

        // Step 3: match the high-res resolutions, then append the new objects
        let res = maskOutput.highResMasks.dim(-1)
        prevOutput.predMasksHighRes = resizeTrackerMasks(prevOutput.predMasksHighRes, h: res, w: res)
        mergeMaskOutput(
            prevOutput, maskOutput, multiplexState: multiplexState, conditioned: newObjectIdx,
            existingPointers: existingPointers)

        // Step 4: re-encode the spatial memory
        if addMaskToMemory {
            precondition(prevOutput.predMasksHighRes.dim(0) == multiplexState.totalValidEntries)
            reencodeMemory(
                prevOutput, propagationVisionFeat: propagationVisionFeat, multiplexState: multiplexState)
        }
    }

    /// Reconditions existing objects with new masks (in place).
    ///
    /// - Parameter newMasks: `(K, 1, H_im, W_im)` binary masks for the objects `objIdxsInMask`.
    func reconditionMasksInExistingState(
        interactivePixFeat: MLXArray, interactiveHighResFeatures: [MLXArray],
        propagationVisionFeat: MLXArray?,
        newMasks: MLXArray, objIdxsInMask: [Int], objIDsInMask: [Int]?, prevOutput: FrameOutput,
        state: MultiplexTrackerState, addMaskToMemory: Bool = true
    ) {
        precondition(config.useMaskInputAsOutputWithoutSam)
        let multiplexState = state.multiplexState
        precondition(newMasks.dim(0) == objIdxsInMask.count)

        // Step 1: encode the incoming masks
        let maskOutput = useMaskAsOutput(
            backboneFeatures: interactivePixFeat, highResFeatures: interactiveHighResFeatures,
            maskInputs: newMasks, multiplexState: multiplexState)

        // Step 2: replace the reconditioned objects in the existing state
        mergeMaskOutput(
            prevOutput, maskOutput, multiplexState: multiplexState, conditioned: objIdxsInMask,
            objIdxs: objIdxsInMask)

        // Step 3: re-encode the spatial memory
        if addMaskToMemory {
            reencodeMemory(
                prevOutput, propagationVisionFeat: propagationVisionFeat, multiplexState: multiplexState)
        }
    }

    // MARK: - Convenience session API

    /// Initializes (or extends) a session with mask prompts on a frame.
    ///
    /// - Parameter masks: `(N, H_im, W_im)` or `(N, 1, H_im, W_im)` binary/float masks at image
    ///   resolution.
    /// - Returns: the frame's output.
    @discardableResult
    func addMaskPrompt(
        _ state: MultiplexTrackerState, frameIndex: Int, features: TrackerFrameFeatures, masks: MLXArray,
        objectIDs: [Int]?
    ) -> FrameOutput {
        let masks = masks.ndim == 3 ? expandedDimensions(masks, axis: 1) : masks

        guard let prevOutput = state.condFrameOutputs[frameIndex] ?? state.nonCondFrameOutputs[frameIndex]
        else {
            // Fresh frame: the masks are the conditioning input (mask-as-output)
            return trackStep(
                state, frameIndex: frameIndex, isInitCondFrame: true, features: features, pointInputs: nil,
                maskInputs: masks, numFrames: frameIndex + 1)
        }

        // Frame already tracked: merge the masks as new objects
        let newIdxs = state.multiplexState.findNextBatchOfAvailableIndices(
            numObjects: masks.dim(0), allowNewBuckets: true)
        let interactive = features.interactive!
        addNewMasksToExistingState(
            interactivePixFeat: getInteractivePixMem(interactive.visionFeat),
            interactiveHighResFeatures: interactive.highRes,
            propagationVisionFeat: features.propagation!.visionFeat, newMasks: masks, objIdxsInMask: newIdxs,
            objIDsInMask: objectIDs, prevOutput: prevOutput, state: state)
        return prevOutput
    }

    /// Propagates all tracked objects to a new frame.
    ///
    /// - Returns: the frame's output; per-object masks are `predMasks` `(N, 1, h, w)` and
    ///   `predMasksHighRes`.
    @discardableResult
    func propagate(
        _ state: MultiplexTrackerState, frameIndex: Int, features: TrackerFrameFeatures, numFrames: Int?,
        runMemEncoder: Bool = true
    ) -> FrameOutput {
        trackStep(
            state, frameIndex: frameIndex, isInitCondFrame: false, features: features, pointInputs: nil,
            maskInputs: nil, numFrames: numFrames, runMemEncoder: runMemEncoder)
    }
}
