// Port of mlx_vlm/models/sam3_1/tracker.py::MultiplexTrackerModel (mlx-vlm 0.7.3).
// Partial until Task 12 fills in the tracker: only the SAM prompt encoder and mask decoders so far.
import MLXNN

final class MultiplexTrackerModel: Module {
    /// Interactive SAM components (point/box prompts, single object slot).
    @ModuleInfo(key: "interactive_sam_prompt_encoder") var interactiveSamPromptEncoder: SAMPromptEncoder
    @ModuleInfo(key: "interactive_sam_mask_decoder") var interactiveSamMaskDecoder: MultiplexMaskDecoder
    /// Propagation SAM mask decoder (multiplex: 16 objects).
    @ModuleInfo(key: "sam_mask_decoder") var samMaskDecoder: MultiplexMaskDecoder

    init(_ config: TrackerConfig) {
        _interactiveSamPromptEncoder.wrappedValue = SAMPromptEncoder(config.promptEncoderConfig)
        _interactiveSamMaskDecoder.wrappedValue = MultiplexMaskDecoder(config.interactiveMaskDecoderConfig)
        _samMaskDecoder.wrappedValue = MultiplexMaskDecoder(config.maskDecoderConfig)
    }
}
