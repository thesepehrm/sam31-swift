// Port of mlx_vlm/models/sam3/config.py (ViTConfig, TextEncoderConfig, DETREncoderConfig,
// DETRDecoderConfig, GeometryEncoderConfig, DetectorMaskDecoderConfig, PromptEncoderConfig) and
// mlx_vlm/models/sam3_1/config.py (VisionEncoderConfig, TrackerMaskDecoderConfig, TrackerConfig,
// DetectorConfig, ModelConfig) (mlx-vlm 0.7.3)
//
// Every field defaults to the Python dataclass default. A key that is missing or `null` in the JSON
// keeps its default, and unknown keys are ignored, which matches `BaseModelConfig.from_dict`.
import Foundation

extension KeyedDecodingContainer {
    /// Decodes `key`, falling back to `fallback` when the key is absent or `null`.
    fileprivate func decode<T: Decodable>(_ key: Key, or fallback: T) throws -> T {
        try decodeIfPresent(T.self, forKey: key) ?? fallback
    }
}

// MARK: - sam3/config.py

struct ViTConfig: Codable, Sendable {
    var modelType = "sam3_vit_model"
    var hiddenSize = 1024
    var numHiddenLayers = 32
    var numAttentionHeads = 16
    var intermediateSize = 4736
    var hiddenAct = "gelu"
    var imageSize = 1008
    var patchSize = 14
    var numChannels = 3
    var windowSize = 24
    var globalAttnIndexes = [7, 15, 23, 31]
    var qkvBias = true
    var ropeTheta: Float = 10000.0
    var pretrainImageSize = 336
    var layerNormEps: Float = 1e-6
    var layerScaleInitValue: Float? = nil
    var hiddenDropout: Float = 0.0
    var attentionDropout: Float = 0.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case windowSize = "window_size"
        case globalAttnIndexes = "global_attn_indexes"
        case qkvBias = "qkv_bias"
        case ropeTheta = "rope_theta"
        case pretrainImageSize = "pretrain_image_size"
        case layerNormEps = "layer_norm_eps"
        case layerScaleInitValue = "layer_scale_init_value"
        case hiddenDropout = "hidden_dropout"
        case attentionDropout = "attention_dropout"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        numHiddenLayers = try c.decode(.numHiddenLayers, or: numHiddenLayers)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        intermediateSize = try c.decode(.intermediateSize, or: intermediateSize)
        hiddenAct = try c.decode(.hiddenAct, or: hiddenAct)
        imageSize = try c.decode(.imageSize, or: imageSize)
        patchSize = try c.decode(.patchSize, or: patchSize)
        numChannels = try c.decode(.numChannels, or: numChannels)
        windowSize = try c.decode(.windowSize, or: windowSize)
        globalAttnIndexes = try c.decode(.globalAttnIndexes, or: globalAttnIndexes)
        qkvBias = try c.decode(.qkvBias, or: qkvBias)
        ropeTheta = try c.decode(.ropeTheta, or: ropeTheta)
        pretrainImageSize = try c.decode(.pretrainImageSize, or: pretrainImageSize)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
        layerScaleInitValue = try c.decodeIfPresent(Float.self, forKey: .layerScaleInitValue)
        hiddenDropout = try c.decode(.hiddenDropout, or: hiddenDropout)
        attentionDropout = try c.decode(.attentionDropout, or: attentionDropout)
    }
}

struct TextEncoderConfig: Codable, Sendable {
    var modelType = "clip_text_model"
    var hiddenSize = 1024
    var numHiddenLayers = 24
    var numAttentionHeads = 16
    var intermediateSize = 4096
    var hiddenAct = "gelu"
    var vocabSize = 49408
    var maxPositionEmbeddings = 32
    var projectionDim = 512
    var layerNormEps: Float = 1e-5
    var attentionDropout: Float = 0.0
    var bosTokenId = 49406
    var eosTokenId = 49407
    var padTokenId = 1

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case projectionDim = "projection_dim"
        case layerNormEps = "layer_norm_eps"
        case attentionDropout = "attention_dropout"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        numHiddenLayers = try c.decode(.numHiddenLayers, or: numHiddenLayers)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        intermediateSize = try c.decode(.intermediateSize, or: intermediateSize)
        hiddenAct = try c.decode(.hiddenAct, or: hiddenAct)
        vocabSize = try c.decode(.vocabSize, or: vocabSize)
        maxPositionEmbeddings = try c.decode(.maxPositionEmbeddings, or: maxPositionEmbeddings)
        projectionDim = try c.decode(.projectionDim, or: projectionDim)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
        attentionDropout = try c.decode(.attentionDropout, or: attentionDropout)
        bosTokenId = try c.decode(.bosTokenId, or: bosTokenId)
        eosTokenId = try c.decode(.eosTokenId, or: eosTokenId)
        padTokenId = try c.decode(.padTokenId, or: padTokenId)
    }
}

struct DETREncoderConfig: Codable, Sendable {
    var modelType = "sam3_detr_encoder"
    var hiddenSize = 256
    var numLayers = 6
    var numAttentionHeads = 8
    var intermediateSize = 2048
    var hiddenAct = "relu"
    var dropout: Float = 0.1
    var layerNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case dropout
        case layerNormEps = "layer_norm_eps"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        numLayers = try c.decode(.numLayers, or: numLayers)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        intermediateSize = try c.decode(.intermediateSize, or: intermediateSize)
        hiddenAct = try c.decode(.hiddenAct, or: hiddenAct)
        dropout = try c.decode(.dropout, or: dropout)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
    }
}

struct DETRDecoderConfig: Codable, Sendable {
    var modelType = "sam3_detr_decoder"
    var hiddenSize = 256
    var numLayers = 6
    var numAttentionHeads = 8
    var numQueries = 200
    var intermediateSize = 2048
    var hiddenAct = "relu"
    var dropout: Float = 0.1
    var layerNormEps: Float = 1e-6
    var boxRpbMode = "log"
    var usePresenceToken = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case numQueries = "num_queries"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case dropout
        case layerNormEps = "layer_norm_eps"
        case boxRpbMode = "box_rpb_mode"
        case usePresenceToken = "use_presence_token"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        numLayers = try c.decode(.numLayers, or: numLayers)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        numQueries = try c.decode(.numQueries, or: numQueries)
        intermediateSize = try c.decode(.intermediateSize, or: intermediateSize)
        hiddenAct = try c.decode(.hiddenAct, or: hiddenAct)
        dropout = try c.decode(.dropout, or: dropout)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
        boxRpbMode = try c.decode(.boxRpbMode, or: boxRpbMode)
        usePresenceToken = try c.decode(.usePresenceToken, or: usePresenceToken)
    }
}

struct GeometryEncoderConfig: Codable, Sendable {
    var modelType = "sam3_geometry_encoder"
    var hiddenSize = 256
    var numLayers = 3
    var numAttentionHeads = 8
    var intermediateSize = 2048
    var hiddenAct = "relu"
    var dropout: Float = 0.1
    var roiSize = 7
    var layerNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case dropout
        case roiSize = "roi_size"
        case layerNormEps = "layer_norm_eps"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        numLayers = try c.decode(.numLayers, or: numLayers)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        intermediateSize = try c.decode(.intermediateSize, or: intermediateSize)
        hiddenAct = try c.decode(.hiddenAct, or: hiddenAct)
        dropout = try c.decode(.dropout, or: dropout)
        roiSize = try c.decode(.roiSize, or: roiSize)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
    }
}

struct DetectorMaskDecoderConfig: Codable, Sendable {
    var modelType = "sam3_mask_decoder"
    var hiddenSize = 256
    var numAttentionHeads = 8
    var numUpsamplingStages = 3
    var dropout: Float = 0.0
    var layerNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case numUpsamplingStages = "num_upsampling_stages"
        case dropout
        case layerNormEps = "layer_norm_eps"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        numUpsamplingStages = try c.decode(.numUpsamplingStages, or: numUpsamplingStages)
        dropout = try c.decode(.dropout, or: dropout)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
    }
}

struct PromptEncoderConfig: Codable, Sendable {
    var hiddenSize = 256
    var imageSize = 1008
    var patchSize = 14
    var maskInputChannels = 16
    var numPointEmbeddings = 4
    var hiddenAct = "gelu"
    var scale = 1

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case maskInputChannels = "mask_input_channels"
        case numPointEmbeddings = "num_point_embeddings"
        case hiddenAct = "hidden_act"
        case scale
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        imageSize = try c.decode(.imageSize, or: imageSize)
        patchSize = try c.decode(.patchSize, or: patchSize)
        maskInputChannels = try c.decode(.maskInputChannels, or: maskInputChannels)
        numPointEmbeddings = try c.decode(.numPointEmbeddings, or: numPointEmbeddings)
        hiddenAct = try c.decode(.hiddenAct, or: hiddenAct)
        scale = try c.decode(.scale, or: scale)
    }
}

// MARK: - sam3_1/config.py

/// SAM 3.1 vision encoder: TriViTDetNeck with 3 scales.
struct VisionEncoderConfig: Codable, Sendable {
    var modelType = "sam3_vision_model"
    var backboneConfig = ViTConfig()
    var fpnHiddenSize = 256
    var fpnKernelSize = 2
    var fpnStride = 2
    /// SAM 3.1: only 3 scales (no 0.5x downsample).
    var scaleFactors: [Float] = [4.0, 2.0, 1.0]
    var numFeatureLevels = 3
    var backboneFeatureSizes = [[288, 288], [144, 144], [72, 72]]
    var layerNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneConfig = "backbone_config"
        case fpnHiddenSize = "fpn_hidden_size"
        case fpnKernelSize = "fpn_kernel_size"
        case fpnStride = "fpn_stride"
        case scaleFactors = "scale_factors"
        case numFeatureLevels = "num_feature_levels"
        case backboneFeatureSizes = "backbone_feature_sizes"
        case layerNormEps = "layer_norm_eps"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        backboneConfig = try c.decode(.backboneConfig, or: backboneConfig)
        fpnHiddenSize = try c.decode(.fpnHiddenSize, or: fpnHiddenSize)
        fpnKernelSize = try c.decode(.fpnKernelSize, or: fpnKernelSize)
        fpnStride = try c.decode(.fpnStride, or: fpnStride)
        scaleFactors = try c.decode(.scaleFactors, or: scaleFactors)
        numFeatureLevels = try c.decode(.numFeatureLevels, or: numFeatureLevels)
        backboneFeatureSizes = try c.decode(.backboneFeatureSizes, or: backboneFeatureSizes)
        layerNormEps = try c.decode(.layerNormEps, or: layerNormEps)
    }
}

/// SAM 3.1 tracker mask decoder, multiplex version.
struct TrackerMaskDecoderConfig: Codable, Sendable {
    var hiddenSize = 256
    var numHiddenLayers = 2
    var numAttentionHeads = 8
    var attentionDownsampleRate = 2
    var numMultimaskOutputs = 3
    var mlpDim = 2048
    var dynamicMultimaskViaStability = true
    var dynamicMultimaskStabilityDelta: Float = 0.05
    var dynamicMultimaskStabilityThresh: Float = 0.98
    /// SAM 3.1 multiplex.
    var multiplexCount = 16
    /// Propagation decoder has no single-mask token (multimask tokens only).
    var multimaskOutputsOnly = false
    /// Use multimask tokens (not the single-mask token) for object pointers.
    var useMultimaskTokenForObjPtr = true

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case attentionDownsampleRate = "attention_downsample_rate"
        case numMultimaskOutputs = "num_multimask_outputs"
        case mlpDim = "mlp_dim"
        case dynamicMultimaskViaStability = "dynamic_multimask_via_stability"
        case dynamicMultimaskStabilityDelta = "dynamic_multimask_stability_delta"
        case dynamicMultimaskStabilityThresh = "dynamic_multimask_stability_thresh"
        case multiplexCount = "multiplex_count"
        case multimaskOutputsOnly = "multimask_outputs_only"
        case useMultimaskTokenForObjPtr = "use_multimask_token_for_obj_ptr"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(.hiddenSize, or: hiddenSize)
        numHiddenLayers = try c.decode(.numHiddenLayers, or: numHiddenLayers)
        numAttentionHeads = try c.decode(.numAttentionHeads, or: numAttentionHeads)
        attentionDownsampleRate = try c.decode(.attentionDownsampleRate, or: attentionDownsampleRate)
        numMultimaskOutputs = try c.decode(.numMultimaskOutputs, or: numMultimaskOutputs)
        mlpDim = try c.decode(.mlpDim, or: mlpDim)
        dynamicMultimaskViaStability = try c.decode(
            .dynamicMultimaskViaStability, or: dynamicMultimaskViaStability)
        dynamicMultimaskStabilityDelta = try c.decode(
            .dynamicMultimaskStabilityDelta, or: dynamicMultimaskStabilityDelta)
        dynamicMultimaskStabilityThresh = try c.decode(
            .dynamicMultimaskStabilityThresh, or: dynamicMultimaskStabilityThresh)
        multiplexCount = try c.decode(.multiplexCount, or: multiplexCount)
        multimaskOutputsOnly = try c.decode(.multimaskOutputsOnly, or: multimaskOutputsOnly)
        useMultimaskTokenForObjPtr = try c.decode(
            .useMultimaskTokenForObjPtr, or: useMultimaskTokenForObjPtr)
    }
}

/// SAM 3.1 tracker: Object Multiplex.
struct TrackerConfig: Codable, Sendable {
    var modelType = "sam3.1_tracker_video"
    var imageSize = 1008
    var visionConfig = VisionEncoderConfig()
    /// Propagation decoder: multimask tokens only (the `__post_init__` override).
    var maskDecoderConfig: TrackerMaskDecoderConfig = {
        var c = TrackerMaskDecoderConfig()
        c.multimaskOutputsOnly = true
        return c
    }()
    var promptEncoderConfig = PromptEncoderConfig()

    // Multiplex
    var multiplexCount = 16

    // Memory attention (decoupled)
    var memoryAttentionFeedForwardHiddenSize = 2048
    var memoryAttentionHiddenSize = 256
    var memoryAttentionNumAttentionHeads = 8
    var memoryAttentionNumLayers = 4
    var memoryAttentionRopeFeatSizes = [72, 72]
    var memoryAttentionRopeTheta: Float = 10000.0

    // Memory encoder: mask downsampler (SAM 3.1: dim = out_dim = 256)
    var maskDownsamplerEmbedDim = 256
    var maskDownsamplerFirstChannels = 16
    var maskDownsamplerInputSize = 1152
    var maskDownsamplerKernelSize = 3
    var maskDownsamplerPadding = 1
    var maskDownsamplerStride = 2
    var memoryEncoderHiddenSize = 256

    // Memory fuser (CXBlock)
    var memoryFuserEmbedDim = 256
    var memoryFuserIntermediateDim = 1024
    var memoryFuserKernelSize = 7
    var memoryFuserLayerScaleInitValue: Float = 1e-6
    var memoryFuserNumLayers = 2
    var memoryFuserPadding = 3

    // Memory retrieval over past frames
    var maxCondFrameNum = 4
    var maxObjectPointersInEncoder = 16
    var memoryTemporalStrideForEval = 1
    var numMaskmem = 7
    var saveImageFeatures = true
    var useMaskmemTposV2 = true

    // Memory encoding: mask -> memory (SAM 3.1: sigmoid(logits) * 2.0 - 1.0)
    var applySigmoidToMaskLogitsForMemEnc = true
    var sigmoidBiasForMemEnc: Float = -1.0
    var sigmoidScaleForMemEnc: Float = 2.0
    var conditionAsMaskInput = true
    var conditionAsMaskInputBg: Float = 0.0
    var conditionAsMaskInputFg: Float = 1.0

    // SAM head behavior (multimask, mask-as-output)
    var directlyAddNoMemEmbed = true
    var multimaskMaxPtNum = 1
    var multimaskMinPtNum = 0
    var multimaskOutputForTracking = true
    var multimaskOutputInSam = true
    var numMultimaskOutputs = 3
    var useMaskInputAsOutputWithoutSam = true

    // Object presence scores and pointers
    var addOutputSuppressionEmbeddings = true
    var fixedNoObjPtr = true
    var objectScoreLogitThreshold: Float = 0.0
    var predObjScores = true
    var useLinearNoObjPtr = true
    var useNoObjPtr = true
    var useObjPtrsInEncoder = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case imageSize = "image_size"
        case visionConfig = "vision_config"
        case maskDecoderConfig = "mask_decoder_config"
        case promptEncoderConfig = "prompt_encoder_config"
        case multiplexCount = "multiplex_count"
        case memoryAttentionFeedForwardHiddenSize = "memory_attention_feed_forward_hidden_size"
        case memoryAttentionHiddenSize = "memory_attention_hidden_size"
        case memoryAttentionNumAttentionHeads = "memory_attention_num_attention_heads"
        case memoryAttentionNumLayers = "memory_attention_num_layers"
        case memoryAttentionRopeFeatSizes = "memory_attention_rope_feat_sizes"
        case memoryAttentionRopeTheta = "memory_attention_rope_theta"
        case maskDownsamplerEmbedDim = "mask_downsampler_embed_dim"
        case maskDownsamplerFirstChannels = "mask_downsampler_first_channels"
        case maskDownsamplerInputSize = "mask_downsampler_input_size"
        case maskDownsamplerKernelSize = "mask_downsampler_kernel_size"
        case maskDownsamplerPadding = "mask_downsampler_padding"
        case maskDownsamplerStride = "mask_downsampler_stride"
        case memoryEncoderHiddenSize = "memory_encoder_hidden_size"
        case memoryFuserEmbedDim = "memory_fuser_embed_dim"
        case memoryFuserIntermediateDim = "memory_fuser_intermediate_dim"
        case memoryFuserKernelSize = "memory_fuser_kernel_size"
        case memoryFuserLayerScaleInitValue = "memory_fuser_layer_scale_init_value"
        case memoryFuserNumLayers = "memory_fuser_num_layers"
        case memoryFuserPadding = "memory_fuser_padding"
        case maxCondFrameNum = "max_cond_frame_num"
        case maxObjectPointersInEncoder = "max_object_pointers_in_encoder"
        case memoryTemporalStrideForEval = "memory_temporal_stride_for_eval"
        case numMaskmem = "num_maskmem"
        case saveImageFeatures = "save_image_features"
        case useMaskmemTposV2 = "use_maskmem_tpos_v2"
        case applySigmoidToMaskLogitsForMemEnc = "apply_sigmoid_to_mask_logits_for_mem_enc"
        case sigmoidBiasForMemEnc = "sigmoid_bias_for_mem_enc"
        case sigmoidScaleForMemEnc = "sigmoid_scale_for_mem_enc"
        case conditionAsMaskInput = "condition_as_mask_input"
        case conditionAsMaskInputBg = "condition_as_mask_input_bg"
        case conditionAsMaskInputFg = "condition_as_mask_input_fg"
        case directlyAddNoMemEmbed = "directly_add_no_mem_embed"
        case multimaskMaxPtNum = "multimask_max_pt_num"
        case multimaskMinPtNum = "multimask_min_pt_num"
        case multimaskOutputForTracking = "multimask_output_for_tracking"
        case multimaskOutputInSam = "multimask_output_in_sam"
        case numMultimaskOutputs = "num_multimask_outputs"
        case useMaskInputAsOutputWithoutSam = "use_mask_input_as_output_without_sam"
        case addOutputSuppressionEmbeddings = "add_output_suppression_embeddings"
        case fixedNoObjPtr = "fixed_no_obj_ptr"
        case objectScoreLogitThreshold = "object_score_logit_threshold"
        case predObjScores = "pred_obj_scores"
        case useLinearNoObjPtr = "use_linear_no_obj_ptr"
        case useNoObjPtr = "use_no_obj_ptr"
        case useObjPtrsInEncoder = "use_obj_ptrs_in_encoder"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        imageSize = try c.decode(.imageSize, or: imageSize)
        visionConfig = try c.decode(.visionConfig, or: visionConfig)
        maskDecoderConfig = try c.decode(.maskDecoderConfig, or: maskDecoderConfig)
        promptEncoderConfig = try c.decode(.promptEncoderConfig, or: promptEncoderConfig)
        multiplexCount = try c.decode(.multiplexCount, or: multiplexCount)
        memoryAttentionFeedForwardHiddenSize = try c.decode(
            .memoryAttentionFeedForwardHiddenSize, or: memoryAttentionFeedForwardHiddenSize)
        memoryAttentionHiddenSize = try c.decode(.memoryAttentionHiddenSize, or: memoryAttentionHiddenSize)
        memoryAttentionNumAttentionHeads = try c.decode(
            .memoryAttentionNumAttentionHeads, or: memoryAttentionNumAttentionHeads)
        memoryAttentionNumLayers = try c.decode(.memoryAttentionNumLayers, or: memoryAttentionNumLayers)
        memoryAttentionRopeFeatSizes = try c.decode(
            .memoryAttentionRopeFeatSizes, or: memoryAttentionRopeFeatSizes)
        memoryAttentionRopeTheta = try c.decode(.memoryAttentionRopeTheta, or: memoryAttentionRopeTheta)
        maskDownsamplerEmbedDim = try c.decode(.maskDownsamplerEmbedDim, or: maskDownsamplerEmbedDim)
        maskDownsamplerFirstChannels = try c.decode(
            .maskDownsamplerFirstChannels, or: maskDownsamplerFirstChannels)
        maskDownsamplerInputSize = try c.decode(.maskDownsamplerInputSize, or: maskDownsamplerInputSize)
        maskDownsamplerKernelSize = try c.decode(.maskDownsamplerKernelSize, or: maskDownsamplerKernelSize)
        maskDownsamplerPadding = try c.decode(.maskDownsamplerPadding, or: maskDownsamplerPadding)
        maskDownsamplerStride = try c.decode(.maskDownsamplerStride, or: maskDownsamplerStride)
        memoryEncoderHiddenSize = try c.decode(.memoryEncoderHiddenSize, or: memoryEncoderHiddenSize)
        memoryFuserEmbedDim = try c.decode(.memoryFuserEmbedDim, or: memoryFuserEmbedDim)
        memoryFuserIntermediateDim = try c.decode(
            .memoryFuserIntermediateDim, or: memoryFuserIntermediateDim)
        memoryFuserKernelSize = try c.decode(.memoryFuserKernelSize, or: memoryFuserKernelSize)
        memoryFuserLayerScaleInitValue = try c.decode(
            .memoryFuserLayerScaleInitValue, or: memoryFuserLayerScaleInitValue)
        memoryFuserNumLayers = try c.decode(.memoryFuserNumLayers, or: memoryFuserNumLayers)
        memoryFuserPadding = try c.decode(.memoryFuserPadding, or: memoryFuserPadding)
        maxCondFrameNum = try c.decode(.maxCondFrameNum, or: maxCondFrameNum)
        maxObjectPointersInEncoder = try c.decode(
            .maxObjectPointersInEncoder, or: maxObjectPointersInEncoder)
        memoryTemporalStrideForEval = try c.decode(
            .memoryTemporalStrideForEval, or: memoryTemporalStrideForEval)
        numMaskmem = try c.decode(.numMaskmem, or: numMaskmem)
        saveImageFeatures = try c.decode(.saveImageFeatures, or: saveImageFeatures)
        useMaskmemTposV2 = try c.decode(.useMaskmemTposV2, or: useMaskmemTposV2)
        applySigmoidToMaskLogitsForMemEnc = try c.decode(
            .applySigmoidToMaskLogitsForMemEnc, or: applySigmoidToMaskLogitsForMemEnc)
        sigmoidBiasForMemEnc = try c.decode(.sigmoidBiasForMemEnc, or: sigmoidBiasForMemEnc)
        sigmoidScaleForMemEnc = try c.decode(.sigmoidScaleForMemEnc, or: sigmoidScaleForMemEnc)
        conditionAsMaskInput = try c.decode(.conditionAsMaskInput, or: conditionAsMaskInput)
        conditionAsMaskInputBg = try c.decode(.conditionAsMaskInputBg, or: conditionAsMaskInputBg)
        conditionAsMaskInputFg = try c.decode(.conditionAsMaskInputFg, or: conditionAsMaskInputFg)
        directlyAddNoMemEmbed = try c.decode(.directlyAddNoMemEmbed, or: directlyAddNoMemEmbed)
        multimaskMaxPtNum = try c.decode(.multimaskMaxPtNum, or: multimaskMaxPtNum)
        multimaskMinPtNum = try c.decode(.multimaskMinPtNum, or: multimaskMinPtNum)
        multimaskOutputForTracking = try c.decode(
            .multimaskOutputForTracking, or: multimaskOutputForTracking)
        multimaskOutputInSam = try c.decode(.multimaskOutputInSam, or: multimaskOutputInSam)
        numMultimaskOutputs = try c.decode(.numMultimaskOutputs, or: numMultimaskOutputs)
        useMaskInputAsOutputWithoutSam = try c.decode(
            .useMaskInputAsOutputWithoutSam, or: useMaskInputAsOutputWithoutSam)
        addOutputSuppressionEmbeddings = try c.decode(
            .addOutputSuppressionEmbeddings, or: addOutputSuppressionEmbeddings)
        fixedNoObjPtr = try c.decode(.fixedNoObjPtr, or: fixedNoObjPtr)
        objectScoreLogitThreshold = try c.decode(.objectScoreLogitThreshold, or: objectScoreLogitThreshold)
        predObjScores = try c.decode(.predObjScores, or: predObjScores)
        useLinearNoObjPtr = try c.decode(.useLinearNoObjPtr, or: useLinearNoObjPtr)
        useNoObjPtr = try c.decode(.useNoObjPtr, or: useNoObjPtr)
        useObjPtrsInEncoder = try c.decode(.useObjPtrsInEncoder, or: useObjPtrsInEncoder)

        // __post_init__: the facebook/sam3.1 HF config carries a stale SAM 3 (non-multiplex)
        // tracker_config. Re-pin the SAM 3.1 multiplex architectural constants.
        if modelType == "sam3_tracker_video" {
            memoryAttentionNumAttentionHeads = 8
            sigmoidScaleForMemEnc = 2.0
            sigmoidBiasForMemEnc = -1.0
        }
        // Propagation decoder: multimask tokens only (no single-mask token).
        maskDecoderConfig.multimaskOutputsOnly = true
    }
}

/// SAM 3.1 detector config.
struct DetectorConfig: Codable, Sendable {
    var modelType = "sam3.1"
    var visionConfig = VisionEncoderConfig()
    var textConfig = TextEncoderConfig()
    var detrEncoderConfig = DETREncoderConfig()
    var detrDecoderConfig = DETRDecoderConfig()
    var geometryEncoderConfig = GeometryEncoderConfig()
    var maskDecoderConfig = DetectorMaskDecoderConfig()
    var initializerRange: Float = 0.02

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfig = "vision_config"
        case textConfig = "text_config"
        case detrEncoderConfig = "detr_encoder_config"
        case detrDecoderConfig = "detr_decoder_config"
        case geometryEncoderConfig = "geometry_encoder_config"
        case maskDecoderConfig = "mask_decoder_config"
        case initializerRange = "initializer_range"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        visionConfig = try c.decode(.visionConfig, or: visionConfig)
        textConfig = try c.decode(.textConfig, or: textConfig)
        detrEncoderConfig = try c.decode(.detrEncoderConfig, or: detrEncoderConfig)
        detrDecoderConfig = try c.decode(.detrDecoderConfig, or: detrDecoderConfig)
        geometryEncoderConfig = try c.decode(.geometryEncoderConfig, or: geometryEncoderConfig)
        maskDecoderConfig = try c.decode(.maskDecoderConfig, or: maskDecoderConfig)
        initializerRange = try c.decode(.initializerRange, or: initializerRange)
    }
}

/// SAM 3.1 top-level model config.
///
/// Python's `text_config`/`vision_config` placeholders only alias the detector's sub-configs for
/// mlx-vlm compatibility, so they are not stored here.
struct ModelConfig: Codable, Sendable {
    var modelType = "sam3.1_video"
    var detectorConfig = DetectorConfig()
    var trackerConfig = TrackerConfig()
    var initializerRange: Float = 0.02
    var lowResMaskSize = 288

    // Tracking / association thresholds (same as SAM 3)
    var detNmsThresh: Float = 0.1
    var assocIouThresh: Float = 0.1
    var trkAssocIouThresh: Float = 0.5
    var highConfThresh: Float = 0.8
    var highIouThresh: Float = 0.8
    var newDetThresh: Float = 0.7
    var scoreThresholdDetection: Float = 0.5
    var fillHoleArea = 16
    var maxNumObjects = 10000

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case detectorConfig = "detector_config"
        case trackerConfig = "tracker_config"
        case initializerRange = "initializer_range"
        case lowResMaskSize = "low_res_mask_size"
        case detNmsThresh = "det_nms_thresh"
        case assocIouThresh = "assoc_iou_thresh"
        case trkAssocIouThresh = "trk_assoc_iou_thresh"
        case highConfThresh = "high_conf_thresh"
        case highIouThresh = "high_iou_thresh"
        case newDetThresh = "new_det_thresh"
        case scoreThresholdDetection = "score_threshold_detection"
        case fillHoleArea = "fill_hole_area"
        case maxNumObjects = "max_num_objects"
    }

    init() {}

    init(from decoder: any Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(.modelType, or: modelType)
        detectorConfig = try c.decode(.detectorConfig, or: detectorConfig)
        trackerConfig = try c.decode(.trackerConfig, or: trackerConfig)
        initializerRange = try c.decode(.initializerRange, or: initializerRange)
        lowResMaskSize = try c.decode(.lowResMaskSize, or: lowResMaskSize)
        detNmsThresh = try c.decode(.detNmsThresh, or: detNmsThresh)
        assocIouThresh = try c.decode(.assocIouThresh, or: assocIouThresh)
        trkAssocIouThresh = try c.decode(.trkAssocIouThresh, or: trkAssocIouThresh)
        highConfThresh = try c.decode(.highConfThresh, or: highConfThresh)
        highIouThresh = try c.decode(.highIouThresh, or: highIouThresh)
        newDetThresh = try c.decode(.newDetThresh, or: newDetThresh)
        scoreThresholdDetection = try c.decode(.scoreThresholdDetection, or: scoreThresholdDetection)
        fillHoleArea = try c.decode(.fillHoleArea, or: fillHoleArea)
        maxNumObjects = try c.decode(.maxNumObjects, or: maxNumObjects)
    }

    /// Reads `<directory>/config.json`.
    static func load(from directory: URL) throws -> ModelConfig {
        let url = directory.appending(path: "config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SAM31Error.weightsNotFound(directory)
        }
        do {
            return try JSONDecoder().decode(ModelConfig.self, from: Data(contentsOf: url))
        } catch {
            throw SAM31Error.invalidConfig("\(url.lastPathComponent): \(error)")
        }
    }
}
