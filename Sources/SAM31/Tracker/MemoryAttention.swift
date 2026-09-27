// Port of mlx_vlm/models/sam3_1/sam_components.py::{DecoupledMemoryAttentionLayer,
// DecoupledMemoryAttention} (mlx-vlm 0.7.3)
//
// Task 12 declares the modules and their full weight-key trees so `tracker_model.memory_attention.*`
// loads. The forward passes, and the parameter-free `SimpleRoPEAttention` children
// (`self_attention_rope`, `cross_attention_rope`), land in Task 13.
import MLX
import MLXNN

/// Pre-norm decoupled transformer layer with separate self/cross-attention projections plus the
/// image cross-attention Q/K projections. Weight keys: `tracker_model.memory_attention.layers.*`.
final class DecoupledMemoryAttentionLayer: Module {
    @ModuleInfo(key: "self_attn_q_proj") var selfAttnQProj: Linear
    @ModuleInfo(key: "self_attn_k_proj") var selfAttnKProj: Linear
    @ModuleInfo(key: "self_attn_v_proj") var selfAttnVProj: Linear
    @ModuleInfo(key: "self_attn_out_proj") var selfAttnOutProj: Linear

    @ModuleInfo(key: "cross_attn_q_proj") var crossAttnQProj: Linear
    @ModuleInfo(key: "cross_attn_k_proj") var crossAttnKProj: Linear
    @ModuleInfo(key: "cross_attn_v_proj") var crossAttnVProj: Linear
    @ModuleInfo(key: "cross_attn_out_proj") var crossAttnOutProj: Linear

    @ModuleInfo(key: "image_cross_attn_q_proj") var imageCrossAttnQProj: Linear
    @ModuleInfo(key: "image_cross_attn_k_proj") var imageCrossAttnKProj: Linear

    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    init(_ config: TrackerConfig) {
        let d = config.memoryAttentionHiddenSize
        _selfAttnQProj.wrappedValue = Linear(d, d)
        _selfAttnKProj.wrappedValue = Linear(d, d)
        _selfAttnVProj.wrappedValue = Linear(d, d)
        _selfAttnOutProj.wrappedValue = Linear(d, d)
        _crossAttnQProj.wrappedValue = Linear(d, d)
        _crossAttnKProj.wrappedValue = Linear(d, d)
        _crossAttnVProj.wrappedValue = Linear(d, d)
        _crossAttnOutProj.wrappedValue = Linear(d, d)
        _imageCrossAttnQProj.wrappedValue = Linear(d, d)
        _imageCrossAttnKProj.wrappedValue = Linear(d, d)
        _linear1.wrappedValue = Linear(d, config.memoryAttentionFeedForwardHiddenSize)
        _linear2.wrappedValue = Linear(config.memoryAttentionFeedForwardHiddenSize, d)
        _norm1.wrappedValue = LayerNorm(dimensions: d)
        _norm2.wrappedValue = LayerNorm(dimensions: d)
        _norm3.wrappedValue = LayerNorm(dimensions: d)
    }
}

/// SAM 3.1 memory attention with decoupled projections.
/// Weight keys: `tracker_model.memory_attention.*`.
final class DecoupledMemoryAttention: Module {
    @ModuleInfo(key: "layers") var layers: [DecoupledMemoryAttentionLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: TrackerConfig) {
        _layers.wrappedValue = (0..<config.memoryAttentionNumLayers).map { _ in
            DecoupledMemoryAttentionLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.memoryAttentionHiddenSize)
    }

    /// - Parameters:
    ///   - image: `(1, HW, D)` raw current-frame image features.
    ///   - src: `(B, HW, D)` current-frame features.
    ///   - memoryImage: `(1, N_img, D)` image features of the memory frames.
    ///   - memory: `(B, N_mem, D)` mask memories plus object-pointer tokens.
    ///   - srcPos: `(B, HW, D)` pos enc added to `src` at input (scaled by 0.1).
    ///   - memoryPos: `(B, N_mem, D)`; only its object-pointer tail is used.
    ///   - memoryImagePos: `(1, N_img, D)` pos enc for the memory image keys.
    ///   - numKExcludeRope: number of trailing object-pointer tokens.
    func callAsFunction(
        image: MLXArray, src: MLXArray, memoryImage: MLXArray, memory: MLXArray, srcPos: MLXArray?,
        memoryPos: MLXArray?, memoryImagePos: MLXArray?, numKExcludeRope: Int
    ) -> MLXArray {
        fatalError("Task 13")
    }
}
