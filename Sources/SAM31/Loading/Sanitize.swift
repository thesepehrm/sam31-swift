// Port of mlx_vlm/models/sam3_1/sam3_1.py::Model.sanitize (mlx-vlm 0.7.3)
import Foundation
import MLX

/// Mask embed: sequential indices to named convs, checked in this order (first match wins).
private let maskEmbedRemap: [(old: String, new: String)] = [
    ("mask_embed.0.", "mask_embed.conv1."),
    ("mask_embed.1.", "mask_embed.layer_norm1."),
    ("mask_embed.3.", "mask_embed.conv2."),
    ("mask_embed.4.", "mask_embed.layer_norm2."),
    ("mask_embed.6.", "mask_embed.conv3."),
]

private let convTransposePatterns = ["scale_layers.", "upscale_conv", "output_upscaling"]
private let skipPatterns = ["memory_temporal_positional_encoding"]

/// Converts checkpoint weights to MLX format: key remaps for the SAM 3.1 module layout, plus
/// Conv2d/ConvTranspose2d transposes for PyTorch-layout checkpoints.
///
/// Conv transposes are skipped for checkpoints already in MLX layout (detected from the patch
/// embedding being channels-last), so sanitize is idempotent.
func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let alreadyMLX = weights.contains { key, value in
        key.hasSuffix("patch_embeddings.projection.weight")
            && value.ndim == 4
            && value.shape[3] == 3
            && value.shape[1] != 3
    }

    var sanitized: [String: MLXArray] = [:]
    sanitized.reserveCapacity(weights.count)

    for (originalKey, originalValue) in weights {
        var key = originalKey
        var value = originalValue

        // Remap mask_embed indexed keys to named keys
        for (old, new) in maskEmbedRemap where key.contains(old) {
            key = key.replacingOccurrences(of: old, with: new)
            break
        }

        // Remap memory_fuser norm -> layer_norm
        if key.contains("memory_fuser") && key.contains(".norm.") {
            key = key.replacingOccurrences(of: ".norm.", with: ".layer_norm.")
        }

        // Remap mask_downsampler.layers.4.conv -> final_conv
        if key.contains("mask_downsampler.layers.4.conv.") {
            key = key.replacingOccurrences(
                of: "mask_downsampler.layers.4.conv.", with: "mask_downsampler.final_conv.")
        }

        if value.ndim == 4 && !alreadyMLX {
            if skipPatterns.contains(where: { key.contains($0) }) {
                sanitized[key] = value
                continue
            }
            if convTransposePatterns.contains(where: { key.contains($0) }) {
                value = value.transposed(1, 2, 3, 0)
            } else {
                value = value.transposed(0, 2, 3, 1)
            }
        }

        sanitized[key] = value
    }

    return sanitized
}
