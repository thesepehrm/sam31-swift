// Port of mlx_vlm/models/sam3_1/sam3_1.py::Model module tree (mlx-vlm 0.7.3)
import MLXNN

/// Root module whose parameter keys match the checkpoint: `detector_model.*` and `tracker_model.*`.
final class SAM31Root: Module {
    @ModuleInfo(key: "detector_model") var detectorModel: DetectorModel
    @ModuleInfo(key: "tracker_model") var trackerModel: MultiplexTrackerModel

    init(_ config: ModelConfig) {
        _detectorModel.wrappedValue = DetectorModel(config.detectorConfig)
        _trackerModel.wrappedValue = MultiplexTrackerModel(config.trackerConfig)
    }
}
