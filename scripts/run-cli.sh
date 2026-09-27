#!/usr/bin/env bash
# Runs `swift run -c release sam31-cli "$@"` with MLX's Metal kernels in place.
#
# SwiftPM does not compile mlx-swift's Metal shaders (mlx-swift#488), so a `swift build` binary
# fails with "Failed to load the default metallib". xcodebuild does compile them. This script builds
# the metallib once with xcodebuild and copies it next to the SwiftPM binary as `mlx.metallib`, the
# first place MLX looks. It rebuilds it when Package.resolved changes.
set -euo pipefail
cd "$(dirname "$0")/.."

metallib=.build/release/mlx.metallib
xcode_lib=.build/xcode/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib

swift build -c release --product sam31-cli >&2
if [[ ! -f $metallib || Package.resolved -nt $metallib ]]; then
    if [[ ! -f $xcode_lib || Package.resolved -nt $xcode_lib ]]; then
        echo "building MLX Metal kernels with xcodebuild (one time)..." >&2
        xcodebuild build -scheme sam31-cli -configuration Release -destination 'platform=macOS' \
            -derivedDataPath .build/xcode -quiet >&2
    fi
    cp "$xcode_lib" "$metallib"
fi
exec swift run -c release --skip-build sam31-cli "$@"
