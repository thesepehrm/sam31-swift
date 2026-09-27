// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "sam31-swift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "SAM31", targets: ["SAM31"]),
        .executable(name: "sam31-cli", targets: ["sam31-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.4"),
    ],
    targets: [
        .target(
            name: "SAM31",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            resources: [.copy("Text/Resources")]
        ),
        .executableTarget(
            name: "sam31-cli",
            dependencies: ["SAM31", .product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "SAM31Tests",
            dependencies: ["SAM31"],
            resources: [.copy("Resources")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
