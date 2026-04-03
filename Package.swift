// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "swift-tokenizers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "MLXSwiftTokenizers", targets: ["MLXSwiftTokenizers"])
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
        .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.7.0"),
    ],
    targets: [
        .target(
            name: "MLXSwiftTokenizers",
            dependencies: [
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "yyjson", package: "yyjson"),
            ]
        ),
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "MLXSwiftTokenizers",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "yyjson", package: "yyjson"),
            ]
        ),
        .testTarget(
            name: "TokenizersTests",
            dependencies: [
                "MLXSwiftTokenizers",
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            resources: [.process("Resources")]
        ),
    ]
)
