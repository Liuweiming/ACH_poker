// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


import PackageDescription

let package = Package(
  name: "OpenSpiel",
  products: [
    .library(
      name: "OpenSpiel",
      targets: ["OpenSpiel"]),
    .executable(
      name: "GridMazeExample",
      targets: ["GridMazeExample"]),
    .executable(
      name: "KuhnPokerExample",
      targets: ["KuhnPokerExample"]),
    .executable(
      name: "TexasHoldemBenchmark",
      targets: ["TexasHoldemBenchmark"]),
  ],
  targets: [
    .target(
      name: "OpenSpiel",
      dependencies: [],
      path: "swift/Sources/OpenSpiel"),
    .testTarget(
      name: "OpenSpielTests",
      dependencies: ["OpenSpiel"],
      path: "swift/Tests/OpenSpielTests"),
    .target(
      name: "GridMazeExample",
      dependencies: ["OpenSpiel"],
      path: "swift/Examples/GridMaze"),
    .target(
      name: "KuhnPokerExample",
      dependencies: ["OpenSpiel"],
      path: "swift/Examples/KuhnPoker"),
    .target(
      name: "TexasHoldemBenchmark",
      dependencies: ["OpenSpiel"],
      path: "swift/Examples/TexasHoldemBenchmark"),
  ]
)
