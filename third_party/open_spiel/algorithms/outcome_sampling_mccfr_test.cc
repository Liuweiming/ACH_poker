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

#include "open_spiel/algorithms/outcome_sampling_mccfr.h"

#include <cmath>
#include <iostream>
#include <random>

#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

constexpr int kSeed = 230398247;

void MCCFR_2PGameTest(const std::string& game_name, std::mt19937* rng,
                      int iterations, double nashconv_upperbound) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  OutcomeSamplingMCCFRSolver solver(*game);
  for (int i = 0; i < iterations; i++) {
    solver.RunIteration(rng);
  }
  const std::unique_ptr<Policy> average_policy = solver.AveragePolicy();
  double nash_conv = NashConv(*game, *average_policy);
  std::cout << "Game: " << game_name << ", iters = " << iterations
            << ", NashConv: " << nash_conv << std::endl;
  SPIEL_CHECK_LE(nash_conv, nashconv_upperbound);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  std::mt19937 rng(algorithms::kSeed);
  // Values double-checked with the original implementation used in (Lanctot,
  // "Monte Carlo Sampling and Regret Minimization For Equilibrium Computation
  // and Decision-Making in Large Extensive Form Games", 2013).
  algorithms::MCCFR_2PGameTest("kuhn_poker", &rng, 10000, 0.1);
  algorithms::MCCFR_2PGameTest("leduc_poker", &rng, 100000, 1.5);
  algorithms::MCCFR_2PGameTest("liars_dice", &rng, 100000, 1);
}
