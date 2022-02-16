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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_OUTCOME_SAMPLING_MCCFR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_OUTCOME_SAMPLING_MCCFR_H_

#include <memory>
#include <random>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// An implementation of outcome sampling Monte Carlo Counterfactual Regret
// Minimization (CFR). This version is implemented in a way that is closer to
// VR-MCCFR, so that it is compatible with the use of baselines to reduce
// variance (baseline of 0 is equivalent to the original outcome sampling).
//
// Lanctot et al. '09: http://mlanctot.info/files/papers/nips09mccfr.pdf
// Lanctot, 2013: http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
// Schmid et al. '18: https://arxiv.org/abs/1809.03057
// Davis, Schmid, & Bowling '19. https://arxiv.org/abs/1907.09633

namespace open_spiel {
namespace algorithms {

class OutcomeSamplingMCCFRSolver {
 public:
  static constexpr double kInitialTableValues = 0.000001;
  static constexpr double kDefaultEpsilon = 0.6;

  // Creates a solver with a specific seed, average type and an explicit
  // default uniform policy for states that have not been visited.
  OutcomeSamplingMCCFRSolver(const Game& game, double epsilon = kDefaultEpsilon,
                             int seed = -1);

  // Creates a solver with a specific seed and average type, and also allows
  // for a custom default policy for states that have not been visited.
  OutcomeSamplingMCCFRSolver(const Game& game,
                             std::shared_ptr<Policy> default_policy,
                             double epsilon = kDefaultEpsilon, int seed = -1);

  // Performs one iteration of outcome sampling.
  void RunIteration() { RunIteration(&rng_); }

  // Same as above, but uses the specified random number generator instead.
  void RunIteration(std::mt19937* rng);

  // Computes the average policy, containing the policy for all players.
  // The returned policy instance should only be used during the lifetime of
  // the CFRSolver object.
  std::unique_ptr<Policy> AveragePolicy() const {
    return std::unique_ptr<Policy>(
        new CFRAveragePolicy(info_states_, default_policy_));
  }

 private:
  double SampleEpisode(State* state, std::mt19937* rng, double my_reach,
                       double opp_reach, double sample_reach);
  std::vector<double> SamplePolicy(const CFRInfoStateValues& info_state) const;

  // The b_i function from  Schmid et al. '19.
  double Baseline(const State& state, const CFRInfoStateValues& info_state,
                  int aidx) const;

  // Applies Eq. 9 of Schmid et al. '19
  double BaselineCorrectedChildValue(const State& state,
                                     const CFRInfoStateValues& info_state,
                                     int sampled_aidx, int aidx,
                                     double child_value,
                                     double sample_prob) const;

  const Game& game_;
  double epsilon_;
  CFRInfoStateValuesTable info_states_;
  int num_players_;
  int update_player_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> dist_;
  std::shared_ptr<Policy> default_policy_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_OUTCOME_SAMPLING_MCCFR_H_
