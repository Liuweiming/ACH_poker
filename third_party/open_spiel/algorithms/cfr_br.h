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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_BR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_BR_H_

#include <memory>

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// An implementation of CFR-BR (Johanson et al., "Finding Optimal Abstract
// Strategies in Extensive-Form Games", 2012). In CFR-BR, at each iteration,
// each player minimizes regret against their worst-case opponent (a best
// response to its current policy).
namespace open_spiel {
namespace algorithms {

class CFRBRSolver : public CFRSolverBase {
 public:
  explicit CFRBRSolver(const Game& game);

  void EvaluateAndUpdatePolicy() override;

 private:
  // Policies that are used instead of the current policy for some the opponent
  // players.
  std::vector<const Policy*> policy_overrides_;
  TabularPolicy uniform_policy_;
  std::vector<std::unique_ptr<TabularBestResponse>> best_response_computers_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_BR_H_
