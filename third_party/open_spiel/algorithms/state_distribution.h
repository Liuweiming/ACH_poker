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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_STATE_DISTRIBUTION_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_STATE_DISTRIBUTION_H_

#include <string>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Returns a distribution over states at the information state containing the
// specified state given the opponents' policies. That is, it returns
// Pr(h | s, \pi_{-i}) by normalizing the opponents' reach probabilities over
// all h \in s, as described in Section 3.2 of Srinivasan et al. 2018
// https://arxiv.org/abs/1810.09026. Computing this distribution relies strongly
// on the fact that InformationStateString must abide by perfect recall.
//
// This is a game-independent implementation that does a breadth-first search
// from the start of the game to enumerate all possible histories consistent
// with the information state, trimming out histories as they become invalid.
// As such, it may not be very fast.
//
// The returned vectors have an arbitrary ordering, and will include
// zero-probability histories if there are any. If the probability of reaching
// the information state under the given policy is zero (e.g. the Bayes
// normalization term is zero) then a uniform random distribution is returned
// instead.
//
// Note: currently only works for turn-based games of imperfect information,
// and does not work with kSampledStochastic chance modes.
HistoryDistribution GetStateDistribution(const State& state,
                                         const Policy* opponent_policy);

// Incrementally builds the state distribution vectors. Must be called at each
// state in a trajectory. All of the states should correspond to the same
// information state (i.e. all states should have identical
// InformationStateString values, although this is not doublechecked). If
// previous is empty, calls the non-incremental version. This must be called for
// each state in order, starting from the first non-chance node, or it will be
// wrong.
std::unique_ptr<HistoryDistribution> UpdateIncrementalStateDistribution(
    const State& state, const Policy* opponent_policy, int player_id,
    std::unique_ptr<HistoryDistribution> previous);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_STATE_DISTRIBUTION_H_
