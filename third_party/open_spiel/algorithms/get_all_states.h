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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_GET_ALL_STATES_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_GET_ALL_STATES_H_

#include <string>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// Get all states in the game, indexed by their string representation.
// For small games only!
//
// Useful for methods that solve the games explicitly, i.e. value iteration.
//
// Use this implementation with caution as it does a recursive tree
// walk of the game and could easily fill up memory for larger games or games
// with long horizons.
//
// Currently only works for sequential games.
//
// Note: negative depth limit means no limit, 0 means only root, etc..

std::map<std::string, std::unique_ptr<State>> GetAllStates(
    const Game& game, int depth_limit, bool include_terminals,
    bool include_chance_states);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_GET_ALL_STATES_H_
