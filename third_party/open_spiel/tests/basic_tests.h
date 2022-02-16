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

#ifndef THIRD_PARTY_OPEN_SPIEL_TESTS_BASIC_TESTS_H_
#define THIRD_PARTY_OPEN_SPIEL_TESTS_BASIC_TESTS_H_

#include <random>
#include <string>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace testing {

// Checks that the game can be loaded.
void LoadGameTest(const std::string& game_name);

// Test to ensure that there are chance outcomes.
void ChanceOutcomesTest(const Game& game);

// Test to ensure that there are no chance outcomes.
void NoChanceOutcomesTest(const Game& game);

// Perform num_sims random simulations of the specified game.
void RandomSimTest(const Game& game, int num_sims);

// Perform num_sims random simulations of the specified game. Also tests the
// Undo function. Note: for every step in the simulation, the entire simulation
// up to that point is rolled backward all the way to the beginning via undo,
// checking that the states match the ones along the history. Therefore, this
// is very slow! Please use sparingly.
void RandomSimTestWithUndo(const Game& game, int num_sims);

// Check that chance outcomes are valid and consistent.
// Performs an exhaustive search of the game tree, so should only be
// used for smallish games.
void CheckChanceOutcomes(const Game& game);

// Same as above but without checking the serialization functions. Every game
// should support serialization: only use this function when developing a new
// game, in order to test the implementation using the basic tests before having
// to implement the custom serialization (only useful for games that have chance
// mode kSampledStochastic).
void RandomSimTestNoSerialize(const Game& game, int num_sims);

// Verifies that ResampleFromInfostate is correctly implemented.
void ResampleInfostateTest(const Game& game, int num_sims);

}  // namespace testing
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_TESTS_BASIC_TESTS_H_
