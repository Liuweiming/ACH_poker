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

#include "open_spiel/algorithms/evaluate_bots.h"

#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {

std::vector<double> EvaluateBots(State* state, const std::vector<Bot*>& bots,
                                 int seed) {
  const int num_players = bots.size();
  std::mt19937 rng(seed);
  std::vector<Action> joint_actions(bots.size());
  if (state->History().empty()) {
    for (auto bot : bots) bot->Restart();
  } else {
    for (auto bot : bots) bot->RestartAt(*state);
  }
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      Action action = SampleAction(state->ChanceOutcomes(), rng).first;
      for (auto bot : bots) bot->InformAction(*state, kChancePlayerId, action);
      state->ApplyAction(action);
    } else if (state->IsSimultaneousNode()) {
      for (Player p = 0; p < num_players; ++p) {
        if (state->LegalActions(p).empty()) {
          joint_actions[p] = kInvalidAction;
        } else {
          joint_actions[p] = bots[p]->Step(*state);
        }
      }
      state->ApplyActions(joint_actions);
    } else {
      Player current_player = state->CurrentPlayer();
      Action action = bots[current_player]->Step(*state);
      for (Player p = 0; p < num_players; ++p) {
        if (p != current_player) {
          bots[p]->InformAction(*state, current_player, action);
        }
      }
      state->ApplyAction(action);
    }
  }

  // Return terminal utility.
  return state->Returns();
}

}  // namespace open_spiel
