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

#ifndef THIRD_PARTY_OPEN_SPIEL_SIMULTANEOUS_MOVE_GAME_H_
#define THIRD_PARTY_OPEN_SPIEL_SIMULTANEOUS_MOVE_GAME_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// This is the generic superclass for simultaneous move games. A simultaneous
// move game (AKA Markov Game) is one where all agents submit actions on each
// time step, and the state transists to a new state as a function of the
// joint action. All players know the exact current state of the game.
// For normal-form or matrix games, see normal_form_game.h or matrix.h.

namespace open_spiel {

class SimMoveGame;

class SimMoveState : public State {
 public:
  SimMoveState(std::shared_ptr<const Game> game) : State(game) {}
  SimMoveState(const SimMoveState&) = default;

  // Subclasses must implement a per-player LegalActions function.
  std::vector<Action> LegalActions(Player player) const override = 0;

  // LegalActions() returns either the chance outcomes (at a chance node),
  // or a flattened form of the joint legal actions (at simultaneous move
  // nodes) - see discussion below.
  std::vector<Action> LegalActions() const override {
    if (IsSimultaneousNode()) {
      return LegalFlatJointActions();
    } else if (IsTerminal()) {
      return {};
    } else {
      SPIEL_CHECK_TRUE(IsChanceNode());
      return LegalChanceOutcomes();
    }
  }

  // We override this rather than DoApplyAction() since we want to prevent
  // saving the flat action in the history.
  void ApplyAction(Action action) override {
    if (IsSimultaneousNode()) {
      ApplyFlatJointAction(action);
    } else {
      DoApplyAction(action);
      history_.push_back(action);
    }
  }

  // Convert a flat joint action to a list of actions.
  std::vector<Action> FlatJointActionToActions(Action flat_action) const;

 protected:
  // To make the implementation of algorithms which traverse the whole game
  // tree easier, we support the mapping of joint actions (one per player)
  // to a single flat action taken by the player kSimultaneousPlayerId.

  // If we have three players with legal sets (a, b), (x, y, z), (p, q)
  // respectively, then their 12 possible joint actions will be numbered as
  // follows:
  //   0 - (a, x, p)
  //   1 - (b, x, p)
  //   2 - (a, y, p)
  //   ...
  //   10 - (b, y, q)
  //   11 - (a, z, q)
  //   12 - (b, z, q)

  // Implementors of simultaneous move games don't have to worry about this
  // mapping, but simply check for player == kSimultaneousPlayerId and forward
  // method calls as follows:
  //    ActionToString --> FlatJointActionToString
  //    LegalActions --> LegalFlatJointActions
  //    ApplyAction --> ApplyFlatJointAction

  // Since we repeatedly index into the list of legal actions, it is necessary
  // that LegalActions returns the same list (in the same order) when called
  // twice on the same state.

  // if the number of legal actions overflows int64, this will of course not
  // work correctly.

  // Map a flat joint action for the simultaneous player to a string, in the
  // form "[action1, action2, ...]".
  std::string FlatJointActionToString(Action flat_action) const;

  // Return a list of legal flat joint actions. See above for details.
  std::vector<Action> LegalFlatJointActions() const;

  // Apply a flat joint action, updating the state.
  void ApplyFlatJointAction(Action flat_action);

  void DoApplyActions(const std::vector<Action>& actions) override = 0;
};

class SimMoveGame : public Game {
 protected:
  SimMoveGame(GameType game_type, GameParameters game_parameters)
      : Game(game_type, game_parameters) {}
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_SIMULTANEOUS_MOVE_GAME_H_
