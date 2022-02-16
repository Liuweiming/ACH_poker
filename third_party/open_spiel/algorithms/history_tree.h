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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_HISTORY_TREE_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_HISTORY_TREE_H_

#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

// TODO(author1): See if it's possible to remove any fields here.
// Stores all information relevant to exploitability calculation for each
// history in the game.
class HistoryNode {
 public:
  HistoryNode(Player player_id, std::unique_ptr<State> game_state);

  State* GetState() { return state_.get(); }

  const std::string& GetInfoState() { return infostate_; }

  const std::string& GetHistory() { return history_; }

  const StateType& GetType() { return type_; }

  double GetValue() const { return value_; }

  Action NumChildren() const { return child_info_.size(); }

  void AddChild(Action outcome,
                std::pair<double, std::unique_ptr<HistoryNode>> child);

  std::vector<Action> GetChildActions() const;

  std::pair<double, HistoryNode*> GetChild(Action outcome);

 private:
  std::unique_ptr<State> state_;
  std::string infostate_;
  std::string history_;
  StateType type_;
  double value_;

  // Map from legal actions to transition probabilities. Uses a map as we need
  // to preserve the order of the actions.
  std::unordered_set<Action> legal_actions_;
  std::map<Action, std::pair<double, std::unique_ptr<HistoryNode>>> child_info_;
};

// History here refers to the fact that we're using histories- i.e.
// representations of all players private information in addition to the public
// information- as the underlying abstraction. Other trees are possible, such as
// PublicTrees, which use public information as the base abstraction, and
// InformationStateTrees, which use all of the information available to one
// player as the base abstraction.
class HistoryTree {
 public:
  // Builds a tree of histories. player_id is needed here as we view all chance
  // and terminal nodes from the viewpoint of player_id. Decision nodes are
  // viewed from the perspective of the player making the decision.
  HistoryTree(std::unique_ptr<State> state, Player player_id);

  HistoryNode* Root() { return root_.get(); }

  HistoryNode* GetByHistory(const std::string& history);

  // For test use only.
  std::vector<std::string> GetHistories();

  Action NumHistories() { return state_to_node_.size(); }

 private:
  std::unique_ptr<HistoryNode> root_;

  // Maps histories to HistoryNodes.
  std::unordered_map<std::string, HistoryNode*> state_to_node_;
};

// Returns a map of infostate strings to a vector of history nodes with
// corresponding counter-factual probabilities, where counter-factual
// probabilities are calculatd using the passed policy for the opponent's
// actions, a probability of 1 for all of the best_responder's actions, and the
// natural chance probabilty for all change actions. We return all infosets
// (i.e. all sets of history nodes grouped by infostate) for the sub-game rooted
// at state, from the perspective of the player with id best_responder.
std::unordered_map<std::string, std::vector<std::pair<HistoryNode*, double>>>
GetAllInfoSets(std::unique_ptr<State> state, Player best_responder,
               const Policy* policy, HistoryTree* tree);

// For a given state, returns all successor states with accompanying
// counter-factual probabilities.
ActionsAndProbs GetSuccessorsWithProbs(const State& state,
                                       Player best_responder,
                                       const Policy* policy);

// Returns all decision nodes, with accompanying counter-factual probabilities,
// for the sub-game rooted at parent_state.
std::vector<std::pair<std::unique_ptr<State>, double>> DecisionNodes(
    const State& parent_state, Player best_responder, const Policy* policy);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_HISTORY_TREE_H_
