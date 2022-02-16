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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_PUBLIC_TREE_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_PUBLIC_TREE_H_

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
class PublicNode {
 public:
  PublicNode(std::unique_ptr<State> game_state);

  State* GetState() { return state_.get(); }

  const std::string& GetHistory() { return history_; }

  const StateType& GetType() { return type_; }

  Action NumChildren() const { return child_info_.size(); }

  void AddChild(Action outcome, std::unique_ptr<PublicNode> child);

  std::vector<Action> GetChildActions() const;

  PublicNode* GetChild(Action outcome);

 private:
  std::unique_ptr<State> state_;
  std::string history_;
  StateType type_;

  // Map from legal actions to transition probabilities. Uses a map as we need
  // to preserve the order of the actions.
  std::unordered_set<Action> legal_actions_;
  std::map<Action, std::unique_ptr<PublicNode>> child_info_;
};

class PublicTree {
 public:
  PublicTree(std::unique_ptr<State> state);

  PublicNode* Root() { return root_.get(); }

  PublicNode* GetByHistory(const std::string& history);

  // For test use only.
  std::vector<std::string> GetHistories();
  std::vector<std::string> GetDecisionHistoriesl();

  Action NumHistories() { return state_to_node_.size(); }

 private:
  std::unique_ptr<PublicNode> root_;

  // Maps histories to PublicNodes.
  std::unordered_map<std::string, PublicNode*> state_to_node_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_PUBLIC_TREE_H_
