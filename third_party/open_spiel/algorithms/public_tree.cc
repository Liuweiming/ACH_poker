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

#include "open_spiel/algorithms/public_tree.h"

#include <cmath>
#include <limits>
#include <unordered_set>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

std::unique_ptr<PublicNode> RecursivelyBuildGameTree(
    std::unique_ptr<State> state,
    std::unordered_map<std::string, PublicNode*>* state_to_node) {
  std::unique_ptr<PublicNode> node(new PublicNode(std::move(state)));
  if (state_to_node == nullptr) SpielFatalError("state_to_node is null.");
  (*state_to_node)[node->GetHistory()] = node.get();
  State* state_ptr = node->GetState();
  switch (node->GetType()) {
    case StateType::kChance: {
      double probability_sum = 0;
      for (const auto& outcome_and_prob : state_ptr->ChanceOutcomes()) {
        Action outcome = outcome_and_prob.first;
        std::unique_ptr<State> child = state_ptr->Child(outcome);
        if (child == nullptr) {
          SpielFatalError("Can't add child; child is null.");
        }
        std::unique_ptr<PublicNode> child_node =
            RecursivelyBuildGameTree(std::move(child), state_to_node);
        node->AddChild(outcome, std::move(child_node));
        // Add the first child as the placeholder.
        break;
      }
      break;
    }
    case StateType::kDecision: {
      for (const auto& legal_action : state_ptr->LegalActions()) {
        std::unique_ptr<State> child = state_ptr->Child(legal_action);
        node->AddChild(legal_action, RecursivelyBuildGameTree(std::move(child),
                                                              state_to_node));
      }
      break;
    }
    case StateType::kTerminal: {
      // As we assign terminal utilities to node.value in the constructor of
      // PublicNode, we don't have anything to do here.
      break;
    }
  }
  return node;
}

}  // namespace

PublicNode::PublicNode(std::unique_ptr<State> game_state)
    : state_(std::move(game_state)),
      history_(state_->ToString()),
      type_(state_->GetType()) {
  for (Action action : state_->LegalActions()) legal_actions_.insert(action);
}

void PublicNode::AddChild(Action outcome, std::unique_ptr<PublicNode> child) {
  if (!legal_actions_.count(outcome)) SpielFatalError("Child is not legal.");
  if (child == nullptr) {
    SpielFatalError("Error inserting child; child is null.");
  }
  child_info_[outcome] = std::move(child);
  if (child_info_.size() > legal_actions_.size()) {
    SpielFatalError("More children than legal actions.");
  }
}

PublicNode* PublicNode::GetChild(Action outcome) {
  auto it = child_info_.find(outcome);
  if (it == child_info_.end()) {
    SpielFatalError("Error getting child; action not found.");
  }
  PublicNode* child = it->second.get();
  if (child == nullptr) {
    SpielFatalError("Error getting child; child is null.");
  }
  return child;
}

std::vector<Action> PublicNode::GetChildActions() const {
  std::vector<Action> actions;
  actions.reserve(child_info_.size());
  for (const auto& kv : child_info_) actions.push_back(kv.first);
  return actions;
}

PublicNode* PublicTree::GetByHistory(const std::string& history) {
  PublicNode* node = state_to_node_[history];
  if (node == nullptr) {
    SpielFatalError(absl::StrCat("Node is null for history: '", history, "'"));
  }
  return node;
}

std::vector<std::string> PublicTree::GetHistories() {
  std::vector<std::string> histories;
  histories.reserve(state_to_node_.size());
  for (const auto& kv : state_to_node_) {
    histories.push_back(kv.first);
  }
  return histories;
}

std::vector<std::string> PublicTree::GetDecisionHistoriesl() {
  std::vector<std::string> histories;
  histories.reserve(state_to_node_.size());
  for (const auto& kv : state_to_node_) {
    if (kv.second->GetType() == StateType::kDecision) {
      histories.push_back(kv.first);
    }
  }
  return histories;
}

PublicTree::PublicTree(std::unique_ptr<State> state) {
  root_ = RecursivelyBuildGameTree(std::move(state), &state_to_node_);
}

}  // namespace algorithms
}  // namespace open_spiel
