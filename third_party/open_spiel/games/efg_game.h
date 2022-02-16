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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_EFG_GAME_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_EFG_GAME_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "absl/types/optional.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// A parser for the .efg format used by Gambit:
// http://www.gambit-project.org/gambit14/formats.html
//
// Parameters:
//       "filename"   string     name of a file containing the data
//
// Note: not the full EFG is supported as stated on that page. In particular:
//   - Payoffs / outcomes at non-terminal nodes are not supported
//   - Player nodes and chance nodes must each have one child
//

namespace open_spiel {
namespace efg_game {

enum class NodeType {
  kChance,
  kPlayer,
  kTerminal,
};

// A node object that represent a subtree of the game.
struct Node {
  Node* parent;
  NodeType type;
  int id;
  std::string name;
  int infoset_number;
  int player_number;
  std::string infoset_name;
  std::string outcome_name;
  int outcome_number;
  std::vector<std::string> actions;
  std::vector<Node*> children;
  std::vector<double> probs;
  std::vector<double> payoffs;
};

// A few example games used for testing.
std::string GetSampleEFGData();
std::string GetKuhnPokerEFGData();

// A function to load an EFG directly from string data. Note: games loaded
// using this function will not be serializable (nor will their states). Use
// the general LoadGame with the filename argument if serialization is required.
std::shared_ptr<const Game> LoadEFGGame(const std::string& data);

class EFGState : public State {
 public:
  explicit EFGState(std::shared_ptr<const Game> game, const Node* root);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  const Node* cur_node_;
};

class EFGGame : public Game {
 public:
  explicit EFGGame(const GameParameters& params);
  explicit EFGGame(const std::string& data);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new EFGState(shared_from_this(), nodes_[0].get()));
  }

  int NumDistinctActions() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double UtilitySum() const override;
  double MaxUtility() const override;
  int MaxGameLength() const override;

  std::shared_ptr<const Game> Clone() const override {
    if (!filename_.empty()) {
      return LoadGame("efg_game", {{"filename", GameParameter(filename_)}});
    } else {
      return LoadEFGGame(string_data_);
    }
  }

 private:
  std::unique_ptr<Node> NewNode() const;
  void ParseGame();
  void ParsePrologue();
  std::string NextToken();
  bool ParseDoubleValue(const std::string& str, double* value) const;
  bool IsWhiteSpace(char c) const;
  bool IsNodeToken(char c) const;
  void ParseChanceNode(Node* parent, Node* child, int depth);
  void ParsePlayerNode(Node* parent, Node* child, int depth);
  void ParseTerminalNode(Node* parent, Node* child, int depth);
  void RecParseSubtree(Node* parent, Node* child, int depth);
  std::string PrettyTree(const Node* node, const std::string& indent) const;

  std::string filename_;
  std::string string_data_;
  int pos_;
  std::vector<std::unique_ptr<Node>> nodes_;
  std::string name_;
  std::string description_;
  std::vector<std::string> player_names_;
  int num_chance_nodes_;
  int num_players_;
  int max_actions_;
  int max_chance_outcomes_;
  int max_depth_;
  absl::optional<double> util_sum_;
  absl::optional<double> max_util_;
  absl::optional<double> min_util_;
  bool constant_sum_;
  bool identical_payoffs_;
  bool general_sum_;
  bool perfect_information_;
  std::unordered_map<int, int> infoset_num_to_states_count_;
};

}  // namespace efg_game
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_EFG_GAME_H_
