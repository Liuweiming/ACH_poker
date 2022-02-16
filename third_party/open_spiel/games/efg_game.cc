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

#include "open_spiel/games/efg_game.h"

#include <cstdlib>
#include <cstring>
#include <fstream>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace efg_game {
namespace {

constexpr int kBuffSize = 1024;

// Facts about the game. These are defaults that will differ depending on the
// game's descriptions. Using dummy defaults just to register the game.
const GameType kGameType{/*short_name=*/"efg_game",
                         /*long_name=*/"efg_game",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                         {{"filename", GameParameter(std::string(""))}},
                         /*default_loadable=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new EFGGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string NodeToString(const Node* node) {
  std::string str = "";
  if (node->type == NodeType::kTerminal) {
    absl::StrAppend(&str, "Terminal: ", node->name, " ", node->outcome_name);
    for (double payoff : node->payoffs) {
      absl::StrAppend(&str, " ", payoff);
    }
    absl::StrAppend(&str, "\n");
  } else if (node->type == NodeType::kChance) {
    absl::StrAppend(&str, "Chance: ", node->name, " ", node->infoset_number,
                    " ", node->infoset_name);
    for (int i = 0; i < node->children.size(); ++i) {
      absl::StrAppend(&str, " ", node->actions[i], " ", node->probs[i]);
    }
    absl::StrAppend(&str, "\n");
  } else if (node->type == NodeType::kPlayer) {
    absl::StrAppend(&str, "Player: ", node->name, " ", node->player_number, " ",
                    node->infoset_number, " ", node->infoset_name);
    for (int i = 0; i < node->children.size(); ++i) {
      absl::StrAppend(&str, " ", node->actions[i]);
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}
}  // namespace

// A copy of games/efg/kuhn_poker.efg useful to use for tests.
const char* kKuhnEFGData = R"###(
EFG 2 R "Kuhn poker" { "Player 1" "Player 2" } "A simplified poker game: https://en.wikipedia.org/wiki/Kuhn_poker"

c "ROOT" 1 "c1" { "1" 1/3 "0" 1/3 "2" 1/3 } 0
  c "c2" 2 "c2" { "2" 1/2 "0" 1/2 } 0
    p "" 1 1 "1" { "p" "b" } 0
      p "" 2 2 "2p" { "p" "b" } 0
        t "" 3 "Outcome 12pp" { -1.0 1.0 }
        p "" 1 2 "1pb" { "p" "b" } 0
          t "" 4 "Outcome 12pbp" { -1.0 1.0 }
          t "" 5 "Outcome 12pbb" { -2.0 2.0 }
      p "" 2 1 "2b" { "p" "b" } 0
        t "" 1 "Outcome 12bp" { 1.0 -1.0 }
        t "" 2 "Outcome 12bb" { -2.0 2.0 }
    p "" 1 1 "1" { "p" "b" } 0
      p "" 2 3 "0p" { "p" "b" } 0
        t "" 8 "Outcome 10pp" { 1.0 -1.0 }
        p "" 1 2 "1pb" { "p" "b" } 0
          t "" 6 "Outcome 10pbp" { -1.0 1.0 }
          t "" 7 "Outcome 10pbb" { 2.0 -2.0 }
      p "" 2 4 "0b" { "p" "b" } 0
        t "" 9 "Outcome 10bp" { 1.0 -1.0 }
        t "" 10 "Outcome 10bb" { 2.0 -2.0 }
  c "c3" 3 "c3" { "2" 1/2 "1" 1/2 } 0
    p "" 1 3 "0" { "p" "b" } 0
      p "" 2 2 "2p" { "p" "b" } 0
        t "" 13 "Outcome 02pp" { -1.0 1.0 }
        p "" 1 4 "0pb" { "p" "b" } 0
          t "" 14 "Outcome 02pbp" { -1.0 1.0 }
          t "" 15 "Outcome 02pbb" { -2.0 2.0 }
      p "" 2 1 "2b" { "p" "b" } 0
        t "" 11 "Outcome 02bp" { 1.0 -1.0 }
        t "" 12 "Outcome 02bb" { -2.0 2.0 }
    p "" 1 3 "0" { "p" "b" } 0
      p "" 2 5 "1p" { "p" "b" } 0
        t "" 18 "Outcome 01pp" { -1.0 1.0 }
        p "" 1 4 "0pb" { "p" "b" } 0
          t "" 16 "Outcome 01pbp" { -1.0 1.0 }
          t "" 17 "Outcome 01pbb" { -2.0 2.0 }
      p "" 2 6 "1b" { "p" "b" } 0
        t "" 19 "Outcome 01bp" { 1.0 -1.0 }
        t "" 20 "Outcome 01bb" { -2.0 2.0 }
  c "c4" 4 "c4" { "0" 1/2 "1" 1/2 } 0
    p "" 1 5 "2" { "p" "b" } 0
      p "" 2 3 "0p" { "p" "b" } 0
        t "" 21 "Outcome 20pp" { 1.0 -1.0 }
        p "" 1 6 "2pb" { "p" "b" } 0
          t "" 22 "Outcome 20pbp" { -1.0 1.0 }
          t "" 23 "Outcome 20pbb" { 2.0 -2.0 }
      p "" 2 4 "0b" { "p" "b" } 0
        t "" 24 "Outcome 20bp" { 1.0 -1.0 }
        t "" 25 "Outcome 20bb" { 2.0 -2.0 }
    p "" 1 5 "2" { "p" "b" } 0
      p "" 2 5 "1p" { "p" "b" } 0
        t "" 28 "Outcome 21pp" { 1.0 -1.0 }
        p "" 1 6 "2pb" { "p" "b" } 0
          t "" 26 "Outcome 21pbp" { -1.0 1.0 }
          t "" 27 "Outcome 21pbb" { 2.0 -2.0 }
      p "" 2 6 "1b" { "p" "b" } 0
        t "" 29 "Outcome 21bp" { 1.0 -1.0 }
        t "" 30 "Outcome 21bb" { 2.0 -2.0 }
)###";

// A copy of games/efg/sample.efg useful to use within tests.
const char* kSampleEFGData = R"###(
EFG 2 R "General Bayes game, one stage" { "Player 1" "Player 2" }
c "ROOT" 1 "(0,1)" { "1G" 0.500000 "1B" 0.500000 } 0
c "" 2 "(0,2)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 1 "Outcome 1" { 10.000000 2.000000 }
t "" 2 "Outcome 2" { 0.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 3 "Outcome 3" { 2.000000 4.000000 }
t "" 4 "Outcome 4" { 4.000000 0.000000 }
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 5 "Outcome 5" { 10.000000 2.000000 }
t "" 6 "Outcome 6" { 0.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 7 "Outcome 7" { 2.000000 4.000000 }
t "" 8 "Outcome 8" { 4.000000 0.000000 }
c "" 3 "(0,3)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 9 "Outcome 9" { 4.000000 2.000000 }
t "" 10 "Outcome 10" { 2.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 11 "Outcome 11" { 0.000000 4.000000 }
t "" 12 "Outcome 12" { 10.000000 2.000000 }
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 13 "Outcome 13" { 4.000000 2.000000 }
t "" 14 "Outcome 14" { 2.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 15 "Outcome 15" { 0.000000 4.000000 }
t "" 16 "Outcome 16" { 10.000000 0.000000 }
)###";

// Include a few samples that are used for testing.
std::string GetSampleEFGData() { return std::string(kSampleEFGData); }

std::string GetKuhnPokerEFGData() { return std::string(kKuhnEFGData); }

EFGState::EFGState(std::shared_ptr<const Game> game, const Node* root)
    : State(game), cur_node_(root) {}

Player EFGState::CurrentPlayer() const {
  if (cur_node_->type == NodeType::kChance) {
    return kChancePlayerId;
  } else if (cur_node_->type == NodeType::kTerminal) {
    return kTerminalPlayerId;
  } else {
    // Gambit player numbers are between 1 and num_players
    SPIEL_CHECK_GE(cur_node_->player_number, 1);
    SPIEL_CHECK_LE(cur_node_->player_number, num_players_);
    return cur_node_->player_number - 1;
  }
}

std::string EFGState::ActionToString(Player player, Action action) const {
  SPIEL_CHECK_LT(action, cur_node_->actions.size());
  return cur_node_->actions[action];
}

std::string EFGState::ToString() const {
  return absl::StrCat(cur_node_->id, ": ", NodeToString(cur_node_));
}

bool EFGState::IsTerminal() const {
  return cur_node_->type == NodeType::kTerminal;
}

std::vector<double> EFGState::Returns() const {
  if (cur_node_->type == NodeType::kTerminal) {
    SPIEL_CHECK_EQ(cur_node_->payoffs.size(), num_players_);
    return cur_node_->payoffs;
  } else {
    return std::vector<double>(num_players_, 0);
  }
}

std::string EFGState::InformationStateString(Player player) const {
  // The information set number has to uniquely identify the infoset, whereas
  // the names are optional. But the numbers are unique per player, so must
  // add the player number.
  return absl::StrCat(cur_node_->player_number - 1, "-", player, "-",
                      cur_node_->infoset_number, "-", cur_node_->infoset_name);
}

std::string EFGState::ObservationString(Player player) const {
  return absl::StrCat(cur_node_->player_number - 1, "-", player, "-",
                      cur_node_->infoset_number, "-", cur_node_->infoset_name);
}

std::unique_ptr<State> EFGState::Clone() const {
  return std::unique_ptr<State>(new EFGState(*this));
}

void EFGState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_TRUE(cur_node_->parent != nullptr);
  cur_node_ = cur_node_->parent;
}

void EFGState::DoApplyAction(Action action) {
  // Actions in these games are just indices into the legal actions.
  SPIEL_CHECK_FALSE(cur_node_->type == NodeType::kTerminal);
  SPIEL_CHECK_LT(action, cur_node_->children.size());
  SPIEL_CHECK_FALSE(cur_node_->children[action] == nullptr);
  cur_node_ = cur_node_->children[action];
}

std::vector<Action> EFGState::LegalActions() const {
  // Actions in these games are just indices into the legal actions.
  std::vector<Action> actions(cur_node_->actions.size(), 0);
  for (int i = 0; i < cur_node_->actions.size(); ++i) {
    actions[i] = i;
  }
  if (cur_node_->type != NodeType::kTerminal) {
    SPIEL_CHECK_GT(actions.size(), 0);
  }
  return actions;
}

std::vector<std::pair<Action, double>> EFGState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_TRUE(cur_node_->type == NodeType::kChance);
  std::vector<std::pair<Action, double>> outcomes(cur_node_->children.size());
  for (int i = 0; i < cur_node_->children.size(); ++i) {
    outcomes[i].first = i;
    outcomes[i].second = cur_node_->probs[i];
  }
  return outcomes;
}

int EFGGame::NumDistinctActions() const { return max_actions_; }

int EFGGame::NumPlayers() const { return num_players_; }

double EFGGame::MinUtility() const { return min_util_.value(); }

double EFGGame::UtilitySum() const { return util_sum_.value(); }

double EFGGame::MaxUtility() const { return max_util_.value(); }

int EFGGame::MaxGameLength() const { return max_depth_; }

EFGGame::EFGGame(const GameParameters& params)
    : Game(kGameType, params),
      string_data_(""),
      pos_(0),
      num_chance_nodes_(0),
      max_actions_(0),
      max_chance_outcomes_(0),
      max_depth_(0),
      constant_sum_(true),
      identical_payoffs_(true),
      general_sum_(true),
      perfect_information_(true) {
  filename_ = ParameterValue<std::string>("filename");

  std::ifstream file;
  file.open(filename_.c_str());
  if (!file.is_open()) {
    SpielFatalError(absl::StrCat("Could not open input file: ", filename_));
  }

  string_data_ = "";
  char buffer[kBuffSize];
  while (!file.eof()) {
    memset(buffer, 0, kBuffSize);
    file.read(buffer, kBuffSize - 1);
    absl::StrAppend(&string_data_, buffer);
  }
  file.close();

  SPIEL_CHECK_GT(string_data_.size(), 0);

  // Now parse the string data into a data structure.
  ParseGame();
}

EFGGame::EFGGame(const std::string& data)
    : Game(kGameType, {}),
      string_data_(data),
      pos_(0),
      num_chance_nodes_(0),
      max_actions_(0),
      max_chance_outcomes_(0),
      max_depth_(0),
      constant_sum_(true),
      identical_payoffs_(true),
      general_sum_(true),
      perfect_information_(true) {
  ParseGame();
}

std::shared_ptr<const Game> LoadEFGGame(const std::string& data) {
  return std::shared_ptr<const Game>(new EFGGame(data));
}

bool EFGGame::IsWhiteSpace(char c) const {
  return (c == ' ' || c == '\r' || c == '\n');
}

bool EFGGame::IsNodeToken(char c) const {
  return (c == 'c' || c == 'p' || c == 't');
}

std::unique_ptr<Node> EFGGame::NewNode() const {
  std::unique_ptr<Node> new_node = std::unique_ptr<Node>(new Node);
  new_node->id = nodes_.size();
  return new_node;
}

bool EFGGame::ParseDoubleValue(const std::string& str, double* value) const {
  if (str.find('/') != std::string::npos) {
    // Check for rational number of the form X/Y
    std::vector<std::string> parts = absl::StrSplit(str, '/');
    SPIEL_CHECK_EQ(parts.size(), 2);
    int numerator = 0, denominator = 0;
    bool success = absl::SimpleAtoi(parts[0], &numerator);
    if (!success) {
      return false;
    }
    success = absl::SimpleAtoi(parts[1], &denominator);
    if (!success) {
      return false;
    }
    SPIEL_CHECK_FALSE(denominator == 0);
    *value = static_cast<double>(numerator) / denominator;
    return true;
  } else {
    // Otherwise, parse as a double.
    return absl::SimpleAtod(str, value);
  }
}

std::string EFGGame::NextToken() {
  std::string str = "";
  bool reading_quoted_string = false;

  if (string_data_.at(pos_) == '"') {
    reading_quoted_string = true;
    pos_++;
  }

  while (true) {
    // Check stopping condition:
    if (pos_ >= string_data_.length() ||
        (reading_quoted_string && string_data_.at(pos_) == '"') ||
        (!reading_quoted_string && IsWhiteSpace(string_data_.at(pos_)))) {
      break;
    }

    str.push_back(string_data_.at(pos_));
    pos_++;
  }

  if (reading_quoted_string) {
    SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
  }
  pos_++;

  // Advance the position to the next token.
  while (pos_ < string_data_.length() && IsWhiteSpace(string_data_.at(pos_))) {
    pos_++;
  }

  return str;
}

/*
EFG 2 R "General Bayes game, one stage" { "Player 1" "Player 2" }
c "ROOT" 1 "(0,1)" { "1G" 0.500000 "1B" 0.500000 } 0
c "" 2 "(0,2)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 1 "Outcome 1" { 10.000000 2.000000 }
t "" 2 "Outcome 2" { 0.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 3 "Outcome 3" { 2.000000 4.000000 }
t "" 4 "Outcome 4" { 4.000000 0.000000 }
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 5 "Outcome 5" { 10.000000 2.000000 }
t "" 6 "Outcome 6" { 0.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 7 "Outcome 7" { 2.000000 4.000000 }
t "" 8 "Outcome 8" { 4.000000 0.000000 }
c "" 3 "(0,3)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 9 "Outcome 9" { 4.000000 2.000000 }
t "" 10 "Outcome 10" { 2.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 11 "Outcome 11" { 0.000000 4.000000 }
t "" 12 "Outcome 12" { 10.000000 2.000000 }
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 13 "Outcome 13" { 4.000000 2.000000 }
t "" 14 "Outcome 14" { 2.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 15 "Outcome 15" { 0.000000 4.000000 }
t "" 16 "Outcome 16" { 10.000000 0.000000 }
*/
void EFGGame::ParsePrologue() {
  // Parse the first part of the header "EFG 2 R "
  SPIEL_CHECK_TRUE(NextToken() == "EFG");
  SPIEL_CHECK_LT(pos_, string_data_.length());
  SPIEL_CHECK_TRUE(NextToken() == "2");
  SPIEL_CHECK_LT(pos_, string_data_.length());
  SPIEL_CHECK_TRUE(NextToken() == "R");
  SPIEL_CHECK_LT(pos_, string_data_.length());
  SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
  name_ = NextToken();
  std::string token = NextToken();
  SPIEL_CHECK_TRUE(token == "{");
  SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
  token = NextToken();
  while (token != "}") {
    player_names_.push_back(token);
    token = NextToken();
  }
  num_players_ = player_names_.size();
  if (string_data_.at(pos_) == '"') {
    description_ = NextToken();
  }
  SPIEL_CHECK_LT(pos_, string_data_.length());
  SPIEL_CHECK_TRUE(IsNodeToken(string_data_.at(pos_)));
}

void EFGGame::ParseChanceNode(Node* parent, Node* child, int depth) {
  // a text string, giving the name of the node
  // a positive integer specifying the information set number
  // (optional) the name of the information set
  // (optional) a list of actions at the information set with their
  //      corresponding probabilities
  // a nonnegative integer specifying the outcome
  // (optional)the payoffs to each player for the outcome
  //
  // c "ROOT" 1 "(0,1)" { "1G" 0.500000 "1B" 0.500000 } 0
  SPIEL_CHECK_TRUE(NextToken() == "c");
  num_chance_nodes_++;
  max_depth_ = std::max(max_depth_, depth);
  child->type = NodeType::kChance;
  child->parent = parent;
  SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
  child->name = NextToken();
  SPIEL_CHECK_FALSE(string_data_.at(pos_) == '"');
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(NextToken(), &child->infoset_number));
  if (string_data_.at(pos_) == '"') {
    child->infoset_name = NextToken();
  }
  // I do not understand how the list of children can be optional.
  SPIEL_CHECK_TRUE(NextToken() == "{");
  int chance_outcomes = 0;
  double prob_sum = 0.0;
  while (string_data_.at(pos_) == '"') {
    child->actions.push_back(NextToken());
    double prob = -1;
    SPIEL_CHECK_TRUE(ParseDoubleValue(NextToken(), &prob));
    SPIEL_CHECK_GE(prob, 0.0);
    SPIEL_CHECK_LE(prob, 1.0);
    prob_sum += prob;
    child->probs.push_back(prob);
    nodes_.push_back(NewNode());
    child->children.push_back(nodes_.back().get());
    chance_outcomes++;
  }
  SPIEL_CHECK_GT(child->actions.size(), 0);
  SPIEL_CHECK_TRUE(Near(prob_sum, 1.0));
  max_chance_outcomes_ = std::max(max_chance_outcomes_, chance_outcomes);
  SPIEL_CHECK_TRUE(NextToken() == "}");
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(NextToken(), &child->outcome_number));
  // Do not support optional payoffs here for now.

  // Now, recurse:
  for (Node* grand_child : child->children) {
    RecParseSubtree(child, grand_child, depth + 1);
  }
}

void EFGGame::ParsePlayerNode(Node* parent, Node* child, int depth) {
  // a text string, giving the name of the node
  // a positive integer specifying the player who owns the node
  // a positive integer specifying the information set
  // (optional) the name of the information set
  // (optional) a list of action names for the information set
  // a nonnegative integer specifying the outcome
  // (optional) the name of the outcome
  // the payoffs to each player for the outcome
  //
  // p "" 1 1 "(1,1)" { "H" "L" } 0
  SPIEL_CHECK_TRUE(NextToken() == "p");
  max_depth_ = std::max(max_depth_, depth);
  child->type = NodeType::kPlayer;
  child->parent = parent;
  SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
  child->name = NextToken();
  SPIEL_CHECK_FALSE(string_data_.at(pos_) == '"');
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(NextToken(), &child->player_number));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(NextToken(), &child->infoset_number));
  infoset_num_to_states_count_[child->infoset_number] += 1;
  if (infoset_num_to_states_count_[child->infoset_number]) {
    perfect_information_ = false;
  }
  if (string_data_.at(pos_) == '"') {
    child->infoset_name = NextToken();
  }
  // Do not understand how the list of actions can be optional.
  SPIEL_CHECK_TRUE(NextToken() == "{");
  int actions = 0;
  while (string_data_.at(pos_) == '"') {
    child->actions.push_back(NextToken());
    nodes_.push_back(NewNode());
    child->children.push_back(nodes_.back().get());
    actions++;
  }
  SPIEL_CHECK_GT(child->actions.size(), 0);
  max_actions_ = std::max(max_actions_, actions);
  SPIEL_CHECK_TRUE(NextToken() == "}");
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(NextToken(), &child->outcome_number));
  // Do not support optional payoffs here for now.

  // Now, recurse:
  for (Node* grand_child : child->children) {
    RecParseSubtree(child, grand_child, depth + 1);
  }
}

void EFGGame::ParseTerminalNode(Node* parent, Node* child, int depth) {
  // a text string, giving the name of the node
  // a nonnegative integer specifying the outcome
  // (optional) the name of the outcome
  // the payoffs to each player for the outcome
  //
  // t "" 1 "Outcome 1" { 10.000000 2.000000 }
  SPIEL_CHECK_TRUE(NextToken() == "t");
  max_depth_ = std::max(max_depth_, depth);
  child->type = NodeType::kTerminal;
  child->parent = parent;
  SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
  child->name = NextToken();
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(NextToken(), &child->outcome_number));
  if (string_data_.at(pos_) == '"') {
    child->outcome_name = NextToken();
  }
  SPIEL_CHECK_TRUE(NextToken() == "{");

  int idx = 0;
  double util_sum = 0;
  bool identical = true;
  while (string_data_.at(pos_) != '}') {
    double utility = 0;
    SPIEL_CHECK_TRUE(ParseDoubleValue(NextToken(), &utility));
    child->payoffs.push_back(utility);
    util_sum += utility;
    if (!min_util_.has_value()) {
      min_util_ = utility;
    }
    if (!max_util_.has_value()) {
      max_util_ = utility;
    }
    min_util_ = std::min(min_util_.value(), utility);
    max_util_ = std::max(max_util_.value(), utility);

    if (identical && idx >= 1 &&
        Near(child->payoffs[idx - 1], child->payoffs[idx])) {
      identical = true;
    } else {
      identical = false;
    }

    idx++;
  }
  SPIEL_CHECK_EQ(child->payoffs.size(), num_players_);
  SPIEL_CHECK_TRUE(NextToken() == "}");

  // Inspect the utilities to classify the utility type for this game.
  if (!util_sum_.has_value()) {
    util_sum_ = util_sum;
  }

  if (constant_sum_ && Near(util_sum_.value(), util_sum)) {
    constant_sum_ = true;
  } else {
    constant_sum_ = false;
  }

  if (identical_payoffs_ && identical) {
    identical_payoffs_ = true;
  } else {
    identical_payoffs_ = false;
  }
}

void EFGGame::RecParseSubtree(Node* parent, Node* child, int depth) {
  switch (string_data_.at(pos_)) {
    case 'c':
      ParseChanceNode(parent, child, depth);
      break;
    case 'p':
      ParsePlayerNode(parent, child, depth);
      break;
    case 't':
      ParseTerminalNode(parent, child, depth);
      break;
    default:
      SpielFatalError(absl::StrCat("Unexpected character at pos ", pos_, ": ",
                                   string_data_.substr(pos_, 1)));
  }
}

std::string EFGGame::PrettyTree(const Node* node,
                                const std::string& indent) const {
  std::string str = indent + NodeToString(node);
  for (Node* child : node->children) {
    str += PrettyTree(child, indent + "  ");
  }
  return str;
}

void EFGGame::ParseGame() {
  // Skip any initial whitespace.
  while (IsWhiteSpace(string_data_.at(pos_))) {
    pos_++;
  }
  SPIEL_CHECK_LT(pos_, string_data_.length());

  ParsePrologue();
  nodes_.push_back(NewNode());
  RecParseSubtree(nullptr, nodes_[0].get(), 0);
  SPIEL_CHECK_GE(pos_, string_data_.length());

  // Modify the game type.
  if (num_chance_nodes_ > 0) {
    game_type_.chance_mode = GameType::ChanceMode::kExplicitStochastic;
  }

  if (perfect_information_) {
    game_type_.information = GameType::Information::kPerfectInformation;
  } else {
    game_type_.information = GameType::Information::kImperfectInformation;
  }

  if (constant_sum_ && Near(util_sum_.value(), 0.0)) {
    game_type_.utility = GameType::Utility::kZeroSum;
  } else if (constant_sum_) {
    game_type_.utility = GameType::Utility::kConstantSum;
  } else if (identical_payoffs_) {
    game_type_.utility = GameType::Utility::kIdentical;
  } else {
    game_type_.utility = GameType::Utility::kGeneralSum;
  }

  game_type_.max_num_players = num_players_;
  game_type_.min_num_players = num_players_;
}

}  // namespace efg_game
}  // namespace open_spiel
