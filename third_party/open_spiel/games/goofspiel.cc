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

#include "open_spiel/games/goofspiel.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace goofspiel {
namespace {

const GameType kGameType{
    /*short_name=*/"goofspiel",
    /*long_name=*/"Goofspiel",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"imp_info", GameParameter(kDefaultImpInfo)},
     {"num_cards", GameParameter(kDefaultNumCards)},
     {"players", GameParameter(kDefaultNumPlayers)},
     {"points_order",
      GameParameter(static_cast<std::string>(kDefaultPointsOrder))}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GoofspielGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

PointsOrder ParsePointsOrder(const std::string& po_str) {
  if (po_str == "random") {
    return PointsOrder::kRandom;
  } else if (po_str == "descending") {
    return PointsOrder::kDescending;
  } else if (po_str == "ascending") {
    return PointsOrder::kAscending;
  } else {
    SpielFatalError("Unrecognized pointsorder parameter: " + po_str);
  }
}

}  // namespace

GoofspielState::GoofspielState(std::shared_ptr<const Game> game, int num_cards,
                               PointsOrder points_order, bool impinfo)
    : SimMoveState(game),
      num_cards_(num_cards),
      points_order_(points_order),
      impinfo_(impinfo),
      current_player_(kInvalidPlayer),
      winners_({}),
      turns_(0),
      point_card_index_(-1),
      point_card_sequence_({}),
      win_sequence_({}),
      actions_history_({}) {
  // Points and point-card deck.
  points_.resize(num_players_);
  std::fill(points_.begin(), points_.end(), 0);
  point_deck_.resize(num_cards_);
  for (int point_value = 1; point_value <= num_cards_; ++point_value) {
    point_deck_[point_value - 1] = point_value;
  }

  // Player hands.
  player_hands_.clear();
  for (auto p = Player{0}; p < num_players_; ++p) {
    std::vector<bool> hand(num_cards_, true);
    player_hands_.push_back(hand);
  }

  // Set the points card index.
  if (points_order_ == PointsOrder::kRandom) {
    point_card_index_ = -1;
    current_player_ = kChancePlayerId;
  } else if (points_order_ == PointsOrder::kAscending) {
    point_card_index_ = 0;
    current_player_ = kSimultaneousPlayerId;
  } else if (points_order_ == PointsOrder::kDescending) {
    point_card_index_ = num_cards - 1;
    current_player_ = kSimultaneousPlayerId;
  }
}

int GoofspielState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void GoofspielState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  point_card_index_ = action_id;
  SPIEL_CHECK_GE(point_card_index_, 0);
  SPIEL_CHECK_LT(point_card_index_, point_deck_.size());
  point_card_sequence_.push_back(point_deck_[point_card_index_]);
  current_player_ = kSimultaneousPlayerId;
}

void GoofspielState::DoApplyActions(const std::vector<Action>& actions) {
  // Check the actions are valid.
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  for (auto p = Player{0}; p < num_players_; ++p) {
    const int action = actions[p];
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, num_cards_);
    SPIEL_CHECK_TRUE(player_hands_[p][action]);
  }

  // Find the highest bid
  int max_bid = -1;
  int num_max_bids = 0;
  int max_bidder = -1;

  for (int p = 0; p < actions.size(); ++p) {
    if (actions[p] > max_bid) {
      max_bid = actions[p];
      num_max_bids = 1;
      max_bidder = p;
    } else if (actions[p] == max_bid) {
      num_max_bids++;
    }
  }

  if (num_max_bids == 1) {
    // Winner takes the point card.
    points_[max_bidder] += point_deck_[point_card_index_];
    win_sequence_.push_back(max_bidder);
  } else {
    // Tied among several players: discarded.
    win_sequence_.push_back(kInvalidPlayer);
  }

  // Add these actions to the history.
  actions_history_.push_back(actions);

  // Remove the cards from the player's hands.
  for (auto p = Player{0}; p < num_players_; ++p) {
    player_hands_[p][actions[p]] = false;
  }

  // Next player's turn.
  if (points_order_ == PointsOrder::kRandom) {
    current_player_ = kChancePlayerId;
    point_deck_.erase(point_deck_.begin() + point_card_index_);
    point_card_index_ = -1;
  } else if (points_order_ == PointsOrder::kAscending) {
    point_card_index_++;
  } else if (points_order_ == PointsOrder::kDescending) {
    point_card_index_--;
  }

  turns_++;

  // No choice at the last turn, so we can play it now
  // We use DoApplyAction(s) not to modify the history, as these actions are
  // not available in the tree.
  if (turns_ == num_cards_ - 1) {
    // There might be a chance event
    if (IsChanceNode()) {
      auto legal_actions = LegalChanceOutcomes();
      SPIEL_CHECK_EQ(legal_actions.size(), 1);
      DoApplyAction(legal_actions.front());
    }

    // Each player plays their last card
    std::vector<Action> actions(num_players_);
    for (auto p = Player{0}; p < num_players_; ++p) {
      auto legal_actions = LegalActions(p);
      SPIEL_CHECK_EQ(legal_actions.size(), 1);
      actions[p] = legal_actions[0];
    }
    DoApplyActions(actions);
  } else if (turns_ == num_cards_) {
    // Game over - determine winner.
    int max_points = -1;
    for (auto p = Player{0}; p < num_players_; ++p) {
      if (points_[p] > max_points) {
        winners_.clear();
        max_points = points_[p];
        winners_.insert(p);
      } else if (points_[p] == max_points) {
        winners_.insert(p);
      }
    }
  }
}

std::vector<std::pair<Action, double>> GoofspielState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes(point_deck_.size());
  for (int i = 0; i < point_deck_.size(); i++) {
    outcomes[i] = std::pair<Action, double>(i, 1.0 / point_deck_.size());
  }
  return outcomes;
}

std::vector<Action> GoofspielState::LegalActions(Player player) const {
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
  if (player == kChancePlayerId) return LegalChanceOutcomes();
  if (player == kTerminalPlayerId) return std::vector<Action>();
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::vector<Action> movelist;
  for (int bid = 0; bid < player_hands_[player].size(); ++bid) {
    if (player_hands_[player][bid]) {
      movelist.push_back(bid);
    }
  }
  return movelist;
}

std::string GoofspielState::ActionToString(Player player,
                                           Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, num_cards_);
  std::string result = "";
  absl::StrAppend(&result, "[P", player, "]Bid: ", (action_id + 1));
  return result;
}

std::string GoofspielState::ToString() const {
  std::string points_line = "Points: ";
  std::string result = "";

  for (auto p = Player{0}; p < num_players_; ++p) {
    absl::StrAppend(&points_line, points_[p]);
    absl::StrAppend(&points_line, " ");
    absl::StrAppend(&result, "P");
    absl::StrAppend(&result, p);
    absl::StrAppend(&result, " hand: ");
    for (int c = 0; c < num_cards_; ++c) {
      if (player_hands_[p][c]) {
        absl::StrAppend(&result, c + 1);
        absl::StrAppend(&result, " ");
      }
    }
    absl::StrAppend(&result, "\n");
  }

  // In imperfect information, the full state depends on both betting sequences
  if (impinfo_) {
    for (auto p = Player{0}; p < num_players_; ++p) {
      absl::StrAppend(&result, "P", p, " actions: ");
      for (int i = 0; i < actions_history_.size(); ++i) {
        absl::StrAppend(&result, actions_history_[i][p]);
        absl::StrAppend(&result, " ");
      }
      absl::StrAppend(&result, "\n");
    }
  }

  absl::StrAppend(&result, "Point card sequence: ");
  for (int i = 0; i < point_card_sequence_.size(); ++i) {
    absl::StrAppend(&result, point_card_sequence_[i], " ");
  }
  absl::StrAppend(&result, "\n");

  return result + points_line + "\n";
}

bool GoofspielState::IsTerminal() const { return (turns_ == num_cards_); }

std::vector<double> GoofspielState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  if (winners_.size() == num_players_) {
    // All players have same number of points? This is a draw.
    return std::vector<double>(num_players_, 0.0);
  } else {
    int num_winners = winners_.size();
    int num_losers = num_players_ - num_winners;
    std::vector<double> returns(num_players_, (-1.0 / num_losers));
    for (const auto& winner : winners_) {
      returns[winner] = 1.0 / num_winners;
    }
    return returns;
  }
}

std::string GoofspielState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (impinfo_) {
    std::string points_line = "Points: ";
    std::string win_sequence = "Win sequence: ";
    std::string result = "";

    // Show the points for all players.
    // Only know the observing player's hand.
    // Only know the observing player's action sequence.
    // Know the win-loss outcome of each step.
    // Show the point card sequence from the start.

    for (auto p = Player{0}; p < num_players_; ++p) {
      absl::StrAppend(&points_line, points_[p]);
      absl::StrAppend(&points_line, " ");

      if (p == player) {
        // Only show this player's hand
        absl::StrAppend(&result, "P");
        absl::StrAppend(&result, p);
        absl::StrAppend(&result, " hand: ");
        for (int c = 0; c < num_cards_; ++c) {
          if (player_hands_[p][c]) {
            absl::StrAppend(&result, c + 1);
            absl::StrAppend(&result, " ");
          }
        }
        absl::StrAppend(&result, "\n");

        // Also show the player's sequence. We need this to ensure perfect
        // recall because two betting sequences can lead to the same hand and
        // outcomes if the opponent chooses differently.
        absl::StrAppend(&result, "P");
        absl::StrAppend(&result, p);
        absl::StrAppend(&result, " action sequence: ");
        for (int i = 0; i < actions_history_.size(); ++i) {
          absl::StrAppend(&result, actions_history_[i][p]);
          absl::StrAppend(&result, " ");
        }
        absl::StrAppend(&result, "\n");
      }
    }

    for (int i = 0; i < win_sequence_.size(); ++i) {
      absl::StrAppend(&win_sequence, win_sequence_[i]);
      win_sequence.push_back(' ');
    }

    absl::StrAppend(&result, "Point card sequence: ");
    for (int i = 0; i < point_card_sequence_.size(); ++i) {
      absl::StrAppend(&result, point_card_sequence_[i], " ");
    }
    absl::StrAppend(&result, "\n");

    return result + win_sequence + "\n" + points_line + "\n";
  } else {
    // All the information is public.
    return ToString();
  }
}

std::string GoofspielState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Perfect info case, show:
  //   - current point card
  //   - everyone's current points
  //   - everyone's current hands
  // Imperfect info case, show:
  //   - current point card
  //   - everyone's current points
  //   - my current hand
  //   - current win sequence
  std::string current_trick =
      absl::StrCat("Current point card: ", point_card_index_ + 1);
  std::string points_line = "Points: ";
  std::string hands = "";
  std::string win_seq = "Win Sequence: ";
  for (auto p = Player{0}; p < num_players_; ++p) {
    absl::StrAppend(&points_line, points_[p], " ");
  }

  if (impinfo_) {
    // Only my hand
    absl::StrAppend(&hands, "P", player, " hand: ");
    for (int c = 0; c < num_cards_; ++c) {
      if (player_hands_[player][c]) {
        absl::StrAppend(&hands, c + 1, " ");
      }
    }
    absl::StrAppend(&hands, "\n");

    // Show the win sequence.
    for (int i = 0; i < win_sequence_.size(); ++i) {
      absl::StrAppend(&win_seq, win_sequence_[i], " ");
    }
    return absl::StrCat(current_trick, "\n", points_line, "\n", hands, win_seq,
                        "\n");
  } else {
    // Show the hands in the perfect info case.
    for (auto p = Player{0}; p < num_players_; ++p) {
      absl::StrAppend(&hands, "P", p, " hand: ");
      for (int c = 0; c < num_cards_; ++c) {
        if (player_hands_[p][c]) {
          absl::StrAppend(&hands, c + 1, " ");
        }
      }
      absl::StrAppend(&hands, "\n");
    }
    return absl::StrCat(current_trick, "\n", points_line, "\n", hands);
  }
}

void GoofspielState::NextPlayer(int* count, Player* player) const {
  *count += 1;
  *player = (*player + 1) % num_players_;
}

void GoofspielState::InformationStateTensor(Player player,
                                            std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->clear();
  values->reserve(game_->InformationStateTensorSize());

  // Point totals: one-hot vector encoding points, per player.
  Player p = player;
  for (int n = 0; n < num_players_; NextPlayer(&n, &p)) {
    // Cards numbered 1 .. K
    int max_points_slots = (num_cards_ * (num_cards_ + 1)) / 2 + 1;
    for (int i = 0; i < max_points_slots; ++i) {
      values->push_back(i == points_[p] ? 1 : 0);
    }
  }

  if (impinfo_) {
    // Bit vector of observing player's hand.
    for (int c = 0; c < num_cards_; ++c) {
      values->push_back(player_hands_[player][c] ? 1 : 0);
    }

    // Sequence of who won each trick.
    for (int i = 0; i < win_sequence_.size(); ++i) {
      for (auto p = Player{0}; p < num_players_; ++p) {
        values->push_back(win_sequence_[i] == p ? 1 : 0);
      }
    }

    // Padding for future tricks
    int future_tricks = num_cards_ - win_sequence_.size();
    for (int i = 0; i < future_tricks * num_players_; ++i) values->push_back(0);

    // Point card sequence.
    for (int i = 0; i < point_card_sequence_.size(); ++i) {
      for (int j = 0; j < num_cards_; ++j) {
        values->push_back(point_card_sequence_[i] == (j + 1) ? 1 : 0);
      }
    }

    // Padding for future tricks
    future_tricks = num_cards_ - point_card_sequence_.size();
    for (int i = 0; i < future_tricks * num_cards_; ++i) values->push_back(0);

    // The observing player's action sequence.
    for (int i = 0; i < num_cards_; ++i) {
      for (int c = 0; c < num_cards_; ++c) {
        values->push_back(i < actions_history_.size() &&
                                  actions_history_[i][player] == c
                              ? 1
                              : 0);
      }
    }

  } else {
    // Point card sequence.
    for (int i = 0; i < point_card_sequence_.size(); ++i) {
      for (int j = 0; j < num_cards_; ++j) {
        values->push_back(point_card_sequence_[i] == (j + 1) ? 1 : 0);
      }
    }

    // Padding for future tricks
    int future_tricks = num_cards_ - point_card_sequence_.size();
    for (int i = 0; i < future_tricks * num_cards_; ++i) values->push_back(0);

    // Bit vectors encoding all players' hands.
    p = player;
    for (int n = 0; n < num_players_; NextPlayer(&n, &p)) {
      for (int c = 0; c < num_cards_; ++c) {
        values->push_back(player_hands_[p][c] ? 1 : 0);
      }
    }
  }

  SPIEL_CHECK_EQ(values->size(), game_->InformationStateTensorSize());
}

void GoofspielState::ObservationTensor(Player player,
                                       std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->clear();
  values->reserve(game_->ObservationTensorSize());

  // Perfect info case, show:
  //   - one-hot encoding the current point card
  //   - everyone's current points
  //   - everyone's current hands
  // Imperfect info case, show:
  //   - one-hot encoding the current point card
  //   - everyone's current points
  //   - my current hand
  //   - current win sequence

  // Current point card.
  for (int i = 0; i < num_cards_; ++i) {
    values->push_back(i == point_card_index_ ? 1.0 : 0.0);
  }

  // Point totals: one-hot vector encoding points, per player.
  Player p = player;
  for (int n = 0; n < num_players_; NextPlayer(&n, &p)) {
    // Cards numbered 1 .. K
    int max_points_slots = (num_cards_ * (num_cards_ + 1)) / 2 + 1;
    for (int i = 0; i < max_points_slots; ++i) {
      values->push_back(i == points_[p] ? 1 : 0);
    }
  }

  if (impinfo_) {
    // Bit vector of observing player's hand.
    for (int c = 0; c < num_cards_; ++c) {
      values->push_back(player_hands_[player][c] ? 1 : 0);
    }

    // Sequence of who won each trick.
    for (int i = 0; i < win_sequence_.size(); ++i) {
      for (auto p = Player{0}; p < num_players_; ++p) {
        values->push_back(win_sequence_[i] == p ? 1 : 0);
      }
    }

    // Padding for future tricks
    int future_tricks = num_cards_ - win_sequence_.size();
    for (int i = 0; i < future_tricks * num_players_; ++i) values->push_back(0);
  } else {
    // Bit vectors encoding all players' hands.
    p = player;
    for (int n = 0; n < num_players_; NextPlayer(&n, &p)) {
      for (int c = 0; c < num_cards_; ++c) {
        values->push_back(player_hands_[p][c] ? 1 : 0);
      }
    }
  }

  SPIEL_CHECK_EQ(values->size(), game_->ObservationTensorSize());
}

std::unique_ptr<State> GoofspielState::Clone() const {
  return std::unique_ptr<State>(new GoofspielState(*this));
}

GoofspielGame::GoofspielGame(const GameParameters& params)
    : Game(kGameType, params),
      num_cards_(ParameterValue<int>("num_cards")),
      num_players_(ParameterValue<int>("players")),
      points_order_(
          ParsePointsOrder(ParameterValue<std::string>("points_order"))),
      impinfo_(ParameterValue<bool>("imp_info")) {}

std::unique_ptr<State> GoofspielGame::NewInitialState() const {
  return std::unique_ptr<State>(new GoofspielState(
      shared_from_this(), num_cards_, points_order_, impinfo_));
}

int GoofspielGame::MaxChanceOutcomes() const {
  if (points_order_ == PointsOrder::kRandom) {
    return num_cards_;
  } else {
    return 0;
  }
}

std::vector<int> GoofspielGame::InformationStateTensorShape() const {
  if (impinfo_) {
    return {// 1-hot bit vector for point total per player; upper bound is 1 +
            // 2 + ... + K = K*(K+1) / 2, but must add one to include 0 points.
            num_players_ * ((num_cards_ * (num_cards_ + 1)) / 2 + 1) +
            // Bit vector for my remaining cards:
            num_cards_ +
            // A sequence of 1-hot bit vectors encoding the player who won that
            // turn, where max number of turns is num_cards
            num_cards_ * num_players_ +
            // A sequence of 1-hot bit vectors encoding the point card sequence
            num_cards_ * num_cards_ +
            // The observing player's own action sequence
            num_cards_ * num_cards_};
  } else {
    return {// 1-hot bit vector for point total per player; upper bound is 1 +
            // 2 + ... + K = K*(K+1) / 2, but must add one to include 0 points.
            num_players_ * ((num_cards_ * (num_cards_ + 1)) / 2 + 1) +
            // A sequence of 1-hot bit vectors encoding the point card sequence
            num_cards_ * num_cards_ +
            // Bit vector for each card per player
            num_players_ * num_cards_};
  }
}

std::vector<int> GoofspielGame::ObservationTensorShape() const {
  // Perfect info case, show:
  //   - current point card showing
  //   - everyone's current points
  //   - everyone's current hands
  // Imperfect info case, show:
  //   - current point card showing
  //   - everyone's current points
  //   - my current hand
  //   - current win sequence
  if (impinfo_) {
    return {// 1-hot bit to encode the current point card
            num_cards_ +
            // 1-hot bit vector for point total per player; upper bound is 1 +
            // 2 + ... + K = K*(K+1) / 2, but must add one to include 0 points.
            num_players_ * ((num_cards_ * (num_cards_ + 1)) / 2 + 1) +
            // Bit vector for my remaining cards:
            num_cards_ +
            // A sequence of 1-hot bit vectors encoding the player who won that
            // turn, where max number of turns is num_cards
            num_cards_ * num_players_};
  } else {
    return {// 1-hot bit to encode the current point card
            num_cards_ +
            // 1-hot bit vector for point total per player; upper bound is 1 +
            // 2 + ... + K = K*(K+1) / 2, but must add one to include 0 points.
            num_players_ * ((num_cards_ * (num_cards_ + 1)) / 2 + 1) +
            // Bit vector for each card per player
            num_players_ * num_cards_};
  }
}

}  // namespace goofspiel
}  // namespace open_spiel
