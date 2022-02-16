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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_GOOFSPIEL_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_GOOFSPIEL_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// Goofspiel, or the Game of Pure Strategy, is a bidding card game where players
// are trying to obtain the most points. In, Goofspiel(K), each player has bid
// cards numbered 1-K and a point card deck containing cards numbered 1-K is
// shuffled and set face-down. Each turn, the top point card is revealed, and
// players simultaneously play a bid card; the point card is given to the
// highest bidder or discarded if the bids are equal. For more detail, see:
// https://en.wikipedia.org/wiki/Goofspiel
//
// This implementation of Goofspiel is slightly more general than the standard
// game. First, more than 2 players can play it. Second, the deck can take on
// pre-determined orders rather than randomly determined. Third, there is an
// option to enable the imperfect information variant described in Sec 3.1.4
// of http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf, where only
// the sequences of wins / losses is revealed (not the players' hands).
//
// Parameters:
//   "imp_info"      bool     Enable the imperfect info variant (default: false)
//   "num_cards"     int      The highest bid card, and point card (default: 13)
//   "players"       int      number of players (default: 2)
//   "points_order"  string   "random" (default), "descending", or "ascending"

namespace open_spiel {
namespace goofspiel {

constexpr int kDefaultNumPlayers = 2;
constexpr int kDefaultNumCards = 13;
constexpr const char* kDefaultPointsOrder = "random";
constexpr const bool kDefaultImpInfo = false;

enum class PointsOrder {
  kRandom,
  kDescending,
  kAscending,
};

constexpr int kInvalidCard = -1;

class GoofspielState : public SimMoveState {
 public:
  explicit GoofspielState(std::shared_ptr<const Game> game, int num_cards,
                          PointsOrder points_order, bool impinfo);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  void InformationStateTensor(Player player,
                              std::vector<double>* values) const override;
  void ObservationTensor(Player player,
                         std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  // Increments the count and increments the player mod num_players_.
  void NextPlayer(int* count, Player* player) const;

  int num_cards_;
  PointsOrder points_order_;
  bool impinfo_;

  Player current_player_;
  std::set<int> winners_;
  int turns_;
  int point_card_index_;
  std::vector<int> points_;
  std::vector<int> point_deck_;                  // Current point deck.
  std::vector<std::vector<bool>> player_hands_;  // true if card is in hand.
  std::vector<int> point_card_sequence_;
  std::vector<int> win_sequence_;  // Which player won
  std::vector<std::vector<Action>> actions_history_;
};

class GoofspielGame : public Game {
 public:
  explicit GoofspielGame(const GameParameters& params);

  int NumDistinctActions() const override { return num_cards_; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return +1; }
  double UtilitySum() const override { return 0; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new GoofspielGame(*this));
  }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return num_cards_; }

 private:
  int num_cards_;    // The K in Goofspiel(K)
  int num_players_;  // Number of players
  PointsOrder points_order_;
  bool impinfo_;
};

}  // namespace goofspiel
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_GOOFSPIEL_H_
