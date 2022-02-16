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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_GIN_RUMMY_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_GIN_RUMMY_H_

// Implementation of the classic card game:
// https://en.wikipedia.org/wiki/Gin_rummy
//
// Gin rummy is played with many variations. Here we closely follow
// the rules described in http://ginrummytournaments.com/pdfs/GRA_Rules.pdf
//
// A game consists of a single hand of gin (i.e. this implementation does not
// support a full game, consisting of multiple hands, played to some given
// point total, usually in the 100-300 point range).
//
// Gin is a large game with over 10^85 information states and a large number
// of states per information state. Off the deal there are 41 choose 10 =
// 1,121,099,408 possible opponent hands, compared to 50 choose 2 = 1,225 in
// heads up Texas hold 'em poker.
//
// Parameters:
//  "oklahoma"        bool   use oklahoma variation?       (default = false)
//  "knock_card"      int    set a specific knock card     (default = 10)
//  "gin_bonus"       int    bonus for getting gin         (default = 25)
//  "undercut_bonus"  int    bonus for an undercut         (default = 25)

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace gin_rummy {

constexpr int kNumPlayers = 2;
constexpr int kMaxPossibleDeadwood = 98;  // E.g. KsKcQdQhJsJcTdTh9s9c
constexpr int kMaxNumDrawUpcardActions = 50;
constexpr int kHandSize = 10;
constexpr int kMaxStockSize = 31;  // Stock size when play begins
constexpr int kWallStockSize = 2;
constexpr int kDefaultKnockCard = 10;
constexpr int kDefaultGinBonus = 25;
constexpr int kDefaultUndercutBonus = 25;
constexpr int kDrawUpcardAction = 52;
constexpr int kDrawStockAction = 53;
constexpr int kPassAction = 54;
constexpr int kKnockAction = 55;
constexpr int kMeldActionBase = 56;  // First lay meld action
constexpr int kNumMeldActions = 185;
constexpr int kNumDistinctActions = kMeldActionBase + kNumMeldActions;
constexpr int kObservationTensorSize =
    kNumPlayers          // Player turn
    + kDefaultKnockCard  // Knock card
    + kNumCards          // Player hand
    + kNumCards          // Upcard
    + kNumCards          // Discard pile
    + kMaxStockSize      // Stock size
    + kNumMeldActions;   // Opponent's layed melds

class GinRummyState : public State {
 public:
  explicit GinRummyState(std::shared_ptr<const Game> game, bool oklahoma,
                         int knock_card, int gin_bonus, int undercut_bonus);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  enum class Phase {
    kDeal,
    kFirstUpcard,
    kDraw,
    kDiscard,
    kKnock,
    kLayoff,
    kWall,
    kGameOver
  };

  inline static constexpr std::array<absl::string_view, 8> kPhaseString = {
      "Deal",  "FirstUpcard", "Draw", "Discard",
      "Knock", "Layoff",      "Wall", "GameOver"};

  std::vector<Action> DealLegalActions() const;
  std::vector<Action> FirstUpcardLegalActions() const;
  std::vector<Action> DrawLegalActions() const;
  std::vector<Action> DiscardLegalActions() const;
  std::vector<Action> KnockLegalActions() const;
  std::vector<Action> LayoffLegalActions() const;
  std::vector<Action> WallLegalActions() const;

  void ApplyDealAction(Action action);
  void ApplyFirstUpcardAction(Action action);
  void ApplyDrawAction(Action action);
  void ApplyDiscardAction(Action action);
  void ApplyKnockAction(Action action);
  void ApplyLayoffAction(Action action);
  void ApplyWallAction(Action action);

  void StockToHand(Player player, Action card);
  void StockToUpcard(Action card);
  void UpcardToHand(Player player);
  void HandToUpcard(Player player, Action card);
  void RemoveFromHand(Player player, Action card);

  int Opponent(int player) const { return 1 - player; }

  const bool oklahoma_;  // If true, will override the knock card value.
  int knock_card_;       // The maximum deadwood total for a legal knock.
  const int gin_bonus_;
  const int undercut_bonus_;

  Phase phase_ = Phase::kDeal;
  Player cur_player_ = kChancePlayerId;
  Player prev_player_ = kChancePlayerId;
  bool finished_layoffs_ = false;
  absl::optional<int> upcard_;
  absl::optional<int> prev_upcard_;  // Used to track repeated moves.
  int stock_size_ = kNumCards;       // Number of cards remaining in stock.
  // True if the prev player drew the upcard only to immediately discard it.
  // If both players do this in succession the game is declared a draw.
  bool repeated_move_ = false;
  // Incremented every time a player draws the upcard. Used to ensure the game
  // is finite. See gin_rummy_test for an example of why this is needed.
  int num_draw_upcard_actions_ = 0;

  // Each player's hand. Indexed by pid.
  std::vector<std::vector<int>> hands_ =
      std::vector<std::vector<int>>(kNumPlayers, std::vector<int>());
  // True if the card is still in the deck. Cards from 0-51 using the suit order
  // "scdh".
  std::vector<bool> deck_{};
  // Discard pile consists of cards that are out of play.
  std::vector<int> discard_pile_{};
  // Prior to a knock, deadwood tracks the minimum possible deadwood count
  // over all meld arrangements, indexed by pid. When player has 11 cards, it
  // counts the best 10 of 11 (the player can discard).
  // After a knock, deadwood counts the total card value of all cards in the
  // player's hand. Points are deducted as the player lays the hand into melds
  // or lays off cards onto opponent melds.
  std::vector<int> deadwood_ = std::vector<int>(kNumPlayers, 0);
  // Flag for whether the player has knocked. Indexed by pid.
  std::vector<bool> knocked_ = std::vector<bool>(kNumPlayers, false);
  // Flag for whether the player has passed on first upcard. Indexed by pid.
  std::vector<bool> pass_on_first_upcard_ =
      std::vector<bool>(kNumPlayers, false);
  // Each player's layed melds during the knock phase. Indexed by pid.
  std::vector<std::vector<int>> layed_melds_ =
      std::vector<std::vector<int>>(kNumPlayers, std::vector<int>());
  // Cards that have been layed off onto knocking player's layed melds.
  std::vector<int> layoffs_{};
};

class GinRummyGame : public Game {
 public:
  explicit GinRummyGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  int MaxChanceOutcomes() const override { return kNumCards; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override {
    return -(kMaxPossibleDeadwood + gin_bonus_);
  }
  double MaxUtility() const override {
    return kMaxPossibleDeadwood + gin_bonus_;
  }
  double UtilitySum() const override { return 0; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new GinRummyState(shared_from_this(), oklahoma_, knock_card_,
                          gin_bonus_, undercut_bonus_));
  }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new GinRummyGame(*this));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }
  // All games should terminate before reaching this upper bound.
  int MaxGameLength() const override { return 300; }

 private:
  const bool oklahoma_;
  const int knock_card_;
  const int gin_bonus_;
  const int undercut_bonus_;
};

}  // namespace gin_rummy
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_GIN_RUMMY_H_
