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

// A generalized version of a Leduc poker, a simple but non-trivial poker game
// described in http://poker.cs.ualberta.ca/publications/UAI05.pdf .
//
// Taken verbatim from the linked paper above: "In Leduc hold'em, the deck
// consists of two suits with three cards in each suit. There are two rounds.
// In the first round a single private card is dealt to each player. In the
// second round a single board card is revealed. There is a two-bet maximum,
// with raise amounts of 2 and 4 in the first and second round, respectively.
// Both players start the first round with 1 already in the pot.
//
// So the maximin sequence is of the form:
// private card player 0, private card player 1, [bets], public card, [bets]
// Parameters:
//     "players"       int    number of players               (default = 2)

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_LEDUC_POKER_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_LEDUC_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace leduc_poker {

// Default parameters.

// TODO(b/127425075): Use absl::optional instead of sentinel values once absl is
// added as a dependency.
constexpr int kInvalidCard = -10000;
constexpr int kDefaultPlayers = 2;
constexpr int kNumSuits = 2;
constexpr int kFirstRaiseAmount = 2;
constexpr int kSecondRaiseAmount = 4;
constexpr int kTotalRaisesPerRound = 2;
constexpr int kMaxRaises = 2;
constexpr int kStartingMoney = 100;
constexpr int kNumInfoStates = 936;  // Number of info state in the 2P game.

class LeducGame;

enum ActionType { kFold = 0, kCall = 1, kRaise = 2 };

class LeducState : public State {
 public:
  explicit LeducState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
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
  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Additional methods
  int round() const { return round_; }
  int deck_size() const { return deck_size_; }
  int public_card() const { return public_card_; }
  int raises() const { return num_raises_; }
  int private_card(Player player) const { return private_cards_[player]; }
  std::vector<Action> LegalActions() const override;

  // Returns a vector of MaxGameLength containing all of the betting actions
  // taken so far. If the round has ended, the actions are kInvalidAction.
  std::vector<int> padded_betting_sequence() const;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

 protected:
  // The meaning of `action_id` varies:
  // - At decision nodes, one of ActionType::{kFold, kCall, kRaise}.
  // - At a chance node, indicates the card to be dealt to the player or
  // revealed publicly. The interpretation of each chance outcome depends on
  // the number of players, but always follows:
  //    lowest value of first suit,
  //    lowest value of second suit,
  //    next lowest value of first suit,
  //    next lowest value of second suit,
  //             .
  //             .
  //             .
  //    highest value of first suit,
  //    highest value of second suit.
  // So, e.g. in the two player case (6 cards): 0 = Jack1, 1 = Jack2,
  // 2 = Queen1, ... , 5 = King2.
  void DoApplyAction(Action move) override;

 private:
  int NextPlayer() const;
  void ResolveWinner();
  bool ReadyForNextRound() const;
  void NewRound();
  int RankHand(Player player) const;
  void SequenceAppendMove(int move);
  void Ante(Player player, int amount);
  void SetPrivate(Player player, Action move);

  // Fields sets to bad/invalid values. Use Game::NewInitialState().
  Player cur_player_;

  int num_calls_;    // Number of calls this round (total, not per player).
  int num_raises_;   // Number of raises made in the round (not per player).
  int round_;        // Round number (1 or 2).
  int stakes_;       // The current 'level' of the bet.
  int num_winners_;  // Number of winning players.
  int pot_;          // Number of chips in the pot.
  int public_card_;  // The public card revealed after round 1.
  int deck_size_;    // Number of cards remaining; not equal deck_.size()
  int private_cards_dealt_;  // How many private cards currently dealt.
  int remaining_players_;    // Num. players still in (not folded).

  // Is this player a winner? Indexed by pid.
  std::vector<bool> winner_;
  // Each player's single private card. Indexed by pid.
  std::vector<int> private_cards_;
  // Cards by value (0-6 for standard 2-player game, -1 if no longer in the
  // deck.)
  std::vector<int> deck_;
  // How much money each player has, indexed by pid.
  std::vector<double> money_;
  // How much each player has contributed to the pot, indexed by pid.
  std::vector<int> ante_;
  // Flag for whether the player has folded, indexed by pid.
  std::vector<bool> folded_;
  // Sequence of actions for each round. Needed to report information state.
  std::vector<int> round1_sequence_;
  std::vector<int> round2_sequence_;
};

class LeducGame : public Game {
 public:
  explicit LeducGame(const GameParameters& params);

  int NumDistinctActions() const override { return 3; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return total_cards_; }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new LeducGame(*this));
  }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    // 2 rounds. Longest one for e.g. 4-player is, e.g.:
    //   check, check, check, raise, call, call, raise, call, call, call
    // = 2 raises + (num_players_-1)*2 calls + (num_players_-2) calls
    return 2 * (2 + (num_players_ - 1) * 2 + (num_players_ - 2));
  }

 private:
  int num_players_;  // Number of players.
  int total_cards_;  // Number of cards total cards in the game.
};

}  // namespace leduc_poker
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_LEDUC_POKER_H_
