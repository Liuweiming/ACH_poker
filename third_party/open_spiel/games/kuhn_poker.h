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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_KUHN_POKER_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_KUHN_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple game that includes chance and imperfect information
// http://en.wikipedia.org/wiki/Kuhn_poker
//
// For more information on this game (e.g. equilibrium sets, etc.) see
// http://poker.cs.ualberta.ca/publications/AAAI05.pdf
//
// The multiplayer (n>2) version is the one described in
// http://mlanctot.info/files/papers/aamas14sfrd-cfr-kuhn.pdf
//
// Parameters:
//     "players"       int    number of players               (default = 2)

namespace open_spiel {
namespace kuhn_poker {

constexpr int kNumInfoStatesP0 = 6;
constexpr int kNumInfoStatesP1 = 6;

enum ActionType { kPass = 0, kBet = 1 };

class KuhnGame;

class KuhnState : public State {
 public:
  explicit KuhnState(std::shared_ptr<const Game> game);
  KuhnState(const KuhnState&) = default;

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
  void UndoAction(Player player, Action move) override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<int> hand() const { return {card_dealt_[CurrentPlayer()]}; }
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  // Whether the specified player made a bet
  bool DidBet(Player player) const;

  // The move history and number of players are sufficient information to
  // specify the state of the game. We keep track of more information to make
  // extracting legal actions and utilities easier.
  // The cost of the additional book-keeping is more complex ApplyAction() and
  // UndoAction() functions.
  int first_bettor_;             // the player (if any) who was first to bet
  std::vector<int> card_dealt_;  // the player (if any) who has each card
  int winner_;                   // winning player, or kInvalidPlayer if the
                                 // game isn't over yet.
  int pot_;                      // the size of the pot
  // How much each player has contributed to the pot, indexed by pid.
  std::vector<int> ante_;
};

class KuhnGame : public Game {
 public:
  explicit KuhnGame(const GameParameters& params);
  int NumDistinctActions() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return num_players_ + 1; }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new KuhnGame(*this));
  }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return num_players_ * 2 - 1; }

 private:
  // Number of players.
  int num_players_;
};

}  // namespace kuhn_poker
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_KUHN_POKER_H_
