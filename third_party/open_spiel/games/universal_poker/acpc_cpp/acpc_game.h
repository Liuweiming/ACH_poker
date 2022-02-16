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

#ifndef OPEN_SPIEL_ACPC_GAME_H
#define OPEN_SPIEL_ACPC_GAME_H

#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"

namespace open_spiel {
namespace universal_poker {
namespace acpc_cpp {

// Same definition from project_acpc_server::game.h
constexpr int kMaxRounds = 4;
constexpr int kMaxPlayers = 10;
constexpr int kMaxBoardCards = 7;
constexpr int kMaxHoleCards = 3;
constexpr int kMaxNumActions = 64;
constexpr int kMaxSuits = 4;
constexpr int kMaxRanks = 13;

struct RawACPCGame : public ::project_acpc_server::Game {};
struct RawACPCState : public ::project_acpc_server::State {};
struct RawACPCAction : public ::project_acpc_server::Action {};

class ACPCGame;

class ACPCState {
 public:
  enum ACPCActionType { ACPC_FOLD, ACPC_CALL, ACPC_RAISE, ACPC_INVALID };

 public:
  ACPCState(const ACPCGame* game);
  ACPCState(const ACPCState& other);
  ACPCState(const ACPCGame* game, const RawACPCState& raw_state);
  virtual ~ACPCState();

  void SetCards(const uint8_t* cards, const uint8_t numCards);
  void SetHoleCards(uint8_t player, const uint8_t holeCards[3],
                    uint8_t nbHoleCards) const;
  void SetBoardCards(const uint8_t boardCards[7], uint8_t nbBoardCards) const;

  // The current player is the first player in a new round, or the next player
  // within a round.
  uint8_t CurrentPlayer() const;

  bool IsFinished() const;
  int RaiseIsValid(int32_t* minSize, int32_t* maxSize) const;
  int IsValidAction(const ACPCActionType actionType, const int32_t size) const;
  void DoAction(const ACPCActionType actionType, const int32_t size);
  double ValueOfState(const uint8_t player) const;
  uint32_t MaxSpend() const;
  // Returns the current round 0-indexed round id (<= game.NumRounds() - 1).
  // A showdown is still in game.NumRounds()-1, not a separate round
  int GetRound() const;
  int GetRaiseSize() const;
  int NumRaises() const;
  // Returns how many board cards are revealed.
  uint8_t SumBoardCards() const;
  uint8_t NumFolded() const;
  bool PlayerFolded(const uint8_t player) const {
    return acpc_state_->playerFolded[player];
  }
  uint32_t Money(const uint8_t player) const;
  uint32_t Ante(const uint8_t player) const;
  uint32_t TotalSpent() const;
  uint32_t CurrentSpent(const uint8_t player) const;
  std::string ToString() const;
  std::string BettingSequence(uint8_t round) const;
  std::vector<uint8_t> DealCards(std::mt19937* rng);
  std::vector<uint8_t> HoleCards(const uint8_t player) const;
  std::vector<uint8_t> BoardCards(uint8_t num_board_cards) const;
  std::string HoleCardsToString(const uint8_t player) const;
  std::string BoardCardsToString(uint8_t num_board_cards) const;

  const ACPCGame* game_;
  std::unique_ptr<RawACPCState> acpc_state_;
};

class ACPCGame {
 public:
  ACPCGame(const std::string& gameDef);
  ACPCGame(const ACPCGame& other);
  virtual ~ACPCGame();

  std::string ToString() const;
  bool IsLimitGame() const;
  // The total number of betting rounds.
  int NumRounds() const;
  std::vector<uint32_t> GetRaiseSize() const;
  std::vector<uint8_t> GetMaxRaises() const;
  uint32_t GetRaiseSize(uint8_t round) const;
  uint8_t GetMaxRaises(uint8_t round) const;
  int GetNbPlayers() const;
  // Returns the number of private cards for each player in this game.
  uint8_t GetNbHoleCardsRequired() const;
  uint8_t GetNbBoardCardsRequired(uint8_t round) const;
  uint8_t GetNbBoardCardsAtRound(uint8_t round) const;
  uint8_t GetTotalNbBoardCards() const;
  uint8_t NumSuitsDeck() const;
  uint8_t NumRanksDeck() const;
  uint32_t StackSize(uint8_t player) const;
  uint32_t BlindSize(uint8_t player) const;

  uint32_t handId_;
  std::unique_ptr<RawACPCGame> acpc_game_;

  // Checks that the underlying acpc_game_ structs have all their fields equal.
  bool operator==(const ACPCGame& other) const;
};

}  // namespace acpc_cpp
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ACPC_GAME_H
