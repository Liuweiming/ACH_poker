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

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <numeric>
#include <ostream>
#include <sstream>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace acpc_cpp {

static const int STRING_BUFFERSIZE = 4096;

namespace {
RawACPCAction GetAction(ACPCState::ACPCActionType type, int32_t size) {
  RawACPCAction acpc_action;

  acpc_action.size = size;
  switch (type) {
    case ACPCState::ACPC_CALL:
      acpc_action.type = project_acpc_server::a_call;
      break;
    case ACPCState::ACPC_FOLD:
      acpc_action.type = project_acpc_server::a_fold;
      break;
    case ACPCState::ACPC_RAISE:
      acpc_action.type = project_acpc_server::a_raise;
      break;
    default:
      acpc_action.type = project_acpc_server::a_invalid;
      break;
  }
  return acpc_action;
}

void readGameToStruct(const std::string &gameDef, RawACPCGame *acpc_game) {
  char buf[STRING_BUFFERSIZE];
  gameDef.copy(buf, STRING_BUFFERSIZE);

  FILE *f = fmemopen(&buf, STRING_BUFFERSIZE, "r");
  project_acpc_server::Game *game = project_acpc_server::readGame(f);

  memcpy(acpc_game, game, sizeof(RawACPCGame));

  free(game);
  fclose(f);
}

}  // namespace

ACPCGame::ACPCGame(const std::string &gameDef)
    // check this make unique.
    : handId_(0), acpc_game_(std::unique_ptr<RawACPCGame>(new RawACPCGame())) {
  readGameToStruct(gameDef, acpc_game_.get());
}

ACPCGame::ACPCGame(const ACPCGame &other)
    : handId_(other.handId_),
      acpc_game_(
          std::unique_ptr<RawACPCGame>(new RawACPCGame(*other.acpc_game_))) {}

// We compare the values for all the fields. For arrays, note that only the
// first `numPlayers` or `numRounds` values are meaningful, the rest being
// non-initialized.
bool ACPCGame::operator==(const ACPCGame &other) const {
  // See project_acpc_server/game.h:42. 12 fields total.
  // int32_t stack[ MAX_PLAYERS ];
  // int32_t blind[ MAX_PLAYERS ];
  // int32_t raiseSize[ MAX_ROUNDS ];
  // enum BettingType bettingType;
  // uint8_t numPlayers;
  // uint8_t numRounds;
  // uint8_t firstPlayer[ MAX_ROUNDS ];
  // uint8_t maxRaises[ MAX_ROUNDS ];
  // uint8_t numSuits;
  // uint8_t numRanks;
  // uint8_t numHoleCards;
  // uint8_t numBoardCards[ MAX_ROUNDS ];
  const RawACPCGame *first = acpc_game_.get();
  const RawACPCGame *second = other.acpc_game_.get();
  const int num_players = first->numPlayers;
  const int num_rounds = first->numRounds;
  return (  // new line
      first->numPlayers == second->numPlayers &&
      first->numRounds == second->numRounds &&
      std::equal(first->stack, first->stack + num_players, second->stack) &&
      std::equal(first->blind, first->blind + num_players, second->blind) &&
      (first->bettingType == ::project_acpc_server::BettingType::limitBetting
           ? std::equal(first->raiseSize, first->raiseSize + num_rounds,
                        second->raiseSize)
           : true) &&
      first->bettingType == second->bettingType &&
      std::equal(first->firstPlayer, first->firstPlayer + num_rounds,
                 second->firstPlayer) &&
      std::equal(first->maxRaises, first->maxRaises + num_rounds,
                 second->maxRaises) &&
      first->numSuits == second->numSuits &&
      first->numRanks == second->numRanks &&
      first->numHoleCards == second->numHoleCards &&
      std::equal(first->numBoardCards, first->numBoardCards + num_rounds,
                 second->numBoardCards));
}

std::string ACPCGame::ToString() const {
  char buf[STRING_BUFFERSIZE];
  FILE *f = fmemopen(&buf, STRING_BUFFERSIZE, "w");
  project_acpc_server::printGame(f, acpc_game_.get());
  std::ostringstream result;
  rewind(f);
  result << buf;
  fclose(f);

  return result.str();
}

bool ACPCGame::IsLimitGame() const {
  return acpc_game_->bettingType == project_acpc_server::limitBetting;
}

int ACPCGame::GetNbPlayers() const { return acpc_game_->numPlayers; }

std::vector<uint32_t> ACPCGame::GetRaiseSize() const {
  return std::vector<uint32_t>(acpc_game_->raiseSize,
                               acpc_game_->raiseSize + acpc_game_->numRounds);
}

std::vector<uint8_t> ACPCGame::GetMaxRaises() const {
  return std::vector<uint8_t>(acpc_game_->maxRaises,
                              acpc_game_->maxRaises + acpc_game_->numRounds);
}

uint32_t ACPCGame::GetRaiseSize(uint8_t round) const {
  return acpc_game_->raiseSize[round];
}

uint8_t ACPCGame::GetMaxRaises(uint8_t round) const {
  return acpc_game_->maxRaises[round];
}
uint8_t ACPCGame::GetNbHoleCardsRequired() const {
  return acpc_game_->numHoleCards;
}

uint8_t ACPCGame::GetNbBoardCardsRequired(uint8_t round) const {
  SPIEL_CHECK_LT(round, acpc_game_->numRounds);

  uint8_t nbCards = 0;
  for (int r = 0; r <= round; ++r) {
    nbCards += acpc_game_->numBoardCards[r];
  }
  return nbCards;
}

uint8_t ACPCGame::GetNbBoardCardsAtRound(uint8_t round) const {
  SPIEL_CHECK_LT(round, acpc_game_->numRounds);
  return acpc_game_->numBoardCards[round];
}

uint8_t ACPCGame::GetTotalNbBoardCards() const {
  uint8_t nbCards = 0;
  for (uint8_t r = 0; r < acpc_game_->numRounds; ++r) {
    nbCards += acpc_game_->numBoardCards[r];
  }
  return nbCards;
}

uint8_t ACPCGame::NumSuitsDeck() const { return acpc_game_->numSuits; }

uint8_t ACPCGame::NumRanksDeck() const { return acpc_game_->numRanks; }

uint32_t ACPCGame::StackSize(uint8_t player) const {
  SPIEL_CHECK_LE(0, player);
  SPIEL_CHECK_LT(player, acpc_game_->numPlayers);
  return acpc_game_->stack[player];
}

int ACPCGame::NumRounds() const { return acpc_game_->numRounds; }

uint32_t ACPCGame::BlindSize(uint8_t player) const {
  SPIEL_CHECK_LE(0, player);
  SPIEL_CHECK_LT(player, acpc_game_->numPlayers);
  return acpc_game_->blind[player];
}

std::string ACPCState::ToString() const {
  char buf[STRING_BUFFERSIZE];
  project_acpc_server::printState(game_->acpc_game_.get(), acpc_state_.get(),
                                  STRING_BUFFERSIZE, buf);
  std::ostringstream out;

  out << buf << std::endl << "Spent: [";
  for (int p = 0; p < game_->acpc_game_->numPlayers; ++p) {
    out << "P" << p << ": " << acpc_state_->spent[p] << "  ";
  }
  out << "]" << std::endl;

  return out.str();
}

int ACPCState::RaiseIsValid(int32_t *minSize, int32_t *maxSize) const {
  return raiseIsValid(game_->acpc_game_.get(), acpc_state_.get(), minSize,
                      maxSize);
}

double ACPCState::ValueOfState(const uint8_t player) const {
  assert(stateFinished(acpc_state_.get()));
  return project_acpc_server::valueOfState(game_->acpc_game_.get(),
                                           acpc_state_.get(), player);
}

bool ACPCState::IsFinished() const { return stateFinished(acpc_state_.get()); }

uint32_t ACPCState::MaxSpend() const { return acpc_state_->maxSpent; }

int ACPCState::GetRound() const { return acpc_state_->round; }

int ACPCState::GetRaiseSize() const {
  return game_->acpc_game_->raiseSize[acpc_state_->round];
}

int ACPCState::NumRaises() const {
  return project_acpc_server::numRaises(acpc_state_.get());
}

uint8_t ACPCState::SumBoardCards() const {
  return project_acpc_server::sumBoardCards(game_->acpc_game_.get(),
                                            acpc_state_.get()->round);
}

uint8_t ACPCState::NumFolded() const {
  return project_acpc_server::numFolded(game_->acpc_game_.get(),
                                        acpc_state_.get());
}

uint8_t ACPCState::CurrentPlayer() const {
  return project_acpc_server::currentPlayer(game_->acpc_game_.get(),
                                            acpc_state_.get());
}

ACPCState::ACPCState(const ACPCGame *game)
    : game_(game),
      acpc_state_(std::unique_ptr<RawACPCState>(new RawACPCState())) {
  project_acpc_server::initState(
      game_->acpc_game_.get(),
      game_->handId_ /*TODO this make a unit test fail++*/, acpc_state_.get());
}

ACPCState::ACPCState(const ACPCState &other)
    : game_(other.game_),
      acpc_state_(
          std::unique_ptr<RawACPCState>(new RawACPCState(*other.acpc_state_))) {
}

ACPCState::ACPCState(const ACPCGame *game, const RawACPCState &raw_state)
    : game_(game),
      acpc_state_(std::unique_ptr<RawACPCState>(new RawACPCState(raw_state))) {}

void ACPCState::DoAction(const ACPCState::ACPCActionType actionType,
                         const int32_t size) {
  RawACPCAction a = GetAction(actionType, size);
  assert(project_acpc_server::isValidAction(game_->acpc_game_.get(),
                                            acpc_state_.get(), false, &a));
  project_acpc_server::doAction(game_->acpc_game_.get(), &a, acpc_state_.get());
}

int ACPCState::IsValidAction(const ACPCState::ACPCActionType actionType,
                             const int32_t size) const {
  RawACPCAction a = GetAction(actionType, size);
  return project_acpc_server::isValidAction(game_->acpc_game_.get(),
                                            acpc_state_.get(), false, &a);
}

uint32_t ACPCState::Money(const uint8_t player) const {
  assert(player < game_->acpc_game_->numPlayers);
  return game_->acpc_game_->stack[player] - acpc_state_->spent[player];
}

uint32_t ACPCState::Ante(const uint8_t player) const {
  assert(player < game_->acpc_game_->numPlayers);
  return acpc_state_->spent[player];
}

uint32_t ACPCState::TotalSpent() const {
  return static_cast<uint32_t>(std::accumulate(
      std::begin(acpc_state_->spent), std::end(acpc_state_->spent), 0));
}

uint32_t ACPCState::CurrentSpent(const uint8_t player) const {
  assert(player < game_->acpc_game_->numPlayers);
  return acpc_state_->spent[player];
}

std::string ACPCState::BettingSequence(uint8_t round) const {
  assert(round < game_->acpc_game_->numRounds);
  std::ostringstream out;
  char buf[10];
  for (int a = 0; a < acpc_state_->numActions[round]; a++) {
    project_acpc_server::Action *action = &acpc_state_->action[round][a];
    project_acpc_server::printAction(game_->acpc_game_.get(), action, 10, buf);
    out << buf;
  }

  return out.str();
}

static uint8_t DealCard(uint8_t *deck, const int numCards, std::mt19937 *rng,
                        std::uniform_int_distribution<int> &rand_int) {
  int i;
  uint8_t ret;
  SPIEL_CHECK_GE(numCards, 1);
  i = rand_int(*rng,
               std::uniform_int_distribution<int>::param_type{0, numCards - 1});
  ret = deck[i];
  deck[i] = deck[numCards - 1];

  return ret;
}

// Adapted form acpc::game
std::vector<uint8_t> ACPCState::DealCards(std::mt19937 *rng) {
  std::uniform_int_distribution<int> rand_int;
  uint8_t deck[MAX_RANKS * MAX_SUITS];
  int num_deck = 0;
  for (int s = MAX_SUITS - game_->acpc_game_->numSuits; s < MAX_SUITS; ++s) {
    for (int r = MAX_RANKS - game_->acpc_game_->numRanks; r < MAX_RANKS; ++r) {
      deck[num_deck] = makeCard(r, s);
      ++num_deck;
    }
  }
  int num_cards = (game_->GetNbPlayers()) * (game_->GetNbHoleCardsRequired()) +
                  game_->GetTotalNbBoardCards();
  std::vector<uint8_t> cards(num_cards, 0);
  int offset = 0;
  for (int p = 0; p < game_->GetNbPlayers(); ++p) {
    for (int i = 0; i < game_->GetNbHoleCardsRequired(); ++i) {
      SPIEL_CHECK_LE(offset + i, num_cards);
      cards[offset + i] = DealCard(deck, num_deck, rng, rand_int);
      --num_deck;
    }
    // Keep the order of the cards.
    std::sort(cards.begin() + offset,
              cards.begin() + offset + game_->GetNbHoleCardsRequired());
    offset += game_->GetNbHoleCardsRequired();
  }
  for (int r = 0; r < game_->NumRounds(); ++r) {
    for (int i = 0; i < game_->GetNbBoardCardsAtRound(r); ++i) {
      SPIEL_CHECK_LE(offset + i, num_cards);
      cards[offset + i] = DealCard(deck, num_deck, rng, rand_int);
      --num_deck;
    }
    std::sort(cards.begin() + offset,
              cards.begin() + offset + game_->GetNbBoardCardsAtRound(r));
    offset += game_->GetNbBoardCardsAtRound(r);
  }
  SetCards(cards.data(), cards.size());
  return cards;
}

void ACPCState::SetCards(const uint8_t *cards, const uint8_t numCards) {
  int offset = 0;
  for (uint8_t p = 0; p < game_->GetNbPlayers(); ++p) {
    SetHoleCards(p, cards + offset, game_->GetNbHoleCardsRequired());
    offset += game_->GetNbHoleCardsRequired();
  }
  SetBoardCards(cards + offset, game_->GetTotalNbBoardCards());
}

std::vector<uint8_t> ACPCState::HoleCards(const uint8_t player) const {
  return std::vector<uint8_t>(
      acpc_state_->holeCards[player],
      acpc_state_->holeCards[player] + game_->GetNbHoleCardsRequired());
}
std::vector<uint8_t> ACPCState::BoardCards(uint8_t num_board_cards) const {
  // Note: need to take care of how many cards are revealed.
  return std::vector<uint8_t>(acpc_state_->boardCards,
                              acpc_state_->boardCards + num_board_cards);
}

std::string ACPCState::HoleCardsToString(const uint8_t player) const {
  if (game_->acpc_game_->numHoleCards) {
    char buffer[MAX_HOLE_CARDS * 3];
    project_acpc_server::printCards(game_->acpc_game_->numHoleCards,
                                    acpc_state_->holeCards[player],
                                    MAX_HOLE_CARDS * 3, buffer);
    std::string ret(buffer);
    return ret;
  }
  return "";
}

std::string ACPCState::BoardCardsToString(uint8_t num_board_cards) const {
  if (num_board_cards) {
    char buffer[MAX_BOARD_CARDS * 3];
    project_acpc_server::printCards(num_board_cards, acpc_state_->boardCards,
                                    MAX_BOARD_CARDS * 3, buffer);
    std::string ret(buffer);
    return ret;
  }
  return "";
}

void ACPCState::SetHoleCards(uint8_t player, const uint8_t holeCards[3],
                             const uint8_t nbHoleCards) const {
  assert(nbHoleCards == game_->GetNbHoleCardsRequired());
  for (int c = 0; c < nbHoleCards; ++c) {
    acpc_state_->holeCards[player][c] = holeCards[c];
  }
}

void ACPCState::SetBoardCards(const uint8_t boardCards[7],
                              uint8_t nbBoardCards) const {
  assert(nbBoardCards >= game_->GetNbBoardCardsRequired(GetRound()) &&
         nbBoardCards <= game_->GetTotalNbBoardCards());
  for (int c = 0; c < nbBoardCards; ++c) {
    acpc_state_->boardCards[c] = boardCards[c];
  }
}

ACPCState::~ACPCState() = default;
ACPCGame::~ACPCGame() = default;
}  // namespace acpc_cpp
}  // namespace universal_poker
}  // namespace open_spiel
