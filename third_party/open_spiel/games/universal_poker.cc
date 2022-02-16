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

#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {

GameParameters KuhnPokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(1)},
          {"blind", GameParameter(std::string("1 1"))},
          {"raiseSize", GameParameter(std::string("1"))},
          {"firstPlayer", GameParameter(std::string("1"))},
          {"maxRaises", GameParameter(std::string("1"))},
          {"numSuits", GameParameter(1)},
          {"numRanks", GameParameter(3)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0"))}};
}

GameParameters LeducPokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"blind", GameParameter(std::string("1 1"))},
          {"raiseSize", GameParameter(std::string("2 4"))},
          {"firstPlayer", GameParameter(std::string("1 1"))},
          {"maxRaises", GameParameter(std::string("2 2"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(3)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0 1"))}};
}

GameParameters Leduc18PokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"blind", GameParameter(std::string("1 1"))},
          {"raiseSize", GameParameter(std::string("2 4"))},
          {"firstPlayer", GameParameter(std::string("1 1"))},
          {"maxRaises", GameParameter(std::string("2 2"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(9)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0 1"))}};
}

GameParameters NolimitedLeduc5PokerParameters() {
  return {{"betting", GameParameter(std::string("nolimit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"stack", GameParameter(std::string("5 5"))},
          {"blind", GameParameter(std::string("1 1"))},
          {"raiseSize", GameParameter(std::string("2 4"))},
          {"firstPlayer", GameParameter(std::string("1 1"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(3)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0 1"))},
          {"bettingAbstraction", GameParameter(std::string("fullgame"))}};
}

GameParameters NolimitedLeduc10PokerParameters() {
  return {{"betting", GameParameter(std::string("nolimit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"stack", GameParameter(std::string("10 10"))},
          {"blind", GameParameter(std::string("1 1"))},
          {"raiseSize", GameParameter(std::string("2 4"))},
          {"firstPlayer", GameParameter(std::string("1 1"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(3)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0 1"))},
          {"bettingAbstraction", GameParameter(std::string("fullgame"))}};
}

GameParameters FHPPokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"blind", GameParameter(std::string("50 100"))},
          {"raiseSize", GameParameter(std::string("100 100"))},
          {"firstPlayer", GameParameter(std::string("1, 2"))},
          {"maxRaises", GameParameter(std::string("3 3"))},
          {"numSuits", GameParameter(4)},
          {"numRanks", GameParameter(13)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3"))}};
}

GameParameters FHP2PokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"blind", GameParameter(std::string("50 100"))},
          {"raiseSize", GameParameter(std::string("100 100"))},
          {"firstPlayer", GameParameter(std::string("1, 2"))},
          {"maxRaises", GameParameter(std::string("3 3"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(5)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3"))}};
}

GameParameters FHP3PokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"blind", GameParameter(std::string("50 100"))},
          {"raiseSize", GameParameter(std::string("100 100"))},
          {"firstPlayer", GameParameter(std::string("1, 2"))},
          {"maxRaises", GameParameter(std::string("3 3"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(13)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3"))}};
}

GameParameters HULHPokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(4)},
          {"blind", GameParameter(std::string("50 100"))},
          {"raiseSize", GameParameter(std::string("100 100 200 200"))},
          {"firstPlayer", GameParameter(std::string("1 2 2 2"))},
          {"maxRaises", GameParameter(std::string("3 3 4 4"))},
          {"numSuits", GameParameter(4)},
          {"numRanks", GameParameter(13)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3 1 1"))}};
}

GameParameters HULH1PokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(4)},
          {"blind", GameParameter(std::string("50 100"))},
          {"raiseSize", GameParameter(std::string("100 100 200 200"))},
          {"firstPlayer", GameParameter(std::string("1 2 2 2"))},
          {"maxRaises", GameParameter(std::string("3 3 4 4"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(5)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3 1 1"))}};
}

// TODO: review unlimited.
const GameType kGameType{
    /*short_name=*/"universal_poker",
    /*long_name=*/"Universal Poker",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/

    {// The ACPC code uses a specific configuration file to describe the game.
     // The following has been copied from ACPC documentation:
     //
     // Empty lines or lines with '#' as the very first character will be
     // ignored
     //
     // The Game definitions should start with "gamedef" and end with
     // "end gamedef" and can have the fields documented bellow (case is
     // ignored)
     //
     // If you are creating your own game definitions, please note that game.h
     // defines some constants for maximums in games (e.g., number of rounds).
     // These may need to be changed for games outside of the what is being run
     // for the Annual Computer Poker Competition.

     // The ACPC gamedef string.  When present, it will take precedence over
     // everything and no other argument should be provided.
     {"gamedef", GameParameter(std::string(""))},
     // Instead of a single gamedef, specifying each line is also possible.
     // The documentation is adapted from project_acpc_server/game.cc.
     //
     // Number of Players (up to 10)
     {"numPlayers", GameParameter(2)},
     // Betting Type "limit" "nolimit"
     {"betting", GameParameter(std::string("nolimit"))},
     // The stack size for each player at the start of each hand (for
     // no-limit). It will be ignored on "limit".
     // TODO(author2): It's unclear what happens on limit. It defaults to
     // INT32_MAX for all players when not provided.
     {"stack", GameParameter(std::string("1200 1200"))},
     // The size of the blinds for each player (relative to the dealer)
     {"blind", GameParameter(std::string("100 100"))},
     // The size of raises on each round (for limit games only) as numrounds
     // integers. It will be ignored for nolimite games.
     {"raiseSize", GameParameter(std::string("100 100"))},
     // Number of betting rounds per hand of the game
     {"numRounds", GameParameter(2)},
     // The player that acts first (relative to the dealer) on each round
     {"firstPlayer", GameParameter(std::string("1 1"))},
     // maxraises - the maximum number of raises on each round. If not
     // specified, it will default to UINT8_MAX.
     {"maxRaises", GameParameter(std::string(""))},
     // The number of different suits in the deck
     {"numSuits", GameParameter(4)},
     // The number of different ranks in the deck
     {"numRanks", GameParameter(6)},
     // The number of private cards to  deal to each player
     {"numHoleCards", GameParameter(1)},
     // The number of cards revealed on each round
     {"numBoardCards", GameParameter(std::string("0 1"))},
     // Specify which actions are available to the player, in both limit and
     // nolimit games. Available options are: "fc" for fold and check/call.
     // "fcpa" for fold, check/call, bet pot and all in (default).
     // Use "fullgame" for the unabstracted game.
     {"bettingAbstraction", GameParameter(std::string("fcpa"))}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new UniversalPokerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Returns how many actions are available at a choice node.
// limit: fold, check/call, bet/raise.
// unlimite: fold, check/call, bet pot, all in.
inline uint32_t GetMaxBettingActions(const acpc_cpp::ACPCGame &acpc_game) {
  return acpc_game.IsLimitGame() ? 3 : 4;
}

// namespace universal_poker
UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game,
                                         int big_blind,
                                         int starting_stack_big_blinds)
    : State(game),
      big_blind_(big_blind),
      starting_stack_big_blinds_(starting_stack_big_blinds),
      acpc_game_(
          static_cast<const UniversalPokerGame *>(game.get())->GetACPCGame()),
      acpc_state_(acpc_game_),
      num_board_cards_(0),
      betting_abstraction_(static_cast<const UniversalPokerGame *>(game.get())
                               ->betting_abstraction()) {
  actionSequence_.resize(acpc_game_->NumRounds());
  bets_.resize(acpc_game_->NumRounds());
  _CalculateActionsAndNodeType();
}

std::string UniversalPokerState::ActionToString(Player player,
                                                Action move) const {
  if (player == kChancePlayerId) return absl::StrCat("Chance outcome:", move);
  if (move == kFold)
    return "f";
  else if (move == kCall)
    return "c";
  if (betting_abstraction_ != BettingAbstraction::kFULLGAME ||
      acpc_game_->IsLimitGame()) {
    if (move == kRaise)
      return "r";
    else if (move == kAllIn)
      return "a";
  } else if ((int)move >= 2 && move < NumDistinctActions()) {
    return "r" + std::to_string(((int)move - 1) * big_blind_);
  }
  SpielFatalError(
      absl::StrCat("Error in LeducState::ActionToString(). Available actions "
                   "are 0, 1, 2, not ",
                   move));
}

std::string UniversalPokerState::SequenceToString() const {
  std::vector<std::string> seq_strs;
  for (auto &round_s : actionSequence_) {
    std::string seq_str;
    for (auto &a : round_s) {
      seq_str += ActionToString(0, a);
    }
    if (seq_str.size()) {
      seq_strs.push_back(seq_str);
    }
  }
  return absl::StrJoin(seq_strs, ",");
}

std::string UniversalPokerState::ToString() const { return SequenceToString(); }

bool UniversalPokerState::IsTerminal() const {
  bool finished = cur_player_ == kTerminalPlayerId;
  assert(acpc_state_.IsFinished() || !finished);
  return finished;
}

bool UniversalPokerState::IsChanceNode() const {
  return cur_player_ == kChancePlayerId;
}

ChanceData UniversalPokerState::SampleChance(std::mt19937 *rng) {
  ChanceData chance_data = ChanceData{acpc_state_.DealCards(rng)};
  SetChance(chance_data);
  return chance_data;
}

void UniversalPokerState::SetChance(const ChanceData &chance_data) {
  acpc_state_.SetCards(chance_data.data.data(), chance_data.data.size());
}

void UniversalPokerState::SetHoleCards(Player player,
                                       const logic::CardSet &cards) {
  std::vector<uint8_t> card_array = cards.ToCardArray();
  acpc_state_.SetHoleCards(player, card_array.data(), card_array.size());
}

void UniversalPokerState::SetBoardCards(const logic::CardSet &cards) {
  std::vector<uint8_t> card_array = cards.ToCardArray();
  acpc_state_.SetBoardCards(card_array.data(), card_array.size());
}

void UniversalPokerState::SetHoleCards(Player player,
                                       const std::vector<uint8_t> &cards) {
  acpc_state_.SetHoleCards(player, cards.data(), cards.size());
}

void UniversalPokerState::SetBoardCards(const std::vector<uint8_t> &cards) {
  acpc_state_.SetBoardCards(cards.data(), cards.size());
}

Player UniversalPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (IsChanceNode()) {
    return kChancePlayerId;
  }

  return Player(acpc_state_.CurrentPlayer());
}

std::vector<double> UniversalPokerState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(NumPlayers(), 0.0);
  }

  std::vector<double> returns(NumPlayers());
  for (Player player = 0; player < NumPlayers(); ++player) {
    // Money vs money at start.
    returns[player] = acpc_state_.ValueOfState(player);
  }

  return returns;
}

void UniversalPokerState::InformationStateTensor(
    Player player, std::vector<double> *values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::vector<int> tensor_shape = game_->InformationStateTensorShape();
  int tensor_size = 0;
  for (auto &ts : tensor_shape) {
    tensor_size += ts;
  }
  values->resize(tensor_size);
  // Because cards and actions are zero-based,
  // Undisclosed cards and non-existent actions are marked as kInvalidAction(-1)
  std::fill(values->begin(), values->end(), kInvalidAction);

  // Layout of observation:
  //   my cards,
  //   public cards.F
  //   NumRounds() round sequence: (max round seq length)
  int offset = 0;
  std::vector<uint8_t> hole_cards = acpc_state_.HoleCards(player);
  std::vector<uint8_t> board_cards = acpc_state_.BoardCards(num_board_cards_);
  for (uint32_t i = 0; i < hole_cards.size(); i++) {
    (*values)[i + offset] = hole_cards[i];
  }
  offset += acpc_game_->GetNbHoleCardsRequired();

  // Public cards
  for (uint32_t i = 0; i < board_cards.size(); i++) {
    (*values)[i + offset] = board_cards[i];
  }
  offset += acpc_game_->GetTotalNbBoardCards();

  if (acpc_game_->IsLimitGame()) {
    for (uint8_t r = 0; r < acpc_game_->NumRounds(); ++r) {
      const std::vector<float> &bets = bets_[r];
      // std::cout << "bets " << bets_ << std::endl;
      // std::cout << "Actions: " << actionSequence_ << std::endl;
      uint32_t round_length =
          std::dynamic_pointer_cast<const UniversalPokerGame>(game_)
              ->MaxRoundLength(r);
      SPIEL_CHECK_LE(bets.size(), round_length);
      absl::c_copy(bets, (*values).begin() + offset);
      // Move offset up to the next round: 2 bits per move.
      offset += round_length;
    }
  } else {
    for (uint8_t r = 0; r < acpc_game_->NumRounds(); ++r) {
      const std::vector<int> actionSeq = GetActionSequence(r);
      absl::c_copy(actionSeq, (*values).begin() + offset);
      // Move offset up to the next round: 2 bits per move.
      offset += actionSeq.size();
    }
    offset = values->size();
  }
  SPIEL_CHECK_EQ(offset, values->size());
}

void UniversalPokerState::ObservationTensor(Player player,
                                            std::vector<double> *values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());

  values->resize(game_->ObservationTensorShape()[0]);
  // Because cards and actions are zero-based,
  // Undisclosed cards and non-existent actions are marked as kInvalidAction(-1)
  std::fill(values->begin(), values->end(), kInvalidAction);

  // Layout of observation:
  //   my cards,
  //   public cards.
  //   NumRounds() round sequence: (max round seq length)
  int offset = 0;
  std::vector<uint8_t> hole_cards = acpc_state_.HoleCards(player);
  std::vector<uint8_t> board_cards = acpc_state_.BoardCards(num_board_cards_);
  for (uint32_t i = 0; i < hole_cards.size(); i++) {
    (*values)[i + offset] = hole_cards[i];
  }
  offset += acpc_game_->GetNbHoleCardsRequired();

  // Public cards
  for (uint32_t i = 0; i < board_cards.size(); i++) {
    (*values)[i + offset] = board_cards[i];
  }
  offset += acpc_game_->GetTotalNbBoardCards();

  // Adding the contribution of each players to the pot.
  for (auto p = Player{0}; p < NumPlayers(); p++) {
    (*values)[offset + p] = acpc_state_.Ante(p);
  }
  offset += NumPlayers();

  SPIEL_CHECK_EQ(offset, values->size());
}

std::string UniversalPokerState::InformationStateString(Player player) const {
  // SPIEL_CHECK_GE(player, 0);
  // SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());

  return absl::StrFormat("%s:%s:%s", acpc_state_.HoleCardsToString(player),
                         acpc_state_.BoardCardsToString(num_board_cards_),
                         SequenceToString());
}

std::string UniversalPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  std::string result = absl::StrFormat(
      "%s:%s:",
      player != kChancePlayerId ? acpc_state_.HoleCardsToString(player) : "",
      acpc_state_.BoardCardsToString(num_board_cards_));
  // Adding the contribution of each players to the pot
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, ",", acpc_state_.Ante(p));
  }
  return result;
}

std::unique_ptr<State> UniversalPokerState::Clone() const {
  return std::unique_ptr<State>(new UniversalPokerState(*this));
}

std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes()
    const {
  return std::vector<std::pair<Action, double>>{{0, 1}};
}

std::vector<Action> UniversalPokerState::LegalActions() const {
  if (IsChanceNode()) {
    return std::vector<Action>{0};
  }

  std::vector<Action> legal_actions;
  legal_actions.reserve(4);

  if (betting_abstraction_ != BettingAbstraction::kFULLGAME ||
      acpc_game_->IsLimitGame()) {
    if (ACTION_FOLD & possibleActions_) legal_actions.push_back(kFold);
    if (ACTION_CHECK_CALL & possibleActions_) legal_actions.push_back(kCall);
    if (ACTION_RAISE & possibleActions_) legal_actions.push_back(kRaise);
    if (ACTION_ALL_IN & possibleActions_) legal_actions.push_back(kAllIn);
    return legal_actions;
  } else {
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0)) {
      legal_actions.push_back(kFold);
    }
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0)) {
      legal_actions.push_back(kCall);
    }
    int32_t min_bet_size = 0;
    int32_t max_bet_size = 0;
    bool valid_to_raise =
        acpc_state_.RaiseIsValid(&min_bet_size, &max_bet_size);
    if (valid_to_raise) {
      assert(min_bet_size % big_blind_ == 0);
      for (int i = min_bet_size; i <= max_bet_size; i += big_blind_) {
        legal_actions.push_back(1 + i / big_blind_);
      }
    }
  }
  return legal_actions;
}

// We first deal the cards to each player, dealing all the cards to the first
// player first, then the second player, until all players have their private
// cards.
void UniversalPokerState::DoApplyAction(Action action_id) {
  if (IsChanceNode()) {
    // Assume we have dealt the board cards.
    // The card will be dealt by the user, by calling DealCards with a seed.
    num_board_cards_ =
        acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound());
    _CalculateActionsAndNodeType();
  } else {
    int action_int = static_cast<int>(action_id);
    int round = acpc_state_.GetRound();
    int current_player = acpc_state_.CurrentPlayer();
    actionSequence_[round].push_back(action_int);
    uint32_t old_spent = acpc_state_.Ante(current_player);
    if (action_int == kFold) {
      ApplyChoiceAction(ACTION_FOLD, 0);
    } else if (action_int == kCall) {
      ApplyChoiceAction(ACTION_CHECK_CALL, 0);
    } else if (betting_abstraction_ != BettingAbstraction::kFULLGAME ||
               acpc_game_->IsLimitGame()) {
      // Limited game or abstracted game. Set betting value in
      // ApplyChoiceAction.
      if (action_int == kRaise) {
        ApplyChoiceAction(ACTION_RAISE, 0);
      } else if (action_int == kAllIn) {
        ApplyChoiceAction(ACTION_ALL_IN, 0);
      }
    } else if (action_int >= 2 && action_int <= NumDistinctActions()) {
      ApplyChoiceAction(ACTION_RAISE, (action_int - 1) * big_blind_);
    } else {
      SpielFatalError(absl::StrFormat("Action not recognized: %i", action_id));
    }
    uint32_t new_spent = acpc_state_.Ante(current_player);
    bets_[round].push_back((new_spent - old_spent) / (float)big_blind_);
  }
}

double UniversalPokerState::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.

  const int num_players = acpc_game_->GetNbPlayers();
  const int rounds = acpc_game_->NumRounds();
  const int player = acpc_state_.CurrentPlayer();
  if (acpc_game_->IsLimitGame()) {
    double max_utility = acpc_state_.TotalSpent();
    max_utility -= acpc_state_.CurrentSpent(player);
    max_utility += (num_players - 1) *
                   ((int)(acpc_game_->GetMaxRaises(acpc_state_.GetRound())) -
                    acpc_state_.NumRaises()) *
                   acpc_game_->GetRaiseSize(acpc_state_.GetRound());
    for (int i = acpc_state_.GetRound() + 1; i < rounds; ++i) {
      max_utility += (num_players - 1) * acpc_game_->GetRaiseSize(i) *
                     acpc_game_->GetMaxRaises(i);
    }
    return max_utility;
  }
  return (double)acpc_game_->StackSize(0) * (num_players - 1);
}

double UniversalPokerState::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  const int num_players = acpc_game_->GetNbPlayers();
  const int rounds = acpc_game_->NumRounds();
  const int player = acpc_state_.CurrentPlayer();
  if (acpc_game_->IsLimitGame()) {
    double min_utility = acpc_state_.CurrentSpent(player);
    min_utility -= acpc_state_.TotalSpent();
    min_utility -= ((int)(acpc_game_->GetMaxRaises(acpc_state_.GetRound())) -
                    acpc_state_.NumRaises()) *
                   acpc_game_->GetRaiseSize(acpc_state_.GetRound());
    for (int i = acpc_state_.GetRound() + 1; i < rounds; ++i) {
      min_utility -= acpc_game_->GetRaiseSize(i) * acpc_game_->GetMaxRaises(i);
    }
    return min_utility;
  }
  return -1. * (double)acpc_game_->StackSize(0);
}

/**
 * Universal Poker Game Constructor
 * @param params
 */
UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
    : Game(kGameType, params),
      gameDesc_(parseParameters(params)),
      acpc_game_(gameDesc_) {
  max_game_length_ = MaxGameLength();
  big_blind_ = 0;
  for (int i = 0; i < NumPlayers(); ++i) {
    if (acpc_game_.BlindSize(i) > big_blind_) {
      big_blind_ = acpc_game_.BlindSize(i);
    }
  }
  little_blind_ = std::numeric_limits<uint32_t>::max();
  for (int i = 0; i < NumPlayers(); ++i) {
    if (acpc_game_.BlindSize(i) < little_blind_) {
      little_blind_ = acpc_game_.BlindSize(i);
    }
  }
  // assumes all stack sizes are equal
  starting_stack_big_blinds_ = acpc_game_.StackSize(0) / big_blind_;
  SPIEL_CHECK_TRUE(max_game_length_.has_value());
  std::string betting_abstraction =
      ParameterValue<std::string>("bettingAbstraction");
  if (betting_abstraction == "fc") {
    betting_abstraction_ = BettingAbstraction::kFC;
  } else if (betting_abstraction == "fcpa") {
    betting_abstraction_ = BettingAbstraction::kFCPA;
  } else if (betting_abstraction == "fullgame") {
    betting_abstraction_ = BettingAbstraction::kFULLGAME;
  } else {
    SpielFatalError(absl::StrFormat("bettingAbstraction: %s not supported.",
                                    betting_abstraction));
  }
}

std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
  return std::unique_ptr<State>(new UniversalPokerState(
      shared_from_this(), big_blind_, starting_stack_big_blinds_));
}

std::vector<int> UniversalPokerGame::InformationStateTensorShape() const {
  return {acpc_game_.GetNbHoleCardsRequired(),
          acpc_game_.GetTotalNbBoardCards(), MaxGameLength()};
}

int UniversalPokerGame::InformationStateTensorSize() const {
  return acpc_game_.GetNbHoleCardsRequired() +
         acpc_game_.GetTotalNbBoardCards() + MaxGameLength();
}

std::vector<int> UniversalPokerGame::ObservationTensorShape() const {
  return {acpc_game_.GetNbHoleCardsRequired() +
          acpc_game_.GetTotalNbBoardCards() + acpc_game_.GetNbPlayers()};
}

double UniversalPokerGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.

  const int num_players = acpc_game_.GetNbPlayers();
  const int rounds = acpc_game_.NumRounds();
  if (acpc_game_.IsLimitGame()) {
    double max_utility = 0;
    for (int i = 0; i < rounds; ++i) {
      max_utility += (num_players - 1) * acpc_game_.GetRaiseSize(i) *
                     acpc_game_.GetMaxRaises(i);
    }
    for (int i = 0; i < num_players; ++i) {
      max_utility += acpc_game_.BlindSize(i);
    }
    max_utility -= little_blind_;
    // NOTE: Make sure the max utility is smaller than (p - 1 ) * stack size.
    SPIEL_CHECK_LE(max_utility,
                   (double)acpc_game_.StackSize(0) * (num_players - 1));
    return max_utility;
  }
  return (double)acpc_game_.StackSize(0) * (num_players - 1);
}

double UniversalPokerGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  const int num_players = acpc_game_.GetNbPlayers();
  const int rounds = acpc_game_.NumRounds();
  if (acpc_game_.IsLimitGame()) {
    double min_utility = 0;
    for (int i = 0; i < rounds; ++i) {
      min_utility += -1. * (double)(acpc_game_.GetRaiseSize(i) *
                                    acpc_game_.GetMaxRaises(i));
    }
    min_utility -= big_blind_;
    // NOTE: Make sure the - min utility is smaller than stack size. So the
    // player will never takes the action of all-in.
    SPIEL_CHECK_LE(-min_utility, (double)acpc_game_.StackSize(0));
    return min_utility;
  }
  return -1. * (double)acpc_game_.StackSize(0);
}

int UniversalPokerGame::MaxChanceOutcomes() const { return 1; }

int UniversalPokerGame::NumPlayers() const { return acpc_game_.GetNbPlayers(); }

int UniversalPokerGame::NumDistinctActions() const {
  if (betting_abstraction_ == BettingAbstraction::kFULLGAME &&
      !acpc_game_.IsLimitGame()) {
    // fold, check/call, bet/raise some multiple of BBs
    return starting_stack_big_blinds_ + 2;
  } else {
    return GetMaxBettingActions(acpc_game_);
  }
}

std::shared_ptr<const Game> UniversalPokerGame::Clone() const {
  return std::shared_ptr<const Game>(new UniversalPokerGame(*this));
}

int UniversalPokerGame::MaxRoundLength(uint8_t round) const {
  int length = 0;
  if (acpc_game_.IsLimitGame()) {
    int num_player = NumPlayers();
    // for 2 player limited game, the max-length action sequence in a round is
    // [check].[raise*max_raise].[call/fold].
    // So the max length is max_raise + 2.
    // for multiplayer, the max-length in a round is
    // (num_players_-1) checks +
    // max_raises[i] raises + max_raises[i]*(num_players_-2) calls +
    // 1 calls.
    length += acpc_game_.GetMaxRaises(round) * (1 + (num_player - 2)) +
              (num_player - 1) + 1;
  }
  return length;
}

int UniversalPokerGame::MaxGameLength() const {
  // We cache this as this is very slow to calculate.
  if (max_game_length_) return *max_game_length_;
  int length = 0;
  if (acpc_game_.IsLimitGame()) {
    std::vector<uint8_t> max_raises = acpc_game_.GetMaxRaises();
    int num_player = NumPlayers();
    for (int i = 0; i != acpc_game_.NumRounds(); ++i) {
      length += max_raises[i] * (1 + (num_player - 2)) + (num_player - 1) + 1;
    }
    return length;
  } else {
    // Make a good guess here because bruteforcing the tree is far too slow
    // One Terminal Action

    // Deal Actions
    length += acpc_game_.GetTotalNbBoardCards() +
              acpc_game_.GetNbHoleCardsRequired() * acpc_game_.GetNbPlayers();

    // Check Actions
    length += (NumPlayers() * acpc_game_.NumRounds());

    // Bet Actions
    double maxStack = 0;
    double maxBlind = 0;
    for (uint32_t p = 0; p < NumPlayers(); p++) {
      maxStack = acpc_game_.StackSize(p) > maxStack ? acpc_game_.StackSize(p)
                                                    : maxStack;
      maxBlind = acpc_game_.BlindSize(p) > maxStack ? acpc_game_.BlindSize(p)
                                                    : maxBlind;
    }

    while (maxStack > maxBlind) {
      maxStack /= 2.0;         // You have always to bet the pot size
      length += NumPlayers();  // Each player has to react
    }
    return length;
  }
}

/**
 * Parses the Game Paramters and makes a gameDesc out of it
 * @param map
 * @return
 */
std::string UniversalPokerGame::parseParameters(const GameParameters &map) {
  if (map.find("gamedef") != map.end()) {
    // We check for sanity that all parameters are empty
    if (map.size() != 1) {
      std::vector<std::string> game_parameter_keys;
      game_parameter_keys.reserve(map.size());
      for (auto const &imap : map) {
        game_parameter_keys.push_back(imap.first);
      }
      SpielFatalError(
          absl::StrCat("When loading a 'universal_poker' game, the 'gamedef' "
                       "field was present, but other fields were present too: ",
                       absl::StrJoin(game_parameter_keys, ", "),
                       "gamedef is exclusive with other paraemters."));
    }
    return ParameterValue<std::string>("gamedef");
  }

  std::string generated_gamedef = "GAMEDEF\n";

  absl::StrAppend(
      &generated_gamedef, ParameterValue<std::string>("betting"), "\n",
      "numPlayers = ", ParameterValue<int>("numPlayers"), "\n",
      "numRounds = ", ParameterValue<int>("numRounds"), "\n",
      "numsuits = ", ParameterValue<int>("numSuits"), "\n",
      "firstPlayer = ", ParameterValue<std::string>("firstPlayer"), "\n",
      "numRanks = ", ParameterValue<int>("numRanks"), "\n",
      "numHoleCards = ", ParameterValue<int>("numHoleCards"), "\n",
      "numBoardCards = ", ParameterValue<std::string>("numBoardCards"), "\n");

  std::string max_raises = ParameterValue<std::string>("maxRaises");
  if (!max_raises.empty()) {
    absl::StrAppend(&generated_gamedef, "maxRaises = ", max_raises, "\n");
  }

  if (ParameterValue<std::string>("betting") == "limit") {
    std::string raise_size = ParameterValue<std::string>("raiseSize");
    if (!raise_size.empty()) {
      absl::StrAppend(&generated_gamedef, "raiseSize = ", raise_size, "\n");
    }
  } else if (ParameterValue<std::string>("betting") == "nolimit") {
    std::string stack = ParameterValue<std::string>("stack");
    if (!stack.empty()) {
      absl::StrAppend(&generated_gamedef, "stack = ", stack, "\n");
    }
  } else {
    SpielFatalError(absl::StrCat("betting should be limit or nolimit, not ",
                                 ParameterValue<std::string>("betting")));
  }

  absl::StrAppend(&generated_gamedef,
                  "blind = ", ParameterValue<std::string>("blind"), "\n");
  absl::StrAppend(&generated_gamedef, "END GAMEDEF\n");

  return generated_gamedef;
}

void UniversalPokerState::ApplyChoiceAction(ActionType action_type, int size) {
  SPIEL_CHECK_GE(cur_player_, 0);  // No chance not terminal.
  switch (action_type) {
    case ACTION_FOLD:
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0);
      break;
    case ACTION_CHECK_CALL:
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0);
      break;
    case ACTION_RAISE:
      if (acpc_game_->IsLimitGame()) {
        acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE,
                             acpc_state_.GetRaiseSize());
      } else if (betting_abstraction_ == BettingAbstraction::kFULLGAME) {
        acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE,
                             size);
      } else if (betting_abstraction_ == BettingAbstraction::kFCPA) {
        acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE,
                             potSize_);
      }
      break;
    case ACTION_ALL_IN:
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE,
                           allInSize_);
      break;
    case ACTION_DEAL:
    default:
      assert(false);
      break;
  }

  _CalculateActionsAndNodeType();
}

void UniversalPokerState::_CalculateActionsAndNodeType() {
  possibleActions_ = 0;

  if (acpc_state_.IsFinished()) {
    if (acpc_state_.NumFolded() >= acpc_game_->GetNbPlayers() - 1) {
      // All players except one has fold.
      cur_player_ = kTerminalPlayerId;
    } else {
      if (num_board_cards_ <
          acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
        cur_player_ = kChancePlayerId;
        possibleActions_ = ACTION_DEAL;
        return;
      }
      // Showdown!
      cur_player_ = kTerminalPlayerId;
    }
  } else {
    // We need to deal a public card.
    if (num_board_cards_ <
        acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
      cur_player_ = kChancePlayerId;
      possibleActions_ = ACTION_DEAL;
      return;
    }

    // Check for CHOICE Actions
    cur_player_ = acpc_state_.CurrentPlayer();
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0)) {
      possibleActions_ |= ACTION_FOLD;
    }
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0)) {
      possibleActions_ |= ACTION_CHECK_CALL;
    }

    if (betting_abstraction_ == BettingAbstraction::kFC) return;

    potSize_ = 0;
    allInSize_ = 0;
    // Only nolimit game need potSize_ and allInSize.
    if (acpc_state_.RaiseIsValid(&potSize_, &allInSize_)) {
      if (acpc_game_->IsLimitGame()) {
        // There's only one "bet" allowed in Limit, which is "all-in or fixed
        // bet".
        possibleActions_ |= ACTION_RAISE;

      } else {
        int cur_spent = acpc_state_.CurrentSpent(acpc_state_.CurrentPlayer());
        int pot_raise_to =
            acpc_state_.TotalSpent() + 2 * acpc_state_.MaxSpend() - cur_spent;

        if (pot_raise_to >= potSize_ && pot_raise_to <= allInSize_) {
          potSize_ = pot_raise_to;
          possibleActions_ |= ACTION_RAISE;
        }

        if (pot_raise_to != allInSize_) {
          // If the raise to amount happens to match the number of chips I
          // have,
          // then this action was already added as a pot-bet.
          possibleActions_ |= ACTION_ALL_IN;
        }
      }
    }
  }
}

const int UniversalPokerState::GetPossibleActionCount() const {
  // _builtin_popcount(int) function is used to count the number of one's
  return __builtin_popcount(possibleActions_);
}

std::ostream &operator<<(std::ostream &os, const BettingAbstraction &betting) {
  switch (betting) {
    case BettingAbstraction::kFC: {
      os << "BettingAbstration: FC";
      break;
    }
    case BettingAbstraction::kFCPA: {
      os << "BettingAbstration: FCPA";
      break;
    }
    case BettingAbstraction::kFULLGAME: {
      os << "BettingAbstraction: FULLGAME";
      break;
    }
    default:
      SpielFatalError("Unknown betting abstraction.");
      break;
  }
  return os;
}

}  // namespace universal_poker
}  // namespace open_spiel
