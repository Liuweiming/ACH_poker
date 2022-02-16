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

#include <ctime>
#include <functional>
#include <iostream>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/algorithms/public_tree.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace universal_poker {
namespace {

namespace testing = open_spiel::testing;

constexpr absl::string_view kKuhnLimit3P =
    ("GAMEDEF\n"
     "limit\n"
     "numPlayers = 3\n"
     "numRounds = 1\n"
     "blind = 1 1 1\n"
     "raiseSize = 1\n"
     "firstPlayer = 1\n"
     "maxRaises = 1\n"
     "numSuits = 1\n"
     "numRanks = 4\n"
     "numHoleCards = 1\n"
     "numBoardCards = 0\n"
     "END GAMEDEF\n");
GameParameters KuhnLimit3PParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(3)},
          {"numRounds", GameParameter(1)},
          {"blind", GameParameter(std::string("1 1 1"))},
          {"raiseSize", GameParameter(std::string("1"))},
          {"firstPlayer", GameParameter(std::string("1"))},
          {"maxRaises", GameParameter(std::string("1"))},
          {"numSuits", GameParameter(1)},
          {"numRanks", GameParameter(4)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0"))}};
}

constexpr absl::string_view kHoldemNoLimit6P =
    ("GAMEDEF\n"
     "nolimit\n"
     "numPlayers = 6\n"
     "numRounds = 4\n"
     "stack = 20000 20000 20000 20000 20000 20000\n"
     "blind = 50 100 0 0 0 0\n"
     "firstPlayer = 3 1 1 1\n"
     "numSuits = 4\n"
     "numRanks = 13\n"
     "numHoleCards = 2\n"
     "numBoardCards = 0 3 1 1\n"
     "END GAMEDEF\n");
GameParameters HoldemNoLimit6PParameters() {
  return {{"betting", GameParameter(std::string("nolimit"))},
          {"numPlayers", GameParameter(6)},
          {"numRounds", GameParameter(4)},
          {"stack",
           GameParameter(std::string("20000 20000 20000 20000 20000 20000"))},
          {"blind", GameParameter(std::string("50 100 0 0 0 0"))},
          {"firstPlayer", GameParameter(std::string("3 1 1 1"))},
          {"numSuits", GameParameter(4)},
          {"numRanks", GameParameter(13)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3 1 1"))}};
}

void LoadKuhnLimitWithAndWithoutGameDef() {
  std::cout << std::endl;
  std::cout << "LoadKuhnLimitWithAndWithoutGameDef " << std::endl;
  UniversalPokerGame kuhn_limit_3p_gamedef(
      {{"gamedef", GameParameter(std::string(kKuhnLimit3P))}});
  UniversalPokerGame kuhn_limit_3p(KuhnLimit3PParameters());
  std::cout << kuhn_limit_3p_gamedef.GetACPCGame()->ToString() << std::endl;
  std::cout << kuhn_limit_3p.GetACPCGame()->ToString() << std::endl;

  SPIEL_CHECK_EQ(kuhn_limit_3p_gamedef.GetACPCGame()->ToString(),
                 kuhn_limit_3p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(kuhn_limit_3p_gamedef.GetACPCGame())) ==
                   (*(kuhn_limit_3p.GetACPCGame())));
}

void LoadHoldemNoLimit6PWithAndWithoutGameDef() {
  std::cout << std::endl;
  std::cout << "LoadHoldemNoLimit6PWithAndWithoutGameDef " << std::endl;
  UniversalPokerGame holdem_no_limit_6p_gamedef(
      {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
  UniversalPokerGame holdem_no_limit_6p(HoldemNoLimit6PParameters());
  std::cout << holdem_no_limit_6p_gamedef.GetACPCGame()->ToString()
            << std::endl;
  std::cout << holdem_no_limit_6p.GetACPCGame()->ToString() << std::endl;

  SPIEL_CHECK_EQ(holdem_no_limit_6p_gamedef.GetACPCGame()->ToString(),
                 holdem_no_limit_6p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(holdem_no_limit_6p_gamedef.GetACPCGame())) ==
                   (*(holdem_no_limit_6p.GetACPCGame())));
}
void LoadGameFromDefaultConfig() { LoadGame("universal_poker"); }

void LoadAndRunGamesFullParameters() {
  std::cout << std::endl;
  std::cout << "LoadAndRunGamesFullParameters " << std::endl;
  std::shared_ptr<const Game> kuhn_limit_3p =
      LoadGame("universal_poker", KuhnLimit3PParameters());
  std::shared_ptr<const Game> os_kuhn_3p =
      LoadGame("kuhn_poker", {{"players", GameParameter(3)}});
  SPIEL_CHECK_EQ(kuhn_limit_3p->MaxGameLength(), os_kuhn_3p->MaxGameLength());
  testing::RandomSimTestNoSerialize(*kuhn_limit_3p, 10);
  // TODO(b/145688976): The serialization is also broken
  // In particular, the firstPlayer string "1" is converted back to an integer
  // when deserializing, which crashes.
  // testing::RandomSimTest(*kuhn_limit_3p, 1);
  std::shared_ptr<const Game> holdem_nolimit_6p =
      LoadGame("universal_poker", HoldemNoLimit6PParameters());
  testing::RandomSimTestNoSerialize(*holdem_nolimit_6p, 10);
  // testing::RandomSimTest(*holdem_nolimit_6p, 3);
  std::shared_ptr<const Game> holdem_nolimit_fullgame =
      LoadGame(HunlGameString("fullgame"));
  // testing::RandomSimTest(*holdem_nolimit_fullgame, 50);
  testing::RandomSimTestNoSerialize(*holdem_nolimit_fullgame, 10);
}

void LoadAndRunGameFromGameDef() {
  std::cout << std::endl;
  std::cout << "LoadAndRunGameFromGameDef " << std::endl;
  std::shared_ptr<const Game> holdem_nolimit_6p =
      LoadGame("universal_poker",
               {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
  testing::RandomSimTestNoSerialize(*holdem_nolimit_6p, 10);
  // TODO(b/145688976): The serialization is also broken
  // testing::RandomSimTest(*holdem_nolimit_6p, 1);
}

void HUNLRegressionTests() {
  std::cout << std::endl;
  std::cout << "HUNLRegressionTests " << std::endl;
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 "
      "50,firstPlayer=2 1 1 "
      "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=400 "
      "400)");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::cout << state->ToString() << std::endl;
  // Pot bet: call 50, and raise by 200.
  state->ApplyAction(universal_poker::kRaise);
  std::cout << state->ToString() << std::endl;
  // Now, the minimum bet size is larger than the pot, so player 0 can only
  // fold, call, or go all-in.
  std::vector<Action> actions = state->LegalActions();
  absl::c_sort(actions);

  SPIEL_CHECK_EQ(actions.size(), 3);
  SPIEL_CHECK_EQ(actions[0], universal_poker::kFold);
  SPIEL_CHECK_EQ(actions[1], universal_poker::kCall);
  SPIEL_CHECK_EQ(actions[2], universal_poker::kAllIn);

  // Try a similar test with a stacks of size 300.
  game = LoadGame(
      "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 "
      "50,firstPlayer=2 1 1 "
      "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=300 "
      "300)");
  state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::cout << state->ToString() << std::endl;

  // The pot bet exactly matches the number of chips available. This is an edge
  // case where all-in is not available, only the pot bet.

  actions = state->LegalActions();
  absl::c_sort(actions);

  SPIEL_CHECK_EQ(actions.size(), 3);
  SPIEL_CHECK_EQ(actions[0], universal_poker::kFold);
  SPIEL_CHECK_EQ(actions[1], universal_poker::kCall);
  SPIEL_CHECK_EQ(actions[2], universal_poker::kRaise);
}

void LoadAndRunGameFromDefaultConfig() {
  std::cout << std::endl;
  std::cout << "LoadAndRunGameFromDefaultConfig " << std::endl;
  std::shared_ptr<const Game> game = LoadGame("universal_poker");
  testing::RandomSimTestNoSerialize(*game, 10);
}

constexpr absl::string_view kHULHString =
    ("universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,"
     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
     "1 "
     "1,raiseSize=200 200 400 400,maxRaises=3 4 4 4)");

void BasicUniversalPokerTests() {
  std::cout << std::endl;
  std::cout << "BasicUniversalPokerTests " << std::endl;
  testing::LoadGameTest(std::string(kHULHString));
  testing::ChanceOutcomesTest(*LoadGame(std::string(kHULHString)));
  testing::RandomSimTestNoSerialize(*LoadGame(std::string(kHULHString)), 10);

  // testing::RandomSimBenchmark("leduc_poker", 10000, false);
  // testing::RandomSimBenchmark("universal_poker", 10000, false);

  // testing::CheckChanceOutcomes(*LoadGame(std::string(kHULHString)));
}

void CalculateInformationSetTests() {}

void ChumpPolicyTests() {
  std::cout << std::endl;
  std::cout << "ChumpPolicyTests " << std::endl;
  std::shared_ptr<const Game> game = LoadGame(std::string(kHULHString));
  std::vector<std::unique_ptr<Bot>> bots;
  bots.push_back(MakePolicyBot(*game, /*player_id=*/0, /*seed=*/0,
                               std::unique_ptr<open_spiel::UniformPolicy>(
                                   new open_spiel::UniformPolicy)));
  bots.push_back(MakePolicyBot(
      *game, /*player_id=*/0, /*seed=*/0,
      std::unique_ptr<UniformRestrictedActions>(new UniformRestrictedActions(
          std::vector<ActionType>({ActionType::kCall})))));
  bots.push_back(MakePolicyBot(
      *game, /*player_id=*/0, /*seed=*/0,
      std::unique_ptr<UniformRestrictedActions>(new UniformRestrictedActions(
          std::vector<ActionType>({ActionType::kFold})))));
  bots.push_back(MakePolicyBot(
      *game, /*player_id=*/0, /*seed=*/0,
      std::unique_ptr<UniformRestrictedActions>(new UniformRestrictedActions(
          std::vector<ActionType>({ActionType::kCall, ActionType::kRaise})))));
  for (int i = 0; i < bots.size(); ++i) {
    for (int j = 0; j < bots.size(); ++j) {
      std::unique_ptr<State> state = game->NewInitialState();
      std::vector<Bot*> bots_ptrs = {bots[i].get(), bots[j].get()};
      EvaluateBots(state.get(), bots_ptrs, /*seed=*/42);
    }
  }
}

using namespace open_spiel::algorithms;
void TestGameTree() {
  std::cout << std::endl;
  std::cout << "TestGameTree " << std::endl;
  std::vector<std::string> game_names = {"kuhn_poker", "leduc_poker",
                                         "FHP_poker", "HULH_poker"};
  std::unordered_map<std::string, GameParameters> game_parameters = {
      {"kuhn_poker", KuhnPokerParameters()},
      {"leduc_poker", LeducPokerParameters()},
      {"FHP_poker", FHPPokerParameters()},
      {"HULH_poker", HULHPokerParameters()}};
  std::unordered_map<std::string, int> num_histories = {
      // Leduc: 85 =
      // 6(sequence) + 4(terminal) + 5(continuing) * (6(sequence) +
      // 4(terminal) + 5(continuing))
      // Kuhn: 9 =
      // 4(sequence, _, c, cr, r) + 2(terminal, rf, cfr) +
      // 3(continuing, cc, crc, rc)
      // FHP: 162 =
      // 8 + 7 + 7 * (8(seq) + 7(term) + 6(term))
      // HULH: 12902
      // 8 + 7 + 7 * (8 + 6 + 7 * (10 + 8 + 9 * (10 + 17)))
      {"kuhn_poker", 9},
      {"leduc_poker", 85},
      {"FHP_poker", 162},
      {"HULH_poker", 12902}};
  for (const auto& game_name : game_names) {
    std::shared_ptr<const Game> game =
        LoadGame("universal_poker", game_parameters[game_name]);
    PublicTree tree(game->NewInitialState());
    if (tree.NumHistories() != num_histories[game_name]) {
      // TODO(b/126764761): Replace calls to SpielFatalError with more
      // appropriate test macros once they support logging.
      SpielFatalError(absl::StrCat(
          "In the game ", game_name,
          ", tree has wrong number of nodes: ", tree.NumHistories(), " but ",
          num_histories[game_name], " nodes were expected."));
    }

    // Check that the root is not null.
    if (tree.Root() == nullptr) {
      SpielFatalError("Root of PublicTree is null for game: " + game_name);
    }
    int i = 0;
    for (const std::string& history : tree.GetHistories()) {
      ++i;
      PublicNode* node = tree.GetByHistory(history);
      std::cout << i << " " << history << std::endl;
      if (node == nullptr) {
        SpielFatalError(absl::StrCat("node is null for history: ", history,
                                     " in game: ", game_name));
      }
      if (node->GetState() == nullptr) {
        SpielFatalError(absl::StrCat("state is null for history: ", history,
                                     " in game: ", game_name));
      }
      if (node->GetState()->ToString() != node->GetHistory()) {
        SpielFatalError(
            "history generated by state does not match history"
            " stored in PublicNode.");
      }
      if (history != node->GetHistory()) {
        SpielFatalError(
            "history key does not match history stored in "
            "PublicNode.");
      }
      if (node->GetType() != StateType::kTerminal &&
          node->GetType() != StateType::kChance) {
        std::vector<Action> legal_actions = node->GetState()->LegalActions();
        std::vector<Action> child_actions = node->GetChildActions();
        if (legal_actions.size() != child_actions.size()) {
          SpielFatalError(absl::StrCat(
              "For state ", history, ", child actions has a different size (",
              child_actions.size(), ") than legal actions (",
              legal_actions.size(), ")."));
        }
        for (int i = 0; i < legal_actions.size(); ++i) {
          if (legal_actions[i] != child_actions[i]) {
            SpielFatalError(absl::StrCat(
                "legal_actions[i] != child_actions[i]: ", legal_actions[i],
                " != ", child_actions[i]));
          }
        }
      }

      if (node->GetType() != StateType::kTerminal &&
          node->GetType() != StateType::kChance &&
          node->NumChildren() != node->GetState()->LegalActions().size()) {
        SpielFatalError(
            absl::StrCat("number of child nodes does not match number of legal"
                         " actions in history: ",
                         history, " in game: ", game_name));
      }
    }
  }
}

}  // namespace
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char** argv) {
  // open_spiel::universal_poker::LoadKuhnLimitWithAndWithoutGameDef();
  // open_spiel::universal_poker::LoadHoldemNoLimit6PWithAndWithoutGameDef();
  // open_spiel::universal_poker::LoadAndRunGamesFullParameters();
  // open_spiel::universal_poker::LoadGameFromDefaultConfig();
  // open_spiel::universal_poker::LoadAndRunGameFromGameDef();
  // open_spiel::universal_poker::LoadAndRunGameFromDefaultConfig();
  // open_spiel::universal_poker::BasicUniversalPokerTests();
  // open_spiel::universal_poker::HUNLRegressionTests();
  // open_spiel::universal_poker::ChumpPolicyTests();
  open_spiel::universal_poker::TestGameTree();
}
