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

#include "open_spiel/spiel.h"

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/games/liars_dice.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/policy.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace testing {
namespace {

void GeneralTests() {
  // Number of supported games should be > 0.
  std::vector<std::string> game_names = RegisteredGames();
  SPIEL_CHECK_GT(game_names.size(), 0);
}

void KuhnTests() {
  // Default params (2 players)
  RandomSimTest(*LoadGame("kuhn_poker"), /*num_sims=*/100);

  // More than two players.
  for (Player players = 3; players <= 5; players++) {
    RandomSimTest(
        *LoadGame("kuhn_poker", {{"players", GameParameter(players)}}),
        /*num_sims=*/100);
  }
}

void TicTacToeTests() {
  auto tic_tac_toe = LoadGame("tic_tac_toe");
  NoChanceOutcomesTest(*tic_tac_toe);
  RandomSimTest(*tic_tac_toe, /*num_sims=*/100);
}

// Dummy game to test flat joint action logic.
class FlatJointActionTestGame : public SimMoveGame {
 public:
  explicit FlatJointActionTestGame(const GameParameters& params)
      : SimMoveGame(GameType{}, params) {}
  int NumDistinctActions() const override { return 8; }
  std::unique_ptr<State> NewInitialState() const override { return nullptr; }
  int MaxChanceOutcomes() const override { return 4; }
  int NumPlayers() const override { return 3; }
  double MinUtility() const override { return -10; }
  double MaxUtility() const override { return 10; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new FlatJointActionTestGame(*this));
  }
  std::vector<int> InformationStateTensorShape() const override { return {}; }
  int MaxGameLength() const override { return 1; }
};

// Dummy state to test flat joint action logic.
class FlatJointActionTestState : public SimMoveState {
 public:
  FlatJointActionTestState()
      : SimMoveState(std::shared_ptr<const FlatJointActionTestGame>(
            new FlatJointActionTestGame({}))) {}
  const std::vector<Action>& JointAction() const { return joint_action_; }
  std::vector<Action> LegalActions(Player player) const override {
    if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
    switch (player) {
      case 0:
        return {2, 4, 6};
      case 1:
        return {1, 3, 5};
      case 2:
        return {0, 100};
    }
    SpielFatalError("Invalid player id");
  }
  Player CurrentPlayer() const override { return kSimultaneousPlayerId; }
  std::string ActionToString(Player player, Action action_id) const override {
    if (player == kSimultaneousPlayerId)
      return FlatJointActionToString(action_id);
    return absl::StrCat("(p=", player, ",a=", action_id, ")");
  }
  std::string ToString() const override { return ""; }
  bool IsTerminal() const override { return false; }
  std::vector<double> Returns() const override { return {}; }
  std::unique_ptr<State> Clone() const override { return nullptr; }

 protected:
  void DoApplyActions(const std::vector<Action>& actions) override {
    joint_action_ = actions;
  }

 protected:
  std::vector<Action> joint_action_;
};

void FlatJointactionTest() {
  FlatJointActionTestState state;
  auto legal_flat_joint_actions = state.LegalActions(kSimultaneousPlayerId);
  SPIEL_CHECK_EQ(legal_flat_joint_actions.size(), 18);
  for (int i = 0; i < 18; ++i) {
    std::cerr << "Joint action " << i << " expands to "
              << state.ActionToString(kSimultaneousPlayerId, i) << std::endl;
  }
  // Last-but-one joint action --> last action for everyone except p0 (which
  // takes its last-but-one action).
  SPIEL_CHECK_EQ(state.ActionToString(kSimultaneousPlayerId, 16),
                 "[(p=0,a=4), (p=1,a=5), (p=2,a=100)]");
  state.ApplyAction(16);
  std::vector<Action> expected_joint_action{4, 5, 100};
  SPIEL_CHECK_EQ(state.JointAction(), expected_joint_action);
}

using PolicyGenerator = std::function<TabularPolicy(const Game& game)>;

constexpr int kNumSimulations = 10;

void TestPoliciesCanPlay(PolicyGenerator policy_generator, const Game& game) {
  TabularPolicy policy = policy_generator(game);
  std::mt19937 rng(0);
  for (int i = 0; i < kNumSimulations; ++i) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      ActionsAndProbs outcomes;
      if (state->IsChanceNode()) {
        outcomes = state->ChanceOutcomes();
      } else {
        outcomes = policy.GetStatePolicy(state->InformationStateString());
      }
      state->ApplyAction(open_spiel::SampleAction(outcomes, rng).first);
    }
  }
}

void TestPoliciesCanPlay(const Policy& policy, const Game& game) {
  std::mt19937 rng(0);
  for (int i = 0; i < kNumSimulations; ++i) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      ActionsAndProbs outcomes;
      if (state->IsChanceNode()) {
        outcomes = state->ChanceOutcomes();
      } else {
        outcomes = policy.GetStatePolicy(*state);
      }
      state->ApplyAction(open_spiel::SampleAction(outcomes, rng).first);
    }
  }
}

void TestEveryInfostateInPolicy(PolicyGenerator policy_generator,
                                const Game& game) {
  TabularPolicy policy = policy_generator(game);
  std::vector<std::unique_ptr<State>> to_visit;
  to_visit.push_back(game.NewInitialState());
  while (!to_visit.empty()) {
    std::unique_ptr<State> state = std::move(to_visit.back());
    to_visit.pop_back();
    for (Action action : state->LegalActions()) {
      to_visit.push_back(state->Child(action));
    }
    if (!state->IsChanceNode() && !state->IsTerminal()) {
      SPIEL_CHECK_EQ(
          policy.GetStatePolicy(state->InformationStateString()).size(),
          state->LegalActions().size());
    }
  }
}

void PolicyTest() {
  auto random_policy_default_seed = [](const Game& game) {
    return GetRandomPolicy(game);
  };
  std::vector<PolicyGenerator> policy_generators = {
      GetUniformPolicy, random_policy_default_seed, GetFirstActionPolicy};

  // For some reason, this can't seem to be brace-initialized, so instead we use
  // push_back.
  std::unique_ptr<Policy> uniform_policy =
      std::unique_ptr<UniformPolicy>(new UniformPolicy);
  for (const std::string& game_name :
       {"leduc_poker", "kuhn_poker", "liars_dice"}) {
    std::shared_ptr<const Game> game = LoadGame(game_name);
    for (const auto& policy_generator : policy_generators) {
      TestEveryInfostateInPolicy(policy_generator, *game);
      TestPoliciesCanPlay(policy_generator, *game);
    }
    TestPoliciesCanPlay(*uniform_policy, *game);
  }
}

void LeducPokerDeserializeTest() {
  // Example Leduc state: player 1 gets the 0th card, player 2 gets the 3rd card
  // and the first two actions are: check, check.
  std::string serialized_game_and_state =
      "# Automatically generated by OpenSpiel SerializeGameAndState\n"
      "[Meta]\n"
      "Version: 1\n"
      "\n"
      "[Game]\n"
      "leduc_poker()\n"
      "[State]\n"
      "0\n"  // first chance event (deal to first player)
      "3\n"  // second chance event (deal to second player)
      "1\n"  // check
      "1\n"  // check
      "\n";

  std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
      game_and_state =
          open_spiel::DeserializeGameAndState(serialized_game_and_state);

  // Should be at round 2 deal (chance node).
  SPIEL_CHECK_TRUE(game_and_state.second->IsChanceNode());

  // Check that the game got deserialized properly.
  SPIEL_CHECK_EQ(game_and_state.first->ToString(),
                 LoadGame("leduc_poker")->ToString());

  // And now check that serializing this game and state gives the same string as
  // above.
  SPIEL_CHECK_EQ(
      SerializeGameAndState(*game_and_state.first, *game_and_state.second),
      serialized_game_and_state);
}

void GameParametersTest() {
  // Bare name
  auto params = GameParametersFromString("game_one");
  SPIEL_CHECK_EQ(params.size(), 1);
  SPIEL_CHECK_EQ(params["name"].string_value(), "game_one");

  // Name with empty list
  params = GameParametersFromString("game_two()");
  SPIEL_CHECK_EQ(params.size(), 1);
  SPIEL_CHECK_EQ(params["name"].string_value(), "game_two");

  // Single string parameter
  params = GameParametersFromString("game_three(foo=bar)");
  SPIEL_CHECK_EQ(params.size(), 2);
  SPIEL_CHECK_EQ(params["name"].string_value(), "game_three");
  SPIEL_CHECK_EQ(params["foo"].string_value(), "bar");

  // Every type of parameter
  params = GameParametersFromString(
      "game_four(str=strval,int=42,float=-1.2,game1=nested(),"
      "game2=nested2(param=val),bool1=True,bool2=False)");
  SPIEL_CHECK_EQ(params.size(), 8);
  SPIEL_CHECK_EQ(params["name"].string_value(), "game_four");
  SPIEL_CHECK_EQ(params["str"].string_value(), "strval");
  SPIEL_CHECK_EQ(params["int"].int_value(), 42);
  SPIEL_CHECK_EQ(params["float"].double_value(), -1.2);
  SPIEL_CHECK_EQ(params["bool1"].bool_value(), true);
  SPIEL_CHECK_EQ(params["bool2"].bool_value(), false);

  auto game1 = params["game1"].game_value();
  SPIEL_CHECK_EQ(game1.size(), 1);
  SPIEL_CHECK_EQ(game1["name"].string_value(), "nested");

  auto game2 = params["game2"].game_value();
  SPIEL_CHECK_EQ(game2.size(), 2);
  SPIEL_CHECK_EQ(game2["name"].string_value(), "nested2");
  SPIEL_CHECK_EQ(game2["param"].string_value(), "val");
}

}  // namespace
}  // namespace testing
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::GeneralTests();
  open_spiel::testing::KuhnTests();
  open_spiel::testing::TicTacToeTests();
  open_spiel::testing::FlatJointactionTest();
  open_spiel::testing::PolicyTest();
  open_spiel::testing::LeducPokerDeserializeTest();
  open_spiel::testing::GameParametersTest();
}
