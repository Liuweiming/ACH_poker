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

#include "open_spiel/tests/basic_tests.h"

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include "absl/types/optional.h"

#include "absl/time/clock.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace testing {

namespace {

constexpr int kInvalidHistoryPlayer = -300;
constexpr int kInvalidHistoryAction = -301;
constexpr double kUtilitySumTolerance = 1e-9;

// Information about the simulation history. Used to track past states and
// actions for rolling back simulations via UndoAction, and check History.
// For simultaneous games, a simultaneous move will be stored as several items.
// The state will be nullptr and the player kInvalidHistoryPlayer for invalid
// transitions.
// The transition state_0 --[action]--> state_1 --[action2]--> ... is stored as:
// (state_0, state_0.CurrentPlayer(), action),
// (state_1, state_1.CurrentPlayer(), action2), ...
struct HistoryItem {
  std::unique_ptr<State> state;
  Player player;
  Action action;
  HistoryItem(std::unique_ptr<State> _state, Player _player, int _action)
      : state(std::move(_state)), player(_player), action(_action) {}
};

// Apply the action to the specified state. If clone is implemented, then do
// more: clone the state, apply the action to the cloned state, and check the
// original state and cloned state are equal using their string
// representation.
void ApplyActionTestClone(const Game& game, State* state,
                          const std::vector<Action>& joint_action) {
  std::unique_ptr<State> clone = state->Clone();
  state->ApplyActions(joint_action);
  clone->ApplyActions(joint_action);
  SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
  SPIEL_CHECK_EQ(state->History(), clone->History());
}

// Apply the action to the specified state. If clone is implemented, then do
// more: clone the state, apply the action to the cloned state, and check the
// original state and cloned state are equal using their string
// representation.
void ApplyActionTestClone(const Game& game, State* state, Action action) {
  std::unique_ptr<State> clone = state->Clone();
  state->ApplyAction(action);
  clone->ApplyAction(action);
  SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
  SPIEL_CHECK_EQ(state->History(), clone->History());
}

// Check that the legal actions list is empty for the non-current player.
// We only check that for turned-base games.

void LegalActionsIsEmptyForOtherPlayers(const Game& game, State& state) {
  if (game.GetType().dynamics == GameType::Dynamics::kSimultaneous) {
    return;
  }

  Player current_player = state.CurrentPlayer();
  for (Player player = 0; player < game.NumPlayers(); ++player) {
    if (state.IsChanceNode()) {
      continue;
    }
    if (player != current_player) {
      int size = state.LegalActions(player).size();
      // We do not use SPIEL_CHECK_EQ because it does not print the values.
      if (size != 0) {
        std::string str = "";
        absl::StrJoin(state.LegalActions(player), str);
        SpielFatalError(absl::StrCat(
            __FILE__, ":", __LINE__, " ", size, " should be 0 for player ",
            player, "(current_player:", current_player, ")", str));
      }
    }
  }
}

// Check that the legal actions list is sorted.

void LegalActionsAreSorted(const Game& game, State& state) {
  if (state.IsChanceNode()) return;
  for (int player = 0; player < game.NumPlayers(); ++player) {
    auto actions = state.LegalActions(player);
    for (int i = 1; i < actions.size(); ++i) {
      SPIEL_CHECK_LT(actions[i - 1], actions[i]);
    }
  }
}

void LegalActionsMaskTest(const Game& game, const State& state,
                          const std::vector<Action>& legal_actions) {
  std::vector<int> legal_actions_mask =
      state.LegalActionsMask(state.CurrentPlayer());
  SPIEL_CHECK_EQ(legal_actions_mask.size(), game.NumDistinctActions());
  for (Action action : legal_actions) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, game.NumDistinctActions());
    SPIEL_CHECK_EQ(legal_actions_mask[action], 1);
  }

  int num_ones = 0;
  for (int i = 0; i < game.NumDistinctActions(); ++i) {
    SPIEL_CHECK_TRUE(legal_actions_mask[i] == 0 || legal_actions_mask[i] == 1);
    if (legal_actions_mask[i] == 1) {
      num_ones++;
    }
  }

  SPIEL_CHECK_EQ(num_ones, legal_actions.size());
}

bool IsPowerOfTwo(int n) { return n == 0 || (n & (n - 1)) == 0; }

}  // namespace

// Checks that the game can be loaded.
void LoadGameTest(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  SPIEL_CHECK_TRUE(game != nullptr);
}

void NoChanceOutcomesTest(const Game& game) {
  std::cout << "NoChanceOutcomesTest, game = " << game.GetType().short_name
            << std::endl;
  int max_outcomes = game.MaxChanceOutcomes();
  SPIEL_CHECK_EQ(max_outcomes, 0);
}

void ChanceOutcomesTest(const Game& game) {
  std::cout << "ChanceOutcomesTest, game = " << game.GetType().short_name
            << std::endl;
  int max_outcomes = game.MaxChanceOutcomes();
  SPIEL_CHECK_GT(max_outcomes, 0);
}

void TestUndo(std::unique_ptr<State> state,
              const std::vector<HistoryItem>& history) {
  // TODO(author2): We can just check each UndoAction.
  for (auto prev = history.rbegin(); prev != history.rend(); ++prev) {
    state->UndoAction(prev->player, prev->action);
    SPIEL_CHECK_EQ(state->ToString(), prev->state->ToString());
    // We also check that UndoActions correctly updates history_.
    SPIEL_CHECK_EQ(state->History(), prev->state->History());
  }
}

void TestSerializeDeserialize(const Game& game, const State* state) {
  const std::string& ser_str = SerializeGameAndState(game, *state);
  std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
      game_and_state = DeserializeGameAndState(ser_str);
  SPIEL_CHECK_EQ(game.ToString(), game_and_state.first->ToString());
  SPIEL_CHECK_EQ(state->ToString(), game_and_state.second->ToString());
}

void TestHistoryContainsActions(const Game& game,
                                const std::vector<HistoryItem>& history) {
  std::vector<Action> actions = {};
  for (const auto& history_item : history) {
    if (history_item.state != nullptr) {
      SPIEL_CHECK_EQ(history_item.state->History(), actions);
    }
    actions.push_back(history_item.action);
  }
}

void CheckReturnsSum(const Game& game, const State& state) {
  std::vector<double> returns = state.Returns();
  double rsum = std::accumulate(returns.begin(), returns.end(), 0.0);

  switch (game.GetType().utility) {
    case GameType::Utility::kZeroSum: {
      SPIEL_CHECK_TRUE(Near(rsum, 0.0, kUtilitySumTolerance));
      break;
    }
    case GameType::Utility::kConstantSum: {
      SPIEL_CHECK_TRUE(Near(rsum, game.UtilitySum(), kUtilitySumTolerance));
      break;
    }
    case GameType::Utility::kIdentical: {
      for (int i = 1; i < returns.size(); ++i) {
        SPIEL_CHECK_TRUE(
            Near(returns[i], returns[i - 1], kUtilitySumTolerance));
      }
      break;
    }
    case GameType::Utility::kGeneralSum: {
      break;
    }
  }
}

// Tests all observation and information_state related methods which are
// supported by the game, for all players.
void CheckObservables(const Game& game, const State& state) {
  for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
    if (game.GetType().provides_information_state_tensor) {
      std::vector<double> v = state.InformationStateTensor(p);
      SPIEL_CHECK_EQ(v.size(), game.InformationStateTensorSize());
    }
    if (game.GetType().provides_observation_tensor) {
      std::vector<double> v = state.ObservationTensor(p);
      SPIEL_CHECK_EQ(v.size(), game.ObservationTensorSize());
    }
    if (game.GetType().provides_information_state_string) {
      // Checking it does not raise errors.
      state.InformationStateString(p);
    }
    if (game.GetType().provides_observation_string) {
      // Checking it does not have errors.
      state.ObservationString(p);
    }
  }
}

void RandomSimulation(std::mt19937* rng, const Game& game, bool undo,
                      bool serialize) {
  std::vector<HistoryItem> history;
  std::vector<double> episode_returns(game.NumPlayers(), 0);

  int infostate_vector_size = game.GetType().provides_information_state_tensor
                                  ? game.InformationStateTensorSize()
                                  : 0;
  std::cout << "Information state vector size: " << infostate_vector_size
            << std::endl;

  int observation_vector_size = game.GetType().provides_observation_tensor
                                    ? game.ObservationTensorSize()
                                    : 0;
  std::cout << "Observation vector size: " << observation_vector_size
            << std::endl;

  SPIEL_CHECK_TRUE(game.MinUtility() < game.MaxUtility());
  std::cout << "Utility range: " << game.MinUtility() << " "
            << game.MaxUtility() << std::endl;

  std::cout << "Starting new game.." << std::endl;
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();

  std::cout << "Initial state:" << std::endl;
  std::cout << "State:" << std::endl << state->ToString() << std::endl;
  int game_length = 0;

  while (!state->IsTerminal()) {
    std::cout << "player " << state->CurrentPlayer() << std::endl;

    LegalActionsIsEmptyForOtherPlayers(game, *state);
    LegalActionsAreSorted(game, *state);

    // Test cloning the state.
    std::unique_ptr<open_spiel::State> state_copy = state->Clone();
    SPIEL_CHECK_EQ(state->ToString(), state_copy->ToString());
    SPIEL_CHECK_EQ(state->History(), state_copy->History());

    if (serialize && (history.size() < 10 || IsPowerOfTwo(history.size()))) {
      TestSerializeDeserialize(game, state.get());
    }

    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      Action action = open_spiel::SampleAction(outcomes, *rng).first;

      std::cout << "sampled outcome: "
                << state->ActionToString(kChancePlayerId, action) << std::endl;

      history.emplace_back(state->Clone(), kChancePlayerId, action);
      state->ApplyAction(action);

      if (undo && (history.size() < 10 || IsPowerOfTwo(history.size()))) {
        TestUndo(state->Clone(), history);
      }
    } else if (state->CurrentPlayer() == open_spiel::kSimultaneousPlayerId) {
      std::vector<double> rewards = state->Rewards();
      SPIEL_CHECK_EQ(rewards.size(), game.NumPlayers());
      std::cout << "Rewards: " << absl::StrJoin(rewards, " ") << std::endl;
      for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
        episode_returns[p] += rewards[p];
      }

      // Players choose simultaneously.
      std::vector<Action> joint_action;

      // Sample an action for each player
      for (auto p = Player{0}; p < game.NumPlayers(); p++) {
        std::vector<Action> actions = state->LegalActions(p);
        std::uniform_int_distribution<int> dis(0, actions.size() - 1);
        Action action = actions[dis(*rng)];
        joint_action.push_back(action);
        if (p == 0) {
          history.emplace_back(state->Clone(), kInvalidHistoryPlayer, action);
        } else {
          history.emplace_back(nullptr, kInvalidHistoryPlayer, action);
        }
        std::cout << "player " << p << " chose "
                  << state->ActionToString(p, action) << std::endl;

        CheckObservables(game, *state);
      }

      ApplyActionTestClone(game, state.get(), joint_action);
      game_length++;
    } else {
      std::vector<double> rewards = state->Rewards();
      SPIEL_CHECK_EQ(rewards.size(), game.NumPlayers());
      std::cout << "Rewards: " << absl::StrJoin(rewards, " ") << std::endl;
      for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
        episode_returns[p] += rewards[p];
      }

      // Decision node.
      Player player = state->CurrentPlayer();

      CheckObservables(game, *state);

      // Sample an action uniformly.
      std::vector<Action> actions = state->LegalActions();
      LegalActionsMaskTest(game, *state, actions);
      if (state->IsTerminal())
        SPIEL_CHECK_TRUE(actions.empty());
      else
        SPIEL_CHECK_FALSE(actions.empty());
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];

      std::cout << "chose action: " << action << " ("
                << state->ActionToString(player, action) << ")" << std::endl;

      history.emplace_back(state->Clone(), player, action);
      ApplyActionTestClone(game, state.get(), action);
      game_length++;

      if (undo && (history.size() < 10 || IsPowerOfTwo(history.size()))) {
        TestUndo(state->Clone(), history);
      }
    }

    std::cout << "State: " << std::endl << state->ToString() << std::endl;
  }

  SPIEL_CHECK_LE(game_length, game.MaxGameLength());

  std::cout << "Reached a terminal state!" << std::endl;
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kTerminalPlayerId);
  std::vector<double> rewards = state->Rewards();
  std::cout << "Rewards: " << absl::StrJoin(rewards, " ") << std::endl;

  history.emplace_back(state->Clone(), kTerminalPlayerId,
                       kInvalidHistoryAction);
  TestHistoryContainsActions(game, history);

  // Check the information state of the terminal, too. This is commonly needed,
  // for example, as a final observation in an RL environment.
  CheckObservables(game, *state);

  // Check that the returns satisfy the constraints based on the game type.
  CheckReturnsSum(game, *state);

  // Now, check each individual return is within bounds.
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns.size(), game.NumPlayers());
  for (Player player = 0; player < game.NumPlayers(); player++) {
    double final_return = returns[player];
    SPIEL_CHECK_FLOAT_EQ(final_return, state->PlayerReturn(player));
    SPIEL_CHECK_GE(final_return, game.MinUtility());
    SPIEL_CHECK_LE(final_return, game.MaxUtility());
    std::cout << "Final return to player " << player << " is " << final_return
              << std::endl;
    episode_returns[player] += rewards[player];
    SPIEL_CHECK_TRUE(Near(episode_returns[player], final_return));
  }
}

// Perform sims random simulations of the specified game.
void RandomSimTest(const Game& game, int num_sims) {
  std::mt19937 rng;
  std::cout << "\nRandomSimTest, game = " << game.GetType().short_name
            << ", num_sims = " << num_sims << std::endl;
  for (int sim = 0; sim < num_sims; ++sim) {
    RandomSimulation(&rng, game, /*undo=*/false, /*serialize=*/true);
  }
}

void RandomSimTestWithUndo(const Game& game, int num_sims) {
  std::mt19937 rng;
  std::cout << "RandomSimTestWithUndo, game = " << game.GetType().short_name
            << ", num_sims = " << num_sims << std::endl;
  for (int sim = 0; sim < num_sims; ++sim) {
    RandomSimulation(&rng, game, /*undo=*/true, /*serialize=*/true);
  }
}

void RandomSimTestNoSerialize(const Game& game, int num_sims) {
  std::mt19937 rng;
  std::cout << "RandomSimTestNoSerialize, game = " << game.GetType().short_name
            << ", num_sims = " << num_sims << std::endl;
  for (int sim = 0; sim < num_sims; ++sim) {
    RandomSimulation(&rng, game, /*undo=*/false, /*serialize=*/false);
  }
}

// Format chance outcomes as a string, for error messages.
std::string ChanceOutcomeStr(const ActionsAndProbs& chance_outcomes) {
  std::string str;
  for (auto outcome : chance_outcomes) {
    if (!str.empty()) str.append(", ");
    absl::StrAppend(&str, "(", outcome.first, ", ", outcome.second, ")");
  }
  return str;
}

// Check chance outcomes in a state and all child states.
// We check that:
// - That LegalActions(kChancePlayerId) (which often defaults to the actions in
//   ChanceOutcomes) and LegalActions() return the same result.
// - All the chance outcome actions are legal actions
// - All the chance outcome actions are different from each other.
// - That the probabilities are within [0, 1] and sum to 1.
void CheckChanceOutcomes(const State& state) {
  if (state.IsTerminal()) return;
  if (state.IsChanceNode()) {
    auto legal_actions = state.LegalActions(kChancePlayerId);
    auto default_legal_actions = state.LegalActions();
    if (legal_actions != default_legal_actions) {
      SpielFatalError(absl::StrCat(
          "Legalactions() and LegalActions(kChancePlayerId) do not give the "
          "same result:",
          "\nLegalActions():                ",
          absl::StrJoin(default_legal_actions, ", "),
          "\nLegalActions(kChancePlayerId): ",
          absl::StrJoin(legal_actions, ", ")));
    }
    std::set<Action> legal_action_set(legal_actions.begin(),
                                      legal_actions.end());
    auto chance_outcomes = state.ChanceOutcomes();

    std::vector<Action> chance_outcome_actions;
    double sum = 0;
    for (const auto& chance_outcome : chance_outcomes) {
      chance_outcome_actions.push_back(chance_outcome.first);
      if (legal_action_set.count(chance_outcome.first) == 0) {
        SpielFatalError(absl::StrCat("LegalActions()=[",
                                     absl::StrJoin(legal_actions, ", "),
                                     "] inconsistent with ChanceOutcomes()=",
                                     ChanceOutcomeStr(chance_outcomes), "."));
      }
      if (chance_outcome.second <= 0. || chance_outcome.second > 1) {
        SpielFatalError(absl::StrCat("Invalid probability for outcome: P(",
                                     chance_outcome.first, ")=",
                                     chance_outcome.second, "; all outcomes=",
                                     ChanceOutcomeStr(chance_outcomes)));
      }
      sum += chance_outcome.second;
    }
    std::set<Action> chance_outcome_actions_set(chance_outcome_actions.begin(),
                                                chance_outcome_actions.end());
    if (chance_outcome_actions.size() != chance_outcome_actions_set.size()) {
      std::sort(chance_outcome_actions.begin(), chance_outcome_actions.end());
      SpielFatalError(absl::StrCat(
          "There are some duplicate actions in ChanceOutcomes\n. There are: ",
          chance_outcome_actions_set.size(), " unique legal actions over ",
          chance_outcome_actions.size(),
          " chance outcome actions.\n Sorted legal actions:\n",
          absl::StrJoin(chance_outcome_actions, ", ")));
    }
    constexpr double eps = 1e-5;
    if (sum < 1 - eps || sum > 1 + eps) {
      SpielFatalError(absl::StrCat("Invalid probabilities; sum=", sum,
                                   "; all outcomes=",
                                   ChanceOutcomeStr(chance_outcomes)));
    }
  }
  // Handles chance nodes, player nodes, including simultaneous nodes if
  // supported.
  for (auto action : state.LegalActions()) {
    auto next_state = state.Child(action);
    CheckChanceOutcomes(*next_state);
  }
}

void CheckChanceOutcomes(const Game& game) {
  CheckChanceOutcomes(*game.NewInitialState());
}

// Verifies that ResampleFromInfostate is correctly implemented.
void ResampleInfostateTest(const Game& game, int num_sims) {
  std::mt19937 rng;
  UniformProbabilitySampler sampler;
  for (int i = 0; i < num_sims; ++i) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        for (int p = 0; p < state->NumPlayers(); ++p) {
          std::unique_ptr<State> other_state =
              state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         other_state->InformationStateString(p));
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         other_state->InformationStateTensor(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), other_state->CurrentPlayer());
        }
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(rng)];
      state->ApplyAction(action);
    }
  }
}

}  // namespace testing
}  // namespace open_spiel
