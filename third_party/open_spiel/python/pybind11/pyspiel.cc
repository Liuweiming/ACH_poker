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

#include <memory>
#include <unordered_map>

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/cfr_br.h"
#include "open_spiel/algorithms/deterministic_policy.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/matrix_game_utils.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/tensor_game_utils.h"
#include "open_spiel/algorithms/trajectories.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/game_transforms/normal_form_extensive_game.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/normal_form_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/query.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"
#include "pybind11/include/pybind11/functional.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

// This file contains OpenSpiel's Python API. The best place to see an overview
// of the API is to refer to python/examples/example.py. Generally, all the core
// functions are exposed as snake case in Python (i.e. CurrentPlayer becomes
// current_player, ApplyAction becomes apply_action, etc.) but otherwise the
// functions and their effect remain the same. For a more detailed documentation
// of each of the core API functions, please see spiel.h.

namespace open_spiel {
namespace {

using ::open_spiel::algorithms::Evaluator;
using ::open_spiel::algorithms::Exploitability;
using ::open_spiel::algorithms::NashConv;
using ::open_spiel::algorithms::TabularBestResponse;
using ::open_spiel::matrix_game::MatrixGame;
using ::open_spiel::tensor_game::TensorGame;

namespace py = ::pybind11;

// This exception class is used to forward errors from Spiel to Python.
// Do not create exceptions of this type directly! Instead, call
// SpielFatalError, which will raise a Python exception when called from
// Python, and exit the process otherwise.
class SpielException : public std::exception {
 public:
  explicit SpielException(std::string message) : message_(message) {}
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

// Trampoline helper class to allow implementing Bots in Python. See
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
class PyBot : public Bot {
 public:
  // We need the bot constructor
  using Bot::Bot;
  ~PyBot() override = default;

  using step_retval_t = std::pair<ActionsAndProbs, open_spiel::Action>;

  // Choose and execute an action in a game. The bot should return its
  // distribution over actions and also its selected action.
  open_spiel::Action Step(const State& state) override {
    PYBIND11_OVERLOAD_PURE_NAME(
        open_spiel::Action,  // Return type (must be simple token)
        Bot,                 // Parent class
        "step",              // Name of function in Python
        Step,                // Name of function in C++
        state                // Arguments
        );
  }

  // Restart at the specified state.
  void Restart() override {
    PYBIND11_OVERLOAD_NAME(
        void,       // Return type (must be a simple token for macro parser)
        Bot,        // Parent class
        "restart",  // Name of function in Python
        Restart,    // Name of function in C++
        // The trailing coma after Restart is necessary to say "No argument"
        );
  }
  bool ProvidesForceAction() override {
    PYBIND11_OVERLOAD_NAME(
        bool,  // Return type (must be a simple token for macro parser)
        Bot,   // Parent class
        "provides_force_action",  // Name of function in Python
        ProvidesForceAction,      // Name of function in C++
                                  // Arguments
        );
  }
  void ForceAction(const State& state, Action action) override {
    PYBIND11_OVERLOAD_NAME(
        void,  // Return type (must be a simple token for macro parser)
        Bot,   // Parent class
        "force_action",  // Name of function in Python
        ForceAction,     // Name of function in C++
        state,           // Arguments
        action);
  }
  void InformAction(const State& state, Player player_id,
                    Action action) override {
    PYBIND11_OVERLOAD_NAME(
        void,  // Return type (must be a simple token for macro parser)
        Bot,   // Parent class
        "inform_action",  // Name of function in Python
        InformAction,     // Name of function in C++
        state,            // Arguments
        player_id, action);
  }

  void RestartAt(const State& state) override {
    PYBIND11_OVERLOAD_NAME(
        void,          // Return type (must be a simple token for macro parser)
        Bot,           // Parent class
        "restart_at",  // Name of function in Python
        RestartAt,     // Name of function in C++
        state          // Arguments
        );
  }
  bool ProvidesPolicy() override {
    PYBIND11_OVERLOAD_NAME(
        bool,  // Return type (must be a simple token for macro parser)
        Bot,   // Parent class
        "provides_policy",  // Name of function in Python
        ProvidesPolicy,     // Name of function in C++
                            // Arguments
        );
  }
  ActionsAndProbs GetPolicy(const State& state) override {
    PYBIND11_OVERLOAD_NAME(ActionsAndProbs,  // Return type (must be a simple
                                             // token for macro parser)
                           Bot,              // Parent class
                           "get_policy",     // Name of function in Python
                           GetPolicy,        // Name of function in C++
                           state);
  }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override {
    PYBIND11_OVERLOAD_NAME(
        step_retval_t,  // Return type (must be a simple token for macro parser)
        Bot,            // Parent class
        "step_with_policy",  // Name of function in Python
        StepWithPolicy,      // Name of function in C++
        state                // Arguments
        );
  }
};

// Definintion of our Python module.
PYBIND11_MODULE(pyspiel, m) {
  m.doc() = "Open Spiel";

  py::enum_<open_spiel::GameParameter::Type>(m, "GameParameterType")
      .value("UNSET", open_spiel::GameParameter::Type::kUnset)
      .value("INT", open_spiel::GameParameter::Type::kInt)
      .value("DOUBLE", open_spiel::GameParameter::Type::kDouble)
      .value("STRING", open_spiel::GameParameter::Type::kString)
      .value("BOOL", open_spiel::GameParameter::Type::kBool);

  py::class_<GameParameter> game_parameter(m, "GameParameter");
  game_parameter.def(py::init<double>())
      .def(py::init<std::string>())
      .def(py::init<bool>())
      .def(py::init<int>())
      .def(py::init<GameParameters>())
      .def("is_mandatory", &GameParameter::is_mandatory)
      .def("__str__", &GameParameter::ToString)
      .def("__repr__", &GameParameter::ToReprString)
      .def("__eq__", [](const GameParameter& value, GameParameter* value2) {
        return value2 && value.ToReprString() == value2->ToReprString();
      });

  py::class_<UniformProbabilitySampler> uniform_sampler(
      m, "UniformProbabilitySampler");
  uniform_sampler.def(py::init<double, double>())
      .def(py::init<int, double, double>())
      .def("__call__", &UniformProbabilitySampler::operator());

  py::enum_<open_spiel::StateType>(m, "StateType")
      .value("TERMINAL", open_spiel::StateType::kTerminal)
      .value("CHANCE", open_spiel::StateType::kChance)
      .value("DECISION", open_spiel::StateType::kDecision)
      .export_values();

  py::class_<GameType> game_type(m, "GameType");
  game_type
      .def(py::init<std::string, std::string, GameType::Dynamics,
                    GameType::ChanceMode, GameType::Information,
                    GameType::Utility, GameType::RewardModel, int, int, bool,
                    bool, bool, bool, std::map<std::string, GameParameter>>(),
           py::arg("short_name"), py::arg("long_name"), py::arg("dynamics"),
           py::arg("chance_mode"), py::arg("information"), py::arg("utility"),
           py::arg("reward_model"), py::arg("max_num_players"),
           py::arg("min_num_players"),
           py::arg("provides_information_state_string"),
           py::arg("provides_information_state_tensor"),
           py::arg("provides_observation_string"),
           py::arg("provides_observation_tensor"),
           py::arg("parameter_specification"))
      .def_readonly("short_name", &GameType::short_name)
      .def_readonly("long_name", &GameType::long_name)
      .def_readonly("dynamics", &GameType::dynamics)
      .def_readonly("chance_mode", &GameType::chance_mode)
      .def_readonly("information", &GameType::information)
      .def_readonly("utility", &GameType::utility)
      .def_readonly("reward_model", &GameType::reward_model)
      .def_readonly("max_num_players", &GameType::max_num_players)
      .def_readonly("min_num_players", &GameType::min_num_players)
      .def_readonly("provides_information_state_string",
                    &GameType::provides_information_state_string)
      .def_readonly("provides_information_state_tensor",
                    &GameType::provides_information_state_tensor)
      .def_readonly("provides_observation_string",
                    &GameType::provides_observation_string)
      .def_readonly("provides_observation_tensor",
                    &GameType::provides_observation_tensor)
      .def_readonly("parameter_specification",
                    &GameType::parameter_specification)
      .def_readonly("default_loadable", &GameType::default_loadable)
      .def("__repr__", [](const GameType& gt) {
        return "<GameType '" + gt.short_name + "'>";
      });

  py::enum_<GameType::Dynamics>(game_type, "Dynamics")
      .value("SEQUENTIAL", GameType::Dynamics::kSequential)
      .value("SIMULTANEOUS", GameType::Dynamics::kSimultaneous);

  py::enum_<GameType::ChanceMode>(game_type, "ChanceMode")
      .value("DETERMINISTIC", GameType::ChanceMode::kDeterministic)
      .value("EXPLICIT_STOCHASTIC", GameType::ChanceMode::kExplicitStochastic)
      .value("SAMPLED_STOCHASTIC", GameType::ChanceMode::kSampledStochastic);

  py::enum_<GameType::Information>(game_type, "Information")
      .value("ONE_SHOT", GameType::Information::kOneShot)
      .value("PERFECT_INFORMATION", GameType::Information::kPerfectInformation)
      .value("IMPERFECT_INFORMATION",
             GameType::Information::kImperfectInformation);

  py::enum_<GameType::Utility>(game_type, "Utility")
      .value("ZERO_SUM", GameType::Utility::kZeroSum)
      .value("CONSTANT_SUM", GameType::Utility::kConstantSum)
      .value("GENERAL_SUM", GameType::Utility::kGeneralSum)
      .value("IDENTICAL", GameType::Utility::kIdentical);

  py::enum_<GameType::RewardModel>(game_type, "RewardModel")
      .value("REWARDS", GameType::RewardModel::kRewards)
      .value("TERMINAL", GameType::RewardModel::kTerminal);

  py::enum_<open_spiel::PlayerId>(m, "PlayerId")
      .value("INVALID", open_spiel::kInvalidPlayer)
      .value("TERMINAL", open_spiel::kTerminalPlayerId)
      .value("CHANCE", open_spiel::kChancePlayerId)
      .value("SIMULTANEOUS", open_spiel::kSimultaneousPlayerId);

  m.attr("INVALID_ACTION") = py::int_(open_spiel::kInvalidAction);

  py::enum_<open_spiel::TensorLayout>(m, "TensorLayout")
      .value("HWC", open_spiel::TensorLayout::kHWC)
      .value("CHW", open_spiel::TensorLayout::kCHW);

  py::class_<State> state(m, "State");
  state.def("current_player", &State::CurrentPlayer)
      .def("apply_action", &State::ApplyAction)
      .def("legal_actions",
           (std::vector<open_spiel::Action> (State::*)(int) const) &
               State::LegalActions)
      .def("legal_actions",
           (std::vector<open_spiel::Action> (State::*)(void) const) &
               State::LegalActions)
      .def("legal_actions_mask",
           (std::vector<int> (State::*)(int) const) & State::LegalActionsMask)
      .def("legal_actions_mask",
           (std::vector<int> (State::*)(void) const) & State::LegalActionsMask)
      .def("action_to_string", (std::string (State::*)(Player, Action) const) &
                                   State::ActionToString)
      .def("action_to_string",
           (std::string (State::*)(Action) const) & State::ActionToString)
      .def("string_to_action",
           (Action (State::*)(Player, const std::string&) const) &
               State::StringToAction)
      .def("string_to_action", (Action (State::*)(const std::string&) const) &
                                   State::StringToAction)
      .def("__str__", &State::ToString)
      .def("is_terminal", &State::IsTerminal)
      .def("rewards", &State::Rewards)
      .def("returns", &State::Returns)
      .def("player_reward", &State::PlayerReward)
      .def("player_return", &State::PlayerReturn)
      .def("is_chance_node", &State::IsChanceNode)
      .def("is_simultaneous_node", &State::IsSimultaneousNode)
      .def("history", &State::History)
      .def("history_str", &State::HistoryString)
      .def("information_state_string",
           (std::string (State::*)(int) const) & State::InformationStateString)
      .def("information_state_string",
           (std::string (State::*)() const) & State::InformationStateString)
      .def("information_state_tensor",
           (std::vector<double> (State::*)(int) const) &
               State::InformationStateTensor)
      .def("information_state_tensor",
           (std::vector<double> (State::*)() const) &
               State::InformationStateTensor)
      .def("observation_string",
           (std::string (State::*)(int) const) & State::ObservationString)
      .def("observation_string",
           (std::string (State::*)() const) & State::ObservationString)
      .def("observation_tensor", (std::vector<double> (State::*)(int) const) &
                                     State::ObservationTensor)
      .def("observation_tensor",
           (std::vector<double> (State::*)() const) & State::ObservationTensor)
      .def("clone", &State::Clone)
      .def("child", &State::Child)
      .def("undo_action", &State::UndoAction)
      .def("apply_actions", &State::ApplyActions)
      .def("num_distinct_actions", &State::NumDistinctActions)
      .def("num_players", &State::NumPlayers)
      .def("chance_outcomes", &State::ChanceOutcomes)
      .def("get_game", &State::GetGame)
      .def("get_type", &State::GetType)
      .def("serialize", &State::Serialize)
      .def("resample_from_infostate", &State::ResampleFromInfostate)
      .def(py::pickle(              // Pickle support
          [](const State& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return std::move(game_and_state.second);
          }));

  py::class_<Game, std::shared_ptr<Game>> game(m, "Game");
  game.def("num_distinct_actions", &Game::NumDistinctActions)
      .def("new_initial_state", &Game::NewInitialState)
      .def("max_chance_outcomes", &Game::MaxChanceOutcomes)
      .def("get_parameters", &Game::GetParameters)
      .def("num_players", &Game::NumPlayers)
      .def("min_utility", &Game::MinUtility)
      .def("max_utility", &Game::MaxUtility)
      .def("get_type", &Game::GetType)
      .def("utility_sum", &Game::UtilitySum)
      .def("information_state_tensor_shape", &Game::InformationStateTensorShape)
      .def("information_state_tensor_layout",
           &Game::InformationStateTensorLayout)
      .def("information_state_tensor_size", &Game::InformationStateTensorSize)
      .def("observation_tensor_shape", &Game::ObservationTensorShape)
      .def("observation_tensor_layout", &Game::ObservationTensorLayout)
      .def("observation_tensor_size", &Game::ObservationTensorSize)
      .def("policy_tensor_shape", &Game::PolicyTensorShape)
      .def("deserialize_state", &Game::DeserializeState)
      .def("max_game_length", &Game::MaxGameLength)
      .def("__str__", &Game::ToString)
      .def("__eq__",
           [](const Game& value, Game* value2) {
             return value2 && value.ToString() == value2->ToString();
           })
      .def(py::pickle(                            // Pickle support
          [](std::shared_ptr<const Game> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            // Have to remove the const here for this to compile, presumably
            // because the holder type is non-const. But seems like you can't
            // set the holder type to std::shared_ptr<const Game> either.
            return std::const_pointer_cast<Game>(LoadGame(data));
          }));

  py::class_<NormalFormGame, std::shared_ptr<NormalFormGame>> normal_form_game(
      m, "NormalFormGame", game);
  normal_form_game.def(py::pickle(                      // Pickle support
      [](std::shared_ptr<const NormalFormGame> game) {  // __getstate__
        return game->ToString();
      },
      [](const std::string& data) {  // __setstate__
        // Have to remove the const here for this to compile, presumably
        // because the holder type is non-const. But seems like you can't
        // set the holder type to std::shared_ptr<const Game> either.
        return std::const_pointer_cast<NormalFormGame>(
            std::static_pointer_cast<const NormalFormGame>(LoadGame(data)));
      }));

  py::class_<MatrixGame, std::shared_ptr<MatrixGame>> matrix_game(
      m, "MatrixGame", normal_form_game);
  matrix_game
      .def(py::init<GameType, GameParameters, std::vector<std::string>,
                    std::vector<std::string>, std::vector<double>,
                    std::vector<double>>())
      .def(py::init<GameType, GameParameters, std::vector<std::string>,
                    std::vector<std::string>,
                    const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<double>>&>())
      .def("num_rows", &MatrixGame::NumRows)
      .def("num_cols", &MatrixGame::NumCols)
      .def("row_utility", &MatrixGame::RowUtility)
      .def("col_utility", &MatrixGame::ColUtility)
      .def("player_utility", &MatrixGame::PlayerUtility)
      .def("row_utilities",
           [](const MatrixGame& game) {
             const std::vector<double>& row_utilities = game.RowUtilities();
             return py::array_t<double>({game.NumRows(), game.NumCols()},
                                        &row_utilities[0]);
           })
      .def("col_utilities",
           [](const MatrixGame& game) {
             const std::vector<double>& col_utilities = game.ColUtilities();
             return py::array_t<double>({game.NumRows(), game.NumCols()},
                                        &col_utilities[0]);
           })
      .def("player_utilities",
           [](const MatrixGame& game, const Player player) {
             const std::vector<double>& player_utilities =
                 game.PlayerUtilities(player);
             return py::array_t<double>({game.NumRows(), game.NumCols()},
                                        &player_utilities[0]);
           })
      .def("row_action_name", &MatrixGame::RowActionName)
      .def("col_action_name", &MatrixGame::ColActionName)
      .def(py::pickle(                                  // Pickle support
          [](std::shared_ptr<const MatrixGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            // Have to remove the const here for this to compile, presumably
            // because the holder type is non-const. But seems like you can't
            // set the holder type to std::shared_ptr<const Game> either.
            return std::const_pointer_cast<MatrixGame>(
                algorithms::LoadMatrixGame(data));
          }));

  py::class_<TensorGame, std::shared_ptr<TensorGame>> tensor_game(
      m, "TensorGame", normal_form_game);
  tensor_game
      .def(py::init<GameType, GameParameters,
                    std::vector<std::vector<std::string>>,
                    std::vector<std::vector<double>>>())
      .def("shape", &TensorGame::Shape)
      .def("player_utility", &TensorGame::PlayerUtility)
      .def("player_utilities",
           [](const TensorGame& game, const Player player) {
             const std::vector<double>& utilities =
                 game.PlayerUtilities(player);
             return py::array_t<double>(game.Shape(), &utilities[0]);
           })
      .def("action_name", &TensorGame::ActionName)
      .def(py::pickle(                                  // Pickle support
          [](std::shared_ptr<const TensorGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            // Have to remove the const here for this to compile, presumably
            // because the holder type is non-const. But seems like you can't
            // set the holder type to std::shared_ptr<const Game> either.
            return std::const_pointer_cast<TensorGame>(
                algorithms::LoadTensorGame(data));
          }));

  py::class_<Bot, PyBot> bot(m, "Bot");
  bot.def(py::init<>())
      .def("step", &Bot::Step)
      .def("restart", &Bot::Restart)
      .def("restart_at", &Bot::RestartAt)
      .def("provides_force_action", &Bot::ProvidesForceAction)
      .def("force_action", &Bot::ForceAction)
      .def("inform_action", &Bot::InformAction)
      .def("provides_policy", &Bot::ProvidesPolicy)
      .def("get_policy", &Bot::GetPolicy)
      .def("step_with_policy", &Bot::StepWithPolicy);

  py::class_<algorithms::Evaluator, std::shared_ptr<algorithms::Evaluator>>
      mcts_evaluator(m, "Evaluator");
  py::class_<algorithms::RandomRolloutEvaluator, algorithms::Evaluator,
             std::shared_ptr<algorithms::RandomRolloutEvaluator>>(
      m, "RandomRolloutEvaluator")
      .def(py::init<int, int>(), py::arg("n_rollouts"), py::arg("seed"));

  py::enum_<algorithms::ChildSelectionPolicy>(m, "ChildSelectionPolicy")
      .value("UCT", algorithms::ChildSelectionPolicy::UCT)
      .value("PUCT", algorithms::ChildSelectionPolicy::PUCT);

  py::class_<algorithms::MCTSBot, Bot>(m, "MCTSBot")
      .def(py::init<const Game&, std::shared_ptr<Evaluator>, double, int,
                    int64_t, bool, int, bool,
                    ::open_spiel::algorithms::ChildSelectionPolicy>(),
           py::arg("game"), py::arg("evaluator"), py::arg("uct_c"),
           py::arg("max_simulations"), py::arg("max_memory_mb"),
           py::arg("solve"), py::arg("seed"), py::arg("verbose"),
           py::arg("child_selection_policy") =
               algorithms::ChildSelectionPolicy::UCT)
      .def("step", &algorithms::MCTSBot::Step)
      .def("mcts_search", &algorithms::MCTSBot::MCTSearch);

  py::class_<TabularBestResponse>(m, "TabularBestResponse")
      .def(py::init<const open_spiel::Game&, int,
                    const std::unordered_map<std::string,
                                             open_spiel::ActionsAndProbs>&>())
      .def(py::init<const open_spiel::Game&, int, const open_spiel::Policy*>())
      .def("value", &TabularBestResponse::Value)
      .def("get_best_response_policy",
           &TabularBestResponse::GetBestResponsePolicy)
      .def("get_best_response_actions",
           &TabularBestResponse::GetBestResponseActions)
      .def("set_policy",
           static_cast<void (TabularBestResponse::*)(
               const std::unordered_map<std::string,
                                        open_spiel::ActionsAndProbs>&)>(
               &TabularBestResponse::SetPolicy))
      .def("set_policy",
           static_cast<void (TabularBestResponse::*)(const Policy*)>(
               &TabularBestResponse::SetPolicy));

  py::class_<open_spiel::Policy>(m, "Policy")
      .def("action_probabilities",
           (std::unordered_map<Action, double> (Policy::*)(
               const open_spiel::State&) const) &
               open_spiel::Policy::GetStatePolicyAsMap)
      .def("get_state_policy_as_map",
           (std::unordered_map<Action, double> (Policy::*)(const std::string&)
                const) &
               open_spiel::Policy::GetStatePolicyAsMap);

  // A tabular policy represented internally as a map. Note that this
  // implementation is not directly compatible with the Python TabularPolicy
  // implementation; the latter is implemented as a table of size
  // [num_states, num_actions], while this is implemented as a map. It is
  // non-trivial to convert between the two, but we have a function that does so
  // in the open_spiel/python/policy.py file.
  py::class_<open_spiel::TabularPolicy, open_spiel::Policy>(m, "TabularPolicy")
      .def(py::init<const std::unordered_map<std::string, ActionsAndProbs>&>())
      .def("get_state_policy", &open_spiel::TabularPolicy::GetStatePolicy)
      .def("policy_table",
           static_cast<const std::unordered_map<std::string, ActionsAndProbs>& (
               TabularPolicy::*)() const>(
               &open_spiel::TabularPolicy::PolicyTable));
  m.def("UniformRandomPolicy", &open_spiel::GetUniformPolicy);

  py::class_<open_spiel::algorithms::CFRSolver>(m, "CFRSolver")
      .def(py::init<const Game&>())
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy", &open_spiel::algorithms::CFRSolver::AveragePolicy);

  py::class_<open_spiel::algorithms::CFRPlusSolver>(m, "CFRPlusSolver")
      .def(py::init<const Game&>())
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRPlusSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy",
           &open_spiel::algorithms::CFRPlusSolver::AveragePolicy);

  py::class_<open_spiel::algorithms::CFRBRSolver>(m, "CFRBRSolver")
      .def(py::init<const Game&>())
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRPlusSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy",
           &open_spiel::algorithms::CFRPlusSolver::AveragePolicy);

  py::class_<open_spiel::algorithms::TrajectoryRecorder>(m,
                                                         "TrajectoryRecorder")
      .def(py::init<const Game&, const std::unordered_map<std::string, int>&,
                    int>())
      .def("record_batch",
           &open_spiel::algorithms::TrajectoryRecorder::RecordBatch);

  m.def("hulh_game_string", &open_spiel::HulhGameString);
  m.def("hunl_game_string", &open_spiel::HunlGameString);
  m.def("create_matrix_game",
        static_cast<std::shared_ptr<const MatrixGame> (*)(
            const std::string&, const std::string&,
            const std::vector<std::string>&, const std::vector<std::string>&,
            const std::vector<std::vector<double>>&,
            const std::vector<std::vector<double>>&)>(
            &open_spiel::matrix_game::CreateMatrixGame),
        "Creates an arbitrary matrix game from named rows/cols and utilities.");

  m.def("create_matrix_game", static_cast<std::shared_ptr<const MatrixGame> (*)(
                                  const std::vector<std::vector<double>>&,
                                  const std::vector<std::vector<double>>&)>(
                                  &open_spiel::matrix_game::CreateMatrixGame),
        "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def("create_tensor_game", static_cast<std::shared_ptr<const TensorGame> (*)(
                                  const std::string&, const std::string&,
                                  const std::vector<std::vector<std::string>>&,
                                  const std::vector<std::vector<double>>&)>(
                                  &open_spiel::tensor_game::CreateTensorGame),
        "Creates an arbitrary tensor game from named actions and utilities.");

  m.def("create_matrix_game", static_cast<std::shared_ptr<const MatrixGame> (*)(
                                  const std::vector<std::vector<double>>&,
                                  const std::vector<std::vector<double>>&)>(
                                  &open_spiel::matrix_game::CreateMatrixGame),
        "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def("create_tensor_game",
        static_cast<std::shared_ptr<const TensorGame> (*)(
            const std::vector<std::vector<double>>&, const std::vector<int>&)>(
            &open_spiel::tensor_game::CreateTensorGame),
        "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def(
      "create_tensor_game",
      [](const std::vector<py::array_t<
             double, py::array::c_style | py::array::forcecast>>& utilities) {
        const int num_players = utilities.size();
        const std::vector<int> shape(
            utilities[0].shape(), utilities[0].shape() + utilities[0].ndim());
        std::vector<std::vector<double>> flat_utilities;
        for (const auto& player_utilities : utilities) {
          SPIEL_CHECK_EQ(player_utilities.ndim(), num_players);
          SPIEL_CHECK_TRUE(
              std::equal(shape.begin(), shape.end(), player_utilities.shape()));
          flat_utilities.push_back(std::vector<double>(
              player_utilities.data(),
              player_utilities.data() + player_utilities.size()));
        }
        return open_spiel::tensor_game::CreateTensorGame(flat_utilities, shape);
      },
      "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def("load_game",
        static_cast<std::shared_ptr<const Game> (*)(const std::string&)>(
            &open_spiel::LoadGame),
        "Returns a new game object for the specified short name using default "
        "parameters");

  m.def("load_game",
        static_cast<std::shared_ptr<const Game> (*)(
            const std::string&, const GameParameters&)>(&open_spiel::LoadGame),
        "Returns a new game object for the specified short name using given "
        "parameters");

  m.def("load_game_as_turn_based",
        static_cast<std::shared_ptr<const Game> (*)(const std::string&)>(
            &open_spiel::LoadGameAsTurnBased),
        "Converts a simultaneous game into an turn-based game with infosets.");

  m.def("load_game_as_turn_based",
        static_cast<std::shared_ptr<const Game> (*)(const std::string&,
                                                    const GameParameters&)>(
            &open_spiel::LoadGameAsTurnBased),
        "Converts a simultaneous game into an turn-based game with infosets.");

  m.def("load_matrix_game", open_spiel::algorithms::LoadMatrixGame,
        "Loads a game as a matrix game (will fail if not a matrix game.");

  m.def("load_tensor_game", open_spiel::algorithms::LoadTensorGame,
        "Loads a game as a tensor game (will fail if not a tensor game.");

  m.def("load_efg_game", open_spiel::efg_game::LoadEFGGame,
        "Load a gambit extensive form game from data.");
  m.def("get_sample_efg_data", open_spiel::efg_game::GetSampleEFGData,
        "Get Kuhn poker EFG data.");
  m.def("get_kuhn_poker_efg_data", open_spiel::efg_game::GetKuhnPokerEFGData,
        "Get sample EFG data.");

  m.def("extensive_to_matrix_game",
        open_spiel::algorithms::ExtensiveToMatrixGame,
        "Converts a two-player extensive-game to its equivalent matrix game, "
        "which is exponentially larger. Use only with small games.");

  m.def("extensive_to_tensor_game", open_spiel::ExtensiveToTensorGame,
        "Converts an extensive-game to its equivalent tensor game, "
        "which is exponentially larger. Use only with small games.");

  m.def("registered_names", GameRegisterer::RegisteredNames,
        "Returns the names of all available games.");

  m.def("registered_games", GameRegisterer::RegisteredGames,
        "Returns the details of all available games.");

  m.def("evaluate_bots", open_spiel::EvaluateBots, py::arg("state"),
        py::arg("bots"), py::arg("seed"),
        "Plays a single game with the given bots and returns the final "
        "utilities.");

  m.def("make_uniform_random_bot", open_spiel::MakeUniformRandomBot,
        "A uniform random bot, for test purposes.");

  m.def("make_stateful_random_bot", open_spiel::MakeStatefulRandomBot,
        "A stateful random bot, for test purposes.");

  m.def("serialize_game_and_state", open_spiel::SerializeGameAndState,
        "A general implementation of game and state serialization.");

  m.def("deserialize_game_and_state", open_spiel::DeserializeGameAndState,
        "A general implementation of deserialization of a game and state "
        "string serialized by serialize_game_and_state.");

  m.def("exploitability",
        static_cast<double (*)(const Game&, const Policy&)>(&Exploitability),
        "Returns the sum of the utility that a best responder wins when when "
        "playing against 1) the player 0 policy contained in `policy` and 2) "
        "the player 1 policy contained in `policy`."
        "This only works for two player, zero- or constant-sum sequential "
        "games, and raises a SpielFatalError if an incompatible game is passed "
        "to it.");

  m.def("exploitability",
        static_cast<double (*)(
            const Game&,
            const std::unordered_map<std::string, ActionsAndProbs>&)>(
            &Exploitability),
        "Returns the sum of the utility that a best responder wins when when "
        "playing against 1) the player 0 policy contained in `policy` and 2) "
        "the player 1 policy contained in `policy`."
        "This only works for two player, zero- or constant-sum sequential "
        "games, and raises a SpielFatalError if an incompatible game is passed "
        "to it.");

  m.def("nash_conv",
        static_cast<double (*)(const Game&, const Policy&)>(&NashConv),
        "Returns the sum of the utility that a best responder wins when when "
        "playing against 1) the player 0 policy contained in `policy` and 2) "
        "the player 1 policy contained in `policy`."
        "This only works for two player, zero- or constant-sum sequential "
        "games, and raises a SpielFatalError if an incompatible game is passed "
        "to it.");

  m.def(
      "nash_conv",
      static_cast<double (*)(
          const Game&,
          const std::unordered_map<std::string, ActionsAndProbs>&)>(&NashConv),
      "Calculates a measure of how far the given policy is from a Nash "
      "equilibrium by returning the sum of the improvements in the value "
      "that each player could obtain by unilaterally changing their strategy "
      "while the opposing player maintains their current strategy (which "
      "for a Nash equilibrium, this value is 0).");

  m.def("convert_to_turn_based", open_spiel::ConvertToTurnBased,
        "Returns a turn-based version of the given game.");

  m.def("num_deterministic_policies",
        open_spiel::algorithms::NumDeterministicPolicies,
        "Returns number of determinstic policies in this game for a player, "
        "or -1 if there are more than 2^64 - 1 policies.");

  m.def("expected_returns",
        static_cast<std::vector<double> (*)(
            const State&, const std::vector<const Policy*>&, int, bool)>(
            &open_spiel::algorithms::ExpectedReturns),
        "Computes the undiscounted expected returns from a depth-limited "
        "search.");

  py::class_<open_spiel::algorithms::BatchedTrajectory>(m, "BatchedTrajectory")
      .def(py::init<int>())
      .def_readwrite("observations",
                     &open_spiel::algorithms::BatchedTrajectory::observations)
      .def_readwrite("state_indices",
                     &open_spiel::algorithms::BatchedTrajectory::state_indices)
      .def_readwrite("legal_actions",
                     &open_spiel::algorithms::BatchedTrajectory::legal_actions)
      .def_readwrite("actions",
                     &open_spiel::algorithms::BatchedTrajectory::actions)
      .def_readwrite(
          "player_policies",
          &open_spiel::algorithms::BatchedTrajectory::player_policies)
      .def_readwrite("player_ids",
                     &open_spiel::algorithms::BatchedTrajectory::player_ids)
      .def_readwrite("rewards",
                     &open_spiel::algorithms::BatchedTrajectory::rewards)
      .def_readwrite("valid", &open_spiel::algorithms::BatchedTrajectory::valid)
      .def_readwrite(
          "next_is_terminal",
          &open_spiel::algorithms::BatchedTrajectory::next_is_terminal)
      .def_readwrite("batch_size",
                     &open_spiel::algorithms::BatchedTrajectory::batch_size)
      .def_readwrite(
          "max_trajectory_length",
          &open_spiel::algorithms::BatchedTrajectory::max_trajectory_length)
      .def("resize_fields",
           &open_spiel::algorithms::BatchedTrajectory::ResizeFields);

  m.def("record_batched_trajectories",
        static_cast<open_spiel::algorithms::BatchedTrajectory (*)(
            const Game&, const std::vector<open_spiel::TabularPolicy>&,
            const std::unordered_map<std::string, int>&, int, bool, int, int)>(
            &open_spiel::algorithms::RecordBatchedTrajectory),
        "Records a batch of trajectories.");

  // Game-Specific Query API.
  m.def("negotiation_item_pool", &open_spiel::query::NegotiationItemPool);
  m.def("negotiation_agent_utils", &open_spiel::query::NegotiationAgentUtils);

  // Set an error handler that will raise exceptions. These exceptions are for
  // the Python interface only. When used from C++, OpenSpiel will never raise
  // exceptions - the process will be terminated instead.
  open_spiel::SetErrorHandler(
      [](const std::string& string) { throw SpielException(string); });
}  // NOLINT(readability/fn_size)

}  // namespace
}  // namespace open_spiel
