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

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/types/optional.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

constexpr const int kSerializationVersion = 1;
constexpr const char* kSerializeMetaSectionHeader = "[Meta]";
constexpr const char* kSerializeGameSectionHeader = "[Game]";
constexpr const char* kSerializeStateSectionHeader = "[State]";

// Returns the available parameter keys, to be used as a utility function.
std::string ListValidParameters(
    const std::map<std::string, GameParameter>& param_spec) {
  std::vector<std::string> available_keys;
  available_keys.reserve(param_spec.size());
  for (const auto& item : param_spec) {
    available_keys.push_back(item.first);
  }
  std::sort(available_keys.begin(), available_keys.end());
  return absl::StrJoin(available_keys, ", ");
}

// Check on supplied parameters for game creation.
// Issues a SpielFatalError if any are missing, of the wrong type, or
// unexpectedly present.
void ValidateParams(const GameParameters& params,
                    const std::map<std::string, GameParameter>& param_spec) {
  // Check all supplied parameters are supported and of the right type.
  for (const auto& param : params) {
    const auto it = param_spec.find(param.first);
    if (it == param_spec.end()) {
      SpielFatalError(absl::StrCat("Unknown parameter ", param.first,
                                   ". Available parameters are: ",
                                   ListValidParameters(param_spec)));
    }
    if (it->second.type() != param.second.type()) {
      SpielFatalError(absl::StrCat(
          "Wrong type for parameter ", param.first, ". Expected type: ",
          GameParameterTypeToString(it->second.type()), ", got ",
          GameParameterTypeToString(param.second.type()), " with ",
          param.second.ToString()));
    }
  }
  // Check we aren't missing any mandatory parameters.
  for (const auto& param : param_spec) {
    if (param.second.is_mandatory() && !params.count(param.first)) {
      SpielFatalError(absl::StrCat("Missing parameter ", param.first));
    }
  }
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const StateType& type) {
  switch (type) {
    case StateType::kChance: {
      os << "CHANCE";
      break;
    }
    case StateType::kDecision: {
      os << "DECISION";
      break;
    }
    case StateType::kTerminal: {
      os << "TERMINAL";
      break;
    }
  }
  return os;
}

StateType State::GetType() const {
  if (IsChanceNode()) {
    return StateType::kChance;
  } else if (IsTerminal()) {
    return StateType::kTerminal;
  } else {
    return StateType::kDecision;
  }
}

bool GameType::ContainsRequiredParameters() const {
  for (const auto& key_val : parameter_specification) {
    if (key_val.second.is_mandatory()) {
      return true;
    }
  }
  return false;
}

GameRegisterer::GameRegisterer(const GameType& game_type, CreateFunc creator) {
  RegisterGame(game_type, creator);
}

std::shared_ptr<const Game> GameRegisterer::CreateByName(
    const std::string& short_name, const GameParameters& params) {
  auto iter = factories().find(short_name);
  if (iter == factories().end()) {
    SpielFatalError(absl::StrCat("Unknown game '", short_name,
                                 "'. Available games are:\n",
                                 absl::StrJoin(RegisteredNames(), "\n")));

  } else {
    ValidateParams(params, iter->second.first.parameter_specification);
    return (iter->second.second)(params);
  }
}

std::vector<std::string> GameRegisterer::RegisteredNames() {
  std::vector<std::string> names;
  for (const auto& key_val : factories()) {
    names.push_back(key_val.first);
  }
  return names;
}

std::vector<GameType> GameRegisterer::RegisteredGames() {
  std::vector<GameType> games;
  for (const auto& key_val : factories()) {
    games.push_back(key_val.second.first);
  }
  return games;
}

bool GameRegisterer::IsValidName(const std::string& short_name) {
  return factories().find(short_name) != factories().end();
}

void GameRegisterer::RegisterGame(const GameType& game_info,
                                  GameRegisterer::CreateFunc creator) {
  factories()[game_info.short_name] = std::make_pair(game_info, creator);
}

bool IsGameRegistered(const std::string& short_name) {
  return GameRegisterer::IsValidName(short_name);
}

std::vector<std::string> RegisteredGames() {
  return GameRegisterer::RegisteredNames();
}

std::vector<GameType> RegisteredGameTypes() {
  return GameRegisterer::RegisteredGames();
}

std::shared_ptr<const Game> LoadGame(const std::string& game_string) {
  return LoadGame(GameParametersFromString(game_string));
}

std::shared_ptr<const Game> LoadGame(const std::string& short_name,
                                     const GameParameters& params) {
  std::shared_ptr<const Game> result =
      GameRegisterer::CreateByName(short_name, params);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat("Unable to create game: ", short_name));
  }
  return result;
}

std::shared_ptr<const Game> LoadGame(GameParameters params) {
  auto it = params.find("name");
  if (it == params.end()) {
    SpielFatalError(absl::StrCat("No 'name' parameter in params: ",
                                 GameParametersToString(params)));
  }
  std::string name = it->second.string_value();
  params.erase(it);
  std::shared_ptr<const Game> result =
      GameRegisterer::CreateByName(name, params);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat("Unable to create game: ", name));
  }
  return result;
}

State::State(std::shared_ptr<const Game> game)
    : num_distinct_actions_(game->NumDistinctActions()),
      num_players_(game->NumPlayers()),
      game_(game) {}

template <>
GameParameters Game::ParameterValue<GameParameters>(
    const std::string& key,
    absl::optional<GameParameters> default_value) const {
  auto iter = game_parameters_.find(key);
  if (iter != game_parameters_.end()) {
    return iter->second.game_value();
  }

  if (default_value == absl::nullopt) {
    std::vector<std::string> available_keys;
    for (auto const& element : game_parameters_) {
      available_keys.push_back(element.first);
    }
    SpielFatalError(absl::StrCat("The parameter for ", key,
                                 " is missing. Available keys are: ",
                                 absl::StrJoin(available_keys, " ")));
  }
  return default_value.value();
}

template <>
int Game::ParameterValue<int>(const std::string& key,
                              absl::optional<int> default_value) const {
  auto iter = game_parameters_.find(key);
  if (iter == game_parameters_.end()) {
    GameParameter default_game_parameter;
    if (default_value != absl::nullopt) {
      default_game_parameter = GameParameter(default_value.value());
    } else {
      auto default_iter = game_type_.parameter_specification.find(key);
      if (default_iter == game_type_.parameter_specification.end()) {
        SpielFatalError(absl::StrCat("No default parameter for ", key,
                                     " and it was not provided as an argument. "
                                     "It is likely it should be mandatory."));
      }
      default_game_parameter = default_iter->second;
    }
    defaulted_parameters_[key] = default_game_parameter;
    return default_game_parameter.int_value();
  } else {
    return iter->second.int_value();
  }
}

template <>
double Game::ParameterValue<double>(
    const std::string& key, absl::optional<double> default_value) const {
  auto iter = game_parameters_.find(key);
  if (iter == game_parameters_.end()) {
    GameParameter default_game_parameter;
    if (default_value != absl::nullopt) {
      default_game_parameter = GameParameter(default_value.value());
    } else {
      auto default_iter = game_type_.parameter_specification.find(key);
      if (default_iter == game_type_.parameter_specification.end()) {
        SpielFatalError(absl::StrCat("No default parameter for ", key,
                                     " and it was not provided as an argument. "
                                     "It is likely it should be mandatory."));
      }
      default_game_parameter = default_iter->second;
    }
    defaulted_parameters_[key] = default_game_parameter;
    return default_game_parameter.double_value();
  } else {
    return iter->second.double_value();
  }
}

template <>
std::string Game::ParameterValue<std::string>(
    const std::string& key, absl::optional<std::string> default_value) const {
  auto iter = game_parameters_.find(key);
  if (iter == game_parameters_.end()) {
    GameParameter default_game_parameter;
    if (default_value != absl::nullopt) {
      default_game_parameter = GameParameter(default_value.value());
    } else {
      auto default_iter = game_type_.parameter_specification.find(key);
      if (default_iter == game_type_.parameter_specification.end()) {
        SpielFatalError(absl::StrCat("No default parameter for ", key,
                                     " and it was not provided as an argument. "
                                     "It is likely it should be mandatory."));
      }
      default_game_parameter = default_iter->second;
    }
    defaulted_parameters_[key] = default_game_parameter;
    return default_game_parameter.string_value();
  } else {
    return iter->second.string_value();
  }
}

template <>
bool Game::ParameterValue<bool>(const std::string& key,
                                absl::optional<bool> default_value) const {
  auto iter = game_parameters_.find(key);
  if (iter == game_parameters_.end()) {
    GameParameter default_game_parameter;
    if (default_value != absl::nullopt) {
      default_game_parameter = GameParameter(default_value.value());
    } else {
      auto default_iter = game_type_.parameter_specification.find(key);
      if (default_iter == game_type_.parameter_specification.end()) {
        SpielFatalError(absl::StrCat("No default parameter for ", key,
                                     " and it was not provided as an argument. "
                                     "It is likely it should be mandatory."));
      }
      default_game_parameter = default_iter->second;
    }
    defaulted_parameters_[key] = default_game_parameter;
    return default_game_parameter.bool_value();
  } else {
    return iter->second.bool_value();
  }
}

void NormalizePolicy(ActionsAndProbs* policy) {
  double sum = 0;
  for (const std::pair<Action, double>& outcome : *policy) {
    sum += outcome.second;
  }
  for (std::pair<Action, double>& outcome : *policy) {
    outcome.second /= sum;
  }
}

std::pair<Action, double> SampleAction(const ActionsAndProbs& outcomes,
                                       std::mt19937& rng) {
  static std::uniform_real_distribution<double> gen(0.0, 1.0);
  return SampleAction(outcomes, gen(rng));
}
std::pair<Action, double> SampleAction(const ActionsAndProbs& outcomes,
                                       double z) {
  SPIEL_CHECK_GE(z, 0);
  SPIEL_CHECK_LT(z, 1);

  // First do a check that this is indeed a proper discrete distribution.
  double sum = 0;
  for (const std::pair<Action, double>& outcome : outcomes) {
    double prob = outcome.second;
    SPIEL_CHECK_GE(prob, 0);
    SPIEL_CHECK_LE(prob, 1);
    sum += prob;
  }
  SPIEL_CHECK_FLOAT_EQ(sum, 1.0);

  // Now sample an outcome.
  sum = 0;
  for (const std::pair<Action, double>& outcome : outcomes) {
    double prob = outcome.second;
    if (sum <= z && z < (sum + prob)) {
      return outcome;
    }
    sum += prob;
  }

  // If we get here, something has gone wrong
  std::cerr << "Chance sampling failed; outcomes:" << std::endl;
  for (const std::pair<Action, double>& outcome : outcomes) {
    std::cerr << outcome.first << "  " << outcome.second << std::endl;
  }
  SpielFatalError(
      absl::StrCat("Internal error: failed to sample an outcome; z=", z));
}

std::string State::Serialize() const {
  // This simple serialization doesn't work for games with sampled chance
  // nodes, since the history doesn't give us enough information to reconstruct
  // the state. If you wish to serialize states in such games, you must
  // implement custom serialization and deserialization for the state.
  SPIEL_CHECK_NE(game_->GetType().chance_mode,
                 GameType::ChanceMode::kSampledStochastic);
  return absl::StrCat(absl::StrJoin(History(), "\n"), "\n");
}

Action State::StringToAction(Player player,
                             const std::string& action_str) const {
  for (const Action action : LegalActions()) {
    if (action_str == ActionToString(player, action)) return action;
  }
  SpielFatalError(
      absl::StrCat("Couldn't find an action matching ", action_str));
}

std::unique_ptr<State> Game::DeserializeState(const std::string& str) const {
  // This simple deserialization doesn't work for games with sampled chance
  // nodes, since the history doesn't give us enough information to reconstruct
  // the state. If you wish to serialize states in such games, you must
  // implement custom serialization and deserialization for the state.
  SPIEL_CHECK_NE(game_type_.chance_mode,
                 GameType::ChanceMode::kSampledStochastic);

  std::unique_ptr<State> state = NewInitialState();
  if (str.length() == 0) {
    return state;
  }
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  for (int i = 0; i < lines.size(); ++i) {
    if (state->IsSimultaneousNode()) {
      std::vector<Action> actions;
      for (int p = 0; p < state->NumPlayers(); ++p, ++i) {
        SPIEL_CHECK_LT(i, lines.size());
        Action action = static_cast<Action>(std::stol(lines[i]));
        actions.push_back(action);
      }
      state->ApplyActions(actions);
      // Must decrement i here, otherwise it is incremented too many times.
      --i;
    } else {
      Action action = static_cast<Action>(std::stol(lines[i]));
      state->ApplyAction(action);
    }
  }
  return state;
}

std::string SerializeGameAndState(const Game& game, const State& state) {
  std::string str = "";

  // Meta section.
  absl::StrAppend(&str,
                  "# Automatically generated by OpenSpiel "
                  "SerializeGameAndState\n");
  absl::StrAppend(&str, kSerializeMetaSectionHeader, "\n");
  absl::StrAppend(&str, "Version: ", kSerializationVersion, "\n");
  absl::StrAppend(&str, "\n");

  // Game section.
  absl::StrAppend(&str, kSerializeGameSectionHeader, "\n");
  absl::StrAppend(&str, game.ToString(), "\n");

  // State section.
  absl::StrAppend(&str, kSerializeStateSectionHeader, "\n");
  absl::StrAppend(&str, state.Serialize(), "\n");

  return str;
}

std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
DeserializeGameAndState(const std::string& serialized_state) {
  std::vector<std::string> lines = absl::StrSplit(serialized_state, '\n');

  enum Section { kInvalid = -1, kMeta = 0, kGame = 1, kState = 2 };
  std::vector<std::string> section_strings = {"", "", ""};
  Section cur_section = kInvalid;

  std::string game_string = "";
  std::string state_string = "";
  std::shared_ptr<const Game> game = nullptr;
  std::unique_ptr<State> state = nullptr;

  for (int i = 0; i < lines.size(); ++i) {
    if (lines[i].length() == 0 || lines[i].at(0) == '#') {
      // Skip comments and blank lines.
    } else if (lines[i] == kSerializeMetaSectionHeader) {
      SPIEL_CHECK_EQ(cur_section, kInvalid);
      cur_section = kMeta;
    } else if (lines[i] == kSerializeGameSectionHeader) {
      SPIEL_CHECK_EQ(cur_section, kMeta);
      cur_section = kGame;
    } else if (lines[i] == kSerializeStateSectionHeader) {
      SPIEL_CHECK_EQ(cur_section, kGame);
      cur_section = kState;
    } else {
      SPIEL_CHECK_NE(cur_section, kInvalid);
      absl::StrAppend(&section_strings[cur_section], lines[i], "\n");
    }
  }

  // Remove the trailing "\n" from the game and state sections.
  if (section_strings[kGame].length() > 0 &&
      section_strings[kGame].back() == '\n') {
    section_strings[kGame].pop_back();
  }
  if (section_strings[kState].length() > 0 &&
      section_strings[kState].back() == '\n') {
    section_strings[kState].pop_back();
  }

  // We currently just ignore the meta section.
  game = LoadGame(section_strings[kGame]);
  state = game->DeserializeState(section_strings[kState]);

  return std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>(
      game, std::move(state));
}

std::ostream& operator<<(std::ostream& stream, GameType::Dynamics value) {
  switch (value) {
    case GameType::Dynamics::kSimultaneous:
      return stream << "Simultaneous";
    case GameType::Dynamics::kSequential:
      return stream << "Sequential";
    default:
      SpielFatalError("Unknown dynamics.");
      return stream << "This will never return.";
  }
}

std::ostream& operator<<(std::ostream& stream, GameType::ChanceMode value) {
  switch (value) {
    case GameType::ChanceMode::kDeterministic:
      return stream << "Deterministic";
    case GameType::ChanceMode::kExplicitStochastic:
      return stream << "ExplicitStochastic";
    case GameType::ChanceMode::kSampledStochastic:
      return stream << "SampledStochastic";
    default:
      SpielFatalError("Unknown mode.");
      return stream << "This will never return.";
  }
}

std::ostream& operator<<(std::ostream& stream, GameType::Information value) {
  switch (value) {
    case GameType::Information::kOneShot:
      return stream << "OneShot";
    case GameType::Information::kPerfectInformation:
      return stream << "PerfectInformation";
    case GameType::Information::kImperfectInformation:
      return stream << "ImperfectInformation";
    default:
      SpielFatalError("Unknown value.");
      return stream << "This will never return.";
  }
}

std::ostream& operator<<(std::ostream& stream, GameType::Utility value) {
  switch (value) {
    case GameType::Utility::kZeroSum:
      return stream << "ZeroSum";
    case GameType::Utility::kConstantSum:
      return stream << "ConstantSum";
    case GameType::Utility::kGeneralSum:
      return stream << "GeneralSum";
    case GameType::Utility::kIdentical:
      return stream << "Identical";
    default:
      SpielFatalError("Unknown value.");
      return stream << "This will never return.";
  }
}

std::ostream& operator<<(std::ostream& stream, GameType::RewardModel value) {
  switch (value) {
    case GameType::RewardModel::kRewards:
      return stream << "Rewards";
    case GameType::RewardModel::kTerminal:
      return stream << "Terminal";
    default:
      SpielFatalError("Unknown value.");
      return stream << "This will never return.";
  }
}

std::string Game::ToString() const {
  GameParameters params = game_parameters_;
  params["name"] = GameParameter(game_type_.short_name);
  return GameParametersToString(params);
}

}  // namespace open_spiel
