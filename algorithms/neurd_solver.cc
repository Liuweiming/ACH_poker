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

#include "neurd_solver.h"

#include <memory>
#include <numeric>
#include <random>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

std::ostream& operator<<(std::ostream& out, const ReplayNode& node) {
  out << node.information << " " << node.current_player << " "
      << node.legal_actions << " " << node.action_index << " " << node.value
      << " " << node.old_policy;
  return out;
}

NeurdSolver::NeurdSolver(const Game& game,
                         std::shared_ptr<VPNetEvaluator> value_0_eval,
                         std::shared_ptr<VPNetEvaluator> value_1_eval,
                         std::shared_ptr<VPNetEvaluator> policy_0_eval,
                         std::shared_ptr<VPNetEvaluator> policy_1_eval,
                         bool use_regret_net, bool use_policy_net,
                         bool use_tabular, bool anticipatory, double alpha,
                         double eta, double epsilon, bool symmetry,
                         std::mt19937* rng, AverageType avg_type)
    : game_(game.Clone()),
      rng_(rng),
      iterations_(0),
      avg_type_(avg_type),
      dist_(0.0, 1.0),
      value_eval_{value_0_eval, value_1_eval},
      policy_eval_{policy_0_eval, policy_1_eval},
      tree_(game_->NewInitialState()),
      root_node_(tree_.Root()),
      root_state_(root_node_->GetState()),
      use_regret_net(use_regret_net),
      use_policy_net(use_policy_net),
      use_tabular(use_tabular),
      anticipatory_(anticipatory),
      alpha_(alpha),
      eta_(eta),
      epsilon_(epsilon),
      symmetry_(symmetry) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}
std::pair<std::vector<Trajectory>, std::vector<Trajectory>>
NeurdSolver::RunIteration() {
  std::vector<Trajectory> value_trajectories(game_->NumPlayers());
  std::vector<Trajectory> policy_trajectories(game_->NumPlayers());
  for (auto p = Player{0}; p < game_->NumPlayers(); ++p) {
    auto ret_p = RunIteration(rng_, 1e-5, p, 0);
    value_trajectories.push_back(ret_p.first);
    policy_trajectories.push_back(ret_p.second);
  }
  return {value_trajectories, policy_trajectories};
}

std::pair<Trajectory, Trajectory> NeurdSolver::RunIteration(Player player,
                                                            double alpha,
                                                            int step) {
  return RunIteration(rng_, player, alpha, step);
}

std::pair<Trajectory, Trajectory> NeurdSolver::RunIteration(std::mt19937* rng,
                                                            Player player,
                                                            double alpha,
                                                            int step) {
  alpha_ = alpha;
  node_touch_ = 0;
  ++iterations_;
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  // Sample a chace seed at the start of an iteration.
  ChanceData chance_data = root_state_->SampleChance(rng);
  bool current_or_average[2] = {true, true};
  UpdateRegrets(root_node_, player, 1, 1, 1, 1, value_trajectory,
                policy_trajectory, step, rng, chance_data, current_or_average);
  return {value_trajectory, policy_trajectory};
}

double NeurdSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double ave_opponent_reach, double sampling_reach,
    Trajectory& value_trajectory, Trajectory& policy_trajectory, int step,
    std::mt19937* rng, const ChanceData& chance_data,
    bool current_or_average[2]) {
  State& state = *(node->GetState());
  universal_poker::UniversalPokerState* poker_state =
      static_cast<universal_poker::UniversalPokerState*>(node->GetState());
  state.SetChance(chance_data);
  // std::cout << state.ToString() << std::endl;
  if (state.IsTerminal()) {
    double value = state.PlayerReturn(player);
    value_trajectory.states.push_back(
        ReplayNode{state.InformationStateString(player),
                   state.InformationStateTensor(player), player,
                   std::vector<Action>{}, -1, value, 0.0, 1.0, 0.0, 0.0, 0.0});
    return value;
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(node->GetChild(action), player, player_reach,
                         opponent_reach, ave_opponent_reach, sampling_reach,
                         value_trajectory, policy_trajectory, step, rng,
                         chance_data, current_or_average);
  } else if (state.IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  node_touch_ += 1;

  Player cur_player = state.CurrentPlayer();
  std::string is_key = state.InformationStateString(cur_player);
  std::vector<Action> legal_actions = state.LegalActions();
  std::vector<double> information_tensor = state.InformationStateTensor();

  // NOTE: why we need a copy here? don't copy, just create one.
  CFRInfoStateValues current_info_state(legal_actions, kInitialTableValues);
  double current_value = 0;
  if (step != 1) {
    auto inf_output = policy_eval_[cur_player]->Inference(state).value;
    SPIEL_CHECK_EQ(inf_output.size(), legal_actions.size());
    current_info_state.SetPolicy(inf_output);
  }
  double value = 0;
  int aidx = 0;
  if (cur_player == player) {
    aidx = current_info_state.SampleActionIndex(0.0, dist_(*rng));
    double new_reach = current_info_state.current_policy[aidx] * player_reach;
    double new_sampling_reach =
        current_info_state.current_policy[aidx] * sampling_reach;
    value = UpdateRegrets(
        node->GetChild(legal_actions[aidx]), player, new_reach, opponent_reach,
        ave_opponent_reach, new_sampling_reach, value_trajectory,
        policy_trajectory, step, rng, chance_data, current_or_average);
  } else {
    aidx = current_info_state.SampleActionIndex(0.0, dist_(*rng));
    double new_reach = current_info_state.current_policy[aidx] * opponent_reach;
    double new_ave_reach = 1.0 / legal_actions.size() * ave_opponent_reach;
    double new_sampling_reach =
        current_info_state.current_policy[aidx] * sampling_reach;
    value = UpdateRegrets(
        node->GetChild(legal_actions[aidx]), player, player_reach, new_reach,
        new_ave_reach, new_sampling_reach, value_trajectory, policy_trajectory,
        step, rng, chance_data, current_or_average);
  }

  if (cur_player == player) {
    value_trajectory.states.push_back(ReplayNode{
        is_key, information_tensor, cur_player, legal_actions, aidx, 0,
        current_value, 1.0, current_info_state.current_policy[aidx], 0.0, 0.0});
    policy_trajectory.states.push_back(ReplayNode{
        is_key, information_tensor, cur_player, legal_actions, aidx, 0,
        current_value, 1.0, current_info_state.current_policy[aidx], 0.0, 0.0});
  }

  if (cur_player == player && current_or_average[cur_player]) {
    if (!use_policy_net || use_tabular) {
      std::vector<double> policy(legal_actions.size());
      for (int paidx = 0; paidx < legal_actions.size(); ++paidx) {
        policy[paidx] = current_info_state.current_policy[paidx] *
                        player_reach / sampling_reach;
      }
      policy_eval_[cur_player]->AccumulateCFRTabular(state, policy);
    }
  }
  return value;
}

}  // namespace algorithms
}  // namespace open_spiel
