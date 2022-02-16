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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_EXTERNAL_SAMPLING_MCCFR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_EXTERNAL_SAMPLING_MCCFR_H_

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/public_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "vpevaluator.h"
#include "vpnet.h"

// An implementation of external sampling Monte Carlo Counterfactual Regret
// Minimization (CFR). See Lanctot 2009 [0] and Chapter 4 of Lanctot 2013 [1]
// for details.
// [0]: http://mlanctot.info/files/papers/nips09mccfr.pdf
// [1]: http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf

namespace open_spiel {
namespace algorithms {

// How to average the strategy. The 'simple' type does the averaging for
// player i + 1 mod num_players on player i's regret update pass; in two players
// this corresponds to the standard implementation (updating the average
// policy at opponent nodes). In n>2 players, this can be a problem for several
// reasons: first, it does not compute the estimate as described by the
// (unbiased) stochastically-weighted averaging in chapter 4 of Lanctot 2013
// commonly used in MCCFR because the denominator (important sampling
// correction) should include all the other sampled players as well so the
// sample reach no longer cancels with reach of the player updating their
// average policy. Second, if one player assigns zero probability to an action
// (leading to a subtree), the average policy of a different player in that
// subtree is no longer updated. Hence, the full averaging does not update the
// average policy in the regret passes but does a separate pass to update the
// average policy. Nevertheless, we set the simple type as the default because
// it is faster, seems to work better empirically, and it matches what was done
// in Pluribus (Brown and Sandholm. Superhuman AI for multiplayer poker.
// Science, 11, 2019).

// NOTE: We can also just update the average strategy of current traversing
// player. So it is an other AverageType, named kCurrent.
enum class AverageType {
  kSimple,
  kFull,
  kCurrent,
};

struct ReplayNode {
  std::string info_str;
  std::vector<double> information;
  open_spiel::Player current_player;
  std::vector<open_spiel::Action> legal_actions;
  int action_index;
  double reward;
  double value;
  double weight;
  double old_policy;
  double adv;
  double value_target;
};

struct PolicyReplayNode {
  std::string info_str;
  std::vector<double> information;
  open_spiel::Player current_player;
  std::vector<open_spiel::Action> legal_actions;
  int action_index;
  double weight;
};

std::ostream& operator<<(std::ostream& out, const ReplayNode& node);

struct Trajectory {
  std::vector<ReplayNode> states;
};

// TODO: Do we need multithread Solver? For vanilla cfr, we can implement
// multithread cfr by parallelizing random branches. However, for MCCFR
// algorithms, random outcomes are sampled. But to parallel players actions is
// problematic. In fact, we can simply open multiple Solvers if the regret and
// strategy are approximated by neural networks. However, It is still
// problematic to shared strategy if the strategy is not approximated by a
// neural network.
class NeurdSolver {
 public:
  static constexpr double kInitialTableValues = 0.000001;

  // allow to get cfr value or policy by value_eval_.
  NeurdSolver(const Game& game, std::shared_ptr<VPNetEvaluator> value_0_eval_,
              std::shared_ptr<VPNetEvaluator> value_1_eval_,
              std::shared_ptr<VPNetEvaluator> policy_0_eval,
              std::shared_ptr<VPNetEvaluator> policy_1_eval,
              bool use_regret_net, bool use_policy_net, bool use_tabular,
              bool anticipatory, double alpha, double eta, double epsilon,
              bool symmetry, std::mt19937* rng,
              AverageType avg_type = AverageType::kSimple);
  // Performs one iteration of external sampling MCCFR, updating the regrets
  // and average strategy for all players. This method uses the internal random
  // number generator.
  std::pair<std::vector<Trajectory>, std::vector<Trajectory>> RunIteration();

  std::pair<Trajectory, Trajectory> RunIteration(Player player, double alpha,
                                                 int step);

  // Same as above, but uses the specified random number generator instead.
  std::pair<Trajectory, Trajectory> RunIteration(std::mt19937* rng,
                                                 Player player, double alpha,
                                                 int step);

  int NodeTouched() { return node_touch_; }

 protected:
  virtual double UpdateRegrets(PublicNode* node, Player player,
                               double player_reach, double oppoment_reach,
                               double ave_opponent_reach, double sampling_reach,
                               Trajectory& value_trajectory,
                               Trajectory& policy_trajectory, int step,
                               std::mt19937* rng, const ChanceData& chance_data,
                               bool current_or_average[2]);

  std::shared_ptr<const Game> game_;
  std::mt19937* rng_;
  uint32_t iterations_;
  AverageType avg_type_;
  std::uniform_real_distribution<double> dist_;
  std::vector<std::shared_ptr<VPNetEvaluator>> value_eval_;
  std::vector<std::shared_ptr<VPNetEvaluator>> policy_eval_;
  PublicTree tree_;
  PublicNode* root_node_;
  State* root_state_;
  bool use_regret_net;
  bool use_policy_net;
  bool use_tabular;
  bool joint_net;
  bool anticipatory_;
  double alpha_;
  double eta_;
  double epsilon_;
  bool symmetry_;
  int node_touch_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_EXTERNAL_SAMPLING_MCCFR_H_
