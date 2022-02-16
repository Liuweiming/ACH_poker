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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_H_

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/threaded_unordered_map.h"

namespace open_spiel {
namespace algorithms {

// A basic structure to store the relevant quantities.
struct CFRInfoStateValues {
  CFRInfoStateValues() {}
  CFRInfoStateValues(std::vector<Action> la, double init_value)
      : legal_actions(la),
        cumulative_regrets(la.size(), init_value),
        true_regrets(la.size(), init_value),
        cumulative_policy(la.size(), init_value),
        current_policy(la.size(), 1.0 / la.size()) {}
  CFRInfoStateValues(std::vector<Action> la) : CFRInfoStateValues(la, 0) {}

  virtual ~CFRInfoStateValues() {}

  virtual void ApplyRegretMatching();  // Fills current_policy.
  void ApplyRegretMatchingUsingMax();
  void ApplyHedge(double eta);
  void ApplyRegretMatchingUsingEpsilonMax(double epsilon = 0.06);
  virtual void ApplyEpsilonGreedy(double epsilon = 0.1);
  void ApplyRegretMatchingImp(const std::vector<double>& regrets);
  void ApplyRegretMatchingUsingMaxImp(const std::vector<double>& regrets,
                                      double epsilon = 0.0);
  void ApplyEpsilonGreedyImp(const std::vector<double>& regrets,
                             double epsilon = 0.1);
  bool empty() const { return legal_actions.empty(); }
  int num_actions() const { return legal_actions.size(); }

  // A string representation of the information state values
  std::string ToString() const;

  // Samples from current policy using randomly generated z, adding epsilon
  // exploration (mixing in uniform).
  int SampleActionIndex(double epsilon, double z);
  virtual void SetRegret(const std::vector<double>& regret);
  virtual void SetPolicy(const std::vector<double>& policy);
  virtual void SetCumulatePolicy(const std::vector<double>& policy);
  // virtual void CumulateRegret(const std::vector<double>& regret);
  // virtual void CumulatePolicy(const std::vector<double>& policy);

  std::vector<Action> legal_actions;
  std::vector<double> cumulative_regrets;
  std::vector<double> true_regrets;
  std::vector<double> cumulative_policy;
  std::vector<double> current_policy;
};

// A type for tables holding CFR values.
using CFRInfoStateValuesTable =
    std::unordered_map<std::string, CFRInfoStateValues>;

// A policy that extracts the average policy from the CFR table values,
// which can be passed to tabular exploitability.
class CFRAveragePolicy : public Policy {
 public:
  // Returns the average policy from the CFR values.
  // If a state/info state is not found, return the default policy for the
  // state/info state (or an empty policy if default_policy is nullptr).
  // If an info state has zero cumulative regret for all actions,
  // return a uniform policy.
  CFRAveragePolicy(const CFRInfoStateValuesTable& info_states,
                   std::shared_ptr<Policy> default_policy);
  ActionsAndProbs GetStatePolicy(const State& state) const override;
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;

 private:
  const CFRInfoStateValuesTable& info_states_;
  bool default_to_uniform_;
  std::shared_ptr<Policy> default_policy_;
  void GetStatePolicyFromInformationStateValues(
      const CFRInfoStateValues& is_vals,
      ActionsAndProbs* actions_and_probs) const;
};

// A policy that extracts the current policy from the CFR table values.
class CFRCurrentPolicy : public Policy {
 public:
  // Returns the current policy from the CFR values. If a default policy is
  // passed in, then it means that it is used if the lookup fails (use nullptr
  // to not use a default policy).
  CFRCurrentPolicy(const CFRInfoStateValuesTable& info_states,
                   std::shared_ptr<Policy> default_policy);
  ActionsAndProbs GetStatePolicy(const State& state) const override;
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;

 private:
  const CFRInfoStateValuesTable& info_states_;
  std::shared_ptr<Policy> default_policy_;
  ActionsAndProbs GetStatePolicyFromInformationStateValues(
      const CFRInfoStateValues& is_vals,
      ActionsAndProbs& actions_and_probs) const;
};

// Base class supporting different flavours of the Counterfactual Regret
// Minimization (CFR) algorithm.
//
// see https://webdocs.cs.ualberta.ca/~bowling/papers/07nips-regretpoker.pdf
// and http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
//
// The implementation is similar to the Python version:
//   open_spiel/python/algorithms/cfr.py
//
// The algorithm computes an approximate Nash policy for 2 player zero-sum
// games.
//
// CFR can be view as a policy iteration algorithm. Importantly, the policies
// themselves do not converge to a Nash policy, but their average does.
//
class CFRSolverBase {
 public:
  CFRSolverBase(const Game& game, bool alternating_updates,
                bool linear_averaging, bool regret_matching_plus);
  virtual ~CFRSolverBase() = default;

  // Performs one step of the CFR algorithm.
  virtual void EvaluateAndUpdatePolicy();

  // Computes the average policy, containing the policy for all players.
  // The returned policy instance should only be used during the lifetime of
  // the CFRSolver object.
  std::unique_ptr<Policy> AveragePolicy() const {
    return std::unique_ptr<Policy>(new CFRAveragePolicy(info_states_, nullptr));
  }

  // Computes the current policy, containing the policy for all players.
  // The returned policy instance should only be used during the lifetime of
  // the CFRSolver object.
  std::unique_ptr<Policy> CurrentPolicy() const {
    return std::unique_ptr<Policy>(new CFRCurrentPolicy(info_states_, nullptr));
  }

 protected:
  const Game& game_;

  // Iteration to support linear_policy.
  int iteration_ = 0;
  CFRInfoStateValuesTable info_states_;
  const std::unique_ptr<State> root_state_;
  const std::vector<double> root_reach_probs_;

  // Compute the counterfactual regret and update the average policy for the
  // specified player.
  // The optional `policy_overrides` can be used to specify for each player a
  // policy to use instead of the current policy. `policy_overrides=nullptr`
  // will disable this feature. Otherwise it should be a [num_players] vector,
  // and if `policy_overrides[p] != nullptr` it will be used instead of the
  // current policy. This feature exists to support CFR-BR.
  std::vector<double> ComputeCounterFactualRegret(
      const State& state, const absl::optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities,
      const std::vector<const Policy*>* policy_overrides);

  // Update the current policy for all information states.
  void ApplyRegretMatching();

 private:
  std::vector<double> ComputeCounterFactualRegretForActionProbs(
      const State& state, const absl::optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities, const int current_player,
      const std::vector<double>& info_state_policy,
      const std::vector<Action>& legal_actions,
      std::vector<double>* child_values_out,
      const std::vector<const Policy*>* policy_overrides);

  void InitializeInfostateNodes(const State& state);

  // Fills `info_state_policy` to be a [num_actions] vector of the probabilities
  // found in `policy` at the given `info_state`.
  void GetInfoStatePolicyFromPolicy(std::vector<double>* info_state_policy,
                                    const std::vector<Action>& legal_actions,
                                    const Policy* policy,
                                    const std::string& info_state) const;

  // Get the policy at this information state. The probabilities are ordered in
  // the same order as legal_actions.
  std::vector<double> GetPolicy(const std::string& info_state,
                                const std::vector<Action>& legal_actions);

  void ApplyRegretMatchingPlusReset();

  std::vector<double> RegretMatching(const std::string& info_state,
                                     const std::vector<Action>& legal_actions);

  bool AllPlayersHaveZeroReachProb(
      const std::vector<double>& reach_probabilities) const;

  const bool regret_matching_plus_;
  const bool alternating_updates_;
  const bool linear_averaging_;

  const int chance_player_;
};

// Standard CFR implementation.
//
// See https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf
class CFRSolver : public CFRSolverBase {
 public:
  explicit CFRSolver(const Game& game)
      : CFRSolverBase(game,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/false,
                      /*regret_matching_plus=*/false) {}
};

// CFR+ implementation.
//
// See https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf
//
// CFR+ is CFR with the following modifications:
// - use Regret Matching+ instead of Regret Matching.
// - use alternating updates instead of simultaneous updates.
// - use linear averaging.
class CFRPlusSolver : public CFRSolverBase {
 public:
  CFRPlusSolver(const Game& game)
      : CFRSolverBase(game,
                      /*alternating_updates=*/true,
                      /*linear_averaging=*/true,
                      /*regret_matching_plus=*/true) {}
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_CFR_H_
