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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_VPEVALUATOR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_VPEVALUATOR_H_

#include <future>  // NOLINT
#include <vector>

#include "absl/hash/hash.h"
#include "device_manager.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

class VPNetEvaluator {
 public:
  explicit VPNetEvaluator(DeviceManager* device_manager, bool value_or_policy,
                          Player player_, int batch_size, int threads,
                          int cache_size, int cache_shards = 1,
                          int device_id = -1);
  ~VPNetEvaluator();

  CFRNetModel::InferenceOutputs Inference(const State& state);
  CFRNetModel::InferenceOutputs Inference(
      Player player, const CFRNetModel::InferenceInputs& inputs);

  void SetCFRTabular(const State& state, const std::vector<double>& value);
  void SetCFRTabular(const CFRNetModel::TrainInputs& input);

  void LearnCFRTabular(const std::vector<CFRNetModel::TrainInputs>& inputs);

  void AccumulateCFRTabular(const State& state,
                            const std::vector<double>& value);
  void AccumulateCFRTabular(const CFRNetModel::TrainInputs& input);

  std::vector<double> GetCFRTabular(const State& state);
  std::vector<double> GetCFRTabular(const CFRNetModel::InferenceInputs& input);

  void ClearCache();
  LRUCacheInfo CacheInfo();

  void ResetBatchSizeStats();
  open_spiel::BasicStats BatchSizeStats();
  open_spiel::HistogramNumbered BatchSizeHistogram();

 private:
  void Runner(int device_id);

  DeviceManager& device_manager_;
  int device_id_;
  std::vector<
      std::unique_ptr<LRUCache<std::string, CFRNetModel::InferenceOutputs>>>
      cache_;
  const int batch_size_;

  struct QueueItem {
    CFRNetModel::InferenceInputs inputs;
    std::promise<CFRNetModel::InferenceOutputs>* prom;
  };
  bool value_or_policy;
  Player player_;
  ThreadedQueue<QueueItem> queue_;
  StopToken stop_;
  std::vector<Thread> inference_threads_;
  std::mutex inference_queue_m_;  // Only one thread at a time should pop.

  std::mutex stats_m_;
  open_spiel::BasicStats batch_size_stats_;
  open_spiel::HistogramNumbered batch_size_hist_;
};

class CFRPolicy : public Policy {
 public:
  CFRPolicy(CFRNetModel* model) : model_(model) {}
  CFRPolicy(VPNetEvaluator* eval_0, VPNetEvaluator* eval_1)
      : eval_0_(eval_0), eval_1_(eval_1) {}
  ~CFRPolicy(){};

  ActionsAndProbs GetStatePolicy(const State& state) const {
    ActionsAndProbs ap;
    CFRNetModel::InferenceInputs inputs = {state.InformationStateString(),
                                           state.LegalActions(),
                                           state.InformationStateTensor()};
    CFRNetModel::InferenceOutputs policy_out;
    if (eval_0_) {
      if (state.CurrentPlayer() == 0) {
        policy_out = eval_0_->Inference(0, inputs);
      } else {
        policy_out = eval_1_->Inference(1, inputs);
      }
    }
    if (model_) {
      policy_out = model_->InfPolicy(
          state.CurrentPlayer(),
          std::vector<CFRNetModel::InferenceInputs>{inputs})[0];
    }
    for (int i = 0; i != policy_out.legal_actions.size(); ++i) {
      ap.push_back({policy_out.legal_actions[i], policy_out.value[i]});
    }
    // std::cout << state.InformationStateString(state.CurrentPlayer()) << " "
    //           << policy_out.value << std::endl;
    return ap;
  }

 private:
  CFRNetModel* model_{nullptr};
  VPNetEvaluator* eval_0_{nullptr};
  VPNetEvaluator* eval_1_{nullptr};
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_VPEVALUATOR_H_
