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

#include "vpevaluator.h"

#include <chrono>
#include <cstdint>
#include <memory>

#include "absl/hash/hash.h"
#include "open_spiel/utils/stats.h"

namespace open_spiel {
namespace algorithms {

VPNetEvaluator::VPNetEvaluator(DeviceManager *device_manager,
                               bool value_or_policy, Player player_,
                               int batch_size, int threads, int cache_size,
                               int cache_shards, int device_id)
    : device_manager_(*device_manager),
      device_id_(device_id),
      value_or_policy(value_or_policy),
      player_(player_),
      batch_size_(batch_size),
      queue_(batch_size * threads * 100),
      batch_size_hist_(batch_size + 1) {
  if (cache_size != 0) {
    cache_shards = std::max(1, cache_shards);
    cache_.reserve(cache_shards);
    for (int i = 0; i < cache_shards; ++i) {
      cache_.push_back(
          std::unique_ptr<LRUCache<std::string, CFRNetModel::InferenceOutputs>>(
              new LRUCache<std::string, CFRNetModel::InferenceOutputs>(
                  cache_size / cache_shards)));
    }
  }
  if (batch_size_ <= 1) {
    threads = 0;
  }
  inference_threads_.reserve(threads);
  for (int i = 0; i < threads; ++i) {
    inference_threads_.emplace_back(
        [this, i]() { this->Runner(device_id_ + i); });
  }
}

VPNetEvaluator::~VPNetEvaluator() {
  stop_.Stop();
  queue_.BlockNewValues();
  queue_.Clear();
  for (auto &t : inference_threads_) {
    t.join();
  }
}

void VPNetEvaluator::ClearCache() {
  for (auto &c : cache_) {
    c->Clear();
  }
}

LRUCacheInfo VPNetEvaluator::CacheInfo() {
  LRUCacheInfo info;
  for (auto &c : cache_) {
    info += c->Info();
  }
  return info;
}

CFRNetModel::InferenceOutputs VPNetEvaluator::Inference(const State &state) {
  CFRNetModel::InferenceInputs inputs = {state.InformationStateString(),
                                         state.LegalActions(),
                                         state.InformationStateTensor()};
  return Inference(state.CurrentPlayer(), inputs);
}

CFRNetModel::InferenceOutputs VPNetEvaluator::Inference(
    Player player, const CFRNetModel::InferenceInputs &inputs) {
  // NOTE: check if we use the correct evaluator.
  SPIEL_CHECK_EQ(player, player_);
  uint64_t key;
  int cache_shard;
  if (!cache_.empty()) {
    key = absl::Hash<std::string>{}(inputs.info_str);
    cache_shard = key % cache_.size();
    absl::optional<const CFRNetModel::InferenceOutputs> opt_outputs =
        cache_[cache_shard]->Get(inputs.info_str);
    if (opt_outputs) {
      return *opt_outputs;
    }
  }
  CFRNetModel::InferenceOutputs outputs;
  if (batch_size_ <= 1) {
    if (value_or_policy) {
      outputs =
          device_manager_.Get(1, device_id_)
              ->InfValue(player_,
                         std::vector<CFRNetModel::InferenceInputs>{inputs})[0];
    } else {
      outputs =
          device_manager_.Get(1, device_id_)
              ->InfPolicy(player_,
                          std::vector<CFRNetModel::InferenceInputs>{inputs})[0];
    }
  } else {
    std::promise<CFRNetModel::InferenceOutputs> prom;
    std::future<CFRNetModel::InferenceOutputs> fut = prom.get_future();
    queue_.Push(QueueItem{inputs, &prom});
    outputs = fut.get();
  }
  if (!cache_.empty()) {
    cache_[cache_shard]->Set(inputs.info_str, outputs);
  }
  return outputs;
}

void VPNetEvaluator::SetCFRTabular(const State &state,
                                   const std::vector<double> &value) {
  SetCFRTabular(CFRNetModel::TrainInputs{
      state.InformationStateString(), state.LegalActions(),
      state.LegalActions(), state.InformationStateTensor(), value, 1.0});
}

void VPNetEvaluator::LearnCFRTabular(
    const std::vector<CFRNetModel::TrainInputs> &inputs) {
  for (auto &device : device_manager_.GetAll()) {
    if (value_or_policy) {
      device->TrainValueTabular(player_, inputs);
    } else {
      device->TrainPolicyTabular(player_, inputs);
    }
  }
}

void VPNetEvaluator::SetCFRTabular(const CFRNetModel::TrainInputs &input) {
  for (auto &device : device_manager_.GetAll()) {
    if (value_or_policy) {
      device->SetCFRTabularValue(input);
    } else {
      device->SetCFRTabularPolicy(input);
    }
  }
}

void VPNetEvaluator::AccumulateCFRTabular(const State &state,
                                          const std::vector<double> &value) {
  AccumulateCFRTabular(CFRNetModel::TrainInputs{
      state.InformationStateString(), state.LegalActions(),
      state.LegalActions(), state.InformationStateTensor(), value, 1.0});
}

void VPNetEvaluator::AccumulateCFRTabular(
    const CFRNetModel::TrainInputs &input) {
  for (auto &device : device_manager_.GetAll()) {
    if (value_or_policy) {
      device->AccumulateCFRTabularValue(input);
    } else {
      device->AccumulateCFRTabularPolicy(input);
    }
  }
}

std::vector<double> VPNetEvaluator::GetCFRTabular(const State &state) {
  return GetCFRTabular(CFRNetModel::InferenceInputs{
      state.InformationStateString(), state.LegalActions(),
      state.InformationStateTensor()});
}

std::vector<double> VPNetEvaluator::GetCFRTabular(
    const CFRNetModel::InferenceInputs &input) {
  if (value_or_policy) {
    return device_manager_.Get(1, device_id_)->GetCFRTabularValue(input);
  } else {
    return device_manager_.Get(1, device_id_)->GetCFRTabularPolicy(input);
  }
}

void VPNetEvaluator::Runner(int device_id) {
  std::vector<CFRNetModel::InferenceInputs> inputs;
  std::vector<std::promise<CFRNetModel::InferenceOutputs> *> promises;
  inputs.reserve(batch_size_);
  promises.reserve(batch_size_);
  while (!stop_.StopRequested()) {
    {
      // Only one thread at a time should be listening to the queue to maximize
      // batch size and minimize latency.
      std::unique_lock<std::mutex> lock(inference_queue_m_);
      TimePoint deadline = std::chrono::time_point_cast<Duration>(
          std::chrono::system_clock::time_point::max());
      for (int i = 0; i < batch_size_; ++i) {
        absl::optional<QueueItem> item = queue_.Pop(deadline);
        if (!item) {  // Hit the deadline.
          break;
        }
        if (inputs.empty()) {
          deadline = std::chrono::time_point_cast<Duration>(
                         std::chrono::system_clock::now()) +
                     std::chrono::milliseconds(1);
        }
        inputs.push_back(item->inputs);
        promises.push_back(item->prom);
      }
    }

    if (inputs.empty()) {  // Almost certainly StopRequested.
      continue;
    }

    {
      std::unique_lock<std::mutex> lock(stats_m_);
      batch_size_stats_.Add(inputs.size());
      batch_size_hist_.Add(inputs.size());
    }
    std::vector<CFRNetModel::InferenceOutputs> outputs;
    if (value_or_policy) {
      outputs = device_manager_.Get(inputs.size(), device_id)
                    ->InfValue(player_, inputs);
    } else {
      outputs = device_manager_.Get(inputs.size(), device_id)
                    ->InfPolicy(player_, inputs);
    }
    for (int i = 0; i < promises.size(); ++i) {
      promises[i]->set_value(outputs[i]);
    }
    inputs.clear();
    promises.clear();
  }
}

void VPNetEvaluator::ResetBatchSizeStats() {
  std::unique_lock<std::mutex> lock(stats_m_);
  batch_size_stats_.Reset();
  batch_size_hist_.Reset();
}

open_spiel::BasicStats VPNetEvaluator::BatchSizeStats() {
  std::unique_lock<std::mutex> lock(stats_m_);
  return batch_size_stats_;
}

open_spiel::HistogramNumbered VPNetEvaluator::BatchSizeHistogram() {
  std::unique_lock<std::mutex> lock(stats_m_);
  return batch_size_hist_;
}

}  // namespace algorithms
}  // namespace open_spiel
