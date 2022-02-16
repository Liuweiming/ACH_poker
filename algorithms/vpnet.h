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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_VPNET_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_VPNET_H_

#include <cstring>
#include <iostream>
#include <iterator>
#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/time/time.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/read_write_mutex.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace logic = open_spiel::universal_poker::logic;

namespace open_spiel {
namespace algorithms {

// Spawn a python interpreter to call export_vpnet.py.
// There are three options for nn_model: mlp, conv2d and resnet.
// The nn_width is the number of hidden units for the mlp, and filters for
// conv/resnet. The nn_depth is number of layers for all three.
bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& device,
                    const std::string& filename, std::string nn_model,
                    int nn_width, int nn_depth, bool verbose = false);

void CreateModel(const Game& game, double learning_rate, double weight_decay,
                 const std::string& model_path,
                 const std::string& model_name_prefix,
                 std::string regret_nn_model, std::string policy_nn_model,
                 int nn_width, int nn_depth, bool use_gpu = false);

template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, const std::pair<T, U>& v) {
  out << "{" << v.first << ", " << v.second << "}";
  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (int i = 0; i != v.size(); ++i) {
    out << v[i] << ((i == (v.size() - 1)) ? "" : " ");
  }
  out << "]";
  return out;
}

std::vector<double> PolicyNormalize(const std::vector<double>& weights);

std::vector<double> HedgeNormalize(const std::vector<double>& weights);

std::string LoadModel(const std::string& model_path, int num_threads,
                      const std::string& device,
                      tensorflow::Session* tf_session,
                      tensorflow::MetaGraphDef& meta_graph_def,
                      tensorflow::SessionOptions& tf_opts);

class CFRNetModel {
 public:
  struct InferenceInputs {
    std::string info_str;
    std::vector<Action> legal_actions;
    std::vector<double> informations;

    bool operator==(const InferenceInputs& o) const {
      return informations == o.informations;
    }

    template <typename H>
    friend H AbslHashValue(H h, const InferenceInputs& in) {
      return H::combine(std::move(h), in.info_str);
    }
    friend std::ostream& operator<<(std::ostream& io,
                                    const CFRNetModel::InferenceInputs& ti);
  };
  struct InferenceOutputs {
    std::vector<Action> legal_actions;
    std::vector<double> value;
    friend std::ostream& operator<<(std::ostream& io,
                                    const CFRNetModel::InferenceOutputs& ti);
  };

  struct TrainInputs {
    std::string info_str;
    std::vector<Action> legal_actions;
    std::vector<Action> masks;
    std::vector<double> informations;
    std::vector<double> value;
    double weight;
    friend std::ostream& operator<<(std::ostream& io,
                                    const CFRNetModel::TrainInputs& ti);
  };

  struct CFRInfoState {
    CFRInfoState() {}
    CFRInfoState(std::size_t size, double init_value)
        : cumulative_regrets(size, init_value),
          cumulative_policy(size, init_value) {}
    CFRInfoState(std::size_t size) : CFRInfoState(size, 0) {}

    std::size_t Size() { return cumulative_regrets.size(); }

    virtual ~CFRInfoState() {}

    std::vector<double> cumulative_regrets;
    std::vector<double> true_regrets;
    std::vector<double> cumulative_policy;
  };

  using CFRTabularEntry = std::vector<double>;
  using CFRInferenceStateValueTalbe =
      std::unordered_map<std::string, CFRTabularEntry>;

 private:
  class CFRTable {
   public:
    CFRTable(const Game& game);

    // Move only, not copyable.
    CFRTable(CFRTable&& other) = default;
    CFRTable& operator=(CFRTable&& other) = default;
    CFRTable(const CFRTable&) = delete;
    CFRTable& operator=(const CFRTable&) = delete;

    ~CFRTable() {}
    void init() { info_states_.clear(); }

    std::vector<InferenceOutputs> Inference(
        const std::vector<InferenceInputs>& inputs, bool use_target = false);

    std::vector<std::string> ParseInfoStr(const std::string& info_str) {
      std::vector<std::string> ret;
      auto dim = info_str.find(":");
      ret.push_back(info_str.substr(0, dim));
      ret.push_back(info_str.substr(dim));
      return ret;
    }

    // Training: do one (batch) step of neural net training
    double Learn(const std::vector<TrainInputs>& inputs);

    void SetValue(const TrainInputs& input, bool accumulate);

    std::vector<double> GetValue(const InferenceInputs& input,
                                 bool normalize = false);

    void SyncFrom(const CFRTable& from) {
      rw_mutex::UniqueWriteLock<rw_mutex::ReadWriteMutex> lock(m_);
      info_states_ = from.info_states_;
    }

   private:
    const universal_poker::UniversalPokerGame* game_;
    const universal_poker::acpc_cpp::ACPCGame* acpc_game_;
    const universal_poker::acpc_cpp::ACPCState* acpc_state_;
    logic::CardSet deck_;
    std::vector<logic::CardSet> player_outcomes_;
    std::unordered_map<std::string, std::size_t> player_outcome_index_;
    std::vector<logic::CardSet> board_outcomes_;
    std::unordered_map<std::string, std::size_t> board_outcome_index_;
    int num_outcomes_;
    int num_hole_cards_;
    int num_board_cards_;
    CFRInferenceStateValueTalbe info_states_;
    rw_mutex::ReadWriteMutex m_;
  };
  class CFRNet {
   public:
    CFRNet(const Game& game, const std::string& path,
           const std::string& model_path, const std::string& file_name,
           int num_threads = 1, const std::string& device = "/cpu:0",
           bool use_target_net = false);

    // Move only, not copyable.
    CFRNet(CFRNet&& other) = default;
    CFRNet& operator=(CFRNet&& other) = default;
    CFRNet(const CFRNet&) = delete;
    CFRNet& operator=(const CFRNet&) = delete;

    ~CFRNet() {}

    void init();

    std::vector<InferenceOutputs> Inference(
        const std::vector<InferenceInputs>& inputs, bool value_or_policy,
        bool use_target = false);

    // Training: do one (batch) step of neural net training
    double Learn(const std::vector<TrainInputs>& inputs,
                 const std::vector<double>& params);

    void SetFlat(const std::vector<float>& flat, bool use_target = false) {
      const Eigen::Map<const Eigen::ArrayXf> flat_array(flat.data(),
                                                        flat.size());
      SetFlatArray(flat_array, use_target);
    }

    void SetFlatArray(const Eigen::ArrayXf& flat, bool use_target = false);

    Eigen::ArrayXf GetFlatArray(bool use_target = false);

    std::vector<float> GetFlat(bool use_target = false) {
      auto flat = GetFlatArray(use_target);
      return std::vector<float>(flat.begin(), flat.end());
    }

    void SyncTarget(double moving_factor = 1) {
      if (moving_factor == 1) {
        SetFlatArray(GetFlatArray(), true);
      } else {
        auto curr_array = GetFlatArray();
        auto target_array = GetFlatArray(true);
        target_array *= (1 - moving_factor);
        target_array += moving_factor * curr_array;
        SetFlatArray(target_array, true);
      }
    }

    void SyncFrom(CFRNet& from) {
      SetFlat(from.GetFlat());
      // Note: we need to sync target net too.
      if (use_target_net) {
        SetFlat(from.GetFlat(true), true);
      }
    }

    std::string SaveCheckpoint(int step);
    void LoadCheckpoint(const std::string& path);

   private:
    std::string device_;
    std::string path_;
    std::string file_name_;
    // Store the full model metagraph file for writing python compatible
    // checkpoints.
    std::string model_meta_graph_contents_;

    int num_threads_;
    int flat_input_size_;
    int num_actions_;
    bool use_target_net;
    int flat_size_;

    // Inputs for inference & training separated to have different fixed sizes
    tensorflow::Session* tf_session_ = nullptr;
    tensorflow::MetaGraphDef meta_graph_def_;
    tensorflow::SessionOptions tf_opts_;
    tensorflow::Session* target_tf_session_ = nullptr;
    tensorflow::MetaGraphDef target_meta_graph_def_;
  };

 public:
  CFRNetModel(const Game& game, const std::string& path,
              const std::string& model_path, const std::string& file_name,
              int num_threads = 1, const std::string& device = "/cpu:0",
              bool use_value_net = false, bool seperate_value_net = true,
              bool use_policy_net = false, bool seperate_policy_net = true,
              bool use_target = false, bool joint_net = false);

  // Move only, not copyable.
  CFRNetModel(CFRNetModel&& other) = default;
  CFRNetModel& operator=(CFRNetModel&& other) = default;
  CFRNetModel(const CFRNetModel&) = delete;
  CFRNetModel& operator=(const CFRNetModel&) = delete;

  void Init() {
    if (value_net_0 != nullptr) {
      value_net_0->init();
    }
    if (value_net_1 != nullptr) {
      value_net_1->init();
    }
    if (policy_net_0 != nullptr) {
      policy_net_0->init();
    }
    if (policy_net_1 != nullptr) {
      policy_net_1->init();
    }
  }

  void InitValue(Player player) {
    if (use_value_net) {
      if (player == 0) {
        value_net_0->init();
      } else {
        value_net_1->init();
      }
    } else {
    }
  }

  void InitPolicy(Player player) {
    if (use_policy_net) {
      if (player == 0) {
        policy_net_0->init();
      } else {
        policy_net_1->init();
      }
    } else {
    }
  }

  void SetCFRTabularValue(const TrainInputs& input) {
    value_table->SetValue(input, false);
  }

  void SetCFRTabularPolicy(const TrainInputs& input) {
    policy_table->SetValue(input, false);
  }

  void AccumulateCFRTabularValue(const TrainInputs& input) {
    value_table->SetValue(input, true);
  }

  void AccumulateCFRTabularPolicy(const TrainInputs& input) {
    policy_table->SetValue(input, true);
  }

  std::vector<double> GetCFRTabularValue(const InferenceInputs& input) {
    return value_table->GetValue(input);
  }

  std::vector<double> GetCFRTabularPolicy(const InferenceInputs& input,
                                          bool normalize = true) {
    return policy_table->GetValue(input, normalize);
  }

  std::vector<CFRNetModel::InferenceOutputs> GetCFRTabularValues(
      Player player, const std::vector<InferenceInputs>& inputs) {
    std::vector<CFRNetModel::InferenceOutputs> outputs;
    for (auto& input : inputs) {
      InferenceOutputs output;
      output.legal_actions = input.legal_actions;
      output.value = value_table->GetValue(input);
      outputs.push_back(output);
    }
    return outputs;
  }

  std::vector<CFRNetModel::InferenceOutputs> GetCFRTabularPolicies(
      Player player, const std::vector<InferenceInputs>& inputs,
      bool normalize = true) {
    std::vector<CFRNetModel::InferenceOutputs> outputs;
    for (auto& input : inputs) {
      InferenceOutputs output;
      output.legal_actions = input.legal_actions;
      output.value = policy_table->GetValue(input, normalize);
      outputs.push_back(output);
    }
    return outputs;
  }

  std::vector<CFRNetModel::InferenceOutputs> InfValue(
      Player player, const std::vector<InferenceInputs>& inputs,
      bool use_target = false) {
    if (use_value_net) {
      if (player == 0) {
        return value_net_0->Inference(inputs, true, use_target);
      } else {
        return value_net_1->Inference(inputs, true, use_target);
      }
    }
    return value_table->Inference(inputs, use_target);
  }

  std::vector<CFRNetModel::InferenceOutputs> InfTargetValue(
      Player player, const std::vector<InferenceInputs>& inputs) {
    return InfValue(player, inputs, true);
  }

  std::vector<CFRNetModel::InferenceOutputs> InfPolicy(
      Player player, const std::vector<InferenceInputs>& inputs,
      bool normalize = true, bool use_target = false) {
    std::vector<CFRNetModel::InferenceOutputs> ret;
    if (use_policy_net) {
      if (player == 0) {
        ret = policy_net_0->Inference(inputs, false, use_target);
      } else {
        ret = policy_net_1->Inference(inputs, false, use_target);
      }
    } else {
      ret = policy_table->Inference(inputs, use_target);
    }
    if (normalize) {
      for (auto& io : ret) {
        io.value = HedgeNormalize(io.value);
      }
    }
    return ret;
  }

  std::vector<CFRNetModel::InferenceOutputs> InfTargetPolicy(
      Player player, const std::vector<InferenceInputs>& inputs) {
    return InfPolicy(player, inputs, false, true);
  }

  double TrainValueTabular(Player player,
                           const std::vector<TrainInputs>& inputs) {
    return value_table->Learn(inputs);
  }

  double TrainPolicyTabular(Player player,
                            const std::vector<TrainInputs>& inputs) {
    return policy_table->Learn(inputs);
  }

  double TrainValue(Player player, const std::vector<TrainInputs>& inputs,
                    const std::vector<double>& params) {
    if (use_value_net) {
      if (player == 0) {
        return value_net_0->Learn(inputs, params);
      } else {
        return value_net_1->Learn(inputs, params);
      }
    } else {
    }
    return 0;
  }

  double TrainPolicy(Player player, const std::vector<TrainInputs>& inputs,
                     const std::vector<double>& params) {
    if (use_policy_net) {
      if (player == 0) {
        return policy_net_0->Learn(inputs, params);
      } else {
        return policy_net_1->Learn(inputs, params);
      }
    } else {
    }
    return 0;
  }

  std::vector<float> GetFlat(Player player, bool value_or_policy) {
    if (value_or_policy) {
      if (use_value_net) {
        if (player == 0) {
          return value_net_0->GetFlat();
        } else {
          return value_net_1->GetFlat();
        }
      }
    } else if (use_policy_net) {
      if (player == 0) {
        return policy_net_0->GetFlat();
      } else {
        return policy_net_1->GetFlat();
      }
    }
    return std::vector<float>();
  }

  void SetFlat(Player player, const std::vector<float>& flat,
               bool value_or_policy) {
    if (value_or_policy) {
      if (use_value_net) {
        if (player == 0) {
          value_net_0->SetFlat(flat);
        } else {
          value_net_1->SetFlat(flat);
        }
      }
    } else if (use_policy_net) {
      if (player == 0) {
        policy_net_0->SetFlat(flat);
      } else {
        policy_net_1->SetFlat(flat);
      }
    }
  }

  void SyncValueFrom(Player player, CFRNetModel& from) {
    if (use_value_net) {
      if (player == 0) {
        value_net_0->SyncFrom(*(from.value_net_0));
      } else {
        value_net_1->SyncFrom(*(from.value_net_1));
      }
    }
    { value_table->SyncFrom(*(from.value_table)); }
  }

  void SyncPolicyFrom(Player player, CFRNetModel& from) {
    if (use_policy_net) {
      if (player == 0) {
        policy_net_0->SyncFrom(*(from.policy_net_0));
      } else {
        policy_net_1->SyncFrom(*(from.policy_net_1));
      }
    }
    { policy_table->SyncFrom(*(from.policy_table)); }
  }

  std::string SaveValue(Player player, int step) {
    if (use_value_net) {
      if (player == 0) {
        return value_net_0->SaveCheckpoint(step);
      } else {
        return value_net_1->SaveCheckpoint(step);
      }
    }
    return "";
  }
  void RestoreValue(Player player, const std::string& path) {
    if (use_value_net) {
      if (player == 0) {
        value_net_0->LoadCheckpoint(path);
      } else {
        value_net_1->LoadCheckpoint(path);
      }
    }
  }
  std::string SavePolicy(Player player, int step) {
    if (use_policy_net) {
      if (player == 0) {
        return policy_net_0->SaveCheckpoint(step);
      } else {
        return policy_net_1->SaveCheckpoint(step);
      }
    }
    return "";
  }
  void RestorePolicy(Player player, const std::string& path) {
    if (use_policy_net) {
      if (player == 0) {
        policy_net_0->LoadCheckpoint(path);
      } else {
        policy_net_1->LoadCheckpoint(path);
      }
    }
  }

  void SyncPolicy(Player player, double moving_factor = 1) {
    if (use_policy_net) {
      if (player == 0) {
        policy_net_0->SyncTarget(moving_factor);
      } else {
        policy_net_1->SyncTarget(moving_factor);
      }
    }
  }

  void SyncValue(Player player, double moving_factor = 1) {
    if (use_value_net) {
      if (player == 0) {
        value_net_0->SyncTarget(moving_factor);
      } else {
        value_net_1->SyncTarget(moving_factor);
      }
    }
  }

  ~CFRNetModel() {}

  const std::string Device() const { return device_; }

 private:
  std::string device_;
  std::string path_;
  bool use_value_net;
  bool use_policy_net;
  bool seperate_value_net;
  bool seperate_policy_net;
  bool joint_net;
  int num_threads_;
  std::shared_ptr<CFRNet> value_net_0;
  std::shared_ptr<CFRNet> value_net_1;
  std::shared_ptr<CFRNet> policy_net_0;
  std::shared_ptr<CFRNet> policy_net_1;
  std::shared_ptr<CFRTable> value_table;
  std::shared_ptr<CFRTable> policy_table;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_VPNET_H_
