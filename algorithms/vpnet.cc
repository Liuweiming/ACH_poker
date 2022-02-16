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

#include "vpnet.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/run_python.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorMap.h"

namespace tf = tensorflow;
using Tensor = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using TensorMap = Eigen::TensorMap<Tensor, Eigen::Aligned>;
using TensorBool = Eigen::Tensor<bool, 2, Eigen::RowMajor>;
using TensorMapBool = Eigen::TensorMap<TensorBool, Eigen::Aligned>;

namespace open_spiel {
namespace algorithms {

namespace {
long long C(int n, int r) {
  if (r > n - r) r = n - r;
  long long ans = 1;
  int i;

  for (i = 1; i <= r; i++) {
    ans *= n - r + i;
    ans /= i;
  }
  return ans;
}
}  // namespace

std::ostream& operator<<(std::ostream& io, const CFRNetModel::TrainInputs& ti) {
  io << ti.info_str << ": [" << ti.legal_actions << " " << ti.informations
     << " " << ti.value << "]";
  return io;
}

std::vector<double> PolicyNormalize(const std::vector<double>& weights) {
  std::vector<double> probs(weights);
  absl::c_for_each(probs, [](double& w) { w = (w > 0 ? w : 0); });
  const double normalizer = absl::c_accumulate(probs, 0.);
  absl::c_for_each(probs, [&probs, normalizer](double& w) {
    w = (normalizer == 0.0 ? 1.0 / probs.size() : w / normalizer);
  });
  return probs;
}

std::vector<double> HedgeNormalize(const std::vector<double>& weights) {
  double sum_positive_regrets = 0.0;
  std::vector<double> ret(weights.size());
  double max_reg = *std::max_element(weights.begin(), weights.end());
  for (int aidx = 0; aidx < weights.size(); ++aidx) {
    ret[aidx] = std::exp((weights[aidx] - max_reg));
    sum_positive_regrets += ret[aidx];
  }

  for (int aidx = 0; aidx < weights.size(); ++aidx) {
    ret[aidx] /= sum_positive_regrets;
  }
  return ret;
}

std::ostream& operator<<(std::ostream& io,
                         const CFRNetModel::InferenceInputs& ti) {
  io << ti.info_str << ": [" << ti.legal_actions << " " << ti.informations
     << "]";
  return io;
}
std::ostream& operator<<(std::ostream& io,
                         const CFRNetModel::InferenceOutputs& ti) {
  io << "[" << ti.value << "]";
  return io;
}

template <typename T>
void no_null_delete(T* ptr) {
  if (ptr != nullptr) {
    delete ptr;
  }
}

bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& device,
                    const std::string& filename, std::string nn_model,
                    int nn_width, int nn_depth, bool verbose) {
  // NOTE: we use the first dimention of tensor as tensor size.
  std::vector<std::string> parameters{
      "--num_actions",
      absl::StrCat(game.NumDistinctActions()),  //
      "--path",
      absl::StrCat("'", path, "'"),  //
      "--device",
      absl::StrCat("'", device, "'"),  //
      "--graph_def",
      filename,  //
      "--learning_rate",
      absl::StrCat(learning_rate),  //
      "--weight_decay",
      absl::StrCat(weight_decay),  //
      "--nn_model",
      nn_model,  //
      "--nn_depth",
      absl::StrCat(nn_depth),  //
      "--nn_width",
      absl::StrCat(nn_width),  //
      absl::StrCat("--verbose=", verbose ? "true" : "false"),
  };
  for (auto& ts : game.InformationStateTensorShape()) {
    parameters.insert(parameters.end(), {"--tensor_shape", absl::StrCat(ts)});
  }
  return RunPython("algorithms.export_vpnet", parameters);
}

void CreateModel(const Game& game, double learning_rate, double weight_decay,
                 const std::string& model_path,
                 const std::string& model_name_prefix,
                 std::string regret_nn_model, std::string policy_nn_model,
                 int nn_width, int nn_depth, bool use_gpu) {
  std::vector<std::string> model_names{"value_0", "value_1", "policy_0",
                                       "policy_1"};
  for (auto& model_name : model_names) {
    std::string nn_model = model_name.find("policy") == std::string::npos
                               ? regret_nn_model
                               : policy_nn_model;
    SPIEL_CHECK_TRUE(CreateGraphDef(
        game, learning_rate, weight_decay, model_path, "/cpu:0",
        absl::StrJoin({model_name_prefix, model_name, std::string("cpu.pb")},
                      "_"),
        nn_model, nn_width, nn_depth));
    if (use_gpu) {
      SPIEL_CHECK_TRUE(CreateGraphDef(
          game, learning_rate, weight_decay, model_path, "/gpu:0",
          absl::StrJoin({model_name_prefix, model_name, std::string("gpu.pb")},
                        "_"),
          nn_model, nn_width, nn_depth));
    }
  }
}

CFRNetModel::CFRNetModel(const Game& game, const std::string& path,
                         const std::string& model_path,
                         const std::string& file_name, int num_threads,
                         const std::string& device, bool use_value_net,
                         bool seperate_value_net, bool use_policy_net,
                         bool seperate_policy_net, bool use_target,
                         bool joint_net)
    : device_(device),
      path_(path),
      num_threads_(num_threads),
      use_value_net(use_value_net),
      use_policy_net(use_policy_net),
      joint_net(joint_net),
      seperate_value_net(seperate_value_net),
      seperate_policy_net(seperate_policy_net) {
  // Some assumptions that we can remove eventually. The value net returns
  // a single value in terms of player 0 and the game is assumed to be zero-sum,
  // so player 1 can just be -value.
  if (use_value_net) {
    value_net_0 = std::shared_ptr<CFRNet>(
        new CFRNet(game, path, model_path, file_name + "_value_0", num_threads_,
                   device, use_target));
    if (seperate_value_net) {
      value_net_1 = std::shared_ptr<CFRNet>(
          new CFRNet(game, path, model_path, file_name + "_value_1",
                     num_threads_, device, use_target));
    } else {
      value_net_1 = value_net_0;
    }
  }
  if (use_policy_net) {
    if (joint_net) {
      policy_net_0 = value_net_0;
      policy_net_1 = value_net_1;
    } else {
      policy_net_0 = std::shared_ptr<CFRNet>(
          new CFRNet(game, path, model_path, file_name + "_policy_0",
                     num_threads_, device, use_target));
      if (seperate_policy_net) {
        policy_net_1 = std::shared_ptr<CFRNet>(
            new CFRNet(game, path, model_path, file_name + "_policy_1",
                       num_threads_, device, use_target));
      } else {
        policy_net_1 = policy_net_0;
      }
    }
  }
  value_table = std::shared_ptr<CFRTable>(new CFRTable(game));
  policy_table = std::shared_ptr<CFRTable>(new CFRTable(game));

  SPIEL_CHECK_EQ(game.NumPlayers(), 2);
  SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);
}

CFRNetModel::CFRTable::CFRTable(const Game& game)
    : game_(static_cast<const universal_poker::UniversalPokerGame*>(&game)),
      acpc_game_(game_->GetACPCGame()),
      acpc_state_(new universal_poker::acpc_cpp::ACPCState(acpc_game_)),
      deck_(/*num_suits=*/acpc_game_->NumSuitsDeck(),
            /*num_ranks=*/acpc_game_->NumRanksDeck()) {
  num_hole_cards_ = acpc_game_->GetNbHoleCardsRequired();
  num_board_cards_ = acpc_game_->GetTotalNbBoardCards();
  player_outcomes_ = deck_.SampleCards(num_hole_cards_);
  for (std::size_t index = 0; index != player_outcomes_.size(); ++index) {
    auto player_outcome_array = player_outcomes_[index].ToCardArray();
    acpc_state_->SetHoleCards(0, &(player_outcome_array[0]),
                              player_outcome_array.size());
    std::string player_outcome_str = acpc_state_->HoleCardsToString(0);
    player_outcome_index_[player_outcome_str] = index;
  }

  if (num_board_cards_) {
    board_outcomes_ = deck_.SampleCards(num_board_cards_);
  }

  num_outcomes_ = player_outcomes_.size() * player_outcomes_.size();
  int num_cards = acpc_game_->NumSuitsDeck() * acpc_game_->NumRanksDeck();
}

std::vector<CFRNetModel::InferenceOutputs> CFRNetModel::CFRTable::Inference(
    const std::vector<InferenceInputs>& inputs, bool use_target) {
  // only one thread is allowed to edit the info_states_.
  rw_mutex::UniqueReadLock<rw_mutex::ReadWriteMutex> lock(m_);
  std::vector<InferenceOutputs> outputs;
  outputs.reserve(inputs.size());
  for (auto& input : inputs) {
    auto sp = ParseInfoStr(input.info_str);
    std::string hole_cards = sp.front();
    SPIEL_CHECK_TRUE(player_outcome_index_.find(hole_cards) !=
                     player_outcome_index_.end());
    std::size_t index = player_outcome_index_[hole_cards];
    std::string info_str = sp[1];
    std::size_t asize = input.legal_actions.size();
    // std::cout << "inference staring ";
    if (info_states_.find(info_str) != info_states_.end()) {
      CFRTabularEntry& info_value = info_states_[info_str];
      // SPIEL_CHECK_EQ(info_value.size(), player_outcomes_.size() * asize);
      outputs.emplace_back(InferenceOutputs{
          input.legal_actions,
          std::vector<double>(info_value.begin() + index * asize,
                              info_value.begin() + (index + 1) * asize)});
    } else {
      outputs.emplace_back(InferenceOutputs{
          input.legal_actions,
          std::vector<double>(input.legal_actions.size(), 0.0000001)});
    }
    // std::cout << "inference ending " << std::endl;
  }
  return outputs;
}

double CFRNetModel::CFRTable::Learn(const std::vector<TrainInputs>& inputs) {
  // only one thread is allowed to edit the info_states_.
  rw_mutex::UniqueWriteLock<rw_mutex::ReadWriteMutex> lock(m_);
  for (auto& input : inputs) {
    auto sp = ParseInfoStr(input.info_str);
    std::string hole_cards = sp.front();
    std::size_t index = player_outcome_index_[hole_cards];
    std::string info_str = sp[1];
    std::size_t asize = input.legal_actions.size();
    if (info_states_.find(info_str) == info_states_.end()) {
      info_states_.insert(
          {info_str,
           CFRTabularEntry(player_outcomes_.size() * asize, 0.0000001)});
    }
    CFRTabularEntry& info_value = info_states_[info_str];
    // SPIEL_CHECK_EQ(info_value.size(), player_outcomes_.size() * asize);
    // assume the learning rate is 0.001.
    for (int i = 0; i != input.legal_actions.size(); ++i) {
      info_value[index * asize + i] += 0.001 * (input.value[i] * input.weight -
                                                info_value[index * asize + i]);
    }
  }
  return 0.0;
}

void CFRNetModel::CFRTable::SetValue(const TrainInputs& input,
                                     bool accumulate) {
  rw_mutex::UniqueWriteLock<rw_mutex::ReadWriteMutex> lock(m_);
  // std::cout << "set value staring ";
  auto sp = ParseInfoStr(input.info_str);
  std::string hole_cards = sp.front();
  SPIEL_CHECK_TRUE(player_outcome_index_.find(hole_cards) !=
                   player_outcome_index_.end());
  std::size_t index = player_outcome_index_[hole_cards];
  std::string info_str = sp[1];
  std::size_t asize = input.legal_actions.size();
  if (info_states_.find(info_str) == info_states_.end()) {
    info_states_.insert(
        {info_str,
         CFRTabularEntry(player_outcomes_.size() * asize, 0.0000001)});
  }
  CFRTabularEntry& info_value = info_states_[info_str];
  // SPIEL_CHECK_EQ(info_value.size(), player_outcomes_.size() * asize);
  // assume the learning rate is 0.001.
  for (int i = 0; i != input.legal_actions.size(); ++i) {
    if (accumulate) {
      info_value[index * asize + i] += input.value[i] * input.weight;
    } else {
      info_value[index * asize + i] = input.value[i] * input.weight;
    }
  }
  // std::cout << "set value ending" << std::endl;
}

std::vector<double> CFRNetModel::CFRTable::GetValue(
    const InferenceInputs& input, bool normalize) {
  rw_mutex::UniqueReadLock<rw_mutex::ReadWriteMutex> lock(m_);
  auto sp = ParseInfoStr(input.info_str);
  std::string hole_cards = sp.front();
  SPIEL_CHECK_TRUE(player_outcome_index_.find(hole_cards) !=
                   player_outcome_index_.end());
  std::size_t index = player_outcome_index_[hole_cards];
  std::string info_str = sp[1];
  std::size_t asize = input.legal_actions.size();
  // std::cout << "inference staring ";
  std::vector<double> ret_value(asize, 0.0000001);
  if (info_states_.find(info_str) != info_states_.end()) {
    CFRTabularEntry& info_value = info_states_[info_str];
    // SPIEL_CHECK_EQ(info_value.size(), player_outcomes_.size() * asize);
    ret_value = std::vector<double>(info_value.begin() + index * asize,
                                    info_value.begin() + (index + 1) * asize);
  }
  if (normalize) {
    return PolicyNormalize(ret_value);
  } else {
    return ret_value;
  }
  // std::cout << "get value ending " << std::endl;
}

std::string LoadModel(const std::string& model_path, int num_threads,
                      const std::string& device, tf::Session** tf_session,
                      tf::MetaGraphDef& meta_graph_def,
                      tf::SessionOptions& tf_opts) {
  std::cout << "loading model " << model_path << std::endl;
  TF_CHECK_OK(ReadBinaryProto(tf::Env::Default(), model_path, &meta_graph_def));

  // tf::graph::SetDefaultDevice(device, meta_graph_def.mutable_graph_def());
  // auto* graph_def = meta_graph_def.mutable_graph_def();
  // if (!device.empty()) {
  //   for (int i = 0; i < graph_def->node_size(); ++i) {
  //     graph_def->mutable_node(i)->set_device("/device:GPU:0");
  //   }
  // }
  // if (device.find("gpu") != std::string::npos) {
  //   std::string gpu_index = device.substr(std::string("/gpu:").size());
  //   tf_opts.config.mutable_gpu_options()->set_visible_device_list(gpu_index);
  // } else {
  //   tf_opts.config.mutable_gpu_options()->set_visible_device_list("");
  // }

  if ((*tf_session) != nullptr) {
    TF_CHECK_OK((*tf_session)->Close());
  }

  // create a new session
  tf_opts.config.set_intra_op_parallelism_threads(num_threads);
  tf_opts.config.set_inter_op_parallelism_threads(2);
  tf_opts.config.set_use_per_session_threads(true);
  tf_opts.config.set_allow_soft_placement(true);
  tf_opts.config.mutable_gpu_options()->set_allow_growth(true);

  TF_CHECK_OK(NewSession(tf_opts, tf_session));

  // Load graph into session
  TF_CHECK_OK((*tf_session)->Create(meta_graph_def.graph_def()));
  return model_path;
}

CFRNetModel::CFRNet::CFRNet(const Game& game, const std::string& path,
                            const std::string& model_path,
                            const std::string& file_name, int num_threads,
                            const std::string& device, bool use_target_net)
    : path_(path),
      file_name_(file_name),
      num_threads_(num_threads),
      device_(device),
      flat_input_size_(game.InformationStateTensorSize()),
      num_actions_(game.NumDistinctActions()),
      use_target_net(use_target_net) {
  if (device.find("gpu") != std::string::npos) {
    file_name_ += "_gpu";
  } else {
    file_name_ += "_cpu";
  }
  std::string load_path = absl::StrCat(model_path, "/", file_name_, ".pb");
  model_meta_graph_contents_ = file::File(load_path, "r").ReadContents();
  LoadModel(load_path, num_threads_, device, &tf_session_, meta_graph_def_,
            tf_opts_);
  // Initialize our variables
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(tf_session_->Run(run_opt, {}, {}, {"init_all_vars_op"}, nullptr,
                               nullptr));
  if (use_target_net) {
    LoadModel(load_path, num_threads_, device, &target_tf_session_,
              target_meta_graph_def_, tf_opts_);
    TF_CHECK_OK(target_tf_session_->Run(run_opt, {}, {}, {"init_all_vars_op"},
                                        nullptr, nullptr));
    SyncTarget();
  }
  auto flat = GetFlatArray();
  flat_size_ = flat.size();
}

void CFRNetModel::CFRNet::init() {
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  tf_session_->Run(run_opt, {}, {}, {"init_all_vars_op"}, nullptr, nullptr);
}

std::string CFRNetModel::CFRNet::SaveCheckpoint(int step) {
  std::string full_path = absl::StrCat(path_, "/checkpoint-", file_name_, step);
  tf::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
  checkpoint_path.scalar<tf::tstring>()() = full_path;
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(tf_session_->Run(
      run_opt,
      {{meta_graph_def_.saver_def().filename_tensor_name(), checkpoint_path}},
      {}, {meta_graph_def_.saver_def().save_tensor_name()}, nullptr, nullptr));
  // Writing a checkpoint from python writes the metagraph file, but c++
  // doesn't, so do it manually to make loading checkpoints easier.
  file::File(absl::StrCat(full_path, ".meta"), "w")
      .Write(model_meta_graph_contents_);
  return full_path;
}
void CFRNetModel::CFRNet::LoadCheckpoint(const std::string& path) {
  tf::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
  checkpoint_path.scalar<tf::tstring>()() = path;
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(tf_session_->Run(
      run_opt,
      {{meta_graph_def_.saver_def().filename_tensor_name(), checkpoint_path}},
      {}, {meta_graph_def_.saver_def().restore_op_name()}, nullptr, nullptr));
}

void CFRNetModel::CFRNet::SetFlatArray(const Eigen::ArrayXf& flat,
                                       bool use_target) {
  tf::Session* session = tf_session_;
  if (use_target) {
    session = target_tf_session_;
  }
  tf::Tensor flat_input(tf::DT_FLOAT, tf::TensorShape({flat.size()}));
  auto flat_input_vec = flat_input.vec<float>();
  for (int i = 0; i < flat.size(); ++i) {
    flat_input_vec(i) = flat(i);
  }
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(session->Run(run_opt, {{"set_flat_input", flat_input}}, {},
                           {"set_flat"}, nullptr, nullptr));
}

Eigen::ArrayXf CFRNetModel::CFRNet::GetFlatArray(bool use_target) {
  tf::Session* session = tf_session_;
  if (use_target) {
    session = target_tf_session_;
  }
  std::vector<tf::Tensor> tf_output;
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(session->Run(run_opt, {}, {"get_flat"}, {}, &tf_output, nullptr));
  auto tf_output_vec = tf_output[0].vec<float>();
  int flat_size = tf_output_vec.size();
  // std::cout << "flat size " << flat_size << std::endl;
  Eigen::ArrayXf ret(flat_size);
  for (int i = 0; i < flat_size; ++i) {
    ret(i) = tf_output_vec(i);
  }
  return ret;
}

std::vector<CFRNetModel::InferenceOutputs> CFRNetModel::CFRNet::Inference(
    const std::vector<InferenceInputs>& inputs, bool value_or_policy,
    bool use_target) {
  tf::Session* session = tf_session_;
  if (use_target) {
    session = target_tf_session_;
  }
  int inference_batch_size = inputs.size();
  // Fill the inputs and mask
  tf::Tensor inf_input(
      tf::DT_FLOAT, tf::TensorShape({inference_batch_size, flat_input_size_}));
  tf::Tensor inf_legals_mask(
      tf::DT_BOOL, tf::TensorShape({inference_batch_size, num_actions_}));

  auto inf_legals_mask_matrix = inf_legals_mask.matrix<bool>();
  auto inf_input_matrix = inf_input.matrix<float>();

  for (int b = 0; b < inference_batch_size; ++b) {
    // Zero initialize the sparse inputs.
    for (int a = 0; a < num_actions_; ++a) {
      inf_legals_mask_matrix(b, a) = false;
    }
    for (Action action : inputs[b].legal_actions) {
      inf_legals_mask_matrix(b, action) = true;
    }
    for (int i = 0; i < inputs[b].informations.size(); ++i) {
      inf_input_matrix(b, i) = inputs[b].informations[i];
    }
  }
  // training.set_data(std::vector<int32_t>(false));
  std::vector<tf::Tensor> inf_output;
  TF_CHECK_OK(
      session->Run({{"input", inf_input}, {"legal_actions", inf_legals_mask}},
                   {"output"}, {}, &inf_output));

  std::vector<InferenceOutputs> out;
  out.reserve(inference_batch_size);
  auto inf_output_matrix = inf_output[0].matrix<float>();
  for (int b = 0; b < inference_batch_size; ++b) {
    std::vector<double> values;
    for (Action action : inputs[b].legal_actions) {
      values.push_back(inf_output_matrix(b, action));
    }
    if (value_or_policy) {
      for (int i = num_actions_; i < inf_output_matrix.dimension(1); ++i) {
        values.push_back(inf_output_matrix(b, i));
      }
    }
    out.push_back({inputs[b].legal_actions, values});
  }

  return out;
}

double CFRNetModel::CFRNet::Learn(const std::vector<TrainInputs>& inputs,
                                  const std::vector<double>& params) {
  int training_batch_size = inputs.size();
  tf::Tensor train_input(
      tf::DT_FLOAT, tf::TensorShape({training_batch_size, flat_input_size_}));
  tf::Tensor masks(tf::DT_BOOL,
                   tf::TensorShape({training_batch_size, num_actions_}));
  tf::Tensor legal_actions(
      tf::DT_BOOL, tf::TensorShape({training_batch_size, num_actions_}));
  int num_value_size = inputs[0].value.size();
  tf::Tensor targets(tf::DT_FLOAT,
                     tf::TensorShape({training_batch_size, num_value_size}));
  tf::Tensor weights(tf::DT_FLOAT, tf::TensorShape({training_batch_size, 1}));
  tf::Tensor params_input(tf::DT_FLOAT, tf::TensorShape({(long)params.size()}));

  auto train_input_matrix = train_input.matrix<float>();
  auto masks_matrix = masks.matrix<bool>();
  auto legal_actions_matrix = legal_actions.matrix<bool>();
  auto targets_matrix = targets.matrix<float>();
  auto weights_matrix = weights.matrix<float>();
  // auto gradient_input_matrix = gradient_input.vec<float>();
  auto params_input_vec = params_input.vec<float>();
  double loss = 0;

  for (int i = 0; i != params.size(); ++i) {
    params_input_vec(i) = params[i];
  }

  if (!training_batch_size) {
    loss = 0;
  } else {
    // #pragma omp parallel for default(shared) schedule(dynamic)
    for (int b = 0; b < training_batch_size; ++b) {
      // Zero initialize the sparse inputs.
      for (int a = 0; a < num_actions_; ++a) {
        masks_matrix(b, a) = false;
        legal_actions_matrix(b, a) = false;
      }
      for (Action action : inputs[b].masks) {
        masks_matrix(b, action) = true;
      }
      for (Action action : inputs[b].legal_actions) {
        legal_actions_matrix(b, action) = true;
      }
      for (int i = 0; i < inputs[b].informations.size(); ++i) {
        train_input_matrix(b, i) = inputs[b].informations[i];
      }
      for (int i = 0; i != inputs[b].value.size(); ++i) {
        targets_matrix(b, i) = inputs[b].value[i];
      }
      weights_matrix(b, 0) = inputs[b].weight;
    }
    std::vector<tf::Tensor> tf_outputs;
    tensorflow::RunOptions run_opt;
    // run_opt.set_inter_op_thread_pool(-1);
    // run_opt.set_report_tensor_allocations_upon_oom(true);
    // tensorflow::RunMetadata run_meta;
    TF_CHECK_OK(tf_session_->Run(run_opt,
                                 {{"input", train_input},
                                  {"masks", masks},
                                  {"legal_actions", legal_actions},
                                  {"targets", targets},
                                  {"params", params_input}},
                                 {
                                     "loss",
                                     //  "policy_loss",
                                     //  "value_loss",
                                     //  "entropy_loss",
                                 },
                                 {"sim_train"}, &tf_outputs, nullptr));
    loss = tf_outputs[0].scalar<float>()(0);
    // std::vector<double> losses(4);
    // for (int i = 0; i != 4; ++i) {
    //   losses[i] = tf_outputs[i].scalar<float>()(0);
    // }
    // std::vector<std::vector<double>> out;
    // out.reserve(training_batch_size);
    // auto inf_output_matrix = tf_outputs[1].matrix<float>();
    // for (int b = 0; b < training_batch_size; ++b) {
    //   std::vector<double> values;
    //   for (Action action : inputs[b].legal_actions) {
    //     values.push_back(inf_output_matrix(b, action));
    //   }
    //   for (int i = num_actions_; i < inf_output_matrix.dimension(1); ++i) {
    //     values.push_back(inf_output_matrix(b, i));
    //   }
    //   out.push_back(values);
    // }
    // auto c_out_matrix = tf_outputs[5].matrix<float>();
    // auto squard_policy_matrix = tf_outputs[6].matrix<float>();
    // auto old_p_matrix = tf_outputs[7].matrix<float>();
    // auto adv_matrix = tf_outputs[8].matrix<float>();
    // for (int b = 0; b < training_batch_size; ++b) {
    //   c_out.push_back(c_out_matrix(b, 0));
    //   squard_policy.push_back(squard_policy_matrix(b, 0));
    //   old_p.push_back(old_p_matrix(b, 0));
    //   adv.push_back(adv_matrix(b, 0));
    //   out.push_back(values);
    // }
    // int a = 0;
  }
  return loss;
}

}  // namespace algorithms
}  // namespace open_spiel
