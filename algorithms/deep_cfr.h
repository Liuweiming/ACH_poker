#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_DEEP_CFR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_DEEP_CFR_H_

#include "open_spiel/utils/json.h"
#include "open_spiel/utils/thread.h"

namespace open_spiel {
namespace algorithms {

struct DeepCFRConfig {
  std::string game;
  bool play;
  std::string init_strategy_0;
  std::string init_strategy_1;
  std::string host;
  int port;
  std::string path;
  std::string graph_def;
  std::string nn_model;
  int nn_width;
  int nn_depth;
  int num_gpus;
  int num_cpus;
  std::string cuda_id;

  std::string cfr_mode;
  std::string average_type;
  std::string weight_type;
  double cfr_rm_scale;
  double cfr_scale_ub;
  double cfr_scale_lb;
  double cfr_rm_amp;
  double cfr_rm_damp;
  double cfr_rm_lb;
  double cfr_rm_ub;
  bool use_regret_net;
  bool use_policy_net;
  bool use_tabular;
  int cfr_max_iterations;
  int cfr_batch_size;
  int cfr_traverse_steps;

  bool nfsp_anticipatory;
  double nfsp_eta;
  double nfsp_epsilon;

  double ach_eta;
  double ach_alpha;
  double ach_beta;
  double ach_thres;
  double ach_epsilon;
  double ach_reward_scale;

  int memory_size;
  int policy_memory_size;

  double learning_rate;
  double weight_decay;
  int train_batch_size;
  int inference_batch_size;
  int train_steps;
  int policy_train_steps;
  int inference_threads;
  int inference_cache;
  int checkpoint_freq;
  int checkpoint_second;
  int evaluation_window;
  bool exp_evaluation_window;
  int first_evaluation;
  int max_evaluation_window;
  bool local_best_response;
  bool post_evaluation;
  int lbr_batch_size;
  int actors;
  int evaluators;
  int omp_threads;
  bool sync_target_by_peroid;
  int sync_period;
  double target_moving_param;
  bool sync_by_restore;
  bool sync_by_copy;
  int max_steps;
  bool verbose;

  json::Object ToJson() const {
    return json::Object({{"game", game},
                         {"play", play},
                         {"init_strategy_0", init_strategy_0},
                         {"init_strategy_1", init_strategy_1},
                         {"host", host},
                         {"port", port},
                         {"path", path},
                         {"graph_def", graph_def},
                         {"nn_model", nn_model},
                         {"nn_width", nn_width},
                         {"nn_depth", nn_depth},
                         {"num_gpus", num_gpus},
                         {"num_cpus", num_cpus},
                         {"cuda_id", cuda_id},
                         {"cfr_mode", cfr_mode},
                         {"average_type", average_type},
                         {"weight_type", weight_type},
                         {"cfr_rm_scale", cfr_rm_scale},
                         {"cfr_scale_ub", cfr_scale_ub},
                         {"cfr_scale_lb", cfr_scale_lb},
                         {"cfr_rm_amp", cfr_rm_amp},
                         {"cfr_rm_damp", cfr_rm_damp},
                         {"cfr_rm_lb", cfr_rm_lb},
                         {"cfr_rm_ub", cfr_rm_ub},
                         {"nfsp_anticipatory", nfsp_anticipatory},
                         {"nfsp_eta", nfsp_eta},
                         {"ach_eta", ach_eta},
                         {"ach_alpha", ach_alpha},
                         {"ach_beta", ach_beta},
                         {"ach_thres", ach_thres},
                         {"ach_epsilon", ach_epsilon},
                         {"ach_reward_scale", ach_reward_scale},
                         {"use_regret_net", use_regret_net},
                         {"use_policy_net", use_policy_net},
                         {"use_tabular", use_tabular},
                         {"cfr_max_iterations", cfr_max_iterations},
                         {"cfr_batch_size", cfr_batch_size},
                         {"cfr_traverse_steps", cfr_traverse_steps},
                         {"memory_size", memory_size},
                         {"policy_memory_size", policy_memory_size},
                         {"learning_rate", learning_rate},
                         {"weight_decay", weight_decay},
                         {"train_batch_size", train_batch_size},
                         {"inference_batch_size", inference_batch_size},
                         {"train_steps", train_steps},
                         {"policy_train_steps", policy_train_steps},
                         {"inference_threads", inference_threads},
                         {"inference_cache", inference_cache},
                         {"checkpoint_freq", checkpoint_freq},
                         {"checkpoint_second", checkpoint_second},
                         {"evaluation_window", evaluation_window},
                         {"exp_evaluation_window", exp_evaluation_window},
                         {"first_evaluation", first_evaluation},
                         {"max_evaluation_window", max_evaluation_window},
                         {"local_best_response", local_best_response},
                         {"post_evaluation", post_evaluation},
                         {"lbr_batch_size", lbr_batch_size},
                         {"actors", actors},
                         {"evaluators", evaluators},
                         {"omp_threads", omp_threads},
                         {"sync_target_by_peroid", sync_target_by_peroid},
                         {"sync_period", sync_period},
                         {"target_moving_param", target_moving_param},
                         {"sync_by_restore", sync_by_restore},
                         {"sync_by_copy", sync_by_copy},
                         {"max_steps", max_steps},
                         {"verbose", verbose}});
  }
};
bool deep_cfr(DeepCFRConfig config, StopToken* stop);
}  // namespace algorithms
}  // namespace open_spiel

#endif