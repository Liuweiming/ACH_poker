
#include <omp.h>
#include <signal.h>

#include <ctime>
#include <iostream>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "deep_cfr.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/thread.h"

ABSL_FLAG(std::string, game, "leduc_poker", "The name of the game to play.");
ABSL_FLAG(std::string, path, "", "Where to output the logs.");
ABSL_FLAG(std::string, suffix, "", "suffix for path.");
ABSL_FLAG(std::string, graph_def, "deep_cfr",
          ("Where to get the graph. This could be from export_vpnet.py, or "
           "from a checkpoint. If this is empty it'll create one."));
ABSL_FLAG(bool, play, false, "playing");
ABSL_FLAG(std::string, init_strategy_0, "", "checkpoint file to init strategy");
ABSL_FLAG(std::string, init_strategy_1, "", "checkpoint file to init strategy");
ABSL_FLAG(std::string, host, "localhost", "host for playing");
ABSL_FLAG(int, port, 12345, "port for playing");
ABSL_FLAG(std::string, nn_model, "normal", "Model torso type.");
ABSL_FLAG(int, nn_width, 64, "Width of the model, passed to export_vpnet.py.");
ABSL_FLAG(int, nn_depth, 2, "Depth of the model, passed to export_vpnet.py.");
ABSL_FLAG(int, num_gpus, 0, "number of gpus.");
ABSL_FLAG(int, num_cpus, 1, "number of cpus.");
ABSL_FLAG(std::string, cuda_id, "", "id for cuda.");
ABSL_FLAG(std::string, cfr_mode, "ESCFR",
          "cfr mode: CFR, OSCFR, ESCFR, SbCFR, PreSbCFR, PostSbCFR, SubCFR, "
          "SubSbCFR, SubPreSbCFR, SubPostSbCFR.");
ABSL_FLAG(std::string, average_type, "Opponent",
          "average_type: Current, Opponent, LinearOpponent");
ABSL_FLAG(std::string, weight_type, "Linear", "weight type: Linear, Constant");
ABSL_FLAG(double, cfr_rm_scale, 1e-2, "CFR regret metch scaler.");
ABSL_FLAG(double, cfr_scale_ub, 0, "cfr regret match parameter.");
ABSL_FLAG(double, cfr_scale_lb, 0, "cfr regret match parameter.");
ABSL_FLAG(double, cfr_rm_amp, 1.01, "cfr regret match parameter.");
ABSL_FLAG(double, cfr_rm_damp, 0.99, "cfr regret match parameter.");
ABSL_FLAG(double, cfr_rm_lb, 1e-20, "cfr regret match parameter.");
ABSL_FLAG(double, cfr_rm_ub, 100, "cfr regret match parameter.");
ABSL_FLAG(bool, use_regret_net, false, "cfr regret match parameter.");
ABSL_FLAG(bool, use_policy_net, false, "cfr regret match parameter.");
ABSL_FLAG(bool, use_tabular, false, "cfr regret match parameter.");
ABSL_FLAG(int, cfr_max_iterations, 10000, "cfr regret match parameter.");
ABSL_FLAG(int, cfr_batch_size, 100, "cfr regret match parameter.");
ABSL_FLAG(int, cfr_traverse_steps, 1, "cfr regret match parameter.");
ABSL_FLAG(bool, nfsp_anticipatory, true, "nfsp parameter.");
ABSL_FLAG(double, nfsp_eta, 0.1, "nfsp parameter.");
ABSL_FLAG(double, nfsp_epsilon, 0.06, "nfsp parameter.");
ABSL_FLAG(double, ach_eta, 1, "ach parameter.");
ABSL_FLAG(double, ach_alpha, 2, "ach parameter.");
ABSL_FLAG(double, ach_beta, 0.01, "ach parameter.");
ABSL_FLAG(double, ach_thres, 2, "ach parameter.");
ABSL_FLAG(double, ach_epsilon, 0.05, "ach parameter.");
ABSL_FLAG(double, ach_reward_scale, 1, "ach parameter.");
ABSL_FLAG(int, memory_size, 10000, "Memory size for training cfr value.");
ABSL_FLAG(int, policy_memory_size, 400000,
          "Memory size for training cfr policy");
ABSL_FLAG(double, learning_rate, 0.001, "Learning rate.");
ABSL_FLAG(double, weight_decay, 0., "Weight decay.");
ABSL_FLAG(int, train_batch_size, 256,
          "How many states to learn from per batch.");
ABSL_FLAG(int, inference_batch_size, 1,
          "How many threads to wait for for inference.");
ABSL_FLAG(int, train_steps, 100, "How many steps for training cfr value.");
ABSL_FLAG(int, policy_train_steps, 100,
          "How many steps for training cfr policy.");
ABSL_FLAG(int, inference_threads, 1, "How many threads to run inference.");
ABSL_FLAG(int, inference_cache, 100000,
          "Whether to cache the results from inference.");
ABSL_FLAG(int, checkpoint_freq, 1000, "Save a checkpoint every N steps.");
ABSL_FLAG(double, checkpoint_second, 86400,
          "Save a checkpoint every N seconds.");
ABSL_FLAG(int, evaluation_window, 10, "evaluation_window");
ABSL_FLAG(bool, exp_evaluation_window, false, "evaluation_window");
ABSL_FLAG(bool, local_best_response, false,
          "use local best response for the evaluation");
ABSL_FLAG(bool, post_evaluation, false,
          "use local best response for the post evaluation");
ABSL_FLAG(int, lbr_batch_size, 10000, "lbr batch size");
ABSL_FLAG(int, first_evaluation, 1, "first_evaluation");
ABSL_FLAG(int, max_evaluation_window, 10000, "max_evaluation");
ABSL_FLAG(int, actors, 1, "How many actors to run.");
ABSL_FLAG(int, evaluators, 1, "How many evaluators to run.");
ABSL_FLAG(int, omp_threads, 1,
          "How many openmp threads for evaluation and training. Default to 1, "
          "i.e., use one thread.");
ABSL_FLAG(bool, sync_target_by_peroid, false, "sync target network by period");
ABSL_FLAG(int, sync_period, 1, "period to sync target network");
ABSL_FLAG(double, target_moving_param, 1.0,
          "moving average parameter to sync target network");
ABSL_FLAG(bool, sync_by_restore, false, "If sync by restore");
ABSL_FLAG(bool, sync_by_copy, true, "If sync by copy");
ABSL_FLAG(bool, verbose, false, "verbose");
ABSL_FLAG(int, max_steps, 1000000, "how many steps.");

open_spiel::StopToken stop_token;

void signal_handler(int s) {
  if (stop_token.StopRequested()) {
    exit(1);
  } else {
    stop_token.Stop();
  }
}

void signal_installer() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = signal_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, nullptr);
}

int main(int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    std::cout << argv[i] << std::endl;
  }
  absl::ParseCommandLine(argc, argv);
  signal_installer();
  open_spiel::algorithms::DeepCFRConfig config;
  config.game = absl::GetFlag(FLAGS_game);
  config.play = absl::GetFlag(FLAGS_play);
  config.init_strategy_0 = absl::GetFlag(FLAGS_init_strategy_0);
  config.init_strategy_1 = absl::GetFlag(FLAGS_init_strategy_1);
  config.host = absl::GetFlag(FLAGS_host);
  config.port = absl::GetFlag(FLAGS_port);
  config.path = absl::GetFlag(FLAGS_path);
  if (config.path == "") {
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%d_%m_%Y.%X", timeinfo);
    std::string str(buffer);
    std::string::size_type post;
    while ((post = str.find(":")) != str.npos) str.replace(post, 1, "_");
    config.path = "./results/" + str + "." + absl::GetFlag(FLAGS_suffix);
  }
  config.graph_def = absl::GetFlag(FLAGS_graph_def);
  config.nn_model = absl::GetFlag(FLAGS_nn_model);
  config.nn_width = absl::GetFlag(FLAGS_nn_width);
  config.nn_depth = absl::GetFlag(FLAGS_nn_depth);
  config.num_gpus = absl::GetFlag(FLAGS_num_gpus);
  config.num_cpus = absl::GetFlag(FLAGS_num_cpus);
  config.cuda_id = absl::GetFlag(FLAGS_cuda_id);
  if (!config.num_gpus) {
    config.cuda_id = "";
  } else {
    SPIEL_CHECK_FALSE(config.cuda_id.empty());
    setenv("CUDA_VISIBLE_DEVICES", config.cuda_id.c_str(), true);
    std::cout << "CUDA_VISIBLE_DEVICES=" << getenv("CUDA_VISIBLE_DEVICES")
              << std::endl;
  }

  config.cfr_mode = absl::GetFlag(FLAGS_cfr_mode);
  config.average_type = absl::GetFlag(FLAGS_average_type);
  config.weight_type = absl::GetFlag(FLAGS_weight_type);
  config.cfr_rm_scale = absl::GetFlag(FLAGS_cfr_rm_scale);
  config.cfr_scale_ub = absl::GetFlag(FLAGS_cfr_scale_ub);
  config.cfr_scale_lb = absl::GetFlag(FLAGS_cfr_scale_lb);
  config.cfr_rm_amp = absl::GetFlag(FLAGS_cfr_rm_amp);
  config.cfr_rm_damp = absl::GetFlag(FLAGS_cfr_rm_damp);
  config.cfr_rm_lb = absl::GetFlag(FLAGS_cfr_rm_lb);
  config.cfr_rm_ub = absl::GetFlag(FLAGS_cfr_rm_ub);
  config.use_regret_net = absl::GetFlag(FLAGS_use_regret_net);
  config.use_policy_net = absl::GetFlag(FLAGS_use_policy_net);
  config.use_tabular = absl::GetFlag(FLAGS_use_tabular);
  config.cfr_max_iterations = absl::GetFlag(FLAGS_cfr_max_iterations);
  config.cfr_batch_size = absl::GetFlag(FLAGS_cfr_batch_size);
  config.cfr_traverse_steps = absl::GetFlag(FLAGS_cfr_traverse_steps);

  config.nfsp_anticipatory = absl::GetFlag(FLAGS_nfsp_anticipatory);
  config.nfsp_eta = absl::GetFlag(FLAGS_nfsp_eta);
  config.nfsp_epsilon = absl::GetFlag(FLAGS_nfsp_epsilon);

  config.ach_eta = absl::GetFlag(FLAGS_ach_eta);
  config.ach_alpha = absl::GetFlag(FLAGS_ach_alpha);
  config.ach_beta = absl::GetFlag(FLAGS_ach_beta);
  config.ach_thres = absl::GetFlag(FLAGS_ach_thres);
  config.ach_epsilon = absl::GetFlag(FLAGS_ach_epsilon);
  config.ach_reward_scale = absl::GetFlag(FLAGS_ach_reward_scale);

  config.memory_size = absl::GetFlag(FLAGS_memory_size);
  config.policy_memory_size = absl::GetFlag(FLAGS_policy_memory_size);

  config.learning_rate = absl::GetFlag(FLAGS_learning_rate);
  config.weight_decay = absl::GetFlag(FLAGS_weight_decay);
  config.train_batch_size = absl::GetFlag(FLAGS_train_batch_size);
  config.inference_batch_size = absl::GetFlag(FLAGS_inference_batch_size);
  config.train_steps = absl::GetFlag(FLAGS_train_steps);
  config.policy_train_steps = absl::GetFlag(FLAGS_policy_train_steps);
  config.inference_threads = absl::GetFlag(FLAGS_inference_threads);
  config.inference_cache = absl::GetFlag(FLAGS_inference_cache);
  config.checkpoint_freq = absl::GetFlag(FLAGS_checkpoint_freq);
  config.checkpoint_second = absl::GetFlag(FLAGS_checkpoint_second);
  config.evaluation_window = absl::GetFlag(FLAGS_evaluation_window);
  config.exp_evaluation_window = absl::GetFlag(FLAGS_exp_evaluation_window);
  config.local_best_response = absl::GetFlag(FLAGS_local_best_response);
  config.post_evaluation = absl::GetFlag(FLAGS_post_evaluation);
  config.lbr_batch_size = absl::GetFlag(FLAGS_lbr_batch_size);
  config.first_evaluation = absl::GetFlag(FLAGS_first_evaluation);
  config.max_evaluation_window = absl::GetFlag(FLAGS_max_evaluation_window);

  config.actors = absl::GetFlag(FLAGS_actors);
  config.evaluators = absl::GetFlag(FLAGS_evaluators);
  config.omp_threads = absl::GetFlag(FLAGS_omp_threads);
  if (config.omp_threads == -1) {
    config.omp_threads = omp_get_num_procs();
  }
  omp_set_num_threads(config.omp_threads);
  config.sync_target_by_peroid = absl::GetFlag(FLAGS_sync_target_by_peroid);
  config.sync_period = absl::GetFlag(FLAGS_sync_period);
  config.target_moving_param = absl::GetFlag(FLAGS_target_moving_param);
  config.sync_by_restore = absl::GetFlag(FLAGS_sync_by_restore);
  config.sync_by_copy = absl::GetFlag(FLAGS_sync_by_copy);
  config.max_steps = absl::GetFlag(FLAGS_max_steps);

  config.verbose = absl::GetFlag(FLAGS_verbose);

  deep_cfr(config, &stop_token);
  return 0;
}
