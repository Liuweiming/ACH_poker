#!/usr/bin/env bash

source pre_run.sh

omp_thread=16
actors=100
num_cpus=1

./build/algorithms/run_ach --use_regret_net=true --use_policy_net=true --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=1000000 --policy_memory_size=10000000 --cfr_batch_size=1000 \
--train_batch_size=128 --train_steps=1 --policy_train_steps=16 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=100000 \
--omp_threads=$omp_thread --evaluation_window=100 --exp_evaluation_window=true --first_evaluation=10 --game=FHP_poker \
--checkpoint_freq=1000000 --sync_period=1 --max_steps=100000000 --graph_def= --verbose=false --suffix=$RANDOM \
--learning_rate=0.0001 --ach_alpha=2 --ach_beta=0.03 --ach_thres=2 --ach_reward_scale=0.002 --verbose=false
