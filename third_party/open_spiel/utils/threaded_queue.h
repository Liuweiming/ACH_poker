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

#ifndef THIRD_PARTY_OPEN_SPIEL_UTILS_THREADED_QUEUE_H_
#define THIRD_PARTY_OPEN_SPIEL_UTILS_THREADED_QUEUE_H_

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include "absl/types/optional.h"

namespace open_spiel {

using Duration = std::chrono::milliseconds;
using TimePoint = std::chrono::time_point<std::chrono::system_clock, Duration>;
// A threadsafe-queue.
template <class T>
class ThreadedQueue {
 public:
  explicit ThreadedQueue(int max_size) : max_size_(max_size) {}

  // Add an element to the queue.
  bool Push(const T& value) {
    return Push(value, std::chrono::time_point_cast<TimePoint::duration>(
                           std::chrono::system_clock::time_point::max()));
  }
  bool Push(const T& value, const Duration& wait) {
    auto new_wait =
        std::min(wait, std::chrono::duration_cast<Duration>(
                           std::chrono::system_clock::time_point::max() -
                           std::chrono::system_clock::now()));
    return Push(value, std::chrono::time_point_cast<TimePoint::duration>(
                           std::chrono::system_clock::now()) +
                           new_wait);
  }
  bool Push(const T& value, const TimePoint& deadline) {
    std::unique_lock<std::mutex> lock(m_);
    if (block_new_values_) {
      return false;
    }
    while (q_.size() >= max_size_) {
      if (cv_.wait_until(lock, deadline) == std::cv_status::timeout ||
          block_new_values_) {
        return false;
      }
    }
    q_.push(value);
    cv_.notify_one();
    return true;
  }

  absl::optional<T> Pop() { return Pop(Duration::max()); }
  absl::optional<T> Pop(const Duration& wait) {
    return Pop(std::chrono::time_point_cast<TimePoint::duration>(
                   std::chrono::system_clock::now()) +
               wait);
  }
  absl::optional<T> Pop(const TimePoint& deadline) {
    std::unique_lock<std::mutex> lock(m_);
    if (block_new_values_) {
      return absl::nullopt;
    }
    while (q_.empty()) {
      if (cv_.wait_until(lock, deadline) == std::cv_status::timeout ||
          block_new_values_) {
        return absl::nullopt;
      };
    }
    T val = q_.front();
    q_.pop();
    cv_.notify_one();
    return val;
  }

  bool Empty() {
    std::unique_lock<std::mutex> lock(m_);
    return q_.empty();
  }

  void Clear() {
    std::unique_lock<std::mutex> lock(m_);
    while (!q_.empty()) {
      q_.pop();
    }
  }

  int Size() {
    std::unique_lock<std::mutex> lock(m_);
    return q_.size();
  }

  // Causes pushing new values to fail. Useful for shutting down the queue.
  void BlockNewValues() {
    std::unique_lock<std::mutex> lock(m_);
    block_new_values_ = true;
    cv_.notify_all();
  }

 private:
  bool block_new_values_ = false;
  int max_size_;
  std::queue<T> q_;
  std::mutex m_;
  std::condition_variable cv_;
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_UTILS_THREADED_QUEUE_H_
