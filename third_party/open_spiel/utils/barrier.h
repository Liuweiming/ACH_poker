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

#ifndef THIRD_PARTY_OPEN_SPIEL_UTILS_BARRIER_H_
#define THIRD_PARTY_OPEN_SPIEL_UTILS_BARRIER_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace open_spiel {

class Barrier {
 public:
  Barrier(int count);

  bool Block(const std::function<void()> &fist_callback = nullptr,
             const std::function<void()> &last_callback = nullptr);
  int Count() { return counter; }

 private:
  std::mutex m;
  std::mutex em;
  std::condition_variable cv;
  std::condition_variable ecv;
  bool ok;
  int counter;
  int waiting;
  int thread_count;
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_UTILS_BARRIER_H_
