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

#include "open_spiel/utils/barrier.h"

namespace open_spiel {

Barrier::Barrier(int count)
    : thread_count(count), counter(0), waiting(0), ok(true) {}

bool Barrier::Block(const std::function<void()> &fist_callback,
                    const std::function<void()> &last_callback) {
  // fence mechanism
  {
    // Before entering, make sure all the old threads are exited.
    std::unique_lock<std::mutex> elk(em);
    ecv.wait(elk, [&] { return ok; });
  }
  std::unique_lock<std::mutex> lk(m);
  ++counter;
  ++waiting;
  cv.wait(lk, [&] { return counter >= thread_count; });
  {
    // Marks that threds are exiting.
    std::unique_lock<std::mutex> elk(em);
    ok = false;
  }
  if (waiting == counter) {
    // This is the first thread to exit.
    // It is responsible to call the fist_callback.
    if (fist_callback) {
      fist_callback();
    }
  }
  --waiting;
  cv.notify_one();
  if (waiting == 0) {
    // This is the last thread to exit.
    // It is responsible to call the last_callback.
    if (last_callback) {
      last_callback();
    }
    // reset barrier
    counter = 0;
    {
      // Marks that all the threds are exited.
      std::unique_lock<std::mutex> elk(em);
      ok = true;
      ecv.notify_all();
    }
    return true;
  }
  return false;
}

}  // namespace open_spiel
