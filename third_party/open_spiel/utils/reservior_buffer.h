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

#ifndef THIRD_PARTY_OPEN_SPIEL_UTILS_RESERVIOR_BUFFER_H_
#define THIRD_PARTY_OPEN_SPIEL_UTILS_RESERVIOR_BUFFER_H_

#include <algorithm>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "circular_buffer.h"

namespace open_spiel {

// A simple circular buffer of fixed size.
template <class T>
class ReserviorBuffer : public CircularBuffer<T> {
 public:
  explicit ReserviorBuffer(int max_size) : CircularBuffer<T>(max_size) {}

  // Add one element, replacing the oldest once it's full.
  void Add(const T& value, std::mt19937& gen) {
    if (this->data_.size() < this->max_size_) {
      this->data_.push_back(value);
    } else {
      int64_t index =
          this->dis_(gen, std::uniform_int_distribution<int64_t>::param_type{
                              0, this->total_added_});
      if (index < this->max_size_) {
        this->data_[index] = value;
      }
    }
    this->total_added_ += 1;
    this->need_init_ = true;
  }
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_UTILS_RESERVIOR_BUFFER_H_
