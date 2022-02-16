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

#ifndef THIRD_PARTY_OPEN_SPIEL_UTILS_DICT_BUFFER_H_
#define THIRD_PARTY_OPEN_SPIEL_UTILS_DICT_BUFFER_H_

#include <algorithm>
#include <limits>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/reservior_buffer.h"

namespace open_spiel {

// A simple circular buffer of fixed size.
template <typename K, typename T>
class DictBuffer : public ReserviorBuffer<T> {
 public:
  explicit DictBuffer(int max_size) : ReserviorBuffer<T>(max_size) {}

  // Add one element, replacing the oldest once it's full.
  template <typename Op_type>
  void Add(const std::pair<K, T>& kv, Op_type op) {
    const K& key = kv.first;
    const T& value = kv.second;
    if (key_to_index_.find(key) != key_to_index_.end()) {
      int index = key_to_index_[key];
      this->data_[index] = op(this->data_[index], value);
    } else if (this->data_.size() < this->max_size_) {
      int index = this->data_.size();
      key_to_index_.insert({key, index});
      this->data_.push_back(value);
    }
    this->total_added_ += 1;
  }

  void Clear() {
    ReserviorBuffer<T>::Clear();
    key_to_index_.clear();
  }

 protected:
  std::unordered_map<K, int> key_to_index_;
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_UTILS_DICT_BUFFER_H_
