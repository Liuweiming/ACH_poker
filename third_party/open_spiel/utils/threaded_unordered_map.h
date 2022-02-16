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

#ifndef THIRD_PARTY_OPEN_SPIEL_UTILS_THREADED_UNORDERED_MAP_H_
#define THIRD_PARTY_OPEN_SPIEL_UTILS_THREADED_UNORDERED_MAP_H_

#include <mutex>
#include <unordered_map>
#include "absl/types/optional.h"

namespace open_spiel {
// a simple ThreadedUnorderedMap.
// the user should makesure that type _Tp is also thread safe.
template <typename _Key, typename _Tp>
class ThreadedUnorderedMap {
 public:
  using map_type = std::unordered_map<_Key, _Tp>;
  using key_type = typename map_type::key_type;
  using mapped_type = typename map_type::mapped_type;
  using value_type = typename map_type::value_type;
  using size_type = typename map_type::size_type;
  // move only.
  ThreadedUnorderedMap() = default;
  ThreadedUnorderedMap(const ThreadedUnorderedMap&) = delete;
  ThreadedUnorderedMap(ThreadedUnorderedMap&&) = default;
  ThreadedUnorderedMap& operator=(const ThreadedUnorderedMap&) = delete;
  ThreadedUnorderedMap& operator=(ThreadedUnorderedMap&&) = delete;

  ~ThreadedUnorderedMap(){};
  bool exist(const _Key& key) {
    std::unique_lock<std::mutex> lock(mtx);
    return _map.find(key) != _map.end();
  }

  _Tp& operator[](const _Key& key) {
    std::unique_lock<std::mutex> lock(mtx);
    return _map[key];
  }

  void insert(const _Key& key, const _Tp& value) {
    std::unique_lock<std::mutex> lock(mtx);
    _map.insert({key, value});
  }

  void insert(const value_type& v) {
    std::unique_lock<std::mutex> lock(mtx);
    _map.insert(v);
  }

  void erase(const _Key& key) {
    std::unique_lock<std::mutex> lock(mtx);
    _map.erase(key);
  }

  void clear() {
    std::unique_lock<std::mutex> lock(mtx);
    _map.clear();
  }

  bool empty() {
    std::unique_lock<std::mutex> lock(mtx);
    return _map.empty();
  }

  size_type size() {
    std::unique_lock<std::mutex> lock(mtx);
    return _map.size();
  }

 private:
  std::mutex mtx;
  std::unordered_map<_Key, _Tp> _map;
};
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_UTILS_THREADED_UNORDERED_MAP_H_