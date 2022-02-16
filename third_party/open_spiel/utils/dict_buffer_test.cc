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

#include "open_spiel/utils/dict_buffer.h"

#include <algorithm>
#include <random>
#include <string>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestDictBuffer() {
  DictBuffer<std::string, int> buffer(5);
  std::mt19937 rng;
  for (int i = 0; i != 10; ++i) {
    buffer.Clear();
    std::vector<int> sample;
    SPIEL_CHECK_TRUE(buffer.Empty());
    SPIEL_CHECK_EQ(buffer.Size(), 0);

    auto op = [](int a, int b) { return a + b; };
    buffer.Add({"a", 13}, op);
    SPIEL_CHECK_FALSE(buffer.Empty());
    SPIEL_CHECK_EQ(buffer.Size(), 1);
    SPIEL_CHECK_EQ(buffer.TotalAdded(), 1);
    SPIEL_CHECK_EQ(buffer[0], 13);

    sample = buffer.Sample(rng, 1);
    SPIEL_CHECK_EQ(sample.size(), 1);
    SPIEL_CHECK_EQ(sample[0], 13);

    buffer.Add({"b", 14}, op);
    buffer.Add({"c", 15}, op);
    buffer.Add({"a", 16}, op);

    SPIEL_CHECK_EQ(buffer.Size(), 3);
    SPIEL_CHECK_EQ(buffer.TotalAdded(), 4);

    sample = buffer.Sample(rng, 4);
    SPIEL_CHECK_EQ(sample.size(), 3);
    SPIEL_CHECK_EQ(*std::min_element(sample.begin(), sample.end()), 14);
    SPIEL_CHECK_EQ(*std::max_element(sample.begin(), sample.end()), 16 + 13);

    buffer.Add({"d", 17}, op);
    buffer.Add({"b", 18}, op);

    SPIEL_CHECK_EQ(buffer.Size(), 4);
    SPIEL_CHECK_EQ(buffer.TotalAdded(), 6);

    sample = buffer.Sample(rng, 4);
    SPIEL_CHECK_EQ(sample.size(), 4);
    SPIEL_CHECK_EQ(*std::min_element(sample.begin(), sample.end()), 15);
    SPIEL_CHECK_EQ(*std::max_element(sample.begin(), sample.end()), 18 + 14);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::TestDictBuffer(); }
