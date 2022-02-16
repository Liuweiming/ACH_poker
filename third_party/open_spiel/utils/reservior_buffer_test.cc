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

#include "open_spiel/utils/reservior_buffer.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <random>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestReserviorBuffer() {
  ReserviorBuffer<int> buffer(4);
  std::mt19937 rng;
  std::vector<int> sample;
  for (int i = 0; i != 10; ++i) {
    buffer.Clear();
    SPIEL_CHECK_TRUE(buffer.Empty());
    SPIEL_CHECK_EQ(buffer.Size(), 0);

    buffer.Add(13, rng);
    SPIEL_CHECK_FALSE(buffer.Empty());
    SPIEL_CHECK_EQ(buffer.Size(), 1);
    SPIEL_CHECK_EQ(buffer.TotalAdded(), 1);
    SPIEL_CHECK_EQ(buffer[0], 13);

    sample = buffer.GetNext(rng, 1);
    SPIEL_CHECK_EQ(sample.size(), 1);
    SPIEL_CHECK_EQ(sample[0], 13);

    buffer.Add(14, rng);
    buffer.Add(15, rng);
    buffer.Add(16, rng);

    SPIEL_CHECK_EQ(buffer.Size(), 4);
    SPIEL_CHECK_EQ(buffer.TotalAdded(), 4);

    sample = buffer.GetNext(rng, 2);
    SPIEL_CHECK_EQ(sample.size(), 2);
    SPIEL_CHECK_GE(sample[0], 13);
    SPIEL_CHECK_LE(sample[0], 16);
    SPIEL_CHECK_GE(sample[1], 13);
    SPIEL_CHECK_LE(sample[1], 16);

    buffer.Add(17, rng);
    buffer.Add(18, rng);

    SPIEL_CHECK_EQ(buffer.Size(), 4);
    SPIEL_CHECK_EQ(buffer.TotalAdded(), 6);

    sample = buffer.GetNext(rng, 1);
    SPIEL_CHECK_EQ(sample.size(), 1);
    SPIEL_CHECK_GE(sample[0], 13);
    SPIEL_CHECK_LE(sample[0], 18);
  }
}

void TestShuffleAndSample() {
  ReserviorBuffer<int> buffer(100);
  std::random_device rd;
  std::mt19937 rng(rd());
  std::vector<int> sample;
  for (int i = 0; i != 1000; ++i) {
    buffer.Add(i, rng);
  }
  int goal = 0;
  for (int i = 0; i != 100; ++i) {
    buffer.GetNext(rng, i);
    for (int j = 0; j != 100; ++j) {
      auto sample = buffer.GetNext(rng, 100);
      int sum = std::accumulate(sample.begin(), sample.end(), 0);
      if (goal == 0) {
        goal = sum;
      }
      SPIEL_CHECK_EQ(goal, sum);
    }
  }
}

void TestReserviorDist() {
  int size = 1000;
  ReserviorBuffer<double> buffer(size);
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(0, 1);
  std::vector<double> sample;
  double sum = 0;
  std::vector<double> real_dist;
  for (int i = 0; i != 20000; ++i) {
    double r = dist(rng);
    buffer.Add(r, rng);
    sum += r;
    auto s = buffer.GetNext(rng, size);
    double s_mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
    sample.push_back(s_mean);
    real_dist.push_back(sum / (i + 1));
    std::cout << real_dist.back() << " " << sample.back() << std::endl;
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestReserviorBuffer();
  open_spiel::TestShuffleAndSample();
  open_spiel::TestReserviorDist();
}
