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

#ifndef THIRD_PARTY_OPEN_SPIEL_QUERY_H_
#define THIRD_PARTY_OPEN_SPIEL_QUERY_H_

#include "open_spiel/spiel.h"

// A query API to get game-specific properties.

namespace open_spiel {
namespace query {

// Negotiation
std::vector<int> NegotiationItemPool(const State& state);
std::vector<int> NegotiationAgentUtils(const State& state, int player);

}  // namespace query
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_QUERY_H_
