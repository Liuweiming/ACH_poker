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

#include "open_spiel/games/universal_poker/logic/card_set.h"

#include <bitset>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/strings/string_view.h"
#include "open_spiel/spiel_utils.h"

constexpr absl::string_view kSuitChars = "cdhs";
constexpr absl::string_view kRankChars = "23456789TJQKA";

extern "C" {
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/evalHandTables"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
}

namespace open_spiel {
namespace universal_poker {
namespace logic {

bool operator==(const CardSet &lhs, const CardSet &rhs) {
  return lhs.cs.cards == rhs.cs.cards;
}

bool operator<(const CardSet &lhs, const CardSet &rhs) {
  return lhs.cs.cards < rhs.cs.cards;
}

// Returns the lexicographically next permutation of the supplied bits.
// See https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
uint64_t bit_twiddle_permute(uint64_t v) {
  uint64_t t = v | (v - 1);
  uint64_t u = ((~t & -~t) - 1);
  int shift = __builtin_ctzl(v) + 1;
  // Shifting by 64 bits or more is undefined behaviour, so we must avoid it.
  // See for example: http://c0x.coding-guidelines.com/6.5.7.html (1185).
  u = (shift < 64) ? (u >> shift) : 0;
  uint64_t w = (t + 1) | u;
  return w;
}

CardSet::CardSet(std::string cardString) : cs() {
  for (int i = 0; i < cardString.size(); i += 2) {
    char rankChr = cardString[i];
    char suitChr = cardString[i + 1];

    uint8_t rank = (uint8_t)(kRankChars.find(rankChr));
    uint8_t suit = (uint8_t)(kSuitChars.find(suitChr));
    SPIEL_CHECK_LT(rank, MAX_RANKS);
    SPIEL_CHECK_LT(suit, MAX_SUITS);
    cs.bySuit[suit] |= ((uint16_t)1 << rank);
  }
}

CardSet::CardSet(std::vector<int> cards) : cs() {
  for (int i = 0; i < cards.size(); ++i) {
    int rank = rankOfCard(cards[i]);
    int suit = suitOfCard(cards[i]);

    cs.bySuit[suit] |= ((uint16_t)1 << rank);
  }
}

CardSet::CardSet(uint16_t num_suits, uint16_t num_ranks) : cs() {
  for (uint16_t r = MAX_RANKS - num_ranks; r < MAX_RANKS; r++) {
    for (uint16_t s = MAX_SUITS - num_suits; s < MAX_SUITS; s++) {
      cs.bySuit[s] |= ((uint16_t)1 << r);
    }
  }
}

std::string CardSet::ToString() const {
  std::ostringstream result;
  for (int r = MAX_RANKS - 1; r >= 0; r--) {
    for (int s = MAX_SUITS - 1; s >= 0; s--) {
      uint32_t mask = (uint32_t)1 << r;
      if (cs.bySuit[s] & mask) {
        result << kRankChars[r] << kSuitChars[s];
      }
    }
  }

  return result.str();
}

std::vector<uint8_t> CardSet::ToCardArray() const {
  std::vector<uint8_t> result(NumCards(), 0);

  int i = 0;
  for (int r = 0; r < MAX_RANKS; ++r) {
    for (int s = 0; s < MAX_SUITS; ++s) {
      uint32_t mask = (uint32_t)1 << r;
      if (cs.bySuit[s] & mask) {
        result[i++] = makeCard(r, s);
      }
    }
  }
  return result;
}

void CardSet::AddCard(uint8_t card) {
  int rank = rankOfCard(card);
  int suit = suitOfCard(card);

  cs.bySuit[suit] |= ((uint16_t)1 << rank);
}

void CardSet::Combine(const CardSet &new_cs) { cs.cards |= new_cs.cs.cards; }

void CardSet::RemoveCard(uint8_t card) {
  int rank = rankOfCard(card);
  int suit = suitOfCard(card);

  cs.bySuit[suit] ^= ((uint16_t)1 << rank);
}

int CardSet::NumCards() const { return __builtin_popcountl(cs.cards); }

int CardSet::RankCards() const {
  ::Cardset csNative;
  csNative.cards = cs.cards;
  return rankCardset(csNative);
}

std::vector<CardSet> CardSet::SampleCards(int nbCards) {
  std::vector<CardSet> combinations;
  if (nbCards <= 0) {
    return combinations;
  }
  uint64_t p = 0;
  for (int i = 0; i < nbCards; ++i) {
    p += (1 << i);
  }
  int nbZeros = __builtin_clzl(cs.cards);
  for (uint64_t n = bit_twiddle_permute(p); __builtin_clzl(p) >= nbZeros;
       p = n, n = bit_twiddle_permute(p)) {
    uint64_t combo = p & cs.cards;
    if (__builtin_popcountl(combo) == nbCards) {
      CardSet c;
      c.cs.cards = combo;
      combinations.emplace_back(c);
    }
  }

  // std::cout << "combinations.size() " << combinations.size() << std::endl;
  return combinations;
}

bool CardSet::ContainsCards(uint8_t card) const {
  int rank = rankOfCard(card);
  int suit = suitOfCard(card);
  return (cs.bySuit[suit] & ((uint16_t)1 << rank)) > 0;
}
}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel