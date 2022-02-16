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

#ifndef THIRD_PARTY_OPEN_SPIEL_UTILS_JSON_H_
#define THIRD_PARTY_OPEN_SPIEL_UTILS_JSON_H_

#include <cstdint>

#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"

namespace open_spiel {
namespace json {

class Null {
 public:
  bool operator==(const Null& o) const;
  bool operator!=(const Null& o) const;
};

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value : public absl::variant<Null, bool, int64_t, double, std::string,
                                   Array, Object> {
 public:
  using absl::variant<Null, bool, int64_t, double, std::string, Array,
                      Object>::variant;  // Inherit the constructors.
  Value() {}
  Value(int v) : Value(static_cast<int64_t>(v)) {}
  Value(const char* v) : Value(std::string(v)) {}

  bool IsNull() const { return absl::holds_alternative<Null>(*this); }
  bool IsBool() const { return absl::holds_alternative<bool>(*this); }
  bool IsTrue() const { return IsBool() && GetBool(); }
  bool IsFalse() const { return IsBool() && !GetBool(); }
  bool IsInt() const { return absl::holds_alternative<int64_t>(*this); }
  bool IsDouble() const { return absl::holds_alternative<double>(*this); }
  bool IsNumber() const { return IsInt() || IsDouble(); }
  bool IsString() const { return absl::holds_alternative<std::string>(*this); }
  bool IsArray() const { return absl::holds_alternative<Array>(*this); }
  bool IsObject() const { return absl::holds_alternative<Object>(*this); }
  bool GetBool() const { return absl::get<bool>(*this); }
  int64_t GetInt() const { return absl::get<int64_t>(*this); }
  int64_t& GetInt() { return absl::get<int64_t>(*this); }
  double GetDouble() const { return absl::get<double>(*this); }
  double& GetDouble() { return absl::get<double>(*this); }
  const std::string& GetString() const { return absl::get<std::string>(*this); }
  std::string& GetString() { return absl::get<std::string>(*this); }
  const Array& GetArray() const { return absl::get<Array>(*this); }
  Array& GetArray() { return absl::get<Array>(*this); }
  const Object& GetObject() const { return absl::get<Object>(*this); }
  Object& GetObject() { return absl::get<Object>(*this); }

  bool operator==(const Null& o) const { return IsNull(); }
  bool operator==(const bool& o) const { return IsBool() && GetBool() == o; }
  bool operator==(const int& o) const { return IsInt() && GetInt() == o; }
  bool operator==(const int64_t& o) const { return IsInt() && GetInt() == o; }
  bool operator==(const double& o) const {
    return IsDouble() && GetDouble() == o;
  }
  bool operator==(const char* o) const {
    return IsString() && GetString() == o;
  }
  bool operator==(const std::string& o) const {
    return IsString() && GetString() == o;
  }
  bool operator==(const Array& o) const { return IsArray() && GetArray() == o; }
  bool operator==(const Object& o) const {
    return IsObject() && GetObject() == o;
  }
  template <typename T>
  bool operator!=(const T& o) const {
    return !(*this == o);
  }
};

// Accept a std::vector of any of the types that are constructible to Value.
// For example accept std::vector<int>.
template <typename T>
Array CastToArray(std::vector<T> vec) {
  Array out;
  out.reserve(vec.size());
  for (const T& val : vec) {
    out.emplace_back(val);
  }
  return out;
}

// Accept a std::vector of any type with a fn that converts to any type
// constructible to Value.
template <typename T, typename Fn>
Array TransformToArray(std::vector<T> vec, Fn fn) {
  Array out;
  out.reserve(vec.size());
  for (const T& val : vec) {
    out.emplace_back(fn(val));
  }
  return out;
}

// Serialize a JSON value into a string.
// Set wrap to true to pretty print with each element of a list or array on
// its own line. Indent is how much to start indenting by, and is mainly used
// for the recursive printer, which increments it by 2 each level.
std::string ToString(const Array& array, bool wrap = false, int indent = 0);
std::string ToString(const Object& obj, bool wrap = false, int indent = 0);
std::string ToString(const Value& value, bool wrap = false, int indent = 0);

std::ostream& operator<<(std::ostream& os, const Null& n);
std::ostream& operator<<(std::ostream& os, const Array& a);
std::ostream& operator<<(std::ostream& os, const Object& o);
std::ostream& operator<<(std::ostream& os, const Value& v);

// Deserialize a string into a JSON value. Returns nullopt on parse failure.
absl::optional<Value> FromString(absl::string_view str);
}
}

#endif  // THIRD_PARTY_OPEN_SPIEL_UTILS_JSON_H_
