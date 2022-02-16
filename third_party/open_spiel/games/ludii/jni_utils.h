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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_LUDII_JNIUTILS_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_LUDII_JNIUTILS_H_

#include <cstring>
#include <string>

#include "jni.h"  // NOLINT

namespace open_spiel {
namespace ludii {

class JNIUtils {
 public:
  JNIUtils(const std::string jar_location);
  ~JNIUtils();

  JNIEnv *GetEnv() const;

  void InitJVM(std::string jar_location);
  void CloseJVM();

 private:
  JavaVM *jvm;
  JNIEnv *env;
  jint res;
};

}  // namespace ludii
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_LUDII_JNIUTILS_H_
