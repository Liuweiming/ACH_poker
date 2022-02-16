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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_DEVICE_MANAGER_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_DEVICE_MANAGER_H_

#include <mutex>
#include <vector>
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

// Keeps track of a bunch of VPNet models, intended to be one per device, and
// gives them out based on usage. When you request a device you specify how much
// work you're going to give it, which is assumed done once the loan is
// returned.
class DeviceManager {
 public:
  DeviceManager() {}

  void AddDevice(CFRNetModel model) {  // Not thread safe.
    devices.emplace_back(Device(std::move(model)));
  }
  void FreezeDevice(int device_id) { devices[device_id].SetFrozen(true); }
  void UnfreezeDevice(int device_id) { devices[device_id].SetFrozen(false); }

  // Acts as a pointer to the model, but lets the manager know when you're
  // done.
  class DeviceLoan {
   public:
    // DeviceLoan is not public constructible and is move only.
    DeviceLoan(DeviceLoan&& other) = default;
    DeviceLoan& operator=(DeviceLoan&& other) = default;
    DeviceLoan(const DeviceLoan&) = delete;
    DeviceLoan& operator=(const DeviceLoan&) = delete;

    ~DeviceLoan() { manager_->Return(device_id_, requests_); }
    CFRNetModel* operator->() { return model_; }
    CFRNetModel* model() { return model_; }

   private:
    DeviceLoan(DeviceManager* manager, CFRNetModel* model, int device_id,
               int requests)
        : manager_(manager),
          model_(model),
          device_id_(device_id),
          requests_(requests) {}
    DeviceManager* manager_;
    CFRNetModel* model_;
    int device_id_;
    int requests_;
    friend DeviceManager;
  };

  // Gives the device with the fewest outstanding requests.
  // Do not return frozen devices, unless requested explicitly.
  DeviceLoan Get(int requests, int device_id = -1) {
    std::unique_lock<std::mutex> lock(m_);
    if (device_id < 0) {
      device_id = 0;
      for (int i = 1; i < devices.size(); ++i) {
        if (!devices[i].frozen() &&
            devices[i].requests < devices[device_id].requests) {
          device_id = i;
        }
      }
    }
    devices[device_id].requests += requests;
    return DeviceLoan(this, &devices[device_id].model, device_id, requests);
  }

  std::vector<DeviceLoan> GetAll() {
    std::vector<DeviceLoan> all;
    for (int i = 0; i < devices.size(); ++i) {
      if (!devices[i].frozen()) {
        all.push_back(DeviceLoan(this, &devices[i].model, i, 0));
      }
    }
    return all;
  }

  int Count() const { return devices.size(); }

 private:
  void Return(int device_id, int requests) {
    std::unique_lock<std::mutex> lock(m_);
    devices[device_id].requests -= requests;
  }

  struct Device {
    CFRNetModel model;
    int requests = 0;
    bool frozen_ = false;
    bool frozen() const { return frozen_; }
    void SetFrozen(bool f) { frozen_ = f; }
    Device(CFRNetModel&& m) : model(std::move(m)) {}
  };

  std::vector<Device> devices;
  std::mutex m_;
};
}
}

#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_DEEP_CFR_DEVICE_MANAGER_H_
