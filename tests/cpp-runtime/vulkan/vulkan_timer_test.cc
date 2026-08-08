/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "../../../src/backend/vulkan/runtime/vulkan_timer.h"

#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/timer.h>

#include <iostream>

#include "../../../src/backend/vulkan/runtime/vulkan_device_api.h"

using namespace tvm::runtime;
using namespace tvm::runtime::vulkan;

namespace {

constexpr int32_t kBufferSize = 1024;

tvm::Device VulkanTestDevice() {
  VulkanDeviceAPI* api = VulkanDeviceAPI::Global();
  return tvm::Device{kDLVulkan, api->GetActiveDeviceID()};
}

/*! \brief Queues a host->device copy and flushes it, so a timer has work to measure. */
void RunGpuWork(tvm::Device dev) {
  Tensor src = Tensor::Empty({kBufferSize}, {kDLInt, 32, 1}, {kDLCPU, 0});
  Tensor dst = Tensor::Empty({kBufferSize}, {kDLInt, 32, 1}, dev);
  for (int32_t i = 0; i < kBufferSize; ++i) {
    static_cast<int32_t*>(src->data)[i] = i;
  }
  src.CopyTo(dst);
  // Force Vulkan to flush and sync the queued work.
  VulkanDeviceAPI::Global()->StreamSync(dev, nullptr);
}

}  // namespace

// Timer::Start() resolves the device timer through the global function
// "runtime.timer.<device name>".  If that registration is missing or misnamed,
// it silently falls back to DefaultTimerNode (host wall-clock), and every other
// assertion in this file would still pass while VulkanTimerNode never runs.
TEST(VulkanTimerNode, RegisteredUnderRuntimeTimerName) {
  ASSERT_TRUE(tvm::ffi::Function::GetGlobal("runtime.timer.vulkan").has_value())
      << "VulkanTimerNode is not registered under the name Timer::Start() looks up.";

  Timer timer = Timer::Start(VulkanTestDevice());
  timer->Stop();
  EXPECT_TRUE(timer->IsInstance<VulkanTimerNode>())
      << "Timer::Start() on a Vulkan device did not return a VulkanTimerNode.";
}

TEST(VulkanTimerNode, TimerCorrectness) {
  tvm::Device dev = VulkanTestDevice();

  // Construct the node directly so this exercises VulkanTimerNode regardless of
  // how Timer::Start() dispatches.
  auto node = tvm::ffi::make_object<VulkanTimerNode>(dev);
  bool uses_gpu_timestamps = node->UsesGpuTimestamps();
  Timer timer(node);

  timer->Start();
  RunGpuWork(dev);
  timer->Stop();
  int64_t elapsed_nanos = timer->SyncAndGetElapsedNanos();

  std::cout << "Elapsed time (nanoseconds): " << elapsed_nanos
            << (uses_gpu_timestamps ? " [gpu timestamps]" : " [host fallback]") << std::endl;

  // Check that some time was measured
  ASSERT_GT(elapsed_nanos, 0);
  // A single 4 KB copy cannot plausibly take a second; a value this large means
  // the raw timestamps were misinterpreted (e.g. undefined high bits not masked
  // off, or a bad timestampPeriod).
  ASSERT_LT(elapsed_nanos, 1000000000LL);
}

// A timer that is dropped without Stop() must not leave commands referencing a
// destroyed query pool, or a captured `this`, queued on the stream.
TEST(VulkanTimerNode, DestroyedWithoutStop) {
  tvm::Device dev = VulkanTestDevice();

  { Timer timer = Timer::Start(dev); }

  // Would submit a command buffer referencing the destroyed query pool.
  RunGpuWork(dev);

  Timer timer = Timer::Start(dev);
  RunGpuWork(dev);
  timer->Stop();
  ASSERT_GT(timer->SyncAndGetElapsedNanos(), 0);
}

// Back-to-back measurements on the same node must each reset the query pool.
TEST(VulkanTimerNode, ReusableAcrossMeasurements) {
  tvm::Device dev = VulkanTestDevice();
  Timer timer(tvm::ffi::make_object<VulkanTimerNode>(dev));

  for (int i = 0; i < 3; ++i) {
    timer->Start();
    RunGpuWork(dev);
    timer->Stop();
    ASSERT_GT(timer->SyncAndGetElapsedNanos(), 0) << "measurement " << i << " returned no time";
  }
}
