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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_TIMER_H_
#define TVM_RUNTIME_VULKAN_VULKAN_TIMER_H_

#include <tvm/runtime/timer.h>

#include <chrono>

#include "vulkan/vulkan_core.h"
#include "vulkan_device.h"
#include "vulkan_stream.h"

namespace tvm {
namespace runtime {
namespace vulkan {

/*!
 * \brief Timer node that measures GPU execution time using Vulkan timestamp queries.
 *
 * `Start()` and `Stop()` record timestamps into a two-entry query pool through the
 * VulkanStream of the calling thread, so the measured interval covers exactly the
 * commands that are submitted to that stream in between.  Reading the timestamps back
 * requires the GPU to have finished executing them, and is therefore deferred to
 * `SyncAndGetElapsedNanos()`, the only blocking call.
 *
 * If the compute queue family used by TVM does not support timestamp queries
 * (`VkQueueFamilyProperties::timestampValidBits == 0`), this timer falls back to
 * host-side timing around a stream synchronization.  That is less accurate, but is
 * always available and matches the behavior of the default TVM timer.
 */
class VulkanTimerNode : public TimerNode {
 public:
  /*!
   * \brief Construct a timer for the given device.
   * \param dev The TVM device to be timed.  Must be a Vulkan device.
   */
  explicit VulkanTimerNode(Device dev);

  /*! \brief Destructor, releases the query pool. */
  ~VulkanTimerNode() override;

  /*! \brief Record the timestamp that marks the start of the measured interval. */
  void Start() override;

  /*! \brief Record the timestamp that marks the end of the measured interval. */
  void Stop() override;

  /*!
   * \brief Wait for the recorded commands to complete and return the elapsed time.
   * \return The time in nanoseconds between `Start` and `Stop`.
   */
  int64_t SyncAndGetElapsedNanos() override;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.vulkan.VulkanTimerNode", VulkanTimerNode, TimerNode);

 private:
  /*! \brief Number of queries held by the pool, one for `Start` and one for `Stop`. */
  static constexpr uint32_t kNumQueries = 2;
  /*! \brief Index of the query written by `Start`. */
  static constexpr uint32_t kStartQuery = 0;
  /*! \brief Index of the query written by `Stop`. */
  static constexpr uint32_t kStopQuery = 1;

  /*! \brief Create the timestamp query pool. */
  void CreateQueryPool();

  /*!
   * \brief Read both timestamps back and convert their difference to nanoseconds.
   *
   * Must only be called once the commands recorded by `Start`/`Stop` have completed.
   *
   * \return The elapsed time in nanoseconds.
   */
  int64_t CollectTimestamps();

  /*! \brief The Vulkan device being timed, owned by the VulkanDeviceAPI. */
  VulkanDevice* vk_device_{nullptr};
  /*!
   * \brief The stream the timestamps are recorded into, set by `Start`.
   *
   * Captured once so that `Stop`, `SyncAndGetElapsedNanos` and the destructor all act on
   * the stream that holds the recorded commands, even if they run on another thread.
   */
  VulkanStream* stream_{nullptr};
  /*! \brief The timestamp query pool, VK_NULL_HANDLE when timing on the host. */
  VkQueryPool query_pool_{VK_NULL_HANDLE};
  /*! \brief Number of nanoseconds per timestamp tick. */
  float timestamp_period_{0.0f};
  /*! \brief Number of meaningful low-order bits in a timestamp. */
  uint32_t timestamp_valid_bits_{0};
  /*! \brief Whether to fall back to host-side timing. */
  bool use_host_timer_{false};
  /*! \brief Whether `Start` has recorded commands writing into the query pool. */
  bool start_recorded_{false};
  /*! \brief Whether `Stop` has recorded commands writing into the query pool. */
  bool stop_recorded_{false};
  /*! \brief Whether those commands have been waited on. */
  bool synchronized_{false};
  /*! \brief Start of the measured interval, used only by the host-side fallback. */
  std::chrono::high_resolution_clock::time_point host_start_;
  /*! \brief The measured duration in nanoseconds. */
  int64_t duration_{0};
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_TIMER_H_
