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

#include "vulkan_timer.h"

#include <tvm/runtime/logging.h>

#include <exception>
#include <mutex>

#include "vulkan_common.h"
#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanTimerNode::VulkanTimerNode(Device dev) {
  vk_device_ = &VulkanDeviceAPI::Global()->device(dev.device_id);

  const auto& prop = vk_device_->device_properties;
  use_host_timer_ = !prop.supports_timestamp_queries;
  if (use_host_timer_) {
    static std::once_flag warning_flag;
    std::call_once(warning_flag, [&]() {
      LOG(WARNING) << "Vulkan device " << prop.device_name
                   << " does not support timestamp queries on its compute queue, "
                   << "falling back to host-side timing.  Measurements may be inaccurate "
                   << "or have extra overhead.";
    });
    return;
  }

  timestamp_period_ = prop.timestamp_period;
  timestamp_valid_bits_ = prop.timestamp_valid_bits;
  CreateQueryPool();
}

VulkanTimerNode::~VulkanTimerNode() {
  if (query_pool_ == VK_NULL_HANDLE) return;

  if (start_recorded_ && !synchronized_) {
    // Start()/Stop() may have left commands that write into this query pool queued in
    // the stream, either not yet recorded into the command buffer or not yet executed.
    // Both must be flushed before the pool can be destroyed.
    try {
      stream_->Synchronize();
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to synchronize the Vulkan stream while destroying a timer: "
                   << e.what();
    }
  }

  vkDestroyQueryPool(*vk_device_, query_pool_, nullptr);
}

void VulkanTimerNode::CreateQueryPool() {
  VkQueryPoolCreateInfo query_pool_info{};
  query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_pool_info.queryCount = kNumQueries;

  VULKAN_CALL(vkCreateQueryPool(*vk_device_, &query_pool_info, nullptr, &query_pool_));
}

void VulkanTimerNode::Start() {
  stream_ = &vk_device_->ThreadLocalStream();

  if (use_host_timer_) {
    stream_->Synchronize();
    host_start_ = std::chrono::high_resolution_clock::now();
    return;
  }

  VkQueryPool query_pool = query_pool_;
  stream_->Launch([query_pool](VulkanStreamState* state) {
    // A query must be reset before it is written to.
    vkCmdResetQueryPool(state->cmd_buffer_, query_pool, 0, kNumQueries);
    // Both timestamps are written at the bottom of the pipeline, so the interval
    // measured is the time between the completion of the commands preceding Start() and
    // the completion of the commands preceding Stop().
    vkCmdWriteTimestamp(state->cmd_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool,
                        kStartQuery);
  });
  start_recorded_ = true;
}

void VulkanTimerNode::Stop() {
  TVM_FFI_ICHECK(stream_ != nullptr) << "VulkanTimerNode::Stop called before Start.";

  if (use_host_timer_) {
    stream_->Synchronize();
    duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - host_start_)
                    .count();
    return;
  }

  VkQueryPool query_pool = query_pool_;
  stream_->Launch([query_pool](VulkanStreamState* state) {
    vkCmdWriteTimestamp(state->cmd_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool,
                        kStopQuery);
  });
  stop_recorded_ = true;
}

int64_t VulkanTimerNode::SyncAndGetElapsedNanos() {
  if (use_host_timer_) return duration_;

  // Reading a query that was never written would block forever below.
  TVM_FFI_ICHECK(stop_recorded_)
      << "VulkanTimerNode::SyncAndGetElapsedNanos called before Start and Stop.";

  // The commands recorded by Start()/Stop() may still be queued in the stream, so submit
  // them and wait for the GPU before reading the timestamps back.
  stream_->Synchronize();
  synchronized_ = true;

  duration_ = CollectTimestamps();
  return duration_;
}

int64_t VulkanTimerNode::CollectTimestamps() {
  uint64_t timestamps[kNumQueries] = {0};

  VULKAN_CALL(vkGetQueryPoolResults(*vk_device_, query_pool_, 0, kNumQueries, sizeof(timestamps),
                                    timestamps, sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

  // Only the lowest timestamp_valid_bits_ of each timestamp are meaningful, and the
  // counter wraps around within those bits.  Masking the difference therefore gives the
  // correct number of ticks even if the counter wrapped between the two timestamps.
  uint64_t mask = ~uint64_t(0);
  if (timestamp_valid_bits_ < 64) {
    mask = (uint64_t(1) << timestamp_valid_bits_) - 1;
  }
  uint64_t ticks = (timestamps[kStopQuery] - timestamps[kStartQuery]) & mask;

  return static_cast<int64_t>(static_cast<double>(ticks) * timestamp_period_);
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
