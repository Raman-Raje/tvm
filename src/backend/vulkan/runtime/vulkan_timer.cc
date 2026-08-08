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
#include <vector>

#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

namespace {

/*! \brief Number of meaningful bits in a timestamp written by the compute queue.
 *
 * Zero means the queue family cannot write timestamps at all.  Bits above the
 * returned count are undefined and must be masked off before use.
 */
uint32_t TimestampValidBits(const VulkanDevice& device) {
  uint32_t queue_prop_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_prop_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_props(queue_prop_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_prop_count, queue_props.data());

  if (device.queue_family_index >= queue_prop_count) return 0;
  return queue_props[device.queue_family_index].timestampValidBits;
}

}  // namespace

VulkanTimerNode::VulkanTimerNode(Device dev) : dev_(dev) {
  // Get the Vulkan device and stream
  auto& vk_dev = VulkanDeviceAPI::Global()->device(dev_.device_id);
  stream_ = &vk_dev.ThreadLocalStream();
  device_ = vk_dev;

  // Retrieve the timestamp period from device properties
  timestamp_period_ = vk_dev.device_properties.timestamp_period;

  // Timestamp queries are only usable when the queue family we submit to
  // reports a non-zero number of valid bits.  SelectComputeQueueFamily()
  // prefers a compute-only queue family, which is exactly the kind of family
  // that some drivers report as having no timestamp support, so this has to be
  // checked at runtime rather than assumed.
  uint32_t valid_bits = TimestampValidBits(vk_dev);
  use_gpu_timer_ = valid_bits > 0 && timestamp_period_ > 0.0f;
  timestamp_mask_ = valid_bits >= 64 ? ~uint64_t(0) : (uint64_t(1) << valid_bits) - 1;

  if (use_gpu_timer_) {
    CreateQueryPool();
  } else {
    static std::once_flag warned;
    std::call_once(warned, [&]() {
      LOG(WARNING) << "Vulkan device " << vk_dev.device_properties.device_name
                   << " does not support timestamp queries on its compute queue family; "
                   << "falling back to host-side timing, which includes submission overhead.";
    });
  }
}

VulkanTimerNode::~VulkanTimerNode() { Cleanup(); }

void VulkanTimerNode::CreateQueryPool() {
  VkQueryPoolCreateInfo query_pool_info{};
  query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_pool_info.queryCount = 2;

  VkResult res = vkCreateQueryPool(device_, &query_pool_info, nullptr, &query_pool_);
  TVM_FFI_ICHECK(res == VK_SUCCESS) << "Failed to create Vulkan query pool.";
}

void VulkanTimerNode::Start() {
  duration_ = 0;

  if (!use_gpu_timer_) {
    stream_->Synchronize();
    host_start_ = std::chrono::high_resolution_clock::now();
    return;
  }

  // Marks that commands referencing `query_pool_` are queued on the stream but
  // not yet flushed, so that Cleanup() knows it must synchronize first.
  query_pending_ = true;
  stream_->Launch([this](VulkanStreamState* state) {
    // Reset the query pool before writing timestamps
    vkCmdResetQueryPool(state->cmd_buffer_, query_pool_, 0, 2);
    vkCmdWriteTimestamp(state->cmd_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_,
                        start_query_);
  });
}

void VulkanTimerNode::Stop() {
  if (!use_gpu_timer_) {
    stream_->Synchronize();
    duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - host_start_)
                    .count();
    return;
  }

  stream_->Launch([this](VulkanStreamState* state) {
    vkCmdWriteTimestamp(state->cmd_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_,
                        end_query_);
  });

  // Ensure GPU has finished writing timestamps before collecting them
  stream_->Synchronize();
  query_pending_ = false;
  CollectTimestamps();
}

int64_t VulkanTimerNode::SyncAndGetElapsedNanos() { return duration_; }

void VulkanTimerNode::CollectTimestamps() {
  uint64_t timestamps[2] = {0};

  VkResult result =
      vkGetQueryPoolResults(device_, query_pool_, 0, 2, sizeof(timestamps), timestamps,
                            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  TVM_FFI_ICHECK(result == VK_SUCCESS) << "Failed to get Vulkan query pool results.";

  // Only the low `timestampValidBits` of each result are defined, so the high
  // bits must be discarded before subtracting.  Masking the difference as well
  // gives the correct interval across a counter wraparound.
  uint64_t diff = ((timestamps[1] & timestamp_mask_) - (timestamps[0] & timestamp_mask_)) &
                  timestamp_mask_;
  duration_ = static_cast<int64_t>(diff * static_cast<double>(timestamp_period_));
}

void VulkanTimerNode::Cleanup() {
  if (query_pool_ == VK_NULL_HANDLE) return;

  // Start() records commands referencing `query_pool_` onto the stream, and in
  // deferred mode the recorded lambda also captures `this`.  If Stop() was
  // never reached (e.g. the timed function threw), both would outlive this
  // object, so the stream has to be flushed before the pool is released.
  if (query_pending_) {
    query_pending_ = false;
    try {
      stream_->Synchronize();
    } catch (const std::exception& e) {
      // Destructors must not throw; the pool is leaked rather than destroyed
      // while a submitted command buffer may still reference it.
      LOG(WARNING) << "Failed to flush the Vulkan stream while destroying a timer: " << e.what();
      return;
    }
  }

  vkDestroyQueryPool(device_, query_pool_, nullptr);
  query_pool_ = VK_NULL_HANDLE;
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
