// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "follow_me/template_pool.hpp"

namespace follow_me
{

TemplatePool::TemplatePool(float update_threshold, float match_threshold, float drift_threshold)
  : initialized_(false)
  , has_latest_template_(false)
  , update_threshold_(update_threshold)
  , match_threshold_(match_threshold)
  , drift_threshold_(drift_threshold)
{
}

void TemplatePool::initialize(const std::vector<float> & initial_feature)
{
  initial_template_ = initial_feature;
  initialized_ = true;
  has_latest_template_ = false;
  latest_template_.clear();
}

bool TemplatePool::tryUpdate(const std::vector<float> & feature, float similarity)
{
  if (!initialized_) {
    return false;
  }

  // Only update when similarity is not above the update threshold
  if (similarity <= update_threshold_) {
    latest_template_ = feature;
    has_latest_template_ = true;
    return true;
  }

  return false;
}

bool TemplatePool::match(const std::vector<float> & /* feature */,
    float similarity_with_initial,
    float similarity_with_latest)
{
  if (!initialized_) {
    return false;
  }

  // Check if drift has occurred
  if (isDrifted(similarity_with_initial)) {
    return false;
  }

  // Match with initial template
  if (similarity_with_initial <= match_threshold_) {
    return true;
  }

  // If there's a latest template, also check matching with the latest template
  if (has_latest_template_ && similarity_with_latest <= match_threshold_) {
    return true;
  }

  return false;
}

bool TemplatePool::isDrifted(float similarity_with_initial) const
{
  if (!initialized_) {
    return true;
  }

  // Consider drift occurred when similarity with initial template is above anti-drift threshold
  return similarity_with_initial > drift_threshold_;
}

void TemplatePool::reset()
{
  initialized_ = false;
  has_latest_template_ = false;
  initial_template_.clear();
  latest_template_.clear();
}

std::optional<std::vector<float>> TemplatePool::getLatestTemplate() const
{
  if (has_latest_template_) {
    return latest_template_;
  }
  return std::nullopt;
}

}  // namespace follow_me
