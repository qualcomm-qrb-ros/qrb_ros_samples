// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef FOLLOW_ME__TEMPLATE_POOL_HPP_
#define FOLLOW_ME__TEMPLATE_POOL_HPP_

#include <memory>
#include <optional>
#include <vector>

namespace follow_me
{

/**
 * @brief Template pool management class
 * Capacity of 2: initial template + latest high-quality template
 */
class TemplatePool
{
public:
  /**
   * @brief Constructor
   * @param update_threshold Template update threshold (default 0.85)
   * @param match_threshold Matching threshold (default 0.75)
   * @param drift_threshold Anti-drift threshold (default 0.5)
   */
  TemplatePool(float update_threshold, float match_threshold, float drift_threshold);

  ~TemplatePool() = default;

  /**
   * @brief Initialize template pool
   * @param initial_feature Initial feature vector
   */
  void initialize(const std::vector<float> & initial_feature);

  /**
   * @brief Try to update template pool
   * @param feature New feature vector
   * @param similarity Similarity with initial template
   * @return Whether template was updated
   */
  bool tryUpdate(const std::vector<float> & feature, float similarity);

  /**
   * @brief Match with template pool
   * @param feature Feature to match
   * @param similarity_with_initial Similarity with initial template
   * @param similarity_with_latest Similarity with latest template (if exists)
   * @return Whether matching succeeded
   */
  bool match(const std::vector<float> & feature,
      float similarity_with_initial,
      float similarity_with_latest);

  /**
   * @brief Check if drift occurred
   * @param similarity_with_initial Similarity with initial template
   * @return Whether drift occurred
   */
  bool isDrifted(float similarity_with_initial) const;

  /**
   * @brief Reset template pool
   */
  void reset();

  /**
   * @brief Get initial template
   */
  const std::vector<float> & getInitialTemplate() const { return initial_template_; }

  /**
   * @brief Get latest template (if exists)
   */
  std::optional<std::vector<float>> getLatestTemplate() const;

  /**
   * @brief Whether initialized
   */
  bool isInitialized() const { return initialized_; }

  /**
   * @brief Whether has latest template
   */
  bool hasLatestTemplate() const { return has_latest_template_; }

private:
  bool initialized_;
  bool has_latest_template_;

  std::vector<float> initial_template_;  // Initial template (permanently retained)
  std::vector<float> latest_template_;   // Latest high-quality template

  float update_threshold_;
  float match_threshold_;
  float drift_threshold_;
};

}  // namespace follow_me

#endif  // FOLLOW_ME__TEMPLATE_POOL_HPP_
