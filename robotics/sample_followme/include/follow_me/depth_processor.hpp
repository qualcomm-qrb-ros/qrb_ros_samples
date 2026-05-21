// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef FOLLOW_ME__DEPTH_PROCESSOR_HPP_
#define FOLLOW_ME__DEPTH_PROCESSOR_HPP_

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

namespace follow_me
{

/**
 * @brief Bounding box structure
 */
struct BoundingBox
{
  int x1, y1, x2, y2;
  float score;

  BoundingBox() : x1(0), y1(0), x2(0), y2(0), score(0.0f) {}
  BoundingBox(int x1_, int y1_, int x2_, int y2_, float score_ = 1.0f)
    : x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(score_)
  {
  }

  int width() const { return x2 - x1; }
  int height() const { return y2 - y1; }
  int centerX() const { return (x1 + x2) / 2; }
  int centerY() const { return (y1 + y2) / 2; }
  int area() const { return width() * height(); }
};

/**
 * @brief Depth processor class
 */
class DepthProcessor
{
public:
  DepthProcessor() = default;
  ~DepthProcessor() = default;

  /**
   * @brief Compute depth median within bounding box
   * @param depth_image Depth image
   * @param bbox Bounding box (based on RGB image coordinates)
   * @param rgb_width RGB image width
   * @param rgb_height RGB image height
   * @return Depth value (meters), returns nullopt if failed
   */
  std::optional<float> computeDepthMedian(const cv::Mat & depth_image,
      const BoundingBox & bbox,
      int rgb_width,
      int rgb_height);

  /**
   * @brief Compute IOU (Intersection over Union)
   * @param bbox1 First bounding box
   * @param bbox2 Second bounding box
   * @return IOU value [0, 1]
   */
  static float computeIOU(const BoundingBox & bbox1, const BoundingBox & bbox2);

  /**
   * @brief Compute distance from bounding box to image center
   * @param bbox Bounding box
   * @param image_width Image width
   * @param image_height Image height
   * @return Normalized distance [0, 1]
   */
  static float computeDistanceToCenter(const BoundingBox & bbox, int image_width, int image_height);

  /**
   * @brief Compute target angle relative to image center
   * @param bbox Bounding box
   * @param image_width Image width
   * @param fov_horizontal Horizontal field of view in radians (default 90 degrees)
   * @return Angle in radians, positive means target is on the right, negative means on the left
   */
  static float computeAngleToTarget(const BoundingBox & bbox,
      int image_width,
      float fov_horizontal = 1.5708f);  // 90 degrees = 1.5708 radians

private:
  /**
   * @brief Filter invalid depth values
   * @param depth_values List of depth values
   * @return Filtered valid depth values
   */
  std::vector<float> filterValidDepths(const std::vector<float> & depth_values);

  /**
   * @brief Compute median value
   * @param values List of values
   * @return Median value
   */
  float computeMedian(std::vector<float> values);
};

}  // namespace follow_me

#endif  // FOLLOW_ME__DEPTH_PROCESSOR_HPP_
