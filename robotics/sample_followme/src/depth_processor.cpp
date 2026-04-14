// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "follow_me/depth_processor.hpp"

#include <algorithm>
#include <cmath>

namespace follow_me
{

std::optional<float> DepthProcessor::computeDepthMedian(const cv::Mat & depth_image,
    const BoundingBox & bbox,
    int rgb_width,
    int rgb_height)
{
  if (depth_image.empty()) {
    return std::nullopt;
  }

  int depth_width = depth_image.cols;
  int depth_height = depth_image.rows;

  // Calculate scaling ratio from RGB to depth image
  float scale_x = static_cast<float>(depth_width) / static_cast<float>(rgb_width);
  float scale_y = static_cast<float>(depth_height) / static_cast<float>(rgb_height);

  // Convert bbox coordinates from RGB image coordinate system to depth image coordinate system
  int x1 = static_cast<int>(bbox.x1 * scale_x);
  int y1 = static_cast<int>(bbox.y1 * scale_y);
  int x2 = static_cast<int>(bbox.x2 * scale_x);
  int y2 = static_cast<int>(bbox.y2 * scale_y);

  // Ensure bounding box is within depth image range
  x1 = std::max(0, std::min(depth_width - 1, x1));
  y1 = std::max(0, std::min(depth_height - 1, y1));
  x2 = std::max(0, std::min(depth_width, x2));
  y2 = std::max(0, std::min(depth_height, y2));

  if (x1 >= x2 || y1 >= y2) {
    return std::nullopt;
  }

  // Collect depth values
  std::vector<float> depth_values;
  depth_values.reserve((x2 - x1) * (y2 - y1));

  for (int y = y1; y < y2; ++y) {
    for (int x = x1; x < x2; ++x) {
      float depth = 0.0f;

      // Read depth value based on depth image type
      if (depth_image.type() == CV_16UC1) {
        // Depth value unit is millimeters
        uint16_t depth_mm = depth_image.at<uint16_t>(y, x);
        depth = static_cast<float>(depth_mm) / 1000.0f;  // Convert to meters
      } else if (depth_image.type() == CV_32FC1) {
        // Depth value unit is meters
        depth = depth_image.at<float>(y, x);
      } else {
        continue;
      }

      // Filter invalid depth values
      if (depth > 0.1f && depth < 10.0f) {  // Valid range: 0.1m - 10m
        depth_values.push_back(depth);
      }
    }
  }

  if (depth_values.empty()) {
    return std::nullopt;
  }

  // Calculate median
  return computeMedian(depth_values);
}

float DepthProcessor::computeIOU(const BoundingBox & bbox1, const BoundingBox & bbox2)
{
  // Calculate intersection
  int x1 = std::max(bbox1.x1, bbox2.x1);
  int y1 = std::max(bbox1.y1, bbox2.y1);
  int x2 = std::min(bbox1.x2, bbox2.x2);
  int y2 = std::min(bbox1.y2, bbox2.y2);

  if (x1 >= x2 || y1 >= y2) {
    return 0.0f;
  }

  int intersection = (x2 - x1) * (y2 - y1);
  int area1 = bbox1.area();
  int area2 = bbox2.area();
  int union_area = area1 + area2 - intersection;

  if (union_area == 0) {
    return 0.0f;
  }

  return static_cast<float>(intersection) / static_cast<float>(union_area);
}

float DepthProcessor::computeDistanceToCenter(const BoundingBox & bbox,
    int image_width,
    int image_height)
{
  int center_x = bbox.centerX();
  int center_y = bbox.centerY();
  int image_center_x = image_width / 2;
  int image_center_y = image_height / 2;

  float dx = static_cast<float>(center_x - image_center_x) / image_width;
  float dy = static_cast<float>(center_y - image_center_y) / image_height;

  return std::sqrt(dx * dx + dy * dy);
}

float DepthProcessor::computeAngleToTarget(const BoundingBox & bbox,
    int image_width,
    float fov_horizontal)
{
  int bbox_center_x = bbox.centerX();
  int image_center_x = image_width / 2;

  // Calculate horizontal offset
  float offset = static_cast<float>(bbox_center_x - image_center_x);

  // Convert to angle
  float angle = (offset / image_center_x) * (fov_horizontal / 2.0f);

  return angle;
}

std::vector<float> DepthProcessor::filterValidDepths(const std::vector<float> & depth_values)
{
  std::vector<float> valid_depths;
  valid_depths.reserve(depth_values.size());

  for (float depth : depth_values) {
    if (depth > 0.1f && depth < 10.0f) {
      valid_depths.push_back(depth);
    }
  }

  return valid_depths;
}

float DepthProcessor::computeMedian(std::vector<float> values)
{
  if (values.empty()) {
    return 0.0f;
  }

  size_t n = values.size();
  std::nth_element(values.begin(), values.begin() + n / 2, values.end());

  if (n % 2 == 0) {
    float median1 = values[n / 2];
    std::nth_element(values.begin(), values.begin() + n / 2 - 1, values.end());
    float median2 = values[n / 2 - 1];
    return (median1 + median2) / 2.0f;
  } else {
    return values[n / 2];
  }
}

}  // namespace follow_me
