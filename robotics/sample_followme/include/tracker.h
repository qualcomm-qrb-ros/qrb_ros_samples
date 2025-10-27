/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: BSD-3-Clause-Clear

*/

#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <chrono>
#include <future>
#include <geometry_msgs/msg/twist.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <net.h>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/laser_scan.hpp>

#define TRACKING_TYPE_CV 1
#define TRACKING_TYPE_LIDAR 2
#define TRACKING_TYPE_DEFAULT TRACKING_TYPE_LIDAR

#define TRACKING_CV_PERSON 0
#define TRACKING_METHOD_DEFAULT TRACKING_CV_PERSON

// ncnn model parameters
#define NCNN_INPUT_NAME ("input.1")
#define NCNN_OUTPUT_NAME ("758")
#define NCNN_MODEL_INPUT_WIDTH (352)
#define NCNN_MODEL_INPUT_HEIGTH (352)
#define NCNN_MODEL_THRESH (0.7)
#define LABEL_PERSON ("person")

#define IMAGE_WIDTH frame_.rows
#define IMAGE_HEIGHT frame_.cols

typedef struct cam_intrinsics {
  int width;
  int height;
  float ppx;
  float ppy;
  float fx;
  float fy;
  float coeffs[5];
} cam_intrinsics;

class TargetBox {
public:
  int GetWidth() { return (x2 - x1); };
  int GetHeight() { return (y2 - y1); };

  TargetBox() {}
  TargetBox(int _x1, int _y1, int _x2, int _y2, int _category = 0,
            float _score = 0)
      : x1(_x1), y1(_y1), x2(_x2), y2(_y2), category(_category), score(_score) {
  }

  int x1;
  int y1;
  int x2;
  int y2;

  int category;
  float score;

  float area() { return GetWidth() * GetHeight(); };
};

class MyTracker {
public:
  MyTracker();
  MyTracker(int type, int method, std::string model_param_file,
            std::string model_bin_file, std::string label_file);
  ~MyTracker();
  MyTracker(const MyTracker &) = delete;

  bool trackingOnce();
  void getTargetPose(float &theta, float &d);
  void stopTracking();
  void getImage_thread(bool &running);
  void getImageOnce(const cv::Mat &m);
  void getDepthOnce(const cv::Mat &m);
  void setIntrinsics(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void camLock();
  void camUnlock();
  void kcfTrackingInit(bool &running);

  float dist__m = 0.0;
  float theta__m = 0.0;

private:
  void initCvTracker(void);
  bool cvPersionTracking(const cv::Mat &img, cv::Rect &bbox);
  bool cvTemplateTracking(const cv::Mat &img, cv::Rect &bbox);
  bool cvKcfTracking(const cv::Mat &img, cv::Rect &bbox);

  void getTargetDistance(cv::Rect &bbox, float &horizon_dist_to_target);
  bool trackingTargetSelect(std::vector<TargetBox> nms_boxes,
                            cv::Rect &targetRect);

  void deproject_pixel_to_point(float point[3], const cam_intrinsics *intrin,
                                const float pixel[2], float depth);

  void calculateDepthAndAngle(const cv::Rect &bbox);

  /* params */
  int tracking_type;
  int tracking_method;

  /* camera input data */
  // uint16_t depth[IMAGE_WIDTH * IMAGE_HEIGHT];
  cv::Mat frame_;
  cv::Mat depth_;
  std::mutex camMutex;
  /* cv tracking result */
  cv::Rect bbox;
  std::vector<TargetBox> target_boxes;

  /* matchTemp */
  bool initFlag;
  cv::Mat lastFrame;
  cv::Rect lastBbox;
  /* cv tracking center */
  cv::Point last_box_center;
  cv::Mat lastROI;
  // float lastDistance;
  int tracking_failure_num;
  // int distance_error_num;

  /* tracking person */
  ncnn::Net net;
  float lastTheta;

  // cncc model file
  std::string model_param_file;
  std::string model_bin_file;
  std::string label_file;
  float thresh;
  int input_width;
  int input_height;
  std::vector<std::string> labels;
  std::vector<int> labelsIndexToDetect;
  uint8_t *data_int8;
  float *data_f32;

  float minDis;
  float minDisAngle;
  cam_intrinsics intrinsics_;
  bool is_init_ = false;
};

#endif
