/*
/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: BSD-3-Clause-Clear

Copyright (c) 2022, xuehao.ma All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "tracker.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>

#include "benchmark.h"
#include "common.h"
#include "cpu.h"
#include "rclcpp/rclcpp.hpp"

using namespace cv;
using namespace std;

float Sigmoid(float x)
{
  return 1.0f / (1.0f + exp(-x));
}

float Tanh(float x)
{
  return 2.0f / (1.0f + exp(-2 * x)) - 1;
}

// 用于计算两个目标框面积的交集
float IntersectionArea(const TargetBox & a, const TargetBox & b)
{
  if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
    // no intersection
    return 0.f;
  }

  float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

  return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b)
{
  return (a.score > b.score);
}

// NMS handle
int nmsHandle(std::vector<TargetBox> & src_boxes,
    std::vector<TargetBox> & dst_boxes,
    int rows,
    int cols)
{
  std::vector<int> picked;

  sort(src_boxes.begin(), src_boxes.end(), scoreSort);

  for (int i = 0; i < src_boxes.size(); i++) {
    int keep = 1;
    for (int j = 0; j < picked.size(); j++) {
      // intersection
      float inter_area = IntersectionArea(src_boxes[i], src_boxes[picked[j]]);
      // union
      float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
      float IoU = inter_area / union_area;

      if (IoU > 0.45 && src_boxes[i].category == src_boxes[picked[j]].category) {
        keep = 0;
        break;
      }
    }

    if (keep) {
      picked.push_back(i);
    }
  }

  for (int i = 0; i < picked.size(); i++) {
    src_boxes[picked[i]].x1 = src_boxes[picked[i]].x1 > 0 ? src_boxes[picked[i]].x1 : 0;
    src_boxes[picked[i]].y1 = src_boxes[picked[i]].y1 > 0 ? src_boxes[picked[i]].y1 : 0;
    src_boxes[picked[i]].x2 =
        src_boxes[picked[i]].x2 >= src_boxes[picked[i]].x1 ? src_boxes[picked[i]].x2 : cols - 2;
    src_boxes[picked[i]].y2 =
        src_boxes[picked[i]].y2 >= src_boxes[picked[i]].y1 ? src_boxes[picked[i]].y2 : rows - 2;
    dst_boxes.push_back(src_boxes[picked[i]]);
  }

  return 0;
}

// 最大化交并比(IoU)处理
int maxIoU_Handle(std::vector<TargetBox> & src_boxes, TargetBox & target_boxes, float & maxIoU)
{
  float maxArea = 0;
  int index = 0;

  for (int i = 0; i < src_boxes.size(); i++) {
    // intersection
    float inter_area = IntersectionArea(src_boxes[i], target_boxes);
    // union
    //  float union_area = src_boxes[i].area() + target_boxes.area() -
    //  inter_area; float IoU = inter_area / union_area;
    float IoU = inter_area / target_boxes.area();

    if (IoU > maxArea) {
      maxArea = IoU;
      index = i;
    }
  }

  maxIoU = maxArea;

  return index;
}

template <typename T>
std::vector<int> argsort(const std::vector<T> & array)
{
  const int array_len(array.size());
  std::vector<int> array_index(array_len, 0);
  for (int i = 0; i < array_len; ++i)
    array_index[i] = i;

  std::sort(array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

  return array_index;
}

bool imageMatch(const cv::Mat & roi, const cv::Mat & frame, cv::Point & point, double & score)
{
  int method = TM_CCOEFF_NORMED;
  cv::Point minLoc, maxLoc;
  double minScore, maxScore;
  cv::Mat resultFrame;

  int result_cols = frame.cols - roi.cols + 1;
  int result_rows = frame.rows - roi.rows + 1;
  resultFrame.create(result_rows, result_cols, CV_32FC1);

  matchTemplate(frame, roi, resultFrame, method);

  minMaxLoc(resultFrame, &minScore, &maxScore, &minLoc, &maxLoc, Mat());

  if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED) {
    point = minLoc;
    score = 1.0f - minScore;  // convert the result
  } else {
    point = maxLoc;
    score = maxScore;
  }

  if (score > 0.65)
    return true;
  else
    return false;
}

/* MyTracker class */
MyTracker::MyTracker()
{
  tracking_type = TRACKING_TYPE_DEFAULT;
  tracking_method = TRACKING_METHOD_DEFAULT;
}

MyTracker::MyTracker(int type,
    int method,
    std::string model_param_file,
    std::string model_bin_file,
    std::string label_file)
  : tracking_type(type)
  , tracking_method(method)
  , model_param_file(model_param_file)
  , model_bin_file(model_bin_file)
  , label_file(label_file)
{
  if (type == TRACKING_TYPE_CV) {
    initCvTracker();
  }
}

MyTracker::~MyTracker() {}

void MyTracker::initCvTracker(void)
{
  if (tracking_method == TRACKING_CV_PERSON) {
    // ncnn::set_cpu_powersave(2);
    // net.opt.num_threads = 1;
    net.opt.num_threads = 1;
    ncnn::CpuSet mask;
    mask.enable(7);
    // mask.enable(5);
    // mask.enable(6);
    ncnn::set_cpu_thread_affinity(mask);

    if (net.load_param(model_param_file.c_str())) {
      cout << "Failed to load the model param file." << std::endl;
      exit(-1);
    }

    if (net.load_model(model_bin_file.c_str())) {
      cout << "Failed to load the model bin file." << std::endl;
      exit(-1);
    }

    input_width = NCNN_MODEL_INPUT_WIDTH;
    input_height = NCNN_MODEL_INPUT_HEIGTH;
    thresh = NCNN_MODEL_THRESH;

    std::ifstream file(label_file);
    if (!file) {
      std::cerr << "Failed to read " << label_file << "." << std::endl;
      exit(-1);
    }

    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
      labels.push_back(line);
      if (line == LABEL_PERSON) {
        labelsIndexToDetect.push_back(index);
      }
      index++;
    }
    if (labelsIndexToDetect.size()) {
      for (auto & i : labelsIndexToDetect) {
        std::cout << "index of lable " << labels[i] << "to detect is: " << i << std::endl;
      }
    } else {
      std::cerr << "no matching labels to detect." << std::endl;
      exit(-1);
    }

    initFlag = false;
    tracking_failure_num = 0;
    // distance_error_num = 0;
  }
}
void MyTracker::deproject_pixel_to_point(float point[3],
    const cam_intrinsics * intrin,
    const float pixel[2],
    float depth)
{
  point[0] = (pixel[0] - intrin->ppx) * depth / intrin->fx;
  point[1] = (pixel[1] - intrin->ppy) * depth / intrin->fy;
  point[2] = depth;
}

void MyTracker::getImageOnce(const cv::Mat & m)
{
  frame_ = m;
  std::cout << "color image received: " << std::endl;
}

void MyTracker::getDepthOnce(const cv::Mat & m)
{
  depth_ = m;
  // print("depth_:",depth_)
  // std::cout<<"depth_ is "<<depth_<<std::endl;
}
void MyTracker::setIntrinsics(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  intrinsics_.ppx = msg->k[2];
  intrinsics_.ppy = msg->k[5];
  intrinsics_.fx = msg->k[0];
  intrinsics_.fy = msg->k[4];
  intrinsics_.width = msg->width;
  intrinsics_.height = msg->height;
}

/* MyTracker class */
void MyTracker::kcfTrackingInit(bool & running)
{
#if 0
    std::cout << ">>> kcfTrackingInit done" << std::endl;
#endif
}

// CV tracking
bool MyTracker::trackingOnce(void)
{
  bool ret = false;

  if (tracking_method == TRACKING_CV_PERSON) {
    ret = cvPersionTracking(frame_, bbox);
  }

  return ret;
}

void MyTracker::getTargetDistance(cv::Rect & bbox, float & horizon_dist_to_target)
{
  /* update last_box_center here */
  int box_center_x = bbox.width / 2 + bbox.x;
  int box_center_y = bbox.height / 2 + bbox.y;

  // float dist_to_center = depth_[box_center_x, box_center_y];
  ushort dist_to_center_tmp = depth_.at<ushort>(box_center_y, box_center_x);
  float dist_to_center = float(dist_to_center_tmp) / 1000;
  std::cout << "dist_to_center is " << dist_to_center << std::endl;
  if (dist_to_center == 0.0) {
    std::cout << "WARNING:dist_to_center is 0 ,continue ...";
    horizon_dist_to_target = -1;
    return;
  }
#if 0
    // Cast the frame that arrived to motion frame
    // auto motion = cam.frames.as<rs2::motion_frame>();
    auto accel_frame = cam.frames.first(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    rs2::motion_frame accel = accel_frame.as<rs2::motion_frame>();
    // If casting succeeded and the arrived frame is from accelerometer stream
    if (accel && accel.get_profile().stream_type() == RS2_STREAM_ACCEL &&
        accel.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
    {
        // Get accelerometer measurements
        rs2_vector accel_data = accel.get_motion_data();
        // Holds the angle as calculated from accelerometer data
        float accel_angle_z;
        // Calculate rotation angle from accelerometer data
        accel_angle_z = atan2(accel_data.y, accel_data.z);
        std::cout << "accel angle z is: " << accel_angle_z << ", " << accel_angle_z / 3.1415 * 180 << std::endl;
    }
#endif

  float upixel[2];  // From pixel
  float upoint[3];  // From point (in 3D)

  float vpixel[2];  // To pixel
  float vpoint[3];  // To point (in 3D)

  // Copy pixels into the arrays (to match rsutil signatures)
  upixel[0] = box_center_x;
  upixel[1] = bbox.y + bbox.height / 8;
  vpixel[0] = box_center_x;
  vpixel[1] = bbox.y + bbox.height * 7 / 8;

  // Query the frame for distance
  // Note: this can be optimized
  // It is not recommended to issue an API call for each pixel
  // (since the compiler can't inline these)
  // However, in this example it is not one of the bottlenecks
  ushort udist_tmp = depth_.at<ushort>(static_cast<int>(upixel[0]), static_cast<int>(upixel[1]));
  ushort vdist_tmp = depth_.at<ushort>(static_cast<int>(vpixel[0]), static_cast<int>(vpixel[1]));
  float udist = float(udist_tmp) / 1000;
  float vdist = float(vdist_tmp) / 1000;
  // auto udist = depth_.at<float>(static_cast<int>(upixel[0]),
  // static_cast<int>(upixel[1])); auto vdist =
  // depth_.at<float>(static_cast<int>(vpixel[0]), static_cast<int>(vpixel[1]));

  float udist_pow2 = pow(udist, 2.f);
  float vdist_pow2 = pow(vdist, 2.f);

  // Deproject from pixel to point in 3D
  deproject_pixel_to_point(upoint, &intrinsics_, upixel, udist);
  deproject_pixel_to_point(vpoint, &intrinsics_, vpixel, vdist);

  // Calculate euclidean distance between the two points
  float dist_points_pow2 = pow(upoint[0] - vpoint[0], 2.f) + pow(upoint[1] - vpoint[1], 2.f) +
                           pow(upoint[2] - vpoint[2], 2.f);
  horizon_dist_to_target = sqrt(
      vdist_pow2 - pow((udist_pow2 - vdist_pow2 - dist_points_pow2), 2.f) / 4 / dist_points_pow2);
  // horizon_dist_to_target = dist_to_center;
  // std::cout << ">>> len of two point is: " << sqrt(dist_points_pow2) << " m"
  // << std::endl;
}

bool MyTracker::trackingTargetSelect(std::vector<TargetBox> nms_boxes, cv::Rect & targetRect)
{
  bool ret = false;
  int boxesNum = nms_boxes.size();
  int index = 0, areaMax = 0;
  if (boxesNum) {
    for (int i = 0; i < boxesNum; i++) {
      if (nms_boxes[i].area() > areaMax) {
        index = i;
        areaMax = nms_boxes[i].area();
      }
    }
    cv::Rect rec;
    rec.x = nms_boxes[index].x1 > 0 ? nms_boxes[index].x1 : 0;
    rec.y = nms_boxes[index].y1 > 0 ? nms_boxes[index].y1 : 0;
    rec.width = min(nms_boxes[index].GetWidth(), frame_.cols - rec.x);
    rec.height = min(nms_boxes[index].GetHeight(), frame_.rows - rec.y);

    float horizon_dist_to_target = 0.0;
    getTargetDistance(rec, horizon_dist_to_target);
    if ((int)horizon_dist_to_target == -1) {
      return false;
    }
    if (horizon_dist_to_target > 0.5 && horizon_dist_to_target < 2 &&
        nms_boxes[index].score > 0.8) {
      targetRect = rec;
      ret = true;
      std::cout << " >>> boxes num: " << boxesNum
                << "  horizon_dist_to_target: " << horizon_dist_to_target
                << "  score: " << nms_boxes[index].score << std::endl;
    }
  }
  return ret;
}

void MyTracker::getTargetPose(float & theta, float & d)
{
  if (tracking_type == TRACKING_TYPE_LIDAR) {
    theta = minDisAngle;
    d = minDis;
  } else {
    if (tracking_method == TRACKING_CV_PERSON) {
      /* calc theta and distance */
      // int box_center_x = bbox.width/2 + bbox.x;
      // int diff = IMAGE_WIDTH/2 - box_center_x;
      // float h_fov = 86.0f;
      // theta = h_fov / IMAGE_WIDTH * diff * M_PI / 180.f;

      int box_center_x = bbox.width / 2 + bbox.x;
      float diff = frame_.cols / 2 - box_center_x;
      float h_fov = 86.0f;
      theta = h_fov / frame_.cols * diff * M_PI / 180.f;
#if 0
            if(abs(theta - lastTheta) / 3.1415 * 180 > 45) {
                std::cout << ">>> theta is: " << theta << ", lastTheta is: " << lastTheta << std::endl;
                theta = lastTheta;
            }
            lastTheta = theta;
#endif

      int box_y_min = bbox.y + 5;
      if (box_center_x < 5 || box_center_x > (IMAGE_WIDTH - 5) || box_y_min > (IMAGE_HEIGHT - 6)) {
        d = DEFAULT_TARGET_DISTANCE;  // keep still
        return;
      }

      /* update last_box_center here */
      last_box_center.x = bbox.width / 2 + bbox.x;
      last_box_center.y = bbox.height / 2 + bbox.y;

      float horizon_dist_to_target = 0.0;
      getTargetDistance(bbox, horizon_dist_to_target);
      if ((int)horizon_dist_to_target == -1) {
        d = DEFAULT_TARGET_DISTANCE;
        return;
      }

      std::cout << "horizon_dist_to_target is " << horizon_dist_to_target << std::endl;  // isnan
      if (!isnan(horizon_dist_to_target) && horizon_dist_to_target > MIN_TARGET_DISTANCE &&
          horizon_dist_to_target < MAX_TARGET_DISTANCE) {
        d = horizon_dist_to_target;
      } else {
        d = DEFAULT_TARGET_DISTANCE;  // keep still
      }

      if (abs(box_center_x - last_box_center.x) < 10 && d < (DEFAULT_TARGET_DISTANCE + 0.1) &&
          d > (DEFAULT_TARGET_DISTANCE - 0.1)) {
        d = DEFAULT_TARGET_DISTANCE;  // keep still
      }

      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "image tracking [theta,d] = [%0.3f rad, %0.3f m]",
          theta, horizon_dist_to_target);
    }
  }
}

bool MyTracker::cvPersionTracking(const cv::Mat & m, cv::Rect & bbox_ret)
{
  bool ret = false;

  cv::Mat frame;
  m.copyTo(frame);

  int frame_width = frame.cols;
  int frame_height = frame.rows;
  // std::cout<<"frame_width:"<<frame_width<<"
  // frame_height:"<<frame_height<<std::endl;

  // resize of input image data
  ncnn::Mat input = ncnn::Mat::from_pixels_resize(
      frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows, input_width, input_height);
  // Normalization of input image data
  const float mean_vals[3] = { 0.f, 0.f, 0.f };
  const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
  input.substract_mean_normalize(mean_vals, norm_vals);

  double start = ncnn::get_current_time();

  // creat extractor
  ncnn::Extractor ex = net.create_extractor();

  // set input tensor
  ex.input(NCNN_INPUT_NAME, input);

  // get output tensor
  ncnn::Mat output;
  ex.extract(NCNN_OUTPUT_NAME, output);

  // handle output tensor
  std::vector<TargetBox> output_boxes;

  for (int h = 0; h < output.h; h++) {
    for (int w = 0; w < output.w; w++) {
      // Prospect probability
      int obj_score_index = (0 * output.h * output.w) + (h * output.w) + w;
      float obj_score = output[obj_score_index];

      // Analytic class
      int category;
      float max_score = 0.0f;
      for (size_t i = 0; i < labelsIndexToDetect.size(); i++) {
        int obj_score_index =
            ((5 + labelsIndexToDetect[i]) * output.h * output.w) + (h * output.w) + w;
        float cls_score = output[obj_score_index];
        if (cls_score > max_score) {
          max_score = cls_score;
          category = labelsIndexToDetect[i];
        }
      }
      float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

      // Threshold filtering
      if (score > thresh) {
        // Analytic coordinate
        int x_offset_index = (1 * output.h * output.w) + (h * output.w) + w;
        int y_offset_index = (2 * output.h * output.w) + (h * output.w) + w;
        int box_width_index = (3 * output.h * output.w) + (h * output.w) + w;
        int box_height_index = (4 * output.h * output.w) + (h * output.w) + w;

        float x_offset = Tanh(output[x_offset_index]);
        float y_offset = Tanh(output[y_offset_index]);
        float box_width = Sigmoid(output[box_width_index]);
        float box_height = Sigmoid(output[box_height_index]);

        float cx = (w + x_offset) / output.w;
        float cy = (h + y_offset) / output.h;

        int x1 = (int)((cx - box_width * 0.5) * frame_width);
        int y1 = (int)((cy - box_height * 0.5) * frame_height);
        int x2 = (int)((cx + box_width * 0.5) * frame_width);
        int y2 = (int)((cy + box_height * 0.5) * frame_height);

        output_boxes.push_back(TargetBox{ x1, y1, x2, y2, category, score });
      }
    }
  }

  // NMS processing
  std::vector<TargetBox> nms_boxes;
  nmsHandle(output_boxes, nms_boxes, frame.rows, frame.cols);
  target_boxes = nms_boxes;
  cv::Mat result_frame;
  cv::Rect kcf_result;
  cv::Rect rec;
  frame.copyTo(result_frame);

  // Printing time
  double end = ncnn::get_current_time();
  double time = end - start;
  // printf("Time:%7.2f ms\n",time);

  // draw result
  for (size_t i = 0; i < nms_boxes.size(); i++) {
    TargetBox box = nms_boxes[i];
    cv::rectangle(result_frame, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2),
        cv::Scalar(0, 0, 255), 2);
  }
  // cv::imshow("detection",result_frame);
  // cv::waitKey(1);

  bool found = false;
  cv::Rect best_bbox;
  if (is_init_ == false) {
    if (nms_boxes.size() >= 2) {
      std::cout << "multi people detected, init failed!" << std::endl;
      return false;
    } else if (nms_boxes.size() == 1) {
      is_init_ = true;
    }
  }

  if (nms_boxes.size() >= 1) {
    found = true;
    auto x1 = nms_boxes[0].x1 > 0 ? nms_boxes[0].x1 : 0;
    auto y1 = nms_boxes[0].y1 > 0 ? nms_boxes[0].y1 : 0;
    auto x2 = min(nms_boxes[0].GetWidth(), result_frame.cols - rec.x);
    auto y2 = min(nms_boxes[0].GetHeight(), result_frame.rows - rec.y);
    // best_bbox = cv::Rect(
    //             nms_boxes[0].x1,
    //             nms_boxes[0].y1,
    //             nms_boxes[0].x2,
    //             nms_boxes[0].y2
    //         );
    best_bbox = cv::Rect(x1, y1, x2, y2);
  }

  if (found) {
    calculateDepthAndAngle(best_bbox);
  }
  // if(dist__m<0.1)
  //     std::cout<<"dist__m too low, failed to detect pelple"<<std::endl;
  //     return found;
  return found;

  //     if(nms_boxes.size() == 0)
  //     {
  //         std::cout << "nobody here." << std::endl;
  //         if(initFlag) //False at first
  //             return true;
  //         return false;
  //     }

  //     else if (nms_boxes.size() == 1)  //only one persion
  //     {
  //         std::cout << "One persion detected." << std::endl;
  //         int idx = nms_boxes[0].category;

  //         rec.x = nms_boxes[0].x1 > 0 ? nms_boxes[0].x1 : 0;
  //         rec.y = nms_boxes[0].y1 > 0 ? nms_boxes[0].y1 : 0;
  //         rec.width = min(nms_boxes[0].GetWidth(), frame.cols - rec.x);
  //         rec.height = min(nms_boxes[0].GetHeight(), frame.rows - rec.y);

  //         if(!initFlag) //initFlag False at first
  //         {
  //             if(trackingTargetSelect(nms_boxes, rec)) {

  //                 cv::Rect rectTrackRoi;
  //                 rectTrackRoi.x = rec.x + rec.width / 6;
  //                 rectTrackRoi.y = rec.y + rec.height / 5;
  //                 rectTrackRoi.width = rec.width * 2 / 3;
  //                 rectTrackRoi.height = rec.height / 2 - rec.height / 6;
  //                 if(rectTrackRoi.width <= 0 || rectTrackRoi.height <= 0) {
  //                     std::cout << "!!! kcf ERROR, width is " <<
  //                     rectTrackRoi.width << ", heigth is " <<
  //                     rectTrackRoi.height << std::endl; goto out;
  //                 }
  //                 else if(rectTrackRoi.width + rectTrackRoi.x > frame.cols -
  //                 2 || rectTrackRoi.height + rectTrackRoi.y > frame.rows - 2)
  //                 {
  //                     std::cout << "!!! kcf OVERFLOW, width is " <<
  //                     rectTrackRoi.width << ", heigth is " <<
  //                     rectTrackRoi.height << std::endl; goto out;
  //                 }
  //                 // std::cout << "about to kcf init " << std::endl;
  //                 kcfTracker.init(rectTrackRoi, frame);

  //                 cv::rectangle(result_frame, rectTrackRoi,
  //                 cv::Scalar(0,255,0), 2); std::cout << "kcf init
  //                 successfully " << std::endl;

  //                 initFlag = true;
  //                 ret = true;
  //                 return ret;
  //             }

  //             else
  //             {
  //                 std::cout << "One person, trackingTargetSelect or
  //                 KCFintialize failed!!!" << std::endl; ret = false; return
  //                 ret;
  //             }
  //         }

  //         else
  //         {
  //             goto TRACKING;
  //         }
  //     }

  //     else //more than 1 persion
  //     {
  //         if(!initFlag)
  //         {
  //             std::cout << "Multi people, initialize failed!!!" << std::endl;
  //             ret = false;
  //         }
  //         else
  //         {
  //             TRACKING:
  //                 cv::Rect kcfResult = kcfTracker.update(frame);
  //                 kcfResult.x = kcfResult.x > 0 ? kcfResult.x : 0;
  //                 kcfResult.y = kcfResult.y > 0 ? kcfResult.y : 0;
  //                 if(kcfResult.width <= 0 || kcfResult.height <= 0)
  //                 {
  //                     std::cout << "!!! kcf ERROR, width is " <<
  //                     kcfResult.width << ", heigth is " << kcfResult.height
  //                     << std::endl; goto out;
  //                 }

  //                 else if(kcfResult.width + kcfResult.x > frame.cols - 2 ||
  //                 kcfResult.height + kcfResult.y > frame.rows - 2)
  //                 {
  //                     std::cout << "!!! kcf OVERFLOW, width is " <<
  //                     kcfResult.width << ", heigth is " << kcfResult.height
  //                     << std::endl; goto out;
  //                 }

  //                 cv::rectangle(result_frame, kcfResult, cv::Scalar(255,0,0),
  //                 2); TargetBox kcfBox(kcfResult.x, kcfResult.y, kcfResult.x
  //                 + kcfResult.width, kcfResult.y + kcfResult.height); float
  //                 maxIoU; int index = maxIoU_Handle(nms_boxes, kcfBox,
  //                 maxIoU); if(maxIoU > 0.9 && kcfBox.GetWidth() >
  //                 nms_boxes[index].GetWidth() / 2)
  //                 {
  //                     rec = kcfResult;
  //                     ret = true;
  //                 }

  //                 else if(maxIoU > 0.5)
  //                 {
  //                     rec.x = nms_boxes[index].x1;
  //                     rec.y = nms_boxes[index].y1;
  //                     rec.width = nms_boxes[index].GetWidth();
  //                     rec.height = nms_boxes[index].GetHeight();
  //                     cv::Rect rectTrackRoi;
  //                     rectTrackRoi.x = rec.x + rec.width / 6;
  //                     rectTrackRoi.y = rec.y + rec.height / 5;
  //                     rectTrackRoi.width = rec.width * 2 / 3;
  //                     rectTrackRoi.height = rec.height / 2 - rec.height / 6;
  //                     kcfTracker.init(rectTrackRoi, frame);
  //                     rec = rectTrackRoi;
  //                     std::cout << "warning: fix kcf tracking, maxIoU is " <<
  //                     maxIoU << std::endl;

  //                     ret = true;
  //                 }

  //                 else
  //                 {
  //                     std::cout << "tracking failed due to IOU too low ,
  //                     maxIoU is " << maxIoU << std::endl; ret = false;
  //                 }
  //         }
  //     }

  // out:
  //     /* update lastframe, lastROI and last_box_center */
  //     if(ret)
  //     {
  //         frame.copyTo(lastFrame);
  //         Mat roi(frame, rec);
  //         roi.copyTo(lastROI);
  //         lastBbox = rec;
  //         bbox_ret = rec;
  //     }
  //     else
  //     {

  //     }

  //     return ret;
}

void MyTracker::calculateDepthAndAngle(const cv::Rect & bbox)
{
  // Get depth at center of bounding box
  cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
  // cv::Point center2(bbox.x + bbox.width/3, bbox.y + bbox.height/2);
  //
  int type = depth_.type();
  float depth;
  if (type == CV_32FC1) {
    depth = depth_.at<float>(center);
  } else if (type == CV_16UC1) {
    depth = depth_.at<unsigned short>(center) / 1000.0f;  // Convert mm to meters
  } else {
    std::cout << "Unsupported depth_ type" << type << std::endl;
    return;
  }
  // Calculate angle offset from camera center
  float x_offset = center.x - intrinsics_.ppx;
  float y_offset = center.y - intrinsics_.ppy;
  // float angle_x = atan2(x_offset, intrinsics_.fx) * 180.0 / M_PI;
  // float angle_y = atan2(y_offset, intrinsics_.fy) * 180.0 / M_PI;

  float angle_x = atan2(x_offset, intrinsics_.fx);
  float angle_y = atan2(y_offset, intrinsics_.fy);

  std::cout << "Human detected - Depth: " << depth << ", Angle X: " << angle_x << std::endl;

  dist__m = depth;
  theta__m = -angle_x / 2;
}

bool MyTracker::cvTemplateTracking(const cv::Mat & img, cv::Rect & bbox) {}

bool MyTracker::cvKcfTracking(const cv::Mat & img, cv::Rect & bbox) {}
