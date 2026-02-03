// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// Copyright (c) 2026, Qualcomm Innovation Center, Inc.
// All rights reserved.

#ifndef SAMPLE_FACE_RECOGNITION_LIB_HPP_
#define SAMPLE_FACE_RECOGNITION_LIB_HPP_

#include <dirent.h>
#include <sys/stat.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

struct face_landmark
{
  float bbox_x, bbox_y;
  float width, height;
  float right_eye_x, right_eye_y;
  float left_eye_x, left_eye_y;
  float nose_x, nose_y;
  float right_corner_x, right_corner_y;
  float left_corner_x, left_corner_y;
  float face_score;
  char name[20];
};

namespace sample_face_recognition
{
const std::map<std::string, int> str2backend{ { "opencv", cv::dnn::DNN_BACKEND_OPENCV },
  { "cuda", cv::dnn::DNN_BACKEND_CUDA }, { "timvx", cv::dnn::DNN_BACKEND_TIMVX } };
const std::map<std::string, int> str2target{ { "cpu", cv::dnn::DNN_TARGET_CPU },
  { "cuda", cv::dnn::DNN_TARGET_CUDA }, { "npu", cv::dnn::DNN_TARGET_NPU },
  { "cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16 } };

class FaceDetect
{
public:
  FaceDetect(const std::string & fd_model,
      const Size & input_size,
      float score_threshold = 0.9f,
      float nms_threshold = 0.3f,
      int top_k = 5000,
      const std::string & fr_model = "");
  FaceDetect(const std::string & fd_model, const std::string & fr_model);
  void
  detect(cv::Mat & image, cv::Mat & faces, std::vector<std::string> & names, bool overlay_flag);
  void overlay(cv::Mat & image, cv::Mat & faces, std::vector<std::string> & names);
  void load_database(const std::string & database_path);
  void update_database(const std::string & target_image_path);
  bool add_database(const std::string & image_file);
  void set_similar_thresh(float value);
  bool check_database(const std::string & target);

private:
  std::string fd_model_;
  std::string fr_model_;
  std::string face_features_path_;
  std::string image_data_path_;
  Size input_size_;
  float score_threshold_;
  float nms_threshold_;
  float scale_;
  int top_k_;
  bool save_;
  int once_run_cnt_;
  size_t count_;
  int backend_id_;
  int target_id_;

  float cosine_similar_thresh_;

  Ptr<FaceDetectorYN> face_detector_;
  Ptr<FaceRecognizerSF> face_recognizer_;

  std::vector<Mat> faces_dst_features_;
  std::vector<std::string> faces_dst_names_;

  cv::Mat get_face_data(const std::string & image_path);
  void write_vetor_to_file(const std::vector<Mat> & mats, const std::string & filename);
  void save_name_to_file(const std::vector<std::string> & vec, const std::string & file_path);
  std::string get_file_name_without_extension(const std::string & file_path);
  std::vector<Mat> read_vector_from_file(const std::string & filename);
  std::vector<std::string> read_name_from_file(const std::string & filename);
  void visualize_single_base(Mat & input,
      Mat & faces,
      int idx,
      bool flag,
      std::string id,
      int thickness);
  void visualize_result_id(Mat & input, Mat & faces, int idx, bool flag, std::string id);
  void visualize_result(Mat & input, Mat & faces, int idx, bool flag);
};

}  // namespace sample_face_recognition

#endif  // SAMPLE_FACE_RECOGNITION_LIB_HPP_
