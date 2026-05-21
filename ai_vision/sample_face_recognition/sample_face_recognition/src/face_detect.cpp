// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// Copyright (c) 2026, Qualcomm Innovation Center, Inc.
// All rights reserved.

#include "sample_face_recognition/face_detect.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

#define FACE_FEATURES_FILE "/face_features.bin"
#define FACE_NAME_FILE "/face_names.txt"

namespace sample_face_recognition
{
FaceDetect::FaceDetect(const std::string & fd_model, const std::string & fr_model)
{
  cosine_similar_thresh_ = 0.363;
  fd_model_ = fd_model;
  fr_model_ = fr_model;
  input_size_ = Size(320, 320);
  score_threshold_ = 0.9;
  nms_threshold_ = 0.3;
  top_k_ = 5000;
  backend_id_ = str2backend.at("opencv");
  target_id_ = str2target.at("cpu");

  face_detector_ = FaceDetectorYN::create(fd_model_, "", input_size_, score_threshold_,
      nms_threshold_, top_k_, backend_id_, target_id_);
  face_recognizer_ = FaceRecognizerSF::create(fr_model_, "");
}

FaceDetect::FaceDetect(const std::string & fd_model,
    const Size & input_size,
    float score_threshold,
    float nms_threshold,
    int top_k,
    const std::string & fr_model)

{
  cosine_similar_thresh_ = 0.363;
  fd_model_ = fd_model;
  fr_model_ = fr_model;
  input_size_ = input_size;
  score_threshold_ = score_threshold;
  nms_threshold_ = nms_threshold;
  top_k_ = top_k;
  backend_id_ = str2backend.at("opencv");
  target_id_ = str2target.at("cpu");

  face_detector_ = FaceDetectorYN::create(fd_model_, "", input_size_, score_threshold_,
      nms_threshold_, top_k_, backend_id_, target_id_);
  face_recognizer_ = FaceRecognizerSF::create(fr_model_, "");
}

void FaceDetect::set_similar_thresh(float value)
{
  cosine_similar_thresh_ = value;
}

void FaceDetect::load_database(const std::string & database_path)
{
  faces_dst_features_ = read_vector_from_file(database_path + FACE_FEATURES_FILE);
  faces_dst_names_ = read_name_from_file(database_path + FACE_NAME_FILE);
}

int find_index(const std::vector<std::string> & vec, const std::string & value)
{
  auto it = std::find(vec.begin(), vec.end(), value);
  if (it != vec.end()) {
    return std::distance(vec.begin(), it);
  } else {
    return -1;
  }
}

bool FaceDetect::add_database(const std::string & image_file)
{
  if (!image_file.empty()) {
    cv::Mat face_data = get_face_data(image_file);
    if (!face_data.empty()) {
      int index = find_index(faces_dst_names_, get_file_name_without_extension(image_file));
      if (index == -1) {
        faces_dst_names_.push_back(get_file_name_without_extension(image_file));
        faces_dst_features_.push_back(face_data);
        std::cout << "add new face info: " << get_file_name_without_extension(image_file)
                  << std::endl;
      } else {
        faces_dst_features_[index] = face_data;
        std::cout << "update face info: " << get_file_name_without_extension(image_file)
                  << std::endl;
      }
      std::string database_name = image_data_path_ + FACE_FEATURES_FILE;
      write_vetor_to_file(faces_dst_features_, database_name);
      std::string database_name_txt = image_data_path_ + FACE_NAME_FILE;
      save_name_to_file(faces_dst_names_, database_name_txt);
      return true;
    }
  }
  return false;
}

bool FaceDetect::check_database(const std::string & target)
{
  auto it = std::find(faces_dst_names_.begin(), faces_dst_names_.end(), target);

  return (it != faces_dst_names_.end()) ? true : false;
}

cv::Mat FaceDetect::get_face_data(const std::string & image_path)
{
  cv::Mat faces;
  cv::Mat aligned_face1, aligned_face2;
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  face_detector_->setInputSize(img.size());
  face_detector_->detect(img, faces);
  if (faces.rows < 1) {
    std::cerr << "Cannot find a face in " << image_path << std::endl;
    return cv::Mat();
  }
  face_recognizer_->alignCrop(img, faces.row(0), aligned_face1);
  face_recognizer_->feature(aligned_face1, aligned_face2);
  aligned_face2 = aligned_face2.clone();
  return aligned_face2;
}

void FaceDetect::update_database(const std::string & target_image_path)
{
  std::vector<Mat> feature_all;
  std::vector<std::string> name_all;
  DIR * dir;
  struct dirent * ent;
  std::string path = target_image_path;
  image_data_path_ = target_image_path;

  if ((dir = opendir(path.c_str())) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      std::string file_name = ent->d_name;
      if (file_name.size() >= 4 && file_name.substr(file_name.size() - 4) == ".jpg") {
        std::string full_path = path + "/" + file_name;
        cv::Mat face_data = get_face_data(full_path);
        if (!face_data.empty()) {
          std::cout << "detect people: " << get_file_name_without_extension(file_name) << std::endl;
          name_all.push_back(get_file_name_without_extension(file_name));
          feature_all.push_back(face_data);
        }
      }
    }
    closedir(dir);
    if (!feature_all.empty()) {
      std::string database_name = path + FACE_FEATURES_FILE;
      write_vetor_to_file(feature_all, database_name);
      std::string database_name_txt = path + FACE_NAME_FILE;
      save_name_to_file(name_all, database_name_txt);
      std::cout << "create: " << database_name << " " << database_name_txt << std::endl;
    }
  } else {
    std::cerr << "Could not open directory: " << path << std::endl;
  }
}

std::string FaceDetect::get_file_name_without_extension(const std::string & file_path)
{
  size_t last_slash_pos = file_path.find_last_of("/\\");
  size_t last_dot_pos = file_path.find_last_of('.');

  std::string file_name =
      (last_slash_pos == std::string::npos) ? file_path : file_path.substr(last_slash_pos + 1);

  if (last_slash_pos == std::string::npos && last_dot_pos != std::string::npos) {
    file_name = file_name.substr(0, last_dot_pos);
  }

  if (last_slash_pos != std::string::npos && last_dot_pos != std::string::npos &&
      last_dot_pos > last_slash_pos) {
    file_name = file_name.substr(0, last_dot_pos - last_slash_pos - 1);
  }

  return file_name;
}

void FaceDetect::save_name_to_file(const std::vector<std::string> & vec,
    const std::string & file_path)
{
  std::ofstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return;
  }

  for (const auto & line : vec) {
    file << line << std::endl;
  }

  file.close();
}

void FaceDetect::write_vetor_to_file(const std::vector<Mat> & mats, const std::string & filename)
{
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    std::cerr << "cannot open the file!!!" << std::endl;
    return;
  }

  size_t count = mats.size();
  ofs.write(reinterpret_cast<const char *>(&count), sizeof(count));

  for (const auto & mat : mats) {
    int rows = mat.rows;
    int cols = mat.cols;
    int type = mat.type();
    ofs.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    ofs.write(reinterpret_cast<const char *>(&type), sizeof(type));

    ofs.write(reinterpret_cast<const char *>(mat.data), mat.elemSize() * mat.total());
  }

  ofs.close();
}

std::vector<Mat> FaceDetect::read_vector_from_file(const std::string & filename)
{
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cout << "open face_features file failed" << std::endl;
    return {};
  }

  size_t count;
  ifs.read(reinterpret_cast<char *>(&count), sizeof(count));
  std::vector<Mat> mats;
  for (size_t i = 0; i < count; ++i) {
    int rows, cols, type;
    ifs.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    ifs.read(reinterpret_cast<char *>(&type), sizeof(type));
    Mat mat(rows, cols, type);
    ifs.read(reinterpret_cast<char *>(mat.data), mat.elemSize() * mat.total());
    mats.push_back(mat);
  }

  ifs.close();
  return mats;
}

std::vector<std::string> FaceDetect::read_name_from_file(const std::string & filename)
{
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cout << "open face_name file failed" << std::endl;
    return {};
  }
  std::vector<std::string> lines;
  std::string line;

  while (std::getline(ifs, line)) {
    lines.push_back(line);
  }
  ifs.close();
  return lines;
}

void FaceDetect::visualize_single_base(Mat & input,
    Mat & faces,
    int idx,
    bool flag,
    std::string id,
    int thickness)
{
  int i = idx;
  // Draw bounding box
  rectangle(input,
      Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)),
          int(faces.at<float>(i, 3))),
      flag ? Scalar(0, 255, 0) : Scalar(0, 0, 255), thickness);
  if (!id.empty()) {
    std::string text = id;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    cv::Point textOrg(int(faces.at<float>(i, 0)) - 10, int(faces.at<float>(i, 1)));

    cv::putText(input, text, textOrg, fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
  }
}

void FaceDetect::visualize_result_id(Mat & input, Mat & faces, int idx, bool flag, std::string id)
{
  visualize_single_base(input, faces, idx, flag, id, 2);
}

void FaceDetect::detect(cv::Mat & image,
    cv::Mat & faces,
    std::vector<std::string> & names,
    bool overlay_flag)
{
  face_detector_->setInputSize(image.size());
  face_detector_->detect(image, faces);

  if (faces.rows) {
    std::vector<Mat> face_features;
    for (int i = 0; i < faces.rows; i++) {
      Mat aligned_face, feature;
      face_recognizer_->alignCrop(image, faces.row(i), aligned_face);
      face_recognizer_->feature(aligned_face, feature);
      feature = feature.clone();
      face_features.push_back(feature);
    }
    for (int i = 0; i < int(face_features.size()); i++) {
      float max_cos = 0;
      int target_id = 0;
      for (int j = 0; j < int(faces_dst_features_.size()); j++) {
        double cos_score = face_recognizer_->match(
            face_features[i], faces_dst_features_[j], FaceRecognizerSF::DisType::FR_COSINE);

        if (cos_score > max_cos) {
          max_cos = cos_score;
          target_id = j;
        }
      }

      faces.at<float>(i, 14) = max_cos;
      if (max_cos >= cosine_similar_thresh_) {
        names.push_back(faces_dst_names_[target_id]);
      } else {
        names.push_back("");
      }
    }
    if (overlay_flag)
      overlay(image, faces, names);
  }
}

void FaceDetect::overlay(cv::Mat & image, cv::Mat & faces, std::vector<std::string> & names)
{
  for (int i = 0; i < faces.rows; i++) {
    visualize_result_id(image, faces, i, true, names[i]);
  }
}

}  // namespace sample_face_recognition
