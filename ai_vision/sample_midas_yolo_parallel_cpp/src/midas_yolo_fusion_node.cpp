// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "sample_midas_yolo_parallel_cpp/midas_yolo_fusion_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace sample_midas_yolo_parallel_cpp
{

// ── COCO class names ─────────────────────────────────────────────────────────

static const char * COCO_CLASSES[] = {
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "remote","keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
  "toothbrush"
};
static constexpr int NUM_COCO_CLASSES = 80;

static const cv::Scalar SEG_COLORS[] = {
  {56,56,255},{151,157,255},{31,112,255},{29,178,255},{49,210,207},
  {10,249,72},{23,204,146},{134,219,61},{52,147,26},{187,212,0},
  {168,153,44},{255,194,0},{147,69,52},{255,115,100},{236,24,0},
  {255,56,132},{133,0,82},{255,56,203},{200,149,255},{199,55,255}
};
static constexpr int NUM_SEG_COLORS = 20;

// ── constructor ──────────────────────────────────────────────────────────────

MidasYoloFusionNode::MidasYoloFusionNode(const rclcpp::NodeOptions & options)
: Node("midas_yolo_fusion_node", options),
  last_log_time_(this->now())
{
  // Declare and read parameters
  input_topic_      = declare_parameter("input_topic",            "/image_raw");
  midas_input_name_ = declare_parameter("midas_input_tensor_name","image");
  yolo_input_name_  = declare_parameter("yolo_input_tensor_name", "image");
  midas_data_type_  = declare_parameter("midas_tensor_data_type", 0);
  yolo_data_type_   = declare_parameter("yolo_tensor_data_type",  0);
  yolo_pack_uint16_ = declare_parameter("yolo_pack_uint16_input", false);
  score_thresh_     = static_cast<float>(declare_parameter("score_thresh",  0.25));
  iou_thresh_       = static_cast<float>(declare_parameter("iou_thresh",    0.45));
  overlay_alpha_    = static_cast<float>(declare_parameter("overlay_alpha", 0.45));
  max_pending_      = declare_parameter("max_pending_frames", 4);

  auto midas_size = declare_parameter("midas_input_size", std::vector<int64_t>{256, 256});
  auto yolo_size  = declare_parameter("yolo_input_size",  std::vector<int64_t>{640, 640});
  midas_h_ = static_cast<int>(midas_size[0]);
  midas_w_ = static_cast<int>(midas_size[1]);
  yolo_h_  = static_cast<int>(yolo_size[0]);
  yolo_w_  = static_cast<int>(yolo_size[1]);

  // Callback group: Reentrant so midas and yolo output callbacks can fire concurrently
  auto output_cbg = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions out_opts;
  out_opts.callback_group = output_cbg;

  // Publishers — enable intra-process for tensor topics
  rclcpp::PublisherOptions ipc_opts;
  ipc_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  midas_in_pub_   = create_publisher<custom_msg::TensorList>(
      "midas_inference_input_tensor",     10, ipc_opts);
  yolo_in_pub_    = create_publisher<custom_msg::TensorList>(
      "yolo_seg_inference_input_tensor",  10, ipc_opts);
  overlay_pub_    = create_publisher<sensor_msgs::msg::Image>("midas_yolo_overlay", 10);
  depth_color_pub_= create_publisher<sensor_msgs::msg::Image>("midas_depth_map",    10);
  depth_gray_pub_ = create_publisher<sensor_msgs::msg::Image>("midas_depth_gray",   10);

  // Subscriptions
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic_, 10,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { image_callback(msg); });

  midas_out_sub_ = create_subscription<custom_msg::TensorList>(
      "midas_inference_output_tensor", 10,
      [this](custom_msg::TensorList::ConstSharedPtr msg) { midas_output_callback(msg); },
      out_opts);

  yolo_out_sub_ = create_subscription<custom_msg::TensorList>(
      "yolo_seg_inference_output_tensor", 10,
      [this](custom_msg::TensorList::ConstSharedPtr msg) { yolo_output_callback(msg); },
      out_opts);

  RCLCPP_INFO(get_logger(),
      "MidasYoloFusionNode started (yolo_pack_uint16=%s, midas=%dx%d, yolo=%dx%d)",
      yolo_pack_uint16_ ? "true" : "false", midas_h_, midas_w_, yolo_h_, yolo_w_);
}

// ── key helpers ──────────────────────────────────────────────────────────────

FrameKey MidasYoloFusionNode::extract_key(const std_msgs::msg::Header & hdr)
{
  if (hdr.stamp.sec == 0 && hdr.stamp.nanosec == 0) {
    return {-1, static_cast<uint32_t>(++synthetic_seq_)};
  }
  return {hdr.stamp.sec, hdr.stamp.nanosec};
}

FrameKey MidasYoloFusionNode::match_pending_key(const std_msgs::msg::Header & hdr)
{
  auto key = extract_key(hdr);
  std::lock_guard<std::mutex> lk(pending_mutex_);
  if (pending_.count(key)) return key;
  if (pending_.size() == 1) return pending_.begin()->first;
  if (pending_.size() > 1) {
    // Return oldest
    return std::min_element(pending_.begin(), pending_.end(),
        [](const auto & a, const auto & b) {
          return a.second.created_at < b.second.created_at;
        })->first;
  }
  return {-2, 0};  // sentinel: not found
}

// ── image decode ─────────────────────────────────────────────────────────────

cv::Mat MidasYoloFusionNode::decode_image(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  const auto & enc = msg->encoding;
  if (enc == "bgr8") {
    return cv::Mat(msg->height, msg->width, CV_8UC3,
        const_cast<uint8_t *>(msg->data.data())).clone();
  }
  if (enc == "rgb8") {
    cv::Mat rgb(msg->height, msg->width, CV_8UC3,
        const_cast<uint8_t *>(msg->data.data()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  if (enc == "nv12") {
    cv::Mat yuv(msg->height * 3 / 2, msg->width, CV_8UC1,
        const_cast<uint8_t *>(msg->data.data()));
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
  }
  RCLCPP_ERROR(get_logger(), "Unsupported encoding: %s", enc.c_str());
  return {};
}

// ── preprocessing ────────────────────────────────────────────────────────────

cv::Mat MidasYoloFusionNode::prep_midas(const cv::Mat & bgr)
{
  cv::Mat resized, rgb;
  cv::resize(bgr, resized, {midas_w_, midas_h_}, 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  if (midas_data_type_ == 2) {
    // float32 normalised with ImageNet mean/std
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    const cv::Scalar mean(0.485f, 0.456f, 0.406f);
    const cv::Scalar std (0.229f, 0.224f, 0.225f);
    cv::subtract(f32, mean, f32);
    cv::divide(f32, std, f32);
    return f32;  // HxWx3 float32
  }
  return rgb;  // HxWx3 uint8
}

cv::Mat MidasYoloFusionNode::prep_yolo(const cv::Mat & bgr)
{
  cv::Mat resized, rgb;
  cv::resize(bgr, resized, {yolo_w_, yolo_h_}, 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  if (yolo_pack_uint16_) {
    // Quantise to uint16: q = round(pixel/255 / 1.5259e-5)
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    f32 *= (1.0f / 0.000015259021893143654f);
    cv::Mat u16;
    f32.convertTo(u16, CV_16UC3);
    return u16;
  }
  if (yolo_data_type_ == 2) {
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    return f32;
  }
  return rgb;  // uint8
}

// ── tensor message builder ───────────────────────────────────────────────────

custom_msg::TensorList MidasYoloFusionNode::make_tensor_msg(
    const std::string & name, const cv::Mat & data, int data_type,
    const std_msgs::msg::Header & hdr)
{
  custom_msg::TensorList msg;
  msg.header = hdr;

  custom_msg::Tensor t;
  t.name      = name;
  t.data_type = data_type;

  // Shape: [1, H, W, C]
  t.shape = {1,
      static_cast<uint32_t>(data.rows),
      static_cast<uint32_t>(data.cols),
      static_cast<uint32_t>(data.channels())};

  const size_t nbytes = data.total() * data.elemSize();
  t.data.resize(nbytes);
  std::memcpy(t.data.data(), data.data, nbytes);

  msg.tensor_list.push_back(std::move(t));
  return msg;
}

// ── image callback ───────────────────────────────────────────────────────────

void MidasYoloFusionNode::image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv::Mat bgr = decode_image(msg);
  if (bgr.empty()) return;

  auto key = extract_key(msg->header);

  cv::Mat midas_in = prep_midas(bgr);
  cv::Mat yolo_in  = prep_yolo(bgr);

  auto midas_msg = std::make_unique<custom_msg::TensorList>(
      make_tensor_msg(midas_input_name_, midas_in, midas_data_type_, msg->header));
  auto yolo_msg = std::make_unique<custom_msg::TensorList>(
      make_tensor_msg(yolo_input_name_,  yolo_in,  yolo_data_type_,  msg->header));

  {
    std::lock_guard<std::mutex> lk(pending_mutex_);
    if (static_cast<int>(pending_.size()) >= max_pending_) {
      auto oldest = std::min_element(pending_.begin(), pending_.end(),
          [](const auto & a, const auto & b) {
            return a.second.created_at < b.second.created_at;
          });
      pending_.erase(oldest);
    }
    PendingFrame frame;
    frame.header_sec  = msg->header.stamp.sec;
    frame.header_nsec = msg->header.stamp.nanosec;
    frame.image_bgr   = bgr;
    frame.yolo_input_w = yolo_w_;
    frame.yolo_input_h = yolo_h_;
    frame.created_at  = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    pending_[key] = std::move(frame);
  }

  midas_in_pub_->publish(std::move(midas_msg));
  yolo_in_pub_->publish(std::move(yolo_msg));
}

// ── output callbacks ─────────────────────────────────────────────────────────

void MidasYoloFusionNode::midas_output_callback(custom_msg::TensorList::ConstSharedPtr msg)
{
  auto key = match_pending_key(msg->header);
  if (key.first == -2) return;

  bool ready = false;
  {
    std::lock_guard<std::mutex> lk(pending_mutex_);
    auto it = pending_.find(key);
    if (it == pending_.end()) return;
    it->second.midas_tensors = msg;
    ready = (it->second.yolo_tensors != nullptr);
  }
  if (ready) fuse_and_publish(key);
}

void MidasYoloFusionNode::yolo_output_callback(custom_msg::TensorList::ConstSharedPtr msg)
{
  auto key = match_pending_key(msg->header);
  if (key.first == -2) return;

  bool ready = false;
  {
    std::lock_guard<std::mutex> lk(pending_mutex_);
    auto it = pending_.find(key);
    if (it == pending_.end()) return;
    it->second.yolo_tensors = msg;
    ready = (it->second.midas_tensors != nullptr);
  }
  if (ready) fuse_and_publish(key);
}

// ── tensor → cv::Mat ─────────────────────────────────────────────────────────

cv::Mat MidasYoloFusionNode::tensor_to_mat(const custom_msg::Tensor & tensor)
{
  const auto & shape = tensor.shape;
  if (shape.empty()) return {};

  // Flatten shape to total elements
  size_t elem_count = 1;
  for (auto d : shape) elem_count *= d;

  const uint8_t * src = nullptr;
  size_t byte_count = 0;

  // DMA-BUF path
#ifdef QRB_TENSOR_HAS_DMABUF
  if (tensor.dmabuf_fd >= 0 && tensor.dmabuf_size > 0) {
    // mmap the ION buffer
    void * ptr = ::mmap(nullptr, tensor.dmabuf_size, PROT_READ, MAP_SHARED,
        tensor.dmabuf_fd, static_cast<off_t>(tensor.dmabuf_offset));
    if (ptr != MAP_FAILED) {
      src        = static_cast<const uint8_t *>(ptr);
      byte_count = tensor.dmabuf_size;
      // We'll copy below then unmap
    }
  }
#endif

  bool from_dmabuf = (src != nullptr);
  if (!from_dmabuf) {
    src        = tensor.data.data();
    byte_count = tensor.data.size();
  }

  if (byte_count == 0 || src == nullptr) return {};

  // Determine dtype
  int cv_type = CV_32F;
  if      (byte_count == elem_count)     cv_type = CV_8U;
  else if (byte_count == elem_count * 2) cv_type = CV_16U;
  else if (byte_count == elem_count * 4) cv_type = CV_32F;
  else if (byte_count == elem_count * 8) cv_type = CV_64F;

  // Build a flat 1D mat then reshape
  cv::Mat flat(1, static_cast<int>(elem_count), cv_type);
  std::memcpy(flat.data, src, byte_count);

#ifdef QRB_TENSOR_HAS_DMABUF
  if (from_dmabuf) {
    ::munmap(const_cast<void *>(static_cast<const void *>(src)), byte_count);
  }
#endif

  // Reshape to [N, ...] — we keep it flat; callers squeeze as needed
  return flat;
}

// ── MiDaS depth decode ───────────────────────────────────────────────────────

void MidasYoloFusionNode::decode_midas_depth(
    const custom_msg::Tensor & tensor, int out_w, int out_h,
    cv::Mat & depth_f32, cv::Mat & depth_gray, cv::Mat & depth_color)
{
  cv::Mat raw = tensor_to_mat(tensor);
  if (raw.empty()) {
    depth_f32   = cv::Mat::zeros(out_h, out_w, CV_32F);
    depth_gray  = cv::Mat::zeros(out_h, out_w, CV_8U);
    depth_color = cv::Mat::zeros(out_h, out_w, CV_8UC3);
    return;
  }

  // Squeeze to 2D: shape is typically [1,1,H,W] or [1,H,W] or [H,W]
  const auto & shape = tensor.shape;
  int h = 1, w = 1;
  (void)w;
  // Find the last two non-1 dims
  std::vector<int> non1;
  for (auto d : shape) if (d > 1) non1.push_back(static_cast<int>(d));
  if (non1.size() >= 2) {
    h = non1[non1.size() - 2];
    w = non1[non1.size() - 1];
  } else if (non1.size() == 1) {
    h = 1; w = non1[0];
  }

  cv::Mat depth2d = raw.reshape(1, h);
  if (depth2d.type() != CV_32F) depth2d.convertTo(depth2d, CV_32F);

  cv::resize(depth2d, depth_f32, {out_w, out_h}, 0, 0, cv::INTER_LINEAR);

  cv::normalize(depth_f32, depth_gray, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::applyColorMap(depth_gray, depth_color, cv::COLORMAP_INFERNO);
}

// ── proto tensor → HxWx32 ────────────────────────────────────────────────────

cv::Mat MidasYoloFusionNode::proto_to_hwc(const cv::Mat & proto_flat)
{
  // proto_flat is a 1D mat of total_elements floats
  // Expected shapes: [1,32,H,W] or [1,H,W,32]
  // We need to return HxWx32 float32
  // Since we don't have the original shape here, we infer from the tensor shape
  // stored in the caller — this function receives the already-shaped mat.
  // For now just return as-is; caller handles reshaping.
  return proto_flat;
}

// ── NMS ──────────────────────────────────────────────────────────────────────

std::vector<Detection> MidasYoloFusionNode::nms(std::vector<Detection> dets)
{
  std::sort(dets.begin(), dets.end(),
      [](const Detection & a, const Detection & b) { return a.score > b.score; });

  std::vector<Detection> kept;
  for (auto & d : dets) {
    bool suppress = false;
    for (const auto & k : kept) {
      if (d.cls != k.cls) continue;
      float ix1 = std::max(d.x1, k.x1), iy1 = std::max(d.y1, k.y1);
      float ix2 = std::min(d.x2, k.x2), iy2 = std::min(d.y2, k.y2);
      if (ix2 <= ix1 || iy2 <= iy1) continue;
      float inter = (ix2 - ix1) * (iy2 - iy1);
      float area_d = (d.x2 - d.x1) * (d.y2 - d.y1);
      float area_k = (k.x2 - k.x1) * (k.y2 - k.y1);
      float iou = inter / std::max(area_d + area_k - inter, 1e-6f);
      if (iou > iou_thresh_) { suppress = true; break; }
    }
    if (!suppress) kept.push_back(std::move(d));
  }
  return kept;
}

// ── YOLO split-output decode ─────────────────────────────────────────────────

std::vector<Detection> MidasYoloFusionNode::decode_yolo_split(
    const cv::Mat & boxes_flat, const cv::Mat & scores_flat,
    const cv::Mat & class_idx_flat, const cv::Mat & coeffs_flat,
    int input_w, int input_h)
{
  // boxes_flat:     N*4 floats  → Nx4
  // scores_flat:    N floats
  // class_idx_flat: N floats
  // coeffs_flat:    N*32 floats → Nx32

  int n = static_cast<int>(scores_flat.total());
  if (n == 0) return {};

  // Normalise scores to [0,1] if needed
  cv::Mat scores_f32;
  if (scores_flat.type() != CV_32F) scores_flat.convertTo(scores_f32, CV_32F);
  else scores_f32 = scores_flat;

  float s_max = *std::max_element(scores_f32.begin<float>(), scores_f32.end<float>());
  if (s_max > 1.0f) {
    // Likely raw logits or uint8 — normalise
    scores_f32 /= (s_max > 255.0f ? s_max : 255.0f);
  }

  cv::Mat boxes_f32;
  if (boxes_flat.type() != CV_32F) boxes_flat.convertTo(boxes_f32, CV_32F);
  else boxes_f32 = boxes_flat;

  cv::Mat cidx_f32;
  if (class_idx_flat.type() != CV_32F) class_idx_flat.convertTo(cidx_f32, CV_32F);
  else cidx_f32 = class_idx_flat;

  cv::Mat coeffs_f32;
  if (coeffs_flat.type() != CV_32F) coeffs_flat.convertTo(coeffs_f32, CV_32F);
  else coeffs_f32 = coeffs_flat;

  // Reshape
  cv::Mat b = boxes_f32.reshape(1, n);   // Nx4
  cv::Mat m = coeffs_f32.reshape(1, n);  // Nx32

  // Transpose if needed
  if (b.cols != 4 && b.rows == 4) cv::transpose(b, b);
  if (m.cols != 32 && m.rows == 32) cv::transpose(m, m);
  if (b.cols != 4 || m.cols != 32) return {};

  std::vector<Detection> dets;
  for (int i = 0; i < n; ++i) {
    float score = scores_f32.at<float>(i);
    if (score < score_thresh_) continue;

    int cls = static_cast<int>(std::round(cidx_f32.at<float>(i)));
    float x1 = b.at<float>(i, 0), y1 = b.at<float>(i, 1);
    float x2 = b.at<float>(i, 2), y2 = b.at<float>(i, 3);

    // Scale from normalised [0,1] if needed
    float max_coord = std::max({std::abs(x1), std::abs(y1), std::abs(x2), std::abs(y2)});
    if (max_coord <= 2.0f) {
      x1 *= input_w; x2 *= input_w;
      y1 *= input_h; y2 *= input_h;
    }
    x1 = std::max(0.0f, std::min(static_cast<float>(input_w), x1));
    x2 = std::max(0.0f, std::min(static_cast<float>(input_w), x2));
    y1 = std::max(0.0f, std::min(static_cast<float>(input_h), y1));
    y2 = std::max(0.0f, std::min(static_cast<float>(input_h), y2));
    if (x2 <= x1 || y2 <= y1) continue;

    Detection d;
    d.cls = cls; d.score = score;
    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    d.coeff.resize(32);
    for (int k = 0; k < 32; ++k) d.coeff[k] = m.at<float>(i, k);
    dets.push_back(std::move(d));
  }
  return nms(std::move(dets));
}

// ── parse YOLO outputs ───────────────────────────────────────────────────────

void MidasYoloFusionNode::parse_yolo_outputs(
    const custom_msg::TensorList & tensors, int input_w, int input_h,
    std::vector<Detection> & dets_out, cv::Mat & proto_out)
{
  dets_out.clear();
  proto_out = cv::Mat();

  // Collect all tensors as flat mats with their shapes
  struct TensorInfo { cv::Mat flat; std::vector<uint32_t> shape; };
  std::vector<TensorInfo> infos;
  for (const auto & t : tensors.tensor_list) {
    infos.push_back({tensor_to_mat(t), t.shape});
  }

  // Identify proto (4D with last or second dim == 32)
  // and detection outputs (boxes, scores, class_idx, coeffs)
  cv::Mat proto_flat;
  std::vector<uint32_t> proto_shape;
  cv::Mat boxes_flat, scores_flat, cidx_flat, coeffs_flat;

  for (auto & info : infos) {
    const auto & sh = info.shape;
    size_t total = 1;
    for (auto d : sh) total *= d;

    // Proto: 4D tensor with a dim == 32 (mask prototypes)
    if (sh.size() == 4) {
      if (sh[1] == 32 || sh[3] == 32) {
        proto_flat  = info.flat;
        proto_shape = sh;
        continue;
      }
    }
    // 3D: could be boxes [1,4,N] or coeffs [1,32,N]
    if (sh.size() == 3) {
      uint32_t d1 = sh[1], d2 = sh[2];
      if ((d1 == 4 || d2 == 4) && boxes_flat.empty()) {
        boxes_flat = info.flat; continue;
      }
      if ((d1 == 32 || d2 == 32) && coeffs_flat.empty()) {
        coeffs_flat = info.flat; continue;
      }
    }
    // 2D: scores [1,N] or class_idx [1,N]
    if (sh.size() == 2) {
      uint32_t d1 = sh[0], d2 = sh[1];
      uint32_t n = std::max(d1, d2);
      if (n >= 1000 && scores_flat.empty()) {
        scores_flat = info.flat; continue;
      }
      if (!scores_flat.empty() && cidx_flat.empty()) {
        uint32_t sn = static_cast<uint32_t>(scores_flat.total());
        if (n == sn) { cidx_flat = info.flat; continue; }
      }
    }
  }

  if (proto_flat.empty()) return;

  // Reshape proto to HxWx32
  // proto_shape is [1, 32, H, W] or [1, H, W, 32]
  int ph = 1, pw = 1;
  if (proto_shape.size() == 4) {
    if (proto_shape[1] == 32) { ph = proto_shape[2]; pw = proto_shape[3]; }
    else                       { ph = proto_shape[1]; pw = proto_shape[2]; }
  }
  // Reshape flat to [32, ph*pw] then transpose to [ph*pw, 32] then reshape to [ph, pw, 32]
  if (proto_shape.size() == 4 && proto_shape[1] == 32) {
    // CHW → HWC
    cv::Mat chw = proto_flat.reshape(1, 32);  // 32 x (ph*pw)
    cv::Mat hwc;
    cv::transpose(chw, hwc);                  // (ph*pw) x 32
    proto_out = hwc.reshape(1, ph);           // ph x (pw*32) — we'll treat as ph rows
    // Actually reshape to ph*pw rows of 32 cols, then interpret as ph x pw x 32
    // For the mask decode we need proto[y,x,:] = 32 coeffs
    // Store as (ph*pw) x 32 and use ph, pw for indexing
    proto_out = hwc;  // (ph*pw) x 32
    // Tag the mat with ph, pw via extra member — use a 3D mat
    int sizes[3] = {ph, pw, 32};
    proto_out = hwc.reshape(1, 3, sizes);  // ph x pw x 32 (3D)
  } else {
    // Already HWC
    proto_out = proto_flat.reshape(1, ph);
  }

  if (!boxes_flat.empty() && !scores_flat.empty() &&
      !cidx_flat.empty()  && !coeffs_flat.empty()) {
    dets_out = decode_yolo_split(
        boxes_flat, scores_flat, cidx_flat, coeffs_flat, input_w, input_h);
  }
}

// ── depth p85 ────────────────────────────────────────────────────────────────

float MidasYoloFusionNode::depth_p85(const cv::Mat & depth, const cv::Mat & mask)
{
  std::vector<float> vals;
  vals.reserve(depth.total() / 4);
  for (int r = 0; r < depth.rows; ++r) {
    for (int c = 0; c < depth.cols; ++c) {
      if (mask.at<uint8_t>(r, c) > 0) vals.push_back(depth.at<float>(r, c));
    }
  }
  if (vals.empty()) return 0.0f;
  size_t idx = static_cast<size_t>(vals.size() * 0.85f);
  std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
  return vals[idx];
}

// ── draw overlay ─────────────────────────────────────────────────────────────

void MidasYoloFusionNode::draw_overlay(
    cv::Mat & out, const cv::Mat & depth_f32,
    const std::vector<Detection> & dets, const cv::Mat & proto_hwc,
    int input_w, int input_h)
{
  if (proto_hwc.empty() || dets.empty()) return;

  // proto_hwc is 3D: ph x pw x 32
  int ph = proto_hwc.size[0];
  int pw = proto_hwc.size[1];

  float sx = static_cast<float>(out.cols) / input_w;
  float sy = static_cast<float>(out.rows) / input_h;
  float psx = static_cast<float>(pw) / input_w;
  float psy = static_cast<float>(ph) / input_h;

  double dmin, dmax;
  cv::minMaxLoc(depth_f32, &dmin, &dmax);
  float dr = std::max(static_cast<float>(dmax - dmin), 1e-6f);

  for (const auto & det : dets) {
    // Compute mask logits: proto_hwc @ coeff → ph x pw
    // proto_hwc[y,x,:] dot coeff
    cv::Mat mask_logits(ph, pw, CV_32F);
    for (int y = 0; y < ph; ++y) {
      for (int x = 0; x < pw; ++x) {
        float val = 0.0f;
        const float * row = proto_hwc.ptr<float>(y, x);
        for (int k = 0; k < 32; ++k) val += row[k] * det.coeff[k];
        mask_logits.at<float>(y, x) = val;
      }
    }
    cv::Mat mask;
    cv::threshold(mask_logits, mask, 0.0, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    // Crop mask to detection box in proto space
    int mx1 = std::max(0, static_cast<int>(det.x1 * psx));
    int my1 = std::max(0, static_cast<int>(det.y1 * psy));
    int mx2 = std::min(pw, static_cast<int>(det.x2 * psx + 0.5f));
    int my2 = std::min(ph, static_cast<int>(det.y2 * psy + 0.5f));
    if (mx2 <= mx1 || my2 <= my1) continue;

    // Detection box in output space
    int bx1 = std::max(0, static_cast<int>(det.x1 * sx));
    int by1 = std::max(0, static_cast<int>(det.y1 * sy));
    int bx2 = std::min(out.cols, static_cast<int>(det.x2 * sx + 0.5f));
    int by2 = std::min(out.rows, static_cast<int>(det.y2 * sy + 0.5f));
    if (bx2 <= bx1 || by2 <= by1) continue;

    cv::Mat crop = mask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1));
    cv::Mat mask_roi;
    cv::resize(crop, mask_roi, {bx2 - bx1, by2 - by1}, 0, 0, cv::INTER_NEAREST);

    // Blend colour overlay
    const cv::Scalar & color = SEG_COLORS[det.cls % NUM_SEG_COLORS];
    cv::Mat out_roi = out(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1));
    cv::Mat color_img(out_roi.size(), CV_8UC3, color);
    cv::Mat blended;
    cv::addWeighted(out_roi, 1.0 - overlay_alpha_, color_img, overlay_alpha_, 0.0, blended);
    blended.copyTo(out_roi, mask_roi);

    // Label
    const char * cls_name = (det.cls >= 0 && det.cls < NUM_COCO_CLASSES)
        ? COCO_CLASSES[det.cls] : "unknown";
    cv::Mat depth_roi = depth_f32(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1));
    float dval = depth_p85(depth_roi, mask_roi);
    float dnorm = (dval - static_cast<float>(dmin)) / dr;

    char label[128];
    std::snprintf(label, sizeof(label), "%s %.0f%% d=%.2f",
        cls_name, det.score * 100.0f, dnorm);

    cv::Scalar cv_color(color[0], color[1], color[2]);
    cv::rectangle(out, {bx1, by1}, {bx2, by2}, cv_color, 1);
    int baseline = 0;
    cv::Size txt_sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int txt_x = bx1;
    int txt_y = std::min(out.rows - 1, by1 + txt_sz.height + 2);
    cv::rectangle(out, {txt_x, by1},
        {std::min(out.cols - 1, txt_x + txt_sz.width + 2), txt_y + baseline},
        cv_color, cv::FILLED);
    cv::putText(out, label, {txt_x + 1, txt_y - 1},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
  }
}

// ── image msg builder ────────────────────────────────────────────────────────

sensor_msgs::msg::Image MidasYoloFusionNode::mat_to_image_msg(
    const cv::Mat & mat, const std::string & encoding,
    const std_msgs::msg::Header & hdr)
{
  sensor_msgs::msg::Image msg;
  msg.header   = hdr;
  msg.height   = static_cast<uint32_t>(mat.rows);
  msg.width    = static_cast<uint32_t>(mat.cols);
  msg.encoding = encoding;
  msg.step     = static_cast<uint32_t>(mat.step);
  msg.is_bigendian = false;
  const size_t nbytes = mat.total() * mat.elemSize();
  msg.data.resize(nbytes);
  std::memcpy(msg.data.data(), mat.data, nbytes);
  return msg;
}

// ── fuse and publish ─────────────────────────────────────────────────────────

void MidasYoloFusionNode::fuse_and_publish(FrameKey key)
{
  PendingFrame frame;
  {
    std::lock_guard<std::mutex> lk(pending_mutex_);
    auto it = pending_.find(key);
    if (it == pending_.end()) return;
    frame = std::move(it->second);
    pending_.erase(it);
  }
  if (!frame.midas_tensors || !frame.yolo_tensors) return;
  if (frame.midas_tensors->tensor_list.empty() ||
      frame.yolo_tensors->tensor_list.empty()) return;

  try {
    // Decode MiDaS depth
    cv::Mat depth_f32, depth_gray, depth_color;
    decode_midas_depth(
        frame.midas_tensors->tensor_list[0],
        frame.image_bgr.cols, frame.image_bgr.rows,
        depth_f32, depth_gray, depth_color);

    // Parse YOLO outputs
    std::vector<Detection> dets;
    cv::Mat proto_hwc;
    parse_yolo_outputs(
        *frame.yolo_tensors, frame.yolo_input_w, frame.yolo_input_h,
        dets, proto_hwc);

    // Draw overlay on depth_color
    cv::Mat overlay = depth_color.clone();
    draw_overlay(overlay, depth_f32, dets, proto_hwc,
        frame.yolo_input_w, frame.yolo_input_h);

    // Build header
    std_msgs::msg::Header hdr;
    hdr.stamp.sec    = frame.header_sec;
    hdr.stamp.nanosec = frame.header_nsec;

    overlay_pub_->publish(mat_to_image_msg(overlay,     "bgr8",  hdr));
    depth_color_pub_->publish(mat_to_image_msg(depth_color, "bgr8",  hdr));
    depth_gray_pub_->publish(mat_to_image_msg(depth_gray,  "mono8", hdr));

    uint64_t cnt = ++processed_count_;
    auto now = this->now();
    if ((now - last_log_time_).seconds() > 2.0) {
      RCLCPP_INFO(get_logger(),
          "processed=%lu pending=%zu detections=%zu",
          cnt, pending_.size(), dets.size());
      last_log_time_ = now;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "fusion failed: %s", e.what());
  }
}

}  // namespace sample_midas_yolo_parallel_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(
    sample_midas_yolo_parallel_cpp::MidasYoloFusionNode)
