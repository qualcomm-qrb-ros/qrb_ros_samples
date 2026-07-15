// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "sample_midas_yolo_parallel_cpp/midas_yolo_fusion_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <numeric>
#include <stdexcept>

#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace sample_midas_yolo_parallel_cpp
{

using clk = std::chrono::steady_clock;
using ms  = std::chrono::duration<double, std::milli>;

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
  out_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Publishers — enable intra-process for tensor topics
  rclcpp::PublisherOptions ipc_opts;
  ipc_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  midas_in_pub_   = create_publisher<custom_msg::TensorList>(
      "midas_inference_input_tensor",     10, ipc_opts);
  yolo_in_pub_    = create_publisher<custom_msg::TensorList>(
      "yolo_seg_inference_input_tensor",  10, ipc_opts);

  // Camera-stream outputs: use image_transport (auto-advertises compressed/theora/
  // zstd variants alongside the raw topic) with SensorDataQoS (best-effort,
  // keep-last-5). Raw bgr8/mono8 frames at 1280x720+ are multi-MB each; RELIABLE
  // QoS on messages this large causes FastDDS fragment retransmission storms
  // that can stall the publisher itself under any slow/lossy subscriber
  // (including foxglove_bridge or a remote client over Wi-Fi). BEST_EFFORT lets
  // the publisher drop stale frames instead of blocking.
  const auto image_qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();
  overlay_pub_    = image_transport::create_publisher(this, "midas_yolo_overlay", image_qos);
  depth_color_pub_= image_transport::create_publisher(this, "midas_depth_map",    image_qos);
  depth_gray_pub_ = image_transport::create_publisher(this, "midas_depth_gray",   image_qos);

  // Subscriptions
  //
  // NOTE: image_sub_ intentionally stays on the default RELIABLE QoS (matching
  // usb_cam's hardcoded rclcpp::QoS{100} publisher). Switching this subscription
  // to SensorDataQoS/BEST_EFFORT was tried and measured to cause near-total loss
  // of /image_raw frames: at 1280x720 each raw bgr8/rgb8 frame is ~2.76MB, and a
  // BEST_EFFORT reader does not participate in the RELIABLE writer's ACKNACK/
  // fragment-recovery protocol, so on this device's small (208KB) kernel UDP
  // recv buffers almost every large fragmented frame was dropped, collapsing
  // throughput from ~30fps to well under 1fps. The retransmission-storm problem
  // that BEST_EFFORT solves for our own *publishers* below does not apply here
  // since we don't control usb_cam's publisher reliability anyway.
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
  // No exact match: the frame this output belongs to is no longer pending
  // (already fused, evicted under load, or the source never stamped
  // header.stamp so keys can't be correlated). Do NOT guess a pending frame
  // here — pairing an output with the wrong frame silently fuses depth and
  // detection results from different images.
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

  // The QNN MiDaS model input is UFIXED_POINT_8 (quantized uint8) in NCHW layout.
  // Split the interleaved HWC uint8 image into three separate H×W planes and
  // concatenate them to produce a contiguous NCHW buffer: [R-plane | G-plane | B-plane].
  std::vector<cv::Mat> planes(3);
  cv::split(rgb, planes);                    // planes[i] is H×W CV_8U
  cv::Mat nchw;
  cv::vconcat(planes, nchw);                 // (3H)×W CV_8U — contiguous NCHW
  return nchw.reshape(1, 1);                 // 1×(3*H*W) flat, same data
}

cv::Mat MidasYoloFusionNode::prep_yolo(const cv::Mat & bgr)
{
  cv::Mat resized, rgb;
  cv::resize(bgr, resized, {yolo_w_, yolo_h_}, 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  // The QNN YOLO model input has declared shape [1, 3, H, W] (NCHW), same as
  // MiDaS.  cv::Mat channel-conversion above produces interleaved HWC data
  // (R,G,B,R,G,B,...); it must be split into three separate H×W planes and
  // concatenated into a contiguous NCHW buffer before being sent to the
  // inference node, exactly like prep_midas() does. Without this conversion
  // the model receives spatially-scrambled channel data: box/mask outputs
  // still look superficially plausible (they are more tolerant of bad
  // input), but the class-score branch (Sigmoid) collapses to all-zero.
  if (yolo_pack_uint16_) {
    // Quantise to uint16: q = round(pixel/255 / 1.5259e-5)
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    f32 *= (1.0f / 0.000015259021893143654f);
    cv::Mat u16;
    f32.convertTo(u16, CV_16UC3);
    std::vector<cv::Mat> planes(3);
    cv::split(u16, planes);
    cv::Mat nchw;
    cv::vconcat(planes, nchw);
    return nchw.reshape(1, 1);
  }
  if (yolo_data_type_ == 2) {
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> planes(3);
    cv::split(f32, planes);
    cv::Mat nchw;
    cv::vconcat(planes, nchw);
    return nchw.reshape(1, 1);
  }
  // uint8
  std::vector<cv::Mat> planes(3);
  cv::split(rgb, planes);
  cv::Mat nchw;
  cv::vconcat(planes, nchw);
  return nchw.reshape(1, 1);
}

// ── tensor message builder ───────────────────────────────────────────────────

custom_msg::TensorList MidasYoloFusionNode::make_tensor_msg(
    const std::string & name, const cv::Mat & data, int data_type,
    const std_msgs::msg::Header & hdr,
    const std::vector<uint32_t> & explicit_shape)
{
  custom_msg::TensorList msg;
  msg.header = hdr;

  custom_msg::Tensor t;
  t.name      = name;
  t.data_type = data_type;

  if (!explicit_shape.empty()) {
    t.shape = explicit_shape;
  } else {
    // Default: infer [1, H, W, C] from mat dimensions
    t.shape = {1,
        static_cast<uint32_t>(data.rows),
        static_cast<uint32_t>(data.cols),
        static_cast<uint32_t>(data.channels())};
  }

  const size_t nbytes = data.total() * data.elemSize();
  t.data.resize(nbytes);
  std::memcpy(t.data.data(), data.data, nbytes);

  msg.tensor_list.push_back(std::move(t));
  return msg;
}

// ── image callback ───────────────────────────────────────────────────────────

void MidasYoloFusionNode::image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  auto t_start = clk::now();
  cv::Mat bgr = decode_image(msg);
  if (bgr.empty()) return;
  auto t_decode = clk::now();

  auto key = extract_key(msg->header);

  // Prep both inputs sequentially — each takes ~2ms, total ~4ms.
  // std::async was tried but caused thread oversubscription with the executor pool.
  cv::Mat midas_in = prep_midas(bgr);
  cv::Mat yolo_in  = prep_yolo(bgr);
  auto t_prep = clk::now();

  // MiDaS input is NCHW uint8: shape [1, 3, H, W]
  auto midas_msg = std::make_unique<custom_msg::TensorList>(
      make_tensor_msg(midas_input_name_, midas_in, midas_data_type_, msg->header,
          {1, 3, static_cast<uint32_t>(midas_h_), static_cast<uint32_t>(midas_w_)}));
  // YOLO input is NCHW: shape [1, 3, H, W] (same layout convention as MiDaS).
  auto yolo_msg = std::make_unique<custom_msg::TensorList>(
      make_tensor_msg(yolo_input_name_,  yolo_in,  yolo_data_type_,  msg->header,
          {1, 3, static_cast<uint32_t>(yolo_h_), static_cast<uint32_t>(yolo_w_)}));
  auto t_make = clk::now();

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
  auto t_pub = clk::now();

  // Log image callback timing every 2s
  static std::atomic<uint64_t> img_count{0};
  static clk::time_point last_img_log = clk::now();
  if (++img_count % 20 == 0) {
    auto now = clk::now();
    if (ms(now - last_img_log).count() > 2000) {
      RCLCPP_INFO(get_logger(),
          "image_cb: decode=%.2fms prep=%.2fms make=%.2fms pub=%.2fms total=%.2fms pending=%zu",
          ms(t_decode - t_start).count(),
          ms(t_prep   - t_decode).count(),
          ms(t_make   - t_prep).count(),
          ms(t_pub    - t_make).count(),
          ms(t_pub    - t_start).count(),
          pending_.size());
      last_img_log = now;
    }
  }
}

// ── output callbacks ─────────────────────────────────────────────────────────

void MidasYoloFusionNode::midas_output_callback(custom_msg::TensorList::ConstSharedPtr msg)
{
  auto key = match_pending_key(msg->header);
  if (key.first == -2) {
    RCLCPP_WARN(get_logger(),
        "midas_output_callback: no pending frame matches header stamp %d.%09u — "
        "dropping output (frame likely evicted or unstamped source)",
        msg->header.stamp.sec, msg->header.stamp.nanosec);
    return;
  }

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
  if (key.first == -2) {
    RCLCPP_WARN(get_logger(),
        "yolo_output_callback: no pending frame matches header stamp %d.%09u — "
        "dropping output (frame likely evicted or unstamped source)",
        msg->header.stamp.sec, msg->header.stamp.nanosec);
    return;
  }

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
  if (depth2d.cols != w) {
    RCLCPP_WARN(get_logger(), "decode_midas_depth: shape mismatch cols=%d expected w=%d",
        depth2d.cols, w);
  }
  if (depth2d.type() != CV_32F) depth2d.convertTo(depth2d, CV_32F);

  cv::resize(depth2d, depth_f32, {out_w, out_h}, 0, 0, cv::INTER_LINEAR);

  cv::normalize(depth_f32, depth_gray, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::applyColorMap(depth_gray, depth_color, cv::COLORMAP_INFERNO);
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

// ── YOLO split-output decode (INT8-fixed binary: full 80-class scores) ───────
//
// The INT8 split binary exposes four separate output tensors:
//   boxes  (1, 4,  N) — cx, cy, w, h in pixel space
//   scores (1, 80, N) — sigmoid class scores in [0,1] for all 80 COCO classes
//   coeffs (1, 32, N) — mask prototype coefficients
//   proto  (1, 32, H, W) — mask prototypes (handled in parse_yolo_outputs)
//
// This function performs per-anchor argmax over the 80 class scores to obtain
// the best class and its confidence, then applies score threshold and NMS.

std::vector<Detection> MidasYoloFusionNode::decode_yolo_split_all_scores(
    const cv::Mat & boxes_flat, const cv::Mat & all_scores_flat,
    const cv::Mat & coeffs_flat,
    int input_w, int input_h)
{
  // all_scores_flat: 80*N floats (from tensor shape [1,80,N])
  // boxes_flat:       4*N floats (from tensor shape [1, 4,N])
  // coeffs_flat:     32*N floats (from tensor shape [1,32,N])

  int total_score_elems = static_cast<int>(all_scores_flat.total());
  if (total_score_elems == 0) return {};

  // Number of anchors
  int n = total_score_elems / NUM_COCO_CLASSES;
  if (n == 0 || total_score_elems % NUM_COCO_CLASSES != 0) {
    RCLCPP_WARN(get_logger(),
        "decode_yolo_split_all_scores: unexpected scores size %d (expected multiple of %d)",
        total_score_elems, NUM_COCO_CLASSES);
    return {};
  }

  cv::Mat scores_f32;
  if (all_scores_flat.type() != CV_32F)
    all_scores_flat.convertTo(scores_f32, CV_32F);
  else
    scores_f32 = all_scores_flat;

  cv::Mat boxes_f32;
  if (boxes_flat.type() != CV_32F) boxes_flat.convertTo(boxes_f32, CV_32F);
  else boxes_f32 = boxes_flat;

  cv::Mat coeffs_f32;
  if (coeffs_flat.type() != CV_32F) coeffs_flat.convertTo(coeffs_f32, CV_32F);
  else coeffs_f32 = coeffs_flat;

  // Reshape to [NUM_COCO_CLASSES, N] then transpose to [N, NUM_COCO_CLASSES]
  cv::Mat s = scores_f32.reshape(1, NUM_COCO_CLASSES);  // 80 x N
  cv::Mat st;
  cv::transpose(s, st);                                  // N x 80

  // Reshape boxes to [4, N] then transpose to [N, 4]
  cv::Mat b = boxes_f32.reshape(1, 4);  // 4 x N
  cv::Mat bt;
  cv::transpose(b, bt);                 // N x 4

  // Reshape coeffs to [32, N] then transpose to [N, 32]
  cv::Mat m = coeffs_f32.reshape(1, 32);  // 32 x N
  cv::Mat mt;
  cv::transpose(m, mt);                   // N x 32

  if (bt.cols != 4 || mt.cols != 32 || st.cols != NUM_COCO_CLASSES) {
    RCLCPP_WARN(get_logger(),
        "decode_yolo_split_all_scores: shape mismatch after reshape/transpose");
    return {};
  }

  std::vector<Detection> dets;
  for (int i = 0; i < n; ++i) {
    // Argmax over 80 class scores for this anchor
    const float * row = st.ptr<float>(i);
    int best_cls = 0;
    float best_score = row[0];
    for (int c = 1; c < NUM_COCO_CLASSES; ++c) {
      if (row[c] > best_score) { best_score = row[c]; best_cls = c; }
    }
    if (best_score < score_thresh_) continue;

    float cx = bt.at<float>(i, 0), cy = bt.at<float>(i, 1);
    float bw = bt.at<float>(i, 2), bh = bt.at<float>(i, 3);

    // Convert cx,cy,w,h → x1,y1,x2,y2
    float x1 = cx - bw * 0.5f, y1 = cy - bh * 0.5f;
    float x2 = cx + bw * 0.5f, y2 = cy + bh * 0.5f;

    // Scale from normalised [0,1] if needed (heuristic: cx > 2.0 → pixel space)
    if (cx <= 2.0f) {
      x1 *= input_w; x2 *= input_w;
      y1 *= input_h; y2 *= input_h;
    }
    x1 = std::max(0.0f, std::min(static_cast<float>(input_w),  x1));
    x2 = std::max(0.0f, std::min(static_cast<float>(input_w),  x2));
    y1 = std::max(0.0f, std::min(static_cast<float>(input_h), y1));
    y2 = std::max(0.0f, std::min(static_cast<float>(input_h), y2));
    if (x2 <= x1 || y2 <= y1) continue;

    Detection d;
    d.cls = best_cls; d.score = best_score;
    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    d.coeff.resize(32);
    for (int k = 0; k < 32; ++k) d.coeff[k] = mt.at<float>(i, k);
    dets.push_back(std::move(d));
  }
  return nms(std::move(dets));
}

// ── parse YOLO outputs ───────────────────────────────────────────────────────
//
// Supports the INT8-fixed split binary which emits four named tensors:
//   boxes  (1,  4, 8400) — box coords (cx,cy,w,h) in pixel space
//   scores (1, 80, 8400) — sigmoid class scores for all 80 COCO classes
//   coeffs (1, 32, 8400) — mask prototype coefficients
//   proto  (1, 32, 160, 160) — mask prototypes
//
// Tensor identification is done by shape alone so it works regardless of
// the order in which the inference backend returns the tensors.

void MidasYoloFusionNode::parse_yolo_outputs(
    const custom_msg::TensorList & tensors, int input_w, int input_h,
    std::vector<Detection> & dets_out, cv::Mat & proto_out)
{
  dets_out.clear();
  proto_out = cv::Mat();

  // Collect all tensors as flat mats with their shapes
  struct TensorInfo { cv::Mat flat; std::vector<uint32_t> shape; std::string name; };
  std::vector<TensorInfo> infos;
  for (const auto & t : tensors.tensor_list) {
    infos.push_back({tensor_to_mat(t), t.shape, t.name});
  }

  // One-time log of the detected output tensor layout, useful for verifying
  // that the loaded model binary matches the expected split-output format.
  static std::once_flag shape_dump_flag;
  std::call_once(shape_dump_flag, [&]() {
    RCLCPP_INFO(get_logger(), "YOLO output tensors (%zu):", infos.size());
    for (const auto & info : infos) {
      std::string sh_str;
      for (size_t i = 0; i < info.shape.size(); ++i) {
        sh_str += (i ? "," : "[");
        sh_str += std::to_string(info.shape[i]);
      }
      sh_str += "]";
      RCLCPP_INFO(get_logger(), "  name='%s' shape=%s elems=%zu",
          info.name.c_str(), sh_str.c_str(), info.flat.total());
    }
  });

  cv::Mat proto_flat;
  std::vector<uint32_t> proto_shape;
  cv::Mat boxes_flat, all_scores_flat, coeffs_flat;

  for (auto & info : infos) {
    const auto & sh = info.shape;

    // Proto: 4D tensor with a dim == 32 (mask prototypes, shape [1,32,H,W])
    if (sh.size() == 4) {
      if (sh[1] == 32 || sh[3] == 32) {
        proto_flat  = info.flat;
        proto_shape = sh;
        continue;
      }
    }

    // 3D tensors: boxes [1,4,N], all_scores [1,80,N], coeffs [1,32,N]
    if (sh.size() == 3) {
      uint32_t d1 = sh[1], d2 = sh[2];
      if ((d1 == 4 || d2 == 4) && boxes_flat.empty()) {
        boxes_flat = info.flat; continue;
      }
      if ((d1 == static_cast<uint32_t>(NUM_COCO_CLASSES) ||
           d2 == static_cast<uint32_t>(NUM_COCO_CLASSES)) &&
          all_scores_flat.empty()) {
        all_scores_flat = info.flat; continue;
      }
      if ((d1 == 32 || d2 == 32) && coeffs_flat.empty()) {
        coeffs_flat = info.flat; continue;
      }
    }
  }

  if (proto_flat.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "parse_yolo_outputs: proto tensor not found in %zu tensors",
        tensors.tensor_list.size());
    return;
  }
  if (boxes_flat.empty() || all_scores_flat.empty() || coeffs_flat.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "parse_yolo_outputs: missing tensors — boxes=%s scores=%s coeffs=%s",
        boxes_flat.empty() ? "missing" : "ok",
        all_scores_flat.empty() ? "missing" : "ok",
        coeffs_flat.empty() ? "missing" : "ok");
    return;
  }

  // Reshape proto to 3D cv::Mat [ph, pw, 32] (HWC)
  int ph = 1, pw = 1;
  if (proto_shape.size() == 4) {
    if (proto_shape[1] == 32) { ph = proto_shape[2]; pw = proto_shape[3]; }
    else                       { ph = proto_shape[1]; pw = proto_shape[2]; }
  }
  if (proto_shape.size() == 4 && proto_shape[1] == 32) {
    // Layout [1, 32, H, W] → transpose to [H, W, 32]
    cv::Mat chw = proto_flat.reshape(1, 32);  // 32 x (ph*pw)
    cv::Mat hwc;
    cv::transpose(chw, hwc);                  // (ph*pw) x 32
    int sizes[3] = {ph, pw, 32};
    proto_out = hwc.reshape(1, 3, sizes);     // ph x pw x 32 (3D)
  } else {
    // Already [1, H, W, 32] — just reshape
    int sizes[3] = {ph, pw, 32};
    proto_out = proto_flat.reshape(1, 3, sizes);
  }

  dets_out = decode_yolo_split_all_scores(
      boxes_flat, all_scores_flat, coeffs_flat, input_w, input_h);
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
  auto t0 = clk::now();
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
    // Decode MiDaS depth and parse YOLO outputs sequentially.
    // std::async caused thread oversubscription with the executor pool.
    cv::Mat depth_f32, depth_gray, depth_color;
    decode_midas_depth(
        frame.midas_tensors->tensor_list[0],
        frame.image_bgr.cols, frame.image_bgr.rows,
        depth_f32, depth_gray, depth_color);
    std::vector<Detection> dets;
    cv::Mat proto_hwc;
    parse_yolo_outputs(
        *frame.yolo_tensors, frame.yolo_input_w, frame.yolo_input_h,
        dets, proto_hwc);
    auto t1 = clk::now();

    // Draw overlay on depth_color
    cv::Mat overlay = depth_color.clone();
    draw_overlay(overlay, depth_f32, dets, proto_hwc,
        frame.yolo_input_w, frame.yolo_input_h);
    auto t2 = clk::now();

    // Build header
    std_msgs::msg::Header hdr;
    hdr.stamp.sec    = frame.header_sec;
    hdr.stamp.nanosec = frame.header_nsec;

    overlay_pub_.publish(mat_to_image_msg(overlay,     "bgr8",  hdr));
    depth_color_pub_.publish(mat_to_image_msg(depth_color, "bgr8",  hdr));
    depth_gray_pub_.publish(mat_to_image_msg(depth_gray,  "mono8", hdr));
    auto t3 = clk::now();

    uint64_t cnt = ++processed_count_;
    auto now = this->now();
    if ((now - last_log_time_).seconds() > 2.0) {
      RCLCPP_INFO(get_logger(),
          "processed=%lu pending=%zu dets=%zu | decode+parse=%.2fms overlay=%.2fms pub=%.2fms total=%.2fms",
          cnt, pending_.size(), dets.size(),
          ms(t1-t0).count(), ms(t2-t1).count(), ms(t3-t2).count(), ms(t3-t0).count());
      last_log_time_ = now;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "fusion failed: %s", e.what());
  }
}

}  // namespace sample_midas_yolo_parallel_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(
    sample_midas_yolo_parallel_cpp::MidasYoloFusionNode)
