# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import cv2
import numpy as np
import subprocess
import os

def nv12_to_bgr(nv12_image, width, height):
    """
    Convert NV12 image to BGR format.
    """
    yuv = nv12_image.reshape((height * 3 // 2, width))
    bgr_data = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    return bgr_data

def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0]
        padh = 0
        padw = 256 - w1
        scale = size0[1] / w1
    else:
        h1 = 256 * size0[0] // size0[1]
        w1 = 256
        padh = 256 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))

    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)), mode='constant', constant_values=0)

    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128,128))

    return img1, img2, scale, pad


def denormalize_detections(detections, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        scale: scalar that was used to resize the image
        pad: padding in the x and y dimensions

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """

    if isinstance(detections, list):
        detections = detections[0]

    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections

def intersect(box_a, box_b):
    """Compute the intersection area between sets of bounding boxes.
    Args:
      box_a: (numpy array) bounding boxes, Shape: [A, 4].
      box_b: (numpy array) bounding boxes, Shape: [B, 4].
    Return:
      (numpy array) intersection area, Shape: [A, B].

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(np.expand_dims(box_a[:, 2:], axis=1), np.expand_dims(box_b[:, 2:], axis=0))
    min_xy = np.maximum(np.expand_dims(box_a[:, :2], axis=1), np.expand_dims(box_b[:, :2], axis=0))
    inter_dim = np.maximum(max_xy - min_xy, 0)
    return inter_dim[:, :, 0] * inter_dim[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the Jaccard overlap of two sets of boxes. Also known as IOU.
    Args:
        box_a: (numpy array) Ground truth bounding boxes, Shape: [num_objects, 4]
        box_b: (numpy array) Prior boxes, Shape: [num_priors, 4]
    Return:
        jaccard overlap: (numpy array) Shape: [box_a.shape[0], box_b.shape[0]]

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    inter_area = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union_area = np.expand_dims(area_a, axis=1) + np.expand_dims(area_b, axis=0) - inter_area
    return inter_area / union_area

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and a set of other boxes.

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    return jaccard(np.expand_dims(box, axis=0), other_boxes)

class BlazeDetector():
    """ Base class for detector models.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/
    """
    def __init__(self):
        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/hand_tracking/subgraphs/hand_detection_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 2944
        self.num_coords = 18
        self.score_clipping_thresh = 100.0
        self.x_scale = 256.0
        self.y_scale = 256.0
        self.h_scale = 256.0
        self.w_scale = 256.0
        self.min_score_thresh = 0.5
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 7

        # These settings are for converting detections to ROIs which can then
        # be extracted and feed into the landmark network
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        self.detection2roi_method = 'box'
        # mediapipe/graphs/hand_tracking/subgraphs/hand_detection_cpu.pbtxt
        self.kp1 = 0
        self.kp2 = 2
        self.theta0 = np.pi/2
        self.dscale = 2.6
        self.dy = -0.5

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_anchors(self, path):
        """Load anchors from a given path."""
        self.anchors = np.load(path)
        assert self.anchors.ndim == 2, "Anchors must be a 2-dimensional array"
        assert self.anchors.shape[0] == self.num_anchors, "Number of anchors does not match"
        assert self.anchors.shape[1] == 4, "Each anchor must have 4 coordinates"

    def palm_detector_qnn_preprocess(self, img):
        """Preprocesses the image for the palm detector.
        Arguments:
            img: a NumPy array of shape (H, W, 3). The image's height and width should be
                256 pixels.
        Returns:
            A NumPy array of shape (1, 256, 256, 3) with the image pixels normalized to the range [-1, 1].
        """
        if isinstance(img, np.ndarray):
            img_batch = np.expand_dims(img, axis=0)
        return self.predict_on_image_np(img_batch)[0]

    def predict_on_image_np(self, image):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3). The image's height and width should be
                256 pixels.

        Returns:
            A numpy array with hand detections.
        """
        assert image.shape[3] == 3
        assert image.shape[2] == self.y_scale
        assert image.shape[1] == self.x_scale

        # Converts the image pixels to the range [-1, 1].
        image = image.astype(np.float32) / 255.0

        return image

    def palm_tensor_to_data(self, msg):
        """Parse the palm_detector outputs: box_coords[1,2944,18] and box_scores[1,2944].

        Distinguish the two outputs by element count to avoid depending on the
        TensorList order:
          - 2944*18 = 52992 -> box_coords
          - 2944      = 2944  -> box_scores
        """
        box_coords_data = None
        box_scores_data = None
        for t in msg.tensor_list:
            arr = np.array(t.data).view(np.float32)
            if arr.size == 2944 * 18:
                box_coords_data = arr.reshape((1, 2944, 18))
            elif arr.size == 2944:
                box_scores_data = arr.reshape((1, 2944, 1))

        # Fallback: if the size-based match failed, assume the original order.
        if box_coords_data is None or box_scores_data is None:
            box_coords_data = np.array(msg.tensor_list[1].data).view(np.float32).reshape((1, 2944, 18))
            box_scores_data = np.array(msg.tensor_list[0].data).view(np.float32).reshape((1, 2944, 1))

        return box_coords_data, box_scores_data

    def palm_detector_qnn_postprocess(self, palm_tensor_msg):
        # 1. Postprocess the raw predictions:
        out1, out2 = self.palm_tensor_to_data(palm_tensor_msg)

        out1= np.expand_dims(out1[0], axis=0)
        out2= np.expand_dims(out2[0], axis=0)

        detections = self._tensors_to_detections(out1, out2, self.anchors)

        # 2. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            if len(faces) > 0:
                faces = np.stack(faces)
            else:
                faces = np.zeros((0, self.num_coords + 1))
            filtered_detections.append(faces)       

        return filtered_detections

    def detection2roi(self, detection):
        """ Convert detections from detector to an oriented bounding box.

        Adapted from:
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

        The center and size of the box is calculated from the center 
        of the detected box. Rotation is calcualted from the vector
        between kp1 and kp2 relative to theta0. The box is scaled
        and shifted by dscale and dy.

        """
        if self.detection2roi_method == 'box':
            # compute box center and scale
            # use mediapipe/calculators/util/detections_to_rects_calculator.cc
            xc = (detection[:,1] + detection[:,3]) / 2
            yc = (detection[:,0] + detection[:,2]) / 2
            scale = (detection[:,3] - detection[:,1]) # assumes square boxes

        elif self.detection2roi_method == 'alignment':
            # compute box center and scale
            # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            xc = detection[:,4+2*self.kp1]
            yc = detection[:,4+2*self.kp1+1]
            x1 = detection[:,4+2*self.kp2]
            y1 = detection[:,4+2*self.kp2+1]
            scale = ((xc-x1)**2 + (yc-y1)**2).sqrt() * 2
        else:
            raise NotImplementedError(
                "detection2roi_method [%s] not supported"%self.detection2roi_method)

        yc += self.dy * scale
        scale *= self.dscale

        # compute box rotation
        x0 = detection[:,4+2*self.kp1]
        y0 = detection[:,4+2*self.kp1+1]
        x1 = detection[:,4+2*self.kp2]
        y1 = detection[:,4+2*self.kp2+1]
        theta = np.arctan2(y0-y1, x0-x1) - self.theta0
        # theta = torch.atan2(y0-y1, x0-x1) - self.theta0
        return xc, yc, scale, theta


    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor 
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert len(raw_box_tensor.shape) == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert len(raw_score_tensor.shape) == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        # The AI Hub quantized model's box_scores are already sigmoid
        # probabilities in [0,1]; only apply sigmoid again when the values look
        # like raw logits (outside [0,1]), to avoid a double sigmoid pushing
        # scores through the threshold.
        if np.nanmax(raw_score_tensor) > 1.0 or np.nanmin(raw_score_tensor) < 0.0:
            thresh = self.score_clipping_thresh
            raw_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
            detection_scores = 1 / (1 + np.exp(-raw_score_tensor))
        else:
            detection_scores = raw_score_tensor
        detection_scores = np.squeeze(detection_scores, axis=-1)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i][mask[i]]
            scores = detection_scores[i][mask[i]]
            scores = np.expand_dims(scores, axis=-1)
            concatenated_detections = np.concatenate([boxes, scores], axis=-1)
            output_detections.append(concatenated_detections)

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        # Standard MediaPipe/blaze weights output anchor-relative offsets (the
        # mean over all anchors is ~0); the AI Hub export used here bakes the
        # anchor decode into the model graph, so box_coords are ABSOLUTE
        # coordinates directly in the 256 input space (the mean over all anchors
        # is ~128 = x_scale/2). Distinguish the two automatically by the mean:
        # in the absolute case we must NOT add the anchor center again
        # (otherwise the coordinates would be pushed outside [0,1]).
        if np.nanmean(np.abs(raw_boxes[..., :4])) > self.x_scale * 0.25:
            norm = raw_boxes / self.x_scale  # 256 space -> normalized [0,1]
            # ch0..3 = corners (x1,y1,x2,y2); clamp corner order with min/max to
            # get [ymin,xmin,ymax,xmax]
            x1, y1, x2, y2 = norm[..., 0], norm[..., 1], norm[..., 2], norm[..., 3]
            xmin = np.minimum(x1, x2); xmax = np.maximum(x1, x2)
            ymin = np.minimum(y1, y2); ymax = np.maximum(y1, y2)
            boxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)
            kps = []
            for k in range(self.num_keypoints):
                off = 4 + k * 2
                kps.append(norm[..., off])       # keypoint x
                kps.append(norm[..., off + 1])   # keypoint y
            if kps:
                boxes = np.concatenate([boxes, np.stack(kps, axis=-1)], axis=-1)
            return boxes

        boxes = np.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        ymin = y_center - h / 2.  # ymin
        xmin = x_center - w / 2.  # xmin
        ymax = y_center + h / 2.  # ymax
        xmax = x_center + w / 2.  # xmax

        boxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)

        keypoints = []
        for k in range(self.num_keypoints):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            keypoints.append(keypoint_x)
            keypoints.append(keypoint_y)
        if keypoints:
            keypoints = np.stack(keypoints, axis=-1)
            boxes = np.concatenate([boxes, keypoints], axis=-1)
        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a numpy array of shape (count, 17).

        Returns a list of numpy arrays, one for each detected face.
        
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        scores = detections[:, self.num_coords]
        remaining = np.argsort(scores)[::-1]

        while remaining.shape[0] > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]

            ious = overlap_similarity(first_box, other_boxes)
            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold

            overlapping = remaining[mask[0]]
            remaining = remaining[~mask[0]]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.copy()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :self.num_coords]
                scores = detections[overlapping, self.num_coords:self.num_coords+1]
                total_score = np.sum(scores)
                weighted = np.sum(coordinates * scores, axis=0) / total_score
                weighted_detection = np.concatenate([weighted, [total_score / len(overlapping)]], axis=0)

            output_detections.append(weighted_detection)

        return output_detections

class BlazeLandmark():
    """ Base class for landmark models. 

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """

    def __init__(self):
        # size of ROIs used for input
        # NOTE: the AI Hub hand_landmark_detector input is 224 (the older .bin
        # used by sample_hand_detection is 256)
        self.resolution = 224

    def landmark_tensor_to_data(self, msg):
        """Parse scores / lr / landmarks from the landmark inference TensorList.

        The AI Hub hand_landmark_detector.dlc has 4 outputs:
            landmarks[1,63], scores[1,1], lr[1,1], world_landmarks[1,63]
        The QNN TensorList order is not guaranteed to match the metadata
        declaration, so match by tensor.name first and fall back to guessing by
        shape/order when the name match fails.
        """
        # Keep the original TensorList order + name + data (QNN does not
        # guarantee order and may not carry semantic names through).
        items = [(t.name, t.name.lower(), np.array(t.data).view(np.float32))
                 for t in msg.tensor_list]

        # Print the actual output names/sizes once on the first frame, to help
        # verify on the board whether QNN carried the semantic names through.
        if not getattr(self, '_names_logged', False):
            print('[landmark] output tensors: '
                  + ', '.join(f'{n}[{a.size}]' for n, _, a in items), flush=True)
            self._names_logged = True

        size63 = [(n, ln, a) for (n, ln, a) in items if a.size == 63]
        size1 = [(n, ln, a) for (n, ln, a) in items if a.size == 1]

        # landmarks (used for classification, NOT world_landmarks):
        # prefer a name containing 'landmark' but not 'world'; when names are
        # not semantic, take the first non-world size-63; failing that, fall
        # back to the first size-63 (in the metadata order landmarks comes
        # before world_landmarks).
        landmarks_data = None
        for n, ln, a in size63:
            if 'landmark' in ln and 'world' not in ln:
                landmarks_data = a
                break
        if landmarks_data is None:
            for n, ln, a in size63:
                if 'world' not in ln:
                    landmarks_data = a
                    break
        if landmarks_data is None and size63:
            landmarks_data = size63[0][2]

        # scores / lr are both size-1: pick by name first, then fill the rest
        # in order of appearance (metadata: scores comes before lr).
        scores_data = next((a for n, ln, a in size1 if 'score' in ln), None)
        lr_data = next((a for n, ln, a in size1 if ('lr' in ln or 'hand' in ln)), None)
        leftovers = [a for n, ln, a in size1
                     if a is not scores_data and a is not lr_data]
        if scores_data is None and leftovers:
            scores_data = leftovers.pop(0)
        if lr_data is None and leftovers:
            lr_data = leftovers.pop(0)

        assert landmarks_data is not None and scores_data is not None and lr_data is not None, (
            'landmark outputs unresolved: '
            + ', '.join(f'{n}[{a.size}]' for n, _, a in items))
        assert landmarks_data.size == 63, f'landmarks size={landmarks_data.size} != 63'

        landmarks_data = landmarks_data.reshape((1, 21, 3))
        return scores_data, lr_data, landmarks_data

    def extract_roi(self, frame, xc, yc, theta, scale):
        """Extract region of interest from the frame."""
        # take points on unit square and transform them according to the roi
        points = np.array([[-1, -1, 1, 1],
                        [-1, 1, -1, 1]], dtype=np.float32)
        points = np.reshape(points, (1, 2, 4))
        points = points * np.reshape(scale, (-1, 1, 1)) / 2
        theta = np.reshape(theta, (-1, 1, 1))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.concatenate([
            np.concatenate([cos_theta, -sin_theta], axis=2),
            np.concatenate([sin_theta, cos_theta], axis=2)
        ], axis=1)
        center = np.concatenate([np.reshape(xc, (-1, 1, 1)), np.reshape(yc, (-1, 1, 1))], axis=1)
        points = np.matmul(R, points) + center

        # use the points to compute the affine transform that maps 
        # these points back to the output square
        res = self.resolution
        points1 = np.array([[0, 0, res-1],
                            [0, res-1, 0]], dtype=np.float32).T
        affines = []
        imgs = []
        for i in range(points.shape[0]):
            pts = points[i, :, :3].T  # Convert to NumPy for cv2 functions
            pts = np.float32(pts)
            M = cv2.getAffineTransform(pts, points1)
            img = cv2.warpAffine(frame, M, (res, res))
            imgs.append(img)
            affine = cv2.invertAffineTransform(M).astype('float32')
            affines.append(affine)
        if imgs:
            imgs = np.stack(imgs) / 255.0
            affines = np.stack(affines)
        else:
            imgs = np.zeros((0, 3, res, res), dtype=np.float32)
            affines = np.zeros((0, 2, 3), dtype=np.float32)

        return imgs, affines, points

    def denormalize_landmarks(self, landmarks, affines):
        """Denormalize the landmarks."""
        # The AI Hub landmark model outputs pixel coordinates directly within
        # the 224 crop box; only multiply by the resolution when the coordinates
        # look normalized [0,1], to avoid re-scaling coordinates that are already
        # in pixels.
        if np.nanmax(np.abs(landmarks[:, :, :2])) <= 2.0:
            landmarks[:, :, :2] *= self.resolution
        for i in range(landmarks.shape[0]):
            landmark = landmarks[i]
            affine = affines[i]
            landmark = np.matmul(affine[:, :2], np.transpose(landmark[:, :2])) + affine[:, 2:]
            landmarks[i, :, :2] = np.transpose(landmark)
        return landmarks
