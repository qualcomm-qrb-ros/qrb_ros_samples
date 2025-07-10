# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import cv2
import numpy as np
import subprocess
import os
from ament_index_python.packages import get_package_share_directory
# import rospy

package_share_directory = get_package_share_directory("sample_hand_detection")

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
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
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

    def qnn_preprocess(self, image):
        """Converts the image pixels to the range [-1, 1]."""
        image.astype(np.float32).tofile(os.path.join(package_share_directory,"image_detector.raw"))
    
    def qnn_postprocess(self):
        output0_path = os.path.join(package_share_directory,"output_MediaPipeHandDetector/Result_0/output_0.raw")
        output0_raw = open(output0_path, 'rb')
        out0 = np.fromfile(output0_raw, dtype=np.float32)
        out0 = out0.reshape((1, 2944, 18))
        output1_path = os.path.join(package_share_directory,"output_MediaPipeHandDetector/Result_0/output_1.raw")
        output1_raw = open(output1_path, 'rb')
        out1 = np.fromfile(output1_raw, dtype=np.float32)
        out1 = out1.reshape((1, 2944, 1))

        return out0, out1

    def _preprocess_np(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.astype(np.float32) / 255.0

    def predict_on_image_np(self, img, model_file_path):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3). The image's height and width should be 
                128 pixels.

        Returns:
            A numpy array with face detections.
        """
        if isinstance(img, np.ndarray):
            img = np.transpose(img, axes=[2, 0, 1])
            img_batch = np.expand_dims(img, axis=0)
        return self.predict_on_batch_np(img_batch, model_file_path)[0]

    def predict_on_batch_np(self, x, model_file_path):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        assert x.shape[1] == 3
        assert x.shape[2] == self.y_scale
        assert x.shape[3] == self.x_scale

        # 1. Preprocess the images into tensors:
        x = self._preprocess_np(x)
        self.qnn_preprocess(x)

        # 2. Run the neural network:
        input_file_path = os.path.join(package_share_directory,"image_detector.txt")
        output_file_path = os.path.join(package_share_directory,"output_MediaPipeHandDetector")
        with open(input_file_path, 'w') as file:
            file.write(os.path.join(package_share_directory,"image_detector.raw"))
        cmd = f"qtld-net-run --model {model_file_path} --input {input_file_path} --output {output_file_path}"
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            # rospy.logdebug("Command execution failed:", e)
            print("Command execution failed:", e)


        # 3. Postprocess the raw predictions:
        out1, out2 = self.qnn_postprocess()

        detections = self._tensors_to_detections(out1, out2, self.anchors)

        # return detections
        # 4. Non-maximum suppression to remove overlapping detections:
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

        thresh = self.score_clipping_thresh
        raw_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
        detection_scores = 1 / (1 + np.exp(-raw_score_tensor))
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
        self.resolution = 256

    def qnn_preprocess(self, image):
        """Converts the image pixels to the range [-1, 1]."""
        image.astype(np.float32).tofile(os.path.join(package_share_directory,"image_landmark.raw"))
    
    def qnn_postprocess(self):
        output0_path = os.path.join(package_share_directory,"output_MediaPipeHandLandmarkDetector/Result_0/output_0.raw")
        output0_raw = open(output0_path, 'rb')
        out0 = np.fromfile(output0_raw, dtype=np.float32)
        output1_path = os.path.join(package_share_directory,"output_MediaPipeHandLandmarkDetector/Result_0/output_1.raw")
        output1_raw = open(output1_path, 'rb')
        out1 = np.fromfile(output1_raw, dtype=np.float32)
        output2_path = os.path.join(package_share_directory,"output_MediaPipeHandLandmarkDetector/Result_0/output_2.raw")
        output2_raw = open(output2_path, 'rb')
        out2 = np.fromfile(output2_raw, dtype=np.float32)
        out2 = out2.reshape((1, 21, 3))

        return out0, out1, out2

    def run_network(self, model_file_path):
        input_file_path = os.path.join(package_share_directory,"image_landmark.txt")
        output_file_path = os.path.join(package_share_directory,"output_MediaPipeHandLandmarkDetector")
        with open(input_file_path, 'w') as file:
            file.write(os.path.join(package_share_directory,"image_landmark.raw"))
        cmd = f"qtld-net-run --model {model_file_path} --input {input_file_path} --output {output_file_path}"
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True, shell=True)
            return True
        except subprocess.CalledProcessError as e:
            # rospy.logdebug("Command execution failed:", e)
            return False

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
            imgs = np.stack(imgs)
            imgs = np.transpose(imgs, axes=[0, 3, 1, 2]) / 255.0
            affines = np.stack(affines)
        else:
            imgs = np.zeros((0, 3, res, res), dtype=np.float32)
            affines = np.zeros((0, 2, 3), dtype=np.float32)

        return imgs, affines, points

    def denormalize_landmarks(self, landmarks, affines):
        """Denormalize the landmarks."""
        landmarks[:,:,:2] *= self.resolution
        for i in range(landmarks.shape[0]):
            landmark = landmarks[i]
            affine = affines[i]
            landmark = np.matmul(affine[:, :2], np.transpose(landmark[:, :2])) + affine[:, 2:]
            landmarks[i, :, :2] = np.transpose(landmark)
        return landmarks
