# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import time
import rclpy
import numpy as np
from sample_3d_lidar_semantic_segmentation import preprocess, postprocess
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList

# raw pointcloud subscriber -> preprocess -> nn_inference_input_tensor publisher -> nn_inference_output_tensor subscriber -> postprocess -> semantic_pointcloud publisher

class LidarSemanticSegmentation(Node):
    def __init__(self):
        super().__init__('lidar_semantic_segmentation_node')

        # Declare parameters
        self.declare_parameter('fov_up', 3.0)
        self.declare_parameter('fov_down', -25.0)
        
        # Get parameter values
        self.fov_up = self.get_parameter('fov_up').value
        self.fov_down = self.get_parameter('fov_down').value
        
        # Create subscriptions and publishers
        self.raw_pointcloud_subscription = self.create_subscription(
            PointCloud2, 
            '/velodyne_points', 
            self.raw_pointcloud_callback, 
            10
        )
        
        self.semantic_pointcloud_publisher = self.create_publisher(
            PointCloud2, 
            '/semantic_points', 
            10
        )
        
        self.nn_inference_output_tensor_subscription = self.create_subscription(
            TensorList, 
            'qrb_inference_output_tensor', 
            self.nn_inference_callback, 
            10
        )
        
        self.nn_inference_input_tensor_publisher = self.create_publisher(
            TensorList, 
            'qrb_inference_input_tensor', 
            10
        )
        
        # State variables
        self.raw_pointcloud_processed_flag = True
        self.current_pointcloud_msg = None
        self.current_point_indices = None
        self.current_num_points = 0
        self.current_points_data = None
        self.first_publish_done = False  # Track if first publish has been done
        self.node_start_time = time.perf_counter()  # Record node start time

        # Timing statistics
        self.preprocess_time_ms = 0.0
        self.inference_time_ms = 0.0
        self.postprocess_time_ms = 0.0
        self.inference_start_time = 0.0
        
        self.get_logger().info(f"Lidar Semantic Segmentation Node initialized with FOV: up={self.fov_up}, down={self.fov_down}")
        self.get_logger().info("First tensor publish will be delayed by 3 seconds to wait for nn inference node")

    def raw_pointcloud_callback(self, msg):
        """Callback for raw point cloud messages"""
        self.get_logger().debug('Received raw pointcloud message')
        
        # Check if this is the first publish and add 3-second delay
        if not self.first_publish_done:
            current_time = time.perf_counter()
            elapsed_time = current_time - self.node_start_time
            
            if elapsed_time < 3.0:
                self.get_logger().debug(f"Waiting for nn inference node, {3.0 - elapsed_time:.1f} seconds remaining")
                return
            else:
                self.first_publish_done = True
                self.get_logger().info("3-second delay completed, starting to publish tensors")
        
        # Preprocess point cloud
        t0 = time.perf_counter()
        preprocessed_tensor, point_indices, num_points, points_data = preprocess.preprocess(
            msg,
            fov_up=self.fov_up,
            fov_down=self.fov_down
        )
        preprocess_time_ms = (time.perf_counter() - t0) * 1000.0

        if preprocessed_tensor is None or num_points == 0:
            self.get_logger().debug("Received empty point cloud, skipping.")
            return

        if self.raw_pointcloud_processed_flag:
            self.raw_pointcloud_processed_flag = False
            # Save the current frame state and publish for inference
            self.current_pointcloud_msg = msg
            self.current_point_indices = point_indices
            self.current_num_points = num_points
            self.current_points_data = points_data
            # Save preprocess time of the dispatched frame so it can be
            # printed together with the matching inference/postprocess times.
            self.preprocess_time_ms = preprocess_time_ms
            nn_inference_input_tensor_msg = self.make_nn_inference_input_tensor_msg(msg, preprocessed_tensor)
            self.nn_inference_input_tensor_publisher.publish(nn_inference_input_tensor_msg)
            # Record time after publish so inference_time only covers the
            # actual inference duration, excluding serialization overhead.
            self.inference_start_time = time.perf_counter()
            self.get_logger().debug("Published nn_inference_input_tensor_msg")
        else:
            self.get_logger().debug("Skip nn_inference_input_tensor_msg (previous inference still processing)")

    def make_nn_inference_input_tensor_msg(self, original_msg, preprocessed_tensor):
        """Create TensorList message for NN inference input"""
        nn_inference_input_tensor_msg = TensorList()
        nn_inference_input_tensor_msg.header = original_msg.header

        tensor = Tensor()
        tensor.name = "lidar_semantic_segmentation_nn_inference_input_tensor"
        tensor.data_type = 2  # float32
        tensor.shape = preprocessed_tensor.shape
        tensor.data = preprocessed_tensor.tobytes()

        nn_inference_input_tensor_msg.tensor_list.append(tensor)
        return nn_inference_input_tensor_msg

    def nn_inference_callback(self, nn_inference_output_tensor_msg):
        """Callback for NN inference output tensor messages"""
        self.get_logger().debug('Received nn_inference_output_tensor message')
        
        if self.current_pointcloud_msg is None:
            self.get_logger().warning("No pointcloud data available for postprocessing")
            return

        t_inference_end = time.perf_counter()
        self.inference_time_ms = (t_inference_end - self.inference_start_time) * 1000.0

        for result in nn_inference_output_tensor_msg.tensor_list:
            # Convert tensor data to numpy array and reshape to [1, 20, 64, 2048]
            output_np_array = np.array(result.data).view(np.float32)
            output_tensor = output_np_array.reshape((1, 20, 64, 2048))

            t_post_start = time.perf_counter()
            semantic_pointcloud_msg = postprocess.postprocess(
                self.current_pointcloud_msg,
                output_tensor,
                self.current_point_indices,
                self.current_num_points,
                self.current_points_data
            )
            self.postprocess_time_ms = (time.perf_counter() - t_post_start) * 1000.0
            total_time_ms = self.preprocess_time_ms + self.inference_time_ms + self.postprocess_time_ms

            if semantic_pointcloud_msg:
                self.semantic_pointcloud_publisher.publish(semantic_pointcloud_msg)
                self.get_logger().info(
                    f"Points: {self.current_num_points} | "
                    f"Preprocess: {self.preprocess_time_ms:.1f} ms | "
                    f"Inference: {self.inference_time_ms:.1f} ms | "
                    f"Postprocess: {self.postprocess_time_ms:.1f} ms | "
                    f"Total: {total_time_ms:.1f} ms"
                )

        self.raw_pointcloud_processed_flag = True

def main(args=None):
    rclpy.init(args=args)
    node = LidarSemanticSegmentation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node shutdown by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    