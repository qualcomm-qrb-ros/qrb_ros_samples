# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # kept if needed elsewhere
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np
import struct
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList


class ResNet101PostProcessNode(Node):
    def __init__(self):
        super().__init__('resnet101_postprocess_node')

        # Internal active flag to track node lifecycle (used to avoid publishing after destroy)
        self._active = True

        # Subscriber and publisher
        self.subscriber = self.create_subscription(
            TensorList,
            'qrb_inference_output_tensor',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            String,
            'resnet101_output',
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info('Initial ROS Node: resnet101_postprocess_node')

        # Model path parameter (default)
        self.declare_parameter('model_path', '/opt/model/')
        self.model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'model path: {self.model_path}')

        # Read labels safely
        labels_path = os.path.join(self.model_path, "imagenet_labels.txt")
        self.class_names = []
        if os.path.isfile(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    # strip newline characters for clean publishing
                    self.class_names = [line.strip() for line in f.readlines() if line.strip()]
                self.get_logger().info(f'Loaded {len(self.class_names)} class labels.')
            except Exception as e:
                self.get_logger().error(f'Failed to read labels file "{labels_path}": {e}')
        else:
            self.get_logger().warning(f'Labels file not found at "{labels_path}". Publishing raw indices.')

    def is_running(self) -> bool:
        """
        Return True if rclpy is still OK and this node has not been destroyed.
        """
        return rclpy.ok() and getattr(self, "_active", False)

    def destroy_node(self):
        """
        Override destroy_node to flip internal active flag for safe shutdown.
        """
        # Mark node inactive before delegating to base destroy_node
        self._active = False
        try:
            super().destroy_node()
        except Exception as e:
            # Log but swallow exceptions during destroy to avoid crashes during shutdown
            print(f'Exception while destroying node: {e}')

    def close(self):
        """
        Helper to deterministically cleanup the node. Can be called from external code or main.
        """
        try:
            # In rclpy there is no explicit unregister for publisher/subscriber beyond destroy_node,
            # but we'll attempt to set them to None for GC and to avoid accidental use.
            self.subscriber = None
            self.publisher = self.publisher  # keep publisher reference for the moment
        except Exception:
            pass
        # Call destroy_node to complete lifecycle transition
        self.destroy_node()

    def _safe_publish_string(self, text: str):
        """
        Publish a string if the node is running and the publisher looks valid.
        This centralizes publish safety checks.
        """
        if not self.is_running():
            print(f"Node not running; skipping publish.")
            return False

        if not hasattr(self, "publisher") or self.publisher is None:
            return False

        # Ensure payload is a string
        if text is None:
            return False

        try:
            # Build the ROS message explicitly to ensure correct type
            msg = String()
            # Guard: ensure we only publish a string (no binary/objects)
            if isinstance(text, bytes):
                # decode bytes (best-effort)
                try:
                    text = text.decode("utf-8", errors="replace")
                except Exception:
                    text = str(text)
            else:
                text = str(text)
            msg.data = text
            # Final safety check: publisher has publish method
            if not hasattr(self.publisher, "publish"):
                #self.get_logger().error('Publisher object does not expose publish(); cannot publish.')
                return False
            self.publisher.publish(msg)
            return True
        except Exception as e:
            #self.get_logger().error(f'Failed to publish message: {e}')
            return False

    def image_callback(self, msg: TensorList):
        """
        Callback invoked when inference output tensor list arrives.
        The node extracts the predicted class index and publishes either the label (if available)
        or the raw index as a string. Defensive checks prevent publishing after shutdown and avoid
        invalid message types.
        """
        if not self.is_running():
            # Defensive: do nothing if node has been shut down
            return

        if not msg or not hasattr(msg, 'tensor_list'):
            return

        try:
            for result_tensor in msg.tensor_list:
                # result_tensor.data could be a list of floats/ints
                if not hasattr(result_tensor, 'data') or result_tensor.data is None:
                    continue

                # Convert to numpy array safely
                try:
                    output_data = np.array(result_tensor.data, dtype=np.float32)
                except Exception:
                    # If conversion fails, attempt best-effort fallback
                    try:
                        output_data = np.asarray([float(x) for x in result_tensor.data], dtype=np.float32)
                    except Exception as e:
                        continue

                if output_data.size == 0:
                    continue

                # Find predicted class index
                predicted_class = int(np.argmax(output_data))

                # Resolve label if available
                class_label = None
                if 0 <= predicted_class < len(self.class_names):
                    class_label = self.class_names[predicted_class]
                else:
                    class_label = str(predicted_class)

                # Publish safely
                success = self._safe_publish_string(class_label)
                
        except Exception as e:
            # Guard against unexpected exceptions and keep node alive
           print(f'Error processing TensorList: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ResNet101PostProcessNode()
        # Spin until interrupted. If rclpy.shutdown() or node.destroy_node() is called, spin will exit.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print('KeyboardInterrupt received; shutting down node.')
    except Exception as e:
        # Log unexpected exceptions
        print(f'Unhandled exception before node creation: {e}', file=sys.stderr)
    finally:
        # Ensure deterministic cleanup
        try:
            if node:
                node.close()
        except Exception as e:
            # Best effort cleanup; log to stderr if node cannot log.          
            print(f'Error during node cleanup: {e}', file=sys.stderr)
        # Shutdown rclpy if still running
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # ignore shutdown errors
            pass


if __name__ == '__main__':
    main()
