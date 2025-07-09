import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np
import subprocess
import sys
from ament_index_python.packages import get_package_share_directory
import shutil
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList
import struct
from std_msgs.msg import String


class ResNet101QuantizedNode(Node):
    def __init__(self):
        super().__init__('resnet101_quantized_node')
        self.subscriber = self.create_subscription(
            TensorList,
            'qrb_inference_output_tensor',  # subscriber topic
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            String,
            'resnet101_quantized_output',  # publish topic
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info(f'Initial ROS Node resnet101 quantized')
        
        self.declare_parameter('model_path', '/opt/model/')
        
        self.model_path = self.get_parameter('model_path').value
        
        self.get_logger().info(f'model path: {self.model_path}')
    
        
    def softmax(self,x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))  # minus the max value in the sample data to improve the stability
            return e_x / e_x.sum()
            
        
    def image_callback(self, msg):
        try:
            for result_tensor in msg.tensor_list:  # 遍历 tensor_list
                self.get_logger().info(f'result_tensor shape is {result_tensor.shape}')
                output_data = np.array(result_tensor.data)
                output_data = output_data.view(np.float32)
                confidences = np.squeeze(output_data)
                confidences = self.softmax(confidences)
                max_value = np.max(confidences)
                predicted_class = np.argmax(confidences)
                self.get_logger().info(f'predicted_class is {predicted_class}, max_value is {max_value}')
               
                # # open the lable file
                with open(self.model_path+"imagenet_labels.txt", "r") as file:
                    # read the content of the lable
                    class_names = file.readlines()
                    class_id = class_names[predicted_class]
                    self.publisher.publish(String(data=class_id))
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    package_share_directory = get_package_share_directory('sample_resnet101_quantized')
    print(f"package_share_directory is {package_share_directory}")
    global workdir 
    workdir = package_share_directory
    if not os.path.exists(workdir+"/input"):
        os.makedirs(workdir+"/input" )
        shutil.copy(package_share_directory+"/input.txt", workdir+"/input/input.txt")
    if not os.path.exists(workdir + "/output"):
        os.makedirs(workdir+"/output" )
        os.chmod(workdir+"/output", 0o777)
         
    rclpy.init(args=args)
    node = ResNet101QuantizedNode() 
       
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
