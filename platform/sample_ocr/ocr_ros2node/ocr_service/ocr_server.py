'''
Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

from ocr_msg.srv import OcrRequest
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty
import time
import threading

from pytesseract import Output
import pytesseract
import cv2
from numpy import *
from PIL import Image
import os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


screenLevels = 255.0
Debug = False
bridge = CvBridge()

nodes = {}

class OcrProcess(Node):
    def __init__(self ,name,image_node):
        super().__init__(name)
        self.check_image_node_alive(image_node)
        self.create_subscription(Image, image_node, self.get_res, 10)
        self.publisher = self.create_publisher(String, image_node+"_ocr", 10)
        self.get_logger().info("init image Subscriber %s" % name)
        #self.timer = threading.Timer(10.0, self.timer_callback)
        #self.timer.start()

    def process_function(self,msg):
        #self.get_logger().info('I heard: "%s"' % msg.data)
        # Reset the timer every time a message is received
        self.get_res(msg.data)
        self.timer.cancel()
        self.timer = threading.Timer(10.0, self.timer_callback)
        self.timer.start()

    def timer_callback(self):
        self.get_logger().info('No message received in the last 10 seconds, shutting down')
        self.destroy_node()

    def check_image_node_alive(self,topic_name):
        while True:
            topics = self.get_topic_names_and_types()
            print(topics)
            #self.get_logger().info("Topic %s exist" % topic_name)
            if any(topic == topic_name for topic, types in topics):
                self.get_logger().info("Topic %s exist" % topic_name)
                break
            elif time.time() - self.start_time > 10:
                self.get_logger().info("Topic %s does not exist after 10 seconds" % topic_name)
                rclpy.shutdown()
                break
            else:
                time.sleep(0.1)

    def get_res(self ,data):
        self.get_logger().debug("translate image")
        #self.timer.cancel()
        #self.timer = threading.Timer(10.0, self.timer_callback)
        #self.timer.start()
        rgb = bridge.imgmsg_to_cv2(data, "rgb8")
        img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if not Debug :
            results = pytesseract.image_to_string(img)
            self.get_logger().debug(results)
            msg = String()
            msg.data = str(results)
            self.publisher.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
            return True
        else :
            results = pytesseract.image_to_data(img, output_type=Output.DICT)
            for i in range(0, len(results["text"])):
                # extract the bounding box coordinates of the text region from the current result
                tmp_tl_x = results["left"][i]
                tmp_tl_y = results["top"][i]
                tmp_br_x = tmp_tl_x + results["width"][i]
                tmp_br_y = tmp_tl_y + results["height"][i]
                tmp_level = results["level"][i]
                conf = results["conf"][i]
                text = results["text"][i]
                if ((tmp_level == 5) and (len(text)>0)):
                    time.sleep(1)
                    msg = String()
                    msg.data = str(str(text))
                    self.publisher.publish(msg)
                    self.get_logger().info('Publishing: "%s"' % msg.data)

class OcrService(Node):

    def __init__(self):
        super().__init__('ocr_service')
        self.srv = self.create_service(OcrRequest, 'OcrRequest', self.ocr_process)

    def ocr_process(self, request, response):
        if len(request.image_node.strip()) == 0 :
            reponse.success = False
            reponse.ocr_node = " "
            self.get_logger().info('Incoming request image Node %s is invaild' % (request.image_node))
            return response
        response.ocr_node = request.image_node + "_response"
        node_name = request.image_node + "_ocr"
        image_node  = request.image_node

        new_thread = threading.Thread(target=self.ocr_process_node, args=(node_name, image_node))
        new_thread.start()
        response.success = True
        self.get_logger().info('Incoming request image Node %s' % (request.image_node))
        return response

    def ocr_process_node(self,node_name,image_node):
        #rclpy.init()
        node = OcrProcess(node_name,image_node)
        executor_node = MultiThreadedExecutor()
        rclpy.spin(node, executor=executor_node)
        #rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)

    ocr_service = OcrService()
    executor = MultiThreadedExecutor()
    rclpy.spin(ocr_service, executor=executor)
    ocr_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
