'''
Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import sys
from ocr_msg.srv import OcrRequest
import rclpy
from rclpy.node import Node


class OcrClientAsync(Node):

    def __init__(self):
        super().__init__('Ocr_client_async')
        self.cli = self.create_client(OcrRequest, 'OcrRequest')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = OcrRequest.Request()

    def send_request(self):
        self.req.image_node = str(sys.argv[1])
        self.future = self.cli.call_async(self.req)



def main(args=None):
    rclpy.init(args=args)

    ocr_client = OcrClientAsync()
    ocr_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(ocr_client)
        if ocr_client.future.done():
            try:
                response = ocr_client.future.result()
            except Exception as e:
                ocr_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                ocr_client.get_logger().info(
                    'Result of ocr_request: for %s  %s reponse  %s' %
                    (ocr_client.req.image_node, response.success, response.ocr_node))
            break

    ocr_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
