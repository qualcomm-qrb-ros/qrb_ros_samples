# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory
import yaml
import sys
import threading
import os
import rclpy
import signal


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')
        self.result_event = threading.Event()
        self.target_class = None
        self.nav_destination = None
        self.nav_action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.detection_enabled = False
        self.yolo_sub = self.create_subscription(
            Detection2DArray, '/yolo_detect_result', self.yolo_callback, 10
        )

    def goto_location_action(self, yaml_file, location_name, done_cb):
        pkg_share = get_package_share_directory('simulation_remote_assistant')
        yaml_path = os.path.join(pkg_share, 'config', yaml_file)
        with open(yaml_path, 'r') as f:
            locations = yaml.safe_load(f)
        if location_name not in locations:
            self.get_logger().error(f"Location '{location_name}' not found in {yaml_file}")
            sys.exit(1)
        p = locations[location_name]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = p["position"]["x"]
        goal.pose.pose.position.y = p["position"]["y"]
        goal.pose.pose.position.z = p["position"]["z"]
        goal.pose.pose.orientation.x = p["orientation"]["x"]
        goal.pose.pose.orientation.y = p["orientation"]["y"]
        goal.pose.pose.orientation.z = p["orientation"]["z"]
        goal.pose.pose.orientation.w = p["orientation"]["w"]
        self.get_logger().info(f"Navigating to {location_name}: ({goal.pose.pose.position.x}, {goal.pose.pose.position.y})")

        self.nav_action_client.wait_for_server()
        send_goal_future = self.nav_action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(
            lambda fut: self._goal_response_callback(fut, done_cb)
        )

    def _goal_response_callback(self, future, done_cb):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal was rejected')
            done_cb(False)
            return
        self.get_logger().info('Navigation goal accepted, waiting to arrive...')

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(
            lambda fut: self._nav_result_callback(fut, done_cb)
        )

    def _nav_result_callback(self, future, done_cb):
        result = future.result().result
        nav_code = getattr(result, 'error_code', None)
        success = nav_code == 0
        if success:
            self.get_logger().info('Navigation target reached, starting detection')
            self.detection_enabled = True
        else:
            self.get_logger().error(f'Navigation failed, code={nav_code}')
        done_cb(success)

    def yolo_callback(self, msg):
        if not self.detection_enabled:
            return
        for det in msg.detections:
            for res in det.results:
                class_id = getattr(res.hypothesis, 'class_id', '')
                if class_id == self.target_class:
                    print(f'Found {class_id} at {self.nav_destination}.')
                    self.result_event.set()
                    self.detection_enabled = False
                    return

def main(args=None):

    def sigint_handler(signum, frame):
        print("\nReceived kill signal, input enter to quit...")
        rclpy.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)

    rclpy.init(args=args)
    node = TaskManagerNode()
    pkg_share = get_package_share_directory('simulation_remote_assistant')
    locations_path = os.path.join(pkg_share, 'config', 'locations.yaml')
    objects_path = os.path.join(pkg_share, 'config', 'objects.yaml')

    if not os.path.exists(locations_path):
        print(f"Locations file not found: {locations_path}")
        sys.exit(1)
    if not os.path.exists(objects_path):
        print(f"Objects file not found: {objects_path}")
        sys.exit(1)

    with open(locations_path, 'r') as f:
        locations = yaml.safe_load(f)
    valid_locations = list(locations.keys())

    with open(objects_path, 'r') as f:
        objects_data = yaml.safe_load(f)
    if 'objects' not in objects_data or not isinstance(objects_data['objects'], list):
        print(f"objects.yaml must contain a list under key 'objects'")
        sys.exit(1)
    valid_objects = objects_data['objects']

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    while rclpy.ok():
        print("\nWhat can I help you with?")
        print(f"Supported locations: {', '.join(valid_locations)}")
        print(f"Supported objects: {', '.join(valid_objects)}")
        print('Example: Go to office to check person\n')
        sys.stdout.flush()
        user_cmd = sys.stdin.readline()
        if not user_cmd:
            break
        user_cmd = user_cmd.strip().lower()
        location = next((loc for loc in valid_locations if loc in user_cmd), None)
        obj = next((o for o in valid_objects if o in user_cmd), None)

        if not location or not obj:
            print("Command must specify both a location and an object to run.\n"
                  f"Available locations: {', '.join(valid_locations)}\n"
                  f"Available objects: {', '.join(valid_objects)}")
            continue

        node.target_class = obj
        node.nav_destination = location

        nav_done_event = threading.Event()
        nav_result_holder = {}

        def nav_done_cb(success):
            nav_result_holder['success'] = success
            nav_done_event.set()

        node.result_event.clear()
        node.goto_location_action('locations.yaml', location, nav_done_cb)
        print(f"Navigation command sent to '{location}', waiting to arrive at destination...")

        nav_done = nav_done_event.wait(timeout=180)
        if not nav_done or not nav_result_holder.get('success', False):
            print("Navigation timed out or failed")
            continue

        print(f"Arrived at '{location}', waiting for detection of '{obj}'...")
        detected = node.result_event.wait(timeout=60.0)
        if not detected:
            print(f"Target '{obj}' not detected within 60 seconds!")

        print("Task finished. You can input new command or Ctrl+C to quit.\n")

    rclpy.shutdown()
    spin_thread.join(timeout=2)

if __name__ == '__main__':
    main()