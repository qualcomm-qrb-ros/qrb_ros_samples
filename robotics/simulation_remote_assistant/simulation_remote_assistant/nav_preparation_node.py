# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from rclpy.node import Node
from qrb_ros_slam_msgs.srv import SlamCommand
from geometry_msgs.msg import Twist
import rclpy
import subprocess
import signal


class NavPreparationNode(Node):
    def __init__(self):
        super().__init__('nav_preparation_node')
        self.slam_cli = self.create_client(SlamCommand, '/qrb_slam_command')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state = 'LOAD_MAP'
        self.timer = self.create_timer(1.0, self.control_loop)
        self.relocalize_spin_speed = 0.3
        self.nav2_process = None

    def send_slam_command(self, command_id):
        if not self.slam_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/qrb_slam_command service unavailable')
            return
        req = SlamCommand.Request()
        req.command_id = command_id
        future = self.slam_cli.call_async(req)
        future.add_done_callback(lambda fut: self.handle_slam_response(fut, command_id))

    def handle_slam_response(self, future, command_id):
        res = future.result()
        msg_str = ""
        success = False
        if res is not None and hasattr(res, 'response'):
            msg_str = res.response.message
            success = getattr(res.response, "result", False) and getattr(res.response, "code", -1) == 0

        log_prefix = {
            3: "Load map response",
            4: "Start localization response",
            5: "Validate relocalization response"
        }.get(command_id, f"Command {command_id} service response")
        self.get_logger().info(f"{log_prefix}: {msg_str}")

        if command_id == 3 and success:
            self.state = 'START_RELOCALIZE'
        elif command_id == 4 and success:
            self.state = 'WAIT_RELOCALIZE'
        elif command_id == 5:
            if success and msg_str and 'Relocalization Successful' in msg_str:
                self.get_logger().info('Relocalization successful, starting navigation')
                self.cmd_vel_pub.publish(Twist())
                self.start_nav2()
                self.state = 'DONE'
        elif not success:
            self.get_logger().error(f'Command {command_id} execution failed!')

    def start_nav2(self):
        if self.nav2_process is None or self.nav2_process.poll() is not None:
            self.get_logger().info('Starting nav2 process...')
            self.nav2_process = subprocess.Popen([
                'ros2', 'launch', 'nav2_bringup', 'navigation_launch.py', 'use_sim_time:=true'
            ])
        else:
            self.get_logger().warn('nav2 process already running.')

    def kill_nav2(self):
        if self.nav2_process and self.nav2_process.poll() is None:
            self.get_logger().info('Killing nav2 process...')
            self.nav2_process.terminate()
            try:
                self.nav2_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.get_logger().warn('Force killing nav2 process...')
                self.nav2_process.kill()

    def control_loop(self):
        if self.state == 'LOAD_MAP':
            self.get_logger().info('Loading map...')
            self.send_slam_command(3)
            self.state = 'WAIT_LOADMAP'
        elif self.state == 'START_RELOCALIZE':
            self.get_logger().info('Starting relocalization...')
            self.send_slam_command(4)
        elif self.state == 'WAIT_RELOCALIZE':
            twist = Twist()
            twist.angular.z = self.relocalize_spin_speed
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info('Spinning in place during relocalization and checking...')
            self.send_slam_command(5)

def main(args=None):
    rclpy.init(args=args)
    node = NavPreparationNode()

    def shutdown_handler(signum=None, frame=None):
        node.get_logger().info('Shutting down, cleaning up nav2 process...')
        node.kill_nav2()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(node)
    finally:
        node.kill_nav2()
        node.get_logger().info('Exited cleanly.')

if __name__ == '__main__':
    main()