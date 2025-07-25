# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from qrb_ros_slam_msgs.srv import SlamCommand
import rclpy
import math


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class BuildMapNode(Node):
    def __init__(self):
        super().__init__('build_map_node')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.slam_cli = self.create_client(SlamCommand, '/qrb_slam_command')
        self.timer = self.create_timer(0.2, self.control_loop)

        self.state = 'START'
        self.start_position = None
        self.start_yaw = None
        self.distance_moved = 0.0
        self.yaw_rotated = 0.0
        self.linear_speed = 0.8
        self.angular_speed = math.radians(30)

        # Each step: ('FORWARD', meters) or ('TURN', degrees)
        self.steps = [
            ('FORWARD', 4.0),
            ('TURN', -90),
            ('FORWARD', 7.0),
            ('TURN', -90),
            ('TURN', -90),
            ('FORWARD', 7.0),
            ('TURN', 90),
            ('FORWARD', 3.3),
            ('TURN', -90),
            ('FORWARD', 1.5),
        ]
        self.current_step_idx = -1

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
        else:
            msg_str = "No response"
        self.get_logger().info(f"Command {command_id} service response: {msg_str}")

        if command_id == 0 and success:
            self.next_step()
        elif command_id == 1 and success:
            self.send_slam_command(2)
        elif command_id == 2 and success:
            self.get_logger().info('Map saved successfully, shutting down the script.')
            rclpy.shutdown()
        elif not success:
            self.get_logger().error(f'Command {command_id} execution failed!')
            rclpy.shutdown()

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        if getattr(self, 'motion_state', None) == 'FORWARD':
            if self.start_position is None:
                self.start_position = pos
            dx = pos.x - self.start_position.x
            dy = pos.y - self.start_position.y
            self.distance_moved = math.sqrt(dx*dx + dy*dy)
        elif getattr(self, 'motion_state', None) == 'TURN':
            if self.start_yaw is None:
                self.start_yaw = current_yaw
            delta_yaw = normalize_angle(current_yaw - self.start_yaw)
            self.yaw_rotated = abs(delta_yaw)

    def next_step(self):
        self.current_step_idx += 1
        if self.current_step_idx >= len(self.steps):
            self.state = 'END_MAP'
            self.send_slam_command(1)
            return
        action, param = self.steps[self.current_step_idx]
        self.motion_state = action
        if action == 'FORWARD':
            self.start_position = None
            self.distance_moved = 0.0
            self.target_distance = param
            self.get_logger().info(f'Entering FORWARD, target: {param} m')
        elif action == 'TURN':
            self.start_yaw = None
            self.yaw_rotated = 0.0
            self.target_turn_angle = math.radians(abs(param))
            self.turn_direction = 1 if param > 0 else -1
            self.get_logger().info(f'Entering TURN {param:+d}°')

    def control_loop(self):
        if self.state == 'START':
            self.get_logger().info('Preparing to start mapping...')
            self.send_slam_command(0)
            self.state = 'WAIT_START'
        elif hasattr(self, 'motion_state'):
            if self.motion_state == 'FORWARD':
                if self.distance_moved < self.target_distance:
                    twist = Twist()
                    twist.linear.x = self.linear_speed
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info(f'FORWARD: {self.distance_moved:.2f}/{self.target_distance} m')
                else:
                    self.cmd_vel_pub.publish(Twist())
                    self.get_logger().info('FORWARD completed, moving to next action')
                    self.next_step()
            elif self.motion_state == 'TURN':
                if self.yaw_rotated < self.target_turn_angle:
                    twist = Twist()
                    twist.angular.z = self.turn_direction * self.angular_speed
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info(f'TURN: {math.degrees(self.yaw_rotated):.1f}/{math.degrees(self.target_turn_angle):.1f}°')
                else:
                    self.cmd_vel_pub.publish(Twist())
                    self.get_logger().info('TURN completed, moving to next action')
                    self.next_step()

def main(args=None):
    rclpy.init(args=args)
    node = BuildMapNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()