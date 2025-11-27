# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


class AMRSimpleMotionControl(Node):
    """AMR simple motion controller with configurable speeds."""

    def __init__(self):
        """Initialize AMR simple motion controller with configurable speeds."""
        super().__init__('amr_simple_motion_control')
 
        # Configurable velocity parameters
        self.FORWARD_SPEED = 0.5    # m/s
        self.BACKWARD_SPEED = -0.5  # m/s
        self.CCW_SPEED = 1.0        # rad/s (counter-clockwise)
        self.CW_SPEED = -1.0        # rad/s (clockwise)
        self.STOP_SPEED = 0.0

        self.publisher_ = self.create_publisher(Twist, '/qrb_robot_base/cmd_vel', 10)

    def send_movement(self, linear, angular, command_type):
        """
        Publish movement command with given velocities.

        :param linear: Linear velocity (m/s)
        :param angular: Angular velocity (rad/s)
        :param command_type: Human-readable movement description
        """
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.publisher_.publish(twist)
        self.get_logger().info(f'{command_type} [Linear: {linear} m/s, Angular: {angular} rad/s]')


def print_usage():
    """Display command interface instructions."""
    print('\n=== AMR Simple Motion Control Usage: ===')
    print('i - Forward movement')
    print(', - Backward movement')
    print('j - Counter-clockwise rotation')
    print('l - Clockwise rotation')
    print('k - Stop')
    print('q - Quit program')
    print('=========================================\n')


def main(args=None):
    rclpy.init(args=args)
    controller = AMRSimpleMotionControl()
    print_usage()
 
    try:
        while rclpy.ok():
            try:
                cmd = input('Input command > ').strip().lower()
 
                if cmd == 'q':
                    controller.get_logger().info('Shutting down controller...')
                    break
 
                match cmd:
                    case 'i':
                        controller.send_movement(
                            controller.FORWARD_SPEED,
                            0.0,
                            'Moving forward'
                        )
                    case ',':
                        controller.send_movement(
                            controller.BACKWARD_SPEED,
                            0.0,
                            'Moving backward'
                        )
                    case 'j':
                        controller.send_movement(
                            0.0,
                            controller.CCW_SPEED,
                            'Rotating counter-clockwise'
                        )
                    case 'l':
                        controller.send_movement(
                            0.0,
                            controller.CW_SPEED,
                            'Rotating clockwise'
                        )
                    case 'k':
                        controller.send_movement(
                            controller.STOP_SPEED,
                            controller.STOP_SPEED,
                            'Stop'
                        )
                    case _:
                        print('Invalid command! Use help commands.')
 
                # Process ROS events
                rclpy.spin_once(controller, timeout_sec=0.01)
 
            except KeyboardInterrupt:
                print('\nExiting...')
                break
 
    finally:
        # Cleanup resources
        controller.destroy_node()
        rclpy.shutdown()
        print('Controller shutdown completed')


if __name__ == '__main__':
    main()
