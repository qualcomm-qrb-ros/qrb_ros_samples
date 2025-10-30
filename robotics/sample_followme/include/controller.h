/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#ifndef __CONTROLLER_H_
#define __CONTROLLER_H_

#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <unistd.h>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <iostream>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"

using namespace std;

class Controller
{
public:
  Controller();
  Controller(float theta, float d);
  ~Controller();
  Controller(const Controller &) = delete;

  void controlOnce(float theta, float d);
  void stop(void);
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr speed_pub;

private:
  void initParams(void);
  void PublishSpeed(float angle_speed, float linear_speed);
  void update(float theta, float d, float (&adjust_speed)[2]);

  float ref_theta;
  float ref_d;
  float kp[2], ki[2], kd[2];  // pid params, angle and distance
  float integrator[2];        // integral of the errors
  float last_error[2];        // for error derivative
  double last_call;

  float linear_max_speed;
  float angular_max_speed;
  float last_linear_speed;
  geometry_msgs::msg::Twist msg;
};
#endif
