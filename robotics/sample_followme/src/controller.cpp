/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include <functional>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <thread>

#include "common.h"
#include "controller.h"

using namespace std;

Controller::Controller() : ref_theta(0), ref_d(DEFAULT_TARGET_DISTANCE) {
  initParams();
}

Controller::Controller(float theta, float d) : ref_theta(theta), ref_d(d) {
  initParams();
}

void Controller::initParams(void) {
  kp[0] = 2.5;
  kp[1] = 0.8;
  ki[0] = 0;
  ki[1] = 0;
  kd[0] = 0;
  kd[1] = 0;

  integrator[0] = 0;
  integrator[1] = 0;
  last_error[0] = 0;
  last_error[1] = 0;
  last_call = rclcpp::Clock().now().seconds(); // initial time

  linear_max_speed = 0.8;
  angular_max_speed = 1.0;
}

Controller::~Controller() {}

void Controller::controlOnce(float theta, float d) {
  float adjust_speed[2];
  update(theta, d, adjust_speed);

  if (abs(adjust_speed[0]) > angular_max_speed) {
    if (adjust_speed[0] > 0)
      adjust_speed[0] = angular_max_speed;
    else
      adjust_speed[0] = -angular_max_speed;
  }

  if (abs(adjust_speed[1]) > linear_max_speed) {
    if (adjust_speed[1] > 0)
      adjust_speed[1] = linear_max_speed;
    else
      adjust_speed[1] = -linear_max_speed;
  }

  if (d < 0.6) {
    printf("d = %.3f, the distance is less than 0.6 meter!!!\n", d);
  }

  else {
    PublishSpeed(adjust_speed[0], adjust_speed[1]);
  }
}

void Controller::update(float theta, float d, float (&adjusted_speed)[2]) {
  float error_angle = theta - ref_theta;
  float error_d = d - ref_d;

  if (abs(error_angle) < 0.1)
    error_angle = 0;
  if (abs(error_d) < 0.2)
    error_d = 0;

  /* amplify distance for backward */
  if (error_d < 0 && DEFAULT_TARGET_DISTANCE < AMPLIFY_SCALE)
    error_d = error_d * AMPLIFY_SCALE / DEFAULT_TARGET_DISTANCE;
  double curTime = rclcpp::Clock().now().seconds();
  double deltaT = curTime - last_call;
  /* PID for angle */
  integrator[0] += error_angle * deltaT;
  if (integrator[0] > 0.8)
    integrator[0] = 0.8;

  float P, I, D;
  P = error_angle;
  I = integrator[0];
  D = (error_angle - last_error[0]) / deltaT;
  adjusted_speed[0] = kp[0] * P + ki[0] * I + kd[0] * D;

  /* PID for distance */
  integrator[1] += error_d * deltaT;
  if (integrator[1] > 5)
    integrator[1] = 5;

  P = error_d;
  I = integrator[1];
  D = (error_d - last_error[1]) / deltaT;
  adjusted_speed[1] = kp[1] * P + ki[1] * I + kd[1] * D;

  last_call = curTime;
}

void Controller::PublishSpeed(float angle_speed, float linear_speed) {
  if (abs(linear_speed) < 0.001 && abs(angle_speed) > 0) {
    if (last_linear_speed > 0)
      linear_speed = 0.2;
    else if (last_linear_speed < 0)
      linear_speed = -0.2;
  }
  if (std::isnan(linear_speed) || std::isnan(angle_speed))
    return;

  msg.linear.x = linear_speed;
  msg.angular.z = angle_speed;

  // printf("linear speed: %.3f, angle speed: %.3f\n", linear_speed,
  // angle_speed);
  speed_pub->publish(msg);
  last_linear_speed = linear_speed;
}

void Controller::stop(void) {
  msg.linear.x = 0;
  msg.angular.z = 0;

  speed_pub->publish(msg);
}
