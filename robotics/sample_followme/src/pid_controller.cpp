// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "follow_me/pid_controller.hpp"

#include <algorithm>

namespace follow_me
{

PIDController::PIDController(double kp, double ki, double kd, double max_output, double min_output)
  : kp_(kp)
  , ki_(ki)
  , kd_(kd)
  , max_output_(max_output)
  , min_output_(min_output)
  , integral_(0.0)
  , prev_error_(0.0)
  , first_run_(true)
{
}

double PIDController::compute(double error, double dt)
{
  // Proportional term
  double p_term = kp_ * error;

  // Integral term
  integral_ += error * dt;
  double i_term = ki_ * integral_;

  // Derivative term
  double d_term = 0.0;
  if (!first_run_) {
    double derivative = (error - prev_error_) / dt;
    d_term = kd_ * derivative;
  }
  first_run_ = false;
  prev_error_ = error;

  // Calculate output
  double output = p_term + i_term + d_term;

  // Limit output range
  output = std::max(min_output_, std::min(max_output_, output));

  return output;
}

void PIDController::reset()
{
  integral_ = 0.0;
  prev_error_ = 0.0;
  first_run_ = true;
}

void PIDController::setGains(double kp, double ki, double kd)
{
  kp_ = kp;
  ki_ = ki;
  kd_ = kd;
}

void PIDController::setOutputLimits(double min_output, double max_output)
{
  min_output_ = min_output;
  max_output_ = max_output;
}

// DualPIDController implementation

DualPIDController::DualPIDController(double linear_kp,
    double linear_ki,
    double linear_kd,
    double angular_kp,
    double angular_ki,
    double angular_kd,
    double max_linear_speed,
    double max_angular_speed)
  : linear_pid_(linear_kp, linear_ki, linear_kd, max_linear_speed, -max_linear_speed)
  , angular_pid_(angular_kp, angular_ki, angular_kd, max_angular_speed, -max_angular_speed)
{
}

void DualPIDController::compute(double distance_error,
    double angle_error,
    double dt,
    double & linear_vel,
    double & angular_vel)
{
  // Calculate linear velocity (based on distance error)
  linear_vel = linear_pid_.compute(distance_error, dt);

  // Calculate angular velocity (based on angle error)
  // In ROS, angular.z > 0 means turn left, < 0 means turn right
  // If target is on the right side (angle_error > 0), need to turn right (angular_vel < 0)
  angular_vel = angular_pid_.compute(-angle_error, dt);
}

void DualPIDController::reset()
{
  linear_pid_.reset();
  angular_pid_.reset();
}

void DualPIDController::setLinearGains(double kp, double ki, double kd)
{
  linear_pid_.setGains(kp, ki, kd);
}

void DualPIDController::setAngularGains(double kp, double ki, double kd)
{
  angular_pid_.setGains(kp, ki, kd);
}

}  // namespace follow_me
