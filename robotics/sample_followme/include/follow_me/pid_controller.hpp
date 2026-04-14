// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef FOLLOW_ME__PID_CONTROLLER_HPP_
#define FOLLOW_ME__PID_CONTROLLER_HPP_

#include <chrono>

namespace follow_me
{

/**
 * @brief PID controller class
 */
class PIDController
{
public:
  /**
   * @brief Constructor
   * @param kp Proportional gain
   * @param ki Integral gain
   * @param kd Derivative gain
   * @param max_output Maximum output value
   * @param min_output Minimum output value
   */
  PIDController(double kp = 1.0,
      double ki = 0.0,
      double kd = 0.0,
      double max_output = 1.0,
      double min_output = -1.0);

  ~PIDController() = default;

  /**
   * @brief Compute control output
   * @param error Current error
   * @param dt Time interval (seconds)
   * @return Control output
   */
  double compute(double error, double dt);

  /**
   * @brief Reset PID state
   */
  void reset();

  /**
   * @brief Set PID parameters
   */
  void setGains(double kp, double ki, double kd);

  /**
   * @brief Set output limits
   */
  void setOutputLimits(double min_output, double max_output);

private:
  double kp_;  // Proportional gain
  double ki_;  // Integral gain
  double kd_;  // Derivative gain

  double max_output_;  // Maximum output
  double min_output_;  // Minimum output

  double integral_;    // Integral term
  double prev_error_;  // Previous error
  bool first_run_;     // Whether first run
};

/**
 * @brief Dual PID controller - for linear and angular velocity control
 */
class DualPIDController
{
public:
  /**
   * @brief Constructor
   * @param linear_kp Linear velocity proportional gain
   * @param linear_ki Linear velocity integral gain
   * @param linear_kd Linear velocity derivative gain
   * @param angular_kp Angular velocity proportional gain
   * @param angular_ki Angular velocity integral gain
   * @param angular_kd Angular velocity derivative gain
   * @param max_linear_speed Maximum linear speed
   * @param max_angular_speed Maximum angular speed
   */
  DualPIDController(double linear_kp = 0.5,
      double linear_ki = 0.0,
      double linear_kd = 0.1,
      double angular_kp = 2.0,
      double angular_ki = 0.0,
      double angular_kd = 0.2,
      double max_linear_speed = 0.5,
      double max_angular_speed = 0.5);

  ~DualPIDController() = default;

  /**
   * @brief Compute control commands
   * @param distance_error Distance error (meters)
   * @param angle_error Angle error (radians)
   * @param dt Time interval (seconds)
   * @param linear_vel Output linear velocity
   * @param angular_vel Output angular velocity
   */
  void compute(double distance_error,
      double angle_error,
      double dt,
      double & linear_vel,
      double & angular_vel);

  /**
   * @brief Reset controller
   */
  void reset();

  /**
   * @brief Set linear velocity PID parameters
   */
  void setLinearGains(double kp, double ki, double kd);

  /**
   * @brief Set angular velocity PID parameters
   */
  void setAngularGains(double kp, double ki, double kd);

private:
  PIDController linear_pid_;   // Linear velocity PID
  PIDController angular_pid_;  // Angular velocity PID
};

}  // namespace follow_me

#endif  // FOLLOW_ME__PID_CONTROLLER_HPP_
