// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef FOLLOW_ME__STATE_MACHINE_HPP_
#define FOLLOW_ME__STATE_MACHINE_HPP_

#include <functional>
#include <string>

namespace follow_me
{

/**
 * @brief System state enumeration
 */
enum class SystemState
{
  IDLE,     // Idle state (initial state)
  RUNNING,  // Running state
  PAUSED    // Paused state
};

/**
 * @brief State machine class
 */
class StateMachine
{
public:
  StateMachine();
  ~StateMachine() = default;

  /**
   * @brief Start state machine (IDLE -> RUNNING)
   * @return Whether succeeded
   */
  bool start();

  /**
   * @brief Pause state machine (RUNNING -> PAUSED)
   * @return Whether succeeded
   */
  bool pause();

  /**
   * @brief Resume state machine (PAUSED -> RUNNING)
   * @return Whether succeeded
   */
  bool resume();

  /**
   * @brief Finish state machine (any state -> IDLE)
   * @return Whether succeeded
   */
  bool finish();

  /**
   * @brief Get current state
   */
  SystemState getState() const { return state_; }

  /**
   * @brief Whether running
   */
  bool isRunning() const { return state_ == SystemState::RUNNING; }

  /**
   * @brief Whether paused
   */
  bool isPaused() const { return state_ == SystemState::PAUSED; }

  /**
   * @brief Whether idle (not started)
   */
  bool isIdle() const { return state_ == SystemState::IDLE; }

  /**
   * @brief Get state string
   */
  std::string getStateString() const;

private:
  SystemState state_;  // Current state

  /**
   * @brief Validate whether state transition is valid
   */
  bool isValidTransition(SystemState from, SystemState to) const;
};

}  // namespace follow_me

#endif  // FOLLOW_ME__STATE_MACHINE_HPP_
