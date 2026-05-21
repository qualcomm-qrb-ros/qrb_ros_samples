// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "follow_me/state_machine.hpp"

namespace follow_me
{

StateMachine::StateMachine() : state_(SystemState::IDLE) {}

bool StateMachine::start()
{
  // Can only start when in IDLE state
  if (state_ == SystemState::IDLE) {
    state_ = SystemState::RUNNING;
    return true;
  }
  return false;
}

bool StateMachine::pause()
{
  // Can only pause when in RUNNING state
  if (state_ == SystemState::RUNNING) {
    state_ = SystemState::PAUSED;
    return true;
  }
  return false;
}

bool StateMachine::resume()
{
  // Resume from PAUSED state returns to RUNNING state
  if (state_ == SystemState::PAUSED) {
    state_ = SystemState::RUNNING;
    return true;
  }
  return false;
}

bool StateMachine::finish()
{
  // Any state can transition to IDLE
  state_ = SystemState::IDLE;
  return true;
}

std::string StateMachine::getStateString() const
{
  switch (state_) {
    case SystemState::IDLE:
      return "IDLE (Not started)";
    case SystemState::RUNNING:
      return "RUNNING";
    case SystemState::PAUSED:
      return "PAUSED";
    default:
      return "UNKNOWN";
  }
}

bool StateMachine::isValidTransition(SystemState from, SystemState to) const
{
  // Define valid state transitions
  switch (from) {
    case SystemState::IDLE:
      return to == SystemState::RUNNING;  // start

    case SystemState::RUNNING:
      return to == SystemState::PAUSED ||  // pause
             to == SystemState::IDLE;      // finish

    case SystemState::PAUSED:
      return to == SystemState::RUNNING ||  // resume
             to == SystemState::IDLE;       // finish

    default:
      return false;
  }
}

}  // namespace follow_me
