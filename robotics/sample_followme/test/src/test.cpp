/*****************************************************************************
@copyright
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause-Clear
*******************************************************************************/
#include "followme_apicheck.hpp"
#include "gtest/gtest.h"

class NodeTestSuite : public ::testing::Test {
protected:
  void SetUp() override { rclcpp::init(0, nullptr); }

  void TearDown() override { rclcpp::shutdown(); }
};

TEST_F(NodeTestSuite, RosMessageTest1) {
  rclcpp::Node::SharedPtr test_node =
      std::make_shared<rclcpp::Node>("test_node");

  uint16_t h = 0;
  auto pub = test_node->create_publisher<SPEED_TYPE>(SPEED_NAME, 10);
  auto sub = test_node->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      [&h](const geometry_msgs::msg::Twist::SharedPtr msg) { h = 1U; });

  EXPECT_EQ(pub->get_subscription_count(), 1U);
  EXPECT_EQ(sub->get_publisher_count(), 1U);

  auto message = geometry_msgs::msg::Twist();

  pub->publish(message);
  std::this_thread::sleep_for(std::chrono::seconds(1));

  rclcpp::spin_some(test_node);

  EXPECT_EQ(h, 1U);

  pub.reset();
  sub.reset();
  test_node.reset();
}
