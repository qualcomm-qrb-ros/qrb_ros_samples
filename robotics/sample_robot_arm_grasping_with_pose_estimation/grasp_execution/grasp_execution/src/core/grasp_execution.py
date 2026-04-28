#!/usr/bin/env python3
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# -*- coding=UTF-8 -*-
import sys
import os
import time
import argparse
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from grasp_perception_msgs.msg import PoseEstimationResult
    ROS2_POSE_MSG_AVAILABLE = True
except ImportError:
    ROS2_POSE_MSG_AVAILABLE = False

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)

try:
    # RM API: support both `python src/main.py` (src on path) and editable installs (Robotic_Arm package).
    _src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    try:
        from Robotic_Arm.rm_robot_interface import *  # noqa: F401,F403
    except ImportError:
        from src.Robotic_Arm.rm_robot_interface import *  # noqa: F401,F403
    print("RM API interface imported successfully")
except ImportError:
    raise RuntimeError("Failed to import RM API. This program supports real robot mode only")


def normalize_pose_translation_m(translation_vec):
    """Normalize translation from PoseEstimationResult in meters."""
    t_m = np.asarray(translation_vec, dtype=float)
    # Warn if value is unusually large for near-field grasping.
    if np.max(np.abs(t_m)) > 1.0:
        print(
            "[grasp_execution][Unit Note] translation_vector magnitude is large. "
            "This code interprets it as meters (m); if upstream uses millimeters (mm), "
            "please normalize upstream output to meters."
        )
    return t_m


def log_pose_xyzrpy(tag, pose):
    """Log pose in xyz offset and rpy Euler format."""
    rpy_deg = np.degrees(np.asarray(pose[3:6], dtype=float))
    print(
        f"[{tag}] xyz=[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}] m, "
        f"rpy=[{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}] deg"
    )


def pad_joint_degree_7(joint_deg):
    """Pad/truncate joint degree list to 7 elements."""
    vals = [float(v) for v in joint_deg]
    if len(vals) < 7:
        vals.extend([0.0] * (7 - len(vals)))
    return vals[:7]


def angle_diff(a, b):
    """Wrap angle difference into [-pi, pi]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


# Camera vs downstream EE pose convention: flip camera +X (orthogonal; det = -1).
# Baked into the hand-eye rotation as R_ce' = R_handeye @ R_align (see transform_camera_pose_to_base).
R_CAMERA_POSE_AXIS_ALIGN = np.diag([-1.0, 1.0, 1.0]).astype(float)


def transform_camera_pose_to_base(
    obj_camera_xyz,
    end_effector_pose_xyzrpy,
    camera_to_ee_rotation,
    camera_to_ee_translation,
    obj_rotation_matrix=None,
):
    """Compute object base-frame pose from camera-frame translation and rotation."""
    x, y, z = [float(v) for v in obj_camera_xyz]
    obj_camera_coordinates = np.array([x, y, z], dtype=float)

    # Effective camera->EE rotation: hand-eye R with axis convention baked in.
    camera_to_ee_rotation_effective = np.asarray(camera_to_ee_rotation, dtype=float).reshape(3, 3) @ R_CAMERA_POSE_AXIS_ALIGN

    x1, y1, z1, rx, ry, rz = [float(v) for v in end_effector_pose_xyzrpy]
    end_effector_pose = np.array([x1, y1, z1, rx, ry, rz], dtype=float)

    t_camera_to_end_effector = np.eye(4)
    t_camera_to_end_effector[:3, :3] = camera_to_ee_rotation_effective
    t_camera_to_end_effector[:3, 3] = camera_to_ee_translation

    position = end_effector_pose[:3]
    orientation = R.from_euler("xyz", end_effector_pose[3:], degrees=False).as_matrix()
    t_base_to_end_effector = np.eye(4)
    t_base_to_end_effector[:3, :3] = orientation
    t_base_to_end_effector[:3, 3] = position

    obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1.0])
    obj_end_effector_coordinates_homo = t_camera_to_end_effector.dot(obj_camera_coordinates_homo)
    obj_base_coordinates_homo = t_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
    obj_base_coordinates = obj_base_coordinates_homo[:3]

    if obj_rotation_matrix is None:
        obj_rotation_matrix = np.eye(3, dtype=float)
    else:
        obj_rotation_matrix = np.asarray(obj_rotation_matrix, dtype=float).reshape(3, 3)

    obj_orientation_matrix = (
        t_base_to_end_effector[:3, :3]
        .dot(camera_to_ee_rotation_effective)
        .dot(obj_rotation_matrix)
    )
    # Project to the nearest proper rotation to avoid left-handed or degenerate matrices.
    u, _, vh = np.linalg.svd(obj_orientation_matrix)
    obj_orientation_matrix = u @ vh
    if np.linalg.det(obj_orientation_matrix) <= 0.0:
        u[:, -1] *= -1.0
        obj_orientation_matrix = u @ vh
    obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler("xyz", degrees=False)

    return obj_base_coordinates, obj_orientation_matrix, obj_orientation_euler


class VisionGraspingSystem:
    """
    Main class for vision-guided grasping.
    Implements all features defined in requirements.md.
    """
    
    def __init__(
        self,
        ip="192.168.1.18",
        port=8080,
        pose_topic="/pose_estimation_result",
    ):
        """
        Initialize the vision grasping system.
        
        Args:
            ip: Robot arm IP address.
            port: Robot arm port.
        """
        self.ip = ip
        self.port = port
        
        # Hand-eye calibration parameters (camera -> end-effector).
        # Camera pose axis alignment is composed in transform_camera_pose_to_base as R @ R_align.
        self.rotation_matrix = np.array([[0.01206237, 0.99929647, 0.03551135],
                                        [-0.99988374, 0.01172294, 0.00975125],
                                        [0.00932809, -0.03562485, 0.9993217]])
        self.translation_vector = np.array([-0.14039019, -0.05225555, -0.12256825])

        # Axis mapping: PoseEstimationResult -> end-effector frame.
        self.pose_result_to_ee_axes = np.array([
            [0.0, 1.0, 0.0],    # x_ee = y_pose
            [-1.0, 0.0, 0.0],   # y_ee = -x_pose
            [0.0, 0.0, 1.0],   # z_ee = z_pose
        ], dtype=float)
        
        # State machine variables.
        self.current_state = "INIT"  # Current state
        self.retry_count = 0
        self.max_retries = 3
        self.object_detected = False
        self.object_pose_in_base = None  # Object pose in base frame
        self.detection_attempts = 0
        self.max_detection_attempts = 5
        self.state_failures = {}
        self.max_state_retries = 2
        self.state_retry_limits = {
            "INIT": 2,
            "MOVE_TO_HOME": 2,
            "MOVE_TO_DETECTION_POSE": 2,
            "CHECK_OBJECT": 2,
            "MOVE_TO_PRE_GRASP": 2,
            "OPEN_GRIPPER": 2,
            "MOVE_TO_GRASP": 2,
            "CLOSE_GRIPPER": 2,
            "CHECK_GRASP_SUCCESS": 1,
            "PLACE_OBJECT": 2,
            "RETURN_TO_DETECTION_POSE": 2,
            "RETURN_TO_HOME": 2,
        }
        self.last_error = ""
        self.safe_stop_reason = ""
        self.recovery_target_state = "CHECK_OBJECT"

        # ROS2 PoseEstimationResult subscription state.
        self.pose_topic = str(pose_topic)
        self.ros_node = None
        self.pose_subscriber = None
        self.pose_queue_maxlen = 1
        self.pose_msg_queue = deque()
        self.latest_pose_msg = None
        self.latest_pose_msg_time = 0.0
        self.use_topic_pose = False
        self._active_test_frame_id = None
        self._verification_records = []
        
        # Initialize real arm + topic subscription mode.
        self._init_robot_arm()
        self._init_pose_subscriber()
        
        # Home and detection poses.
        self.home_joint_angles = [2.272, 30.837, -115.824, -2.83, -15.69, 4.866]  # home mode
        # self.detection_joint_angles = [2.272, 30.837, -115.824, -2.83, -15.69, 4.866]  # detect mode
        self.detection_joint_angles = [6.817, 4.14, -112.66, 24.456, 23.72, -19.90] # detect mode
        
        print("Vision grasping execution system initialized")

    def _init_pose_subscriber(self):
        """Initialize required PoseEstimationResult subscription."""
        if not ROS2_POSE_MSG_AVAILABLE:
            raise RuntimeError("[grasp_execution] Missing rclpy or grasp_perception_msgs.msg; cannot run")

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.ros_node = Node("grasp_execution_pose_listener")
            latest_only_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=self.pose_queue_maxlen,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self.pose_subscriber = self.ros_node.create_subscription(
                PoseEstimationResult,
                self.pose_topic,
                self._pose_result_callback,
                latest_only_qos,
            )
            self.use_topic_pose = True
            print(f"[grasp_execution] Subscribed to topic: {self.pose_topic}")
        except Exception as e:
            raise RuntimeError(f"[grasp_execution] Failed to initialize subscription: {e}")

    def _pose_result_callback(self, msg):
        """Store PoseEstimationResult and keep only the latest frame."""
        msg_time = time.time()
        self.latest_pose_msg = msg
        self.latest_pose_msg_time = msg_time

        # Keep only newest message to avoid stale backlog.
        self.pose_msg_queue.clear()
        self.pose_msg_queue.append((msg_time, msg))

    def _convert_pose_result_vector_to_ee(self, vec3):
        """Convert PoseEstimationResult vector to end-effector frame."""
        return self.pose_result_to_ee_axes.dot(np.asarray(vec3, dtype=float))

    def _convert_pose_result_rotation_to_ee(self, rotation_matrix):
        """Convert PoseEstimationResult rotation to end-effector frame."""
        c = self.pose_result_to_ee_axes
        return c.dot(rotation_matrix).dot(c.T)

    def get_pose_from_topic(
        self,
        timeout_sec=3.0,
        require_fresh=True,
        max_msg_age_sec=1.0,
        log_timeout=True,
    ):
        """
        Get one pose frame from PoseEstimationResult topic.
        Returns [rx, ry, rz, tx, ty, tz] in end-effector frame or None.
        """
        if not self.use_topic_pose or self.ros_node is None:
            return None

        start_time = time.time()
        # Accept only new messages after this call starts.
        request_time = start_time
        while time.time() - start_time < timeout_sec:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            latest_valid_msg = self.latest_pose_msg
            latest_valid_msg_time = self.latest_pose_msg_time
            if latest_valid_msg is not None:
                # Skip old/expired messages to avoid false positives.
                if require_fresh and latest_valid_msg_time < request_time:
                    continue
                if require_fresh and (time.time() - latest_valid_msg_time) > max_msg_age_sec:
                    continue
                if len(latest_valid_msg.rotation_matrix) != 9 or len(latest_valid_msg.translation_vector) != 3:
                    print("[grasp_execution] Invalid message field length; skipping this frame")
                    continue
                self.latest_pose_msg_time = latest_valid_msg_time
                # Object rotation and translation in camera frame.
                r_pose = np.array(latest_valid_msg.rotation_matrix, dtype=float).reshape(3, 3)
                t_pose_m = normalize_pose_translation_m(latest_valid_msg.translation_vector)

                rvec_camera = R.from_matrix(r_pose).as_rotvec()
                euler_camera = R.from_matrix(r_pose).as_euler('xyz', degrees=False)
                euler_camera_deg = np.degrees(euler_camera)
                print("[grasp_execution] Detection result received (object pose in camera frame)")
                print(f"  offset(xyz): [{t_pose_m[0]:.4f}, {t_pose_m[1]:.4f}, {t_pose_m[2]:.4f}] m")
                print(
                    f"  euler_xyz: [{euler_camera[0]:.4f}, "
                    f"{euler_camera[1]:.4f}, {euler_camera[2]:.4f}] rad"
                )
                print(
                    f"  euler_xyz: [{euler_camera_deg[0]:.2f}, "
                    f"{euler_camera_deg[1]:.2f}, {euler_camera_deg[2]:.2f}] deg"
                )
                print(f"  rvec_camera: [{rvec_camera[0]:.4f}, {rvec_camera[1]:.4f}, {rvec_camera[2]:.4f}] rad")
                return [rvec_camera[0], rvec_camera[1], rvec_camera[2], t_pose_m[0], t_pose_m[1], t_pose_m[2]]

        if log_timeout:
            print(f"[grasp_execution] Timed out waiting for fresh messages: {self.pose_topic}")
        return None

    def check_no_fresh_pose_stable(self, required_no_msg=3, per_check_timeout=1.0):
        """
        Debounce check for object disappearance.
        Requires `required_no_msg` consecutive misses to pass.
        Returns:
          - True: object considered removed (grasp success)
          - False: fresh message received (object still present)
          - None: topic mode disabled
        """
        if not self.use_topic_pose or self.ros_node is None:
            return None

        no_msg_count = 0
        for i in range(required_no_msg):
            pose = self.get_pose_from_topic(
                timeout_sec=per_check_timeout,
                require_fresh=True,
                max_msg_age_sec=1.0,
                log_timeout=False,
            )
            if pose is None:
                no_msg_count += 1
                print(
                    f"[Debounce Check] Attempt {i + 1}/{required_no_msg}: no fresh message"
                )
            else:
                print(
                    f"[Debounce Check] Attempt {i + 1}/{required_no_msg}: fresh message received, object still present"
                )
                return False

        return no_msg_count >= required_no_msg
    
    def _init_robot_arm(self):
        """Initialize real robot arm."""
        try:
            self.thread_mode = rm_thread_mode_e(2)  # 3-thread mode
            self.robot = RoboticArm(self.thread_mode)
            self.handle = self.robot.rm_create_robot_arm(self.ip, self.port, 3)
            
            if self.handle.id == -1:
                raise ConnectionError("Failed to connect to robot arm")
            else:
                print(f"Connected to robot arm: {self.handle.id}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize robot arm: {e}")

    def _plan_movej_p_joint_solution(self, pose):
        """
        Solve IK for target pose.
        Returns (ok, q_solve_deg, message).
        """
        current_joint = self.get_current_joint_angles()
        if current_joint is None:
            return False, None, "Failed to get current joint angles; IK planning failed"

        try:
            ik_params = rm_inverse_kinematics_params_t(
                q_in=pad_joint_degree_7(current_joint),
                q_pose=[float(v) for v in pose[:6]],
                flag=1,  # Euler angles
            )
            ik_ret, q_solve = self.robot.rm_algo_inverse_kinematics(ik_params)
        except Exception as e:
            return False, None, f"IK exception: {e}"

        if ik_ret != 0:
            return False, None, f"IK failed, ret={ik_ret}"

        return True, [float(v) for v in q_solve], "IK succeeded"

    def _check_movej_p_self_collision(self, pose):
        """
        Run pre-checks before movej_p.
        Returns (safe, reason).
        """

        ok, q_solve, msg = self._plan_movej_p_joint_solution(pose)
        if not ok:
            return False, msg
        # if q_solve[4] < -66.0 or q_solve[4] > 115.0:
        #     return False, "Joint 5 out of limit, range: [-66, 115]"
        # For 6-axis arms: >0 means joint i limit exceeded.
        if len(q_solve) >= 6:
            try:
                limit_ret = self.robot.rm_algo_ikine_check_joint_position_limit(
                    q_solve[:6]
                )
                if limit_ret not in (0, -1):
                    return False, f"Joint limit check failed, joint={limit_ret}"
            except Exception:
                # Ignore for non-6-axis arms or unsupported firmware.
                pass

        # Self-collision check: 0=no collision, non-zero=collision/limit.
        collision_ret = self.robot.rm_algo_safety_robot_self_collision_detection(
            pad_joint_degree_7(q_solve)
        )
        if collision_ret != 0:
            return False, f"Self-collision check failed, ret={collision_ret}"

        return True, "Self-collision check passed"
    
    def movej(self, joint_angles, v=15, r=0, connect=0, block=1):
        """
        Joint-space move.
        
        Args:
            joint_angles: Joint angles (radians).
            v: Speed percentage.
            r: Blend radius.
            connect: Trajectory connection flag.
            block: Blocking mode.
        """
        print(f"[MoveJ] Moving to joint angles: {joint_angles}")
        
        movej_result = self.robot.rm_movej(joint_angles, v, r, connect, block)
        if movej_result == 0:
            print("Joint motion succeeded")
            ok = True
        else:
            print(f"Joint motion failed, error code: {movej_result}")
            ok = False
        time.sleep(2)
        return ok
    
    def movej_p(self, pose, v=15, r=0, connect=0, block=1):
        """
        Cartesian move using joint interpolation.
        
        Args:
            pose: Pose [x, y, z, rx, ry, rz] in m/rad.
            v: Speed percentage.
            r: Blend radius.
            connect: Trajectory connection flag.
            block: Blocking mode.
        """
        print(f"[MoveJ_P] Moving to Cartesian pose: {pose[:3]}")
        log_pose_xyzrpy("Robot command (movej_p)", pose)
        
        safe, reason = self._check_movej_p_self_collision(pose)
        if not safe:
            print(f"[MoveJ_P] Motion blocked before execution: {reason}")
            return False

        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("Cartesian joint interpolation motion succeeded")
            ok = True
        else:
            print(f"Cartesian joint interpolation motion failed, error code: {movej_p_result}")
            ok = False
        time.sleep(2)
        return ok
    
    def movel(self, pose, v=15, r=0, connect=0, block=1):
        """
        Cartesian linear move.
        
        Args:
            pose: Pose [x, y, z, rx, ry, rz] in m/rad.
            v: Speed percentage.
            r: Blend radius.
            connect: Trajectory connection flag.
            block: Blocking mode.
        """
        print(f"[MoveL] Linear move to pose: {pose[:3]}")
        log_pose_xyzrpy("Robot command (movel)", pose)
        
        movel_result = self.robot.rm_movel(pose, v, r, connect, block)
        if movel_result == 0:
            print("Linear motion succeeded")
            ok = True
        else:
            print(f"Linear motion failed, error code: {movel_result}")
            ok = False
        time.sleep(2)
        return ok
    
    def set_gripper_pick_on(self, speed=500, force=200, block=True, timeout=30):
        """Force-controlled gripper close."""
        print("[Gripper] Executing force-controlled grasp")
        try:
            gripper_result = self.robot.rm_set_gripper_pick_on(speed, force, block, timeout)
            if gripper_result == 0:
                print("Gripper grasp succeeded")
                ok = True
            else:
                print(f"Gripper grasp failed, error code: {gripper_result}")
                ok = False
        except Exception as e:
            print(f"Gripper grasp exception: {e}")
            ok = False

        time.sleep(1)
        return ok
    
    def set_gripper_release(self, speed=500, block=True, timeout=30):
        """Open gripper."""
        print("[Gripper] Executing gripper release")
        try:
            gripper_result = self.robot.rm_set_gripper_release(speed, block, timeout)
            if gripper_result == 0:
                print("Gripper release succeeded")
                ok = True
            else:
                print(f"Gripper release failed, error code: {gripper_result}")
                ok = False
        except Exception as e:
            print(f"Gripper release exception: {e}")
            ok = False

        time.sleep(1)
        return ok
    
    def get_current_joint_angles(self):
        """Get current joint angles."""
        ret_code, joint_degrees = self.robot.rm_get_joint_degree()
        if ret_code == 0:
            return joint_degrees
        print(f"Failed to get joint angles, error code: {ret_code}")
        return None
    
    def get_current_cartesian_pose(self):
        """Get current Cartesian pose."""
        joint_angles = self.get_current_joint_angles()
        if joint_angles is None:
            return None
        pose = self.robot.rm_algo_forward_kinematics(joint_angles, flag=1)
        return pose
    
    # Convert object pose from camera frame to arm base frame.
    def convert(self, x, y, z, x1, y1, z1, rx, ry, rz, obj_rotation_matrix=None):
        """
        Convert camera-frame object pose to base-frame pose
        using hand-eye calibration and current end-effector pose.
        """
        obj_base_coordinates, obj_orientation_matrix, obj_orientation_euler = transform_camera_pose_to_base(
            obj_camera_xyz=(x, y, z),
            end_effector_pose_xyzrpy=(x1, y1, z1, rx, ry, rz),
            camera_to_ee_rotation=self.rotation_matrix,
            camera_to_ee_translation=self.translation_vector,
            obj_rotation_matrix=obj_rotation_matrix,
        )
        obj_orientation_euler_deg = np.degrees(obj_orientation_euler)
        print("[Rotation Debug] R_camera_obj:")
        debug_obj_rotation = np.eye(3, dtype=float) if obj_rotation_matrix is None else np.asarray(obj_rotation_matrix, dtype=float).reshape(3, 3)
        print(np.array2string(debug_obj_rotation, precision=6, suppress_small=True))
        print("[Rotation Debug] R_base_obj:")
        print(np.array2string(obj_orientation_matrix, precision=6, suppress_small=True))
        print(
            "[Rotation Debug] obj_orientation_euler(rad): "
            f"[{obj_orientation_euler[0]:.6f}, {obj_orientation_euler[1]:.6f}, {obj_orientation_euler[2]:.6f}]"
        )
        print(
            "[Rotation Debug] obj_orientation_euler(deg): "
            f"[{obj_orientation_euler_deg[0]:.3f}, {obj_orientation_euler_deg[1]:.3f}, {obj_orientation_euler_deg[2]:.3f}]"
        )
        # Compose final result.
        obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
        obj_base_pose[3:] = rx,ry,rz
        
        return obj_base_pose

    def convert_camera_to_base(self, obj_camera_pose, input_frame="camera"):
        """
        Convert object pose from camera frame to arm base frame.
        
        Args:
            obj_camera_pose: [rx, ry, rz, tx, ty, tz]
                rx, ry, rz: Rodrigues rotation vector (radians).
                tx, ty, tz: Translation vector (meters).
            input_frame: Input pose frame.
                - "camera": use hand-eye transform T_camera->ee
                - "end_effector": pose already in ee frame
        
        Returns:
            list: [x, y, z, rx, ry, rz] in arm base frame.
        """
        # Parse input in two compatible formats:
        # 1) [rx, ry, rz, tx, ty, tz] rotvec + translation
        # 2) [tx, ty, tz, qx, qy, qz, qw] translation + quaternion
        if len(obj_camera_pose) == 6:
            rvec_x, rvec_y, rvec_z = obj_camera_pose[0], obj_camera_pose[1], obj_camera_pose[2]
            tvec_x_m, tvec_y_m, tvec_z_m = obj_camera_pose[3], obj_camera_pose[4], obj_camera_pose[5]
            rvec_obj = np.array([rvec_x, rvec_y, rvec_z], dtype=float)
            R_in_o = R.from_rotvec(rvec_obj).as_matrix()
        elif len(obj_camera_pose) == 7:
            tvec_x_m, tvec_y_m, tvec_z_m = obj_camera_pose[0], obj_camera_pose[1], obj_camera_pose[2]
            qx, qy, qz, qw = obj_camera_pose[3], obj_camera_pose[4], obj_camera_pose[5], obj_camera_pose[6]
            R_in_o = R.from_quat([qx, qy, qz, qw]).as_matrix()
            rvec_obj = None
        else:
            raise ValueError(
                "Invalid obj_camera_pose format: expected 6D (rotvec+trans) or 7D (trans+quat)"
            )

        # Read current end-effector pose.
        end_effector_pose = self.get_current_cartesian_pose()
        if end_effector_pose is None:
            print("Failed to get robot arm pose")
            return None

        x1, y1, z1 = end_effector_pose[0], end_effector_pose[1], end_effector_pose[2]
        rx, ry, rz = end_effector_pose[3], end_effector_pose[4], end_effector_pose[5]

        # Pass camera-frame object rotation into convert() for base-frame orientation.
        obj_base_pose = self.convert(x=tvec_x_m, y=tvec_y_m, z=tvec_z_m, x1=x1, y1=y1, z1=z1, rx=rx, ry=ry, rz=rz, obj_rotation_matrix=R_in_o)
        return obj_base_pose.tolist()
    
    def get_object_detection_pose(self):
        """
        Get detection result only from PoseEstimationResult topic.
        Returns: (pose, frame)
          - pose: [rx, ry, rz, tx, ty, tz] or None
          - frame: "end_effector"
        """
        self.detection_attempts += 1
        topic_pose = self.get_pose_from_topic(timeout_sec=3.0)
        if topic_pose is not None:
            print(f"[Topic Detection] Attempt {self.detection_attempts}: received message from {self.pose_topic}")
            return topic_pose, "camera"
        print(f"[Topic Detection] Attempt {self.detection_attempts}: no message from {self.pose_topic}")
        return None, "camera"

    def get_current_pose_from_arm_state(self):
        """
        Get current arm pose from rm_get_current_arm_state.
        Falls back to forward kinematics if needed.
        Returns [x, y, z, rx, ry, rz] or None.
        """
        try:
            ret, state = self.robot.rm_get_current_arm_state()
            if ret == 0 and isinstance(state, dict) and "pose" in state:
                pose = state["pose"]
                if isinstance(pose, list) and len(pose) >= 6:
                    return [float(v) for v in pose[:6]]
        except Exception as e:
            print(f"[State Query] rm_get_current_arm_state exception: {e}")
        return self.get_current_cartesian_pose()

    
    def check_object(self):
        """
        Detect object.
        
        Returns:
            bool: True if object is detected.
        """
        print("\n" + "="*60)
        print("Starting object detection")
        print("="*60)
        
        object_pose, pose_frame = self.get_object_detection_pose()
        
        if object_pose is not None:
            print(f"[Object Detection] Object detected, pose: {object_pose}")
            
            # Convert to base frame.
            self.object_pose_in_base = self.convert_camera_to_base(
                object_pose,
                input_frame=pose_frame,
            )
            if self.object_pose_in_base is not None:
                base_rpy_deg = np.degrees(np.asarray(self.object_pose_in_base[3:6], dtype=float))
                print(f"[Coordinate Transform] Base-frame pose:")
                print(f"  Position: [{self.object_pose_in_base[0]:.4f}, {self.object_pose_in_base[1]:.4f}, {self.object_pose_in_base[2]:.4f}] m")
                print(f"  Orientation: [{self.object_pose_in_base[3]:.4f}, {self.object_pose_in_base[4]:.4f}, {self.object_pose_in_base[5]:.4f}] rad")
                print(f"  Orientation: [{base_rpy_deg[0]:.2f}, {base_rpy_deg[1]:.2f}, {base_rpy_deg[2]:.2f}] deg")
                self.object_detected = True
                return True
            else:
                print("[Coordinate Transform] Conversion failed")
                self.object_detected = False
                return False
        else:
            print("[Object Detection] No object detected")
            self.object_detected = False
            return False
    
    def print_current_frame(self):
        """Print current frame information."""
        print("\n" + "="*60)
        print("Printing current frame information")
        print("="*60)
        
        # Read current joint angles.
        joint_angles = self.get_current_joint_angles()
        if joint_angles is not None:
            print(f"Current joint angles: {joint_angles}")
        
        # Read current Cartesian pose.
        cartesian_pose = self.get_current_cartesian_pose()
        if cartesian_pose is not None:
            cart_rpy_deg = np.degrees(np.asarray(cartesian_pose[3:6], dtype=float))
            print(f"Current Cartesian pose:")
            print(f"  Position: [{cartesian_pose[0]:.4f}, {cartesian_pose[1]:.4f}, {cartesian_pose[2]:.4f}] m")
            print(f"  Orientation: [{cartesian_pose[3]:.4f}, {cartesian_pose[4]:.4f}, {cartesian_pose[5]:.4f}] rad")
            print(f"  Orientation: [{cart_rpy_deg[0]:.2f}, {cart_rpy_deg[1]:.2f}, {cart_rpy_deg[2]:.2f}] deg")

    def _mark_state_success(self, state_name):
        """Clear failure count after state succeeds."""
        if state_name in self.state_failures:
            del self.state_failures[state_name]

    def _enter_failure(self, state_name, reason, recover_state="CHECK_OBJECT", max_retries=None):
        """Unified failure handler with recovery and safe-stop fallback."""
        if max_retries is not None:
            limit = max_retries
        else:
            limit = self.state_retry_limits.get(state_name, self.max_state_retries)
        fail_count = self.state_failures.get(state_name, 0) + 1
        self.state_failures[state_name] = fail_count
        self.last_error = f"{state_name}: {reason}"
        self.recovery_target_state = recover_state
        print(
            f"[Failure Handler] State {state_name} failed "
            f"({fail_count}/{limit}), reason: {reason}"
        )
        if fail_count >= limit:
            self.safe_stop_reason = self.last_error
            print("[Failure Handler] Retry limit reached, entering SAFE_STOP")
            self.current_state = "SAFE_STOP"
        else:
            self.current_state = "ERROR_RECOVERY"
    
    def  run_state_machine(self):
        """Run one state-machine step."""
        print(f"\nCurrent state: {self.current_state}")
        state_before_run = self.current_state

        try:
            if self.current_state == "INIT":
                print("\n" + "="*60)
                print("State: INIT - robot arm initialization")
                print("="*60)
                print("[Initialization] Opening gripper")
                if not self.set_gripper_release():
                    self._enter_failure("INIT", "Failed to open gripper during initialization", recover_state="INIT")
                    return
                self._mark_state_success("INIT")
                self.current_state = "MOVE_TO_HOME"

            elif self.current_state == "MOVE_TO_HOME":
                print("\n" + "="*60)
                print("State: MOVE_TO_HOME - move to home pose")
                print("="*60)
                print("[Motion] Moving to home pose")
                if not self.movej(self.home_joint_angles, v=15):
                    self._enter_failure("MOVE_TO_HOME", "Robot arm failed to reach home pose", recover_state="MOVE_TO_HOME")
                    return
                self._mark_state_success("MOVE_TO_HOME")
                self.current_state = "PRINT_FRAME"

            elif self.current_state == "PRINT_FRAME":
                print("\n" + "="*60)
                print("State: PRINT_FRAME - print current frame")
                print("="*60)
                self.print_current_frame()
                self._mark_state_success("PRINT_FRAME")
                self.current_state = "MOVE_TO_DETECTION_POSE"

            elif self.current_state == "MOVE_TO_DETECTION_POSE":
                print("\n" + "="*60)
                print("State: MOVE_TO_DETECTION_POSE - move to detection pose")
                print("="*60)
                print("[Motion] Moving to detection pose")
                if not self.movej(self.detection_joint_angles, v=15):
                    self._enter_failure(
                        "MOVE_TO_DETECTION_POSE",
                        "Robot arm failed to reach detection pose",
                        recover_state="MOVE_TO_DETECTION_POSE",
                    )
                    return
                self._mark_state_success("MOVE_TO_DETECTION_POSE")
                self.current_state = "CHECK_OBJECT"

            elif self.current_state == "CHECK_OBJECT":
                print("\n" + "="*60)
                print("State: CHECK_OBJECT - detect object")
                print("="*60)
                if self.check_object():
                    print("[Object Detection] Object detected, starting grasp workflow")
                    self.detection_attempts = 0
                    self._mark_state_success("CHECK_OBJECT")
                    self.current_state = "MOVE_TO_PRE_GRASP"
                else:
                    self.detection_attempts += 1
                    print(
                        f"[Object Detection] No object detected, retrying after 3 seconds "
                        f"({self.detection_attempts}/{self.max_detection_attempts})"
                    )
                    if self.detection_attempts >= self.max_detection_attempts:
                        self._enter_failure(
                            "CHECK_OBJECT",
                            "Object was not detected for multiple consecutive attempts",
                            recover_state="MOVE_TO_DETECTION_POSE",
                            max_retries=2,
                        )
                        return
                    time.sleep(3)

            elif self.current_state == "MOVE_TO_PRE_GRASP":
                print("\n" + "="*60)
                print("State: MOVE_TO_PRE_GRASP - move to pre-grasp pose")
                print("="*60)
                if self.object_pose_in_base is not None:
                    pre_grasp_pose = self.object_pose_in_base.copy()
                    # Pre-grasp pose is in base frame; add +0.15 m on X.
                    pre_grasp_pose[0] += 0.15  # x +15 cm
                    pre_grasp_pose[2] += 0.05  # z +5 cm

                    print(f"[Motion] Moving to pre-grasp pose")
                    print(f"  Original pose: {self.object_pose_in_base[:3]}")
                    print(f"  Pre-grasp pose: {pre_grasp_pose[:3]}")
                    if not self.movel(pre_grasp_pose, v=15):
                        self._enter_failure(
                            "MOVE_TO_PRE_GRASP",
                            "Robot arm failed to reach pre-grasp pose",
                            recover_state="CHECK_OBJECT",
                        )
                        return
                    self._mark_state_success("MOVE_TO_PRE_GRASP")
                    self.current_state = "OPEN_GRIPPER"
                else:
                    print("[Error] Object pose is empty, returning to detection state")
                    self.current_state = "CHECK_OBJECT"

            elif self.current_state == "OPEN_GRIPPER":
                print("\n" + "="*60)
                print("State: OPEN_GRIPPER - open gripper")
                print("="*60)
                if not self.set_gripper_release():
                    self._enter_failure("OPEN_GRIPPER", "Failed to open gripper", recover_state="MOVE_TO_PRE_GRASP")
                    return
                self._mark_state_success("OPEN_GRIPPER")
                self.current_state = "MOVE_TO_GRASP"

            elif self.current_state == "MOVE_TO_GRASP":
                print("\n" + "="*60)
                print("State: MOVE_TO_GRASP - move to grasp pose")
                print("="*60)
                if self.object_pose_in_base is not None:
                    print(f"[Motion] Moving to grasp pose: {self.object_pose_in_base[:3]}")
                    if not self.movel(self.object_pose_in_base, v=10):
                        self._enter_failure("MOVE_TO_GRASP", "Robot arm failed to reach grasp pose", recover_state="MOVE_TO_PRE_GRASP")
                        return
                    self._mark_state_success("MOVE_TO_GRASP")
                    self.current_state = "CLOSE_GRIPPER"
                else:
                    print("[Error] Object pose is empty, returning to detection state")
                    self.current_state = "CHECK_OBJECT"

            elif self.current_state == "CLOSE_GRIPPER":
                print("\n" + "="*60)
                print("State: CLOSE_GRIPPER - close gripper")
                print("="*60)
                if not self.set_gripper_pick_on():
                    self._enter_failure("CLOSE_GRIPPER", "Failed to close gripper", recover_state="MOVE_TO_PRE_GRASP")
                    return
                self._mark_state_success("CLOSE_GRIPPER")
                self.current_state = "PLACE_OBJECT"

            elif self.current_state == "PLACE_OBJECT":
                print("\n" + "="*60)
                print("State: PLACE_OBJECT - place object after grasp")
                print("="*60)
                print("[Motion] Moving to place pose...")
                current_pose = self.get_current_cartesian_pose()
                if current_pose is not None:
                    place_pose = current_pose.copy()
                    # base frame coordinate system
                    # place_pose[0] += -0.05  # x -5 cm (forward)
                    place_pose[1] += 0.3  # y +30 cm (right)
                    place_pose[2] += 0.2  # z +20 cm (up)
                    if not self.movej_p(place_pose, v=15):
                        self._enter_failure("PLACE_OBJECT", "Failed to move to place pose", recover_state="MOVE_TO_DETECTION_POSE")
                        return
                    time.sleep(2)
                    print("[Gripper] Releasing gripper")
                    if not self.set_gripper_release():
                        self._enter_failure("PLACE_OBJECT", "Failed to release gripper during placement", recover_state="PLACE_OBJECT")
                        return
                    time.sleep(2)
                    self._mark_state_success("PLACE_OBJECT")
                    self.current_state = "RETURN_TO_DETECTION_POSE"
                else:
                    self._enter_failure(
                        "PLACE_OBJECT",
                        "Failed to get current pose; cannot perform placement",
                        recover_state="MOVE_TO_DETECTION_POSE",
                    )
                    return

            elif self.current_state == "RETURN_TO_DETECTION_POSE":
                print("\n" + "="*60)
                print("State: RETURN_TO_DETECTION_POSE - return to detection pose")
                print("="*60)
                print("[Motion] Returning to detection pose")
                if not self.movej(self.detection_joint_angles, v=15):
                    self._enter_failure(
                        "RETURN_TO_DETECTION_POSE",
                        "Failed to return to detection pose after grasp",
                        recover_state="MOVE_TO_DETECTION_POSE",
                    )
                    return
                time.sleep(2)
                print("\n" + "="*60)
                print("One grasp-verification round completed; robot arm has returned to detection pose")
                print("="*60)
                self._mark_state_success("RETURN_TO_DETECTION_POSE")
                self.current_state = "DONE"

            elif self.current_state == "RETURN_TO_HOME":
                print("\n" + "="*60)
                print("State: RETURN_TO_HOME - return to home pose")
                print("="*60)
                print("[Motion] Returning to home pose")
                if not self.movej(self.home_joint_angles, v=15):
                    self._enter_failure("RETURN_TO_HOME", "Failed to return to home pose", recover_state="RETURN_TO_HOME")
                    return
                time.sleep(2)
                print("\n" + "="*60)
                print("Grasping workflow completed")
                print("="*60)
                self._mark_state_success("RETURN_TO_HOME")
                self.current_state = "FINISHED"

            elif self.current_state == "ERROR_RECOVERY":
                print("\n" + "="*60)
                print("State: ERROR_RECOVERY - execute recovery strategy")
                print("="*60)
                print(f"[Recovery] Last error: {self.last_error}")
                print("[Recovery] Attempting to return to detection pose first")
                if self.movej(self.detection_joint_angles, v=15):
                    print(f"[Recovery] Recovery succeeded, returning to state: {self.recovery_target_state}")
                    self.current_state = self.recovery_target_state
                else:
                    self.safe_stop_reason = f"Recovery failed: {self.last_error}"
                    print("[Recovery] Failed to return to detection pose, entering SAFE_STOP")
                    self.current_state = "SAFE_STOP"

            elif self.current_state == "SAFE_STOP":
                print("\n" + "="*60)
                print("State: SAFE_STOP - safe shutdown")
                print("="*60)
                print(f"[Safe Shutdown] Reason: {self.safe_stop_reason}")
                print("[Safe Shutdown] Trying to release gripper; waiting for manual intervention")
                self.set_gripper_release()
                self.current_state = "DONE"

            elif self.current_state == "FINISHED":
                print("\nWorkflow finished")
                self.current_state = "DONE"

        except Exception as e:
            self._enter_failure(
                state_before_run,
                f"State execution exception: {e}",
                recover_state="CHECK_OBJECT",
                max_retries=1,
            )

    def reset_state(self):
        """Reset state-machine variables."""
        self.current_state = "INIT"
        self.retry_count = 0
        self.object_detected = False
        self.object_pose_in_base = None
        self.detection_attempts = 0
        self.state_failures = {}
        self.state_retry_limits = {
            "INIT": 2,
            "MOVE_TO_HOME": 2,
            "MOVE_TO_DETECTION_POSE": 2,
            "CHECK_OBJECT": 2,
            "MOVE_TO_PRE_GRASP": 2,
            "OPEN_GRIPPER": 2,
            "MOVE_TO_GRASP": 2,
            "CLOSE_GRIPPER": 2,
            "CHECK_GRASP_SUCCESS": 1,
            "PLACE_OBJECT": 2,
            "RETURN_TO_DETECTION_POSE": 2,
            "RETURN_TO_HOME": 2,
        }
        self.last_error = ""
        self.safe_stop_reason = ""
        self.recovery_target_state = "CHECK_OBJECT"
        print("[System] State has been reset")

    def wait_user_command_after_done(self):
        """
        Wait for user command after entering DONE.
        - Input R: run again from the initial flow.
        - Input Q: quit the program.

        Returns:
            bool: True to continue, False to exit.
        """
        print("\n" + "=" * 60)
        print("System is in DONE state, waiting for user command")
        print("Input R to run again, input Q to quit")
        print("=" * 60)

        while True:
            cmd = input("[Please enter command R/Q]: ").strip().upper()
            if cmd == "R":
                print("[User Command] Received R, continuing to next grasp-verification round")
                # Keep arm at detection pose and continue detection stage.
                self.current_state = "CHECK_OBJECT"
                self.retry_count = 0
                self.object_detected = False
                self.object_pose_in_base = None
                self.detection_attempts = 0
                self.state_failures = {}
                self.last_error = ""
                self.safe_stop_reason = ""
                return True
            if cmd == "Q":
                print("[User Command] Received Q, program will exit")
                return False
            print("[User Command] Invalid input, please enter R or Q")
    
    def run_complete_cycle(self, cycles=1):
        """
        Run full grasping cycles.
        
        Args:
            cycles: Number of cycles.
        """
        print("\n" + "="*60)
        print(f"Starting vision-guided grasping workflow, planned cycles: {cycles}")
        print("="*60)
        
        for cycle in range(cycles):
            print(f"\n{'='*60}")
            print(f"Cycle {cycle+1}/{cycles}")
            print(f"{'='*60}")
            
            self.reset_state()
            
            # Run state machine until done.
            while True:
                self.run_state_machine()
                if self.current_state == "DONE":
                    should_continue = self.wait_user_command_after_done()
                    if not should_continue:
                        print("\n[System] User chose to exit, stopping execution")
                        return
                time.sleep(0.5)  # Short delay.
                    
        print("\n" + "="*60)
        print("All grasping cycles completed")
        print("="*60)

    def shutdown(self):
        """Release ROS2 resources."""
        if self.ros_node is not None:
            try:
                self.ros_node.destroy_node()
            except Exception:
                pass
            self.ros_node = None
        if ROS2_POSE_MSG_AVAILABLE and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

        
def main():
    """Program entry point."""
    print("\n" + "="*60)
    print("Vision Grasping Execution System")
    print("="*60)
    
    parser = argparse.ArgumentParser(description="RM vision-guided grasping execution system")
    parser.add_argument("--ip", default="192.168.1.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pose-topic", default="/pose_estimation_result")
    args = parser.parse_args()

    # Create system instance (real arm + grasp_perception_msgs.msg topic only).
    grasp_system = VisionGraspingSystem(
        ip=args.ip,
        port=args.port,
        pose_topic=args.pose_topic,
    )
    
    try:
        grasp_system.run_complete_cycle(cycles=1)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred during program execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        grasp_system.shutdown()
        print("\n" + "="*60)
        print("Program finished")
        print("="*60)


if __name__ == "__main__":
    main()
