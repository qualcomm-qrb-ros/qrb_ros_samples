# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
import yaml
import xacro
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    robot_gazebo_pkg = 'qrb_ros_sim_gazebo'
    default_laser_config = os.path.join(
        get_package_share_directory(robot_gazebo_pkg), 'config', 'params', 'qrb_robot_base_laser_params.yaml')
    default_imu_config = os.path.join(
        get_package_share_directory(robot_gazebo_pkg), 'config', 'params', 'imu_params.yaml')

    launch_args = [
        DeclareLaunchArgument('launch_config_file', default_value=''),
        DeclareLaunchArgument('world_model', default_value='warehouse'),
        DeclareLaunchArgument('robot_entity_name', default_value='qrb_amr_robot'),
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('enable_laser', default_value='true'),
        DeclareLaunchArgument('laser_config_file', default_value=default_laser_config),
        DeclareLaunchArgument('enable_imu', default_value='true'),
        DeclareLaunchArgument('imu_config_file', default_value=default_imu_config),
        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        DeclareLaunchArgument('initial_z', default_value='0.0'),
        DeclareLaunchArgument('initial_roll', default_value='0.0'),
        DeclareLaunchArgument('initial_pitch', default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
    ]

    return LaunchDescription(launch_args + [
        OpaqueFunction(function=launch_setup)
    ])

def launch_setup(context):
    config = {
        'launch_config_file': LaunchConfiguration('launch_config_file').perform(context),
        'world_model': LaunchConfiguration('world_model').perform(context),
        'robot_entity_name': LaunchConfiguration('robot_entity_name').perform(context),
        'namespace': LaunchConfiguration('namespace').perform(context),
        'enable_laser': LaunchConfiguration('enable_laser').perform(context),
        'laser_config_file': LaunchConfiguration('laser_config_file').perform(context),
        'enable_imu': LaunchConfiguration('enable_imu').perform(context),
        'imu_config_file': LaunchConfiguration('imu_config_file').perform(context),
        'initial_x': LaunchConfiguration('initial_x').perform(context),
        'initial_y': LaunchConfiguration('initial_y').perform(context),
        'initial_z': LaunchConfiguration('initial_z').perform(context),
        'initial_roll': LaunchConfiguration('initial_roll').perform(context),
        'initial_pitch': LaunchConfiguration('initial_pitch').perform(context),
        'initial_yaw': LaunchConfiguration('initial_yaw').perform(context),
    }

    if config['launch_config_file']:
        with open(config['launch_config_file'], 'r') as f:
            yaml_config = yaml.safe_load(f)
            for key in config:
                if key in yaml_config:
                    config[key] = yaml_config[key]

    gz_pkg_share = get_package_share_directory('qrb_ros_sim_gazebo')
    robot_model_path = os.path.join(gz_pkg_share, 'urdf', 'gazebo_qrb_robot_base.urdf.xacro')

    # Launch the gazebo and load world model
    load_world_model = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={
            'gz_args': [os.path.join(gz_pkg_share, 'worlds', f'{config["world_model"]}.sdf') + ' -v 1']
        }.items()
    )

    robot_launch = generate_robot_launch(config, robot_model_path)

    return [load_world_model, *robot_launch]

def generate_robot_launch(config, robot_model_path):
    doc = xacro.process_file(
        robot_model_path,
        mappings={
            'robot_name': config['robot_entity_name'],
            'enable_laser': config['enable_laser'],
            'laser_config': config['laser_config_file'],
            'enable_imu': config['enable_imu'],
            'imu_config': config['imu_config_file'],
        }
    )

    # After starting the robot_state_publisher node, it will publish the robot_description topic,
    # which contains the content of the URDF model file. It will also subscribe to the /joint_states
    # topic to get joint data, and then publish the tf and tf_static topics.
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{"use_sim_time": True}, {'robot_description': doc.toxml()}],
        namespace=config['namespace'],
        output='screen'
    )

    # Create robot model in the gazebo world
    spawn_robot_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        namespace=config['namespace'],
        arguments=[
            '-entity', config['robot_entity_name'],
            '-topic', 'robot_description',
            '-x', config['initial_x'],
            '-y', config['initial_y'],
            '-z', config['initial_z'],
            '-R', config['initial_roll'],
            '-P', config['initial_pitch'],
            '-Y', config['initial_yaw']
        ],
    )

    # Ros-Gazebo bridge
    ros_gz_bridge_configs = []
    ## Lidar
    if config['enable_laser'].lower() == 'true':
        ros_gz_bridge_configs.append(
            {'args': ['scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan']}
        )
    ## IMU
    if config['enable_imu'].lower() == 'true':
        ros_gz_bridge_configs.append(
            {'args': ['imu@sensor_msgs/msg/Imu[gz.msgs.IMU']}
        )
    ## Differential driver
    ros_gz_bridge_configs.append(
        {'args': [
            'odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            'cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'
        ]}
    )
    ## Robot state
    ros_gz_bridge_configs.append(
        {'args': [
            'joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
        ]}
    )
    ## Clock
    ros_gz_bridge_configs.append(
        {'args': ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']}
    )
    ros_gz_bridges = []
    for bridge_config in ros_gz_bridge_configs:
        ros_gz_bridges.append(
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=bridge_config['args'],
                namespace=config['namespace'],
                **({'remappings': bridge_config['remappings']} if 'remappings' in bridge_config else {}),
                output='screen'
            )
        )

    return [
        robot_state_publisher_node,
        spawn_robot_entity,
        *ros_gz_bridges,
    ]