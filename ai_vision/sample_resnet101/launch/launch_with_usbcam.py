import os  # For accessing environment variables
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.logging import get_logger

def generate_launch_description():
    # Get USB_CAMERA_PATH from environment variables
    logger = get_logger('usb_cam_launch')
    usb_camera_path = os.environ.get('USB_CAMERA_PATH','/dev/video0')  # Default to /dev/video0 if not set

    
    logger.info(f'USB_CAMERA_PATH set to: {usb_camera_path}')
    return LaunchDescription([
        # Node for sample_resnet101_quantized
        Node(
            package='sample_resnet101_quantized',  # Replace with the actual package name
            executable='qrb_ros_resnet101',  # Replace with the actual executable name
            output='screen',  # Output logs to terminal
        ),

        # Node for usb_cam
        Node(
            package='usb_cam',  # Package name
            executable='usb_cam_node_exe',  # Executable name
            name='usb_cam_node',  # Node name (optional)
            output='screen',  # Output logs to terminal
            parameters=[
                {'video_device': usb_camera_path},  # Fetch USB_CAMERA_PATH from environment
                {'pixel_format': 'mjpeg2rgb'},
                {'image_width': 640},
                {'image_height': 480},
                {'framerate': 10.0},
                {'queue_size': 1000},
            ],
        ),
    ])
