
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    sjtu_drone_bringup_path = get_package_share_directory('sjtu_drone_bringup')

    rviz_path = os.path.join(
        sjtu_drone_bringup_path, "rviz", "rviz.rviz"
    )
    
    yaml_file_path = os.path.join(
        get_package_share_directory('sjtu_drone_bringup'),
        'config', 'drone.yaml'
    )

    model_ns = "drone"

    with open(yaml_file_path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        model_ns = yaml_dict["namespace"]

    return LaunchDescription([
        DeclareLaunchArgument(
            "controller",
            default_value="keyboard",
            description="Type of controller: keyboard (default) or joystick",
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=[
                "-d", rviz_path
            ],
            output="screen",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sjtu_drone_bringup_path, 'launch', 'sjtu_drone_gazebo.launch.py')
            )
        ),
    ])
