from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Paths
    drone_bringup_dir = get_package_share_directory('sjtu_drone_bringup')
    robot_model_dir = get_package_share_directory('robot_model_pkg')
    sjtu_desc_dir = get_package_share_directory('sjtu_drone_description')

    # World file
    world_file = os.path.join(sjtu_desc_dir, 'worlds', 'simple_world.world')

    # Gazebo + Drone (already spawns drone & sets world)
    gazebo_with_drone = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(drone_bringup_dir, 'launch', 'sjtu_drone_gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # (Optional) If you need extra drone config on top of gazebo_with_drone
    # uncomment this, otherwise remove:
    # drone_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(drone_bringup_dir, 'launch', 'drone.launch.py')
    #     )
    # )

    # Wheeled robot launch
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_model_dir, 'launch', 'robot_xacro.launch.py')
        )
    )

    return LaunchDescription([
        gazebo_with_drone,
        robot_launch
        # drone_launch  # only if really needed
    ])
