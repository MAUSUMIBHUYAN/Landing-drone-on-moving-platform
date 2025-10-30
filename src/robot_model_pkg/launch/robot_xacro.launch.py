from launch import LaunchDescription
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to your xacro
    robot_pkg = get_package_share_directory('robot_model_pkg')
    xacro_file = os.path.join(robot_pkg, 'urdf', 'robot.xacro')

    # Convert xacro → urdf
    robot_description = Command(['xacro ', xacro_file])

    return LaunchDescription([
        # Launch Gazebo (empty world)
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'),

        # Publish robot description
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),

        # Spawn robot in Gazebo
        # Node(
        #     package='gazebo_ros',
        #     executable='spawn_entity.py',
        #     arguments=['-topic', 'robot_description',
        #                '-entity', 'robot_model'],
        #     output='screen'
        # ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description',
                       '-entity', 'robot_model',
                       '-x', '2.0', '-y', '-3.0', '-z', '0.1'],
            output='screen'
        ),
        Node(
	    package='controller_manager',
	    executable='spawner',
	    arguments=['diff_cont'],
	),
        Node(
            package='random_motion',
            executable='random_move',
            output='screen'
        )
    ])
