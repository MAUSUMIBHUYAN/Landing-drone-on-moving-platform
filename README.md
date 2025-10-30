Setup Instructions:- 

Download the zip file:
unzip project_ws.zip
cd project_ws

Remove old build folders (if any):
rm -rf build/ install/ log/

Source your ROS2 environment:
source /opt/ros/humble/setup.bash

Build the workspace:
colcon build

Source the workspace setup:
source install/setup.bash

Launch the simulation:
ros2 launch sjtu_drone_bringup multi_robot.launch.py


This will open the Gazebo simulation with the drone spawned and ready.

Now, in another terminal (Terminal 2), run the control script:
python3 mpc_pid.py

