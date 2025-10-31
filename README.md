## ⚙️ Setup Instructions

Follow the steps below to set up and run the simulation properly 👇  

---

### 1️⃣ Download and Extract the Project
```bash
# Download and unzip the workspace
unzip project_ws.zip
cd project_ws


```

### 2️⃣ Clean Previous Build Files (if any)
```bash
rm -rf build/ install/ log/

```

### 3️⃣ Source Your ROS2 Environment
Make sure you have ROS2 Humble installed:
```bash
source /opt/ros/humble/setup.bash

```

### 4️⃣ Build the Workspace
```bash
colcon build
```

### 5️⃣ Source the Workspace Setup
```bash
source install/setup.bash
```

### 6️⃣ Launch the Simulation
```bash
ros2 launch sjtu_drone_bringup multi_robot.launch.py
```
This will open the Gazebo environment with the drone spawned and ready.


### 7️⃣ Run the Control Script (Terminal 2)
```bash
python3 mpc_pid.py
```

## 🎥 Demonstration 
https://github.com/user-attachments/assets/298ca848-afca-43bd-9505-4675c09f31d6

## 📊 Plots and Analysis

All generated plots are stored inside the **`plots/`** folder.  

---

## ⚠️ Troubleshooting & Notes
---

You may occasionally encounter a **common Gazebo issue** where the **robot or drone spawns below the ground**, or the **Gazebo world doesn’t load properly**.  

---
👉 **Fix:**
```bash
# Close Gazebo completely and relaunch
ros2 launch sjtu_drone_bringup multi_robot.launch.py










