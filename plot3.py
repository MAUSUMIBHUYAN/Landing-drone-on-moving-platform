#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import time
import casadi as ca
import matplotlib.pyplot as plt

class MPCController:
    """Optimized Model Predictive Controller for drone trajectory tracking."""
    def __init__(self, dt, N=8):
        self.dt = dt; self.N = N
        self.A = ca.DM([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = ca.DM([[0.5*self.dt**2, 0], [0, 0.5*self.dt**2], [self.dt, 0], [0, self.dt]])
        self.Q = ca.diag(ca.DM([10., 10., 1., 1.])); self.R = ca.diag(ca.DM([0.5, 0.5]))
        self.max_accel = 2.0
        self.X_sol = None; self.U_sol = None
        self._setup_solver()
        


    def _setup_solver(self):
        self.opti = ca.Opti() # Use default SX symbolic framework for this problem size
        self.X = self.opti.variable(4, self.N + 1); self.U = self.opti.variable(2, self.N)
        self.x0 = self.opti.parameter(4); self.X_ref = self.opti.parameter(4, self.N + 1)
        cost = 0
        self.opti.subject_to(self.X[:, 0] == self.x0)
        for k in range(self.N):
            state_error = self.X[:, k] - self.X_ref[:, k]
            control_input = self.U[:, k]
            cost += ca.mtimes([state_error.T, self.Q, state_error])
            cost += ca.mtimes([control_input.T, self.R, control_input])
            x_next = self.A @ self.X[:, k] + self.B @ self.U[:, k]
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        self.opti.subject_to(self.opti.bounded(-self.max_accel, self.U[0,:], self.max_accel))
        self.opti.subject_to(self.opti.bounded(-self.max_accel, self.U[1,:], self.max_accel))
        self.opti.minimize(cost)
        p_opts = {"expand": True}
        s_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 
                  "ipopt.warm_start_init_point": "yes", 'ipopt.tol': 1e-4}
        self.opti.solver('ipopt', p_opts, s_opts)

    def compute_control(self, current_state, ref_trajectory, logger):
        self.opti.set_value(self.x0, current_state); self.opti.set_value(self.X_ref, ref_trajectory)
        if self.X_sol is not None and self.U_sol is not None:
            self.opti.set_initial(self.X, self.X_sol); self.opti.set_initial(self.U, self.U_sol)
        try: 
            sol = self.opti.solve()
            self.X_sol = sol.value(self.X); self.U_sol = sol.value(self.U)
            return sol.value(self.U)[:, 0]
        except Exception as e:
            logger.error(f"MPC solver failed: {e}", throttle_duration_sec=2)
            self.X_sol = None; self.U_sol = None
            return np.zeros(2)

class KalmanFilter:
    """A Kalman Filter for tracking position, velocity, and acceleration."""
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 10.0
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)

    def predict(self, dt):
        # Overridden by child classes
        pass

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

class PlanarVehicleKF(KalmanFilter):
    """Kalman filter for a 6D planar state [x, y, vx, vy, ax, ay]."""
    def __init__(self, dt, q_accel=1.0, r_meas=5.0):
        super().__init__(dim_x=6, dim_z=2)
        self.dt = dt
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.R = np.eye(self.dim_z) * r_meas
        q_val = np.array([0.25*dt**4, 0.25*dt**4, dt**2, dt**2, dt**2, dt**2]) * q_accel
        self.Q = np.diag(q_val)

    def predict(self, dt):
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, dt, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

class YawKF(KalmanFilter):
    """Kalman filter for a 3D yaw state [yaw, omega, alpha]."""
    def __init__(self, dt, q_pos=0.05, q_vel=0.1, r_meas=0.3):
        super().__init__(dim_x=3, dim_z=1)
        self.dt = dt
        self.H = np.array([[1, 0, 0]])
        self.R = np.array([[r_meas]])
        self.Q = np.diag([q_pos, q_vel, 0.1])
    
    def predict(self, dt):
        self.F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

class FollowDescentDroneController(Node):
    def __init__(self):
        super().__init__('follow_descent_drone_controller')
        self.get_logger().info("Optimized Drone Landing Controller Initializing...")

        self.declare_parameter('visualize', True)
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10)
        self.robot_pose_sub = self.create_subscription(Odometry, '/odom', self.robot_pose_callback, qos_profile)
        self.image_sub = self.create_subscription(Image, '/simple_drone/bottom/image_raw', self.image_callback, qos_profile)
        self.drone_odom_sub = self.create_subscription(Odometry, '/simple_drone/odom', self.drone_odom_callback, qos_profile)
        
        self.dt = 0.1; self.max_xy_velocity = 1.2; self.max_z_velocity = 0.5
        self.kp_xy, self.ki_xy, self.kd_xy = 0.7, 0.02, 0.25; self.kp_z, self.ki_z, self.kd_z = 0.8, 0.01, 0.1
        self.integral_xy_max, self.integral_z_max = 2.0, 1.0

        self.bridge = CvBridge(); self.altitude_target = 2.5; self.robot_height = 0.30
        self.follow_area_threshold = 1250; self.descent_area_threshold = 5000; self.land_area_threshold = 8750
        self.follow_altitude = 1.0; self.landing_speed = 0.9; self.land_altitude_threshold = 0.15
        self.cart_front_offset = 0.25; self.prediction_latency = 0.15
        self.FOLLOW_GRACE_PERIOD = 0.5; self.LAND_GRACE_PERIOD = 0.3
        self.MPC_PID_SWITCH_ERROR = 0.1

        self.drone_position = np.zeros(3); self.drone_velocity = np.zeros(3); self.drone_yaw = 0.0
        self.cos_yaw = 1.0; self.sin_yaw = 0.0
        self.robot_position = np.zeros(2); self.robot_velocity = np.zeros(2)
        self.robot_acceleration = np.zeros(2); self.robot_yaw = 0.0; self.robot_angular_velocity = 0.0; self.unwrapped_yaw = 0.0
        self.integral_xy = np.zeros(2); self.prev_error_xy = np.zeros(2); self.integral_z = 0.0; self.prev_error_z = 0.0
        self.target_pixel = None; self.target_area = 0; self.last_bbox = None
        self.frame_center = None; self.vision_active = False; self.last_vision_time = 0
        self.state = "TAKEOFF"; self.has_drone_odom = False; self.has_robot_odom = False; self.is_prediction_stable = False
        self.vision_loss_timestamp = 0.0; self.frame_count = 0

        self.kf_xy = PlanarVehicleKF(self.dt, q_accel=1.5, r_meas=5.0)
        self.kf_yaw = YawKF(self.dt, r_meas=0.3)
        self.mpc_controller = MPCController(self.dt)
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        self.takeoff_timer = self.create_timer(1.0, self.initiate_takeoff)
        self.kf_logs = {
            "time": [],
            "x_true": [], "y_true": [],
            "x_est": [], "y_est": [],
            "vx_true": [], "vy_true": [],
            "vx_est": [], "vy_est": []
        }

    def initiate_takeoff(self):
        self.get_logger().info("State: TAKEOFF - Publishing takeoff command.")
        self.takeoff_pub.publish(Empty()); self.state = "ASCENDING"; self.takeoff_timer.cancel()

    def euler_from_quaternion(self, q):
        t3 = +2.0*(q.w*q.z + q.x*q.y); t4 = +1.0 - 2.0*(q.y**2 + q.z**2)
        return 0, 0, math.atan2(t3, t4)

    def drone_odom_callback(self, msg: Odometry):
        if not self.has_drone_odom: self.get_logger().info("Received first drone odometry message.")
        self.has_drone_odom = True
        self.drone_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        _, _, self.drone_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        vx_b=msg.twist.twist.linear.x; vy_b=msg.twist.twist.linear.y
        self.cos_yaw = math.cos(self.drone_yaw); self.sin_yaw = math.sin(self.drone_yaw)
        self.drone_velocity[0] = vx_b*self.cos_yaw - vy_b*self.sin_yaw
        self.drone_velocity[1] = vx_b*self.sin_yaw + vy_b*self.cos_yaw
        self.drone_velocity[2] = msg.twist.twist.linear.z

    def robot_pose_callback(self, msg: Odometry):
        current_time = time.time()
        new_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        _, _, new_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        
        # --- TRUE VELOCITY IN LOCAL FRAME ---
        vx_local = msg.twist.twist.linear.x
        vy_local = msg.twist.twist.linear.y
        
        # --- GET WORLD FRAME VELOCITY (vx_true, vy_true) ---
        # Get the yaw used for this Odometry reading (new_yaw)
        cos_yaw = math.cos(new_yaw)
        sin_yaw = math.sin(new_yaw)
        
        # Transform the local frame velocities (vx_local, vy_local) to the world frame (vx_true, vy_true)
        vx_true_world = vx_local * cos_yaw - vy_local * sin_yaw
        vy_true_world = vx_local * sin_yaw + vy_local * cos_yaw

        if not self.has_robot_odom:
            # ... (Existing initialization code remains the same) ...
            self.get_logger().info("Received first robot odometry message.")
            self.kf_xy.x[0] = new_pos[0]; self.kf_xy.x[1] = new_pos[1]
            self.unwrapped_yaw=new_yaw; self.kf_yaw.x[0]=new_yaw; self.robot_yaw=new_yaw
            self.last_robot_odom_time=current_time; self.has_robot_odom=True; return
            
        dt_odom = current_time - self.last_robot_odom_time
        if dt_odom < 0.01: return
        self.last_robot_odom_time = current_time
        dyaw = new_yaw - self.robot_yaw; dyaw = (dyaw + math.pi) % (2*math.pi) - math.pi; self.unwrapped_yaw += dyaw
        self.kf_xy.predict(dt_odom); self.kf_yaw.predict(dt_odom)
        self.kf_xy.update(new_pos); self.kf_yaw.update(np.array([self.unwrapped_yaw]))
        
        # --- Log actual vs estimated state (USING WORLD FRAME VELOCITY) ---
        t = time.time()
        
        self.kf_logs["time"].append(t)
        self.kf_logs["x_true"].append(new_pos[0])
        self.kf_logs["y_true"].append(new_pos[1])
        self.kf_logs["vx_true"].append(vx_true_world)  # <-- CORRECTED
        self.kf_logs["vy_true"].append(vy_true_world)  # <-- CORRECTED

        self.kf_logs["x_est"].append(self.kf_xy.x[0])
        self.kf_logs["y_est"].append(self.kf_xy.x[1])
        self.kf_logs["vx_est"].append(self.kf_xy.x[2])
        self.kf_logs["vy_est"].append(self.kf_xy.x[3])

        
        self.robot_position = self.kf_xy.x[:2]; self.robot_velocity = self.kf_xy.x[2:4]; self.robot_acceleration = self.kf_xy.x[4:]
        self.robot_yaw=(self.kf_yaw.x[0]+math.pi)%(2*math.pi)-math.pi; self.robot_angular_velocity = self.kf_yaw.x[1]

        if not self.is_prediction_stable and self.kf_xy.P[0,0] < 1.0 and self.kf_xy.P[1,1] < 1.0:
            self.is_prediction_stable = True; self.get_logger().info("Kalman Filters converged. Prediction is now STABLE.")

    def predict_robot_state(self, prediction_time: float):
        if not self.is_prediction_stable: return self.robot_position, self.robot_velocity
        t = prediction_time + self.prediction_latency
        x, y, vx, vy, ax, ay = self.kf_xy.x
        yaw, omega, _ = self.kf_yaw.x
        if abs(omega) > 0.01:
            speed = math.hypot(vx, vy); radius = speed/omega if abs(omega)>1e-6 else float('inf'); d_theta = omega*t
            s_dt, c_dt = math.sin(d_theta), math.cos(d_theta); dx_r, dy_r = radius*s_dt, radius*(1-c_dt)
            c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
            pred_x = x + dx_r*c_yaw - dy_r*s_yaw; pred_y = y + dx_r*s_yaw + dy_r*c_yaw
            pred_yaw=yaw+d_theta; pred_vx=speed*math.cos(pred_yaw); pred_vy=speed*math.sin(pred_yaw)
        else:
            pred_x=x+vx*t+0.5*ax*t**2; pred_y=y+vy*t+0.5*ay*t**2
            pred_vx=vx+ax*t; pred_vy=vy+ay*t; pred_yaw=yaw
        offset_vec=np.array([math.cos(pred_yaw), math.sin(pred_yaw)])*self.cart_front_offset
        return np.array([pred_x, pred_y])+offset_vec, np.array([pred_vx, pred_vy])

    def image_callback(self, img_msg: Image):
        try: frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError: return
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        h, w = frame.shape[:2]; self.frame_center = (w//2, h//2)
        roi = frame; roi_offset = (0,0)
        if self.last_bbox is not None:
            lx, ly, lw, lh = [int(v) for v in self.last_bbox]
            roi_x=max(0, lx-lw//2); roi_y=max(0, ly-lh//2); roi_w=min(w-roi_x, lw*2); roi_h=min(h-roi_y, lh*2)
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]; roi_offset = (roi_x, roi_y)
        processed_frame = cv2.GaussianBlur(roi, (11, 11), 0)
        hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        lower_red=cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])); upper_red=cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(lower_red, upper_red)
        kernel = np.ones((20,20), np.uint8); mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_target = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > self.follow_area_threshold:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx=int(M["m10"]/M["m00"])+roi_offset[0]; cy=int(M["m01"]/M["m00"])+roi_offset[1]
                    self.target_pixel=(cx, cy); self.target_area=area; self.vision_active=True; self.last_vision_time=time.time()
                    x, y, w_c, h_c = cv2.boundingRect(largest_contour)
                    self.last_bbox = (x+roi_offset[0], y+roi_offset[1], w_c, h_c); found_target = True
        if not found_target: self.vision_active = False; self.last_bbox = None
        if self.get_parameter('visualize').get_parameter_value().bool_value and (self.frame_count % 3 == 0):
            if self.vision_active:
                x, y, w_c, h_c = self.last_bbox
                cv2.rectangle(frame, (x, y), (x+w_c, y+h_c), (0, 255, 0), 2)
                cv2.circle(frame, self.target_pixel, 10, (0, 255, 0), 2)
            cv2.imshow("Drone Camera Feed", frame); cv2.imshow("Red Mask", mask); cv2.waitKey(1)
        self.frame_count+=1
        
    def compute_pid_control(self, target_pos: np.ndarray):
        error_xy = target_pos[:2] - self.drone_position[:2]
        self.integral_xy = np.clip(self.integral_xy+error_xy*self.dt, -self.integral_xy_max, self.integral_xy_max)
        derivative_xy=(error_xy-self.prev_error_xy)/self.dt; self.prev_error_xy=error_xy
        control_world_xy=self.kp_xy*error_xy+self.ki_xy*self.integral_xy+self.kd_xy*derivative_xy
        error_z=target_pos[2]-self.drone_position[2]
        self.integral_z=np.clip(self.integral_z+error_z*self.dt, -self.integral_z_max, self.integral_z_max)
        derivative_z=(error_z-self.prev_error_z)/self.dt; self.prev_error_z=error_z
        vz=self.kp_z*error_z+self.ki_z*self.integral_z+self.kd_z*derivative_z
        vx_w, vy_w = control_world_xy[0], control_world_xy[1]
        vx_b = vx_w*self.cos_yaw + vy_w*self.sin_yaw; vy_b = -vx_w*self.sin_yaw + vy_w*self.cos_yaw
        return np.array([vx_b, vy_b, vz])
        
    def compute_mpc_control(self):
        ref_traj = np.zeros((4, self.mpc_controller.N + 1))
        for i in range(self.mpc_controller.N + 1):
            t_pred=i*self.dt; pred_pos, pred_vel = self.predict_robot_state(t_pred)
            ref_traj[:2, i] = pred_pos; ref_traj[2:, i] = pred_vel
        current_state = np.array([self.drone_position[0],self.drone_position[1],self.drone_velocity[0],self.drone_velocity[1]])
        accel_cmd = self.mpc_controller.compute_control(current_state, ref_traj, self.get_logger())
        vx_w=self.drone_velocity[0]+accel_cmd[0]*self.dt; vy_w=self.drone_velocity[1]+accel_cmd[1]*self.dt
        vx_b=vx_w*self.cos_yaw + vy_w*self.sin_yaw; vy_b=-vx_w*self.sin_yaw + vy_w*self.cos_yaw
        return np.array([vx_b, vy_b])

    def control_loop(self):
        if not self.has_drone_odom: self.get_logger().info("Waiting for drone odometry...", throttle_duration_sec=5); return
        state_handlers = {"ASCENDING": self.handle_ascending, "SEARCHING": self.handle_searching,
                          "FOLLOWING": self.handle_following, "LANDING": self.handle_landing}
        handler = state_handlers.get(self.state)
        control = handler(time.time()) if handler else np.zeros(3)
        if self.state not in ["LANDED", "TAKEOFF"]:
            speed_xy = np.linalg.norm(control[:2])
            if speed_xy > self.max_xy_velocity: control[:2] *= self.max_xy_velocity / speed_xy
            self.publish_control(control)

    def handle_ascending(self, current_time):
        if not self.has_robot_odom: self.get_logger().info("Ascending: Waiting for robot odometry...", throttle_duration_sec=5); return np.array([0.,0.,0.2])
        if self.drone_position[2] >= self.altitude_target-0.1: self.state="SEARCHING"; self.get_logger().info("State: SEARCHING - At altitude."); return np.zeros(3)
        return np.array([0., 0., 0.4])

    def handle_searching(self, current_time):
        if self.vision_active: self.state="FOLLOWING"; self.get_logger().info("State: FOLLOWING - Target acquired."); return np.zeros(3)
        pred_pos, _ = self.predict_robot_state(0.5)
        self.get_logger().info("SEARCHING with predictive odometry...", throttle_duration_sec=1)
        return self.compute_pid_control(np.array([pred_pos[0], pred_pos[1], self.altitude_target]))

    def handle_following(self, current_time):
        if not self.vision_active:
            if self.vision_loss_timestamp == 0.0: self.vision_loss_timestamp = current_time; self.get_logger().info(f"Vision lost. Grace period: {self.FOLLOW_GRACE_PERIOD}s.", throttle_duration_sec=2)
            if current_time - self.vision_loss_timestamp > self.FOLLOW_GRACE_PERIOD: self.state = "SEARCHING"; self.get_logger().info("State: SEARCHING - Grace period expired."); return np.zeros(3)
        else: self.vision_loss_timestamp = 0.0
        if self.vision_active and self.target_area > self.land_area_threshold: self.state = "LANDING"; self.get_logger().info("State: LANDING - Target close for landing."); return np.zeros(3)
        target_pos_now, _ = self.predict_robot_state(0.0)
        error_now = np.linalg.norm(self.drone_position[:2] - target_pos_now)
        control_xy = self.compute_mpc_control() if error_now > self.MPC_PID_SWITCH_ERROR else self.compute_pid_control(np.array([target_pos_now[0], target_pos_now[1], self.follow_altitude]))[:2]
        current_area = self.target_area if self.vision_active else self.descent_area_threshold
        target_alt = self.follow_altitude if current_area >= self.descent_area_threshold else self.altitude_target
        control_z = np.clip(0.4*(target_alt - self.drone_position[2]), -self.max_z_velocity, self.max_z_velocity)
        return np.array([control_xy[0], control_xy[1], control_z])

    def handle_landing(self, current_time):
        if not self.vision_active:
            if self.vision_loss_timestamp == 0.0: self.vision_loss_timestamp = current_time; self.get_logger().warning(f"Vision lost in LANDING. Grace period: {self.LAND_GRACE_PERIOD}s.", throttle_duration_sec=2)
            if current_time - self.vision_loss_timestamp > self.LAND_GRACE_PERIOD: self.state = "SEARCHING"; self.get_logger().warning("State: SEARCHING - Vision lost during landing! Aborting."); return np.zeros(3)
        else: self.vision_loss_timestamp = 0.0
        if self.drone_position[2] <= self.robot_height + self.land_altitude_threshold:
            self.land_pub.publish(Empty()); self.state = "LANDED"; self.get_logger().info("State: LANDED - Final land command sent."); self.control_timer.cancel()
            return np.zeros(3)
        control_xy = self.compute_mpc_control()
        return np.array([control_xy[0], control_xy[1], -self.landing_speed])

    def publish_control(self, control: np.ndarray):
        twist = Twist(); twist.linear = Vector3(x=float(control[0]), y=float(control[1]), z=float(control[2]))
        self.cmd_vel_pub.publish(twist)

    def shutdown_hook(self):
        self.get_logger().warn("Shutdown initiated, sending emergency land command.")
        if self.state != "LANDED": self.land_pub.publish(Empty())
        if self.takeoff_timer: self.takeoff_timer.cancel()
        if self.control_timer: self.control_timer.cancel()

        # --- Plot Actual vs Estimated States ---
        if len(self.kf_logs["time"]) > 10:
            t0 = self.kf_logs["time"][0]
            time_axis = [t - t0 for t in self.kf_logs["time"]]

            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(time_axis, self.kf_logs["x_true"], label="x_true", color="blue")
            plt.plot(time_axis, self.kf_logs["x_est"], '--', label="x_est", color="red")
            plt.plot(time_axis, self.kf_logs["y_true"], label="y_true", color="green")
            plt.plot(time_axis, self.kf_logs["y_est"], '--', label="y_est", color="orange")
            plt.title("Position: Actual vs Estimated")
            plt.xlabel("Time [s]"); plt.ylabel("Position [m]")
            plt.legend(); plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(time_axis, self.kf_logs["vx_true"], label="vx_true", color="blue")
            plt.plot(time_axis, self.kf_logs["vx_est"], '--', label="vx_est", color="red")
            plt.plot(time_axis, self.kf_logs["vy_true"], label="vy_true", color="green")
            plt.plot(time_axis, self.kf_logs["vy_est"], '--', label="vy_est", color="orange")
            plt.title("Velocity: Actual vs Estimated")
            plt.xlabel("Time [s]"); plt.ylabel("Velocity [m/s]")
            plt.legend(); plt.grid(True)

            plt.tight_layout()
            plt.savefig("kf_estimation_plot.png", dpi=300)  # <- saves the plot
            plt.show()

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    controller = FollowDescentDroneController()
    try: rclpy.spin(controller)
    except KeyboardInterrupt: pass
    finally:
        controller.shutdown_hook()
        if rclpy.ok():
            controller.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()