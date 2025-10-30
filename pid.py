#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import NavSatFix, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import time

class DroneGPSVisionFollower(Node):
    def __init__(self):
        super().__init__('drone_gps_vision_follower')

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10)

        # --- Subscribers ---
        self.robot_pose_sub = self.create_subscription(Odometry, '/odom', self.robot_pose_callback, 10)
        self.drone_gps_sub = self.create_subscription(NavSatFix, '/simple_drone/gps/nav', self.drone_gps_callback, 10)

        # Image topic
        self.camera_topic = '/simple_drone/bottom/image_raw'
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.bridge = CvBridge()

        # --- Parameters (tune these) ---
        self.altitude_target = 2.0
        self.robot_height = 0.30        # robot body top above ground (m)
        self.safety_margin = 0.05       # stop just above robot (m)
        self.KP_XY = 0.8
        self.MAX_SPEED = 1.5
        self.ALIGN_PIX_TOL = 30
        self.AREA_LAND_THRESHOLD = 6000    # area at which we consider "close enough to land"
        self.AREA_TOO_CLOSE = 20000        # area meaning "we're on/over it" -> immediate land
        self.ALIGN_CONFIRM_TIME = 0.8      # seconds of stable alignment before descent
        self.linear_speed = 0.6

        # Position states
        self.robot_x = None
        self.robot_y = None
        self.drone_x = None
        self.drone_y = None
        self.drone_z = None

        # GPS reference
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None

        # Vision state
        self.last_frame_width = None
        self.last_frame_height = None
        self.last_centroid = None
        self.last_area = 0
        self.vision_age = 999.0
        self.last_detection_time = 0.0

                # --- additional smoothing/prediction state ---
        self.centroid_ema = None         # (cx, cy) smoothed
        self.area_ema = 0.0
        self.EMA_ALPHA = 0.45            # smoothing factor (0..1) - higher = more responsive

        # require sustained large-area for immediate land (consecutive frames or time)
        self.area_confirm_count = 0
        self.AREA_CONFIRM_FRAMES = 6     # e.g. at 10 Hz, ~0.6 s

        # velocity smoothing for published cmd_vel
        self.prev_cmd = (0.0, 0.0, 0.0)
        self.CMD_ALPHA = 0.5             # smoothing for control outputs

        # descent limits
        self.MIN_DESCENT = -0.12         # fastest allowed descent (m/s)
        self.MAX_DESCENT = -0.02         # slowest descent while descending

        # simple robot predictor (linear) using odom history
        self.robot_prev = None           # (t, x, y)
        self.prediction_dt = 0.5         # seconds ahead to predict

        # State machine
        self.state = "takeoff"

        # Timing / flags
        self.align_start_time = None
        self.landed = False

        # Timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Start procedure (takeoff and climb)
        self.get_logger().info("Taking off...")
        self.takeoff_pub.publish(Empty())
        time.sleep(2.0)
        self.get_logger().info(f"Climbing to {self.altitude_target} m...")
        self.climb_to_altitude(self.altitude_target)
        self.state = "track"
        self.get_logger().info("Tracking red robot body with vision (GPS as backup).")

    def climb_to_altitude(self, target_altitude: float):
        climb_rate = 0.35
        duration = target_altitude / climb_rate
        # simple open-loop climb (keeps publishing until time elapsed)
        self.publish_cmd_vel(0.0, 0.0, climb_rate)
        time.sleep(duration)
        self.publish_cmd_vel(0.0, 0.0, 0.0)

    # --- Callbacks ---
    def drone_gps_callback(self, msg: NavSatFix):
        # set reference on first GPS message
        if self.ref_lat is None:
            self.ref_lat = msg.latitude
            self.ref_lon = msg.longitude
            self.ref_alt = msg.altitude
            self.get_logger().info(f"Reference GPS set: lat={self.ref_lat}, lon={self.ref_lon}, alt={self.ref_alt}")
            return
        # approximate meters per degree
        k_lat = 111320.0
        k_lon = 111320.0 * math.cos(math.radians(self.ref_lat))
        self.drone_x = (msg.longitude - self.ref_lon) * k_lon
        self.drone_y = (msg.latitude - self.ref_lat) * k_lat
        self.drone_z = msg.altitude - self.ref_alt

    def robot_pose_callback(self, msg: Odometry):
        # keep previous for velocity estimate
        if self.robot_x is not None and self.robot_y is not None:
            self.robot_prev = (time.time(), self.robot_x, self.robot_y)
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y


    def image_callback(self, img_msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return
        h, w = frame.shape[:2]
        self.last_frame_width = w
        self.last_frame_height = h

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 110, 60]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 110, 60]); upper2 = np.array([179, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.last_area = 0
            self.last_centroid = None
            self.vision_age = time.time() - self.last_detection_time if self.last_detection_time else 999.0
        else:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                self.last_centroid = (cx, cy)
                self.last_area = int(area)
                self.last_detection_time = time.time()
                self.vision_age = 0.0

                                # smoothing (EMA) for centroid and area
                if self.centroid_ema is None:
                    self.centroid_ema = (cx, cy)
                    self.area_ema = float(area)
                else:
                    ex, ey = self.centroid_ema
                    alpha = self.EMA_ALPHA
                    self.centroid_ema = (alpha * cx + (1 - alpha) * ex,
                                         alpha * cy + (1 - alpha) * ey)
                    self.area_ema = alpha * area + (1 - alpha) * self.area_ema

                # use smoothed values for logic / visualization
                smooth_cx = int(self.centroid_ema[0])
                smooth_cy = int(self.centroid_ema[1])
                smooth_area = int(self.area_ema)

                self.last_centroid = (smooth_cx, smooth_cy)
                self.last_area = smooth_area
                self.last_detection_time = time.time()
                self.vision_age = 0.0

                # draw smoothed centroid (replace previous draws if present)
                cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                cv2.circle(frame, (smooth_cx, smooth_cy), 6, (255, 0, 0), -1)
                cv2.putText(frame, f"Area(s): {smooth_area}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


                # --- Visualization ---
                cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)   # contour
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)         # centroid
                cv2.putText(frame, f"Area: {int(area)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show both windows
        cv2.imshow("Drone Camera", frame)
        cv2.imshow("Red Mask", mask)
        cv2.waitKey(1)


    # --- Utils ---
    def publish_cmd_vel(self, lx=0.0, ly=0.0, lz=0.0):
        twist = Twist()
        twist.linear = Vector3(x=float(lx), y=float(ly), z=float(lz))
        self.cmd_vel_pub.publish(twist)

    def control_loop(self):
        # update vision_age
        if self.last_detection_time:
            self.vision_age = time.time() - self.last_detection_time
        else:
            self.vision_age = 999.0

        if self.state == "done" or self.landed:
            # once done, ensure zero velocity and nothing else
            self.publish_cmd_vel(0.0, 0.0, 0.0)
            return

        use_vision = (self.last_centroid is not None) and (self.vision_age < 0.8)

        if self.state == "track":
            if use_vision:
                self.vision_track_and_maybe_land()
            else:
                self.gps_fallback_track()
        elif self.state == "land":
            if use_vision:
                self.vision_track_and_maybe_land()
            else:
                self.gps_fallback_land()

    def gps_fallback_track(self):
        if None not in (self.robot_x, self.robot_y, self.drone_x, self.drone_y):
            # --- Estimate robot velocity if possible ---
            pred_x, pred_y = self.robot_x, self.robot_y
            if self.robot_prev is not None:
                t_prev, x_prev, y_prev = self.robot_prev
                dt = max(1e-3, time.time() - t_prev)
                vx_robot = (self.robot_x - x_prev) / dt
                vy_robot = (self.robot_y - y_prev) / dt
                pred_x += vx_robot * self.prediction_dt
                pred_y += vy_robot * self.prediction_dt

            # --- Compute error vector ---
            dx = pred_x - self.drone_x
            dy = pred_y - self.drone_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # --- Proportional control toward predicted robot position ---
            k_xy = self.KP_XY
            vx = k_xy * dx
            vy = k_xy * dy

            # --- Clamp max horizontal speed ---
            speed = math.sqrt(vx ** 2 + vy ** 2)
            if speed > self.MAX_SPEED:
                factor = self.MAX_SPEED / speed
                vx *= factor
                vy *= factor

            # --- If close enough, switch to land ---
            if distance < 0.5:
                self.get_logger().info(f"GPS fallback: Near robot (d={distance:.2f} m) → switching to 'land' state")
                self.state = "land"

            # --- Publish command ---
            self.publish_cmd_vel(vx, vy, 0.0)
        else:
            self.get_logger().warn("GPS data incomplete. Holding position.")
            self.publish_cmd_vel(0.0, 0.0, 0.0)



    def gps_fallback_land(self):
        if self.drone_z is not None and self.drone_z <= (self.robot_height + self.safety_margin + 0.02):
            self.get_logger().info("Landing now (GPS fallback)...")
            self.publish_cmd_vel(0.0, 0.0, 0.0)
            self.land_pub.publish(Empty())
            self.land_pub.publish(Empty())
            self.state = "done"
            self.landed = True
        else:
            # Slowly descend and maintain position
            self.publish_cmd_vel(0.0, 0.0, -0.06)


    def vision_track_and_maybe_land(self):
        cx, cy = self.last_centroid
        w, h = self.last_frame_width, self.last_frame_height
        center_x, center_y = w / 2.0, h / 2.0
        err_x, err_y = cx - center_x, cy - center_y

        # normalized errors [-1,1]
        nx = err_x / (w / 2.0)
        ny = err_y / (h / 2.0)

        vy = -self.KP_XY * nx
        vx = -self.KP_XY * ny

        # deadzone
        if abs(err_x) < self.ALIGN_PIX_TOL:
            vy = 0.0
        if abs(err_y) < self.ALIGN_PIX_TOL:
            vx = 0.0

        # clamp horizontal speeds
        vx = max(-self.MAX_SPEED, min(self.MAX_SPEED, vx))
        vy = max(-self.MAX_SPEED, min(self.MAX_SPEED, vy))

        aligned = abs(err_x) < self.ALIGN_PIX_TOL and abs(err_y) < self.ALIGN_PIX_TOL

        # area-based sustained confirmation (avoid single-frame spikes)
        if self.last_area >= self.AREA_TOO_CLOSE:
            self.area_confirm_count += 1
        else:
            self.area_confirm_count = 0

        if self.area_confirm_count >= self.AREA_CONFIRM_FRAMES:
            # only after sustained frames do we consider immediate landing
            self.get_logger().info("AREA_TOO_CLOSE sustained -> immediate confirmed land")
            self.publish_cmd_vel(0.0, 0.0, 0.0)
            self.land_pub.publish(Empty())
            self.land_pub.publish(Empty())
            self.state = "done"
            self.landed = True
            return

        # handle alignment timing
        if aligned:
            if self.align_start_time is None:
                self.align_start_time = time.time()

            align_duration = time.time() - self.align_start_time

            # reduce horizontal speed while descending
            vx *= 0.35
            vy *= 0.35

            # if area suggests near, reduce to zero horizontal
            if self.last_area >= self.AREA_LAND_THRESHOLD:
                vx = 0.0
                vy = 0.0

            if align_duration > self.ALIGN_CONFIRM_TIME:
                # compute descent rate smoothly based on altitude/area
                lz = -0.03  # default gentle

                if self.drone_z is not None:
                    dz = self.drone_z - (self.robot_height + self.safety_margin)
                    if dz > 1.0:
                        lz = -0.12       # fast descent if high
                    elif dz > 0.6:
                        lz = -0.08
                    elif dz > 0.3:
                        lz = -0.05
                    elif dz > 0.15:
                        lz = -0.03
                    else:
                        lz = -0.015      # feather-touch near ground
                else:
                    lz = -0.04    
                
                alpha = 0.4
                prev_lz = self.prev_cmd[2]
                lz = alpha * lz + (1 - alpha) * prev_lz

                # If vision area indicates we are close enough and stable -> land
                area_cond = (self.last_area >= self.AREA_LAND_THRESHOLD)
                alt_cond = (self.drone_z is not None and self.drone_z <= (self.robot_height + self.safety_margin + 0.02))

                # publish smoothed velocities (we will low-pass them below)
                desired_cmd = (vx, vy, lz)
            else:
                desired_cmd = (vx, vy, 0.0)
        else:
            # reset align timer
            self.align_start_time = None
            desired_cmd = (vx, vy, 0.0)

        # optional: predict small robot movement and adjust vx/vy (if odom present)
        if None not in (self.robot_x, self.robot_y, self.robot_prev):
            # estimate robot velocity
            t_prev, x_prev, y_prev = self.robot_prev
            dt = max(1e-3, time.time() - t_prev)
            vx_robot = (self.robot_x - x_prev) / dt
            vy_robot = (self.robot_y - y_prev) / dt
            # predict robot position after prediction_dt
            pred_x = self.robot_x + vx_robot * self.prediction_dt
            pred_y = self.robot_y + vy_robot * self.prediction_dt
            # TODO: convert predicted robot pos into image pixel offset if needed
            # (this is optional and depends on your mapping from world->image)

        # low-pass filter on cmd to avoid jerks
        alpha = self.CMD_ALPHA
        px, py, pz = self.prev_cmd
        dx, dy, dz = desired_cmd
        sx = alpha * dx + (1 - alpha) * px
        sy = alpha * dy + (1 - alpha) * py
        sz = alpha * dz + (1 - alpha) * pz
        self.prev_cmd = (sx, sy, sz)

        # publish
        self.publish_cmd_vel(sx, sy, sz)

        # If aligned+stable and either area_cond or alt_cond -> final landing
        if (self.align_start_time is not None) and ((self.last_area >= self.AREA_LAND_THRESHOLD) or (self.drone_z is not None and self.drone_z <= (self.robot_height + self.safety_margin + 0.02))):
            # ensure stability window
            if time.time() - self.align_start_time > max(self.ALIGN_CONFIRM_TIME, 0.6):
                self.get_logger().info("Conditions met -> publishing land and holding.")
                self.publish_cmd_vel(0.0, 0.0, 0.0)
                self.land_pub.publish(Empty())
                self.land_pub.publish(Empty())
                self.state = "done"
                self.landed = True
                return

        self.get_logger().info(
            f"Track: err=({err_x:.0f},{err_y:.0f}), area={self.last_area}, cmd=({sx:.2f},{sy:.2f},{sz:.3f}), z={self.drone_z}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DroneGPSVisionFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Emergency landing...")
        node.land_pub.publish(Empty())
        time.sleep(2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()