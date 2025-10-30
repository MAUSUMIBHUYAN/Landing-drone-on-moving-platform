#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import cv2
import numpy as np
from cv_bridge import CvBridge
import math

class RedMarkerLanding(Node):
    def __init__(self):
        super().__init__('red_marker_landing')

        # ROS2 Publishers & Subscribers
        self.image_sub = self.create_subscription(Image, '/simple_drone/bottom/image_raw', self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10)

        self.bridge = CvBridge()

        # Drone Status
        self.taken_off = False
        self.reached_altitude = False
        self.searching_marker = False
        self.altitude_step = 0

        # Spiral search parameters
        self.spiral_radius = 0.2
        self.angle_step = math.pi / 8
        self.spiral_angle = 0
        self.spiral_timer = None

        # Tolerances
        self.center_tolerance = 20  
        self.altitude_velocity = 0.2  

        self.get_logger().info("Taking off...")
        self.takeoff_pub.publish(Empty())
        self.altitude_timer = self.create_timer(1.0, self.ascend_to_altitude)

    def ascend_to_altitude(self):
        """ Ascend to a target altitude in steps. """
        if self.altitude_step < 5:  # Approximate 3m altitude
            twist = Twist()
            twist.linear.z = 0.3
            self.cmd_vel_pub.publish(twist)
            self.altitude_step += 1
        else:
            self.cmd_vel_pub.publish(Twist())  # Stop ascent
            self.reached_altitude = True
            self.get_logger().info("Reached altitude. Starting marker detection.")
            self.altitude_timer.cancel()
            self.start_spiral_search()

    def image_callback(self, msg):
        """ Process bottom camera image to detect red marker. """
        if not self.reached_altitude:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width, _ = frame.shape

        # Convert to HSV and mask red color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            if self.searching_marker:
                self.spiral_timer.cancel()
                self.searching_marker = False
                self.get_logger().info("Marker found! Aligning...")

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            marker_center_x = x + w // 2
            marker_center_y = y + h // 2

            image_center_x = width // 2
            image_center_y = height // 2
            offset_x = marker_center_x - image_center_x
            offset_y = marker_center_y - image_center_y

            self.align_and_land(offset_x, offset_y)

    def start_spiral_search(self):
        """ Starts a spiral search movement. """
        self.searching_marker = True
        self.spiral_timer = self.create_timer(0.5, self.spiral_move)
        self.get_logger().info("Marker not detected. Starting spiral search...")

    def spiral_move(self):
        """ Moves the drone in a spiral pattern to search for the marker. """
        if not self.searching_marker:
            return

        twist = Twist()
        twist.linear.x = self.spiral_radius * math.cos(self.spiral_angle)
        twist.linear.y = self.spiral_radius * math.sin(self.spiral_angle)
        self.spiral_angle += self.angle_step
        
        # Gradually increase the radius
        if self.spiral_angle >= 2 * math.pi:
            self.spiral_radius += 0.1
            self.spiral_angle = 0

        self.cmd_vel_pub.publish(twist)

    def align_and_land(self, offset_x, offset_y):
        """ Aligns and lands the drone when centered over the marker. """
        twist = Twist()
        move_x = move_y = False

        if abs(offset_x) > self.center_tolerance:
            twist.linear.y = -0.2 if offset_x > 0 else 0.2  # Move left/right
            move_x = True

        if abs(offset_y) > self.center_tolerance:
            twist.linear.x = -0.2 if offset_y > 0 else 0.2  # Move forward/backward
            move_y = True

        if move_x or move_y:
            self.get_logger().info(f"Aligning: offset_x={offset_x}, offset_y={offset_y}")
            self.cmd_vel_pub.publish(twist)
        else:
            self.smooth_landing()

    def smooth_landing(self):
        """ Lands the drone smoothly. """
        self.get_logger().info("Marker centered. Landing now...")
        if self.searching_marker:
            self.spiral_timer.cancel()
            self.searching_marker = False
        
        twist = Twist()
        for _ in range(10):
            twist.linear.z = -self.altitude_velocity
            self.cmd_vel_pub.publish(twist)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.5))

        self.cmd_vel_pub.publish(Twist())
        self.land_pub.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    node = RedMarkerLanding()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
