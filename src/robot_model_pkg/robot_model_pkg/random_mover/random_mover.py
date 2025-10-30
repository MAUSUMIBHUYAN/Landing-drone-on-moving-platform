#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random
import time

class RandomMover(Node):
    def __init__(self):
        super().__init__('random_mover')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(1.0, self.move_robot)  # change every 1s

    def move_robot(self):
        twist = Twist()

        # Random linear velocity forward/backward
        twist.linear.x = random.uniform(0.1, 0.5)  # 0.1~0.5 m/s

        # Random angular velocity
        twist.angular.z = random.uniform(-1.0, 1.0)  # -1~1 rad/s

        self.pub.publish(twist)
        self.get_logger().info(f"Linear: {twist.linear.x:.2f}, Angular: {twist.angular.z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = RandomMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
