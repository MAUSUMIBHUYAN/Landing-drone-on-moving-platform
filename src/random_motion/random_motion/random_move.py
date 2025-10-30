# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist

# class StationaryNode(Node):
#     def __init__(self):
#         super().__init__('stationary_node')
#         self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
#         self.timer = self.create_timer(1.0, self.stop_robot)  # publish every 1 second
#         self.get_logger().info('StationaryNode')

#     def stop_robot(self):
#         msg = Twist()
#         msg.linear.x = 0.0
#         msg.linear.y = 0.0
#         msg.linear.z = 0.0
#         msg.angular.x = 0.0
#         msg.angular.y = 0.0
#         msg.angular.z = 0.0

#         self.publisher.publish(msg)
#         self.get_logger().info('Published stop command: robot stationary.')

# def main(args=None):
#     rclpy.init(args=args)
#     node = StationaryNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()




# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist

# class StraightMove(Node):
#     def __init__(self):
#         super().__init__('straight_move')
#         self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
#         self.timer = self.create_timer(1.0, self.move_straight)  # every 1 sec

#     def move_straight(self):
#         msg = Twist()

#         # Constant forward speed
#         msg.linear.x = 0.2   # adjust speed (m/s) as per your robot
#         msg.angular.z = 0.0  # no rotation → straight line

#         self.publisher.publish(msg)
#         self.get_logger().info(f"Moving straight with linear={msg.linear.x:.2f}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = StraightMove()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import math
import random

class RandomMove(Node):
    def __init__(self):
        super().__init__('random_move')

        # Publisher for cmd_vel
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer for publishing velocity
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.move_robot)

        # Drone position subscriber
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.create_subscription(PoseStamped, '/drone/pose', self.drone_pose_callback, 10)

        # Orbit parameters
        self.orbit_radius = 1.5      # distance from drone
        self.angular_speed = 0.5      # rad/s
        self.linear_speed = 0.5       # m/s

        # Current angle around drone
        self.theta = random.uniform(0, 2*math.pi)

        # Deviation for varied path
        self.max_dev = 0.5  # meters

        self.get_logger().info('RandomMove Node Initialized! Orbiting drone in real-time.')

    def drone_pose_callback(self, msg):
        self.drone_x = msg.pose.position.x
        self.drone_y = msg.pose.position.y

    def move_robot(self):
        msg = Twist()

        # Increment angle for circular motion
        self.theta += self.angular_speed * self.timer_period

        # Apply small random deviation to orbit radius
        dev = random.uniform(-self.max_dev, self.max_dev)

        # Compute target position around drone
        target_x = self.drone_x + (self.orbit_radius + dev) * math.cos(self.theta)
        target_y = self.drone_y + (self.orbit_radius + dev) * math.sin(self.theta)

        # Compute simple circular velocity command
        msg.linear.x = self.linear_speed
        msg.angular.z = self.angular_speed

        # Publish velocity
        self.pub.publish(msg)

        self.get_logger().info(
            f'Orbiting drone: drone_x={self.drone_x:.2f}, drone_y={self.drone_y:.2f}, target_x={target_x:.2f}, target_y={target_y:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = RandomMove()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()









