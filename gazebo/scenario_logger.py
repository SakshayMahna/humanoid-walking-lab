"""Run one command scenario on the G1 in Gazebo and log pelvis odometry to CSV.

Timed on SIMULATION time (odometry stamps): zero command for STAND s, then [VX, VY, WZ] until SECONDS.
The logger publishes /cmd_vel itself, so the command starts at exactly t = STAND in the log
(an external `ros2 topic pub` started with a wall-clock sleep lands 2-4 s late/early depending on startup).

Run inside the container:  python3 scenario_logger.py OUT.csv SECONDS STAND VX VY WZ
CSV columns: t, x, y, z, yaw, vx_body, vy_body, wz, cmd_vx, cmd_vy, cmd_wz, seen_vx (last vx on /cmd_vel)
"""

import csv
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class ScenarioLogger(Node):
    def __init__(self, path, seconds, stand, command):
        super().__init__("scenario_logger")
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["t", "x", "y", "z", "yaw", "vx_body", "vy_body", "wz", "cmd_vx", "cmd_vy", "cmd_wz", "seen_vx"])
        self.seconds, self.stand, self.command = seconds, stand, command
        self.t0 = None
        self.last_log = -1.0
        self.last_cmd = -1.0
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.seen_vx = float("nan")   # last vx actually seen on /cmd_vel (from any publisher)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Odometry, "/g1/pelvis/odometry", self.on_odom, 50)

    def on_cmd(self, msg):
        self.seen_vx = msg.linear.x

    def on_odom(self, msg):
        t_abs = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9   # simulation time
        if self.t0 is None:
            self.t0 = t_abs
        t = t_abs - self.t0
        cmd = self.command if t >= self.stand else (0.0, 0.0, 0.0)

        if t - self.last_cmd >= 0.1:   # 10 Hz, like `ros2 topic pub -r 10`
            self.last_cmd = t
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.angular.z = cmd
            self.cmd_pub.publish(twist)

        if t - self.last_log < 0.05:   # log at 20 Hz
            return
        self.last_log = t
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        p, v = msg.pose.pose.position, msg.twist.twist
        self.writer.writerow([f"{t:.3f}", f"{p.x:.4f}", f"{p.y:.4f}", f"{p.z:.4f}", f"{yaw:.4f}",
                              f"{v.linear.x:.4f}", f"{v.linear.y:.4f}", f"{v.angular.z:.4f}",
                              *(f"{c:.3f}" for c in cmd), f"{self.seen_vx:.3f}"])
        if t > self.seconds:
            self.cmd_pub.publish(Twist())   # stop
            self.file.close()
            raise SystemExit


def main():
    out, seconds, stand = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    command = tuple(float(v) for v in sys.argv[4:7])
    rclpy.init()
    node = ScenarioLogger(out, seconds, stand, command)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
