"""Run an ONNX policy on the G1 in Gazebo.

    ros2 launch g1_policy_runner policy.launch.py policy:=walk clock:=wall
    ros2 launch g1_policy_runner policy.launch.py policy:=unitree clock:=sim gui:=false

Same order as Diego's g1_locomotion.launch.py: the policy node starts first, Gazebo (+ bridge + spawn) 2 s later.
Gazebo is started with Diego's g1_sim/sim_core.py, so the robot, world and bridge are identical for both policies.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    policy = LaunchConfiguration("policy").perform(context)
    clock = LaunchConfiguration("clock").perform(context)
    config = os.path.join(get_package_share_directory("g1_policy_runner"), "config", f"{policy}.yaml")

    policy_node = Node(
        package="g1_policy_runner",
        executable="policy_runner",
        name="policy_runner",
        output="screen",
        parameters=[
            config,
            {"clock": clock,                      # which timer drives the policy: "wall" or "sim"
             "use_sim_time": clock == "sim",      # node clock follows Gazebo's /clock (bridged by g1_sim)
             "policy_period": float(LaunchConfiguration("policy_period").perform(context))},
        ],
    )

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("g1_sim"), "g1_sim", "sim_core.py")),
        launch_arguments={
            "world": LaunchConfiguration("world"),
            "robot_model": LaunchConfiguration("robot_model"),
            "bridge_config": "g1_pos",            # position-command bridge (what "auto" picks for g1_free_roam)
            "spawn_z": LaunchConfiguration("spawn_z"),
            "gui": LaunchConfiguration("gui"),
            "run": "true",
        }.items(),
    )

    return [policy_node, TimerAction(period=2.0, actions=[sim])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("policy", default_value="walk", description="walk | unitree (config/<policy>.yaml)"),
        DeclareLaunchArgument("clock", default_value="wall", description="wall | sim: clock that drives the policy timer"),
        DeclareLaunchArgument("policy_period", default_value="0.02", description="s between policy steps (training: 0.02)"),
        DeclareLaunchArgument("gui", default_value="true", description="Start the Gazebo GUI"),
        DeclareLaunchArgument("robot_model", default_value="g1_free_roam",
                              description="g1_sim robot alias, or a path to a model.sdf (e.g. a gain variant)"),
        DeclareLaunchArgument("world", default_value="empty", description="g1_sim world alias"),
        DeclareLaunchArgument("spawn_z", default_value="0.80", description="Spawn height (m), as in Diego's launch"),
        OpaqueFunction(function=launch_setup),
    ])
