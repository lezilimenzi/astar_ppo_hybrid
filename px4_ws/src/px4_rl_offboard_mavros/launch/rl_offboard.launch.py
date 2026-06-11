#!/usr/bin/env python3
"""RL Offboard 노드 실행용 launch 파일.
course_goal_publisher가 /planner/goal에 R/L 코스 목표를 publish.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    course = LaunchConfiguration('course', default='R')

    goals_yaml = PathJoinSubstitution([
        FindPackageShare('px4_nav_planner'), 'config', 'goals.yaml'
    ])

    params_file = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params.yaml'
    ])
    params_astar_overlay = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params_astar.yaml'
    ])
    params_ppo_overlay = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params_ppo.yaml'
    ])
    planner_only = LaunchConfiguration('planner_only', default='false')
    rl_only = LaunchConfiguration('rl_only', default='false')

    # homing 활성화: goal_pub가 position setpoint로 homing 후 /planner/homing_done publish
    # RL은 homing_done 수신 후 velocity 제어 시작 (방안 A)
    goal_pub = Node(
        package='px4_nav_planner',
        executable='course_goal_publisher_node',
        name='course_goal_publisher',
        output='screen',
        parameters=[
            goals_yaml,
            {'default_course': course},
            {'enable_homing': True},
            {'use_sim_time': True},
        ],
    )

    rl_node = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}],
    )
    rl_node_planner_only = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, params_astar_overlay, {'use_sim_time': True}],
    )
    rl_node_rl_only = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, params_ppo_overlay, {'use_sim_time': True}],
    )
    planner_only_cond = PythonExpression(["'", planner_only, "' == 'true'"])
    rl_only_cond = PythonExpression(["'", planner_only, "' != 'true' and '", rl_only, "' == 'true'"])
    default_cond = PythonExpression(["'", planner_only, "' != 'true' and '", rl_only, "' != 'true'"])

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='R',
                             description='R(ight) 또는 L(eft) 코스'),
        DeclareLaunchArgument('planner_only', default_value='false',
                             description='true: global planner(A*) 전용, RL 비활성화'),
        DeclareLaunchArgument('rl_only', default_value='false',
                             description='true: RL 전용, global planner 비활성화'),
        goal_pub,
        GroupAction([rl_node], condition=IfCondition(default_cond)),
        GroupAction([rl_node_planner_only], condition=IfCondition(planner_only_cond)),
        GroupAction([rl_node_rl_only], condition=IfCondition(rl_only_cond)),
    ])
