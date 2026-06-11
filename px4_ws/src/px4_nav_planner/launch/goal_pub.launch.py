from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Args (course만 유지 - goal_pub에서 사용)
    course = LaunchConfiguration('course', default='R')

    goals_yaml = PathJoinSubstitution([
        FindPackageShare('px4_nav_planner'), 'config', 'goals.yaml'
    ])

    goal_pub = Node(
        package='px4_nav_planner',
        executable='course_goal_publisher_node',
        name='course_goal_publisher',
        output='screen',
        parameters=[
            goals_yaml,
            {'default_course': course},
            # 호밍 라이트(필요 시만 변경)
            {'homing_duration_sec': 8.0,
             'homing_rate_hz': 20.0,
             'home_xyz': [0.0, 5.0, 2.5]},
            {'use_sim_time': True},
        ]
    )

    # astar_apf_planner2_node 사용 (노드의 기본값 사용)
    astar = Node(
        package='px4_nav_planner',
        executable='astar_apf_planner2_node',
        name='astar_apf_planner',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 고정 지연 8초
    delayed_astar = TimerAction(period=8.0, actions=[astar])

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='R'),
        goal_pub,
        delayed_astar,
    ])

