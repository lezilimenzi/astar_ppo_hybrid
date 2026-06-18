from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='R'),
        goal_pub,
    ])

