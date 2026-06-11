from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_path

def generate_launch_description():
    # 패키지 share 경로
    pkg_share = FindPackageShare('px4_nav_perception')
    # install 공간의 실제 경로(파일 읽기용)
    pkg_dir = get_package_share_path('px4_nav_perception')

    # 파라미터/rviz 설정 파일은 Substitution 경로로 넘김
    params   = PathJoinSubstitution([pkg_share, 'config', 'grid_fuser.yaml'])
    rviz_cfg = PathJoinSubstitution([pkg_share, 'rviz', 's3_local_mapping.rviz'])

    # URDF 실제 파일 경로를 읽어서 문자열로 주입
    urdf_file = pkg_dir / 'urdf' / 'iris_rplidar_depth.urdf'
    robot_description_str = urdf_file.read_text()

    use_sim_time = {'use_sim_time': True}

    # 1) URDF를 TF로 브로드캐스트 (base_link -> rplidar_link, camera_link 등)
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_str,  # ★ 문자열 직접 주입
            'use_sim_time': True
        }],
    )

    # 2) MAVROS Odom -> TF(odom->base_link) 브리지
    odom_to_tf = Node(
        package='px4_nav_perception',
        executable='odom_to_tf',
        name='odom_to_tf',
        output='screen',
        parameters=[
            use_sim_time,
            {
                'odom_topic': '/mavros/local_position/odom',
                'pose_topic': '/mavros/local_position/pose',
                'parent_frame': 'odom',
                'child_frame': 'base_link',
            }
        ],
        respawn=True, respawn_delay=1.0
    )

    # 3) OccupancyGrid 생성 노드
    grid_fuser = Node(
        package='px4_nav_perception',
        executable='grid_fuser',
        name='grid_fuser',
        output='screen',
        parameters=[params, use_sim_time],
        respawn=True, respawn_delay=1.0
    )

    # 4) RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],   # rviz 설정 파일 없으면 이 인자 제거
        output='screen',
        parameters=[use_sim_time],
    )

    return LaunchDescription([
        robot_state_pub,
        odom_to_tf,
        grid_fuser,
        rviz,
    ])
