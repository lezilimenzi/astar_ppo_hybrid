from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'px4_nav_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 🔹 launch 파일 설치
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # (옵션) config 파일도 있으면 같이
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'course_goal_publisher_node = px4_nav_planner.nodes.course_goal_publisher_node:main',
        ],
    },
)
