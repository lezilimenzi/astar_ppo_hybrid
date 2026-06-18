#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool

class CourseGoalPublisher(Node):
    """호밍(home_xyz 이동) 후 선택 코스(R/L) 목표를 /planner/goal(latched)로 발행.
    호밍 중 OFFBOARD/ARM 요청 수행. /planner/course_select 로 코스 변경. 프레임: odom(spawn 기준 ENU)."""

    def __init__(self):
        super().__init__('course_goal_publisher')

        # Params (좌표·코스는 goals.yaml에서 덮어씀)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('default_course', 'R')
        self.declare_parameter('goal_R', [8.0, -8.2, 2.5])
        self.declare_parameter('goal_L', [-8.0, -8.2, 2.5])
        self.declare_parameter('spawn_enu', [0.0, 0.0, 0.83])  # = Gazebo 스폰 (sitl_run.sh)

        # 호밍 (시작점 이동)
        self.declare_parameter('enable_homing', True)
        self.declare_parameter('home_xyz', [0.0, 8.0, 2.5])
        self.declare_parameter('home_tolerance', 0.30)
        self.declare_parameter('home_timeout_sec', 20.0)
        self.declare_parameter('home_rate_hz', 20.0)
        self.declare_parameter('hold_after_homing', False)

        # OFFBOARD/ARM (호밍 중만 사용)
        self.declare_parameter('request_offboard_during_homing', True)
        self.declare_parameter('pre_setpoint_sec', 2.0)
        self.declare_parameter('mode_retry_sec', 1.0)
        self.declare_parameter('arm_retry_sec', 1.0)

        # Read
        self.frame_id = self.get_parameter('frame_id').value
        self.course = self.get_parameter('default_course').value
        if self.course not in ('R', 'L'):
            self.get_logger().warn(f'Invalid default_course "{self.course}", using "R"')
            self.course = 'R'

        try:
            goal_r_param = self.get_parameter('goal_R').value
            goal_l_param = self.get_parameter('goal_L').value
            if goal_r_param is None or goal_l_param is None:
                raise ValueError('goal_R/goal_L parameters must be provided via YAML or launch parameters.')
            self.goal_R = self._as_xyz(goal_r_param)
            self.goal_L = self._as_xyz(goal_l_param)
        except (ValueError, TypeError, IndexError) as e:
            self.get_logger().error(f'Failed to parse goal parameters: {e}')
            raise

        self.spawn_enu       = self._as_xyz(self.get_parameter('spawn_enu').value)
        self.enable_homing   = self.get_parameter('enable_homing').value
        home_xyz_param = self.get_parameter('home_xyz').value  # goals.yaml 필수
        self.home_xyz        = self._as_xyz(home_xyz_param)
        self.home_tol        = float(self.get_parameter('home_tolerance').value)
        self.home_timeout    = float(self.get_parameter('home_timeout_sec').value)
        self.home_rate_hz    = float(self.get_parameter('home_rate_hz').value)
        self.hold_after_homing = bool(self.get_parameter('hold_after_homing').value)
        self.request_offboard_during_homing = bool(self.get_parameter('request_offboard_during_homing').value)
        pre_setpoint_sec = float(self.get_parameter('pre_setpoint_sec').value)
        self.mode_retry_sec = float(self.get_parameter('mode_retry_sec').value)
        self.arm_retry_sec = float(self.get_parameter('arm_retry_sec').value)

        self.get_logger().info(
            f'spawn_enu={self.spawn_enu}, home_xyz={self.home_xyz} (ENU, MAVROS가 변환)'
        )

        # ---------------- State ----------------
        self.pose: PoseStamped = None
        self.have_pose = False
        self.homing_done = not self.enable_homing
        self.homing_start_time = None
        self._goal_published_without_homing = False  # enable_homing=False 시 1회 publish용
        self.mav_state = None
        self._pre_setpoint_count = 0
        self._pre_setpoint_needed = int(max(0, pre_setpoint_sec) * self.home_rate_hz)
        self._t_last_mode_req = 0.0
        self._t_last_arm_req = 0.0

        # QoS & I/O
        q_latched = QoSProfile(   # 목표 latching
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        q_mavros_sub = QoSProfile(   # 센서 구독 (BEST_EFFORT)
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        q_setpoint = QoSProfile(   # setpoint 스트리밍
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._on_pose, q_mavros_sub,
        )
        if self.request_offboard_during_homing and self.enable_homing:
            self.create_subscription(State, '/mavros/state', self._on_state, q_mavros_sub)
            self._cli_mode = self.create_client(SetMode, '/mavros/set_mode')
            self._cli_arm = self.create_client(CommandBool, '/mavros/cmd/arming')

        self.pub_goal = self.create_publisher(PoseStamped, '/planner/goal', q_latched)
        # RL 노드가 velocity 제어 시작 시점 판단
        self.pub_homing_done = self.create_publisher(Bool, '/planner/homing_done', q_latched)
        self.pub_setpoint = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', q_setpoint
        )
        self.sub_select = self.create_subscription(
            String, '/planner/course_select', self._on_select, 10
        )
        self.timer = self.create_timer(1.0 / max(1.0, self.home_rate_hz), self._tick)

        self.get_logger().info(
            f'Started course_goal_publisher | frame={self.frame_id}, '
            f'default_course={self.course}, homing={"on" if self.enable_homing else "off"}'
        )

    # ---------- Callbacks ----------
    def _on_pose(self, msg: PoseStamped):
        self.pose = msg
        self.have_pose = True

    def _on_state(self, msg: State):
        self.mav_state = msg

    def _on_select(self, msg: String):
        course = msg.data.strip().upper()
        if course in ('R', 'L'):
            self.course = course
            if self.homing_done:
                self.publish_goal(self.course)  # 호밍 완료 상태면 즉시 갱신
        else:
            self.get_logger().warn(f'Unknown course "{course}" (use "R" or "L")')

    # ---------- Helpers ----------
    def _as_xyz(self, arr):
        """(x,y,z) 튜플로 변환(부족하면 보정)"""
        if not arr or len(arr) == 0:
            raise ValueError('XYZ cannot be empty')
        if len(arr) == 1:
            return float(arr[0]), 0.0, 2.0
        if len(arr) == 2:
            return float(arr[0]), float(arr[1]), 2.0
        return float(arr[0]), float(arr[1]), float(arr[2])

    def _dist2D(self, ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    # ---------- Main loop ----------
    def _tick(self):
        # 0) 호밍 비활성화 시: 시작 시 goal + homing_done 1회 퍼블리시 (RL 등 velocity 제어와 함께 사용)
        if self.homing_done and not self.enable_homing and not self._goal_published_without_homing:
            self.publish_goal(self.course)
            self._publish_homing_done(True)
            self._goal_published_without_homing = True
            self.get_logger().info(f'Published goal for course {self.course} (homing disabled).')
            return

        # 1) 호밍 단계: (0,8,2)로 setpoint 퍼블리시
        if not self.homing_done and self.enable_homing:
            # 호밍 시작 시간 기록
            if self.homing_start_time is None:
                self.homing_start_time = self.get_clock().now()
                self.get_logger().info(
                    f'Homing to {self.home_xyz} (tol={self.home_tol} m, timeout={self.home_timeout}s)'
                )

            # setpoint: (home - spawn) ENU. MAVROS가 ENU→NED 변환
            hx, hy, hz = self.home_xyz
            sx, sy, sz = self.spawn_enu
            x, y, z = hx - sx, hy - sy, hz - sz

            reached = False
            if self.have_pose:
                rx, ry = self.pose.pose.position.x, self.pose.pose.position.y  # ENU
                if self._dist2D(rx, ry, x, y) <= self.home_tol:
                    reached = True

            self._publish_setpoint(x, y, z)

            # OFFBOARD/ARM 요청 (pre_setpoint 완료 후)
            if self.request_offboard_during_homing and self.mav_state is not None and self.mav_state.connected:
                if self._pre_setpoint_count < self._pre_setpoint_needed:
                    self._pre_setpoint_count += 1
                else:
                    now = time.time()
                    if self.mav_state.mode != 'OFFBOARD':
                        if now - self._t_last_mode_req >= self.mode_retry_sec and self._cli_mode.service_is_ready():
                            self._t_last_mode_req = now
                            req = SetMode.Request()
                            req.base_mode, req.custom_mode = 0, 'OFFBOARD'
                            self._cli_mode.call_async(req)
                            self.get_logger().info('Requested OFFBOARD.')
                    elif not self.mav_state.armed:
                        if now - self._t_last_arm_req >= self.arm_retry_sec and self._cli_arm.service_is_ready():
                            self._t_last_arm_req = now
                            req = CommandBool.Request()
                            req.value = True
                            self._cli_arm.call_async(req)
                            self.get_logger().info('Requested ARM.')

            # 종료 조건: 도달 or 타임아웃
            elapsed = (self.get_clock().now() - self.homing_start_time).nanoseconds / 1e9
            if reached or elapsed >= self.home_timeout:
                self.homing_done = True
                status = "reached" if reached else "timeout"
                self.get_logger().info(f'Homing {status}. Publishing course goal and homing_done...')
                self.publish_goal(self.course)
                self._publish_homing_done(True)
            return

        # 2) 호밍 완료 후: 필요 시에만 마지막 위치를 유지
        if self.homing_done and self.hold_after_homing:
            hx, hy, hz = self.home_xyz
            sx, sy, sz = self.spawn_enu
            self._publish_setpoint(hx - sx, hy - sy, hz - sz)

    def _publish_setpoint(self, x, y, z):
        sp = PoseStamped()
        sp.header.frame_id = self.frame_id
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.pose.position.x = float(x)
        sp.pose.position.y = float(y)
        sp.pose.position.z = float(z)
        # L 코스: 호밍 위치는 동일, 수평 방향만 ENU -x 를 보도록 (yaw = π, +x 가 yaw=0 일 때)
        if self.course == 'L':
            yaw = math.pi
            sp.pose.orientation.x = 0.0
            sp.pose.orientation.y = 0.0
            sp.pose.orientation.z = math.sin(yaw * 0.5)
            sp.pose.orientation.w = math.cos(yaw * 0.5)
        elif self.have_pose:
            sp.pose.orientation = self.pose.pose.orientation
        else:
            sp.pose.orientation.w = 1.0
        self.pub_setpoint.publish(sp)

    # ---------- Goal publish ----------
    def publish_goal(self, course: str):
        if course not in ('R', 'L'):
            self.get_logger().warn(f'Unknown course "{course}" (use "R" or "L")')
            return
        gx, gy, gz = self.goal_R if course == 'R' else self.goal_L
        sx, sy, sz = self.spawn_enu
        # goal: (goal - spawn) 상대 좌표. odom 로컬 프레임 기준 (spawn 기준 ENU)
        x, y, z = gx - sx, gy - sy, gz - sz

        goal = PoseStamped()
        goal.header.frame_id = self.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z
        goal.pose.orientation.w = 1.0

        self.pub_goal.publish(goal)
        self.get_logger().info(f'Published /planner/goal for {course}: ({x:.2f}, {y:.2f}, {z:.2f}) ENU')

    def _publish_homing_done(self, done: bool):
        msg = Bool()
        msg.data = bool(done)
        self.pub_homing_done.publish(msg)


def main():
    rclpy.init()
    node = CourseGoalPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
