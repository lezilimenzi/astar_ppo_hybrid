#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, heapq, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

class AStarAPFPlanner(Node):
    def __init__(self):
        super().__init__('astar_apf_planner_node')

        # -----------------------
        # Parameters (ROS2)
        # -----------------------
        self.declare_parameter('occ_threshold', 50)           # Grid 점유 임계
        self.declare_parameter('allow_diag', True)
        self.declare_parameter('k_att', 1.0)                  # APF attractive
        self.declare_parameter('k_rep', 0.3)                  # APF repulsive
        self.declare_parameter('rep_range', 1.0)              # ρ0 - 증가: 더 먼 거리까지 장애물 고려
        self.declare_parameter('apf_activate_dist', 1.0)      # 근접 시 APF 가중↑
        self.declare_parameter('waypoint_tolerance', 0.4)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('grid_topic', '/local_grid')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('replan_period', 0.5)          # s
        self.declare_parameter('step_ahead', 0.6)             # setpoint 전진 거리[m]
        self.declare_parameter('z_setpoint_mode', 'hold_pose_z')  # hold_pose_z | fixed
        self.declare_parameter('z_fixed', 2.0)                # fixed일 때 사용
        self.declare_parameter('safe_distance', 0.2)          # LiDAR 최소 안전 거리[m]
        self.declare_parameter('min_step_ahead', 0.15)        # 장애물 근처 최소 step[m]

        # Local avoidance parameters
        self.declare_parameter('d_stop', 1.5)                 # LOCAL_AVOID 진입 거리 (m) - 증가: 더 일찍 회피
        self.declare_parameter('d_clear', 1.5)                # GLOBAL_FOLLOW 복귀 거리 (m) - 증가: 더 확실히 뚫렸을 때 복귀
        self.declare_parameter('step_forward', 0.3)           # LOCAL_AVOID 전진 거리 (m) - 감소: 더 작은 단위로 이동
        self.declare_parameter('yaw_tol', 0.087)             # heading 정렬 허용 오차 (rad, ~5deg)
        self.declare_parameter('pos_tol', 0.3)               # 로컬 타깃 근접 판단 오차 (m)
        self.declare_parameter('w_range', 0.85)                # 거리 가중치 (scoring) - 증가: 안전성 우선
        self.declare_parameter('w_align', 0.15)                # goal 정렬 가중치 (scoring) - 감소: goal 방향 선호도 감소
        self.declare_parameter('publish_setpoint', True)       # False: path만 publish (hybrid 시 rl_offboard가 velocity 사용)

        # Read params
        p = self.get_parameter
        self.occ_threshold       = p('occ_threshold').get_parameter_value().integer_value
        self.allow_diag          = p('allow_diag').get_parameter_value().bool_value
        self.k_att               = p('k_att').get_parameter_value().double_value
        self.k_rep               = p('k_rep').get_parameter_value().double_value
        self.rep_range           = p('rep_range').get_parameter_value().double_value
        self.apf_activate_dist   = p('apf_activate_dist').get_parameter_value().double_value
        self.waypoint_tolerance  = p('waypoint_tolerance').get_parameter_value().double_value
        self.global_frame        = p('global_frame').get_parameter_value().string_value
        self.grid_topic          = p('grid_topic').get_parameter_value().string_value
        self.scan_topic          = p('scan_topic').get_parameter_value().string_value
        self.replan_period       = p('replan_period').get_parameter_value().double_value
        self.step_ahead          = p('step_ahead').get_parameter_value().double_value
        self.z_setpoint_mode     = p('z_setpoint_mode').get_parameter_value().string_value
        self.z_fixed             = p('z_fixed').get_parameter_value().double_value
        self.safe_distance       = p('safe_distance').get_parameter_value().double_value
        self.min_step_ahead      = p('min_step_ahead').get_parameter_value().double_value

        # Local avoidance params
        self.d_stop              = p('d_stop').get_parameter_value().double_value
        self.d_clear             = p('d_clear').get_parameter_value().double_value
        self.step_forward        = p('step_forward').get_parameter_value().double_value
        self.yaw_tol             = p('yaw_tol').get_parameter_value().double_value
        self.pos_tol             = p('pos_tol').get_parameter_value().double_value
        self.w_range             = p('w_range').get_parameter_value().double_value
        self.w_align             = p('w_align').get_parameter_value().double_value
        self.publish_setpoint    = p('publish_setpoint').get_parameter_value().bool_value

        # -----------------------
        # State
        # -----------------------
        self.grid_data = None
        self.w = self.h = None
        self.res = None
        self.ox = self.oy = None
        self.have_grid = False

        self.pose: PoseStamped = None
        self.have_pose = False

        self.goal: PoseStamped = None
        self.have_goal = False

        self.scan: LaserScan = None
        self.have_scan = False

        self.global_path_xy = []  # [(x,y), ...]
        self.cur_wp_idx = 0

        # State machine for local avoidance
        self.state = "GLOBAL_FOLLOW"  # "GLOBAL_FOLLOW" or "LOCAL_AVOID"

        # Sector distances (front only - left/right/back are computed but not used)
        self.d_front = float('inf')

        # LOCAL_AVOID state variables
        self.yaw_target = None      # Target heading (absolute)
        self.local_target = None     # (x, y) local target position
        self.last_avoid_heading = None  # 이전에 선택한 회피 방향 (상대각도, rad)
        self.avoid_attempts = 0      # 같은 방향 재시도 횟수

        # -----------------------
        # QoS
        # -----------------------
        q_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        q_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        # Setpoint 발행용 - 스트리밍 데이터이므로 TRANSIENT_LOCAL 제거, RELIABLE + depth=10
        # course_goal_publisher_node와 동일한 QoS 사용
        q_setpoint = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # -----------------------
        # I/O
        # -----------------------
        self.sub_grid = self.create_subscription(
            OccupancyGrid, self.grid_topic, self.cb_grid, q_sensor)

        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.cb_pose, q_sensor)

        self.sub_goal = self.create_subscription(
            PoseStamped, '/planner/goal', self.cb_goal, q_latched)

        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self.cb_scan, q_sensor)

        self.pub_path = self.create_publisher(Path, '/planner/path_raw', q_latched)
        self.pub_setpoint = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', q_setpoint)

        # Control / planning timers
        self.timer_ctrl = self.create_timer(1.0/20.0, self.tick_control)  # 20Hz
        self.timer_replan = self.create_timer(self.replan_period, self.tick_replan)

        # 핵심 시작 로그 (1회만 출력)
        self.get_logger().info('A* + APF hybrid planner (ROS2) started.')

    # -----------------------
    # Callbacks
    # -----------------------
    def cb_grid(self, msg: OccupancyGrid):
        self.h = msg.info.height
        self.w = msg.info.width
        self.res = msg.info.resolution
        self.ox = msg.info.origin.position.x
        self.oy = msg.info.origin.position.y
        self.grid_data = np.array(msg.data, dtype=np.int16).reshape(self.h, self.w)
        self.have_grid = True
        # 디버깅용: 필요 시에만 DEBUG 레벨로 확인
        self.get_logger().debug(
            f'Received grid: origin=({self.ox:.2f},{self.oy:.2f}) size=({self.w},{self.h}) res={self.res:.3f}'
        )

    def cb_pose(self, msg: PoseStamped):
        self.pose = msg
        self.have_pose = True

    def cb_goal(self, msg: PoseStamped):
        # 새 goal 수신은 핵심 이벤트이므로 INFO 유지
        self.get_logger().info(
            f"Received goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})"
        )
        self.goal = msg
        self.have_goal = True
        # 즉시 리플랜
        self.plan_global_path()

    def cb_scan(self, msg: LaserScan):
        self.scan = msg
        self.have_scan = True
        self.update_sector_distances(msg)

    # -----------------------
    # Helpers: map/world
    # -----------------------
    def xy_to_ij(self, x: float, y: float):
        if not self.have_grid: return None
        j = int((x - self.ox) / self.res)   # col (x)
        i = int((y - self.oy) / self.res)   # row (y)
        if 0 <= i < self.h and 0 <= j < self.w:
            return i, j
        return None

    def ij_to_xy(self, i: int, j: int):
        x = self.ox + (j + 0.5) * self.res
        y = self.oy + (i + 0.5) * self.res
        return x, y

    def is_occ(self, i: int, j: int):
        # out-of-bounds는 점유 취급
        if i < 0 or i >= self.h or j < 0 or j >= self.w: return True
        return self.grid_data[i, j] >= self.occ_threshold

    # -----------------------
    # A* (global)
    # -----------------------
    def astar_neighbors(self, i, j):
        steps = [(1,0),( -1,0),(0,1),(0,-1)]
        if self.allow_diag:
            steps += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for di, dj in steps:
            ni, nj = i+di, j+dj
            if 0 <= ni < self.h and 0 <= nj < self.w and not self.is_occ(ni, nj):
                cost = math.hypot(di, dj)
                yield ni, nj, cost

    @staticmethod
    def h_octile(i, j, gi, gj):
        dx = abs(i - gi); dy = abs(j - gj)
        return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)

    def astar(self, start_ij, goal_ij):
        si, sj = start_ij; gi, gj = goal_ij
        if (si, sj) == (gi, gj):
            return [(si, sj)]

        openq = []
        heapq.heappush(openq, (self.h_octile(si, sj, gi, gj), 0.0, (si, sj)))
        gscore = {(si, sj): 0.0}
        parent = {(si, sj): None}
        visited = set()

        while openq:
            f, gc, (i, j) = heapq.heappop(openq)
            if (i, j) in visited:
                continue
            visited.add((i, j))
            if (i, j) == (gi, gj):
                break
            for ni, nj, step in self.astar_neighbors(i, j):
                ng = gc + step
                if ng < gscore.get((ni, nj), 1e18):
                    gscore[(ni, nj)] = ng
                    parent[(ni, nj)] = (i, j)
                    heapq.heappush(openq, (ng + self.h_octile(ni, nj, gi, gj), ng, (ni, nj)))

        if (gi, gj) not in parent:
            return None

        # reconstruct
        path = []
        node = (gi, gj)
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def plan_global_path(self):
        if not (self.have_grid and self.have_pose and self.have_goal):
            self.get_logger().debug(
                f'plan_global_path: missing data grid={self.have_grid} pose={self.have_pose} goal={self.have_goal}'
            )
            return

        sx, sy = self.pose.pose.position.x, self.pose.pose.position.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        # 경로 계획 정보는 DEBUG 로만 출력 (필요 시 활성화)
        self.get_logger().debug(f'plan_global_path: start=({sx:.2f},{sy:.2f}) goal=({gx:.2f},{gy:.2f})')

        s_ij = self.xy_to_ij(sx, sy)
        g_ij = self.xy_to_ij(gx, gy)

        if s_ij is None or g_ij is None:
            grid_x_max = self.ox + self.w * self.res
            grid_y_max = self.oy + self.h * self.res
            self.get_logger().warn(
                f'Start/Goal outside grid. '
                f'start=({sx:.2f},{sy:.2f}) start_ij={s_ij}, '
                f'goal=({gx:.2f},{gy:.2f}) goal_ij={g_ij}, '
                f'grid_origin=({self.ox:.2f},{self.oy:.2f}) grid_max=({grid_x_max:.2f},{grid_y_max:.2f}) '
                f'grid_size=({self.w},{self.h}) res={self.res:.3f}'
            )
            return

        if self.is_occ(*g_ij):
            self.get_logger().warn(f'Goal cell occupied at {g_ij}.')
            return

        self.get_logger().debug(f'Running A* from {s_ij} to {g_ij}')
        path_ij = self.astar(s_ij, g_ij)
        if path_ij is None:
            self.get_logger().warn('A* path not found.')
            return

        self.global_path_xy = [self.ij_to_xy(i, j) for (i, j) in path_ij]
        # 마지막 waypoint를 goal로 설정 (goal을 명확히 목표로 하기 위해)
        if self.global_path_xy:
            self.global_path_xy[-1] = (gx, gy)
        self.cur_wp_idx = 0

        # publish Path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.global_frame
        for (x, y) in self.global_path_xy:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = self._z_setpoint_current()
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)
        self.get_logger().info(f'Global path planned: {len(self.global_path_xy)} wps')

    # -----------------------
    # Local: APF hybrid → position setpoint
    # -----------------------
    def _advance_wp_if_close(self):
        if not self.global_path_xy: return
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        while self.cur_wp_idx < len(self.global_path_xy)-1:
            wx, wy = self.global_path_xy[self.cur_wp_idx]
            if math.hypot(wx - rx, wy - ry) < self.waypoint_tolerance:
                self.cur_wp_idx += 1
            else:
                break

    def _current_wp(self):
        if not self.global_path_xy: return None
        return self.global_path_xy[self.cur_wp_idx]

    def _min_obs_dist_from_scan(self):
        if not self.have_scan or self.scan is None:
            return float('inf')
        arr = np.array(self.scan.ranges, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return float('inf')
        return float(np.min(arr))

    def update_sector_distances(self, scan: LaserScan):
        """LiDAR 360°를 front 섹터로 요약 (d_front만 사용)."""
        if scan is None:
            return

        angle = scan.angle_min
        d_front = []

        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r) or r <= scan.range_min:
                angle += scan.angle_increment
                continue

            # 각도 normalize (-pi, pi)
            a = self.normalize_angle(angle)

            # front 섹터만 계산 (-45°~45°)
            if -math.pi/4 <= a <= math.pi/4:
                d_front.append(r)

            angle += scan.angle_increment

        def safe_min(lst):
            return min(lst) if lst else float('inf')

        self.d_front = safe_min(d_front)

    @staticmethod
    def normalize_angle(a):
        """(-pi, pi] 범위로 정규화."""
        return math.atan2(math.sin(a), math.cos(a))

    def _compute_apf_force_world(self, target_x, target_y):
        # Attractive
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        yaw = self._yaw_from_quat()
        dx, dy = (target_x - rx), (target_y - ry)
        F_att_x = self.k_att * dx
        F_att_y = self.k_att * dy

        F_rep_x = 0.0; F_rep_y = 0.0
        if self.have_scan and self.scan is not None:
            ang = self.scan.angle_min
            for r in self.scan.ranges:
                if not math.isfinite(r) or r < 0.05:
                    ang += self.scan.angle_increment
                    continue
                if r <= self.rep_range:
                    # obst in robot frame
                    ox_r = r * math.cos(ang)
                    oy_r = r * math.sin(ang)
                    # to world
                    ox = rx + ox_r * math.cos(yaw) - oy_r * math.sin(yaw)
                    oy = ry + ox_r * math.sin(yaw) + oy_r * math.cos(yaw)
                    rho = math.hypot(ox - rx, oy - ry)
                    if rho < 1e-3:
                        ang += self.scan.angle_increment
                        continue
                    coeff = self.k_rep * (1.0 / rho - 1.0 / self.rep_range) * (1.0 / (rho ** 2))
                    if coeff > 0.0:
                        ux = (rx - ox) / rho
                        uy = (ry - oy) / rho
                        F_rep_x += coeff * ux
                        F_rep_y += coeff * uy
                ang += self.scan.angle_increment
        return (F_att_x + F_rep_x, F_att_y + F_rep_y)

    def _yaw_from_quat(self):
        q = self.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _z_setpoint_current(self):
        if self.z_setpoint_mode == 'fixed':
            return self.z_fixed
        # hold current pose z
        return self.pose.pose.position.z if self.have_pose else self.z_fixed

    def tick_control(self):
        # 최소 조건: pose (goal은 경로 계획에만 필요)
        if not self.have_pose:
            self.get_logger().debug(
                f'skip control: no pose yet grid={self.have_grid} pose={self.have_pose} goal={self.have_goal}'
            )
            return

        # goal 없을 때는 현재 위치 유지 (OFFBOARD failsafe 방지)
        if not self.have_goal:
            self.get_logger().debug('No goal yet, holding position')
            self.publish_hold_setpoint()
            return

        # 글로벌 안전 버블: LiDAR 최소 거리 < safe_distance 이면 응급 회피(backoff)
        min_obs = self._min_obs_dist_from_scan()
        if min_obs < self.safe_distance:
            self._emergency_backoff(min_obs)
            return

        # State machine: check for state transition
        if self.state == "GLOBAL_FOLLOW":
            # Check if we need to enter LOCAL_AVOID
            if self.d_front < self.d_stop:
                self.get_logger().warn(
                    f"[GLOBAL_FOLLOW] Obstacle ahead ({self.d_front:.2f} m) → LOCAL_AVOID"
                )
                self.enter_local_avoid()
            elif self.global_path_xy:
                # Continue with normal path following
                self.follow_global_path()
                return
            else:
                # No path yet, try to plan path if we have grid data
                if self.have_grid:
                    self.plan_global_path()
                # Hold position while planning (OFFBOARD failsafe 방지)
                self.publish_hold_setpoint()
                return

        elif self.state == "LOCAL_AVOID":
            self.run_local_avoid()
            return

    def follow_global_path(self):
        """GLOBAL_FOLLOW 상태: 기존 A* 경로 추종 로직."""
        if not self.global_path_xy:
            # 경로가 없으면 현재 위치 유지 (OFFBOARD failsafe 방지)
            self.publish_hold_setpoint()
            return

        self._advance_wp_if_close()
        wp = self._current_wp()
        if wp is None:
            # 이론적으로 발생하지 않아야 하지만 방어 코드 (경로 재계획 중 발생 가능)
            self.get_logger().warn('no current waypoint despite path data, resetting path')
            self.global_path_xy = []  # 경로 초기화하여 다음 tick에서 재계획 유도
            self.publish_hold_setpoint()
            return

        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        wx, wy = wp

        # Goal 종료 조건 (마지막 WP 근접)
        if self.cur_wp_idx == len(self.global_path_xy)-1:
            final_dist = math.hypot(wx - rx, wy - ry)
            if final_dist < self.waypoint_tolerance:
                self.get_logger().info(
                    f'Goal reached! Final distance: {final_dist:.3f} m, '
                    f'goal=({wx:.2f}, {wy:.2f}), current=({rx:.2f}, {ry:.2f})'
                )
                sp = PoseStamped()
                sp.header.stamp = self.get_clock().now().to_msg()
                sp.header.frame_id = self.global_frame
                sp.pose.position.x = wx
                sp.pose.position.y = wy
                sp.pose.position.z = self._z_setpoint_current()
                sp.pose.orientation.w = 1.0
                self.pub_setpoint.publish(sp)
                return

        # 전역(단순 추종) 방향: waypoint 방향
        gx_wp, gy_wp = (wx - rx), (wy - ry)
        gnorm_wp = math.hypot(gx_wp, gy_wp)
        if gnorm_wp > 1e-6:
            gx_wp /= gnorm_wp; gy_wp /= gnorm_wp

        # Goal 방향도 고려 (goal을 "까먹지 않도록")
        gx_goal, gy_goal = (self.goal.pose.position.x - rx), (self.goal.pose.position.y - ry)
        gnorm_goal = math.hypot(gx_goal, gy_goal)
        if gnorm_goal > 1e-6:
            gx_goal /= gnorm_goal; gy_goal /= gnorm_goal

        # Waypoint와 Goal 방향을 블렌딩 (goal에 가까울수록 goal 방향 가중↑)
        goal_dist = gnorm_goal
        goal_weight = max(0.1, min(0.5, 5.0 / max(goal_dist, 1.0)))  # goal에 가까울수록 0.5까지 증가
        gx = (1.0 - goal_weight) * gx_wp + goal_weight * gx_goal
        gy = (1.0 - goal_weight) * gy_wp + goal_weight * gy_goal
        gnorm = math.hypot(gx, gy)
        if gnorm > 1e-6:
            gx /= gnorm; gy /= gnorm

        min_obs = self._min_obs_dist_from_scan()

        alpha = 0.0

        if min_obs > self.apf_activate_dist:
            # 충분히 떨어져 있으면 단순 방향 추종
            Fx, Fy = gx, gy
        else:
            # 장애물에 가까우면 APF 사용 (attractive는 goal, repulsive는 LiDAR)
            apfx, apfy = self._compute_apf_force_world(self.goal.pose.position.x, self.goal.pose.position.y)
            # blend: 가까울수록 APF 영향↑
            alpha = max(0.5, min(1.0, (self.apf_activate_dist / max(min_obs, 0.05))))
            Fx = (1.0 - alpha) * gx + alpha * apfx
            Fy = (1.0 - alpha) * gy + alpha * apfy

        # Force → 로컬 setpoint (lookahead)
        normF = math.hypot(Fx, Fy)
        if normF < 1e-6:
            Fx, Fy = gx, gy
            normF = 1.0
        Fx /= normF; Fy /= normF
        # 장애물 근접 시 step 크기 자동 축소
        step = self._adaptive_step(min_obs)
        tx = rx + Fx * step
        ty = ry + Fy * step

        self.get_logger().debug(
            f'setpoint tx={tx:.2f} ty={ty:.2f} wp_idx={self.cur_wp_idx} min_obs={min_obs:.2f} alpha={alpha:.2f}'
        )

        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = self.global_frame
        sp.pose.position.x = tx
        sp.pose.position.y = ty
        sp.pose.position.z = self._z_setpoint_current()
        sp.pose.orientation.w = 1.0
        self.pub_setpoint.publish(sp)

    # -----------------------
    # Local Avoidance Logic
    # -----------------------
    def enter_local_avoid(self):
        """LOCAL_AVOID 진입 시 최적 heading / 로컬 타깃을 계산."""
        self.state = "LOCAL_AVOID"
        self.yaw_target = None
        self.local_target = None

        # 1) 최적 heading 계산 (이전 방향이 막혀있으면 제외)
        exclude_heading = None
        if self.avoid_attempts >= 2 and self.last_avoid_heading is not None:
            # 같은 방향을 2번 이상 시도했는데 막혀있으면, 그 방향 제외
            exclude_heading = self.last_avoid_heading
            self.get_logger().warn(
                f"[LOCAL_AVOID] Excluding previous heading {math.degrees(exclude_heading):.1f} deg (blocked {self.avoid_attempts} times)"
            )

        result = self.compute_best_heading(exclude_heading=exclude_heading)

        # 제외 방향 때문에 유효한 방향이 없으면, 제외 없이 다시 시도
        if result is None and exclude_heading is not None:
            self.get_logger().warn(
                "[LOCAL_AVOID] No valid heading found with exclusion. Trying without exclusion."
            )
            result = self.compute_best_heading(exclude_heading=None)
            # 제외 없이도 없으면 재시도 횟수 리셋 (막다른 길일 수 있음)
            if result is None:
                self.avoid_attempts = 0

        if result is None:
            best_heading = None
            best_distance = None
        else:
            best_heading, best_distance = result

        if best_heading is None:
            self.get_logger().warn(
                "[LOCAL_AVOID] No valid heading found. Attempting backward motion."
            )
            # 앞뒤 전부 막혀있으면 뒤로 회피
            current_yaw = self._yaw_from_quat()
            self.yaw_target = self.normalize_angle(current_yaw + math.pi)
            rx = self.pose.pose.position.x
            ry = self.pose.pose.position.y
            self.local_target = (
                rx - self.step_forward * 0.5 * math.cos(current_yaw),
                ry - self.step_forward * 0.5 * math.sin(current_yaw),
            )
            self.get_logger().info(
                f"[LOCAL_AVOID] Backward target=({self.local_target[0]:.2f}, {self.local_target[1]:.2f})"
            )
            return

        current_yaw = self._yaw_from_quat()
        self.yaw_target = self.normalize_angle(current_yaw + best_heading)

        # 이전 방향과 비교하여 재시도 횟수 관리
        if self.last_avoid_heading is not None:
            heading_diff = abs(self.normalize_angle(best_heading - self.last_avoid_heading))
            if heading_diff < math.pi / 6:  # ±30도 이내면 같은 방향으로 간주
                self.avoid_attempts += 1
            else:
                # 다른 방향 선택했으므로 재시도 횟수 리셋
                self.avoid_attempts = 0
        else:
            self.avoid_attempts = 0

        self.last_avoid_heading = best_heading

        # 2) 그 방향으로 한 스텝 전진할 로컬 타깃점 계산
        # 중요: 선택한 방향의 실제 거리를 확인하고, 그 거리보다 작은 거리로만 전진
        rx = self.pose.pose.position.x
        ry = self.pose.pose.position.y

        # 선택한 방향의 실제 거리 확인 (안전 마진 포함)
        safe_margin = 0.2  # 20cm 안전 마진
        max_step = best_distance - safe_margin if best_distance is not None and math.isfinite(best_distance) else self.step_forward

        # 실제 거리가 step_forward보다 작으면, 실제 거리만큼만 전진
        actual_step = min(self.step_forward, max(0.1, max_step))  # 최소 0.1m는 전진

        if actual_step < self.step_forward * 0.5:
            # 실제 거리가 너무 가까우면 경고
            self.get_logger().warn(
                f"[LOCAL_AVOID] Selected direction distance ({best_distance:.2f} m) is too close. "
                f"Reducing step from {self.step_forward:.2f} m to {actual_step:.2f} m"
            )

        self.local_target = (
            rx + actual_step * math.cos(self.yaw_target),
            ry + actual_step * math.sin(self.yaw_target),
        )

        # LOCAL_AVOID 진입 로그
        self.get_logger().info(
            f"[LOCAL_AVOID] best_heading(rel)={math.degrees(best_heading):.1f} deg, "
            f"distance={best_distance:.2f} m, "
            f"yaw_target={math.degrees(self.yaw_target):.1f} deg, "
            f"local_target=({self.local_target[0]:.2f}, {self.local_target[1]:.2f}), "
            f"step={actual_step:.2f} m, attempts={self.avoid_attempts}"
        )

    def run_local_avoid(self):
        """LOCAL_AVOID 상태에서 회전 → 전진 수행."""
        if self.yaw_target is None or self.local_target is None:
            # 아직 타깃 없으면 다시 계산 시도
            self.enter_local_avoid()
            # enter_local_avoid가 실패할 수 있으므로 현재 위치 유지 (OFFBOARD failsafe 방지)
            if self.yaw_target is None or self.local_target is None:
                self.publish_hold_setpoint()
            return

        current_yaw = self._yaw_from_quat()
        rx = self.pose.pose.position.x
        ry = self.pose.pose.position.y

        # 1) 우선 heading 맞추기 (회전)
        yaw_err = self.normalize_angle(self.yaw_target - current_yaw)
        if abs(yaw_err) > self.yaw_tol:
            # 제자리에서 회전: 위치는 고정, yaw만 타깃으로 보내기
            self.publish_yaw_setpoint(self.yaw_target)
            return

        # 2) yaw 정렬이 끝났으면 로컬 타깃까지 전진 setpoint
        dx = self.local_target[0] - rx
        dy = self.local_target[1] - ry
        dist = math.hypot(dx, dy)

        # 앞쪽이 너무 다시 막혔으면 다시 best heading 계산 (이전 방향 제외)
        if self.d_front < self.d_stop * 0.8:
            self.get_logger().warn(
                f"[LOCAL_AVOID] Front blocked again ({self.d_front:.2f} m < {self.d_stop * 0.8:.2f} m). "
                f"Recomputing heading (excluding previous direction)."
            )
            # 이전 방향이 막혀있으므로 재시도 횟수 증가
            self.avoid_attempts += 1
            self.enter_local_avoid()
            # enter_local_avoid 후에도 타깃이 없으면 현재 위치 유지 (OFFBOARD failsafe 방지)
            if self.yaw_target is None or self.local_target is None:
                self.publish_hold_setpoint()
            return

        if dist > self.pos_tol:
            # 로컬 타깃을 향한 setpoint 퍼블리시
            self.publish_position_setpoint(
                self.local_target[0],
                self.local_target[1],
                self._z_setpoint_current(),
                self.yaw_target,
            )
        else:
            # 로컬 타깃 도달 → 전방이 충분히 뚫려 있다면 GLOBAL_FOLLOW 복귀 시도
            if self.d_front > self.d_clear:
                self.get_logger().info(
                    "[LOCAL_AVOID] Local target reached and front clear → GLOBAL_FOLLOW"
                )
                self.state = "GLOBAL_FOLLOW"
                # 회피 상태 변수 리셋
                self.last_avoid_heading = None
                self.avoid_attempts = 0
                # 즉시 경로 재계획
                self.plan_global_path()
            else:
                # 아직 앞이 막혀 있으면 다시 회피 방향 탐색 (이전 방향 제외)
                self.get_logger().warn(
                    "[LOCAL_AVOID] Local target reached but front still blocked. Recomputing heading (excluding previous direction)."
                )
                # 이전 방향이 막혀있으므로 재시도 횟수 증가
                self.avoid_attempts += 1
                self.enter_local_avoid()
                # enter_local_avoid 후에도 타깃이 없으면 현재 위치 유지 (OFFBOARD failsafe 방지)
                if self.yaw_target is None or self.local_target is None:
                    self.publish_hold_setpoint()

    def compute_best_heading(self, exclude_heading=None):
        """
        LiDAR 360°와 goal 방향을 이용해
        - goal 방향과 유사하고
        - 측정 거리가 먼
        방향을 선택.

        Args:
            exclude_heading: 제외할 방향 (상대각도, rad). None이면 모든 방향 고려.

        반환: (best_angle, best_distance) 튜플 또는 None
            - best_angle: 로봇 기준 상대각도(0=정면), rad
            - best_distance: 해당 방향의 실제 거리, m
        """
        if not self.have_scan or self.scan is None or self.goal is None or not self.have_pose:
            return None

        scan = self.scan

        # 현재 위치, goal 기준 yaw 계산 (맵 좌표계)
        rx = self.pose.pose.position.x
        ry = self.pose.pose.position.y
        gx = self.goal.pose.position.x
        gy = self.goal.pose.position.y
        yaw_goal_world = math.atan2(gy - ry, gx - rx)

        # 로봇 heading 기준 goal 상대각도
        current_yaw = self._yaw_from_quat()
        goal_rel = self.normalize_angle(yaw_goal_world - current_yaw)

        angle = scan.angle_min
        best_score = -float('inf')
        best_angle = None
        best_distance = None

        # maxRange normalize용
        r_max = scan.range_max

        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r) or r <= scan.range_min:
                angle += scan.angle_increment
                continue

            a_rel = self.normalize_angle(angle)  # 라이다 기준 상대 각도

            # 제외할 방향이면 스킵
            if exclude_heading is not None:
                heading_diff = abs(self.normalize_angle(a_rel - exclude_heading))
                if heading_diff < math.pi / 6:  # ±30도 이내면 제외
                    angle += scan.angle_increment
                    continue

            # range term (0~1 사이 정도로 normalizing)
            # 안전마진: rep_range 이상 떨어진 곳만 고려
            if r < self.rep_range:
                angle += scan.angle_increment
                continue

            range_term = min(r / r_max, 1.0)

            # goal alignment term: cos(각도 차이) - goal 방향 우선
            align_term = math.cos(a_rel - goal_rel)

            # goal 방향에 매우 가까우면 선호도 증가 (가중치 감소)
            angle_diff = abs(self.normalize_angle(a_rel - goal_rel))
            if angle_diff < math.pi / 6:  # goal 방향 ±30도 이내
                align_term = align_term * 2.0 + 0.3  # 3.0 -> 2.0, 0.5 -> 0.3: goal 방향 선호도 감소

            score = self.w_range * range_term + self.w_align *  align_term

            if score > best_score:
                best_score = score
                best_angle = a_rel
                best_distance = r

            angle += scan.angle_increment

        if best_angle is None:
            return None

        return (best_angle, best_distance)

    def publish_hold_setpoint(self):
        """현재 위치/자세를 그대로 유지하는 setpoint."""
        if not self.have_pose:
            return

        p = self.pose.pose
        current_yaw = self._yaw_from_quat()
        self.publish_position_setpoint(
            p.position.x,
            p.position.y,
            self._z_setpoint_current(),
            current_yaw,
        )

    def publish_yaw_setpoint(self, yaw_target: float):
        """위치는 그대로, yaw만 바꾸는 setpoint."""
        if not self.have_pose:
            return

        p = self.pose.pose
        self.publish_position_setpoint(
            p.position.x,
            p.position.y,
            self._z_setpoint_current(),
            yaw_target,
        )

    def publish_position_setpoint(self, x, y, z, yaw):
        """/mavros/setpoint_position/local 에 PoseStamped 퍼블리시."""
        if not self.publish_setpoint:
            return
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = self.global_frame
        sp.pose.position.x = x
        sp.pose.position.y = y
        sp.pose.position.z = z

        # yaw -> quaternion
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        sp.pose.orientation.z = qz
        sp.pose.orientation.w = qw

        self.pub_setpoint.publish(sp)

    # 주기적 리플랜 (맵 변화 대응)
    def tick_replan(self):
        if self.state == "GLOBAL_FOLLOW" and self.have_grid and self.have_pose and self.have_goal:
            self.get_logger().debug('tick_replan: periodic replan triggered')
            self.plan_global_path()

    def _emergency_backoff(self, min_obs: float):
        """
        안전 버블(safe_distance) 안으로 들어왔을 때
        - APF의 repulsive 성분만 이용해서
        - 장애물에서 멀어지는 방향으로 한 스텝 물러나기.
        """
        if not self.have_pose:
            return
        rx = self.pose.pose.position.x
        ry = self.pose.pose.position.y

        # target을 현재 위치로 주면 attractive는 0 → repulsive만 남음
        Fx, Fy = self._compute_apf_force_world(rx, ry)
        normF = math.hypot(Fx, Fy)
        if normF < 1e-6:
            # 주변이 전부 막혀있으면 goal 반대 방향으로 회피 시도
            if self.have_goal:
                # goal 반대 방향
                gx = self.goal.pose.position.x - rx
                gy = self.goal.pose.position.y - ry
                gnorm = math.hypot(gx, gy)
                if gnorm > 1e-6:
                    Fx = -gx / gnorm
                    Fy = -gy / gnorm
                else:
                    # goal도 현재와 같으면 현재 heading의 반대 방향
                    current_yaw = self._yaw_from_quat()
                    Fx = -math.cos(current_yaw)
                    Fy = -math.sin(current_yaw)
            else:
                # goal 없으면 HOLD
                self.get_logger().warn(
                    f'Emergency backoff: no valid direction (min_obs={min_obs:.2f} m), holding.'
                )
                self.publish_hold_setpoint()
                return
        else:
            Fx /= normF
            Fy /= normF

        # 응급 회피 스텝: 너무 크지 않게 제한
        step = max(self.min_step_ahead, min(self.step_ahead * 0.5, 0.3))
        tx = rx + Fx * step
        ty = ry + Fy * step

        self.get_logger().warn(
            f'Emergency backoff: min_obs={min_obs:.2f} m < {self.safe_distance:.2f} m, '
            f'moving to ({tx:.2f}, {ty:.2f})'
        )
        self.publish_position_setpoint(
            tx, ty, self._z_setpoint_current(), self._yaw_from_quat()
        )

    def _adaptive_step(self, min_obs_dist: float) -> float:
        """
        장애물과의 최소 거리(min_obs_dist)에 따라 step_ahead를 축소.
        주의: min_obs_dist는 반드시 safe_distance 이상이어야 함 (호출부에서 체크)
        - 멀리 있을 때: 원래 step_ahead 유지
        - 2m 이내: 0.5배
        - 1m 이내: 0.25배 (최소 min_step_ahead 보장)
        """
        step = self.step_ahead
        if not math.isfinite(min_obs_dist):
            return step
        # 1m 미만: 더 강하게 축소
        if min_obs_dist < 1.0:
            return max(self.min_step_ahead, step * 0.25)
        # 2m 미만: 보통 수준 축소
        if min_obs_dist < 2.0:
            return max(self.min_step_ahead, step * 0.5)
        # 그 외: 원래 step 유지
        return step


def main():
    rclpy.init()
    node = AStarAPFPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
