#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import heapq
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan


class AStarAPFPlanner(Node):
    """
    A* global path + APF local steering + LOCAL_AVOID + ESCAPE wall-follow hybrid planner.

    Added improvements:
      (A) Strong no-return:
          - visited heatmap penalty + visited mask hard-block (optional).
      (B) Local avoid candidate generation:
          - scan-based heading + grid-sampling fallback/override (robust near walls/corners).
      (C) Safety gate:
          - every setpoint candidate must pass grid raycast + clearance (EDT) checks.
    """

    def __init__(self):
        super().__init__('astar_apf_planner_node')

        # -----------------------
        # Parameters
        # -----------------------
        # Grid / A*
        self.declare_parameter('occ_threshold', 50)
        self.declare_parameter('allow_diag', True)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('grid_topic', '/local_grid')
        self.declare_parameter('replan_period', 0.5)

        # APF
        self.declare_parameter('k_att', 0.9)
        self.declare_parameter('k_rep', 0.5)
        self.declare_parameter('rep_range', 1.0)
        self.declare_parameter('apf_activate_dist', 1.0)

        # Tracking / step
        self.declare_parameter('waypoint_tolerance', 0.4)
        self.declare_parameter('step_ahead', 0.6)
        self.declare_parameter('min_step_ahead', 0.15)
        self.declare_parameter('safe_distance', 0.2)      # very close -> emergency

        # Z control
        self.declare_parameter('z_setpoint_mode', 'hold_pose_z')  # fixed / hold_pose_z
        self.declare_parameter('z_fixed', 2.0)
        self.declare_parameter('z_lock_enable', True)
        self.declare_parameter('z_slew_rate_max', 0.5)
        self.declare_parameter('z_outlier_threshold', 0.5)

        # Topics
        self.declare_parameter('scan_topic', '/scan')

        # Local avoid trigger
        self.declare_parameter('d_stop', 2.0)
        self.declare_parameter('d_clear', 1.5)
        self.declare_parameter('step_forward', 0.3)
        self.declare_parameter('yaw_tol', 0.087)
        self.declare_parameter('pos_tol', 0.3)

        # Heading scoring weights
        self.declare_parameter('w_range', 0.60)
        self.declare_parameter('w_align', 0.15)
        self.declare_parameter('w_clear', 0.15)
        self.declare_parameter('w_visited', 0.60)

        # Stuck / Escape
        self.declare_parameter('stuck_time_horizon', 2.0)
        self.declare_parameter('stuck_progress_eps', 0.05)
        self.declare_parameter('no_heading_limit', 5)

        self.declare_parameter('escape_follow_dist', 0.7)
        self.declare_parameter('escape_step', 0.6)
        self.declare_parameter('escape_kn', 1.2)
        self.declare_parameter('escape_release_margin', 0.2)

        self.declare_parameter('escape_make_subgoal', True)
        self.declare_parameter('escape_subgoal_steps', 14)
        self.declare_parameter('escape_subgoal_min_edt_ratio', 0.8)

        # -----------------------
        # (A) Strong no-return params
        # -----------------------
        self.declare_parameter('visited_keep_secs', 25.0)
        self.declare_parameter('visited_cell_decay', 0.98)
        self.declare_parameter('visited_penalty_radius', 2)
        self.declare_parameter('visited_mark_stride', 1)

        # HARD no-return mask (recent trail exclusion)
        self.declare_parameter('no_return_hard_block', True)
        self.declare_parameter('no_return_mask_secs', 12.0)     # consider last N seconds as "do not enter"
        self.declare_parameter('no_return_mask_radius', 2)      # cells radius
        self.declare_parameter('no_return_allow_if_stuck', True)

        # -----------------------
        # (C) Safety check params
        # -----------------------
        self.declare_parameter('segment_check_step', 0.10)      # meters along segment
        self.declare_parameter('min_clearance', 0.30)           # meters (EDT)
        self.declare_parameter('goal_ray_block_dist', 1.2)      # meters (scan: goal-direction blocked close)

        # -----------------------
        # (B) Grid-sampling local avoid params
        # -----------------------
        self.declare_parameter('local_grid_sample_enable', True)
        self.declare_parameter('local_grid_sample_only_when_blocked', True)
        self.declare_parameter('local_sample_angle_deg', 15.0)  # angle step for radial sampling
        self.declare_parameter('local_sample_radii', [0.35, 0.55, 0.75, 0.95])  # meters
        self.declare_parameter('local_sample_forward_bias_deg', 90.0)  # sample within ± this deg around goal_rel

        # if scan-based best heading points into "blocked goal direction" or unsafe, allow grid sampler to override
        self.declare_parameter('local_grid_override_if_goal_blocked', True)

        p = self.get_parameter
        self.occ_threshold = p('occ_threshold').get_parameter_value().integer_value
        self.allow_diag = p('allow_diag').get_parameter_value().bool_value
        self.global_frame = p('global_frame').get_parameter_value().string_value
        self.grid_topic = p('grid_topic').get_parameter_value().string_value
        self.replan_period = p('replan_period').get_parameter_value().double_value

        self.k_att = p('k_att').get_parameter_value().double_value
        self.k_rep = p('k_rep').get_parameter_value().double_value
        self.rep_range = p('rep_range').get_parameter_value().double_value
        self.apf_activate_dist = p('apf_activate_dist').get_parameter_value().double_value

        self.waypoint_tolerance = p('waypoint_tolerance').get_parameter_value().double_value
        self.step_ahead = p('step_ahead').get_parameter_value().double_value
        self.min_step_ahead = p('min_step_ahead').get_parameter_value().double_value
        self.safe_distance = p('safe_distance').get_parameter_value().double_value

        self.z_setpoint_mode = p('z_setpoint_mode').get_parameter_value().string_value
        self.z_fixed = p('z_fixed').get_parameter_value().double_value
        self.z_lock_enable = p('z_lock_enable').get_parameter_value().bool_value
        self.z_slew_rate_max = p('z_slew_rate_max').get_parameter_value().double_value
        self.z_outlier_threshold = p('z_outlier_threshold').get_parameter_value().double_value

        self.scan_topic = p('scan_topic').get_parameter_value().string_value

        self.d_stop = p('d_stop').get_parameter_value().double_value
        self.d_clear = p('d_clear').get_parameter_value().double_value
        self.step_forward = p('step_forward').get_parameter_value().double_value
        self.yaw_tol = p('yaw_tol').get_parameter_value().double_value
        self.pos_tol = p('pos_tol').get_parameter_value().double_value

        self.w_range = p('w_range').get_parameter_value().double_value
        self.w_align = p('w_align').get_parameter_value().double_value
        self.w_clear = p('w_clear').get_parameter_value().double_value
        self.w_visited = p('w_visited').get_parameter_value().double_value

        self.stuck_time_horizon = p('stuck_time_horizon').get_parameter_value().double_value
        self.stuck_progress_eps = p('stuck_progress_eps').get_parameter_value().double_value
        self.no_heading_limit = p('no_heading_limit').get_parameter_value().integer_value

        self.escape_follow_dist = p('escape_follow_dist').get_parameter_value().double_value
        self.escape_step = p('escape_step').get_parameter_value().double_value
        self.escape_kn = p('escape_kn').get_parameter_value().double_value
        self.escape_release_margin = p('escape_release_margin').get_parameter_value().double_value

        self.escape_make_subgoal = p('escape_make_subgoal').get_parameter_value().bool_value
        self.escape_subgoal_steps = p('escape_subgoal_steps').get_parameter_value().integer_value
        self.escape_subgoal_min_edt_ratio = p('escape_subgoal_min_edt_ratio').get_parameter_value().double_value

        self.visited_keep_secs = p('visited_keep_secs').get_parameter_value().double_value
        self.visited_cell_decay = p('visited_cell_decay').get_parameter_value().double_value
        self.visited_penalty_radius = p('visited_penalty_radius').get_parameter_value().integer_value
        self.visited_mark_stride = p('visited_mark_stride').get_parameter_value().integer_value

        self.no_return_hard_block = p('no_return_hard_block').get_parameter_value().bool_value
        self.no_return_mask_secs = p('no_return_mask_secs').get_parameter_value().double_value
        self.no_return_mask_radius = p('no_return_mask_radius').get_parameter_value().integer_value
        self.no_return_allow_if_stuck = p('no_return_allow_if_stuck').get_parameter_value().bool_value

        self.segment_check_step = p('segment_check_step').get_parameter_value().double_value
        self.min_clearance = p('min_clearance').get_parameter_value().double_value
        self.goal_ray_block_dist = p('goal_ray_block_dist').get_parameter_value().double_value

        self.local_grid_sample_enable = p('local_grid_sample_enable').get_parameter_value().bool_value
        self.local_grid_sample_only_when_blocked = p('local_grid_sample_only_when_blocked').get_parameter_value().bool_value
        self.local_sample_angle_deg = p('local_sample_angle_deg').get_parameter_value().double_value
        self.local_sample_radii = [float(x) for x in p('local_sample_radii').value]
        self.local_sample_forward_bias_deg = p('local_sample_forward_bias_deg').get_parameter_value().double_value
        self.local_grid_override_if_goal_blocked = p('local_grid_override_if_goal_blocked').get_parameter_value().bool_value

        # -----------------------
        # State: grid / edt
        # -----------------------
        self.grid_data = None
        self.w = self.h = None
        self.res = None
        self.ox = self.oy = None
        self.have_grid = False

        self.edt = None
        self.have_edt = False

        # visited memory (no-return)
        self.visited_grid = None                  # float heatmap
        self.recent_cells = deque()               # (time, i, j) used for heat
        self.trail_cells = deque()                # (time, i, j) used for hard mask
        self.visited_mask = None                  # bool mask (recent forbidden area)

        self._ctrl_tick_count = 0

        # -----------------------
        # State: pose/goal/scan
        # -----------------------
        self.pose: PoseStamped = None
        self.have_pose = False

        self.goal: PoseStamped = None
        self.have_goal = False

        self.scan: LaserScan = None
        self.have_scan = False
        self.d_front = float('inf')
        self.min_scan = float('inf')

        # global path
        self.global_path_xy = []
        self.cur_wp_idx = 0

        # state machine
        self.state = "GLOBAL_FOLLOW"

        # local avoid vars
        self.yaw_target = None
        self.local_target = None
        self.last_avoid_heading = None
        self.avoid_attempts = 0
        self._no_heading_count = 0

        # stuck detector
        self._prog_hist = deque()

        # escape vars
        self.hand = None
        self.hit_point_dist = None
        self.subgoal = None

        # z lock/slew
        self._z_setpoint_locked = None
        self._z_setpoint_last = None
        self._z_lock_initialized = False
        self._last_control_time = None

        # QoS
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
        q_setpoint = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # I/O
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

        # timers
        self.timer_ctrl = self.create_timer(1.0 / 20.0, self.tick_control)
        self.timer_replan = self.create_timer(self.replan_period, self.tick_replan)

        self.get_logger().info('A*+APF hybrid planner started (Strong no-return + Grid-sampling avoid + Safety gate).')

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

        if self.visited_grid is None or self.visited_grid.shape != (self.h, self.w):
            self.visited_grid = np.zeros((self.h, self.w), dtype=np.float32)
        if self.visited_mask is None or self.visited_mask.shape != (self.h, self.w):
            self.visited_mask = np.zeros((self.h, self.w), dtype=bool)

        self.edt = self._compute_edt_bfs()
        self.have_edt = (self.edt is not None)

        # update mask from trail (cheap)
        self._refresh_visited_mask()

    def cb_pose(self, msg: PoseStamped):
        self.pose = msg
        self.have_pose = True

        # Z lock 초기화 (첫 pose 수신 시, hold_pose_z 모드에서만)
        if self.z_lock_enable and (not self._z_lock_initialized) and self.z_setpoint_mode == 'hold_pose_z':
            z_initial = float(msg.pose.position.z)
            # 초기 z 값이 합리적인 범위인지 확인 (0.5 ~ 10.0 m)
            if 0.5 <= z_initial <= 10.0:
                self._z_setpoint_locked = z_initial
                self._z_setpoint_last = z_initial
                self._z_lock_initialized = True
                self.get_logger().info(f'Z lock initialized at {self._z_setpoint_locked:.2f} m')
            else:
                self.get_logger().warn(
                    f'Z lock: initial z value {z_initial:.2f} m out of range [0.5, 10.0], using z_fixed={self.z_fixed:.2f} m'
                )
                self._z_setpoint_locked = self.z_fixed
                self._z_setpoint_last = self.z_fixed
                self._z_lock_initialized = True

        self._ctrl_tick_count += 1
        if (self._ctrl_tick_count % max(1, self.visited_mark_stride)) == 0:
            self._mark_visited_current_cell()

    def cb_goal(self, msg: PoseStamped):
        self.goal = msg
        self.have_goal = True
        self._reset_escape()
        self.plan_global_path()
        self.get_logger().info(
            f"Goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})"
        )

    def cb_scan(self, msg: LaserScan):
        self.scan = msg
        self.have_scan = True
        self._update_scan_metrics(msg)

    # -----------------------
    # Map helpers
    # -----------------------
    def xy_to_ij(self, x: float, y: float):
        if not self.have_grid:
            return None
        j = int((x - self.ox) / self.res)
        i = int((y - self.oy) / self.res)
        if 0 <= i < self.h and 0 <= j < self.w:
            return i, j
        return None

    def ij_to_xy(self, i: int, j: int):
        x = self.ox + (j + 0.5) * self.res
        y = self.oy + (i + 0.5) * self.res
        return x, y

    def is_occ(self, i: int, j: int):
        if i < 0 or i >= self.h or j < 0 or j >= self.w:
            return True
        return self.grid_data[i, j] >= self.occ_threshold

    def _find_nearest_free_cell(self, gi: int, gj: int, max_radius: int = 10):
        if not self.is_occ(gi, gj):
            return (gi, gj)

        q = deque([(gi, gj, 0)])
        visited = set([(gi, gj)])
        neigh = [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while q:
            i, j, dist = q.popleft()
            if dist > max_radius:
                break
            for di, dj in neigh:
                ni, nj = i + di, j + dj
                if (ni, nj) in visited:
                    continue
                if ni < 0 or ni >= self.h or nj < 0 or nj >= self.w:
                    continue
                visited.add((ni, nj))
                if not self.is_occ(ni, nj):
                    return (ni, nj)
                q.append((ni, nj, dist + 1))
        return None

    # -----------------------
    # Strong no-return memory
    # -----------------------
    def _mark_visited_current_cell(self):
        if not (self.have_pose and self.have_grid and self.visited_grid is not None):
            return
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        ij = self.xy_to_ij(rx, ry)
        if ij is None:
            return
        i, j = ij
        now = self.get_clock().now()

        self.visited_grid[i, j] += 1.0
        self.recent_cells.append((now, i, j))

        # also add to trail for hard mask
        self.trail_cells.append((now, i, j))

        # prune trail queue
        horizon_ns = int(max(0.1, self.no_return_mask_secs) * 1e9)
        while self.trail_cells and (now - self.trail_cells[0][0]).nanoseconds > horizon_ns:
            self.trail_cells.popleft()

        self._refresh_visited_mask()

    def _refresh_visited_mask(self):
        if not (self.have_grid and self.visited_mask is not None):
            return
        self.visited_mask.fill(False)
        if not self.no_return_hard_block:
            return

        r = max(0, int(self.no_return_mask_radius))
        for _, i, j in self.trail_cells:
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.h and 0 <= nj < self.w:
                        self.visited_mask[ni, nj] = True

    def _visited_penalty_at_cell(self, ci: int, cj: int) -> float:
        if self.visited_grid is None:
            return 0.0
        r = max(0, int(self.visited_penalty_radius))
        s = 0.0
        cnt = 0
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.h and 0 <= nj < self.w:
                    s += float(self.visited_grid[ni, nj])
                    cnt += 1
        if cnt == 0:
            return 0.0
        return s / cnt

    def _hard_blocked_cell(self, i: int, j: int) -> bool:
        """Hard no-return mask check (optionally relaxed if stuck/escape)."""
        if not self.no_return_hard_block:
            return False
        if self.visited_mask is None:
            return False

        # If we are in ESCAPE, allow mask to be relaxed (escape is priority)
        if self.state.startswith("ESCAPE"):
            return False

        # If we detected stuck recently, optionally relax
        if self.no_return_allow_if_stuck:
            if self._is_stuck_recent():
                return False

        return bool(self.visited_mask[i, j])

    def _is_stuck_recent(self) -> bool:
        """Heuristic: if progress over horizon is too small."""
        if len(self._prog_hist) < 2:
            return False
        progress = self._prog_hist[0][1] - self._prog_hist[-1][1]
        return (progress < self.stuck_progress_eps)

    # -----------------------
    # EDT
    # -----------------------
    def _compute_edt_bfs(self):
        if not self.have_grid or self.grid_data is None:
            return None

        INF = 1e9
        dist = np.full((self.h, self.w), INF, dtype=np.float32)
        q = deque()

        occ = (self.grid_data >= self.occ_threshold)
        occ_idx = np.argwhere(occ)
        if occ_idx.size == 0:
            return dist * self.res

        for (i, j) in occ_idx:
            dist[i, j] = 0.0
            q.append((i, j))

        neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            i, j = q.popleft()
            nd = dist[i, j] + 1.0
            for di, dj in neigh:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.h and 0 <= nj < self.w and nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    q.append((ni, nj))

        return dist * float(self.res)

    def _edt_value_at(self, x: float, y: float) -> float:
        if not self.have_edt or self.edt is None:
            return float('inf')
        ij = self.xy_to_ij(x, y)
        if ij is None:
            return 0.0
        i, j = ij
        return float(self.edt[i, j])

    def _grad_edt_unit_at(self, x: float, y: float):
        if not self.have_edt or self.edt is None:
            return None
        ij = self.xy_to_ij(x, y)
        if ij is None:
            return None
        i, j = ij

        im1 = max(i - 1, 0); ip1 = min(i + 1, self.h - 1)
        jm1 = max(j - 1, 0); jp1 = min(j + 1, self.w - 1)

        d_x = (self.edt[i, jp1] - self.edt[i, jm1]) / max((jp1 - jm1), 1)
        d_y = (self.edt[ip1, j] - self.edt[im1, j]) / max((ip1 - im1), 1)

        gx = float(d_x)
        gy = float(d_y)
        n = math.hypot(gx, gy)
        if n < 1e-6:
            return None
        return (gx / n, gy / n)

    # -----------------------
    # A* global
    # -----------------------
    def astar_neighbors(self, i, j):
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.allow_diag:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for di, dj in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.h and 0 <= nj < self.w and not self.is_occ(ni, nj):
                # Hard no-return: do not plan through recent trail unless relaxed
                if self._hard_blocked_cell(ni, nj):
                    continue
                yield ni, nj, math.hypot(di, dj)

    @staticmethod
    def h_octile(i, j, gi, gj):
        dx = abs(i - gi); dy = abs(j - gj)
        return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)

    def astar(self, start_ij, goal_ij):
        si, sj = start_ij
        gi, gj = goal_ij
        if (si, sj) == (gi, gj):
            return [(si, sj)]

        openq = []
        heapq.heappush(openq, (self.h_octile(si, sj, gi, gj), 0.0, (si, sj)))
        gscore = {(si, sj): 0.0}
        parent = {(si, sj): None}
        closed = set()

        while openq:
            _, gc, (i, j) = heapq.heappop(openq)
            if (i, j) in closed:
                continue
            closed.add((i, j))
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

        path = []
        node = (gi, gj)
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def plan_global_path(self):
        if not (self.have_grid and self.have_pose and self.have_goal):
            return

        sx, sy = self.pose.pose.position.x, self.pose.pose.position.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        s_ij = self.xy_to_ij(sx, sy)
        g_ij = self.xy_to_ij(gx, gy)
        if s_ij is None or g_ij is None:
            self.get_logger().warn("Start/Goal outside grid -> cannot plan.")
            return

        if self.is_occ(*g_ij):
            free_g = self._find_nearest_free_cell(*g_ij, max_radius=10)
            if free_g is None:
                self.get_logger().warn("Goal cell occupied and no free nearby -> cannot plan.")
                return
            g_ij = free_g

        path_ij = self.astar(s_ij, g_ij)
        if path_ij is None:
            self.get_logger().warn("A* path not found.")
            return

        self.global_path_xy = [self.ij_to_xy(i, j) for (i, j) in path_ij]
        if self.global_path_xy:
            self.global_path_xy[-1] = (gx, gy)
        self.cur_wp_idx = 0

        self._publish_path(self.global_path_xy)
        self.get_logger().info(f'Global path planned: {len(self.global_path_xy)} wps')

    def _publish_path(self, pts_xy):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.global_frame
        z = self._z_setpoint_current()
        for (x, y) in pts_xy:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)

    # -----------------------
    # Scan metrics
    # -----------------------
    def _update_scan_metrics(self, scan: LaserScan):
        if scan is None:
            self.d_front = float('inf')
            self.min_scan = float('inf')
            return

        arr = np.array(scan.ranges, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        self.min_scan = float(np.min(arr)) if arr.size else float('inf')

        ang = scan.angle_min
        front = []
        for r in scan.ranges:
            if not math.isfinite(r) or r <= scan.range_min:
                ang += scan.angle_increment
                continue
            a = self.normalize_angle(ang)
            if -math.pi / 3 <= a <= math.pi / 3:
                front.append(r)
            ang += scan.angle_increment
        self.d_front = min(front) if front else float('inf')

    def _goal_ray_min_dist(self) -> float:
        if not (self.have_scan and self.scan is not None and self.have_pose and self.have_goal):
            return float('inf')
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        yaw_goal_world = math.atan2(gy - ry, gx - rx)
        yaw = self._yaw_from_quat()
        goal_rel = self.normalize_angle(yaw_goal_world - yaw)

        scan = self.scan
        ang = scan.angle_min
        m = float('inf')
        for r in scan.ranges:
            if math.isfinite(r) and r > scan.range_min:
                a_rel = self.normalize_angle(ang)
                if abs(self.normalize_angle(a_rel - goal_rel)) < (math.pi / 12):
                    m = min(m, r)
            ang += scan.angle_increment
        return m

    # -----------------------
    # Control tick
    # -----------------------
    def tick_control(self):
        if not self.have_pose:
            return

        if not self.have_goal:
            self.publish_hold_setpoint()
            return

        self._update_stuck_detector()

        if self.have_scan:
            if self.min_scan < self.safe_distance:
                self._emergency_backoff(self.min_scan)
                return

        if self.state == "GLOBAL_FOLLOW":
            self._global_follow_step()
        elif self.state == "LOCAL_AVOID":
            self._local_avoid_step()
        elif self.state == "ESCAPE_WF":
            self._escape_wall_follow_step()
        elif self.state == "ESCAPE_GO_SUBGOAL":
            self._escape_subgoal_step()
        else:
            self.publish_hold_setpoint()

    def tick_replan(self):
        if self.visited_grid is not None:
            self.visited_grid *= float(self.visited_cell_decay)

        if self.state == "GLOBAL_FOLLOW" and self.have_grid and self.have_pose and self.have_goal:
            self.plan_global_path()

    # -----------------------
    # GLOBAL follow
    # -----------------------
    def _advance_wp_if_close(self):
        if not self.global_path_xy:
            return
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        while self.cur_wp_idx < len(self.global_path_xy) - 1:
            wx, wy = self.global_path_xy[self.cur_wp_idx]
            if math.hypot(wx - rx, wy - ry) < self.waypoint_tolerance:
                self.cur_wp_idx += 1
            else:
                break

    def _current_wp(self):
        if not self.global_path_xy:
            return None
        return self.global_path_xy[self.cur_wp_idx]

    def _global_follow_step(self):
        d_check = min(self.d_front, self.min_scan)
        goal_ray = self._goal_ray_min_dist()
        if goal_ray < self.goal_ray_block_dist:
            d_check = min(d_check, goal_ray)

        if d_check < self.d_stop:
            self._enter_local_avoid()
            return

        if not self.global_path_xy:
            if self.have_grid:
                self.plan_global_path()
            if not self.global_path_xy:
                self.publish_hold_setpoint()
                return

        self._advance_wp_if_close()
        wp = self._current_wp()
        if wp is None:
            self.publish_hold_setpoint()
            return

        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        wx, wy = wp

        # goal reached
        if self.cur_wp_idx == (len(self.global_path_xy) - 1):
            if math.hypot(wx - rx, wy - ry) < self.waypoint_tolerance:
                self.get_logger().info("Goal reached -> hold at goal.")
                self.publish_position_setpoint(wx, wy, self._z_setpoint_current(), self._yaw_from_quat())
                return

        vx, vy = (wx - rx), (wy - ry)
        n = math.hypot(vx, vy)
        if n < 1e-6:
            self.publish_hold_setpoint()
            return
        ux_wp, uy_wp = (vx / n, vy / n)

        # APF blend near obstacles
        min_obs = self.min_scan
        if min_obs > self.apf_activate_dist:
            ux, uy = ux_wp, uy_wp
        else:
            apfx, apfy = self._compute_apf_force_world(wx, wy)
            an = math.hypot(apfx, apfy)
            if an < 1e-6:
                ux, uy = ux_wp, uy_wp
            else:
                apfx /= an; apfy /= an
                alpha = max(0.4, min(1.0, self.apf_activate_dist / max(min_obs, 0.05)))
                ux = (1.0 - alpha) * ux_wp + alpha * apfx
                uy = (1.0 - alpha) * uy_wp + alpha * apfy
                un = math.hypot(ux, uy)
                if un > 1e-6:
                    ux /= un; uy /= un
                else:
                    ux, uy = ux_wp, uy_wp

        step = self._adaptive_step(min_obs)
        tx = rx + step * ux
        ty = ry + step * uy

        # hard no-return check
        if self._candidate_hard_blocked(tx, ty):
            # try alternate angles
            tx2, ty2 = self._find_safe_candidate_angles(rx, ry, math.atan2(uy, ux), step)
            if tx2 is None:
                self.get_logger().warn("[GLOBAL] hard no-return blocks -> LOCAL_AVOID")
                self._enter_local_avoid()
                return
            tx, ty = tx2, ty2

        # safety gate
        if not self.is_segment_safe(rx, ry, tx, ty):
            self.get_logger().warn("[GLOBAL] candidate unsafe -> LOCAL_AVOID")
            self._enter_local_avoid()
            return

        self.publish_position_setpoint(tx, ty, self._z_setpoint_current(), math.atan2(uy, ux))

    def _candidate_hard_blocked(self, x: float, y: float) -> bool:
        if not (self.have_grid and self.no_return_hard_block):
            return False
        ij = self.xy_to_ij(x, y)
        if ij is None:
            return True
        i, j = ij
        return self._hard_blocked_cell(i, j)

    def _find_safe_candidate_angles(self, rx, ry, base_yaw, step):
        """Try angles to escape hard no-return and unsafe segments."""
        for deg in [15, -15, 25, -25, 35, -35, 45, -45, 60, -60, 90, -90]:
            a = base_yaw + math.radians(deg)
            tx = rx + step * math.cos(a)
            ty = ry + step * math.sin(a)
            if self._candidate_hard_blocked(tx, ty):
                continue
            if not self.is_segment_safe(rx, ry, tx, ty):
                continue
            return tx, ty
        return None, None

    # -----------------------
    # Local avoid (scan + grid sampling)
    # -----------------------
    def _enter_local_avoid(self):
        self.state = "LOCAL_AVOID"
        self.yaw_target = None
        self.local_target = None

        exclude = None
        if self.avoid_attempts >= 2 and self.last_avoid_heading is not None:
            exclude = self.last_avoid_heading
            self.get_logger().warn(
                f"[LOCAL_AVOID] Excluding previous heading {math.degrees(exclude):.1f} deg (blocked {self.avoid_attempts} times)"
            )

        # 1) scan-based best heading
        scan_choice = self._compute_best_heading_scan(exclude_heading=exclude)
        if scan_choice is None and exclude is not None:
            self.get_logger().warn("[LOCAL_AVOID] No valid heading found with exclusion. Trying without exclusion.")
            scan_choice = self._compute_best_heading_scan(exclude_heading=None)

        # 2) grid-sampling fallback/override
        use_grid = False
        goal_ray = self._goal_ray_min_dist()
        goal_blocked = (goal_ray < self.goal_ray_block_dist)

        if self.local_grid_sample_enable and self.have_grid:
            if not self.local_grid_sample_only_when_blocked:
                use_grid = True
            else:
                # use grid sampler if scan choice missing OR goal direction is blocked (common wall-crash) OR front too close
                if (scan_choice is None) or (goal_blocked and self.local_grid_override_if_goal_blocked) or (min(self.d_front, self.min_scan) < (0.9 * self.d_stop)):
                    use_grid = True

        if use_grid:
            grid_choice = self._compute_best_heading_grid_sample()
        else:
            grid_choice = None

        # choose winner by score (grid returns score too)
        chosen_rel = None
        chosen_dist = None

        if grid_choice is not None:
            (rel_g, dist_g, score_g) = grid_choice
        else:
            rel_g, dist_g, score_g = None, None, -1e18

        if scan_choice is not None:
            (rel_s, dist_s, score_s) = scan_choice
        else:
            rel_s, dist_s, score_s = None, None, -1e18

        if score_g > score_s:
            chosen_rel, chosen_dist = rel_g, dist_g
        else:
            chosen_rel, chosen_dist = rel_s, dist_s

        if chosen_rel is None:
            self._no_heading_count += 1
            self.get_logger().warn(
                f"[LOCAL_AVOID] No valid heading found (no_heading_count={self._no_heading_count}) -> ESCAPE"
            )
            self._enter_escape_mode()
            return

        self._no_heading_count = 0
        yaw = self._yaw_from_quat()
        self.yaw_target = self.normalize_angle(yaw + chosen_rel)

        # attempts / memory
        if self.last_avoid_heading is not None:
            if abs(self.normalize_angle(chosen_rel - self.last_avoid_heading)) < math.pi / 6:
                self.avoid_attempts += 1
            else:
                self.avoid_attempts = 0
        else:
            self.avoid_attempts = 0
        self.last_avoid_heading = chosen_rel

        rx = self.pose.pose.position.x
        ry = self.pose.pose.position.y

        safe_margin = 0.25
        if chosen_dist is not None and math.isfinite(chosen_dist):
            max_step = chosen_dist - safe_margin
        else:
            max_step = self.step_forward
        actual_step = min(self.step_forward, max(0.12, max_step))

        tx = rx + actual_step * math.cos(self.yaw_target)
        ty = ry + actual_step * math.sin(self.yaw_target)

        # hard no-return + safety
        if self._candidate_hard_blocked(tx, ty) or (not self.is_segment_safe(rx, ry, tx, ty)):
            self.get_logger().warn("[LOCAL] candidate blocked/unsafe -> ESCAPE")
            self._enter_escape_mode()
            return

        self.local_target = (tx, ty)
        self.get_logger().info(
            f"[LOCAL] chosen_rel={math.degrees(chosen_rel):.1f}deg target=({tx:.2f},{ty:.2f}) "
            f"attempts={self.avoid_attempts}"
        )

    def _local_avoid_step(self):
        if self.yaw_target is None or self.local_target is None:
            self._enter_local_avoid()
            if self.yaw_target is None or self.local_target is None:
                self.publish_hold_setpoint()
            return

        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        yaw = self._yaw_from_quat()

        yaw_err = self.normalize_angle(self.yaw_target - yaw)
        if abs(yaw_err) > self.yaw_tol:
            self.publish_yaw_setpoint(self.yaw_target)
            return

        tx, ty = self.local_target
        dist = math.hypot(tx - rx, ty - ry)

        if min(self.d_front, self.min_scan) < (0.8 * self.d_stop):
            self.get_logger().warn(
                f"[LOCAL_AVOID] Front blocked again ({min(self.d_front, self.min_scan):.2f} m < {0.8 * self.d_stop:.2f} m). Recomputing heading."
            )
            self.avoid_attempts += 1
            self._enter_local_avoid()
            return

        if dist > self.pos_tol:
            if self._candidate_hard_blocked(tx, ty) or (not self.is_segment_safe(rx, ry, tx, ty)):
                self.get_logger().warn("[LOCAL] segment blocked/unsafe -> ESCAPE")
                self._enter_escape_mode()
                return
            self.publish_position_setpoint(tx, ty, self._z_setpoint_current(), self.yaw_target)
            return

        if min(self.d_front, self.min_scan) > self.d_clear:
            self.state = "GLOBAL_FOLLOW"
            self.last_avoid_heading = None
            self.avoid_attempts = 0
            self.plan_global_path()
            return

        self.avoid_attempts += 1
        self._enter_local_avoid()

    def _compute_best_heading_scan(self, exclude_heading=None):
        """
        Scan-based selection.
        Returns: (best_rel, best_dist, best_score)
        """
        if not (self.have_scan and self.scan is not None and self.have_pose and self.have_goal):
            return None

        scan = self.scan
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y

        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        yaw_goal_world = math.atan2(gy - ry, gx - rx)
        yaw = self._yaw_from_quat()
        goal_rel = self.normalize_angle(yaw_goal_world - yaw)

        goal_ray = self._goal_ray_min_dist()
        goal_blocked = (goal_ray < self.goal_ray_block_dist)

        best_score = -1e18
        best_angle = None
        best_dist = None
        r_max = max(1e-3, scan.range_max)

        ang = scan.angle_min
        for r in scan.ranges:
            a_rel = self.normalize_angle(ang)

            if (not math.isfinite(r)) or r <= scan.range_min:
                ang += scan.angle_increment
                continue

            if exclude_heading is not None:
                if abs(self.normalize_angle(a_rel - exclude_heading)) < (math.pi / 6):
                    ang += scan.angle_increment
                    continue

            if r < self.rep_range:
                ang += scan.angle_increment
                continue

            step = min(self.step_forward, max(0.12, r - 0.25))
            cand_x = rx + step * math.cos(yaw + a_rel)
            cand_y = ry + step * math.sin(yaw + a_rel)

            if self._candidate_hard_blocked(cand_x, cand_y):
                ang += scan.angle_increment
                continue
            if not self.is_segment_safe(rx, ry, cand_x, cand_y):
                ang += scan.angle_increment
                continue

            range_term = min(r / r_max, 1.0)

            align_term = math.cos(a_rel - goal_rel)
            if goal_blocked:
                if abs(self.normalize_angle(a_rel - goal_rel)) < (math.pi / 6):
                    align_term *= 0.15  # stronger suppression than before

            clear_term = 0.0
            if self.have_edt:
                D = self._edt_value_at(cand_x, cand_y)
                clear_term = max(0.0, min(1.0, D / max(self.min_clearance, 1e-3)))

            visited_pen = 0.0
            if self.have_grid and self.visited_grid is not None:
                ij = self.xy_to_ij(cand_x, cand_y)
                if ij is None:
                    visited_pen = 5.0
                else:
                    visited_pen = self._visited_penalty_at_cell(*ij)

            score = (self.w_range * range_term) + (self.w_align * align_term) + (self.w_clear * clear_term) - (self.w_visited * visited_pen)

            if score > best_score:
                best_score = score
                best_angle = a_rel
                best_dist = r

            ang += scan.angle_increment

        if best_angle is None:
            return None
        return best_angle, best_dist, best_score

    def _compute_best_heading_grid_sample(self):
        """
        Grid-based radial sampling around the robot.
        Returns: (best_rel, best_free_dist, best_score)
        - best_free_dist: approximate (for step sizing). We'll use max radius as proxy.
        """
        if not (self.have_pose and self.have_goal and self.have_grid):
            return None

        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        yaw = self._yaw_from_quat()

        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        yaw_goal_world = math.atan2(gy - ry, gx - rx)
        goal_rel = self.normalize_angle(yaw_goal_world - yaw)

        goal_ray = self._goal_ray_min_dist()
        goal_blocked = (goal_ray < self.goal_ray_block_dist)

        angle_step = math.radians(max(5.0, self.local_sample_angle_deg))
        forward_bias = math.radians(max(30.0, min(180.0, self.local_sample_forward_bias_deg)))

        # sample angles around goal direction primarily (to stay purposeful) but avoid crash if blocked
        # if goal is blocked, widen search around perpendicular directions too
        centers = [goal_rel]
        if goal_blocked:
            centers += [goal_rel + math.pi/2, goal_rel - math.pi/2, goal_rel + math.pi]

        best_score = -1e18
        best_rel = None
        best_dist = None

        # precompute radii (largest gives more progress but requires safety)
        radii = [r for r in self.local_sample_radii if r > 0.1]
        if not radii:
            radii = [0.35, 0.55, 0.75]

        # Score uses same weights; treat "range_term" as normalized radius
        r_max = max(radii)

        for c in centers:
            a0 = c - forward_bias
            a1 = c + forward_bias
            a = a0
            while a <= a1 + 1e-9:
                a_rel = self.normalize_angle(a)
                # choose farthest safe radius in this direction (greedy)
                best_r_here = None
                best_score_here = -1e18

                for r in radii:
                    cand_x = rx + r * math.cos(yaw + a_rel)
                    cand_y = ry + r * math.sin(yaw + a_rel)

                    ij = self.xy_to_ij(cand_x, cand_y)
                    if ij is None:
                        continue
                    if self.is_occ(*ij):
                        continue
                    if self._hard_blocked_cell(*ij):
                        continue
                    if not self.is_segment_safe(rx, ry, cand_x, cand_y):
                        continue

                    # range_term (progress proxy)
                    range_term = min(r / r_max, 1.0)

                    # align_term
                    align_term = math.cos(a_rel - goal_rel)
                    if goal_blocked and abs(self.normalize_angle(a_rel - goal_rel)) < (math.pi / 6):
                        align_term *= 0.1

                    # clearance term
                    clear_term = 0.0
                    if self.have_edt:
                        D = self._edt_value_at(cand_x, cand_y)
                        clear_term = max(0.0, min(1.0, D / max(self.min_clearance, 1e-3)))

                    # visited penalty
                    visited_pen = 0.0
                    if self.visited_grid is not None:
                        visited_pen = self._visited_penalty_at_cell(*ij)

                    score = (self.w_range * range_term) + (self.w_align * align_term) + (self.w_clear * clear_term) - (self.w_visited * visited_pen)

                    if score > best_score_here:
                        best_score_here = score
                        best_r_here = r

                if best_r_here is not None and best_score_here > best_score:
                    best_score = best_score_here
                    best_rel = a_rel
                    best_dist = best_r_here  # treat as available free distance proxy

                a += angle_step

        if best_rel is None:
            return None
        return best_rel, best_dist, best_score

    # -----------------------
    # Stuck detector
    # -----------------------
    def _update_stuck_detector(self):
        if not (self.have_pose and self.have_goal):
            return
        now = self.get_clock().now()
        d = self.dist_to_goal()
        self._prog_hist.append((now, d))
        while self._prog_hist and (now - self._prog_hist[0][0]).nanoseconds > int(self.stuck_time_horizon * 1e9):
            self._prog_hist.popleft()

        if len(self._prog_hist) < 2:
            return

        progress = self._prog_hist[0][1] - self._prog_hist[-1][1]
        stuck_by_progress = (progress < self.stuck_progress_eps)
        stuck_by_heading = (self._no_heading_count >= self.no_heading_limit)

        if self.state.startswith("ESCAPE"):
            return

        if stuck_by_progress or stuck_by_heading:
            self.get_logger().warn(
                f"[STUCK] progress={progress:.3f}/{self.stuck_time_horizon:.1f}s "
                f"no_heading={self._no_heading_count}/{self.no_heading_limit} -> ESCAPE"
            )
            self._enter_escape_mode()

    # -----------------------
    # Safety: segment check
    # -----------------------
    def is_segment_safe(self, x0, y0, x1, y1) -> bool:
        """
        Segment safety:
          - OccupancyGrid ray check
          - EDT clearance check: reject if clearance is too small
        """
        if not self.have_grid:
            return True

        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return True

        steps = max(2, int(length / max(self.segment_check_step, 1e-3)))
        for k in range(steps + 1):
            t = k / steps
            x = x0 + t * dx
            y = y0 + t * dy
            ij = self.xy_to_ij(x, y)
            if ij is None:
                return False
            i, j = ij
            if self.is_occ(i, j):
                return False
            if self._hard_blocked_cell(i, j):
                return False
            if self.have_edt:
                if self._edt_value_at(x, y) < self.min_clearance:
                    return False
        return True

    # -----------------------
    # APF
    # -----------------------
    def _compute_apf_force_world(self, target_x, target_y):
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        yaw = self._yaw_from_quat()

        dx, dy = (target_x - rx), (target_y - ry)
        F_att_x = self.k_att * dx
        F_att_y = self.k_att * dy

        F_rep_x = 0.0
        F_rep_y = 0.0
        if self.have_scan and self.scan is not None:
            ang = self.scan.angle_min
            for r in self.scan.ranges:
                if (not math.isfinite(r)) or r < 0.05:
                    ang += self.scan.angle_increment
                    continue
                if r <= self.rep_range:
                    ox_r = r * math.cos(ang)
                    oy_r = r * math.sin(ang)
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

    # -----------------------
    # Escape mode (same as before, but uses hard block + safety)
    # -----------------------
    @staticmethod
    def rot_cw_90(v):
        x, y = v
        return (y, -x)

    @staticmethod
    def rot_ccw_90(v):
        x, y = v
        return (-y, x)

    def _goal_vec_unit(self):
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        vx, vy = (gx - rx), (gy - ry)
        n = math.hypot(vx, vy)
        if n < 1e-6:
            return (1.0, 0.0)
        return (vx / n, vy / n)

    def _choose_hand(self):
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        g = self._grad_edt_unit_at(rx, ry)
        if g is None:
            self.hand = "R"
            return
        tL = self.rot_ccw_90(g)
        tR = self.rot_cw_90(g)
        gv = self._goal_vec_unit()
        dl = tL[0] * gv[0] + tL[1] * gv[1]
        dr = tR[0] * gv[0] + tR[1] * gv[1]
        self.hand = "R" if dr > dl else "L"

    def _has_line_of_sight(self, x0, y0, x1, y1) -> bool:
        if not self.have_grid:
            return False
        p0 = self.xy_to_ij(x0, y0)
        p1 = self.xy_to_ij(x1, y1)
        if p0 is None or p1 is None:
            return False

        i0, j0 = p0
        i1, j1 = p1
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1

        err = dj - di
        i, j = i0, j0
        while True:
            if self.is_occ(i, j):
                return False
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -di:
                err -= di
                j += sj
            if e2 < dj:
                err += dj
                i += si
        return True

    def dist_to_goal(self) -> float:
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        return math.hypot(gx - rx, gy - ry)

    def _escape_release_check(self) -> bool:
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        if self.hit_point_dist is None:
            return False
        if self._has_line_of_sight(rx, ry, gx, gy):
            if self.dist_to_goal() < (self.hit_point_dist - self.escape_release_margin):
                return True
        return False

    def _make_escape_subgoal(self):
        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        g = self._grad_edt_unit_at(rx, ry)
        if g is None:
            return None
        t = self.rot_cw_90(g) if self.hand == "R" else self.rot_ccw_90(g)
        tn = math.hypot(t[0], t[1])
        if tn < 1e-6:
            return None
        ux, uy = (t[0] / tn, t[1] / tn)

        p_x, p_y = rx, ry
        min_ok = self.escape_follow_dist * self.escape_subgoal_min_edt_ratio
        for _ in range(int(self.escape_subgoal_steps)):
            nx = p_x + self.escape_step * ux
            ny = p_y + self.escape_step * uy
            ij = self.xy_to_ij(nx, ny)
            if ij is None or self.is_occ(*ij):
                break
            if self.have_edt:
                if self._edt_value_at(nx, ny) < min_ok:
                    break
            p_x, p_y = nx, ny

        if math.hypot(p_x - rx, p_y - ry) < 0.5:
            return None
        # avoid hard mask even for subgoal (but allow if we are stuck: mask relax already handled in _hard_blocked_cell)
        ij = self.xy_to_ij(p_x, p_y)
        if ij is not None and self._hard_blocked_cell(*ij):
            return None
        return (p_x, p_y)

    def _enter_escape_mode(self):
        if not (self.have_grid and self.have_edt and self.have_pose and self.have_goal):
            self.get_logger().warn("[ESCAPE] prerequisites missing -> LOCAL_AVOID retry")
            self._enter_local_avoid()
            return

        self.hit_point_dist = self.dist_to_goal()
        self._choose_hand()
        self.subgoal = None

        if self.escape_make_subgoal:
            sg = self._make_escape_subgoal()
            if sg is not None:
                self.subgoal = sg
                self.state = "ESCAPE_GO_SUBGOAL"
                self.get_logger().warn(f"[ESCAPE] GO_SUBGOAL hand={self.hand} hit={self.hit_point_dist:.2f} sg={sg}")
                return

        self.state = "ESCAPE_WF"
        self.get_logger().warn(f"[ESCAPE] WALL_FOLLOW hand={self.hand} hit={self.hit_point_dist:.2f}")

    def _escape_wall_follow_step(self):
        if not (self.have_pose and self.have_goal and self.have_edt):
            self.publish_hold_setpoint()
            return

        if self._escape_release_check():
            self.get_logger().warn("[ESCAPE_WF] release -> GLOBAL (replan)")
            self.state = "GLOBAL_FOLLOW"
            self.plan_global_path()
            return

        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        g = self._grad_edt_unit_at(rx, ry)
        if g is None:
            gv = self._goal_vec_unit()
            tx = rx + self.escape_step * gv[0]
            ty = ry + self.escape_step * gv[1]
            if self.is_segment_safe(rx, ry, tx, ty):
                self.publish_position_setpoint(tx, ty, self._z_setpoint_current(), math.atan2(gv[1], gv[0]))
            else:
                self.publish_hold_setpoint()
            return

        nx, ny = g
        t = self.rot_cw_90((nx, ny)) if self.hand == "R" else self.rot_ccw_90((nx, ny))
        tn = math.hypot(t[0], t[1])
        txu, tyu = (t[0] / tn, t[1] / tn)

        D = self._edt_value_at(rx, ry)
        e = D - self.escape_follow_dist

        ux = txu + self.escape_kn * e * nx
        uy = tyu + self.escape_kn * e * ny
        un = math.hypot(ux, uy)
        if un < 1e-6:
            ux, uy = txu, tyu
            un = 1.0
        ux /= un; uy /= un

        tx = rx + self.escape_step * ux
        ty = ry + self.escape_step * uy

        if not self.is_segment_safe(rx, ry, tx, ty):
            self.hand = "L" if self.hand == "R" else "R"
            self.get_logger().warn("[ESCAPE_WF] blocked -> switch hand")
            return

        self.publish_position_setpoint(tx, ty, self._z_setpoint_current(), math.atan2(uy, ux))

    def _escape_subgoal_step(self):
        if not (self.have_pose and self.have_goal):
            self.publish_hold_setpoint()
            return
        if self.subgoal is None:
            self.state = "ESCAPE_WF"
            return

        if self.have_edt and self._escape_release_check():
            self.get_logger().warn("[ESCAPE_SG] release -> GLOBAL (replan)")
            self.state = "GLOBAL_FOLLOW"
            self.subgoal = None
            self.plan_global_path()
            return

        rx, ry = self.pose.pose.position.x, self.pose.pose.position.y
        sx, sy = self.subgoal
        d = math.hypot(sx - rx, sy - ry)
        if d < max(self.pos_tol, 0.4):
            self.get_logger().warn("[ESCAPE_SG] reached -> WALL_FOLLOW")
            self.subgoal = None
            self.state = "ESCAPE_WF"
            return

        ux, uy = ((sx - rx) / d, (sy - ry) / d)
        step = min(self.escape_step, max(self.min_step_ahead, d))
        tx = rx + step * ux
        ty = ry + step * uy

        if not self.is_segment_safe(rx, ry, tx, ty):
            self.get_logger().warn("[ESCAPE_SG] unsafe -> WALL_FOLLOW")
            self.subgoal = None
            self.state = "ESCAPE_WF"
            return

        self.publish_position_setpoint(tx, ty, self._z_setpoint_current(), math.atan2(uy, ux))

    def _reset_escape(self):
        self.hand = None
        self.hit_point_dist = None
        self.subgoal = None

    # -----------------------
    # Emergency / step adapt
    # -----------------------
    def _adaptive_step(self, min_obs_dist: float) -> float:
        step = self.step_ahead
        if not math.isfinite(min_obs_dist):
            return step
        if min_obs_dist < 1.0:
            return max(self.min_step_ahead, step * 0.25)
        if min_obs_dist < 2.0:
            return max(self.min_step_ahead, step * 0.5)
        return step

    def _emergency_backoff(self, min_obs: float):
        rx = self.pose.pose.position.x
        ry = self.pose.pose.position.y

        if self.have_edt:
            g = self._grad_edt_unit_at(rx, ry)
            if g is not None:
                ux, uy = g
            else:
                yaw = self._yaw_from_quat()
                ux, uy = (-math.cos(yaw), -math.sin(yaw))
        else:
            yaw = self._yaw_from_quat()
            ux, uy = (-math.cos(yaw), -math.sin(yaw))

        step = max(self.min_step_ahead, min(self.step_ahead * 0.5, 0.3))
        tx = rx + ux * step
        ty = ry + uy * step

        if self.have_grid and (not self.is_segment_safe(rx, ry, tx, ty)):
            self.publish_hold_setpoint()
            return

        self.get_logger().warn(f"[EMERGENCY] min_obs={min_obs:.2f} -> backoff ({tx:.2f},{ty:.2f})")
        self.publish_position_setpoint(tx, ty, self._z_setpoint_current(), self._yaw_from_quat())

    # -----------------------
    # Setpoints
    # -----------------------
    def publish_hold_setpoint(self):
        p = self.pose.pose.position
        self.publish_position_setpoint(p.x, p.y, self._z_setpoint_current(), self._yaw_from_quat())

    def publish_yaw_setpoint(self, yaw_target: float):
        p = self.pose.pose.position
        self.publish_position_setpoint(p.x, p.y, self._z_setpoint_current(), yaw_target)

    def publish_position_setpoint(self, x, y, z, yaw):
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = self.global_frame
        sp.pose.position.x = float(x)
        sp.pose.position.y = float(y)
        sp.pose.position.z = float(z)

        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        sp.pose.orientation.z = qz
        sp.pose.orientation.w = qw

        self.pub_setpoint.publish(sp)

    # -----------------------
    # Z setpoint
    # -----------------------
    def _z_setpoint_current(self):
        """Z setpoint 계산 (fixed / hold_pose_z with lock/slew limit)"""
        if self.z_setpoint_mode == 'fixed':
            z_target = self.z_fixed
        elif self.z_setpoint_mode == 'hold_pose_z':
            if not self.have_pose:
                z_target = self.z_fixed
            elif self.z_lock_enable and self._z_lock_initialized and (self._z_setpoint_locked is not None):
                # Z lock 모드: 초기 z 값 사용
                z_target = self._z_setpoint_locked
            else:
                # 일반 hold_pose_z 모드: 현재 pose z 사용 (outlier 필터링)
                z_pose = float(self.pose.pose.position.z)
                if self._z_setpoint_last is not None:
                    # Outlier 감지: 이전 값과 차이가 임계값을 초과하면 무시
                    z_diff = abs(z_pose - self._z_setpoint_last)
                    if z_diff > self.z_outlier_threshold:
                        self.get_logger().warn(
                            f'Z outlier detected: {z_pose:.2f} m (diff={z_diff:.2f} m), using last value {self._z_setpoint_last:.2f} m'
                        )
                        z_target = self._z_setpoint_last
                    else:
                        z_target = z_pose
                else:
                    z_target = z_pose
        else:
            z_target = self.z_fixed

        # Slew limit 적용 (변화율 제한)
        if self._z_setpoint_last is not None and self._last_control_time is not None:
            now = self.get_clock().now()
            dt = (now - self._last_control_time).nanoseconds / 1e9
            if 0.0 < dt < 1.0:
                dz = z_target - self._z_setpoint_last
                dz_max = self.z_slew_rate_max * dt
                if abs(dz) > dz_max:
                    # 변화율 제한
                    z_original = z_target
                    z_target = self._z_setpoint_last + math.copysign(dz_max, dz)
                    # Z lock 모드에서는 locked 값으로 강제
                    if self.z_lock_enable and self._z_lock_initialized and (self._z_setpoint_locked is not None):
                        z_target = self._z_setpoint_locked
                    # 일반 모드에서도 큰 변화는 경고
                    elif abs(z_original - self._z_setpoint_last) > 0.3:
                        self.get_logger().warn(
                            f'Z large change detected: {self._z_setpoint_last:.2f} -> {z_original:.2f} m, '
                            f'slew-limited to {z_target:.2f} m (rate={abs(dz)/dt:.2f} m/s > {self.z_slew_rate_max:.2f} m/s)'
                        )

        # 최종 값 저장
        self._z_setpoint_last = z_target
        self._last_control_time = self.get_clock().now()
        return z_target

    # -----------------------
    # Yaw helpers
    # -----------------------
    def _yaw_from_quat(self):
        q = self.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(a):
        return math.atan2(math.sin(a), math.cos(a))


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
