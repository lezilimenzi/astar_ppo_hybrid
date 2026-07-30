#!/usr/bin/env python3
"""RL Offboard: obs(lidar+goal_rel_body) -> policy -> velocity setpoint (local/world ENU). OFFBOARD/ARM 자동."""

from __future__ import annotations

import heapq
import json
import math
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data

try:
    from scipy import ndimage as _ndimage
    from scipy.interpolate import splprep as _splprep, splev as _splev
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def sanitize_lidar(arr: np.ndarray, max_range: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.where(np.isnan(arr) | np.isinf(arr), max_range, arr)
    return np.clip(arr, 0.0, max_range).astype(np.float32)


def downsample_lidar(lidar: np.ndarray, target: int, mode: str = "min") -> np.ndarray:
    n = len(lidar)
    if n == target:
        return lidar.copy().astype(np.float32)
    if target > n:
        idx = np.linspace(0, n - 1, target).astype(int)
        return lidar[idx].astype(np.float32)
    indices = np.array_split(np.arange(n), target)
    if mode == "min":
        return np.array([float(np.min(lidar[i])) for i in indices], dtype=np.float32)
    if mode == "avg":
        return np.array([float(np.mean(lidar[i])) for i in indices], dtype=np.float32)
    raise ValueError(f"downsample_mode must be 'min' or 'avg', got {mode}")


class RLOffboardNode(Node):
    def __init__(self):
        super().__init__("rl_offboard_node")

        # Params (declare + get in one pass). 상세 근거는 config/params.yaml 참조.
        params = [
            ("mavros_state_topic", "/mavros/state"),
            ("mavros_odom_topic", "/mavros/local_position/odom"),
            ("mavros_setpoint_topic", "/mavros/setpoint_velocity/cmd_vel_unstamped"),
            ("scan_topic", "/scan"),
            ("goal_topic", "/planner/goal"),
            ("homing_done_topic", "/planner/homing_done"),
            ("wait_for_homing", True),
            ("setpoint_rate_hz", 20.0),
            ("pre_setpoint_sec", 2.0),
            ("policy_path", ""),
            ("obs_norm_path", ""),
            ("obs_beams", 64),
            ("max_range", 10.0),
            ("world_size", 20.0),
            ("downsample_mode", "min"),
            ("odom_timeout_sec", 0.3),
            ("scan_timeout_sec", 0.5),
            ("goal_radius", 1.5),
            ("nav_mode", "hybrid"),
            ("grid_topic", "/local_grid"),
            ("occ_threshold", 50),
            ("allow_diag", True),
            ("replan_period", 1.0),
            ("planner_max_yaw_rate", 1.0),
            ("wp_advance_search_window", 10),
            ("rl_front_deg", 90.0),
            ("lidar_range_min_override", 0.35),
            ("emergency_brake_dist", 0.0),
            ("emergency_brake_deg", 30.0),
            ("enable_altitude_hold", True),
            ("altitude_hold_margin", 0.1),
            ("altitude_hold_gain", 0.4),
            ("mode_retry_sec", 1.0),
            ("arm_retry_sec", 1.0),
            ("astar_min_inflation_cells",      1),
            ("shortcut_los_clearance_cells",   1),
            ("hybrid_lookahead_dist_m",        1.0),
            ("smooth_bspline_enabled",         True),
            ("smooth_bspline_s",               0.0),
            ("smooth_bspline_output_points",   50),
            ("smooth_bspline_skip_if_few_wps", 4),
        ]
        for name, default in params:
            self.declare_parameter(name, default)
        for name, _ in params:
            setattr(self, name, self.get_parameter(name).value)
        # scipy 미설치 시 smoothing 비활성
        if not _SCIPY_OK:
            self.smooth_bspline_enabled = False
            self.get_logger().warn("scipy 없음: B-spline smoothing 비활성화.")

        # nav_mode별 default 모델 파일 (policy_path/obs_norm_path 미지정 시).
        #   hybrid → policy_hybrid_ts.pt / obs_norm_hybrid.json
        #   ppo    → policy_ppo_ts.pt    / obs_norm_ppo.json
        #   astar  → policy 불필요
        policy_path = self.policy_path or ""
        obs_norm_path = self.obs_norm_path or ""
        if self.nav_mode != "astar" and (not policy_path or not obs_norm_path):
            if self.nav_mode == "hybrid":
                default_policy  = "policy_hybrid_ts.pt"
                default_obs_norm = "obs_norm_hybrid.json"
            else:  # "ppo"
                default_policy  = "policy_ppo_ts.pt"
                default_obs_norm = "obs_norm_ppo.json"
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg = Path(get_package_share_directory("px4_rl_offboard_mavros"))
            except Exception:
                pkg = Path(__file__).resolve().parents[2]
            policy_path   = policy_path   or str(pkg / "models" / default_policy)
            obs_norm_path = obs_norm_path or str(pkg / "models" / default_obs_norm)

        # Load obs_norm + policy (astar 모드는 불필요)
        self.obs_dim = self.obs_beams + 2
        self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_std  = np.ones(self.obs_dim, dtype=np.float32)
        self.clip_obs = 10.0
        self.v_min = 0.3
        self.v_max = 0.8
        self.max_yaw_rate = 1.0
        self.policy = None
        self._torch = None
        if self.nav_mode != "astar":
            with open(obs_norm_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            obs_cfg = cfg["obs"]
            self.obs_dim = int(obs_cfg.get("obs_dim", self.obs_beams + 2))
            if self.obs_dim != self.obs_beams + 2:
                raise ValueError(f"obs_dim mismatch: {self.obs_dim} != {self.obs_beams + 2}")
            self.obs_mean = np.array(obs_cfg["mean"], dtype=np.float32)
            self.obs_std  = np.array(obs_cfg["std"],  dtype=np.float32)
            self.clip_obs = float(obs_cfg.get("clip_obs", 10.0))
            act = cfg["action"]["scale"]
            self.v_min = float(act["v_min"])
            self.v_max = float(act["v_max"])
            self.max_yaw_rate = float(act["max_yaw_rate"])
            import torch
            self._torch = torch
            self.policy = torch.jit.load(policy_path, map_location="cpu")
            self.policy.eval()

        # State
        self.mav_state: Optional[State] = None
        self.odom: Optional[Odometry] = None
        self.scan: Optional[LaserScan] = None
        self.goal_xy: Optional[np.ndarray] = None   # ENU (odom과 동일)
        # 센서 timestamp는 sim clock 기반 (Gazebo 속도 저하 시 wall clock 오탐 방지)
        self.t_last_odom = self.get_clock().now()
        self.t_last_scan = self.get_clock().now()
        self._pre_setpoint_count = 0
        self._pre_setpoint_needed = int(max(0, self.pre_setpoint_sec) * self.setpoint_rate_hz)
        self._t_last_mode_req = 0.0
        self._t_last_arm_req = 0.0
        self._homing_done = not self.wait_for_homing
        self._goal_reached = False
        self._scan_warn_last = 0.0   # scan 미수신 진단 로그 쓰로틀
        self._emergency_brake_active = False  # emergency brake 진입 시에만 로그
        self._cached_ranges: Optional[np.ndarray] = None  # scan ranges 캐시 (재생성 회피)
        self._brake_count = 0                               # 연속 brake 틱 카운터
        self._brake_rotate_left: Optional[bool] = None      # brake 진입 시 1회 결정, 해제 전 고정
        self._path_xy: List[Tuple[float, float]] = []
        self._path_wp_idx = 0
        self._have_grid = False
        self._grid_data: Optional[np.ndarray] = None
        self._grid_ox = self._grid_oy = 0.0
        self._grid_res = 0.1
        self._grid_w = self._grid_h = 0
        self._hold_z: Optional[float] = None  # Altitude hold reference
        self._plan_s_ij: Optional[Tuple[int, int]] = None
        self._plan_g_ij: Optional[Tuple[int, int]] = None
        self._t_last_debug_log = 0.0

        # QoS — /scan은 Gazebo RPLidar의 qos_profile_sensor_data와 명시 매칭 (불일치 시 미수신)
        q_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        q_scan = qos_profile_sensor_data
        q_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
        q_setpoint = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        # Callback groups — tick/replan/sensor를 별도 스레드로 분리해 A* 블로킹 방지.
        # tick·replan은 MutuallyExclusive(각 단일 실행), sensor는 Reentrant(저장만, 동시 허용).
        self._cb_group_tick   = MutuallyExclusiveCallbackGroup()
        self._cb_group_replan = MutuallyExclusiveCallbackGroup()
        self._cb_group_sensor = ReentrantCallbackGroup()

        # _path_lock: _path_xy/_path_wp_idx의 tick↔replan race 방지 (RLock=동일 스레드 재진입 허용).
        self._path_lock = threading.RLock()
        # _grid_lock: _cb_grid(sensor)↔_plan_path(replan) 간 grid 갱신/읽기 원자성 보장.
        self._grid_lock = threading.Lock()

        # Subs
        self.create_subscription(State, self.mavros_state_topic, self._cb_state, q_sensor,
                                 callback_group=self._cb_group_sensor)
        self.create_subscription(Odometry, self.mavros_odom_topic, self._cb_odom, q_sensor,
                                 callback_group=self._cb_group_sensor)
        self.create_subscription(LaserScan, self.scan_topic, self._cb_scan, q_scan,
                                 callback_group=self._cb_group_sensor)
        self.create_subscription(PoseStamped, self.goal_topic, self._cb_goal, q_goal,
                                 callback_group=self._cb_group_sensor)
        if self.nav_mode in ("hybrid", "astar"):
            self.create_subscription(OccupancyGrid, self.grid_topic, self._cb_grid, q_sensor,
                                     callback_group=self._cb_group_sensor)
        if self.wait_for_homing:
            self.create_subscription(Bool, self.homing_done_topic, self._cb_homing_done, q_goal,
                                     callback_group=self._cb_group_sensor)

        # Pub
        self._pub = self.create_publisher(Twist, self.mavros_setpoint_topic, q_setpoint)
        self._path_pub = self.create_publisher(NavPath, '/planner/path', 10)
        self._cli_mode = self.create_client(SetMode, "/mavros/set_mode")
        self._cli_arm = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.create_timer(1.0 / self.setpoint_rate_hz, self._tick,
                          callback_group=self._cb_group_tick)
        if self.nav_mode in ("hybrid", "astar"):
            self.create_timer(self.replan_period, self._tick_replan,
                              callback_group=self._cb_group_replan)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"RL Offboard node started. odom={self.mavros_odom_topic} scan={self.scan_topic} "
            f"goal={self.goal_topic} rate={self.setpoint_rate_hz}Hz"
        )

    def _cb_state(self, msg: State):
        self.mav_state = msg

    def _cb_odom(self, msg: Odometry):
        self.odom = msg
        self.t_last_odom = self.get_clock().now()

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg
        self.t_last_scan = self.get_clock().now()
        # scan ranges를 1회 캐시 → _dist_in_sector/_build_obs가 공유 (자기감지는 lidar_range_min_override로 처리)
        self._cached_ranges = np.array(msg.ranges, dtype=np.float32)

    def _cb_goal(self, msg: PoseStamped):
        # /planner/goal을 odom 프레임으로 변환
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                msg.header.frame_id,
                rclpy.time.Time()
            )
            transformed_pose = do_transform_pose_stamped(msg, transform)
            self.goal_xy = np.array(
                [transformed_pose.pose.position.x, transformed_pose.pose.position.y],
                dtype=np.float32
            )
            if self.nav_mode in ("hybrid", "astar"):
                self._plan_path()
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for goal: {e}. Ignore this goal update.")
            return

    def _cb_grid(self, msg: OccupancyGrid):
        new_h   = msg.info.height
        new_w   = msg.info.width
        new_res = msg.info.resolution
        new_ox  = msg.info.origin.position.x
        new_oy  = msg.info.origin.position.y
        new_data = np.array(msg.data, dtype=np.int16).reshape(new_h, new_w)
        with self._grid_lock:
            self._grid_h    = new_h
            self._grid_w    = new_w
            self._grid_res  = new_res
            self._grid_ox   = new_ox
            self._grid_oy   = new_oy
            self._grid_data = new_data
            self._have_grid = True
        self.get_logger().debug(
            f"[GRID] Received: {self._grid_w}x{self._grid_h} cells, res={self._grid_res}m, "
            f"origin=({self._grid_ox:.1f},{self._grid_oy:.1f}), "
            f"bounds x:[{self._grid_ox:.1f}, {self._grid_ox + self._grid_w*self._grid_res:.1f}] "
            f"y:[{self._grid_oy:.1f}, {self._grid_oy + self._grid_h*self._grid_res:.1f}]"
        )

    def _cb_homing_done(self, msg: Bool):
        if msg.data:
            self._homing_done = True
            if self.odom is not None:
                self._hold_z = float(self.odom.pose.pose.position.z)
            self.get_logger().info("Homing done. Starting RL velocity control.")

    def _publish(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0, yaw_rate: float = 0.0):
        """vx, vy: local/world(ENU) 기준 속도. cmd_vel_unstamped에 publish."""
        vx = 0.0 if not math.isfinite(vx) else max(-self.v_max, min(self.v_max, float(vx)))
        vy = 0.0 if not math.isfinite(vy) else max(-self.v_max, min(self.v_max, float(vy)))
        vz = 0.0 if not math.isfinite(vz) else max(-0.5, min(0.5, float(vz)))
        # yaw_rate clamp: astar는 planner_max_yaw_rate, hybrid/ppo는 max_yaw_rate(학습 분포 보호)
        if self.nav_mode == "astar":
            yaw_limit = float(getattr(self, 'planner_max_yaw_rate', self.max_yaw_rate))
        else:
            yaw_limit = self.max_yaw_rate
        yaw_rate = 0.0 if not math.isfinite(yaw_rate) else max(-yaw_limit, min(yaw_limit, float(yaw_rate)))
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = vx, vy, vz
        msg.angular.z = yaw_rate
        self._pub.publish(msg)

    def _sensors_ok(self) -> bool:
        """odom/scan/goal 신선도 검사 (sim clock 기반)."""
        now = self.get_clock().now()
        odom_age = (now - self.t_last_odom).nanoseconds / 1e9
        scan_age = (now - self.t_last_scan).nanoseconds / 1e9

        if self.odom is None or odom_age > self.odom_timeout_sec:
            return False
        if self.scan is None or scan_age > self.scan_timeout_sec:
            if self.allow_diag:
                wall_now = time.time()
                if (wall_now - self._scan_warn_last) > 5.0:
                    self._scan_warn_last = wall_now
                    if self.scan is None:
                        self.get_logger().warn(
                            f"[SCAN] {self.scan_topic} never received. "
                            "QoS 불일치 가능: ros2 topic info /scan -v 로 publisher QoS 확인 후 "
                            "qos_profile_sensor_data(BEST_EFFORT+VOLATILE)와 맞는지 확인."
                        )
                    else:
                        self.get_logger().warn(
                            f"[SCAN] {self.scan_topic} timeout (>{self.scan_timeout_sec}s, age={scan_age:.2f}s). "
                            "Gazebo 부하 또는 일시적 scan 중단. QoS는 정상."
                        )
            return False
        if self.goal_xy is None:
            return False
        return True

    def _build_obs(self) -> Optional[np.ndarray]:
        """lidar(64) + goal_rel(body frame, 2) = obs_dim(66) 관측 벡터 생성."""
        if self.odom is None or self.scan is None or self.goal_xy is None:
            return None
        # scan 캐시 우선 사용 (scan 콜백에서 1회 생성)
        if self._cached_ranges is not None and len(self._cached_ranges) == len(self.scan.ranges):
            raw = self._cached_ranges.copy()
        else:
            raw = np.array(self.scan.ranges, dtype=np.float32)
        rmin = float(getattr(self.scan, "range_min", 0.0))
        override = float(getattr(self, "lidar_range_min_override", 0.35) or 0.0)
        # RL 관측엔 자기감지 override 적용 (비상 제동은 use_override=False로 미적용)
        if override > rmin:
            rmin = override
        raw = np.where(np.isfinite(raw) & (raw > rmin), raw, self.max_range)
        ranges = sanitize_lidar(raw, self.max_range)
        if len(ranges) != self.obs_beams:
            ranges = downsample_lidar(ranges, self.obs_beams, self.downsample_mode)
        lidar = np.clip(ranges / self.max_range, 0.0, 1.0) * 2.0 - 1.0
        px, py = float(self.odom.pose.pose.position.x), float(self.odom.pose.pose.position.y)
        yaw = quat_to_yaw(self.odom.pose.pose.orientation)

        # scan-odom dead reckoning 보정: 타임스탬프 차이만큼 위치/yaw 외삽 → goal_rel 정확도 향상
        if self.scan is not None and self.scan.header.stamp.sec != 0:
            scan_t = rclpy.time.Time.from_msg(self.scan.header.stamp).nanoseconds / 1e9
            odom_t = rclpy.time.Time.from_msg(self.odom.header.stamp).nanoseconds / 1e9
            dt = scan_t - odom_t
            if abs(dt) > 0.01:
                vx_b = self.odom.twist.twist.linear.x
                vy_b = self.odom.twist.twist.linear.y
                wz   = self.odom.twist.twist.angular.z
                vx_w = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
                vy_w = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
                px  += vx_w * dt
                py  += vy_w * dt
                yaw += wz   * dt

        # goal_rel target: hybrid는 경로상 lookahead 지점(거리 기반), ppo는 최종 goal.
        # norm_scale은 hybrid에서 학습값 3.0 고정(입력 분포 유지), ppo는 world_size 정규화.
        LOOKAHEAD_NORM_SCALE = 3.0
        lookahead_dist = float(getattr(self, "hybrid_lookahead_dist_m", 1.0))
        if self.nav_mode == "hybrid" and self._path_xy:
            wp = self._path_lookahead_wp_at_dist(px, py, lookahead_dist)
            target_xy = np.array(wp, dtype=np.float32) if wp else self.goal_xy
            norm_scale = LOOKAHEAD_NORM_SCALE
        else:
            target_xy = self.goal_xy
            norm_scale = max(self.world_size, 1e-6)

        dx_world = (target_xy - np.array([px, py], dtype=np.float32)) / norm_scale
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx_body = c * dx_world[0] - s * dx_world[1]
        dy_body = s * dx_world[0] + c * dx_world[1]
        goal_rel = np.clip(np.array([dx_body, dy_body], dtype=np.float32), -1.0, 1.0)
        obs = np.concatenate([lidar.astype(np.float32), goal_rel])
        return obs if obs.shape[0] == self.obs_dim else None

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = np.clip((obs - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs).astype(np.float32)
        if np.any(np.isnan(obs_norm)) or np.any(np.isinf(obs_norm)):
            return np.zeros(2, dtype=np.float32)
        with self._torch.no_grad():
            t = self._torch.from_numpy(obs_norm).float().unsqueeze(0)
            out = self.policy(t).detach().cpu().numpy().flatten().astype(np.float32)
        if out.shape[0] < 2 or np.any(np.isnan(out)) or np.any(np.isinf(out)):
            return np.zeros(2, dtype=np.float32)
        return np.clip(out, -1.0, 1.0)

    def _action_to_vel(self, action: np.ndarray) -> Tuple[float, float]:
        v_cmd = self.v_min + (action[0] + 1.0) * 0.5 * (self.v_max - self.v_min)
        yaw_rate = action[1] * self.max_yaw_rate
        return float(v_cmd), float(yaw_rate)

    def _compute_cmd_vel(self, v_cmd: float, yaw: float) -> Tuple[float, float]:
        """heading 방향 속도를 local/world(ENU) vx·vy로 변환."""
        vx = v_cmd * math.cos(yaw)
        vy = v_cmd * math.sin(yaw)
        return vx, vy

    def _vz_correction(self) -> float:
        if not self.enable_altitude_hold or self.odom is None:
            return 0.0
        # 현재 고도 유지 (최초 저장된 _hold_z 기준)
        if self._hold_z is None:
            return 0.0
        target = self._hold_z
        err = target - float(self.odom.pose.pose.position.z)
        if abs(err) <= self.altitude_hold_margin:
            return 0.0
        return max(-0.5, min(0.5, self.altitude_hold_gain * err))

    def _dist_in_sector(self, low_deg: float, high_deg: float, agg: str, use_override: bool = True) -> float:
        """LiDAR body frame 각도 구간(deg) 거리 집계. agg: min|median|mean.
        use_override=False면 lidar_range_min_override를 rmin에 반영하지 않음(비상 제동 등)."""
        if self.scan is None or not self.scan.ranges:
            return self.max_range
        if self._cached_ranges is not None and len(self._cached_ranges) == len(self.scan.ranges):
            ranges = self._cached_ranges.copy()
        else:
            ranges = np.array(self.scan.ranges, dtype=np.float32)
        rmin = float(getattr(self.scan, "range_min", 0.0))
        override = float(getattr(self, "lidar_range_min_override", 0.35) or 0.0)
        if use_override and override > rmin:
            rmin = override
        ranges = np.where(np.isfinite(ranges) & (ranges > rmin), ranges, self.max_range)
        ang_min = float(self.scan.angle_min)
        inc = float(self.scan.angle_increment)
        # [0, 2π) 공간에서 비교 (atan2 (-π,π] 범위가 180° 초과 섹터를 놓치는 wrap-around 방지)
        _TWO_PI = 2.0 * math.pi
        low_pos  = math.radians(low_deg)  % _TWO_PI
        high_pos = math.radians(high_deg) % _TWO_PI
        vals: List[float] = []
        for i in range(len(ranges)):
            ang_pos = (ang_min + i * inc) % _TWO_PI
            if low_pos <= high_pos:
                in_sector = low_pos <= ang_pos <= high_pos
            else:
                # 섹터가 0/2π 경계를 넘는 경우 (예: 350°~10°)
                in_sector = ang_pos >= low_pos or ang_pos <= high_pos
            if in_sector:
                vals.append(float(ranges[i]))
        if not vals:
            return self.max_range
        a = agg.lower()
        if a == "median":
            return float(np.median(np.array(vals, dtype=np.float32)))
        if a == "mean":
            return float(np.mean(np.array(vals, dtype=np.float32)))
        return float(min(vals))

    def _xy_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if not self._have_grid:
            return None
        j = int(math.floor((x - self._grid_ox) / self._grid_res))
        i = int(math.floor((y - self._grid_oy) / self._grid_res))
        if 0 <= i < self._grid_h and 0 <= j < self._grid_w:
            return (i, j)
        return None

    def _ij_to_xy(self, i: int, j: int) -> Tuple[float, float]:
        x = self._grid_ox + (j + 0.5) * self._grid_res
        y = self._grid_oy + (i + 0.5) * self._grid_res
        return (x, y)

    def _is_occ(self, i: int, j: int) -> bool:
        if i < 0 or i >= self._grid_h or j < 0 or j >= self._grid_w:
            return True
        return self._grid_data[i, j] >= self.occ_threshold

    def _is_plan_blocked(self, i: int, j: int, grid: Optional[np.ndarray] = None) -> bool:
        """A* 셀 막힘 판정 (occ≥threshold). start/goal 셀은 예외.
        grid: None이면 self._grid_data. 병렬 호출 시 독립 grid 전달로 race 방지."""
        if i < 0 or i >= self._grid_h or j < 0 or j >= self._grid_w:
            return True
        g = grid if grid is not None else self._grid_data
        val = int(g[i, j])
        if val >= self.occ_threshold:
            return True
        if self._plan_s_ij is not None and (i, j) == self._plan_s_ij:
            return False
        if self._plan_g_ij is not None and (i, j) == self._plan_g_ij:
            return False
        return False

    @staticmethod
    def _h_octile(i: int, j: int, gi: int, gj: int) -> float:
        dx, dy = abs(i - gi), abs(j - gj)
        return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)

    def _astar_neighbors(self, i: int, j: int, grid: Optional[np.ndarray] = None):
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.allow_diag:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for di, dj in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < self._grid_h and 0 <= nj < self._grid_w and not self._is_plan_blocked(ni, nj, grid):
                yield ni, nj, math.hypot(di, dj)

    def _astar(
        self,
        start_ij: Tuple[int, int],
        goal_ij: Tuple[int, int],
        grid: Optional[np.ndarray] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """A* 경로 탐색. grid=None이면 self._grid_data (독립 grid 전달 시 thread-safe)."""
        si, sj = start_ij
        gi, gj = goal_ij
        if (si, sj) == (gi, gj):
            return [(si, sj)]
        openq = []
        heapq.heappush(openq, (self._h_octile(si, sj, gi, gj), 0.0, (si, sj)))
        gscore = {(si, sj): 0.0}
        parent = {(si, sj): None}
        visited = set()
        while openq:
            _, gc, (i, j) = heapq.heappop(openq)
            if (i, j) in visited:
                continue
            visited.add((i, j))
            if (i, j) == (gi, gj):
                break
            for ni, nj, step in self._astar_neighbors(i, j, grid):
                ng = gc + step
                if ng < gscore.get((ni, nj), 1e18):
                    gscore[(ni, nj)] = ng
                    parent[(ni, nj)] = (i, j)
                    heapq.heappush(openq, (ng + self._h_octile(ni, nj, gi, gj), ng, (ni, nj)))
        if (gi, gj) not in parent:
            return None
        path = []
        node = (gi, gj)
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    # ── 경로 후처리 (LOS shortcut + inflation + B-spline smoothing) ──────────

    def _bresenham_cells(self, i0: int, j0: int, i1: int, j1: int):
        """Bresenham 선분 위의 모든 (i, j) 격자 셀을 yield."""
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            yield i, j
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj

    def _los_clear(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        grid: Optional[np.ndarray] = None,
    ) -> bool:
        """두 월드 좌표 사이 선분에 장애물이 없으면 True. grid=None이면 raw self._grid_data."""
        if not self._have_grid:
            return True
        c0 = self._xy_to_ij(x0, y0)
        c1 = self._xy_to_ij(x1, y1)
        if c0 is None or c1 is None:
            return False
        g = grid if grid is not None else self._grid_data
        for ci, cj in self._bresenham_cells(c0[0], c0[1], c1[0], c1[1]):
            if ci < 0 or ci >= self._grid_h or cj < 0 or cj >= self._grid_w:
                return False
            if int(g[ci, cj]) >= self.occ_threshold:
                return False
        return True

    def _shortcut_path(self, path_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """LOS 기반 그리디 경로 단축. clearance>0이면 inflation 마진으로 LOS 체크(벽 근접 방지)."""
        if len(path_xy) <= 2:
            return list(path_xy)
        clearance = int(getattr(self, 'shortcut_los_clearance_cells', 1))
        los_grid = self._inflate_grid_temp(clearance) if clearance > 0 else None
        result = [path_xy[0]]
        i = 0
        while i < len(path_xy) - 1:
            # 가장 먼 j부터 시도
            j = len(path_xy) - 1
            while j > i + 1:
                if self._los_clear(path_xy[i][0], path_xy[i][1],
                                   path_xy[j][0], path_xy[j][1],
                                   grid=los_grid):
                    break
                j -= 1
            result.append(path_xy[j])
            i = j
        return result

    def _inflate_grid_temp(self, radius_cells: int) -> np.ndarray:
        """장애물을 radius_cells만큼 팽창한 임시 그리드 반환 (원본 _grid_data 불변)."""
        grid = self._grid_data  # 참조 한 번 캡처 (atomic in CPython)
        occ_mask = (grid >= self.occ_threshold)
        if radius_cells > 0:
            inflated = _ndimage.binary_dilation(occ_mask, iterations=radius_cells)
        else:
            inflated = occ_mask
        result = grid.copy()
        result[inflated & ~occ_mask] = int(self.occ_threshold)
        return result

    def _astar_with_inflation(
        self,
        start_ij: Tuple[int, int],
        goal_ij: Tuple[int, int],
        radius_cells: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """팽창 그리드로 A* 실행 (start/goal 셀은 force-clear)."""
        inflated = self._inflate_grid_temp(radius_cells)
        # start/goal 셀이 팽창으로 막혔을 경우 강제 해제
        si, sj = start_ij
        gi, gj = goal_ij
        inflated[si, sj] = min(inflated[si, sj], int(self.occ_threshold) - 1)
        inflated[gi, gj] = min(inflated[gi, gj], int(self.occ_threshold) - 1)
        return self._astar(start_ij, goal_ij, grid=inflated)

    def _smooth_path_bspline(
        self,
        path_xy: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """B-spline으로 경로 스무딩. 출력 점이 장애물 안이면 원본 반환."""
        # 3차 B-spline 은 제어점 4개가 하한. 파라미터가 더 크면 그쪽을 따른다
        # (호출부 게이트 smooth_bspline_skip_if_few_wps 와 단일 기준 유지).
        min_wps = max(4, int(getattr(self, 'smooth_bspline_skip_if_few_wps', 4)))
        if not _SCIPY_OK or len(path_xy) < min_wps:
            return path_xy
        try:
            xs = [p[0] for p in path_xy]
            ys = [p[1] for p in path_xy]
            s = float(getattr(self, 'smooth_bspline_s', 0.0))
            n_out = int(getattr(self, 'smooth_bspline_output_points', 50))
            tck, _ = _splprep([xs, ys], s=s, k=min(3, len(path_xy) - 1))
            u_fine = np.linspace(0.0, 1.0, n_out)
            out_x, out_y = _splev(u_fine, tck)
            result = list(zip(out_x.tolist(), out_y.tolist()))
            if self._have_grid:
                min_r = int(getattr(self, 'astar_min_inflation_cells', 1))
                if min_r > 0:
                    inflated_grid = self._inflate_grid_temp(min_r)
                    for (rx, ry) in result:
                        ij = self._xy_to_ij(rx, ry)
                        if ij is not None and inflated_grid[ij[0], ij[1]] >= self.occ_threshold:
                            return path_xy
                else:
                    for (rx, ry) in result:
                        ij = self._xy_to_ij(rx, ry)
                        if ij is not None and self._is_occ(*ij):
                            return path_xy
            result[-1] = path_xy[-1]
            return result
        except Exception as e:
            self.get_logger().warn(f"B-spline smoothing failed ({type(e).__name__}: {e}), returning original path.")
            return path_xy

    def _publish_path_msg(self) -> None:
        """현재 _path_xy를 nav_msgs/Path로 발행."""
        if not self._path_xy:
            return
        msg = NavPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        z = self._hold_z if self._hold_z is not None else 0.0
        for (x, y) in self._path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    # ── 경로 계획 ────────────────────────────────────────────────────────────

    def _plan_path(self) -> None:
        """A* 경로 계획. grid 없거나 실패 시 직선(goal)."""
        if self.odom is None or self.goal_xy is None:
            return
        px = float(self.odom.pose.pose.position.x)
        py = float(self.odom.pose.pose.position.y)
        gx, gy = float(self.goal_xy[0]), float(self.goal_xy[1])
        if self._have_grid:
            s_ij = self._xy_to_ij(px, py)
            g_ij = self._xy_to_ij(gx, gy)
            self.get_logger().debug(
                f"[ASTAR] start_xy=({px:.2f},{py:.2f}) -> ij={s_ij}, "
                f"goal_xy=({gx:.2f},{gy:.2f}) -> ij={g_ij}"
            )
            if not s_ij:
                if self.allow_diag:
                    self.get_logger().warn(
                        f"Path fallback (straight): start ({px:.1f},{py:.1f}) outside grid "
                        f"[bounds x:[{self._grid_ox:.1f},{self._grid_ox+self._grid_w*self._grid_res:.1f}] "
                        f"y:[{self._grid_oy:.1f},{self._grid_oy+self._grid_h*self._grid_res:.1f}]]"
                    )
            elif not g_ij:
                if self.allow_diag:
                    self.get_logger().warn(
                        f"Path fallback (straight): goal ({gx:.1f},{gy:.1f}) outside grid "
                        f"[bounds x:[{self._grid_ox:.1f},{self._grid_ox+self._grid_w*self._grid_res:.1f}] "
                        f"y:[{self._grid_oy:.1f},{self._grid_oy+self._grid_h*self._grid_res:.1f}]]"
                    )
            else:
                # start/goal이 inflation으로 occupied여도 force-clear 후 A* 시도 (우회 경로 확보)
                si, sj = s_ij
                gi, gj = g_ij
                if self._is_occ(*s_ij):
                    if self.allow_diag:
                        val = int(self._grid_data[s_ij])
                        self.get_logger().warn(
                            f"A* start cell occupied (ij={s_ij}, val={val}): "
                            f"force-clearing start for A* (inflation artifact)"
                        )
                if self._is_occ(*g_ij):
                    if self.allow_diag:
                        val = int(self._grid_data[g_ij])
                        self.get_logger().warn(
                            f"A* goal cell occupied (ij={g_ij}, val={val}): "
                            f"force-clearing goal for A* (inflation artifact)"
                        )
                self._plan_s_ij = s_ij
                self._plan_g_ij = g_ij
                try:
                    # min_r → 0 점진적 fallback으로 좁은 통로에서도 경로 보장
                    min_r = int(getattr(self, 'astar_min_inflation_cells', 1))
                    path_ij = None
                    for r in range(min_r, -1, -1):
                        path_ij = self._astar_with_inflation(s_ij, g_ij, radius_cells=r)
                        if path_ij:
                            if r < min_r:
                                self.get_logger().warn(
                                    f"A* min_inflation reduced {min_r} → {r} (narrow passage)"
                                )
                            break

                    if path_ij:
                        path_xy = [self._ij_to_xy(ci, cj) for ci, cj in path_ij]
                        path_xy = self._shortcut_path(path_xy)           # LOS shortcutting
                        skip_n = int(getattr(self, 'smooth_bspline_skip_if_few_wps', 4))
                        smoothed = False
                        if self.smooth_bspline_enabled and len(path_xy) >= skip_n:
                            sm = self._smooth_path_bspline(path_xy)      # B-spline smoothing
                            # 기각(장애물 충돌·점 부족·예외) 시 입력 객체를 그대로 반환하므로
                            # 동일성으로 "실제 적용 여부"를 판정한다.
                            smoothed = sm is not path_xy
                            path_xy = sm
                        # goal 좌표 고정 후 저장 (_path_lock으로 tick 스레드와 race 방지)
                        path_xy[-1] = (gx, gy)
                        with self._path_lock:
                            self._path_xy = path_xy
                            self._path_wp_idx = 0
                        self.get_logger().info(
                            f"A* succeeded: {len(path_xy)} wps "
                            f"(smoothed={smoothed}), "
                            f"start_ij={s_ij}, goal_ij={g_ij}"
                        )
                        self._publish_path_msg()
                        return
                    if self.allow_diag:
                        self.get_logger().warn(
                            f"Path fallback (straight): A* failed to find path (start={s_ij}, goal={g_ij})"
                        )
                finally:
                    self._plan_s_ij = None
                    self._plan_g_ij = None
        else:
            if self.allow_diag:
                self.get_logger().warn(
                    f"Path fallback (straight): no grid (_have_grid=False)"
                )
        # fallback: 직선 경로(goal)로 대체 (_path_lock으로 tick 스레드와 race 방지)
        self.get_logger().warn("Path fallback (straight): using direct goal line.")
        with self._path_lock:
            self._path_xy = [(gx, gy)]
            self._path_wp_idx = 0
        self._publish_path_msg()

    def _tick_replan(self) -> None:
        """주기적 경로 재계획 (replan 전용 스레드). tick 스레드 직접 호출은 race 위험으로 금지."""
        if not (self.nav_mode in ("hybrid", "astar")
                and self.odom is not None
                and self.goal_xy is not None
                and not self._goal_reached):
            return
        self._plan_path()

    def _path_current_wp(self) -> Optional[Tuple[float, float]]:
        with self._path_lock:
            if not self._path_xy or self._path_wp_idx >= len(self._path_xy):
                return None
            return self._path_xy[self._path_wp_idx]

    def _path_lookahead_wp_at_dist(
        self, px: float, py: float, lookahead_dist: float
    ) -> Optional[Tuple[float, float]]:
        """경로상 lookahead_dist(m) 앞 지점을 거리 기반 interpolation하여 반환.
        거리 기반은 경로 wp 수와 무관하게 일정 거리 앞을 가리킴 (step-count의 goal 점프 문제 회피)."""
        with self._path_lock:
            if not self._path_xy:
                return None
            path = self._path_xy
            n = len(path)
            start_idx = max(0, min(self._path_wp_idx, n - 1))

        # 이미 통과한 시작 세그먼트 건너뛰기: dot product<0(사이각>90°)이면 start_idx advance.
        # (lookahead가 후방 점을 반환해 180° 회전을 유발하는 버그 차단)
        while start_idx + 1 < n:
            ax, ay = path[start_idx]
            bx, by = path[start_idx + 1]
            if (ax - px) * (bx - ax) + (ay - py) * (by - ay) < 0.0:
                start_idx += 1
            else:
                break

        # 현재 위치 → path[start_idx] 까지의 거리를 첫 번째 세그먼트로 포함
        prev_x, prev_y = px, py
        accum = 0.0
        for i in range(start_idx, n):
            wp_x, wp_y = path[i]
            seg = math.hypot(wp_x - prev_x, wp_y - prev_y)
            if seg < 1e-6:
                prev_x, prev_y = wp_x, wp_y
                continue
            if accum + seg >= lookahead_dist:
                t = (lookahead_dist - accum) / seg
                return (prev_x + t * (wp_x - prev_x), prev_y + t * (wp_y - prev_y))
            accum += seg
            prev_x, prev_y = wp_x, wp_y

        # 잔여 경로가 lookahead_dist보다 짧으면 start_idx보다 한 발 앞을 반환 (정지 데드락 방지)
        return path[min(n - 1, start_idx + 1)]

    def _path_advance_wp(self, px: float, py: float):
        """가장 가까운 경로 세그먼트에 수직 투영해 wp_idx 갱신 (측방 이탈에 강건)."""
        with self._path_lock:
            if not self._path_xy or self._path_wp_idx >= len(self._path_xy):
                return
            n = len(self._path_xy)
            search_window = int(getattr(self, 'wp_advance_search_window', 10))
            search_end = min(n - 1, self._path_wp_idx + search_window)
            best_seg_idx = self._path_wp_idx
            best_t       = 0.0
            best_dist    = float('inf')
            for i in range(self._path_wp_idx, search_end):
                ax, ay = self._path_xy[i]
                bx, by = self._path_xy[i + 1]
                dx, dy = bx - ax, by - ay
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-10:
                    continue
                t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                cx = ax + t * dx
                cy = ay + t * dy
                d  = math.hypot(px - cx, py - cy)
                if d < best_dist:
                    best_dist    = d
                    best_seg_idx = i
                    best_t       = t
            if best_t > 0.5 and best_seg_idx + 1 < n:
                self._path_wp_idx = max(self._path_wp_idx, best_seg_idx + 1)
            else:
                self._path_wp_idx = max(self._path_wp_idx, best_seg_idx)

    def _tick_global(self, px: float, py: float, yaw: float, now: float) -> Tuple[float, float, float, float]:
        """Global planning: lookahead wp 방향 heading 정렬 후 target 방향 velocity (local/world ENU 기준)."""
        self._path_advance_wp(px, py)
        wp = self._path_lookahead_wp_at_dist(px, py, self.hybrid_lookahead_dist_m)
        curr_wp = self._path_current_wp()

        if now - self._t_last_debug_log >= 1.0:
            self.get_logger().debug(
                f"[GLOBAL_WP] curr_idx={self._path_wp_idx}, "
                f"curr_wp={curr_wp}, lookahead_wp={wp}, "
                f"robot=({px:.2f},{py:.2f}), path_len={len(self._path_xy)}"
            )
            self._t_last_debug_log = now

        if wp is None:
            gx, gy = self.goal_xy[0] - px, self.goal_xy[1] - py
        else:
            gx, gy = wp[0] - px, wp[1] - py
        gnorm = math.hypot(gx, gy)
        if gnorm < 1e-4:
            return 0.0, 0.0, self._vz_correction(), 0.0
        target_yaw = math.atan2(gy, gx)
        yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        # yaw 오차 비례 회전 + 오차 클수록 감속 (정렬 데드존 없음, 코너에서 자연스러운 꺾임)
        planner_yaw_limit = float(getattr(self, 'planner_max_yaw_rate', self.max_yaw_rate))
        yaw_rate = max(-planner_yaw_limit, min(planner_yaw_limit, 2.0 * yaw_err))
        align_scale = max(0.0, 1.0 - abs(yaw_err) / (math.pi / 2))
        v_cmd = min(self.v_max, 0.8 * gnorm) * align_scale
        vx = v_cmd * math.cos(target_yaw)
        vy = v_cmd * math.sin(target_yaw)
        return vx, vy, self._vz_correction(), yaw_rate

    def _tick(self):
        """20Hz 제어 루프: 연결/OFFBOARD/ARM 확보 → 비상제동 → nav_mode별 제어 → setpoint 발행."""
        now = time.time()
        if self.mav_state is None or not self.mav_state.connected:
            return
        # homing 중에는 velocity 미발행 (course_goal_publisher의 position setpoint와 충돌 방지)
        if self.wait_for_homing and not self._homing_done:
            return
        if self._pre_setpoint_count < self._pre_setpoint_needed:
            self._pre_setpoint_count += 1
            self._publish()
            return
        if self.mav_state.mode != "OFFBOARD":
            if now - self._t_last_mode_req > self.mode_retry_sec:
                self._t_last_mode_req = now
                if self._cli_mode.service_is_ready():
                    req = SetMode.Request()
                    req.base_mode, req.custom_mode = 0, "OFFBOARD"
                    self._cli_mode.call_async(req)
                    self.get_logger().info("Requested OFFBOARD.")
            self._publish()
            return
        if not self.mav_state.armed:
            if now - self._t_last_arm_req > self.arm_retry_sec:
                self._t_last_arm_req = now
                if self._cli_arm.service_is_ready():
                    req = CommandBool.Request()
                    req.value = True
                    self._cli_arm.call_async(req)
                    self.get_logger().info("Requested ARM.")
            self._publish()
            return

        if not self._sensors_ok():
            self._publish()
            return

        px = float(self.odom.pose.pose.position.x)
        py = float(self.odom.pose.pose.position.y)
        dist = math.hypot(self.goal_xy[0] - px, self.goal_xy[1] - py)
        if self._goal_reached or dist < self.goal_radius:
            if not self._goal_reached:
                self._goal_reached = True
                self.get_logger().info(f"Goal reached (dist={dist:.2f}m). Complete stop.")
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        yaw = quat_to_yaw(self.odom.pose.pose.orientation)

        half = getattr(self, "rl_front_deg", 90.0) / 2.0

        # Emergency brake: 정면 최소거리 ≤ 임계값이면 회전+후퇴. use_override로 body 자기감지 필터.
        d_front_min = self._dist_in_sector(-half, half, "min", use_override=True)
        if d_front_min <= self.emergency_brake_dist:
            d_left  = self._dist_in_sector(self.emergency_brake_deg, 90.0, "min", use_override=False)
            d_right = self._dist_in_sector(-90.0, -self.emergency_brake_deg, "min", use_override=False)
            self._brake_count += 1

            # 회전 방향 결정: 진입 시 1회만 결정하고 brake 해제 전까지 고정 (방향 진동 방지)
            if self._brake_rotate_left is None:
                rotate_reason = "lidar"
                rotate_left = (d_left >= d_right)
                wp_la = self._path_lookahead_wp_at_dist(px, py, self.hybrid_lookahead_dist_m)
                if wp_la is not None:
                    wp_angle_world = math.atan2(wp_la[1] - py, wp_la[0] - px)
                    # body frame 기준 상대각: +이면 왼쪽, -이면 오른쪽
                    angle_diff = (wp_angle_world - yaw + math.pi) % (2 * math.pi) - math.pi
                    rotate_left = (angle_diff >= 0.0)
                    rotate_reason = "astar"
                self._brake_rotate_left = rotate_left
            else:
                rotate_left = self._brake_rotate_left
                rotate_reason = "locked"

            yaw_rate = 0.0 if self._brake_count > 30 else (
                self.max_yaw_rate if rotate_left else -self.max_yaw_rate
            )
            if not self._emergency_brake_active:
                self._emergency_brake_active = True
                self.get_logger().info(
                    f"Emergency brake: d_front={d_front_min:.2f}m <= {self.emergency_brake_dist}, "
                    f"rotating {'left' if rotate_left else 'right'} (src={rotate_reason})"
                )
            d_rear = self._dist_in_sector(
                180.0 - self.emergency_brake_deg,
                180.0 + self.emergency_brake_deg, "min", use_override=False)
            retreat_speed = 0.4 if d_rear > 0.8 else 0.0
            vx_r = -retreat_speed * math.cos(yaw)
            vy_r = -retreat_speed * math.sin(yaw)
            self._publish(vx_r, vy_r, self._vz_correction(), yaw_rate)
            return
        self._emergency_brake_active = False
        self._brake_count = 0
        self._brake_rotate_left = None

        # nav_mode별 제어 로직
        if self.nav_mode == "astar":
            vx, vy, vz, yaw_rate = self._tick_astar_mode(px, py, yaw, now)
        elif self.nav_mode == "ppo":
            vx, vy, vz, yaw_rate = self._tick_ppo_mode(yaw)
        else:  # "hybrid"
            vx, vy, vz, yaw_rate = self._tick_hybrid_mode(px, py, yaw)

        self._publish(vx, vy, vz, yaw_rate)

    def _tick_astar_mode(self, px: float, py: float, yaw: float, now: float) -> Tuple[float, float, float, float]:
        """A* 단독 모드: global planner(_tick_global)만 사용. PPO 없음."""
        if not self._path_xy:
            self._plan_path()
        return self._tick_global(px, py, yaw, now)

    def _tick_ppo_mode(self, yaw: float) -> Tuple[float, float, float, float]:
        """PPO 단독 모드: 최종 goal 방향 매 tick 추론. PPO 출력을 가공 없이 그대로 발행."""
        obs = self._build_obs()
        if obs is None:
            return 0.0, 0.0, self._vz_correction(), 0.0
        action = self._infer(obs)
        v_cmd, yaw_rate = self._action_to_vel(action)
        vx, vy = self._compute_cmd_vel(v_cmd, yaw)
        return vx, vy, self._vz_correction(), yaw_rate

    def _tick_hybrid_mode(self, px: float, py: float, yaw: float) -> Tuple[float, float, float, float]:
        """Hybrid 모드: A* 경로 lookahead를 obs로 PPO 매 tick 추론 → v_cmd·yaw_rate.
        이동은 현재 heading 방향, heading은 PPO yaw_rate로 제어 (학습 unicycle 정합)."""
        with self._path_lock:
            have_path = bool(self._path_xy)
        if have_path:
            self._path_advance_wp(px, py)
        else:
            self._plan_path()

        obs = self._build_obs()
        if obs is None:
            return 0.0, 0.0, self._vz_correction(), 0.0
        action = self._infer(obs)
        v_cmd, yaw_rate = self._action_to_vel(action)
        vx = v_cmd * math.cos(yaw)
        vy = v_cmd * math.sin(yaw)
        return vx, vy, self._vz_correction(), yaw_rate


def main(args=None):
    rclpy.init(args=args)
    node = RLOffboardNode()
    # tick / replan / sensor 콜백 그룹을 각 스레드로 분리 (스레드 수 = callback group 수)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
