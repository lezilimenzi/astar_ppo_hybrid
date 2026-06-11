#!/usr/bin/env python3
"""RL Offboard: obs(lidar+goal_rel_body) -> policy -> velocity setpoint (local/world ENU). OFFBOARD/ARM 자동."""

from __future__ import annotations

import concurrent.futures
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

        # Params (declare + get in one pass)
        params = [
            ("mavros_state_topic", "/mavros/state"),
            ("arm_mask_angles_deg", [59.4, 123.0, 237.0, 301.0]),  # #2: Iris 로터 암 방향 (iris.sdf 실측)
            ("arm_mask_half_width_deg", 12.0),                       # #2: 각 암 ±12°
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
            # nav_mode: "hybrid" (A* waypoint → PPO 항상) | "astar" (A* 단독) | "ppo" (PPO 단독)
            ("nav_mode", "hybrid"),
            ("grid_topic", "/local_grid"),
            ("occ_threshold", 50),
            ("allow_diag", True),
            # replan_period: A* 전역경로 재계획 주기(s). 1.0s 근거:
            #   (1) 연산 안전마진 — A* 최악 ~50ms, 전용 스레드(_cb_group_replan)에서 실행되어
            #       주기(1000ms) ≫ 계산시간(~50ms) → 콜백 중첩/큐 적체 없음
            #   (2) 경로 신선도 — v_max=0.8m/s → 1s당 최대 0.8m 이동. 장애물 크기(1.5~3m)
            #       대비 1s 내 경로 무효화량이 작아 staleness가 bounded
            ("replan_period", 1.0),
            ("path_lookahead_steps", 12),  # global planner: 현재 wp보다 N칸 앞을 target으로. 12 = 3.0m/0.25m(학습 env cell_size)
            ("planner_max_yaw_rate", 1.0),  # global planner 전용 yaw rate 상한 (rad/s)
            ("wp_advance_search_window", 10),  # _path_advance_wp 세그먼트 투영 탐색 창 (wp 수)
            ("rl_front_deg", 90.0),
            # LiDAR 구간 거리: min=보수적, mean/median=스파이크 완화(비상 제동은 항상 min)
            ("lidar_sector_agg", "mean"),
            # 드론 기체/로터 암 자기 감지 필터: scan.range_min보다 크면 이 값으로 override
            ("lidar_range_min_override", 0.0),
            ("emergency_brake_dist", 0.0),
            ("emergency_brake_deg", 60.0),
            ("heading_align_tol", 0.26),
            ("goal_bias_dist", 2.5),
            ("goal_bias_yaw_scale", 0.25),
            # ppo 모드 PPO 추론 주기 (tick). 학습 dt=0.3s @ 20Hz tick = 6 (학습 매칭). 1=매 tick(현행, zigzag), 12=보수적.
            ("ppo_frame_skip", 6),
            ("enable_altitude_hold", True),
            ("altitude_hold_margin", 0.1),
            ("altitude_hold_gain", 0.4),
            ("mode_retry_sec", 1.0),
            ("arm_retry_sec", 1.0),
            ("goal_timeout_sec", 0.0),  # 0.0 = disabled
            # --- TopoTraj 스타일 경로 개선 ---
            ("use_topo_diversity",             False),
            ("astar_min_inflation_cells",      2),        # 경로 최소 벽 여유(셀). 좁은 통로 막히면 자동 감소
            ("topo_inflation_radii_cells",     [0, 1, 2]),
            ("topo_homotopy_samples",          10),
            ("shortcut_los_clearance_cells",   0),
            # hybrid 모드 PPO obs용 경로 lookahead 거리 (m).
            # norm_scale은 학습값(3.0)으로 고정 → policy 입력 분포 유지.
            # R 코스처럼 좁은 통로 직전에서 3m 앞 지점이 통로 너머로 꺾이는 경우
            # 더 가까운 지점을 target으로 삼아 벽 방향 velocity 성분 억제.
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

        # Resolve model paths
        # nav_mode별 default 파일명:
        #   "hybrid" → policy_c2_ts.pt   / obs_norm_c2.json   (C2 ablation 모델, 10M step)
        #   "ppo"    → policy_ppo_ts.pt  / obs_norm_ppo.json  (PPO 단독 모드용)
        #   "astar"  → policy 불필요
        policy_path = self.policy_path or ""
        obs_norm_path = self.obs_norm_path or ""
        if self.nav_mode != "astar" and (not policy_path or not obs_norm_path):
            if self.nav_mode == "hybrid":
                default_policy  = "policy_c2dt05_ts.pt"
                default_obs_norm = "obs_norm_c2dt05.json"
            else:  # "ppo"
                default_policy  = "policy_ppo_c2dt05_ts.pt"
                default_obs_norm = "obs_norm_ppo_c2dt05.json"
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
        # #5: sim clock 기반 초기화 (use_sim_time=True 환경에서 Gazebo 속도 저하 시 오탐 방지)
        self.t_last_odom = self.get_clock().now()
        self.t_last_scan = self.get_clock().now()
        self.t_last_goal = 0.0  # goal은 goal_timeout_sec=0 (무제한)이므로 float 유지
        self._pre_setpoint_count = 0
        self._pre_setpoint_needed = int(max(0, self.pre_setpoint_sec) * self.setpoint_rate_hz)
        self._t_last_mode_req = 0.0
        self._t_last_arm_req = 0.0
        self._homing_done = not self.wait_for_homing
        self._goal_reached = False
        self._scan_warn_last = 0.0   # scan 미수신 진단 로그 쓰로틀
        self._emergency_brake_active = False  # emergency brake 진입 시에만 로그
        self._filtered_ranges: Optional[np.ndarray] = None  # #2: arm mask 적용된 캐시
        self._brake_count = 0                               # #3: 연속 brake 틱 카운터
        self._brake_rotate_left: Optional[bool] = None      # brake 진입 시 1회 결정, 해제 전 고정
        self._path_xy: List[Tuple[float, float]] = []
        self._path_wp_idx = 0
        self._have_grid = False
        self._grid_data: Optional[np.ndarray] = None
        self._grid_ox = self._grid_oy = 0.0
        self._grid_res = 0.1
        self._grid_w = self._grid_h = 0
        self._hold_z: Optional[float] = None  # Altitude hold reference
        self._plan_robot_xy: Optional[Tuple[float, float]] = None
        self._plan_s_ij: Optional[Tuple[int, int]] = None
        self._plan_g_ij: Optional[Tuple[int, int]] = None
        self._t_last_debug_log = 0.0
        self._yaw_sat_count = 0
        # ppo 모드 frame skip: 학습 dt(0.3s)=20Hz×6 매칭. PPO 추론·action 갱신만 다운샘플.
        # _tick_ppo_mode 한정 (astar/hybrid 무영향).
        self._ppo_skip_counter = 0
        self._cached_ppo_action: Optional[np.ndarray] = None

        # QoS
        q_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        # /scan: Gazebo RPLidar(ray_sensor)는 qos_profile_sensor_data(BEST_EFFORT+VOLATILE) 사용.
        # QoS 불일치 시 메시지 미수신 → obs=None → RL 미동작. 명시적 매칭으로 호환성 보장.
        q_scan = qos_profile_sensor_data
        q_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
        q_setpoint = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        # #9: callback groups — tick/replan 스레드 분리로 A* 블로킹 방지
        # tick:   MutuallyExclusive → 20Hz 제어 루프 단일 실행 보장
        # replan: MutuallyExclusive → A* 재계획 단일 실행 보장 (tick과 별도 스레드)
        # sensor: Reentrant          → 센서 콜백은 빠른 저장만, 동시 실행 허용
        self._cb_group_tick   = MutuallyExclusiveCallbackGroup()
        self._cb_group_replan = MutuallyExclusiveCallbackGroup()
        self._cb_group_sensor = ReentrantCallbackGroup()

        # #9: _path_xy / _path_wp_idx 를 tick ↔ replan 스레드 간 race condition으로부터 보호
        # RLock: 같은 스레드 내 재진입 허용 (예: _path_advance_wp → _path_current_wp 체인)
        self._path_lock = threading.RLock()

        # _grid_lock: _cb_grid(sensor thread) ↔ _plan_path(replan thread) 간 race 방지
        # _grid_h/_grid_w/_grid_ox/_grid_oy/_grid_data 를 원자적으로 갱신/읽기
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
        self.t_last_odom = self.get_clock().now()  # #5: sim clock

    def _filter_arm_angles(self, ranges: np.ndarray) -> np.ndarray:
        """#2: 로터 암 방향 빔을 max_range로 대체 (numpy 벡터화).
        거리 기반 필터(lidar_range_min_override) 대신 각도 마스크로 자기감지 제거."""
        if self.scan is None:
            return ranges
        out = ranges.copy()
        ang_min = float(self.scan.angle_min)
        inc = float(self.scan.angle_increment)
        n = len(out)
        beam_angles = ang_min + np.arange(n, dtype=np.float32) * inc
        hw = math.radians(float(self.arm_mask_half_width_deg))
        for arm_deg in self.arm_mask_angles_deg:
            arm_rad = math.radians(float(arm_deg))
            diff = np.abs(np.arctan2(
                np.sin(beam_angles - arm_rad),
                np.cos(beam_angles - arm_rad)
            ))
            out[diff < hw] = self.max_range
        return out

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg
        self.t_last_scan = self.get_clock().now()  # #5: sim clock
        # #2: scan 수신 시 1회 arm 마스크 적용 → _dist_in_sector/_build_obs 공유
        raw = np.array(msg.ranges, dtype=np.float32)
        self._filtered_ranges = self._filter_arm_angles(raw)

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
            self.t_last_goal = time.time()
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
        # 모드별 yaw_rate clamp:
        #   astar: planner_max_yaw_rate (기하학적 P-controller)
        #   hybrid (v3 방식 C) / ppo: max_yaw_rate (PPO action[1] × max_yaw_rate, 학습 분포 보호)
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
        # #5: sim clock 기반 — Gazebo 속도 저하 시 wall clock 오탐 방지
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
        if self.goal_timeout_sec > 0 and (time.time() - self.t_last_goal) > self.goal_timeout_sec:
            return False
        return True

    def _build_obs(self) -> Optional[np.ndarray]:
        if self.odom is None or self.scan is None or self.goal_xy is None:
            return None
        # #2: arm mask 캐시 우선 사용 (scan 콜백에서 1회 계산)
        if self._filtered_ranges is not None and len(self._filtered_ranges) == len(self.scan.ranges):
            raw = self._filtered_ranges.copy()
        else:
            raw = np.array(self.scan.ranges, dtype=np.float32)
        rmin = float(getattr(self.scan, "range_min", 0.0))
        override = float(getattr(self, "lidar_range_min_override", 0.0) or 0.0)
        # RL 관측: 자기감지 필터(override) 적용. 비상 제동은 _dist_in_sector(..., use_override=False).
        if override > rmin:
            rmin = override
        raw = np.where(np.isfinite(raw) & (raw > rmin), raw, self.max_range)
        ranges = sanitize_lidar(raw, self.max_range)
        if len(ranges) != self.obs_beams:
            ranges = downsample_lidar(ranges, self.obs_beams, self.downsample_mode)
        lidar = np.clip(ranges / self.max_range, 0.0, 1.0) * 2.0 - 1.0
        px, py = float(self.odom.pose.pose.position.x), float(self.odom.pose.pose.position.y)
        yaw = quat_to_yaw(self.odom.pose.pose.orientation)

        # #7: scan-odom dead reckoning 보정
        # scan(8Hz)과 odom(22Hz) 타임스탬프 차이만큼 위치/yaw 보정 → goal_rel 정확도 향상
        # 최대 보정량: dt=45ms, v=1.5m/s → Δpos≈0.068m (0.2m 셀 기준 유의미)
        if self.scan is not None and self.scan.header.stamp.sec != 0:
            scan_t = rclpy.time.Time.from_msg(self.scan.header.stamp).nanoseconds / 1e9
            odom_t = rclpy.time.Time.from_msg(self.odom.header.stamp).nanoseconds / 1e9
            dt = scan_t - odom_t
            if abs(dt) > 0.01:  # 10ms 이상 차이 시 보정 (원안 50ms → 10ms, 22Hz odom 기준 실질 동작)
                vx_b = self.odom.twist.twist.linear.x
                vy_b = self.odom.twist.twist.linear.y
                wz   = self.odom.twist.twist.angular.z
                vx_w = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
                vy_w = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
                px  += vx_w * dt
                py  += vy_w * dt
                yaw += wz   * dt

        # nav_mode별 goal_rel 계산:
        #   "hybrid": 경로상 hybrid_lookahead_dist_m 앞 지점 (distance 기반 interpolation)
        #             step-count 기반(_path_lookahead_wp)은 3 wps 경로에서 즉시 goal로
        #             점프해 routing point를 무시하는 문제가 있어 distance 기반으로 교체.
        #             norm_scale은 학습값(3.0) 고정: lookahead 거리와 분리해
        #             policy 입력 분포를 유지하면서 target 방향만 조정.
        #   "ppo"   : 최종 goal / world_size (기존 policy와 일치)
        LOOKAHEAD_NORM_SCALE = 3.0  # 학습 시 norm_scale — policy 입력 분포 유지를 위해 고정
        lookahead_dist = float(getattr(self, "hybrid_lookahead_dist_m", 1.5))
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
        # cmd_vel_unstamped 실험 결과: local/world 기준 속도로 해석됨
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

    def _escape_if_spinning(self, yaw_rate: float, yaw: float) -> Optional[Tuple[float, float, float, float]]:
        """요레이트 포화가 길게 지속되면 소량 후진으로 탈출."""
        if abs(yaw_rate) >= self.max_yaw_rate * 0.95:
            self._yaw_sat_count += 1
        else:
            self._yaw_sat_count = 0
        if self._yaw_sat_count > 20:
            self._yaw_sat_count = 0
            return (-0.3 * math.cos(yaw), -0.3 * math.sin(yaw), self._vz_correction(), 0.0)
        return None

    def _dist_in_sector(self, low_deg: float, high_deg: float, agg: str, use_override: bool = True) -> float:
        """LiDAR body frame 각도 구간(deg) 거리 집계. agg: min|median|mean.
        use_override=False면 lidar_range_min_override를 rmin에 반영하지 않음(비상 제동 등)."""
        if self.scan is None or not self.scan.ranges:
            return self.max_range
        if self._filtered_ranges is not None and len(self._filtered_ranges) == len(self.scan.ranges):
            ranges = self._filtered_ranges.copy()
        else:
            ranges = np.array(self.scan.ranges, dtype=np.float32)
        rmin = float(getattr(self.scan, "range_min", 0.0))
        override = float(getattr(self, "lidar_range_min_override", 0.0) or 0.0)
        if use_override and override > rmin:
            rmin = override
        ranges = np.where(np.isfinite(ranges) & (ranges > rmin), ranges, self.max_range)
        ang_min = float(self.scan.angle_min)
        inc = float(self.scan.angle_increment)
        # [0, 2π) 공간에서 비교: atan2의 (-π, π] 범위가 high_deg>180° 섹터를
        # (예: rear 150°~210°) 놓치는 wrap-around 버그 수정.
        _TWO_PI = 2.0 * math.pi
        low_pos  = math.radians(low_deg)  % _TWO_PI   # [0, 2π)
        high_pos = math.radians(high_deg) % _TWO_PI   # [0, 2π)
        vals: List[float] = []
        for i in range(len(ranges)):
            ang_pos = (ang_min + i * inc) % _TWO_PI   # [0, 2π)
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

    def _min_dist_in_sector(self, low_deg: float, high_deg: float) -> float:
        """구간 최소 거리 (비상 제동 등 보수적 경로용). override 미적용."""
        return self._dist_in_sector(low_deg, high_deg, "min", use_override=False)

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
        """A* 전용: occ≥threshold 막힘 + (옵션) 로봇 근처 unknown 막힘. 시작/목표 셀은 unknown만으로는 막지 않음.

        grid: 명시적으로 전달된 그리드 배열. None이면 self._grid_data 사용.
              병렬 A* 실행 시 각 스레드가 독립적인 grid를 전달해야 race condition을 피할 수 있음.
        """
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
        """A* 경로 탐색.

        grid: 탐색에 사용할 점유 격자. None이면 self._grid_data 사용.
              병렬 호출 시 각 호출자가 독립적인 grid를 전달하면 self._grid_data를
              수정하지 않으므로 thread-safe하게 동작함.
        """
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

    # ── TopoTraj 스타일 경로 개선 ──────────────────────────────────────────

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
        """두 월드 좌표 사이의 선분에 장애물이 없으면 True.

        grid: 사전에 팽창된 그리드. None이면 raw self._grid_data 사용.
              _shortcut_path에서 inflation clearance를 적용할 때 전달함.
        """
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
        """LOS 기반 그리디 경로 단축: 시선이 통하는 가장 먼 점으로 바로 연결.

        shortcut_los_clearance_cells > 0이면 A* inflation과 동일한 마진으로 LOS 체크.
        clearance=0(기존)이면 raw grid만 체크해 벽 근접 단축이 허용됐음.
        """
        if len(path_xy) <= 2:
            return list(path_xy)
        clearance = int(getattr(self, 'shortcut_los_clearance_cells', 0))
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
        """장애물을 radius_cells 셀만큼 팽창한 임시 그리드 반환.

        _grid_data 참조를 한 번만 캡처: _cb_grid(sensor thread)가 중간에
        _grid_data를 교체해도 occ_mask와 result가 동일 배열 기준으로 계산됨.
        """
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
        """팽창된 그리드로 A* 실행.

        self._grid_data를 수정하지 않고 팽창 그리드를 _astar에 직접 전달하므로
        여러 radius에 대해 thread-safe하게 병렬 호출 가능.
        """
        inflated = self._inflate_grid_temp(radius_cells)
        # start/goal 셀이 팽창으로 막혔을 경우 강제 해제
        si, sj = start_ij
        gi, gj = goal_ij
        inflated[si, sj] = min(inflated[si, sj], int(self.occ_threshold) - 1)
        inflated[gi, gj] = min(inflated[gi, gj], int(self.occ_threshold) - 1)
        return self._astar(start_ij, goal_ij, grid=inflated)

    def _astar_diverse(
        self,
        start_ij: Tuple[int, int],
        goal_ij: Tuple[int, int],
    ) -> List[List[Tuple[int, int]]]:
        """다양한 팽창 반경으로 여러 A* 경로를 병렬 탐색.

        _astar_with_inflation이 self._grid_data를 수정하지 않으므로
        ThreadPoolExecutor로 race condition 없이 동시 실행 가능.
        (Python GIL로 인해 CPU 바운드 완전 병렬은 아니나, I/O 대기 없이
        numpy 연산이 GIL을 부분 해제해 실질적인 중첩 실행 효과가 있음)
        """
        min_r = int(getattr(self, 'astar_min_inflation_cells', 2))
        base_radii = list(self.topo_inflation_radii_cells) if hasattr(self, 'topo_inflation_radii_cells') else [0, 1, 2]

        # min_r 기반으로 탐색. 좁은 통로에서 경로 없으면 base를 1씩 줄여 재시도
        for base in range(min_r, -1, -1):
            radii = [r + base for r in base_radii]

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(radii)) as executor:
                future_to_radius = {
                    executor.submit(self._astar_with_inflation, start_ij, goal_ij, r): r
                    for r in radii
                }
                paths_unordered = []
                for fut in concurrent.futures.as_completed(future_to_radius):
                    p = fut.result()
                    if p is not None:
                        paths_unordered.append((future_to_radius[fut], p))

            # radius 순서대로 정렬해 결과 재현성 유지
            paths_unordered.sort(key=lambda x: x[0])

            results: List[List[Tuple[int, int]]] = []
            seen: set = set()
            for _, p in paths_unordered:
                key = (p[0], p[-1], len(p))
                if key not in seen:
                    seen.add(key)
                    results.append(p)

            if results:
                if base < min_r:
                    self.get_logger().warn(
                        f"A* diverse: min_inflation reduced {min_r} → {base} (narrow passage)"
                    )
                return results

        # 최후 fallback: inflation 없는 bare A*
        fallback = self._astar(start_ij, goal_ij)
        return [fallback] if fallback else []

    def _sample_path_at_fractions(
        self,
        path_xy: List[Tuple[float, float]],
        n: int,
    ) -> List[Tuple[float, float]]:
        """arc-length 기반 등간격 n개 샘플."""
        if len(path_xy) < 2:
            return list(path_xy)
        pts = np.array(path_xy, dtype=np.float64)
        dists = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
        total = dists[-1]
        if total < 1e-9:
            return [path_xy[0]] * n
        fracs = np.linspace(0.0, total, n)
        xs = np.interp(fracs, dists, pts[:, 0])
        ys = np.interp(fracs, dists, pts[:, 1])
        return list(zip(xs.tolist(), ys.tolist()))

    def _paths_same_homotopy(
        self,
        path_a: List[Tuple[float, float]],
        path_b: List[Tuple[float, float]],
    ) -> bool:
        """두 경로가 같은 homotopy class이면 True (대응 점들이 모두 LOS)."""
        n = int(getattr(self, 'topo_homotopy_samples', 10))
        samps_a = self._sample_path_at_fractions(path_a, n)
        samps_b = self._sample_path_at_fractions(path_b, n)
        for (ax, ay), (bx, by) in zip(samps_a, samps_b):
            if not self._los_clear(ax, ay, bx, by):
                return False
        return True

    def _filter_homotopy_classes(
        self,
        paths_xy: List[List[Tuple[float, float]]],
    ) -> List[List[Tuple[float, float]]]:
        """위상적으로 중복된 경로 제거 — 각 homotopy class에서 대표 1개만 유지."""
        kept: List[List[Tuple[float, float]]] = []
        for p in paths_xy:
            duplicate = any(self._paths_same_homotopy(p, k) for k in kept)
            if not duplicate:
                kept.append(p)
        return kept if kept else [paths_xy[0]]

    def _smooth_path_bspline(
        self,
        path_xy: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """B-spline으로 경로 스무딩. 출력 점이 장애물 안이면 원본 반환."""
        if not _SCIPY_OK or len(path_xy) < 4:
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
                min_r = int(getattr(self, 'astar_min_inflation_cells', 2))
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
                # start/goal 셀이 inflation으로 occupied여도 A* 강제 시도.
                # _astar_with_inflation이 start/goal 셀을 force-clear하므로
                # 직선 fallback 대신 우회 경로를 찾을 수 있음.
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
                self._plan_robot_xy = (px, py)
                self._plan_s_ij = s_ij
                self._plan_g_ij = g_ij
                try:
                    # ── TopoTraj 파이프라인 ──
                    # use_topo_diversity=True: 병렬 다중 경로, False: 단일 경로
                    # 두 경우 모두 _astar_with_inflation(radius=0)으로 start/goal force-clear
                    if self.use_topo_diversity:
                        candidates_ij = self._astar_diverse(s_ij, g_ij)
                    else:
                        # min_r부터 0까지 점진적 fallback: 좁은 통로에서도 경로 보장
                        min_r = int(getattr(self, 'astar_min_inflation_cells', 2))
                        raw = None
                        for r in range(min_r, -1, -1):
                            raw = self._astar_with_inflation(s_ij, g_ij, radius_cells=r)
                            if raw:
                                if r < min_r:
                                    self.get_logger().warn(
                                        f"A* single: min_inflation reduced {min_r} → {r} (narrow passage)"
                                    )
                                break
                        candidates_ij = [raw] if raw else []

                    if candidates_ij:
                        # XY 변환
                        candidates_xy: List[List[Tuple[float, float]]] = [
                            [self._ij_to_xy(ci, cj) for ci, cj in p]
                            for p in candidates_ij
                        ]
                        # homotopy class 필터
                        if self.use_topo_diversity and len(candidates_xy) > 1:
                            candidates_xy = self._filter_homotopy_classes(candidates_xy)
                        # 최단 경로 선택
                        def _plen(p: List[Tuple[float, float]]) -> float:
                            return sum(
                                math.hypot(p[k + 1][0] - p[k][0], p[k + 1][1] - p[k][1])
                                for k in range(len(p) - 1)
                            )
                        path_xy = min(candidates_xy, key=_plen)
                        # LOS shortcutting
                        path_xy = self._shortcut_path(path_xy)
                        # B-spline smoothing
                        skip_n = int(getattr(self, 'smooth_bspline_skip_if_few_wps', 4))
                        if self.smooth_bspline_enabled and len(path_xy) >= skip_n:
                            path_xy = self._smooth_path_bspline(path_xy)
                        # goal 좌표 고정 및 저장 (#9: _path_lock으로 tick 스레드와 race 방지)
                        path_xy[-1] = (gx, gy)
                        with self._path_lock:
                            self._path_xy = path_xy
                            self._path_wp_idx = 0
                        self.get_logger().info(
                            f"A* succeeded: {len(path_xy)} wps "
                            f"(diversity={self.use_topo_diversity}, "
                            f"smoothed={self.smooth_bspline_enabled}), "
                            f"start_ij={s_ij}, goal_ij={g_ij}"
                        )
                        self._publish_path_msg()
                        return
                    if self.allow_diag:
                        self.get_logger().warn(
                            f"Path fallback (straight): A* failed to find path (start={s_ij}, goal={g_ij})"
                        )
                finally:
                    self._plan_robot_xy = None
                    self._plan_s_ij = None
                    self._plan_g_ij = None
        else:
            if self.allow_diag:
                self.get_logger().warn(
                    f"Path fallback (straight): no grid (_have_grid=False)"
                )
        # path fallback: 직선 경로(goal)로 대체 (#9: _path_lock으로 tick 스레드와 race 방지)
        self.get_logger().warn("Path fallback (straight): using direct goal line.")
        with self._path_lock:
            self._path_xy = [(gx, gy)]
            self._path_wp_idx = 0
        self._publish_path_msg()

    def _tick_replan(self) -> None:
        """주기적 경로 재계획 (replan_period 간격 타이머).

        replan_period=1.0s가 안전 최솟값 (A* 최악 ~50ms → 콜백 중첩 방지).
        _plan_path를 tick 스레드에서 직접 호출하면 _cb_group_replan 우회 →
        _path_xy 동시 쓰기(경로 thrashing) 위험으로 금지.
        """
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

    def _path_lookahead_wp(self) -> Optional[Tuple[float, float]]:
        """현재 waypoint보다 N칸 앞을 target으로 사용 (emergency brake 방향 결정 전용)."""
        with self._path_lock:
            if not self._path_xy:
                return None
            steps = getattr(self, "path_lookahead_steps", 12)
            idx = min(self._path_wp_idx + steps, len(self._path_xy) - 1)
            return self._path_xy[idx]

    def _path_lookahead_wp_at_dist(
        self, px: float, py: float, lookahead_dist: float
    ) -> Optional[Tuple[float, float]]:
        """드론 현재 위치로부터 경로상 lookahead_dist(m) 앞 지점을 interpolation하여 반환.

        step-count 기반 lookahead는 A* 경로가 3 wps로 단축됐을 때
        즉시 마지막 wp(goal)로 점프해 routing point를 무시하는 문제가 있음.
        distance 기반은 경로 wps 수와 무관하게 항상 일정 거리 앞 지점을 목표로 함.
        """
        with self._path_lock:
            if not self._path_xy:
                return None
            path = self._path_xy
            n = len(path)
            start_idx = max(0, min(self._path_wp_idx, n - 1))

        # Backward 시작 세그먼트 차단: drone이 path[start_idx]를 path 진행방향으로
        # 이미 통과했으면 start_idx를 advance. (path[start_idx]−drone)와
        # (path[start_idx+1]−path[start_idx])의 dot product가 음수(사이각 > 90°)면 통과한 것.
        # military_airfield 2-wp 단순 경로에서 lookahead가 후방 점을 반환해
        # 180° 회전을 유발하는 버그 차단.
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

        # 경로 잔여가 lookahead_dist보다 짧을 때: 항상 현재 start_idx보다 한 발 앞을 반환.
        # path[-2] 방식은 wp_idx >= n-2 시 현재 위치를 가리켜 드론 정지 데드락 유발.
        return path[min(n - 1, start_idx + 1)]

    def _path_advance_wp(self, px: float, py: float):
        """가장 가까운 경로 세그먼트를 수직 투영하여 wp_idx 결정.

        greedy(d_next<=d_cur): 측방 드리프트 시 코너를 건너뜀.
        threshold(0.5m): lookahead 추종 중 비스듬히 지나치면 idx 동결 (기존 포기 이유).
        세그먼트 투영: 양쪽 문제 없음. 측방 이탈과 무관하게 경로 위 투영점(t)으로 판단.
        """
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

        # [DEBUG] 우선진단 1: Lookahead WP 추적 (1초마다 출력)
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
        tol = self.heading_align_tol
        if abs(yaw_err) > tol:
            # #8: 완전 정지(0,0) 제거 → 오차 비례 감속
            # 90° 이상에서만 실질 정지, 그 이하에서는 감속하며 전진
            planner_yaw_limit = float(getattr(self, 'planner_max_yaw_rate', self.max_yaw_rate))
            yaw_rate = max(-planner_yaw_limit, min(planner_yaw_limit, 2.0 * yaw_err))
            align_scale = max(0.0, 1.0 - abs(yaw_err) / (math.pi / 2))
            v_cmd = min(self.v_max, 0.8 * gnorm) * align_scale
            vx = v_cmd * math.cos(target_yaw)
            vy = v_cmd * math.sin(target_yaw)
            return vx, vy, self._vz_correction(), yaw_rate
        v_cmd = min(self.v_max, 0.8 * gnorm)
        # target_yaw 방향으로 velocity (코너에서 자연스러운 꺾임)
        vx = v_cmd * math.cos(target_yaw)
        vy = v_cmd * math.sin(target_yaw)
        return vx, vy, self._vz_correction(), 0.0

    def _tick(self):
        now = time.time()
        if self.mav_state is None or not self.mav_state.connected:
            return
        # homing 대기 중: course_goal_publisher가 position setpoint를 전송 중이므로
        # velocity setpoint를 동시에 전송하면 PX4가 두 명령을 번갈아 실행 → 드론 제자리 → homing timeout
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

        # Emergency brake (공통): arm mask 적용 후 실제 장애물만 감지 (#2), 후퇴 추가 (#3)
        # use_override=True: lidar_range_min_override=0.35m 적용 → body 자기감지(0.24~0.31m) 필터
        d_front_min = self._dist_in_sector(-half, half, "min", use_override=True)
        if d_front_min <= self.emergency_brake_dist:
            d_left  = self._dist_in_sector(self.emergency_brake_deg, 90.0, "min", use_override=False)
            d_right = self._dist_in_sector(-90.0, -self.emergency_brake_deg, "min", use_override=False)
            self._brake_count += 1

            # 회전 방향 결정: 진입 시 1회만 결정하고 brake 해제 전까지 고정 (방향 진동 방지)
            if self._brake_rotate_left is None:
                rotate_reason = "lidar"
                rotate_left = (d_left >= d_right)
                wp_la = self._path_lookahead_wp()
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

        # ppo 모드의 goal_bias_yaw_scale 게이트용 d_min_sectors 계산
        agg = str(getattr(self, "lidar_sector_agg", "min")).lower()
        if agg not in ("min", "median", "mean"):
            agg = "min"
        d_front       = self._dist_in_sector(-half, half, agg)
        d_front_left  = self._dist_in_sector(half, 90.0, agg)
        d_front_right = self._dist_in_sector(-90.0, -half, agg)
        d_min_sectors = min(d_front, d_front_left, d_front_right)

        # nav_mode별 제어 로직
        if self.nav_mode == "astar":
            vx, vy, vz, yaw_rate = self._tick_astar_mode(px, py, yaw, now)
        elif self.nav_mode == "ppo":
            vx, vy, vz, yaw_rate = self._tick_ppo_mode(yaw, d_min_sectors)
        else:  # "hybrid"
            vx, vy, vz, yaw_rate = self._tick_hybrid_mode(px, py, yaw)

        self._publish(vx, vy, vz, yaw_rate)

    def _tick_astar_mode(self, px: float, py: float, yaw: float, now: float) -> Tuple[float, float, float, float]:
        """A* 단독 모드: global planner(_tick_global)만 사용. PPO 없음."""
        if not self._path_xy:
            self._plan_path()
        return self._tick_global(px, py, yaw, now)

    def _tick_ppo_mode(self, yaw: float, d_min_sectors: float) -> Tuple[float, float, float, float]:
        """PPO 단독 모드: 최종 goal 방향으로 PPO 추론. A* 없음.

        Frame skip (ppo 모드 한정, astar/hybrid 무영향):
        - PPO 추론·action 갱신은 ppo_frame_skip tick마다 (학습 dt=0.3s 매칭, default 6 @ 20Hz).
        - 그 외 tick은 cache된 action 그대로 재사용 → PX4 setpoint는 20Hz 유지(lockstep).
        - 안전 게이트(_escape_if_spinning, goal_bias)는 cache 우회하여 매 tick 평가.
        """
        # frame skip: 학습 dt(0.3s) 매칭 (배포 0.05s × 6 = 0.3s)
        self._ppo_skip_counter += 1
        if self._cached_ppo_action is None or self._ppo_skip_counter >= self.ppo_frame_skip:
            obs = self._build_obs()
            if obs is None:
                return 0.0, 0.0, self._vz_correction(), 0.0
            self._cached_ppo_action = self._infer(obs)
            self._ppo_skip_counter = 0

        v_cmd, yaw_rate = self._action_to_vel(self._cached_ppo_action)

        # 안전 게이트는 매 tick 평가 (cache 우회)
        esc = self._escape_if_spinning(yaw_rate, yaw)
        if esc is not None:
            self._cached_ppo_action = None  # cache invalidate → 다음 tick 즉시 재추론
            return esc

        if d_min_sectors > self.goal_bias_dist:
            yaw_rate *= self.goal_bias_yaw_scale

        vx, vy = self._compute_cmd_vel(v_cmd, yaw)
        return vx, vy, self._vz_correction(), yaw_rate

    def _tick_hybrid_mode(self, px: float, py: float, yaw: float) -> Tuple[float, float, float, float]:
        """Hybrid-v3 (방식 C): A*(전역 경로) lookahead obs + PPO(v_cmd·yaw_rate) 제어.

        학습 env(C2, dt=0.05)와 정합:
        - obs goal_rel = A* 경로상 hybrid_lookahead_dist_m(1.0m) 앞 wp / norm 3.0 (_build_obs 그대로)
        - 제어 = PPO 출력 full: yaw_rate = action[1]*max_yaw_rate, 이동 = 현재 heading 방향(학습 unicycle)
        - 매 tick(20Hz) 추론 = 학습 dt=0.05 와 결정 주기 정합 (frame_skip 없음)
        기하학적 yaw·align_scale 미사용(학습에 없음 → 폐기).
        """
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
        v_cmd, yaw_rate = self._action_to_vel(action)   # 방식 C: PPO yaw_rate 사용

        # 이동 = 현재 heading 방향 (학습 unicycle 정합), heading 은 PPO yaw_rate 로 제어
        vx = v_cmd * math.cos(yaw)
        vy = v_cmd * math.sin(yaw)
        return vx, vy, self._vz_correction(), yaw_rate


def main(args=None):
    rclpy.init(args=args)
    node = RLOffboardNode()
    # #9: MultiThreadedExecutor — tick(20Hz) / replan(A*) / sensor 콜백을 별도 스레드로 분리
    # num_threads=3: tick 전용, replan 전용, sensor 공용 (스레드 수 = callback group 수)
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
