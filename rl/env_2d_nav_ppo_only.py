#!/usr/bin/env python3
"""
env_2d_nav.py 의 PPO-only ablation 변형 (A*+PPO 하이브리드 vs 순수 PPO 학습 차이 비교용 baseline).
parent 와 dynamics/C2 보상/커리큘럼/평가 시나리오/lidar/BFS 거리맵 모두 동일.
차이: A* 경로 계산·replan·waypoint·lookahead obs 제거 → goal_rel = 항상 최종 goal body-frame
(parent 의 use_astar_lookahead=False 분기와 동일). BFS 거리맵은 보상 shaping 용(A* 아님)으로 유지.

보상 (C2 정식 모델): collision -collision_penalty / progress w_progress*(prev_bfs-curr_bfs) / success +success_reward.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


@dataclass
class RectObstacle:
    """Axis-aligned rectangle obstacle in 2D."""
    x: float  # center x
    y: float  # center y
    w: float  # width (x extent)
    h: float  # height (y extent)

    @property
    def xmin(self) -> float:
        return self.x - self.w / 2.0

    @property
    def xmax(self) -> float:
        return self.x + self.w / 2.0

    @property
    def ymin(self) -> float:
        return self.y - self.h / 2.0

    @property
    def ymax(self) -> float:
        return self.y + self.h / 2.0


# Gazebo `obstacle_course_light.world` [-X] S 코스 박스: (cx, cy, w, h) = ENU 중심·SDF <box><size>.
# 학습 env [0, world_size]^2 로 COURSE_L_S_OFFSET_XY 평행이동 (world_size=20 정합).
COURSE_L_S_ENU_BOXES: List[Tuple[float, float, float, float]] = [
    (-10.0, 1.0, 0.3, 18.0),  # s_outer_left
    (-6.0, 10.0, 8.0, 0.3),  # s_outer_top
    (-6.0, -10.0, 8.0, 0.3),  # s_outer_bottom
    (-4.0, 0.0, 0.3, 8.0),  # inner_s_v1
    (-8.0, 0.0, 0.3, 8.0),  # inner_s_v2
    (-6.0, 6.0, 0.3, 8.0),  # inner_s_v3
    (-6.0, -6.0, 0.3, 8.0),  # inner_s_v4
]
COURSE_L_S_OFFSET_XY: Tuple[float, float] = (12.0, 10.0)


class UAVNav2DEnv(gym.Env):
    """
    2D UAV 내비게이션 환경 (연속 위치·연속 액션). PPO-only baseline: goal_rel = 최종 goal body-frame.

    obs: lidar_ranges(obs_beams, downsample→[-1,1]) + goal_rel(body-frame dx,dy, world_size 정규화).
    action: (a_v, a_yaw)∈[-1,1] (C2: PPO 가 v·yaw 모두 제어).
      v_cmd=v_min+(a_v+1)/2*(v_max-v_min), yaw_rate=a_yaw*yaw_rate_max.
    보상(C2): BFS progress + success bonus, terminal collision penalty.
    종료: success(goal_radius 내) / collision / timeout(max_steps).
    """

    _LIDAR_RAY_STEP: float = 0.2   # ray marching 스텝 크기 (m)

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        world_size: float = 20.0,
        num_obstacles: int = 2,
        obstacle_size_range: Tuple[float, float] = (1.5, 3.0),
        num_beams: int = 360,          # 내부 레이 계산용 (ROS scan 해상도에 대응)
        obs_beams: int = 64,           # 관측용 downsample 빔 수
        downsample_mode: str = "min",  # "min" or "avg"
        fov: float = 2 * math.pi,      # 360 deg
        max_range: float = 10.0,       # r_FOV (논문에서는 rFOV)
        v_max: float = 0.8,            # v_cmd 최대 속도 (m/s)
        v_min: float = 0.3,            # v_cmd 최소 속도 (멈춤 방지, replan 주기 내 wp 전진 보장)
        yaw_rate_max: float = 1.0,     # yaw_rate 최대 (rad/s)
        dt: float = 0.05,              # 시뮬 스텝 시간 (s). 배포 setpoint 20Hz 정합
        goal_radius: float = 2.0,      # 목표 판정 반경 (커리큘럼 시작값, 초반 학습 용이)
        max_steps: int = 1800,         # 에피소드 상한. dt=0.05 기준 시간 horizon ~90s
        use_curriculum: bool = True,   # 커리큘럼 학습 활성화
        curriculum_levels: Optional[List[Tuple[float, float, int]]] = None,
        collision_penalty: float = 500.0,  # 충돌 페널티 (stress test collision 감소 목적)
        success_reward: float = 800.0,  # 목표 도달 보상 (sparse anchor)
        w_progress: float = 50.0,        # goal로 1m 가까워지면 +50 (BFS 거리 기반)
        yaw_drift_std: float = 0.0,       # yaw 랜덤 드리프트 표준편차
        min_clearance: float = 1.0,    # 시작/목표 위치 최소 여유거리(m)
        eval_scenario: Optional[str] = None,  # "obstacle_density_*", "narrow_passage", "course_l_s"
        eval_sensor_noise: bool = False,     # 평가 전용: yaw_drift_std=0.05
        curriculum_verbose: bool = True,     # False면 curriculum 진행 로그 생략 (eval/병렬 env 중복 출력 방지)
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.world_size = float(world_size)
        self.num_obstacles = int(num_obstacles)
        self.obstacle_size_range = obstacle_size_range

        self.num_beams = int(num_beams)
        self.obs_beams = int(obs_beams)
        self.downsample_mode = str(downsample_mode).lower()
        if self.downsample_mode not in ("min", "avg"):
            raise ValueError(f"downsample_mode must be 'min' or 'avg', got {downsample_mode!r}")
        self.fov = float(fov)
        self.max_range = float(max_range)

        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.yaw_rate_max = float(yaw_rate_max)
        self.dt = float(dt)
        self.goal_radius_base = float(goal_radius)  # 기본 goal_radius (커리큘럼 시작점)
        self.goal_radius = float(goal_radius)       # 현재 goal_radius (동적 조정)
        self.max_steps = int(max_steps)

        # Curriculum learning
        self.use_curriculum = bool(use_curriculum)
        if curriculum_levels is None:
            # (goal_radius, min_clearance, num_obstacles); num_obstacles tuple(lo,hi)→매 에피소드 lo~hi 랜덤
            curriculum_levels = [
                (2.0, float(min_clearance), (0, 3)),    # 단계 0: 0~3개 랜덤 (기초 내비게이션)
                (2.0, float(min_clearance), (4, 6)),    # 단계 1: 4~6개 랜덤
                (1.5, float(min_clearance), 8),         # 단계 2: 8개 고정
                (1.0, float(min_clearance), (8, 12)),   # 단계 3: 8~12개 랜덤
                (0.5, float(min_clearance), (12, 20)),  # 단계 4: 12~20개 랜덤
            ]
        self.curriculum_levels = [
            (float(r), float(c), n) for r, c, n in curriculum_levels
        ]
        if not self.curriculum_levels:
            self.curriculum_levels = [(self.goal_radius_base, float(min_clearance), int(num_obstacles))]
        self.curriculum_level_idx = 0
        self.curriculum_level_thresholds = [0.60, 0.70, 0.80, 0.90]

        self.collision_penalty = float(collision_penalty)
        self.success_reward = float(success_reward)
        self.w_progress = float(w_progress)
        self.yaw_drift_std = float(yaw_drift_std)
        self.min_clearance = float(min_clearance)

        # 평가 전용 (학습 환경과 분리)
        self.eval_scenario = eval_scenario
        self.eval_sensor_noise = bool(eval_sensor_noise)
        self.curriculum_verbose = bool(curriculum_verbose)
        if self.eval_sensor_noise:
            self.yaw_drift_std = 0.05
        if self.eval_scenario or self.eval_sensor_noise:
            self.use_curriculum = False  # 평가 시 커리큘럼 비활성

        self.rng = np.random.default_rng(seed)

        # Observation space: lidar (N,) + goal_rel body (2,) [heading 제거]
        obs_dim = self.obs_beams + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Continuous actions: (a_v, a_yaw) ∈ [-1, 1] (정규화)
        # env에서 v_cmd = v_min + (a_v+1)/2*(v_max-v_min), yaw_rate_cmd = a_yaw * yaw_rate_max
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # State variables
        self.pos = np.zeros(2, dtype=np.float32)
        self.yaw = 0.0  # heading angle (rad) in 2D
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles: List[RectObstacle] = []
        self.step_count = 0
        self.prev_goal_dist = 0.0
        self._bfs_dist_map: Optional[np.ndarray] = None
        self._bfs_cell_size: float = 0.25       # BFS 거리 맵 해상도 (m/cell). course_l_s 0.3m 벽 포함 보장 위해 ≤0.25 필요
        self.episode_reward = 0.0

        # For logging
        self.last_collision = False
        self.last_success = False

        # Curriculum tracking
        self._curriculum_episode_count = 0
        self._curriculum_success_count = 0
        self._curriculum_window = 100  # 최근 100 에피소드 기준
        # 단계별 시나리오 혼합 비율 (단계↑일수록 stress 비중↑). 누적 경계 확률:
        # normal → density_15 → density_20 → narrow → course_l_s
        self._stage_mix_thresholds = [
            (1.00, 1.00, 1.00, 1.00, 1.00),  # stage 0: normal 100%
            (0.90, 0.90, 0.90, 0.95, 1.00),  # stage 1: normal 90%, narrow 5%, course_l_s 5%
            (0.70, 0.80, 0.80, 0.90, 1.00),  # stage 2: normal 70%, d15 10%, narrow 10%, course_l_s 10%
            (0.45, 0.60, 0.70, 0.85, 1.00),  # stage 3: normal 45%, d15 15%, d20 10%, narrow 15%, course_l_s 15%
            (0.05, 0.20, 0.55, 0.70, 1.00),  # stage 4: normal 5%, d15 15%, d20 35%, narrow 15%, course_l_s 30%
        ]

    # --- Reset & Step ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.last_collision = False
        self.last_success = False
        self._bfs_dist_map = None
        self.episode_reward = 0.0

        # 커리큘럼: 성공률 threshold 만족 시 최고 단계로 한 번에 점프
        if self.use_curriculum and self._curriculum_episode_count >= self._curriculum_window:
            success_rate = self._curriculum_success_count / max(self._curriculum_episode_count, 1)
            old_idx = self.curriculum_level_idx
            target_idx = old_idx
            for next_level in range(old_idx + 1, len(self.curriculum_levels)):
                thr_idx = next_level - 1
                if thr_idx < len(self.curriculum_level_thresholds) and success_rate >= self.curriculum_level_thresholds[thr_idx]:
                    target_idx = next_level
                else:
                    break
            if target_idx > old_idx:
                self.curriculum_level_idx = target_idx
                self._apply_curriculum_level()
                required_rate = self.curriculum_level_thresholds[target_idx - 1] if target_idx > 0 else 0.0
                if self.curriculum_verbose:
                    print(
                        f"[Curriculum] Success {success_rate*100:.1f}% (>= {required_rate*100:.0f}%) → "
                        f"goal_radius={self.goal_radius:.2f}, min_clearance={self.min_clearance:.2f}"
                    )
            if target_idx > old_idx:
                self._curriculum_episode_count = 0
                self._curriculum_success_count = 0

        if self.use_curriculum:
            self._apply_curriculum_level()

        # 평가 시나리오 또는 단계별 혼합 비율 적용
        if self.eval_scenario == "obstacle_density_15":
            self._reset_eval_obstacle_density(num_obstacles=15)
        elif self.eval_scenario == "obstacle_density_20":
            self._reset_eval_obstacle_density(num_obstacles=20)
        elif self.eval_scenario == "narrow_passage":
            self._reset_eval_narrow_passage()
        elif self.eval_scenario == "course_l_s":
            self._reset_course_l_s()
        elif self.use_curriculum:
            idx = min(self.curriculum_level_idx, len(self._stage_mix_thresholds) - 1)
            t = self._stage_mix_thresholds[idx]
            r = float(self.rng.random())
            if r < t[0]:      # normal
                self._reset_normal_scenario()
            elif r < t[1]:    # density_15
                self._reset_eval_obstacle_density(num_obstacles=15)
            elif r < t[2]:    # density_20
                self._reset_eval_obstacle_density(num_obstacles=20)
            elif r < t[3]:    # narrow_passage
                self._reset_eval_narrow_passage()
            else:             # course_l_s
                self._reset_course_l_s()
        else:
            self._reset_normal_scenario()

        # BFS 거리 맵: 에피소드당 1회 계산 (goal·장애물 확정 후)
        self._bfs_dist_map = self._compute_bfs_dist_map(self._bfs_cell_size)

        self.yaw = self.rng.uniform(-math.pi, math.pi)

        obs = self._get_obs()
        info = self._get_info()
        info["episode_reward"] = float(self.episode_reward)
        return obs, info

    def _apply_curriculum_level(self) -> None:
        if not self.curriculum_levels:
            return
        radius, clearance, num_obstacles = self.curriculum_levels[self.curriculum_level_idx]
        self.goal_radius = float(radius)
        self.min_clearance = float(clearance)
        # tuple (lo, hi): 매 에피소드마다 lo~hi개 랜덤
        if isinstance(num_obstacles, tuple):
            lo, hi = num_obstacles
            self.num_obstacles = int(self.rng.integers(lo, hi + 1))
        else:
            self.num_obstacles = int(num_obstacles)

    def _reset_eval_obstacle_density(self, num_obstacles: int) -> None:
        """평가 전용: Obstacle Density Stress Test. goal behind obstacle."""
        self.obstacles = []
        for _ in range(num_obstacles):
            w = float(self.rng.uniform(*self.obstacle_size_range))
            h = float(self.rng.uniform(*self.obstacle_size_range))
            x = float(self.rng.uniform(w / 2, self.world_size - w / 2))
            y = float(self.rng.uniform(h / 2, self.world_size - h / 2))
            self.obstacles.append(RectObstacle(x=x, y=y, w=w, h=h))

        self.pos = self._sample_free_position(min_clearance=self.min_clearance)

        # num_obstacles=0: "obstacle 뒤 goal" 로직 불필요
        if num_obstacles == 0:
            self.goal = self._sample_free_position(
                min_dist=10.0, ref=self.pos, min_clearance=self.min_clearance
            )
            self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))
            return

        # goal: obstacle 바로 뒤에 (obstacle이 start와 goal 사이에 위치)
        for _ in range(500):
            idx = self.rng.integers(0, len(self.obstacles))
            obs = self.obstacles[idx]
            dir_vec = np.array([obs.x, obs.y]) - self.pos
            dist = float(np.linalg.norm(dir_vec))
            if dist < 1e-6:
                continue
            dir_vec = dir_vec / dist
            # goal = obstacle 반대편 (start->obstacle->goal 방향)
            goal_offs = 2.0 + self.rng.uniform(0.5, 2.0)
            goal_cand = np.array([obs.x, obs.y], dtype=np.float32) + dir_vec * goal_offs
            goal_cand = np.clip(goal_cand, 1.0, self.world_size - 1.0)
            if not self._check_collision(goal_cand) and float(np.linalg.norm(goal_cand - self.pos)) >= 5.0:
                if self._is_reachable_bfs(self.pos, goal_cand, cell_size=0.3):
                    self.goal = goal_cand
                    break
        else:
            self.goal = self._sample_free_position(min_dist=10.0, ref=self.pos, min_clearance=self.min_clearance)
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def _reset_eval_narrow_passage(self) -> None:
        """평가 전용: Narrow Passage. 'L 통로'(세로+가로 다리)만 자유공간으로 두고 나머지를 벽으로 채워
        닫힌 관 구성 → 90도 꺾어 주행해야 도달. start/goal=통로 양 끝, flip 으로 4방향 랜덤화."""
        W = self.world_size
        for _ in range(200):
            cw = float(self.rng.uniform(1.8, 2.4))     # 통로 폭(m)
            vx0 = float(self.rng.uniform(2.0, 3.5))    # 세로 다리 좌측 x (코너를 구석으로)
            hy0 = float(self.rng.uniform(2.0, 3.5))    # 가로 다리 하단 y (코너를 구석으로)
            vy1 = float(self.rng.uniform(16.5, 18.0))  # 세로 다리 상단 y (끝점을 가장자리로)
            hx1 = float(self.rng.uniform(16.5, 18.0))  # 가로 다리 우측 x (끝점을 가장자리로)
            vx1, hy1 = vx0 + cw, hy0 + cw
            flip_x = bool(self.rng.integers(0, 2))
            flip_y = bool(self.rng.integers(0, 2))

            def fx(x: float) -> float:
                return W - x if flip_x else x

            def fy(y: float) -> float:
                return W - y if flip_y else y

            def box(xmin: float, xmax: float, ymin: float, ymax: float) -> RectObstacle:
                xs = sorted([fx(xmin), fx(xmax)])
                ys = sorted([fy(ymin), fy(ymax)])
                return RectObstacle(
                    x=(xs[0] + xs[1]) / 2.0, y=(ys[0] + ys[1]) / 2.0,
                    w=xs[1] - xs[0], h=ys[1] - ys[0],
                )

            # L 통로를 제외한 영역을 5개 벽으로 감싸 닫힌 관 구성
            self.obstacles = [
                box(0.0, W, 0.0, hy0),      # 아래
                box(0.0, W, vy1, W),        # 위
                box(0.0, vx0, hy0, vy1),    # 왼쪽
                box(vx1, W, hy1, vy1),      # 안쪽(세로 다리 우측 · 가로 다리 위)
                box(hx1, W, hy0, hy1),      # 가로 다리 우측 막음
            ]

            # 통로 양 끝점: 세로 다리 상단 / 가로 다리 우측
            end_v = np.array([fx((vx0 + vx1) / 2.0), fy(vy1 - 0.9)], dtype=np.float32)
            end_h = np.array([fx(hx1 - 0.9), fy((hy0 + hy1) / 2.0)], dtype=np.float32)
            if self.rng.integers(0, 2):
                self.pos, self.goal = end_v.copy(), end_h.copy()
            else:
                self.pos, self.goal = end_h.copy(), end_v.copy()

            if self._check_collision(self.pos) or self._check_collision(self.goal):
                continue
            if self._is_reachable_bfs(self.pos, self.goal, cell_size=0.2):
                break
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def _reset_course_l_s(self) -> None:
        """평가 전용: Course L(S 코스) 박스를 env 좌표로 고정 배치. BFS 도달 가능한 start/goal 쌍만 채택."""
        ox, oy = COURSE_L_S_OFFSET_XY
        self.obstacles = []
        for cx, cy, w, h in COURSE_L_S_ENU_BOXES:
            self.obstacles.append(RectObstacle(x=cx + ox, y=cy + oy, w=w, h=h))

        # 3단계 커리큘럼 spawn (goal과 같은 corridor 근처일수록 완주 쉬움):
        # easy(50%): BFS ~3-8m, 완주 가능 / medium(30%): min_goal_dist=5.0 으로 v4 동쪽 8~15m 지배
        # hard(20%): 원거리 BFS ~50m 전체 S-커브 탐색. hard 비율 낮게 유지 —
        #   완주 불가 구조라 비중 크면 no_progress 로 수렴해 학습 신호 희석.
        r = self.rng.random()
        if r < 0.50:
            spawn_x = (4.2, 5.7)
            spawn_y = (0.5, 5.5)
            min_goal_dist = 1.5
        elif r < 0.80:
            spawn_x = (10.0, 13.0)
            spawn_y = (2.5, 7.5)
            min_goal_dist = 5.0
        else:
            spawn_x = (12.0, 16.0)
            spawn_y = (14.0, 20.0)
            min_goal_dist = 10.0
        goal_x = (2.6, 4.0)
        goal_y = (1.0, 4.0)

        for _ in range(3000):
            self.pos = np.array(
                [
                    float(self.rng.uniform(*spawn_x)),
                    float(self.rng.uniform(*spawn_y)),
                ],
                dtype=np.float32,
            )
            self.goal = np.array(
                [
                    float(self.rng.uniform(*goal_x)),
                    float(self.rng.uniform(*goal_y)),
                ],
                dtype=np.float32,
            )
            if self._check_collision(self.pos) or self._check_collision(self.goal):
                continue
            if float(np.linalg.norm(self.goal - self.pos)) < min_goal_dist:
                continue
            if self.min_clearance > 0.0:
                if self._min_lidar_at(self.pos) < self.min_clearance or \
                        self._min_lidar_at(self.goal) < self.min_clearance:
                    continue
            if self._is_reachable_bfs(self.pos, self.goal, cell_size=0.2):
                self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))
                return

        # fallback (대표 통로점; 수동으로 장애물 밖·연결 확인)
        self.pos = np.array([13.0, 11.0], dtype=np.float32)
        self.goal = np.array([4.8, 3.0], dtype=np.float32)
        if self._check_collision(self.pos):
            self.pos = np.array([12.5, 10.0], dtype=np.float32)
        if self._check_collision(self.goal):
            self.goal = np.array([5.2, 2.8], dtype=np.float32)
        if not self._is_reachable_bfs(self.pos, self.goal, cell_size=0.25):
            self.goal = np.array([4.0, 4.5], dtype=np.float32)
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Action must have 2 elements (a_v, a_yaw) in [-1,1], got shape {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # yaw_drift_std > 0이면 소량 yaw 노이즈 (eval_sensor_noise 시 0.05)
        if self.yaw_drift_std > 0.0:
            self.yaw = self._wrap_angle(self.yaw + self.rng.normal(0.0, self.yaw_drift_std))

        # Scale normalized action [-1,1] -> (v_cmd, yaw_rate_cmd) — C2: PPO 가 v·yaw 모두 제어
        a_v, a_yaw = float(action[0]), float(action[1])
        v_cmd = self.v_min + (a_v + 1.0) / 2.0 * (self.v_max - self.v_min)
        yaw_rate_cmd = a_yaw * self.yaw_rate_max
        self.yaw = self._wrap_angle(self.yaw + yaw_rate_cmd * self.dt)
        delta = (v_cmd * self.dt) * np.array(
            [math.cos(self.yaw), math.sin(self.yaw)], dtype=np.float32
        )

        prev_bfs_dist = self._get_bfs_dist(self.pos)
        raw_new_pos = self.pos + delta
        # 경로 상 충돌 검사 (선분을 샘플링해 "텔레포트" 충돌 방지)
        collision, hit_pos, collision_kind = self._check_path_collision(self.pos, raw_new_pos, check_bounds=True)
        self.last_collision = collision

        # Update position
        if not collision:
            self.pos = np.clip(raw_new_pos, 0.0, self.world_size)
        elif hit_pos is not None:
            self.pos = hit_pos

        # Success check
        dist_to_goal = float(np.linalg.norm(self.goal - self.pos))
        curr_bfs_dist = self._get_bfs_dist(self.pos)
        success = dist_to_goal <= self.goal_radius
        self.last_success = success

        terminated = bool(collision or success)
        truncated = bool(self.step_count >= self.max_steps)

        reward, lidar = self._compute_reward(
            prev_bfs_dist=prev_bfs_dist,
            curr_bfs_dist=curr_bfs_dist,
            collision=collision,
            success=success,
        )

        obs = self._get_obs(lidar=lidar if not collision else None)
        info = self._get_info()
        info["collision_kind"] = collision_kind if collision else "none"
        if terminated:
            info["done_reason"] = "collision" if collision else "reached"
        elif truncated:
            info["done_reason"] = "timeout"

        # Update curriculum statistics when episode ends
        if terminated or truncated:
            self._curriculum_episode_count += 1
            if success:
                self._curriculum_success_count += 1

        return obs, reward, terminated, truncated, info

    # --- BFS 유틸리티 ---
    def _build_obstacle_grid(self, cell_size: float) -> Tuple[np.ndarray, int]:
        """numpy 벡터화로 장애물 그리드 생성. 반환: (grid[n,n], n)"""
        n = int(math.ceil(self.world_size / cell_size))
        grid = np.zeros((n, n), dtype=np.uint8)
        if not self.obstacles:
            return grid, n
        xs = (np.arange(n, dtype=np.float32) + 0.5) * cell_size
        ys = (np.arange(n, dtype=np.float32) + 0.5) * cell_size
        XX, YY = np.meshgrid(xs, ys)
        for obs in self.obstacles:
            grid |= ((XX >= obs.xmin) & (XX <= obs.xmax) &
                     (YY >= obs.ymin) & (YY <= obs.ymax))
        return grid, n

    def _compute_bfs_dist_map(self, cell_size: float = 0.5) -> np.ndarray:
        """goal에서 역방향 BFS로 전체 그리드의 최단 경로 거리(m) 계산.
        장애물·도달 불가 셀은 np.inf. 에피소드 시작 시 1회 호출."""
        grid, n = self._build_obstacle_grid(cell_size)
        dist = np.full((n, n), np.inf, dtype=np.float32)
        gj = max(0, min(n - 1, int(self.goal[0] / cell_size)))
        gi = max(0, min(n - 1, int(self.goal[1] / cell_size)))
        if grid[gi, gj]:
            return dist
        dist[gi, gj] = 0.0
        q = deque([(gi, gj)])
        while q:
            i, j = q.popleft()
            d_next = dist[i, j] + cell_size
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not grid[ni, nj] and dist[ni, nj] == np.inf:
                    dist[ni, nj] = d_next
                    q.append((ni, nj))
        return dist

    def _get_bfs_dist(self, pos: np.ndarray) -> float:
        """현재 위치의 BFS 거리 조회. 도달 불가 셀이면 유클리드 거리로 fallback."""
        if self._bfs_dist_map is None:
            return float(np.linalg.norm(self.goal - pos))
        n = self._bfs_dist_map.shape[0]
        j = max(0, min(n - 1, int(pos[0] / self._bfs_cell_size)))
        i = max(0, min(n - 1, int(pos[1] / self._bfs_cell_size)))
        d = float(self._bfs_dist_map[i, j])
        return d if not math.isinf(d) else float(np.linalg.norm(self.goal - pos))

    def _is_reachable_bfs(self, start: np.ndarray, goal: np.ndarray, cell_size: float = 0.5) -> bool:
        """start에서 goal까지 장애물을 피해 도달 가능한지 DFS로 검사."""
        if cell_size <= 0.0:
            return True
        grid, n = self._build_obstacle_grid(cell_size)

        def to_cell(p: np.ndarray) -> Tuple[int, int]:
            return (max(0, min(n - 1, int(p[1] / cell_size))),
                    max(0, min(n - 1, int(p[0] / cell_size))))

        si, sj = to_cell(start)
        gi, gj = to_cell(goal)
        if grid[si, sj] or grid[gi, gj]:
            return False

        visited = np.zeros((n, n), dtype=np.uint8)
        stack = [(si, sj)]
        visited[si, sj] = 1
        while stack:
            i, j = stack.pop()
            if i == gi and j == gj:
                return True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not visited[ni, nj] and not grid[ni, nj]:
                    visited[ni, nj] = 1
                    stack.append((ni, nj))
        return False

    def _compute_reward(
        self,
        *,
        prev_bfs_dist: float,
        curr_bfs_dist: float,
        collision: bool,
        success: bool,
    ) -> Tuple[float, Optional[np.ndarray]]:
        # C2 정식 모델: collision penalty + BFS-progress shaping + success bonus
        if collision:
            return -self.collision_penalty, None

        lidar = self._lidar_ranges(self.pos, self.yaw)
        lidar = self._sanitize_lidar(lidar)

        # 1) BFS progress shaping: goal로 가까워진 BFS 거리(m) × w_progress
        reward = self.w_progress * (prev_bfs_dist - curr_bfs_dist)

        # 2) Success bonus
        if success:
            reward += self.success_reward

        return reward, lidar

    # --- Observation ---
    def _get_obs(self, *, lidar: Optional[np.ndarray] = None) -> np.ndarray:
        if lidar is None:
            lidar = self._lidar_ranges(self.pos, self.yaw)  # [0, max_range]
        lidar = self._downsample_lidar(lidar, self.obs_beams, self.downsample_mode)
        lidar_norm = (lidar / self.max_range).astype(np.float32)  # [0,1]
        # map to [-1,1] for NN friendliness
        lidar_norm = lidar_norm * 2.0 - 1.0

        # goal_rel: PPO-only baseline — A* lookahead 제거, final goal 을 target 으로 body-frame 정규화.
        target = self.goal.astype(np.float32)
        norm_scale = self.world_size
        dx_world = (target - self.pos).astype(np.float32)
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        dx_body = c * dx_world[0] - s * dx_world[1]
        dy_body = s * dx_world[0] + c * dx_world[1]
        goal_rel = np.array([dx_body, dy_body], dtype=np.float32) / max(norm_scale, 1e-6)
        goal_rel = np.clip(goal_rel, -1.0, 1.0).astype(np.float32)

        obs = np.concatenate([lidar_norm, goal_rel], axis=0).astype(np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "collision": self.last_collision,
            "success": self.last_success,
            "dist_to_goal": float(np.linalg.norm(self.goal - self.pos)),
        }

    # --- Obstacle generation & collision ---
    def _min_lidar_at(self, pos: np.ndarray) -> float:
        """pos에서 전방위 lidar의 최소 거리(m). clearance 체크용."""
        return float(np.min(self._lidar_ranges(pos, 0.0)))

    def _reset_normal_scenario(self) -> None:
        """커리큘럼 normal 에피소드: 장애물 랜덤 생성 + pos/goal 샘플링."""
        # stage 4: normal 50%를 0~5개 장애물로 → 저밀도 환경 학습 분포 포함, eval domain gap 해소
        if self.curriculum_level_idx == 4 and self.rng.random() < 0.50:
            saved = self.num_obstacles
            self.num_obstacles = int(self.rng.integers(0, 6))
            self._spawn_obstacles()
            self.num_obstacles = saved
        else:
            self._spawn_obstacles()
        self.pos = self._sample_free_position(min_clearance=self.min_clearance)
        self.goal = self._sample_free_position(min_dist=5.0, ref=self.pos, min_clearance=self.min_clearance)
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def _spawn_obstacles(self) -> None:
        self.obstacles = []
        for _ in range(self.num_obstacles):
            w = float(self.rng.uniform(*self.obstacle_size_range))
            h = float(self.rng.uniform(*self.obstacle_size_range))
            x = float(self.rng.uniform(w/2, self.world_size - w/2))
            y = float(self.rng.uniform(h/2, self.world_size - h/2))
            self.obstacles.append(RectObstacle(x=x, y=y, w=w, h=h))

    def _sample_free_position(
        self,
        min_dist: float = 0.0,
        ref: Optional[np.ndarray] = None,
        min_clearance: float = 0.0,
    ) -> np.ndarray:
        for _ in range(2000):
            p = np.array(
                [self.rng.uniform(0.5, self.world_size - 0.5),
                 self.rng.uniform(0.5, self.world_size - 0.5)],
                dtype=np.float32
            )
            if self._check_collision(p):
                continue
            if ref is not None and float(np.linalg.norm(p - ref)) < min_dist:
                continue
            if min_clearance > 0.0 and self._min_lidar_at(p) < min_clearance:
                continue
            return p
        # fallback
        return np.array([1.0, 1.0], dtype=np.float32)

    def _check_collision(self, p: np.ndarray) -> bool:
        # Point-in-rectangle collision (단순화: UAV는 점)
        x, y = float(p[0]), float(p[1])
        for obs in self.obstacles:
            if (obs.xmin <= x <= obs.xmax) and (obs.ymin <= y <= obs.ymax):
                return True
        return False

    def _check_path_collision(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        check_bounds: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """p0→p1 직선 경로를 샘플링해 장애물/경계 충돌 검사. 충돌 시 hit_pos 반환."""
        delta = p1 - p0
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-6:
            if self._check_collision(p0):
                return True, p0.copy(), "obstacle"
            return False, None, "none"

        step = min(0.05, dist / 10.0)
        steps = max(1, int(math.ceil(dist / max(step, 1e-6))))
        for i in range(1, steps + 1):
            t = i / steps
            p = p0 + delta * t
            x, y = float(p[0]), float(p[1])
            # 경계 충돌 (check_bounds=True 시)
            if check_bounds and (x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size):
                return True, p.astype(np.float32), "boundary"
            if self._check_collision(p.astype(np.float32)):
                return True, p.astype(np.float32), "obstacle"
        return False, None, "none"

    # --- Lidar simulation (LaserScan-like ranges) ---
    def _lidar_ranges(self, p: np.ndarray, yaw: float) -> np.ndarray:
        """num_beams 레이 ray marching 거리 (max_range 클리핑), numpy 벡터화.
        배포 시 ROS2 /scan.ranges 주입으로 교체 가능."""
        ray_step = self._LIDAR_RAY_STEP
        max_steps = int(self.max_range / ray_step)

        angles = np.linspace(-self.fov/2, self.fov/2, self.num_beams, dtype=np.float32) + yaw
        dx = np.cos(angles)  # (num_beams,)
        dy = np.sin(angles)  # (num_beams,)

        # 모든 (beam, step) 조합의 x, y 좌표: shape (num_beams, max_steps)
        ks = np.arange(1, max_steps + 1, dtype=np.float32)  # (max_steps,)
        xs = float(p[0]) + dx[:, None] * (ks[None, :] * ray_step)  # (num_beams, max_steps)
        ys = float(p[1]) + dy[:, None] * (ks[None, :] * ray_step)  # (num_beams, max_steps)

        # 경계 밖 hit
        hit = (xs < 0.0) | (xs > self.world_size) | (ys < 0.0) | (ys > self.world_size)

        # 장애물 hit: 각 장애물에 대해 numpy 배열 비교
        for obs in self.obstacles:
            hit |= ((xs >= obs.xmin) & (xs <= obs.xmax) &
                    (ys >= obs.ymin) & (ys <= obs.ymax))

        # 빔별 첫 번째 hit 거리 계산
        has_hit = hit.any(axis=1)  # (num_beams,)
        first_idx = np.argmax(hit, axis=1)  # (num_beams,) — hit 없으면 0 반환(has_hit로 마스킹)
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)
        ranges[has_hit] = np.minimum((first_idx[has_hit] + 1) * ray_step, self.max_range)

        return ranges

    def _sanitize_lidar(self, lidar: np.ndarray) -> np.ndarray:
        arr = np.asarray(lidar, dtype=np.float32)
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        arr = np.where(nan_mask | inf_mask, self.max_range, arr)
        arr = np.clip(arr, 0.0, self.max_range)
        return arr

    @staticmethod
    def _downsample_lidar(lidar: np.ndarray, target_beams: int, mode: str) -> np.ndarray:
        if target_beams <= 0:
            raise ValueError(f"target_beams must be > 0, got {target_beams}")
        if len(lidar) == target_beams:
            return lidar.copy()
        if target_beams > len(lidar):
            idx = np.linspace(0, len(lidar) - 1, target_beams).astype(int)
            return lidar[idx]

        # Split into sectors and pool per sector.
        indices = np.array_split(np.arange(len(lidar)), target_beams)
        if mode == "min":
            return np.array([float(np.min(lidar[idx])) for idx in indices], dtype=lidar.dtype)
        if mode == "avg":
            return np.array([float(np.mean(lidar[idx])) for idx in indices], dtype=lidar.dtype)
        raise ValueError(f"Unknown downsample mode: {mode}")

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi
