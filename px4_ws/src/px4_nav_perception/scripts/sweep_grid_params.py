#!/usr/bin/env python3
"""grid_fuser log-odds 파라미터 오프라인 스윕.

rosbag2 에 기록된 /scan + /tf 를 한 번만 읽어 광선 투사 결과(셀 인덱스)를 캐시한 뒤,
(prob_hit, prob_miss, clamp_min, clamp_max) 조합마다 log-odds 누적만 다시 돌린다.
광선 기하는 파라미터와 무관하므로 비행을 반복할 필요가 없다.

채점은 "A* 가 이 지도를 어떻게 보는가" 기준이다. obstacle_course.world 의 벽
16개가 ground truth 이므로 프레임마다 세 상태 중 하나로 분류한다:

  ok      : home→goal 경로가 존재하고, 실제 벽을 통과하지 않는다
  blocked : 경로가 없다              → 유령 셀이 통로를 막음 (거짓 양성)
  leak    : 경로가 있으나 실제 벽을 관통해야만 성립 → 벽에 구멍 (거짓 음성)

사용법
------
  # 1) 재현 검증: 현재 yaml 값으로 만든 지도가 기록된 /local_grid 와 일치하는지
  ./sweep_grid_params.py --bag ~/bags/course_l_A --validate

  # 2) 스윕
  ./sweep_grid_params.py --bag ~/bags/course_l_A --out sweep.csv

  # 3) 격자 지정
  ./sweep_grid_params.py --bag ~/bags/course_l_A \
      --prob-hit 0.70 0.75 0.80 0.85 --prob-miss 0.25 0.30 0.35 0.40 \
      --clamp-min -2 -4 -6 --clamp-max 2 3 4

ROS 2 환경을 source 한 상태에서 실행해야 한다 (rosbag2_py, tf2_ros 사용).
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rclpy.time
    import tf2_ros
    from builtin_interfaces.msg import Duration as MsgDuration
except ImportError as exc:  # pragma: no cover
    sys.exit(f"ROS 2 환경을 source 해야 합니다 ({exc}).")

from scipy import ndimage


# ---------------------------------------------------------------------------
# 고정 설정 — grid_fuser.yaml / params.yaml 과 일치해야 한다
# ---------------------------------------------------------------------------
@dataclass
class FixedCfg:
    resolution: float = 0.20
    width: int = 261
    height: int = 261
    origin_x: float = -26.1
    origin_y: float = -26.1
    frame_id: str = "odom"
    max_range: float = 10.0
    max_range_no_hit_epsilon: float = 0.08
    lidar_range_min_override: float = 0.35
    occupied_backfill_cells: int = 0        # 히트 셀 뒤(빔 방향) N 셀 추가 점유
    protect_hit_cells_from_miss: bool = False
    occupied_thresh: float = 0.55
    decay_rate: float = 0.001
    publish_rate: float = 5.0
    publish_dilation_cells: int = 1      # grid_fuser.yaml
    astar_min_inflation_cells: int = 1   # params.yaml (A* 가 추가로 거는 팽창)
    occ_threshold: int = 50              # params.yaml (A* 의 점유 판정)
    home_xy: Tuple[float, float] = (0.0, 8.0)
    goal_xy: Tuple[float, float] = (-8.0, -8.2)


# obstacle_course.world 의 static box: (center_x, center_y, size_x, size_y)
WORLD_WALLS: List[Tuple[float, float, float, float]] = [
    # +X ㄹ 코스 (Course R)
    (2.0, -2.0, 0.3, 16.0),    # r_outer_left
    (10.0, 1.0, 0.3, 18.0),    # r_outer_right
    (5.0, 10.0, 10.0, 0.3),    # r_outer_top
    (5.0, -10.0, 10.0, 0.3),   # r_outer_bottom
    (4.5, 6.0, 5.0, 0.3),      # inner_r_h1
    (7.5, 2.0, 5.0, 0.3),      # inner_r_h2
    (4.5, -2.0, 5.0, 0.3),     # inner_r_h3
    (7.5, -6.0, 5.0, 0.3),     # inner_r_h4
    # -X S 코스 (Course L)
    (-10.0, 1.0, 0.3, 18.0),   # s_outer_left
    (-2.0, -2.0, 0.3, 16.0),   # s_outer_right
    (-5.0, 10.0, 10.0, 0.3),   # s_outer_top
    (-5.0, -10.0, 10.0, 0.3),  # s_outer_bottom
    (-4.0, 0.0, 0.3, 8.0),     # inner_s_v1
    (-8.0, 0.0, 0.3, 8.0),     # inner_s_v2
    (-6.0, 6.0, 0.3, 8.0),     # inner_s_v3
    (-6.0, -6.0, 0.3, 8.0),    # inner_s_v4
]


# 반드시 통과해야 하는 관문. (이름, x범위, y범위, 축)
#   축 'x' = 각 y 행에서 x 방향 자유 셀 수를 센다 (세로 통로)
#   축 'y' = 각 x 열에서 y 방향 자유 셀 수를 센다 (가로로 넘는 개구부)
GATES: List[Tuple[str, Tuple[float, float], Tuple[float, float], str]] = [
    # 진입구: s_outer_right(x=-2, y<=6) 상단과 s_outer_top(y=10) 사이. Course L 로
    # 들어가는 유일한 통로이며, 여기가 막히면 A* 가 Course R 쪽으로 우회한다.
    ("E  entrance",   (-2.30, -1.70), (6.2, 9.8), "y"),
    ("C1 v1|s_right", (-3.85, -2.15), (-3.6, 3.6), "x"),   # 첫 통로 1.7m
    ("C2 v2|v4",      (-7.85, -6.15), (-3.8, -2.2), "x"),  # 두번째 통로 1.7m
    ("C3 v2|v3",      (-7.85, -6.15), (2.2, 3.8), "x"),    # 북측 통로 1.7m
    ("G  gap x=-6",   (-6.30, -5.70), (-1.8, 1.8), "y"),   # v3/v4 사이 4m 개구부
]


def _range_to_idx(lo: float, hi: float, origin: float, res: float, n: int) -> Tuple[int, int]:
    a = int(math.floor((lo - origin) / res))
    b = int(math.ceil((hi - origin) / res))
    return max(0, a), min(n, max(a + 1, b))


def build_gates(cfg: FixedCfg) -> List[Tuple[str, slice, slice, str]]:
    out = []
    for name, (x0, x1), (y0, y1), axis in GATES:
        j0, j1 = _range_to_idx(x0, x1, cfg.origin_x, cfg.resolution, cfg.width)
        i0, i1 = _range_to_idx(y0, y1, cfg.origin_y, cfg.resolution, cfg.height)
        out.append((name, slice(i0, i1), slice(j0, j1), axis))
    return out


def build_wall_boxes(cfg: FixedCfg) -> List[Tuple[slice, slice, str]]:
    """벽별 셀 범위 + 두께 축. 'v' = 세로벽(x 방향으로 얇음)."""
    out = []
    for cx, cy, sx, sy in WORLD_WALLS:
        j0, j1 = _range_to_idx(cx - sx / 2, cx + sx / 2, cfg.origin_x, cfg.resolution, cfg.width)
        i0, i1 = _range_to_idx(cy - sy / 2, cy + sy / 2, cfg.origin_y, cfg.resolution, cfg.height)
        out.append((slice(i0, i1), slice(j0, j1), "v" if sx < sy else "h"))
    return out


def build_truth_masks(cfg: FixedCfg) -> Tuple[np.ndarray, np.ndarray]:
    """(overlap, core) 벽 마스크.

    overlap: 셀이 벽 사각형과 조금이라도 겹침 → 침식(FN) 채점용
    core   : 셀 중심이 벽 내부       → leak(경로가 벽을 관통) 판정용
    """
    res = cfg.resolution
    xs_lo = cfg.origin_x + np.arange(cfg.width) * res
    ys_lo = cfg.origin_y + np.arange(cfg.height) * res
    xs_hi, ys_hi = xs_lo + res, ys_lo + res
    xs_c, ys_c = xs_lo + 0.5 * res, ys_lo + 0.5 * res

    overlap = np.zeros((cfg.height, cfg.width), dtype=bool)
    core = np.zeros_like(overlap)
    for cx, cy, sx, sy in WORLD_WALLS:
        x0, x1 = cx - sx / 2.0, cx + sx / 2.0
        y0, y1 = cy - sy / 2.0, cy + sy / 2.0
        ox = (xs_hi > x0) & (xs_lo < x1)
        oy = (ys_hi > y0) & (ys_lo < y1)
        overlap |= oy[:, None] & ox[None, :]
        kx = (xs_c >= x0) & (xs_c <= x1)
        ky = (ys_c >= y0) & (ys_c <= y1)
        core |= ky[:, None] & kx[None, :]
    return overlap, core


# ---------------------------------------------------------------------------
# bag 읽기 + 광선 투사 캐시 (파라미터 무관)
# ---------------------------------------------------------------------------
@dataclass
class ScanCache:
    """스캔 1개의 셀 인덱스. flat 인덱스 = i * width + j."""
    stamp: float
    miss_idx: np.ndarray   # int32, l_miss 를 받을 셀
    miss_cnt: np.ndarray   # int32, 같은 스캔에서 중복 통과한 횟수
    hit_idx: np.ndarray
    hit_cnt: np.ndarray


@dataclass
class BagData:
    scans: List[ScanCache] = field(default_factory=list)
    grids: List[Tuple[float, np.ndarray]] = field(default_factory=list)  # 기록된 /local_grid


def _open_bag(path: str):
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=path, storage_id="")
    conv = rosbag2_py.ConverterOptions("", "")
    reader.open(storage, conv)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, types


def load_bag(path: str, cfg: FixedCfg, verbose: bool = True) -> BagData:
    reader, types = _open_bag(path)
    needed = {"/scan", "/tf", "/tf_static"}
    missing = needed - set(types)
    if missing:
        sys.exit(f"bag 에 필수 토픽이 없습니다: {sorted(missing)}")

    buf = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=120))
    data = BagData()
    n_skip = 0

    while reader.has_next():
        topic, raw, _t = reader.read_next()
        msg_type = get_message(types[topic])
        msg = deserialize_message(raw, msg_type)

        if topic == "/tf_static":
            for tr in msg.transforms:
                buf.set_transform_static(tr, "bag")
            continue
        if topic == "/tf":
            for tr in msg.transforms:
                buf.set_transform(tr, "bag")
            continue
        if topic == "/local_grid":
            arr = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
            data.grids.append((_stamp_sec(msg.header.stamp), arr))
            continue
        if topic != "/scan":
            continue

        pose = _lookup_pose(buf, cfg.frame_id, msg.header)
        if pose is None:
            n_skip += 1
            continue
        cached = _cast_scan(msg, pose, cfg)
        if cached is not None:
            data.scans.append(cached)

    if verbose:
        print(f"[bag] scans={len(data.scans)} (TF 없음으로 건너뜀 {n_skip}), "
              f"기록된 local_grid={len(data.grids)}")
    if not data.scans:
        sys.exit("사용 가능한 스캔이 없습니다. /tf_static 이 bag 에 포함됐는지 확인하세요.")
    return data


def _stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _lookup_pose(buf, target: str, header) -> Optional[Tuple[float, float, float]]:
    """target 프레임에서 본 센서 프레임의 (x, y, yaw). 노드의 lookup_tf 와 동일 경로."""
    src = header.frame_id or ""
    if not src:
        return None
    for stamp in (header.stamp, rclpy.time.Time()):
        try:
            tr = buf.lookup_transform(target, src, stamp)
        except Exception:
            continue
        t, q = tr.transform.translation, tr.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return (t.x, t.y, math.atan2(siny, cosy))
    return None


def _cast_scan(scan, pose: Tuple[float, float, float], cfg: FixedCfg) -> Optional[ScanCache]:
    """grid_fuser.process_scan 의 기하 부분을 그대로 재현한다."""
    tx, ty, yaw = pose
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    rmax = float(scan.range_max)
    r_cap = min(rmax, cfg.max_range) if math.isfinite(rmax) else cfg.max_range
    eps = cfg.max_range_no_hit_epsilon
    n = len(scan.ranges)
    if n == 0:
        return None

    angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment
    ranges_raw = np.asarray(scan.ranges, dtype=np.float32)

    # 자기감지 방어 (grid_fuser 와 동일)
    r_min_eff = max(float(scan.range_min), cfg.lidar_range_min_override)
    valid = np.isfinite(ranges_raw) & (ranges_raw >= r_min_eff) & (ranges_raw <= r_cap)
    if not valid.any():
        return None
    ranges_v, angles_v = ranges_raw[valid], angles[valid]

    sx_g = int((tx - cfg.origin_x) / cfg.resolution)
    sy_g = int((ty - cfg.origin_y) / cfg.resolution)

    px_s, py_s = ranges_v * np.cos(angles_v), ranges_v * np.sin(angles_v)
    gx_f = px_s * cos_y - py_s * sin_y + tx
    gy_f = px_s * sin_y + py_s * cos_y + ty
    ex = ((gx_f - cfg.origin_x) / cfg.resolution).astype(np.int32)
    ey = ((gy_f - cfg.origin_y) / cfg.resolution).astype(np.int32)

    is_max_return = ranges_v >= (r_cap - eps)
    valid_occ = (~is_max_return
                 & (ex >= 0) & (ex < cfg.width)
                 & (ey >= 0) & (ey < cfg.height))

    # 폐색 보정 backfill (grid_fuser 와 동일)
    bf_cells = []
    if cfg.occupied_backfill_cells > 0:
        dxr, dyr = gx_f - tx, gy_f - ty
        rn = np.hypot(dxr, dyr)
        rn = np.where(rn < 1e-6, 1e-6, rn)
        ux, uy = dxr / rn, dyr / rn
        for k in range(1, cfg.occupied_backfill_cells + 1):
            bx = ((gx_f + ux * (k * cfg.resolution) - cfg.origin_x) / cfg.resolution).astype(np.int32)
            by = ((gy_f + uy * (k * cfg.resolution) - cfg.origin_y) / cfg.resolution).astype(np.int32)
            ok = (valid_occ & (bx >= 0) & (bx < cfg.width)
                  & (by >= 0) & (by < cfg.height))
            if np.any(ok):
                bf_cells.append((bx[ok], by[ok]))

    max_steps = int(math.ceil(r_cap / cfg.resolution / 0.95)) + 2
    t_vals = np.linspace(0.0, 1.0, max_steps, dtype=np.float32)
    dx = (ex - sx_g).astype(np.float32)
    dy = (ey - sy_g).astype(np.float32)
    cx_all = (sx_g + np.outer(dx, t_vals)).astype(np.int32)
    cy_all = (sy_g + np.outer(dy, t_vals)).astype(np.int32)

    is_endpoint = (cx_all == ex[:, None]) & (cy_all == ey[:, None])
    cx_prev = np.concatenate([cx_all[:, :1], cx_all[:, :-1]], axis=1)
    cy_prev = np.concatenate([cy_all[:, :1], cy_all[:, :-1]], axis=1)
    is_dup = (cx_all == cx_prev) & (cy_all == cy_prev)
    is_dup[:, 0] = False

    in_bounds = ((cx_all >= 0) & (cx_all < cfg.width)
                 & (cy_all >= 0) & (cy_all < cfg.height)
                 & ~is_endpoint & ~is_dup)

    # 같은 스캔의 모든 히트/backfill 셀을 l_miss 에서 제외 (grid_fuser 와 동일)
    if cfg.protect_hit_cells_from_miss:
        hit_img = np.zeros((cfg.height, cfg.width), dtype=bool)
        hit_img[ey[valid_occ], ex[valid_occ]] = True
        for bx, by in bf_cells:
            hit_img[by, bx] = True
        flat_ib = in_bounds.ravel().copy()
        sel = np.flatnonzero(flat_ib)
        drop = hit_img[cy_all.ravel()[sel], cx_all.ravel()[sel]]
        flat_ib[sel[drop]] = False
        in_bounds = flat_ib.reshape(in_bounds.shape)

    miss_flat = (cy_all[in_bounds].astype(np.int64) * cfg.width
                 + cx_all[in_bounds].astype(np.int64))
    hit_parts = [ey[valid_occ].astype(np.int64) * cfg.width + ex[valid_occ].astype(np.int64)]
    for bx, by in bf_cells:
        hit_parts.append(by.astype(np.int64) * cfg.width + bx.astype(np.int64))
    hit_flat = np.concatenate(hit_parts) if len(hit_parts) > 1 else hit_parts[0]

    m_idx, m_cnt = np.unique(miss_flat, return_counts=True)
    h_idx, h_cnt = np.unique(hit_flat, return_counts=True)
    return ScanCache(
        stamp=_stamp_sec(scan.header.stamp),
        miss_idx=m_idx.astype(np.int32), miss_cnt=m_cnt.astype(np.int32),
        hit_idx=h_idx.astype(np.int32), hit_cnt=h_cnt.astype(np.int32),
    )


# ---------------------------------------------------------------------------
# 파라미터 조합 1개 평가
# ---------------------------------------------------------------------------
@dataclass
class Params:
    prob_hit: float
    prob_miss: float
    clamp_min: float
    clamp_max: float

    @property
    def l_hit(self) -> float:
        return math.log(self.prob_hit / (1.0 - self.prob_hit + 1e-9))

    @property
    def l_miss(self) -> float:
        return math.log(self.prob_miss / (1.0 - self.prob_miss + 1e-9))

    @property
    def rho(self) -> float:
        return abs(self.l_hit / self.l_miss)


@dataclass
class Score:
    frames: int = 0
    gate_evals: int = 0      # (프레임 × 활성 관문) 평가 횟수
    gate_blocked: int = 0    # 그중 자유 셀 0 으로 막힌 횟수  ← 거짓 양성의 실제 피해
    gate_min_w: float = 9.9  # 관문 최소 자유 셀 폭(전체 최소)
    hole_rows: float = 0.0   # 벽을 관통하는 구멍 행 수 (프레임 평균) ← 거짓 음성의 실제 피해
    fp_cells: float = 0.0    # 벽이 아닌데 occupied (프레임 평균, 진단용)
    fn_cells: float = 0.0    # 관측된 벽인데 free (프레임 평균, 진단용)

    @property
    def blocked_frac(self) -> float:
        return self.gate_blocked / self.gate_evals if self.gate_evals else 0.0

    @property
    def cost(self) -> float:
        """막힘을 우선 최소화하되 벽 구멍도 허용하지 않는 단일 비용."""
        return self.blocked_frac * 100.0 + self.hole_rows


def evaluate(params: Params, data: BagData, cfg: FixedCfg,
             truth_overlap: np.ndarray, truth_core: np.ndarray,
             collect_grids: bool = False
             ) -> Tuple[Score, List[Tuple[float, np.ndarray]]]:
    thr_logodds = math.log(cfg.occupied_thresh / (1.0 - cfg.occupied_thresh))
    if params.clamp_max <= thr_logodds:
        # 어떤 셀도 occupied 가 될 수 없는 조합 — 전면 무효
        return Score(frames=1, gate_evals=1, gate_blocked=1, gate_min_w=0.0), []

    n_cells = cfg.height * cfg.width
    grid_log = np.zeros(n_cells, dtype=np.float32)
    ever_hit = np.zeros(n_cells, dtype=bool)
    ever_seen = np.zeros(n_cells, dtype=bool)   # hit 또는 광선 통과 = 관측됨

    struct8 = ndimage.generate_binary_structure(2, 2)   # grid_fuser dilation
    struct4 = ndimage.generate_binary_structure(2, 1)   # A* inflation (scipy 기본)

    gates = build_gates(cfg)
    wall_boxes = build_wall_boxes(cfg)

    score = Score()
    grids_out: List[Tuple[float, np.ndarray]] = []
    publish_period = 1.0 / cfg.publish_rate
    next_publish = data.scans[0].stamp

    for sc in data.scans:
        # --- log-odds 누적: miss 먼저, clip, hit, clip (노드와 동일 순서) ---
        grid_log[sc.miss_idx] += params.l_miss * sc.miss_cnt
        np.clip(grid_log, params.clamp_min, params.clamp_max, out=grid_log)
        grid_log[sc.hit_idx] += params.l_hit * sc.hit_cnt
        np.clip(grid_log, params.clamp_min, params.clamp_max, out=grid_log)
        ever_hit[sc.hit_idx] = True
        ever_seen[sc.hit_idx] = True
        ever_seen[sc.miss_idx] = True

        if sc.stamp < next_publish:
            continue
        next_publish = sc.stamp + publish_period

        occ = (grid_log >= thr_logodds).reshape(cfg.height, cfg.width)
        if cfg.publish_dilation_cells > 0:
            occ = ndimage.binary_dilation(occ, structure=struct8,
                                          iterations=cfg.publish_dilation_cells)
        if collect_grids:
            grids_out.append((sc.stamp, np.where(occ, np.int8(100), np.int8(0))))

        # --- A* 가 보는 그리드 ---
        planner_occ = occ
        if cfg.astar_min_inflation_cells > 0:
            planner_occ = ndimage.binary_dilation(
                occ, structure=struct4, iterations=cfg.astar_min_inflation_cells)

        score.frames += 1
        free = ~planner_occ
        seen2d = ever_seen.reshape(cfg.height, cfg.width)

        # --- 관문: 지나가야 하는 통로가 막혔는가 (거짓 양성의 실제 피해) ---
        for _name, si, sj, axis in gates:
            if seen2d[si, sj].mean() < 0.8:
                continue                      # 아직 관측 안 된 관문은 채점 제외
            counts = free[si, sj].sum(axis=1 if axis == "x" else 0)
            score.gate_evals += 1
            w = int(counts.min())
            score.gate_min_w = min(score.gate_min_w, float(w))
            if w == 0:
                score.gate_blocked += 1

        # --- 벽 관통 구멍: A* 가 실제 벽을 통과할 수 있는가 (거짓 음성의 실제 피해) ---
        for si, sj, orient in wall_boxes:
            occ_sub = planner_occ[si, sj]
            seen_sub = seen2d[si, sj]
            ax = 1 if orient == "v" else 0
            solid = occ_sub.any(axis=ax)      # 그 행/열에 점유 셀이 하나라도 있는가
            observed = seen_sub.any(axis=ax)
            score.hole_rows += float(np.count_nonzero(observed & ~solid))

        observed_wall = ever_hit.reshape(cfg.height, cfg.width) & truth_overlap
        raw_occ = (grid_log >= thr_logodds).reshape(cfg.height, cfg.width)
        score.fp_cells += float(np.count_nonzero(raw_occ & ~truth_overlap))
        score.fn_cells += float(np.count_nonzero(observed_wall & ~raw_occ))

        # 노드는 publish_grid() 말미에 _apply_decay() 를 호출한다 (export 직후)
        if cfg.decay_rate > 0.0:
            grid_log *= (1.0 - cfg.decay_rate)

    if score.frames:
        score.fp_cells /= score.frames
        score.fn_cells /= score.frames
        score.hole_rows /= score.frames
    if score.gate_min_w > 9.0:
        score.gate_min_w = float("nan")   # 활성 관문이 하나도 없었던 경우
    return score, grids_out


def _xy_to_ij(xy: Tuple[float, float], cfg: FixedCfg) -> Tuple[int, int]:
    j = int((xy[0] - cfg.origin_x) / cfg.resolution)
    i = int((xy[1] - cfg.origin_y) / cfg.resolution)
    return (max(0, min(cfg.height - 1, i)), max(0, min(cfg.width - 1, j)))


# ---------------------------------------------------------------------------
# 재현 검증
# ---------------------------------------------------------------------------
def validate(data: BagData, cfg: FixedCfg, params: Params) -> None:
    if not data.grids:
        sys.exit("bag 에 /local_grid 가 없어 검증할 수 없습니다.")
    truth_overlap, truth_core = build_truth_masks(cfg)
    _, mine = evaluate(params, data, cfg, truth_overlap, truth_core, collect_grids=True)
    if not mine:
        sys.exit("비교할 프레임이 없습니다.")

    rec_stamps = np.array([s for s, _ in data.grids], dtype=np.float64)
    print(f"[validate] 재현 프레임 {len(mine)} / 기록 프레임 {len(data.grids)}")
    print(f"{'t(s)':>8} {'재현':>7} {'기록':>7} {'IoU':>7} {'재현만':>7} {'기록만':>7}")

    ious = []
    # 누적 지도이므로 후반부를 본다. 발행 시각이 정확히 일치하지 않으므로 최근접 매칭.
    for k in np.linspace(len(mine) * 0.5, len(mine) - 1, 6).astype(int):
        t_mine, g_mine = mine[k]
        j = int(np.argmin(np.abs(rec_stamps - t_mine)))
        a = (g_mine == 100)
        b = (data.grids[j][1] == 100)
        inter = int(np.count_nonzero(a & b))
        union = int(np.count_nonzero(a | b))
        iou = inter / union if union else 1.0
        ious.append(iou)
        print(f"{t_mine - mine[0][0]:8.1f} {int(a.sum()):7d} {int(b.sum()):7d} "
              f"{iou:7.4f} {int(np.count_nonzero(a & ~b)):7d} "
              f"{int(np.count_nonzero(b & ~a)):7d}")

    mean_iou = float(np.mean(ious))
    print(f"\n[validate] 평균 IoU = {mean_iou:.4f}")
    if mean_iou >= 0.95:
        print("  → 재현이 노드와 일치합니다. 스윕 결과를 신뢰할 수 있습니다.")
    else:
        print("  → 0.95 미만. FixedCfg 값이 yaml/params 와 어긋났거나,")
        print("     녹화 당시 노드 파라미터가 validate 에 넘긴 Params 와 다릅니다.")
        print("     grid_fuser.yaml 의 prob_hit/prob_miss/clamp_min/clamp_max 를 확인하세요.")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="rosbag2 디렉터리")
    ap.add_argument("--out", default=None, help="결과 CSV 경로")
    ap.add_argument("--validate", action="store_true",
                    help="스윕 대신 현재 값 재현 검증만 수행")
    ap.add_argument("--prob-hit", type=float, nargs="+",
                    default=[0.70, 0.75, 0.80, 0.85])
    ap.add_argument("--prob-miss", type=float, nargs="+",
                    default=[0.25, 0.30, 0.35, 0.40])
    ap.add_argument("--clamp-min", type=float, nargs="+", default=[-2.0, -4.0, -6.0])
    ap.add_argument("--clamp-max", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--dilation", type=int, default=None,
                    help="grid_fuser.yaml 의 publish_dilation_cells 덮어쓰기 (발행 시 팽창)")
    ap.add_argument("--astar-inflation", type=int, default=None,
                    help="params.yaml 의 astar_min_inflation_cells 덮어쓰기 (A* 팽창)")
    ap.add_argument("--backfill", type=int, default=None,
                    help="occupied_backfill_cells 덮어쓰기 (히트 셀 뒤 N 셀 점유)")
    ap.add_argument("--protect-hits", action="store_true",
                    help="protect_hit_cells_from_miss 활성화")
    ap.add_argument("--resolution", type=float, default=None,
                    help="격자 해상도(m) 덮어쓰기. 커버 범위 유지하려면 --grid-size 도 함께")
    ap.add_argument("--grid-size", type=int, default=None,
                    help="width/height(셀) 덮어쓰기. origin 은 -res*size/2 로 자동 재계산")
    args = ap.parse_args()

    cfg = FixedCfg()
    if args.resolution is not None:
        cfg.resolution = args.resolution
    if args.grid_size is not None:
        cfg.width = cfg.height = args.grid_size
    if args.resolution is not None or args.grid_size is not None:
        # 격자 중심을 원점에 유지 (기존 261×0.2 → ±26.1m 와 동일한 규칙)
        cfg.origin_x = -cfg.resolution * cfg.width / 2.0
        cfg.origin_y = -cfg.resolution * cfg.height / 2.0
        print(f"[cfg] resolution={cfg.resolution} grid={cfg.width}x{cfg.height} "
              f"origin=({cfg.origin_x:.2f},{cfg.origin_y:.2f}) "
              f"커버 ±{cfg.resolution*cfg.width/2:.1f}m")
    if args.dilation is not None:
        cfg.publish_dilation_cells = args.dilation
    if args.astar_inflation is not None:
        cfg.astar_min_inflation_cells = args.astar_inflation
    if args.backfill is not None:
        cfg.occupied_backfill_cells = args.backfill
    if args.protect_hits:
        cfg.protect_hit_cells_from_miss = True
    print(f"[cfg] publish_dilation={cfg.publish_dilation_cells} "
          f"astar_inflation={cfg.astar_min_inflation_cells} "
          f"backfill={cfg.occupied_backfill_cells} "
          f"protect_hits={cfg.protect_hit_cells_from_miss} "
          f"(측방 팽창 {cfg.publish_dilation_cells + cfg.astar_min_inflation_cells}셀 "
          f"= {(cfg.publish_dilation_cells + cfg.astar_min_inflation_cells) * cfg.resolution:.1f}m)")

    data = load_bag(args.bag, cfg)

    if args.validate:
        validate(data, cfg, Params(0.80, 0.30, -6.0, 3.0))
        return

    truth_overlap, truth_core = build_truth_masks(cfg)
    combos = [Params(*c) for c in itertools.product(
        args.prob_hit, args.prob_miss, args.clamp_min, args.clamp_max)]
    print(f"[sweep] {len(combos)} 조합 × {len(data.scans)} 스캔")

    rows = []
    for k, p in enumerate(combos, 1):
        score, _ = evaluate(p, data, cfg, truth_overlap, truth_core)
        rows.append((p, score))
        print(f"  [{k:3d}/{len(combos)}] hit={p.prob_hit:.2f} miss={p.prob_miss:.2f} "
              f"cmin={p.clamp_min:+.1f} cmax={p.clamp_max:+.1f} | rho={p.rho:5.2f} "
              f"blocked={score.blocked_frac:6.2%} minW={score.gate_min_w:4.1f} "
              f"holes={score.hole_rows:6.2f} FP={score.fp_cells:7.1f} FN={score.fn_cells:6.1f}")

    rows.sort(key=lambda r: (r[1].cost, r[1].fp_cells))
    print("\n=== 상위 15 (비용 = 관문막힘% × 100 + 벽구멍) ===")
    print("  minW_m / hole_m 은 미터 환산 — 해상도가 다른 실행끼리는 이 열로 비교할 것")
    print(f"{'hit':>5} {'miss':>5} {'cmin':>6} {'cmax':>6} {'rho':>6} "
          f"{'blocked':>8} {'minW_m':>7} {'hole_m':>7} {'holes':>7} {'FP':>8} {'FN':>7}")
    for p, s in rows[:15]:
        print(f"{p.prob_hit:5.2f} {p.prob_miss:5.2f} {p.clamp_min:+6.1f} {p.clamp_max:+6.1f} "
              f"{p.rho:6.2f} {s.blocked_frac:8.2%} {s.gate_min_w * cfg.resolution:7.2f} "
              f"{s.hole_rows * cfg.resolution:7.2f} {s.hole_rows:7.2f} "
              f"{s.fp_cells:8.1f} {s.fn_cells:7.1f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("prob_hit,prob_miss,clamp_min,clamp_max,rho,frames,"
                     "gate_evals,gate_blocked,blocked_frac,gate_min_w,gate_min_w_m,"
                     "hole_rows,hole_m,cost,fp_cells,fn_cells,resolution\n")
            for p, s in rows:
                fh.write(f"{p.prob_hit},{p.prob_miss},{p.clamp_min},{p.clamp_max},"
                         f"{p.rho:.4f},{s.frames},{s.gate_evals},{s.gate_blocked},"
                         f"{s.blocked_frac:.6f},{s.gate_min_w:.1f},"
                         f"{s.gate_min_w * cfg.resolution:.3f},{s.hole_rows:.3f},"
                         f"{s.hole_rows * cfg.resolution:.3f},"
                         f"{s.cost:.4f},{s.fp_cells:.2f},{s.fn_cells:.2f},{cfg.resolution}\n")
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
