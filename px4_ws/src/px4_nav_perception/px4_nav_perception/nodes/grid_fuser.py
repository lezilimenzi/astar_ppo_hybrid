#!/usr/bin/env python3
import array
import math
import numpy as np
from typing import Tuple, Optional

try:
    from scipy.ndimage import binary_dilation as _scipy_dilation
    from scipy.ndimage import generate_binary_structure as _gen_struct
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time

# TF2
import tf_transformations
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


class GridFuser(Node):
    def __init__(self):
        super().__init__('grid_fuser')

        p = self.declare_parameter
        p('resolution', 0.1)                  # m/cell
        p('width', 200)                       # cells
        p('height', 200)                      # cells
        p('frame_id', 'odom')                 # grid frame (target frame for TF)
        p('robot_frame', 'base_link')         # robot frame (for rolling window)
        p('lidar_topic', '/scan')             # LiDAR LaserScan
        p('publish_rate', 5.0)                # Hz
        p('max_range', 10.0)                  # hard cap
        # 거리 r이 effective cap에 이 값(m) 이내로 가까우면 "막힘이 없다"로 보고 끝 셀을 occ로 찍지 않음
        p('max_range_no_hit_epsilon', 0.08)
        p('rolling_window', True)
        p('use_log_odds', False)
        p('prob_hit', 0.65)
        p('prob_miss', 0.35)
        p('clamp_min', -2.0)
        p('clamp_max', 2.0)
        p('occupied_thresh', 0.65)
        p('free_value', 0)                    # int grid value for free
        p('occupied_value', 100)              # int grid value for occupied
        p('unknown_value', -1)
        p('decay_rate', 0.0)                  # per publish step toward 0 (log-odds); 0 disables
        p('origin_x', -10.0)                  # used if rolling_window=False
        p('origin_y', -10.0)
        p('publish_dilation_cells', 0)        # 발행 전 occupied 셀 팽창 반경(셀). 0=비활성

        self.resolution = float(self.get_parameter('resolution').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.max_range_no_hit_epsilon = float(self.get_parameter('max_range_no_hit_epsilon').value)
        self.rolling_window = bool(self.get_parameter('rolling_window').value)
        self.use_log_odds = bool(self.get_parameter('use_log_odds').value)
        self.prob_hit = float(self.get_parameter('prob_hit').value)
        self.prob_miss = float(self.get_parameter('prob_miss').value)
        self.clamp_min = float(self.get_parameter('clamp_min').value)
        self.clamp_max = float(self.get_parameter('clamp_max').value)
        self.occ_thresh = float(self.get_parameter('occupied_thresh').value)
        self.free_value = int(self.get_parameter('free_value').value)
        self.occ_value = int(self.get_parameter('occupied_value').value)
        self.unk_value = int(self.get_parameter('unknown_value').value)
        self.decay_rate = float(self.get_parameter('decay_rate').value)
        self.static_origin_x = float(self.get_parameter('origin_x').value)
        self.static_origin_y = float(self.get_parameter('origin_y').value)
        self.publish_dilation_cells = int(self.get_parameter('publish_dilation_cells').value)
        # 8-connected (rank=2): 대각 코너 장애물 커버. __init__에서 1회 생성해 매 호출 재사용.
        self._dilation_struct = (
            _gen_struct(2, 2) if self.publish_dilation_cells > 0 and _SCIPY_OK else None
        )

        if self.use_log_odds:
            self.grid_log = np.zeros((self.height, self.width), dtype=np.float32)  # 0=unknown neutral
            self.grid_int = np.full((self.height, self.width), self.unk_value, dtype=np.int8)
            # 둘 다 표준 log-odds 공식: log(p / (1-p))
            # l_hit  > 0 (prob_hit  > 0.5), l_miss < 0 (prob_miss < 0.5)
            # update 시 둘 다 += 로 통일: hit → 증가, miss → 감소 (음수이므로)
            if not (0.5 < self.prob_hit < 1.0):
                raise ValueError(f"prob_hit={self.prob_hit} 은 (0.5, 1.0) 범위여야 합니다. "
                                 f"0.5 이하면 hit이 free로 처리됩니다.")
            if not (0.0 < self.prob_miss < 0.5):
                raise ValueError(f"prob_miss={self.prob_miss} 은 (0.0, 0.5) 범위여야 합니다. "
                                 f"0.5 이상이면 miss가 occupied로 처리됩니다.")
            if self.occ_thresh <= 0.5:
                raise ValueError(
                    f"occupied_thresh={self.occ_thresh} 은 use_log_odds=True 일 때 **0.5보다 커야** 합니다. "
                    "초기 중립 log-odds=0 → 점유 확률 prob=0.5 인데, thresh≤0.5이면 "
                    "prob >= thresh 가 모든 미관측 셀에 성립해 전부 occupied(100)로 export됩니다."
                )
            self.l_hit  = math.log(self.prob_hit  / (1.0 - self.prob_hit  + 1e-9))
            self.l_miss = math.log(self.prob_miss / (1.0 - self.prob_miss + 1e-9))
        else:
            self.grid_int = np.full((self.height, self.width), self.unk_value, dtype=np.int8)

        # Origin of the grid in frame_id (meters)
        self.origin_x = self.static_origin_x
        self.origin_y = self.static_origin_y

        # TF buffer/listener
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.grid_pub = self.create_publisher(OccupancyGrid, '/local_grid', 10)
        self.marker_pub = self.create_publisher(Marker, '/planner/path_marker', 10)

        # Subscribers
        self.create_subscription(
            LaserScan, self.lidar_topic, self.process_scan, qos_profile_sensor_data
        )
        self.create_subscription(NavPath, '/planner/path', self._cb_path, 10)
        self._latest_path: Optional[NavPath] = None
        self.get_logger().info(f"GridFuser: using {self.lidar_topic}")

        # Timer for publish & decay & (optional) rolling window
        self.timer = self.create_timer(1.0 / max(self.publish_rate, 0.1), self.publish_grid)

        self.get_logger().info("GridFuser node ready.")

    def lookup_tf(self, target_frame: str, source_frame: str, stamp: Optional[Time]) -> Optional[TransformStamped]:
        # stamp 조회 실패 시 최신 TF로 fallback (블로킹 금지)
        if stamp is not None:
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame, source_frame, stamp, rclpy.duration.Duration(seconds=0.02)
                )
            except ExtrapolationException:
                pass
            except (LookupException, ConnectivityException):
                return None
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.0)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _extract_pose_2d(self, tf_msg: TransformStamped) -> Tuple[float, float, float]:
        """TransformStamped → (tx, ty, yaw)"""
        q = tf_msg.transform.rotation
        yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return (tf_msg.transform.translation.x, tf_msg.transform.translation.y, yaw)

    def robot_pose_in_grid(self) -> Tuple[float, float]:
        """Get robot pose (robot_frame) in target frame for rolling window."""
        tx = self.lookup_tf(self.frame_id, self.robot_frame, None)
        if tx is None:
            return 0.0, 0.0
        return tx.transform.translation.x, tx.transform.translation.y

    def process_scan(self, scan: LaserScan):
        sensor_frame = scan.header.frame_id if scan.header.frame_id else ''
        scan_stamp = scan.header.stamp if scan.header.stamp.sec != 0 else None
        tf_msg = self.lookup_tf(self.frame_id, sensor_frame, scan_stamp)
        if tf_msg is None:
            return
        pose0 = self._extract_pose_2d(tf_msg)
        tx, ty, yaw = pose0
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        if self.rolling_window:
            rx, ry = self.robot_pose_in_grid()
            win_w = self.width * self.resolution
            win_h = self.height * self.resolution
            self.origin_x = rx - 0.5 * win_w
            self.origin_y = ry - 0.5 * win_h

        r_cap = min(float(scan.range_max), self.max_range) if math.isfinite(scan.range_max) else self.max_range
        eps = float(self.max_range_no_hit_epsilon)
        n = len(scan.ranges)

        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment
        ranges_raw = np.array(scan.ranges, dtype=np.float32)

        valid = np.isfinite(ranges_raw) & (ranges_raw >= float(scan.range_min)) & (ranges_raw <= r_cap)
        ranges_v = ranges_raw[valid]
        angles_v = angles[valid]
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            return

        ox_f = tx
        oy_f = ty
        sx_g = int((ox_f - self.origin_x) / self.resolution)
        sy_g = int((oy_f - self.origin_y) / self.resolution)

        px_s = ranges_v * np.cos(angles_v)
        py_s = ranges_v * np.sin(angles_v)
        gx_f = px_s * cos_y - py_s * sin_y + tx
        gy_f = px_s * sin_y + py_s * cos_y + ty

        ex_arr = ((gx_f - self.origin_x) / self.resolution).astype(np.int32)
        ey_arr = ((gy_f - self.origin_y) / self.resolution).astype(np.int32)
        is_max_return = ranges_v >= (r_cap - eps)

        valid_occ = (
            ~is_max_return
            & (ex_arr >= 0) & (ex_arr < self.width)
            & (ey_arr >= 0) & (ey_arr < self.height)
        )

        max_steps = int(math.ceil(r_cap / self.resolution / 0.95)) + 2
        # t_vals 0.95→1.0: endpoint 직전 5% 구간 cell들도 l_miss 받도록 수정.
        # endpoint cell은 is_endpoint mask로 l_miss 제외, l_hit만 적용.
        t_vals = np.linspace(0.0, 1.0, max_steps, dtype=np.float32)

        dx = (ex_arr - sx_g).astype(np.float32)
        dy = (ey_arr - sy_g).astype(np.float32)
        cx_all = (sx_g + np.outer(dx, t_vals)).astype(np.int32)
        cy_all = (sy_g + np.outer(dy, t_vals)).astype(np.int32)

        # endpoint는 hit과 겹치면 miss가 장애물 증거를 지울 수 있어 제외
        is_endpoint = (cx_all == ex_arr[:, None]) & (cy_all == ey_arr[:, None])
        cx_prev = np.concatenate([cx_all[:, :1], cx_all[:, :-1]], axis=1)
        cy_prev = np.concatenate([cy_all[:, :1], cy_all[:, :-1]], axis=1)
        is_dup = (cx_all == cx_prev) & (cy_all == cy_prev)
        is_dup[:, 0] = False

        in_bounds = (
            (cx_all >= 0) & (cx_all < self.width)
            & (cy_all >= 0) & (cy_all < self.height)
            & ~is_endpoint
            & ~is_dup
        )

        # log-odds: miss 먼저 적용 후 hit (clamp와 순서 꼬임 방지)
        if self.use_log_odds:
            np.add.at(self.grid_log, (cy_all[in_bounds], cx_all[in_bounds]), self.l_miss)
            np.clip(self.grid_log, self.clamp_min, self.clamp_max, out=self.grid_log)
            np.add.at(self.grid_log, (ey_arr[valid_occ], ex_arr[valid_occ]), self.l_hit)
            np.clip(self.grid_log, self.clamp_min, self.clamp_max, out=self.grid_log)
        else:
            self.grid_int[cy_all[in_bounds], cx_all[in_bounds]] = self.free_value
            self.grid_int[ey_arr[valid_occ], ex_arr[valid_occ]] = self.occ_value

    def _export_int_grid(self) -> np.ndarray:
        if self.use_log_odds:
            # Convert log-odds to probability for export
            prob = 1.0 - 1.0 / (1.0 + np.exp(np.clip(self.grid_log, -87.0, 87.0)))
            out = np.full_like(self.grid_log, self.unk_value, dtype=np.int8)
            out[prob >= self.occ_thresh] = self.occ_value
            out[(prob < self.occ_thresh) & (prob >= 0.0)] = self.free_value
            out = out.astype(np.int8)
        else:
            out = self.grid_int if self._dilation_struct is None else self.grid_int.copy()
        if self._dilation_struct is not None:
            occ_mask = (out == self.occ_value)
            dilated = _scipy_dilation(occ_mask, structure=self._dilation_struct,
                                      iterations=self.publish_dilation_cells)
            out[dilated] = self.occ_value
        return out

    def _apply_decay(self):
        if self.use_log_odds and self.decay_rate > 0.0:
            # decay toward 0 (unknown) to forget stale evidence slightly
            self.grid_log *= (1.0 - self.decay_rate)

    def _cb_path(self, msg: NavPath) -> None:
        self._latest_path = msg
        self._publish_path_marker()

    def _publish_path_marker(self) -> None:
        if self._latest_path is None or not self._latest_path.poses:
            return
        m = Marker()
        m.header = self._latest_path.header
        m.ns = 'planned_path'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.08      # 선 두께(m)
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.2
        m.color.a = 0.9
        m.pose.orientation.w = 1.0
        for ps in self._latest_path.poses:
            pt = Point()
            pt.x = ps.pose.position.x
            pt.y = ps.pose.position.y
            pt.z = ps.pose.position.z
            m.points.append(pt)
        self.marker_pub.publish(m)

    def publish_grid(self):
        self._apply_decay()

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        grid_now = self._export_int_grid()
        msg.data = array.array('b', grid_now.flatten().astype(np.int8).tobytes())
        self.grid_pub.publish(msg)
        self._publish_path_marker()


def main(args=None):
    rclpy.init(args=args)
    node = GridFuser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
