# Pipeline Overview: Perception → A* → PPO → PX4 Control

이 문서는 `px4_ws/src` 워크스페이스의 ROS 2 노드·토픽 흐름과, 학습 코드(`rl/env_2d_nav.py`, `rl/env_2d_nav_ppo_only.py`, `rl/train_ppo_2d.py`)와의 대응 관계를 정리한다. 논문 *"A\* 전역 경로 계획과 PPO 기반 경로 추종을 결합한 UAV 하이브리드 항법 프레임워크"* 의 인지(Perception)–판단(Decision)–제어(Control) 3-모듈 구조(Fig. 1)에 대응하며, 배포 노드 `rl_offboard_node`는 **astar / ppo / hybrid** 3가지 `nav_mode`를 지원한다(논문의 3-모드 비교 실험과 동일).

> **2.5D 항법:** 고도(z)는 altitude hold로 일정하게 유지하고, 수평 평면의 경로 추종·장애물 회피에 집중한다. 행동 공간은 선속도 + 요 각속도 2D이다.

---

## 0. 시뮬레이션 월드

평가에 쓰이는 월드는 두 가지이며, 논문의 4개 시나리오(R/L × obstacle/military)에 대응한다. 둘 다
`PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/` 아래에 있다.

| 월드 파일 | 크기 | 성격 | 논문 시나리오 |
|-----------|------|------|---------------|
| `obstacle_course.world` | 20×20 m | ㄹ/S자 좁은 통로 코스 (**현재 goals.yaml 활성**) | R_obstacle / L_obstacle |
| `military_airfield.world` | 50×50 m | 시가지/군 부대 모사 (건물·차량·방벽) | R_military / L_military |

**읽는 법:** Gazebo `<box><size>`는 **(dx, dy, dz)** = (x 길이, y 길이, 높이). `<pose>`는 모델/링크 **중심** 좌표. 끝점 구간은 `[중심 − size/2, 중심 + size/2]`. 모든 장애물은 `static` 박스이고 **높이 dz=8 m, pose z=4 m(중심)** 이라 z∈[0, 8] 을 막는다(순항 고도 z=4 는 벽 중간 높이).

### 0.1 `obstacle_course.world` (20×20, ㄹ/S 코스) — 활성

**+X ㄹ 코스 (Course R 영역, x>0)**

| 모델명 | 중심 (x, y, z) | size (dx, dy, dz) | 점유 구간 |
|--------|----------------|-------------------|-----------|
| `r_outer_left`   | (2, -2, 4)  | (0.3, 16, 8) | x=2,  y∈[-10, 6] |
| `r_outer_right`  | (10, 1, 4)  | (0.3, 18, 8) | x=10, y∈[-8, 10] |
| `r_outer_top`    | (5, 10, 4)  | (10, 0.3, 8) | y=10, x∈[0, 10] |
| `r_outer_bottom` | (5, -10, 4) | (10, 0.3, 8) | y=-10, x∈[0, 10] |
| `inner_r_h1`     | (4.5, 6, 4) | (5, 0.3, 8)  | y=6,  x∈[2, 7] |
| `inner_r_h2`     | (7.5, 2, 4) | (5, 0.3, 8)  | y=2,  x∈[5, 10] |
| `inner_r_h3`     | (4.5, -2, 4)| (5, 0.3, 8)  | y=-2, x∈[2, 7] |
| `inner_r_h4`     | (7.5, -6, 4)| (5, 0.3, 8)  | y=-6, x∈[5, 10] |

→ 내부 수평 막대 4개가 좌·우 교대로 물려 **ㄹ자(지그재그) 통로**를 형성.

**−X S 코스 (Course L 영역, x<0)**

| 모델명 | 중심 (x, y, z) | size (dx, dy, dz) | 점유 구간 |
|--------|----------------|-------------------|-----------|
| `s_outer_left`   | (-10, 1, 4)  | (0.3, 18, 8) | x=-10, y∈[-8, 10] |
| `s_outer_right`  | (-2, -2, 4)  | (0.3, 16, 8) | x=-2,  y∈[-10, 6] |
| `s_outer_top`    | (-5, 10, 4)  | (10, 0.3, 8) | y=10,  x∈[-10, 0] |
| `s_outer_bottom` | (-5, -10, 4) | (10, 0.3, 8) | y=-10, x∈[-10, 0] |
| `inner_s_v1`     | (-4, 0, 4)   | (0.3, 8, 8)  | x=-4, y∈[-4, 4] |
| `inner_s_v2`     | (-8, 0, 4)   | (0.3, 8, 8)  | x=-8, y∈[-4, 4] |
| `inner_s_v3`     | (-6, 6, 4)   | (0.3, 8, 8)  | x=-6, y∈[2, 10] |
| `inner_s_v4`     | (-6, -6, 4)  | (0.3, 8, 8)  | x=-6, y∈[-10, -2] |

→ 내부 수직 막대 4개가 **S자 통로**를 형성. 중앙 x∈(-2, 2) 부근은 스폰/호밍용 빈 통로.

얇은 쪽 두께는 **0.3 m**. `start_marker` / `goal_R_marker` / `goal_L_marker`는 z=0.1 평면 데칼(비행 무관, 시각 표시용).

### 0.2 `military_airfield.world` (50×50, 군 부대)

50×50 `compound_pad`(z=0.04 바닥) 위에 높이 8 m 구조물이 배치된다. 주요 건물:

| 모델명 | 중심 (x, y) | size (dx, dy) | 비고 |
|--------|-------------|---------------|------|
| `hangar`         | (15, 12)  | 12 × 8  | 북동 격납고 |
| `admin_building` | (-15, 12) | 14 × 6  | 북서 행정동 |
| `barracks`       | (-18, 21) | 10 × 5  | 북서 막사 |

그 외 중앙 `bund_wall`(ㅁ자 방벽 클러스터), `cen_obs_*`/`bunker_*`/`jersey_*`(콘크리트 방벽), `vehicle_*`/`supply_pallet_*`, `comms_tower`/`watchtower` 등 다수의 다중링크 장애물이 외곽·중앙에 분산 배치된다. 정확한 pose는 월드 파일(`military_airfield.world`) 참조. 좌표 변경 시 `start/goal` 시각 마커 pose 도 함께 갱신 필요(주석 참조).

### 0.3 스폰·호밍·목표 (`goals.yaml` + `course_goal_publisher`)

설정 파일: `px4_nav_planner/config/goals.yaml` (현재 값 = `obstacle_course.world` 기준).

| 항목 | 값 (ENU, 스폰 기준) | 의미 |
|------|---------------------|------|
| `spawn_enu` | (0.0, 0.0, 0.83) | odom 원점 = Gazebo 스폰 (`sitl_run.sh`) |
| `home_xyz`  | (0.0, 8.0, 4.0)  | **호밍 목표** — 위쪽 중앙 통로(0, 8)로 먼저 이동 |
| `goal_R`    | (8.0, -8.2, 4.0) | **Course R** 최종 목표 — +X ㄹ 코스 하단 |
| `goal_L`    | (-8.0, -8.2, 4.0)| **Course L** 최종 목표 — −X S 코스 하단 |
| `default_course` | "R" | 활성 코스 선택 |

월드별 레퍼런스(순항 z=4.0):
- `obstacle_course.world` (20×20): home (0, 8), goal_R (8, -8.2), goal_L (-8, -8.2)
- `military_airfield.world` (50×50): home (0, 0), goal_R (20, 20), goal_L (-18, -20)

**`course_goal_publisher_node`** 동작 요약:
1. 활성화 시 `home_xyz`로 호밍(위치 setpoint). 퍼블리시 pose는 `(home_xyz − spawn_enu)` (`odom` 프레임).
2. 호밍 완료 후 `/planner/goal` 에 목표를 한 번 올림. pose는 `(goal_{R|L} − spawn_enu)`. YAML 좌표는 월드 스케일, 노드에서 스폰 오프셋을 뺀 값이 odom 메시지에 실린다.

---

## 1. System Architecture (논문 Fig. 1 매핑)

`rl_offboard_node`의 입력은 `/mavros/local_position/odom`(자기 위치), `/scan`(2D LiDAR), `/local_grid`(OccupancyGrid), `/planner/goal`(최종 목표) 네 가지이며, 출력은 `/mavros/setpoint_velocity/cmd_vel_unstamped`(Twist) 한 가지다. 20 Hz 제어 루프(`_tick`)와 별도 스레드의 주기적 재계획(`_tick_replan`)이 돈다.

| 논문 모듈 | 구현 | 설명 |
|-----------|------|------|
| **Perception(인지)** | `grid_fuser` + `_build_obs` | 360° LiDAR → 64채널 다운샘플(min) + odom → 2D OccupancyGrid(`/local_grid`). 그리드 261×261 @0.2 m(±26.1 m)로 두 월드 모두 커버. |
| **Decision(판단)** | A*(`_astar_with_inflation`) + PPO 정책 | 그리드에서 로봇 셀→목표 셀 inflation A* → LOS shortcut → B-spline 스무딩 → `/planner/path`. PPO는 경로상 lookahead 경유점을 추종하며 (v, ω) 산출. |
| **Control(제어)** | `_publish` → MAVROS → PX4 | heading 방향 ENU 속도 + yaw_rate 를 `cmd_vel`로 발행, PX4 속도 제어기로 전달. OFFBOARD/ARM 자동 전환. |

전역 계층은 좁은 통로에서 `astar_min_inflation_cells`(=1)를 0까지 자동 감소시켜 막힘을 회피하고, start/goal 셀은 force-clear 한다. 경로 추종 lookahead는 **전 모드 거리 기반**(`_path_lookahead_wp_at_dist`, `hybrid_lookahead_dist_m`=1.0)으로 통일되어 있다.

---

## 2. nav_mode 3종 (논문 비교 실험 대상)

세 모드는 동일한 Perception(그리드/관측)을 공유하고 **조향 결정 출처만 다르다**. 모델은 모드별로 분리된다.

| 모드 | 조향 출처 | obs 의 goal_rel | 모델 파일 | 비고 |
|------|-----------|-----------------|-----------|------|
| `astar` | A* 경로 기하 P제어(`_tick_global`) | — (PPO 미사용) | 없음 | yaw_rate = clamp(2·yaw_err, `planner_max_yaw_rate`=1.0), 오차 클수록 `align_scale`로 감속 |
| `ppo`   | PPO 단독 추론(`_tick_ppo_mode`) | **최종 goal**, `world_size`(20) 정규화 | `policy_ppo_ts.pt` / `obs_norm_ppo.json` | 매 tick `_build_obs→_infer→_action_to_vel→publish`, 출력 가공 없음 |
| `hybrid`| PPO 추론(`_tick_hybrid_mode`) | **A* lookahead 경유점**, `LOOKAHEAD_NORM_SCALE`(3.0) 정규화 | `policy_hybrid_ts.pt` / `obs_norm_hybrid.json` | A* 전역 경로를 obs에 주입 → PPO가 추종+회피. yaw 는 PPO 출력 사용 |

세 모드 모두 yaw 회전 상한은 1.0 rad/s 로 통일(astar=`planner_max_yaw_rate`, ppo/hybrid=학습값 `max_yaw_rate`). `policy_path`/`obs_norm_path` 가 비면 nav_mode별 default 모델을 자동 선택한다.

---

## 3. Hybrid Model — MDP (논문 2.2)

| 요소 | 정의 |
|------|------|
| **State** | LiDAR 64빔(정규화) + goal_rel 2 (body frame) = `obs_dim` **66**. goal_rel 은 odom·A* 경유점으로부터 산출 → PPO 부분관측 한계 완화. |
| **Action** | 연속 2D: 선속도 v ∈ [0.3, 0.8] m/s, 요 각속도 ω ∈ [-1.0, 1.0] rad/s |
| **Reward** | 기존 연구[2] 기반, 항법과 무관한 배터리 항 제외, **BFS 기반 진행 보상** 도입. 현재 env 활성 항 = `w_progress`(BFS, 50.0) + `success_reward`(800.0) + `collision_penalty`(500.0) |

학습 환경(`rl/env_2d_nav.py`): 20×20 m 2D, dt=0.05 s, n_envs 병렬, 5단계 커리큘럼(장애물 점증). PPO 단독은 `rl/env_2d_nav_ppo_only.py`(A* lookahead 없음, goal_rel=최종 goal). 추론 시점(obs↔action)과 배포 20 Hz 모두 동일 시간 스케일로, 별도 frame-skip 없이 매 tick 추론한다.

---

## 4. 안전/보조 계층 (현재 코드 기준)

PPO/A* 출력 위에 얹히는 보조 로직의 **현재 활성 상태**:

| 메커니즘 | 상태 | 비고 |
|----------|------|------|
| Altitude hold | **활성** | `_hold_z` 기준 z 유지, vz = gain·err (±0.5 클립) |
| Emergency brake | **비활성** | `emergency_brake_dist=0.0` → 트리거 안 됨. grid 회피로 대체. (켜면 거리 기반 lookahead로 회전 방향 결정) |
| OFFBOARD/ARM 자동, pre-setpoint | 활성 | 노드 내부에서 자동 처리 |

**제거된 것(이전 버전 대비):** map boundary clamp(`enable_map_bounds` 등), `planner_slowdown_dist`, ppo 휴리스틱 `_escape_if_spinning`·`goal_bias_yaw_scale`, astar `heading_align_tol` 데드존(비례제어로 통합), `ppo_frame_skip`(매 tick 추론으로 복귀). 모드 간 confound 제거 및 순수 비교를 위해 정리됨.

---

## 5. 학습·평가 (논문 3장 요약)

- **2D 학습/평가**: 채택 보상 구조를 hybrid·ppo 동일 적용, 20×20 m 2D, 7개 시나리오(장애물 밀도 0/5/10/15/20 + 좁은 통로 + S자), 10M step. 1M마다 평균 성공률(Weighted Score)·고난이도(15/20) 충돌률 평가. hybrid 가 전 구간 높은 성공률·빠른 수렴(특히 1M~4M)·낮은 충돌률로 수렴.
- **3D Gazebo 검증**: 사전 지도 정보 없이, A* 경로 주기적 재계획. obstacle/military 맵 × R/L 목표 × 8회 = 총 32회. 결과(총합): **Hybrid 81.3%(26/32), 평균 74.2 s** > A* 46.9%(15/32) > PPO 25%(8/32).

> 배포 노드의 재계획 주기는 `replan_period`(**1.0 s**, 논문 실험과 동일)이다.
