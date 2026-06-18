# Changelog
코드/파라미터 수정 사항을 이 파일에 기록합니다.

---

## 2026-06-16: `[적용]` `rl_offboard_node` — ppo 모드 frame skip 제거 (매 tick 추론 복귀)

> ppo 모드 추론 주기를 PX4 발행 주기(20Hz)와 분리하던 frame skip(action repeat)을 완전 제거. `ppo_frame_skip` 파라미터(params.yaml + declare_parameter), 상태 변수 `_ppo_skip_counter`·`_cached_ppo_action`, `_tick_ppo_mode`의 skip/cache 분기를 모두 삭제. 이제 ppo 모드는 hybrid와 동일하게 **매 tick** `_build_obs → _infer → _action_to_vel → publish`로 추론한다. (2026-05-11 frame skip 도입 항목을 되돌림.) `grep` 결과 코드/params에서 `ppo_frame_skip`·`_ppo_skip_counter`·`_cached_ppo_action` 0건. `python3 -m py_compile` 통과.

---

## 2026-06-16: `params.yaml` — `planner_max_yaw_rate: 1.5 → 1.0` (모드 간 회전 상한 통일)

> astar 전용 회전 상한을 ppo/hybrid의 `max_yaw_rate=1.0`과 동일하게 맞춤. 2026-05-06에 기하 yaw용으로 1.0→1.5 상향했었으나, 현재 hybrid는 PPO yaw_rate를 직접 발행(1.0 clamp)하므로 `planner_max_yaw_rate`는 astar 전용. astar만 1.5로 더 큰 회전 권한을 갖는 불일치를 제거해 astar/ppo/hybrid 비교를 동일 조건으로 통일.

---

## 2026-06-16: `[적용]` `rl_offboard_node` — lookahead 거리 기반 통일 + step 기반 코드 삭제

> **변경:** 모든 경로 추종을 거리 기반(`_path_lookahead_wp_at_dist`, `hybrid_lookahead_dist_m`)으로 통일. 유일하게 step 기반(`_path_lookahead_wp`)을 쓰던 비상제동 회전 방향 결정도 거리 기반으로 전환하고, step 기반 잔재를 완전 삭제.
>
> **삭제/변경 상세:**
> - `rl_offboard_node.py`:
>   - `_path_lookahead_wp()` 함수(step 기반) 삭제.
>   - 비상제동 블록: `wp_la = self._path_lookahead_wp()` → `self._path_lookahead_wp_at_dist(px, py, self.hybrid_lookahead_dist_m)`.
>   - `declare_parameter`에서 `path_lookahead_steps` 제거.
> - `params.yaml`: `path_lookahead_steps` 줄 제거. `hybrid_lookahead_dist_m` 주석을 "astar/hybrid wp 추종 + 비상제동 회전방향" 공용으로 갱신.
>
> **결과:** astar/hybrid wp 추종과 비상제동이 모두 동일한 거리 기반 lookahead를 공유. step 기반 코드·파라미터 0건. (`emergency_brake_dist=0.0`으로 비상제동 자체는 여전히 비활성이나, 부활 시에도 거리 기반으로 동작.) `python3 -m py_compile` 통과.

---

## 2026-06-16: `[적용]` `rl_offboard_node` — 모드별 deployment 휴리스틱 3종 제거 (heading_align_tol / _escape_if_spinning / goal_bias)

> **변경:** astar의 `heading_align_tol` 정렬 데드존, ppo의 `_escape_if_spinning`(스핀 탈출)·`goal_bias_yaw_scale`(개활 구간 yaw 감쇠)를 코드·파라미터에서 완전 삭제. PPO 출력을 가공 없이 그대로 발행하고, astar는 단순 비례 제어로 통일.
>
> **삭제 상세:**
> - `params.yaml` + `declare_parameter`: `heading_align_tol`, `goal_bias_dist`, `goal_bias_yaw_scale`, `lidar_sector_agg` 제거.
> - `rl_offboard_node.py`:
>   - `_escape_if_spinning` 함수 + `_yaw_sat_count` 상태 변수 삭제.
>   - `_tick_ppo_mode`: escape·goal_bias 게이트 삭제 → `_build_obs → _infer → _action_to_vel → publish` 직선 파이프라인. 시그니처 `_tick_ppo_mode(yaw)`로 단순화.
>   - `_tick_global`(astar): `heading_align_tol` 데드존 분기 제거 → yaw 오차 비례 회전 + `align_scale` 감속을 **항상** 적용 (정렬 시 yaw_rate≈0, align_scale≈1로 자연 수렴).
>   - dead-by-cascade 정리: ppo가 `d_min_sectors`를 더 이상 쓰지 않으므로 `_tick`의 섹터 거리 계산(`agg` / `d_front` / `d_front_left` / `d_front_right` / `d_min_sectors`)과 입력 파라미터 `lidar_sector_agg` 삭제. (`_dist_in_sector`는 비상제동에서 계속 사용.)
>
> **모드별 결과:**
> - **astar**: 비례 P제어만 (정렬 데드존 없음).
> - **ppo**: PPO 추론 + frame skip만. 출력 후처리 게이트 0개.
> - **hybrid**: 변경 없음 (원래 이 세 휴리스틱 미사용).
>
> **참고:** 본 변경은 2026-05-10 "ppo 모드 휴리스틱 원복" 항목을 다시 되돌려(re-remove) 순수 평가 형태로 정리한 것. `grep -nE 'heading_align_tol|_escape_if_spinning|_yaw_sat_count|goal_bias_dist|goal_bias_yaw_scale|lidar_sector_agg|d_min_sectors'` 결과 코드/params에서 0건(CHANGELOG 이력 제외). `python3 -m py_compile` 통과.

---

## 2026-05-10: `rl_offboard_node` — ppo 모드 휴리스틱 원복 (Step 5 + dead-by-cascade 되돌림)

> **배경:** 직전 "3-mode ablation 청결화"의 Step 5에서 ppo 모드의 deployment-time 휴리스틱(`_escape_if_spinning`, `goal_bias_yaw_scale`)을 "순수 PPO 평가" 명분으로 삭제했으나, SITL 검증 결과 ppo 모드가 학습 분포 외 동작(zigzag, 회피 실패)으로 사실상 사용 불가 수준 약화 확인. 두 휴리스틱은 confound가 아니라 PPO 정책의 실제 deployment-time 안정화 레이어로, **PPO yaw_rate에만 작용**하므로 hybrid·astar의 yaw 산출에 영향 없음 → ablation 비교에서 ppo 전용 요소로 분류 가능.
>
> **원복 범위:**
> - `_escape_if_spinning` 함수 + `_yaw_sat_count` 상태 변수 + `_tick_ppo_mode`에서의 호출.
> - `goal_bias_dist`, `goal_bias_yaw_scale` 파라미터 (params.yaml + declare_parameter) + `_tick_ppo_mode`에서의 `if d_min_sectors > goal_bias_dist: yaw_rate *= goal_bias_yaw_scale` 게이트.
> - dead-by-cascade로 함께 삭제됐던 chain 복구: `lidar_sector_agg` 파라미터, `_tick`의 `agg` / `d_front` / `d_front_left` / `d_front_right` / `d_min_sectors` 계산 (ppo 모드의 goal_bias 게이트에 필요).
> - `_tick_ppo_mode(yaw, d_min_sectors)` 시그니처 복구.
>
> **건드리지 않은 것 (원복하지 않음):**
> - Step 1-4 (lookahead 통일, backward 차단, `planner_slowdown_dist` 삭제, hybrid `heading_align_tol` 추가): astar/hybrid에만 영향, ppo 약화와 무관.
> - Step 6 (boundary 메커니즘 정리): `enable_map_bounds=false`로 이미 dead 상태였던 코드 정리. 행동 변화 0.
> - `_tick_hybrid_mode(px, py, yaw)` 시그니처: hybrid는 `d_min_sectors`를 안 씀.
>
> **상태:** ppo 모드는 직전 "3-mode 청결화" 적용 전 동작으로 복귀. astar/hybrid는 청결화 후 상태 유지. 노드 재시작 후 SITL에서 ppo 모드 zigzag/회피 실패 양상 해소 기대.

---

## 2026-05-10: `goals.yaml` — `obstacle_course_light.world` 좌표로 갱신

> military_airfield 좌표(home `(0,0)`, goal_R `(20,20)`, goal_L `(-18,-20)`) → obstacle_course_light(20×20, ㄹ/S 코스) 좌표로 변경. 옵션 A(스폰 → 북쪽 home 이동 → 남쪽 goal 통과) 패턴 채택.
>
> | 항목 | 좌표 | 검증 |
> |---|---|---|
> | `home_xyz` | `(0, 8, 2.5)` | y=10 외벽과 inner_r_h1(y=6) 사이 corridor |
> | `goal_R` | `(8, -8.2, 2.5)` | inner_r_h4(y=-6) 와 r_outer_bottom(y=-10) 사이, r_outer_right(x=10) 서쪽 |
> | `goal_L` | `(-8, -8.2, 2.5)` | 대칭 위치, inner_s_v4(x=-5) 서쪽, s_outer_left(x=-10) 동쪽 |
>
> 좌표는 2026-04-14 entry에서 0.6m 두께 벽과 AABB 겹침 없음으로 검증된 값 그대로. 코드 변경 없음 (yaml만).

---

## 2026-05-10: `[적용]` `rl_offboard_node` — 3-mode ablation 청결화 (lookahead 통일 + 비대칭/dead code 정리)

> **목적:** astar / ppo / hybrid 3-모드 항법 성능 비교 ablation의 confound 제거. 원칙: **astar 로직 + ppo 로직 = hybrid 로직** (단, `obs_norm_*.json` / `policy_*.pt`는 모드별 분리). 현재는 ① lookahead 방식, ② hybrid 전용 추가 감속, ③ 한쪽에만 있는 안전 게이트, ④ ppo 전용 휴리스틱, ⑤ `enable_map_bounds=false`로 사실상 dead인 boundary 코드가 confound로 작용.
>
> **현재 사실 (코드 확인):**
> - **Lookahead 비대칭**: astar `_path_lookahead_wp()` step 기반(`path_lookahead_steps=3`), hybrid `_path_lookahead_wp_at_dist(px, py, 1.0)` dist 기반(`hybrid_lookahead_dist_m=1.0`). + `_path_lookahead_wp_at_dist`에 **backward 시작 세그먼트 버그** (drone이 path[start_idx]를 진행 방향으로 이미 통과 + 둘 사이 거리 ≥ `lookahead_dist`이면 lookahead 점이 drone 뒤쪽 → 180° 회전. military_airfield 2-wp 경로에서 재현).
> - **모드별 비대칭 메커니즘**:
>   - hybrid 전용: `planner_slowdown_dist` ([rl_offboard_node.py#L1450-L1453](rl_offboard_node.py#L1450-L1453)) — obstacle 근접 시 v_cmd 추가 곱셈 감속.
>   - astar 전용: `heading_align_tol` 데드존 게이트 ([line 1247-1261](rl_offboard_node.py#L1247-L1261)) — `|yaw_err|≤15°`일 때 yaw_rate=0 + v_cmd full.
>   - ppo 전용: `_escape_if_spinning` ([line 513-522](rl_offboard_node.py#L513-L522)) yaw_rate 포화 20틱 시 후진 탈출, `goal_bias_yaw_scale` ([line 1401-1402](rl_offboard_node.py#L1401-L1402)) 무장애 공간에서 PPO yaw 0.25× 약화.
> - **Dead code (`enable_map_bounds=false` 게이트로 무력화)**: `_is_heading_outward_near_boundary` + `_compute_boundary_turn_yaw_rate` (ppo/hybrid에서 호출되나 항상 False), `_clamp_bounds` ([line 612-636](rl_offboard_node.py#L612-L636), 모든 모드 _tick 끝에서 호출되나 즉시 return). 관련 파라미터 6개 (`enable_map_bounds`, `map_x_min/max`, `map_y_min/max`, `map_boundary_margin`, `boundary_turn_clearance`).
>
> **가정:**
> 1. PPO 학습 분포 — `_build_obs`의 `goal_rel`은 backward 차단 후에도 normal 케이스 출력 변함 없음. 재학습 불필요.
> 2. ppo 휴리스틱(escape, goal_bias) 삭제 시 노출되는 정책 약점(spin loop, 무장애 zigzag)은 평가에서 보고할 결과로 간주 — 가리지 않음.
> 3. boundary 메커니즘은 `enable_map_bounds=false`로 이미 dead — 삭제는 행동 변화 0.
> 4. astar의 effective lookahead 거리가 ~1.8m(step=3, 50점 경로) → 1.0m(dist)로 0.8m 가까워짐. 직선 구간 오버슛·오실레이션 가능성. (corner 추종은 dist가 step보다 강함 — sparse-path skip 문제 해소.) 회귀 시 `hybrid_lookahead_dist_m`으로 흡수.
> 5. `heading_align_tol` 게이트는 astar/hybrid 공유 — hybrid에서도 정렬 후 PPO v_cmd가 align_scale에 깎이지 않고 full로 발행되어 직진 성능 보완.
>
> **변경 계획 (7 단계, 각 단계별 검증 후 다음 단계 진행):**
>
> 1. **`_path_lookahead_wp_at_dist` backward 시작 세그먼트 차단**
>    - 벡터 `(path[start_idx] − drone)`(드론→start_idx)와 path 진행방향 `(path[start_idx+1] − path[start_idx])`의 dot product가 **음수**(두 벡터 사이각 > 90°)면 drone이 path[start_idx]를 진행방향으로 통과한 것 → `start_idx += 1` 후 반복. 인덱스 끝나면 fallback.
>    - **검증:** 단위 테스트. drone=(1.5, 0) yaw=0, path=[(0,0), (5,0)], `lookahead_dist=1.0` → 반환점 x좌표 > 1.5 (앞쪽). 현재 동작은 ≈0.5 (후방).
>
> 2. **`_tick_global`이 dist 기반 lookahead를 사용하도록 교체**
>    - `wp = self._path_lookahead_wp()` → `wp = self._path_lookahead_wp_at_dist(px, py, self.hybrid_lookahead_dist_m)`. 파라미터는 `hybrid_lookahead_dist_m` 두 모드 공유 (rename 없음).
>    - **검증:** obstacle_course 1x astar 모드 SITL 30초 비행 — goal 도달 성공 + 충돌 0회.
>
> 3. **`planner_slowdown_dist` 메커니즘 코드에서 완전 삭제 (hybrid 비대칭 감속 제거)**
>    - 삭제: (a) `params.yaml`의 줄, (b) `declare_parameter` 항목, (c) `_tick_hybrid_mode` slowdown 블록 ([line 1450-1453](rl_offboard_node.py#L1450-L1453), `have_path_slow` lock 포함). astar에 동일 로직 추가하지 **않음** — "두 모드 모두 슬로우다운 없음"으로 통일.
>    - 안전성: A* inflation으로 경로-벽 여유 0.4m+ 확보 + PPO lidar 학습 + `emergency_brake_dist=0.0`도 이미 비활성. astar의 현재 default와 동일 수준.
>    - **검증:** military_airfield + obstacle_course 1x hybrid 30초 비행 충돌 0회.
>
> 4. **`heading_align_tol` 게이트를 hybrid에 추가 (정렬 데드존 통일, 직진 성능 보완)**
>    - `_tick_hybrid_mode`의 yaw_rate / align_scale 계산을 astar 게이트 구조와 동일하게 변경:
>      - `|yaw_err| > tol`: yaw_rate = clamp(2·err, planner_max_yaw_rate), v_cmd × align_scale (현재 동작과 동일)
>      - `|yaw_err| ≤ tol`: **yaw_rate = 0, v_cmd 그대로 (PPO 출력 full, align_scale 미적용)**
>    - 효과: 정렬된 직선 구간에서 PPO v_cmd가 align_scale로 깎이지 않고 그대로 발행 → 직진 성능 보완. yaw jitter 제거. astar/hybrid가 정렬 후 동일 동작.
>    - **검증:** military_airfield 직선 구간 30초 비행에서 hybrid 평균 v_cmd가 변경 전 대비 ≥10% 상승 (rosbag 또는 DEBUG 로그 분석).
>
> 5. **ppo 모드 inference-time 휴리스틱 2개 삭제 (순수 PPO 평가)**
>    - 삭제 대상:
>      - `_escape_if_spinning` 함수 ([line 513-522](rl_offboard_node.py#L513-L522)) + 호출 ([line 1397-1399](rl_offboard_node.py#L1397-L1399)) + 상태 변수 `_yaw_sat_count` (선언/리셋 포함).
>      - `goal_bias_yaw_scale` 곱셈 ([line 1401-1402](rl_offboard_node.py#L1401-L1402)) + 파라미터 `goal_bias_dist`, `goal_bias_yaw_scale` (params.yaml + declare_parameter).
>    - 삭제 후 ppo 모드는 `_build_obs → _infer → _action_to_vel → _compute_cmd_vel → publish` 직선 파이프라인.
>    - **검증:** 삭제 전후 ppo 모드 SITL 30초 비행 비교. spin loop / 무장애 zigzag 등 정책 약점 노출 시 ablation 결과의 일부로 보고.
>
> 6. **Boundary 메커니즘 완전 정리 (dead code 제거)**
>    - 삭제 대상:
>      - `_is_heading_outward_near_boundary` 함수 ([line 530-544](rl_offboard_node.py#L530-L544))
>      - `_compute_boundary_turn_yaw_rate` 함수 ([line 546-553](rl_offboard_node.py#L546-L553))
>      - `_clamp_bounds` 함수 ([line 612-636](rl_offboard_node.py#L612-L636)) + 호출 ([line 1381](rl_offboard_node.py#L1381))
>      - `_tick_ppo_mode` boundary if-블록 ([line 1403-1406](rl_offboard_node.py#L1403-L1406))
>      - `_tick_hybrid_mode` boundary if-블록 ([line 1446-1449](rl_offboard_node.py#L1446-L1449))
>      - 파라미터 7개: `enable_map_bounds`, `map_x_min`, `map_x_max`, `map_y_min`, `map_y_max`, `map_boundary_margin`, `boundary_turn_clearance` (params.yaml + declare_parameter).
>    - **검증:** 행동 변화 0 (이미 dead). 모든 SITL 시나리오 변경 전과 동일 동작 확인.
>
> 7. **military_airfield hybrid backward 회전 재발 안 함을 직접 확인 (전체 통합 검증)**
>    - `start_px4` + goal_R(20, 20) + nav_mode=hybrid, 30초 비행.
>    - **검증:** `yaw_rate`가 `±planner_max_yaw_rate`에 포화된 횟수 0회. astar/ppo 모드도 같은 시나리오 30초 비행으로 회귀 없음 확인.
>
> **성공 기준 (전체 작업 종료 조건):**
> - ✅ astar/hybrid가 동일 lookahead 함수·파라미터로 wp 산출. 두 `_tick_*`의 lookahead 호출 라인 동일.
> - ✅ astar와 hybrid의 핵심 차이가 **v_cmd 출처 한 줄**로 환원 (geometric `min(v_max, 0.8·gnorm)` vs PPO `action[0]`).
> - ✅ ppo 모드의 핵심 동작이 **PPO 추론 → action_to_vel → publish** 직선 흐름 (휴리스틱 0개).
> - ✅ `grep -nE 'planner_slowdown_dist\|_escape_if_spinning\|_yaw_sat_count\|goal_bias_yaw_scale\|goal_bias_dist\|enable_map_bounds\|_clamp_bounds\|_is_heading_outward_near_boundary\|_compute_boundary_turn_yaw_rate\|map_x_min\|map_x_max\|map_y_min\|map_y_max\|map_boundary_margin\|boundary_turn_clearance'` 결과 0건 (전체 패키지).
> - ✅ military_airfield(goal_R) hybrid 모드 180° 후방 회전 이벤트 0회.
> - ✅ obstacle_course 1x astar 모드 baseline 회귀 없음 (goal 도달 성공률 동등 이상).
> - ✅ obstacle_course 1x + military_airfield hybrid 모드 충돌 0회.
>
> **건드리지 않는 것 (surgical):**
> - `_path_lookahead_wp()` 함수 본체와 `path_lookahead_steps` 파라미터 — emergency brake (현재 `emergency_brake_dist=0.0`로 비활성) 부활 시 재사용 여지. 이번 cleanup의 범위 밖.
> - `params.yaml` rename — `hybrid_lookahead_dist_m` 두 모드 공유 형태로 우선 진행. 이름이 hybrid에 묶여있어 어색하나 변경 범위 최소화 우선.
> - PPO 학습/모델 (`obs_norm_*.json`, `policy_*.pt`) — 이번 변경 무관.
> - emergency brake 메커니즘 자체 — 이번 cleanup 범위 밖 (별도 PR).
>
> **트레이드오프 / 푸시백 메모:**
> - dist 기반(1.0m)은 step 기반(50점 경로 기준 ~1.8m)보다 lookahead가 0.8m 가까움 → 직선 구간 wp 추종은 더 정확해지지만 오버슛·오실레이션 여지 ↑. 회귀 시 `hybrid_lookahead_dist_m: 1.0 → 1.5~1.8`로 흡수.
> - sparse 경로(shortcut으로 3-wp 이하)에서는 dist가 step보다 strictly 우수 — astar 모드의 sparse-path corner 추종을 **개선하는 부수 효과**.
> - ppo 휴리스틱 삭제로 ppo 모드 평가에서 spin loop·무장애 zigzag가 관찰될 수 있음. 정책의 실제 특성으로 ablation에서 보고할 결과.
> - hybrid에 `heading_align_tol` 추가 시 평균 속도 미세 상승 (정렬 데드존에서 align_scale 미적용). ablation에서 PPO 기여 측정에 유리한 방향.
>
> **적용 결과 (2026-05-10):**
> - Step 1-6 코드 적용 완료. `grep -nE 'planner_slowdown_dist|_escape_if_spinning|_yaw_sat_count|goal_bias_yaw_scale|goal_bias_dist|enable_map_bounds|_clamp_bounds|_is_heading_outward_near_boundary|_compute_boundary_turn_yaw_rate|map_x_min|map_x_max|map_y_min|map_y_max|map_boundary_margin|map_boundary_gain|boundary_turn_clearance'` 결과 0건 확인.
> - 계획 외 dead-by-cascade 추가 정리: boundary 제거 후 `_tick_ppo_mode` / `_tick_hybrid_mode`에서 `d_min_sectors`가 미사용이 되어, 그 계산 chain(`agg`, `d_front`, `d_front_left`, `d_front_right`, `d_min_sectors`)과 그 유일한 입력 파라미터 `lidar_sector_agg`까지 함께 삭제. `_tick_ppo_mode(yaw)`, `_tick_hybrid_mode(px, py, yaw)`로 시그니처 단순화.
> - **Step 7 (SITL 통합 검증):** ✅ hybrid 모드 — 180° 후방 회전 이벤트 0회, 목표 지점까지 안정적 비행 성공. backward 시작 세그먼트 차단(Step 1) + lookahead 통일(Step 2) + slowdown 제거(Step 3) + heading_align_tol 게이트 추가(Step 4)의 조합이 의도대로 동작 확인.
> - 후속 검증 (선택): obstacle_course 1x astar baseline 회귀 / ppo 모드 spin loop·zigzag 노출 여부는 별도 ablation 실행에서 측정 예정.

---

## 2026-05-08: `grid_fuser.py` — Ray casting `t_vals` 0.95→1.0 (endpoint 직전 cell 누락 버그 수정)

> `process_scan()` ray casting의 `t_vals = np.linspace(0.0, 0.95, ...)`이 ray의 끝 5% 구간을 sample 안 함 → endpoint 직전 1-2 cells (10m ray 기준 cell 48-49)가 l_miss 받지 못함 → 한번 occupied 마킹된 cell이 영원히 clear 안 되는 버그.
>
> **수정:** `np.linspace(0.0, 1.0, max_steps, ...)`로 변경. cx_all 마지막 step이 endpoint cell과 정확히 일치 → `is_endpoint` mask로 자동 제외 (l_miss 적용 안 됨). endpoint cell의 l_hit은 별도로 처리되므로 동작 동일하나, **endpoint 직전 cells가 정상적으로 l_miss 받음** → 멀리 있는 occupied cell이 ray 통과 시 정상 clear.
>
> 영향: occupied↔free 전환 정상화. clamp_max(3.0) 도달 cell도 ~3-4 ray 통과 시 free 전환됨.

---

## 2026-05-08: `params.yaml` — `path_lookahead_steps: 12→3` (planner_only wp 추종 개선)

> planner_only 모드에서 A* waypoint 미추종 → shortcut 발생 문제 수정. step 기반 lookahead 거리를 ~7m(50점 경로, 12 step) → ~1.8m(3 step)으로 단축하여 hybrid의 1.0m 거리 기반 lookahead와 유사 수준으로 맞춤.
>
> **모드별 영향:**
> - **planner_only (astar)**: `_tick_global`이 `_path_lookahead_wp()` 사용 → **직접 영향**, wp 추종 정확도 개선
> - **hybrid (v2)**: `_path_lookahead_wp_at_dist()` (거리 기반) 사용 → **영향 없음**
> - **ppo only**: 미사용
> - emergency brake 회전 방향 결정에도 사용되나 `emergency_brake_dist=0.0`로 현재 비활성

---

## 2026-05-08: `grid_fuser.yaml` — 격자 ±26.1m로 확장 (military_airfield 커버)

> `width/height: 221→261`, `origin: ±22.1→±26.1`. 261×0.2=52.2m로 50×50 world 전체 커버.
> 메모리 195KB→272KB (+77KB, 무시 가능). obstacle_course 1x/2x도 동일 설정으로 호환.

---

## 2026-05-08: `military_airfield.world` — 레이아웃 재구성 + goal 좌표 적용 (옵션 B)

> **레이아웃 재구성 완료:**
> - **Center (4 건물)**: hangar(10, 8), admin(-10, 8), barracks(-10, -8), control_tower(10, -8)
> - **Inner south (oil 클러스터)**: 4 탱크 (±2, -12)·(±2, -16) + bund_wall(0, -14)
> - **Inner**: bunker_N(0, 14), bunker_S(0, -7)
> - **Edge perimeter**: 모든 obstacle (vehicles, supply_pallets, cen_obs 6개, comm_poles, light_poles, containers, fuel_tank_NE, steel_tower_W, comms_tower 등) 외곽 ±22 영역으로 이동
> - 추가된 4m+ 구조물 10개 (이전 변경): comm_pole_E/W, container_stack_N/SE/N2, light_pole_SW/NW, fuel_tank_NE, steel_tower_W, control_tower
>
> **추천 navigation 좌표 (옵션 B, 비대칭 대각, 코드 미적용):**
>
> ```yaml
> spawn_enu:  [0.0, 0.0, 0.83]
> home_xyz:   [0.0, 0.0, 2.5]        # 스폰 위에서 takeoff
> goal_R:     [20.0, 20.0, 2.5]      # helipad NE, 직선거리 28.3m
> goal_L:     [-18.0, -20.0, 2.5]    # SW, 직선거리 27m
> ```
>
> **path 난이도:**
> - **R (NE helipad)**: hangar(10, 8) 우회 → helipad 영역. 중간 난이도.
> - **L (SW)**: barracks(-10, -8) + oil cluster(0, -14) + cen_obs_6(-22, -18) 3중 우회. 높은 난이도.
>
> **클리어런스 검증:**
> - (20, 20): 주변 장애물 15m+ (helipad 자체는 z=0.06 평면, 비행 무관)
> - (-18, -20): cen_obs_6(-22, -18) 4.5m, cen_obs_3(-15, -22) 3.6m
>
> 적용 시 `px4_nav_planner/config/goals.yaml` 수정 필요.

---

## 2026-05-08: `rl_offboard_node` — `ppo` 모드 default 모델 변경

> nav_mode "ppo"일 때 default 모델을 `policy_ts.pt`/`obs_norm.json` → **`policy_ppo_ts.pt`/`obs_norm_ppo.json`**로 변경. PPO 단독 모드용으로 별도 학습/배포된 모델 사용.

---

## 2026-05-07: `params.yaml` — `enable_map_bounds: false` (다양한 크기 world 지원)

> 다양한 world(`obstacle_course_light_2x` 40×40, `military_airfield` 등)에서 공통 사용 위해 map bounds 비활성. 회피는 LiDAR + occupancy grid + emergency brake에 일임. 물리 벽이 있는 world는 자체 boundary 역할.

---

## 2026-05-07: `obstacle_course_light_2x.world` — 외곽 40×40 봉합

> 외곽 boundary 갭 4곳 봉합: 동/서벽을 풀 길이(40m)로 확장 (`r_outer_right`, `s_outer_left` 각각 length 36→40, center y 2→0), 북/남 중앙 8m 갭에 새 segment 추가 (`outer_top_middle`, `outer_bottom_middle`). 드론이 40×40 영역 밖으로 나갈 수 없게 함.

---

## 2026-05-06: `rl_offboard_node` — 기하학적 yaw 모드 한도 분리 (옵션 A, hybrid 포함)

> `_publish()`의 yaw_rate clamp를 nav_mode별로 분기: `astar` 및 `hybrid`(v2 기하학적 yaw)는 `planner_max_yaw_rate`, `ppo`만 `max_yaw_rate`(PPO 학습값) 사용. `planner_max_yaw_rate: 1.0→1.5` 상향. hybrid도 `_tick_hybrid_mode`에서 기하학적 P-controller로 회전하므로 동일 한도 적용 (이전 버전은 hybrid가 max_yaw_rate=1.0으로 재clamp되는 버그 있음).

---

## 2026-05-06: `grid_fuser.yaml` — `prob_hit`/`prob_miss` 약간 hit-우위로 조정

> occupied 누적 약간 강화: `prob_hit: 0.80→0.85` (l_hit +1.386→+1.735), `prob_miss: 0.30→0.35` (l_miss −0.847→−0.619). hit/miss 비율 1.64×→2.80×. clamp/decay 등 다른 파라미터는 변경 없음.

---

## 2026-04-18: Hybrid 모드 실패 근본 원인 분석 및 Hybrid-v2 설계 (Fix 6)

### 실험 목표

**hybrid 모드가 astar-only, ppo-only 대비 성능 우위를 증명.**

각 모드의 강점:
- **astar-only**: 전역 경로(dead-end 회피). 속도는 고정 공식(`min(v_max, 0.8*gnorm)`).
- **ppo-only**: LiDAR 기반 장애물 반응 속도 제어. 전역 경로 없어 복잡한 코스 탈출 불가.
- **hybrid-v2**: A*(전역 경로) + PPO(속도 제어) 역할 분리 → 두 모드의 장점 결합.

---

### 문제: astar 모드 성공 / 기존 hybrid 모드 실패

Fix 1–5 적용 후 astar 모드는 코스 완주, hybrid 모드는 동일 코스에서
emergency brake 반복 + 경로 탐색 실패 루프로 완주 불가.

---

### 근본 원인 1 (치명): 속도 방향 결정 구조 차이

| | astar (`_tick_global`) | 기존 hybrid |
|---|---|---|
| 속도 방향 | `target_yaw` (waypoint 방향, 매 tick) | `yaw` (현재 heading, PPO yaw_rate로만 변경) |
| heading 오차 처리 | `align_scale` → 90° 이상 실질 정지 | PPO yaw_rate로 서서히 회전, 전진 계속 |
| 좁은 통로 | 즉시 waypoint 방향으로 이동 | 현재 heading(벽 방향)으로 전진 중 회전 |

waypoint가 측방(90°)에 있으면 PPO 회전 완료 전 벽 방향 전진 →
`d_front < emergency_brake_dist` → brake → 회전 멈춤 → 재진입 → 루프.

코드 근거 ([rl_offboard_node.py:1435](px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py)):
```python
vx, vy = self._compute_cmd_vel(v_cmd, yaw)   # vx = v_cmd * cos(yaw) ← 항상 현재 heading
```

---

### 근본 원인 2: `planner_slowdown_dist=0.0` → hybrid 감속 없음

astar: `v_cmd = min(v_max, 0.8 * gnorm)` — waypoint 근접 시 자연 감속.
hybrid: `planner_slowdown_dist=0.0` → 감속 블록 조건 `> 0` 불충족 → 전속 유지.

---

### 근본 원인 3 (부가): `goal_rel` 크기 불일치

`hybrid_lookahead_dist_m=1.0` + `LOOKAHEAD_NORM_SCALE=3.0` →
`goal_rel` ≈ `1.0 / 3.0 = 0.33`. 학습 분포 불일치 가능 → yaw_rate 출력 과소.

---

### Fix 6. `_tick_hybrid_mode` 전체 교체 — Hybrid-v2

**재학습 불필요.** PPO `v_cmd`(속도 제어)만 사용, `yaw_rate`는 기하학적으로 대체.

#### 역할 분리

| 역할 | 담당 | 근거 |
|---|---|---|
| 전역 경로 (어디로) | A* | dead-end 회피, 최적 우회 |
| 이동 방향 | `target_yaw` 기하학적 | LiDAR body-frame → heading=이동방향 일치 필수 |
| heading 정렬 | `planner_max_yaw_rate` 기하학적 | astar에서 이미 검증된 방식 |
| 전진 속도 | PPO `v_cmd` | LiDAR 기반 학습된 장애물 반응 속도 |

> **왜 yaw_rate를 기하학적으로 교체하는가:**
> PPO는 "현재 heading으로 전진 + yaw_rate로 회전"으로 학습됨.
> 이동방향을 `target_yaw`로 바꾸면 PPO yaw_rate 출력과 실제 필요 회전량 관계가 깨짐.
> 기하학적 yaw_rate는 재학습 없이 안정적이며 astar에서 이미 검증됨.
>
> **왜 LiDAR 문제가 없는가:**
> yaw_rate를 기하학적으로 제어하므로 heading이 `target_yaw`로 빠르게 수렴.
> heading ≈ target_yaw → LiDAR forward = 이동방향 → 학습 분포 유지.

#### 기존 hybrid vs. hybrid-v2

```
기존 hybrid:
  이동방향 = yaw (현재 heading)    ← 근본 원인
  yaw_rate  = PPO 출력
  v_cmd     = PPO 출력

hybrid-v2:
  이동방향 = target_yaw (A* lookahead 방향)
  yaw_rate  = clamp(2.0 * yaw_err, ±planner_max_yaw_rate)   ← astar와 동일
  v_cmd     = PPO 출력 * align_scale                         ← PPO의 실질 기여
```

#### 수정 코드

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

```python
def _tick_hybrid_mode(self, px, py, yaw, d_min_sectors):
    """Hybrid-v2: A*(전역 경로) + 기하학적 yaw + PPO v_cmd(장애물 반응 속도)."""
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
    v_cmd, _ = self._action_to_vel(action)   # yaw_rate 출력 미사용

    # [Fix 6] 기하학적 heading 정렬 (astar _tick_global과 동일 로직)
    lookahead_dist = float(getattr(self, 'hybrid_lookahead_dist_m', 1.0))
    wp = self._path_lookahead_wp_at_dist(px, py, lookahead_dist)
    if wp is None:
        wp = tuple(self.goal_xy)
    target_yaw = math.atan2(wp[1] - py, wp[0] - px)
    yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    planner_yaw_limit = float(getattr(self, 'planner_max_yaw_rate', self.max_yaw_rate))
    yaw_rate = max(-planner_yaw_limit, min(planner_yaw_limit, 2.0 * yaw_err))
    align_scale = max(0.0, 1.0 - abs(yaw_err) / (math.pi / 2))
    v_cmd *= align_scale
    # yaw_err > 90°: align_scale=0 → 정지 후 회전
    # yaw_err = 60°: align_scale≈0.33 → 감속하며 회전
    # yaw_err ≤ 15°(heading_align_tol): PPO v_cmd 전속 허용

    if (self._is_heading_outward_near_boundary(px, py, yaw)
            and d_min_sectors < self.boundary_turn_clearance):
        v_cmd = 0.0
        yaw_rate = self._compute_boundary_turn_yaw_rate(px, py, yaw)
    with self._path_lock:
        have_path_slow = bool(self._path_xy)
    if have_path_slow and d_min_sectors < self.planner_slowdown_dist and self.planner_slowdown_dist > 0:
        v_cmd *= max(0.0, d_min_sectors / self.planner_slowdown_dist)

    # 이동방향 = target_yaw (waypoint 방향), 현재 yaw 아님
    vx = v_cmd * math.cos(target_yaw)
    vy = v_cmd * math.sin(target_yaw)
    return vx, vy, self._vz_correction(), yaw_rate
```

> **`_escape_if_spinning` 제거 이유:**
> 기존: PPO yaw_rate 포화 감지용. hybrid-v2는 yaw_rate가 기하학적이므로 불필요.

---

### PPO의 실질 기여 분석 및 두 가지 접근 방식

hybrid-v2에서 PPO는 `v_cmd`(속도) **하나만** 담당합니다.  
yaw_rate와 이동방향이 모두 기하학적이므로 PPO의 역할이 제한됩니다.

**핵심 의문:** PPO는 `(v_cmd, yaw_rate)`를 함께 학습했습니다.  
yaw_rate를 버리고 v_cmd만 사용하면 두 출력이 함께 형성된 행동 정책에서  
v_cmd만 떼어 쓰는 **off-policy 사용**에 가깝습니다. v_cmd 품질 보장 없음.

---

### 방식 A: hybrid-v2 그대로 실험 (현재 코드 적용 가능, 재학습 불필요)

**PPO 기여:** v_cmd(속도 제어)만. yaw_rate 미사용.

astar 대비 이론적 우위:

| | astar | hybrid-v2 (방식 A) |
|---|---|---|
| 속도 결정 | `min(v_max, 0.8 * gnorm)` (waypoint 거리) | PPO `v_cmd` (LiDAR 64빔) |
| 장애물 근접 감속 | waypoint 거리에만 의존 | LiDAR 반응 (이론상 세밀) |
| 열린 공간 | 선형 가속 | PPO 학습된 가속 |

**한계:** v_cmd가 yaw_rate와 함께 학습되어 단독 사용 시 품질 미보장.  
실험 결과가 astar와 유사하면 hybrid 우위 증명 어려움.

**연산 영향:**
- `_path_lookahead_wp_at_dist` 추가 1회 호출: ~5μs. 무시 가능.
- 삼각함수 2회 추가: ~10μs/tick. 무시 가능.
- `_path_lock` 재진입: `RLock` → 데드락 없음.
- `_escape_if_spinning` 제거: 연산 감소.

**검증 지표 (동일 코스 3모드 비교):**

| 지표 | 측정 방법 |
|---|---|
| 완주율 | 5회 실행 성공 횟수 |
| 평균 소요시간 | goal reached 시점 |
| emergency brake 횟수 | 로그 카운트 |
| 경로 부드러움 | cmd_vel 방향 변화량 누적 |

---

### 방식 C: PPO 재학습 (v_cmd 전용 policy)

**목표:** hybrid-v2 구조에 맞게 처음부터 설계된 policy 학습.  
yaw_rate를 action space에서 제거하고 v_cmd만 출력하도록 학습.  
→ PPO가 "속도 제어 전문가"로 명확히 역할 분리됨.

**학습 환경 변경 사항:**

| 항목 | 현재 policy | 재학습 policy |
|---|---|---|
| action space | `[v_cmd, yaw_rate]` | `[v_cmd]` |
| 이동 방향 | 현재 heading | A* target_yaw (고정) |
| heading 제어 | PPO yaw_rate | 기하학적 (환경에서 처리) |
| obs | LiDAR 64 + goal_rel 2 | LiDAR 64 + goal_rel 2 (동일) |

**재학습 후 hybrid-v2의 PPO 기여:**
- v_cmd가 yaw_rate와 분리되어 학습 → 속도 제어 품질 보장
- LiDAR 패턴 → 속도 결정의 인과관계가 명확히 학습됨
- astar 대비 우위가 실험적으로 증명 가능

**재학습 체크리스트:**
- [ ] 학습 환경에서 action space를 `[v_cmd]`로 축소
- [ ] 이동방향을 A* target_yaw로 고정 (환경 내 geometric yaw 제어)
- [ ] LOOKAHEAD_NORM_SCALE 학습 시 실제 사용값(3.0)과 일치 확인
- [ ] obs_norm.json 재생성 (obs 분포 변화 반영)
- [ ] 새 policy 파일명: `policy_hybrid_v2_ts.pt`, `obs_norm_hybrid_v2.json`

**진행 시점:** 방식 A 실험 결과 확인 후 결정.

---

## 2026-04-18: Emergency Brake 반복 · A* 웨이포인트 방향 역전 수정 계획 (버그 검토 + 연산 부하 검토 반영본)

> **배경 분석:** `sim_20260417_134952` 배그(295초, 44,754 메시지) 분석을 통해
> 두 가지 구조적 문제를 확인하고 안전한 개선 조합을 도출함.
>
> - Emergency brake: 295초간 9회 cmd_vel 완전 정지(v=0), 530회 발행 지연(>0.15s)
> - 방향 역전: t=81s · 144s · 148s에서 순간 ~180° 속도 방향 반전 확인
>
> **2026-04-18 재검토 — 발견된 문제 3건 및 연산 부하 검토:**
>
> | Fix | 초안 문제 | 심각도 |
> |-----|-----------|--------|
> | Fix 1 | `ADVANCE_RADIUS=0.5m` threshold = 현재 코드가 이미 포기한 방식 재도입 (regression). `hybrid_lookahead` 추종 중 wp_idx 동결 유발 | **치명** |
> | Fix 2 | `path[-2]` 반환: `wp_idx >= n-2` 시 현재 위치를 목표로 가리켜 드론 정지·맴돌기. 코드에도 `max(start_idx+1, n-1)` 논리 오류 혼재 | **치명** |
> | Fix 3 | `threading.Event` 플래그 `.is_set()` 미확인 → Event 완전 no-op. 연산 부하는 없으나 의도한 reactive 동작 없음 | 중간 |
> | Fix 4 | `generate_binary_structure(2,1)` 4-connected는 대각 코너 미커버 | 경미 |
> | Fix 5 | `smooth_bspline_output_points: 80` 증가가 세그먼트 투영(Fix 1 교체)과 정합 안 됨 | 경미 |
>
> **연산 부하 요약:**
> - Fix 1 세그먼트 투영: 50Hz × 10 iterations = 초당 ~3,500 FLOP. 원본 greedy O(N) 대비 오히려 bounded.
> - Fix 3 `replan_period` 단축(3.0→1.0s): A* ~8ms/call, replan 스레드 독립 → tick 차단 없음. CPU ≤2.4%.
> - Fix 4 8-connected dilation: grid_fuser 3.52Hz 기준 추가 ~1ms/call = 3.5ms/s. 무시 가능.
> - Fix 2·5: O(1) 조건 분기만 추가. 연산 영향 없음.

---

### Fix 1. `rl_offboard_node.py` — `_path_advance_wp` 세그먼트 수직 투영으로 교체

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**원인:**
현재 greedy projection(`d_next ≤ d_cur`)은 드론이 경로에서 측방 드리프트 시 여러 wp를
한 tick에 건너뜀. B-spline 50점 조밀 경로에서 라우팅 포인트(코너)를 skip →
`goal_rel` 급변 → PPO 방향 역전.

배그 증거: t=81.22s, 144.73s, 148.0s 에서 vx/vy 가 순간 ~180° 반전.

> **초안 버그 — `ADVANCE_RADIUS=0.5m` threshold 방식은 이미 포기된 regression:**
>
> 현재 코드 주석 [rl_offboard_node.py:1168-1169] 명시:
> *"기존 0.5m 고정 threshold 방식은 lookahead를 쫓는 드론이 현재 wp를 비스듬히 지나치면
> idx가 멈추는 문제가 있었음"*
>
> `hybrid_lookahead_dist_m=1.0m`에서 lookahead 타깃은 현재 wp 앞 1.0m 지점.
> 드론이 그 방향으로 사선 이동하면 현재 wp를 측면에서 0.5m 이상 거리로 통과.
> `d_cur > 0.5` 유지 → while 루프 즉시 break → wp_idx 동결 → 경로 추종 완전 불능.
> threshold 방식은 greedy 문제를 제거하는 대신 freezing 문제를 재도입하는 regression이다.

**수정 (세그먼트 수직 투영):**

```python
# Before
def _path_advance_wp(self, px: float, py: float):
    with self._path_lock:
        ...
        while self._path_wp_idx < n - 1:
            d_cur  = math.hypot(wp_cur[0]  - px, wp_cur[1]  - py)
            d_next = math.hypot(wp_next[0] - px, wp_next[1] - py)
            if d_next <= d_cur:
                self._path_wp_idx += 1
            else:
                break

# After
def _path_advance_wp(self, px: float, py: float):
    """현재 위치에서 가장 가까운 경로 세그먼트를 수직 투영하여 wp_idx를 결정.

    greedy(d_next<=d_cur): 측방 드리프트 시 코너를 한 번에 건너뜀.
    threshold(ADVANCE_RADIUS=0.5m): lookahead 추종 중 비스듬히 지나치면 idx 동결.
      → 두 방식 모두 현재 코드 주석(line 1168-1169)에 문제가 이미 기록됨.
    세그먼트 투영: 측방 이탈과 무관하게 경로 위 투영점(t)으로 정확히 위치 판단.
      greedy 점프도 freezing도 없음.
    연산: 최대 search_window(10)회 루프 × O(1) = 50Hz 기준 초당 ~3,500 FLOP. 무시 가능.
    _path_lock 보유 시간: ~10-20μs. replan 스레드 대기 영향 없음.
    """
    with self._path_lock:
        if not self._path_xy or self._path_wp_idx >= len(self._path_xy):
            return
        n = len(self._path_xy)

        # 현재 wp_idx 앞 최대 search_window 개 세그먼트에서 가장 가까운 세그먼트 탐색
        search_window = int(getattr(self, 'wp_advance_search_window', 10))
        search_end = min(n - 1, self._path_wp_idx + search_window)

        best_seg_idx = self._path_wp_idx
        best_t        = 0.0
        best_dist     = float('inf')

        for i in range(self._path_wp_idx, search_end):
            ax, ay = self._path_xy[i]
            bx, by = self._path_xy[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-10:
                continue
            # 드론 위치를 세그먼트 AB에 수직 투영, t=[0,1]로 클램프
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx = ax + t * dx
            cy = ay + t * dy
            d  = math.hypot(px - cx, py - cy)
            if d < best_dist:
                best_dist     = d
                best_seg_idx  = i
                best_t        = t

        # 세그먼트 후반부(t > 0.5)에 있으면 다음 wp로 전진.
        # max()로 wp_idx가 뒤로 되돌아가지 않도록 보장.
        if best_t > 0.5 and best_seg_idx + 1 < n:
            self._path_wp_idx = max(self._path_wp_idx, best_seg_idx + 1)
        else:
            self._path_wp_idx = max(self._path_wp_idx, best_seg_idx)
```

**파라미터 추가 (`params.yaml`):**

```yaml
wp_advance_search_window: 10  # 세그먼트 투영 탐색 창. 50점 경로 기준 ~1m 전방까지 탐색.
```

**연산 영향:**
- 최대 10회 루프, 각 O(1). 50Hz 기준 초당 ~3,500 FLOP.
- 원본 greedy 최악 O(N=50)보다 bounded. 오히려 빠를 수 있음.
- `_path_lock` 보유 ~10μs → replan 스레드 경로 쓰기 대기 무시 가능.

---

### Fix 2. `rl_offboard_node.py` — `_path_lookahead_wp_at_dist` 잔여 경로 처리

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**원인:**
`start → routing → goal` 3-wp 경로(LOS shortcut 후, `smooth_bspline_skip_if_few_wps=4`로
B-spline 미적용)에서 전체 잔여 길이 < `hybrid_lookahead_dist_m(1.0m)` 이면
현재 코드가 `path[-1]`(goal)을 반환. PPO에 전달되는 `goal_rel`이 routing 포인트 방향이
아닌 최종 goal 방향을 가리켜 코너 무시 → 장애물 근접 후 brake.

> **초안 버그 1 — `path[-2]` 반환 시 `wp_idx >= n-2` 데드락:**
>
> 경로 `[start(0), routing(1), goal(2)]`, `wp_idx=1` (드론이 routing 근처):
> - `start_idx = 1`
> - 루프 누적: `routing→goal ≈ 1.12m < lookahead(1.0m)` → fall-through
> - `path[-2] = path[1]` = 현재 wp 자신
> - `goal_rel`이 현재 위치를 가리킴 → `gnorm ≈ 0` → PPO 입력 `[0,0]` → 드론 정지·맴돌기
>
> **초안 버그 2 — 수정 코드에 논리 오류:**
>
> ```python
> forward_idx = min(n - 1, max(start_idx + 1, n - 1))  # 항상 n-1이 됨
> if n >= 3 and start_idx < n - 2:
>     forward_idx = start_idx + 1
> ```
> 첫 줄 `max(start_idx+1, n-1)` = `n-1`(항상). 두 번째 줄이 덮어쓰는 구조는
> 의도와 달리 `start_idx >= n-2` 시 여전히 `n-1`(goal)을 반환.
>
> **올바른 수정:** `path[min(n-1, start_idx+1)]`로 단순화.
> - `start_idx=0`: `path[1]` = routing ✓ (코너 추종)
> - `start_idx=1`: `path[2]` = goal ✓ (routing 통과 후 goal 직행)
> - `start_idx=n-1`: `path[n-1]` ✓ (범위 초과 없음)

**수정:**

```python
# _path_lookahead_wp_at_dist() 마지막 반환부 교체

# Before
        # 경로가 lookahead_dist보다 짧으면 마지막 wp 반환
        return path[-1]

# After
        # 경로 잔여가 lookahead_dist보다 짧을 때:
        # path[min(n-1, start_idx+1)]으로 항상 현재보다 한 발 앞을 반환.
        # - start_idx=0, path=[start,routing,goal]: path[1]=routing → 코너 추종 보장
        # - start_idx=1 (routing 통과 후): path[2]=goal → goal 직행 허용
        # - 초안의 path[-2] 방식은 wp_idx>=n-2 시 현재 위치를 가리켜 드론 정지 유발
        return path[min(n - 1, start_idx + 1)]
```

**연산 영향:** O(1) 조건 분기 교체. 무시할 수준.

---

### Fix 3. `rl_offboard_node.py` — reactive replan `threading.Event` 구현

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**원인:**
전방 장애물 근접 시 `replan_period=3.0s` 타이머만 의존 → 최대 3초 동안 stale 경로 유지.
배그에서 grid 발행 실측 3.52Hz 기준 3초 = 약 10프레임 지연.

**위험 회피:**
`threading.Thread(target=self._plan_path)` 직접 호출은 `_cb_group_replan(MutuallyExclusiveCallbackGroup)`을
우회해 경로 thrashing 유발. 대신 `threading.Event` flag로 요청하고 `_tick_replan`에서 처리.

> **초안 버그 — Event 플래그가 완전 no-op:**
>
> ```python
> # 초안 _tick_replan
> self._replan_requested.clear()   # 무조건 clear
> self._plan_path()                # 무조건 실행 (flag 상태 무관)
> ```
> `.is_set()` 체크 없이 항상 `_plan_path()` 호출 → Event 플래그 세팅 여부와 무관하게
> 기존 동작과 identical. `_tick()`의 `set()` 호출이 아무 효과를 주지 못함.
>
> **구조적 한계 (수정 후에도 존재):**
> `_tick_replan`은 ROS2 타이머 콜백. reactive 요청이 있어도 최대 `replan_period`만큼 대기.
> `_plan_path()`를 `_tick()` 내에서 직접 호출하면 `_cb_group_replan` 우회 → thrashing 위험.
> 따라서 **`replan_period` 자체를 단축**하는 것이 실질적인 latency 감소 수단.
> Event 플래그는 의미를 명확히 하는 코드 문서화 역할로 유지.

**수정:**

```python
# __init__ 에 추가
self._replan_requested = threading.Event()

# _tick() 내 d_front_min 계산 직후, emergency brake 조건문 이전에 삽입
REACTIVE_REPLAN_DIST = self.emergency_brake_dist * 2.5  # ≈ 0.93m
if (d_front_min < REACTIVE_REPLAN_DIST
        and not self._goal_reached
        and self._path_xy):
    self._replan_requested.set()   # flag만 세우고 즉시 반환 (tick 블로킹 없음)

# _tick_replan() 교체
def _tick_replan(self) -> None:
    """주기적 경로 재계획. reactive 요청 수신 시에도 이 콜백에서 처리.

    구조적 한계: _tick_replan은 ROS2 타이머 콜백(_cb_group_replan)이므로
    reactive 요청 후 실제 replan은 최대 replan_period 만큼 지연된다.
    _plan_path를 tick 스레드에서 직접 호출 시 MutuallyExclusiveCallbackGroup 우회 →
    경로 thrashing 위험. replan_period=1.0s 단축이 실질적 latency 감소 수단.
    """
    if not (self.nav_mode in ("hybrid", "astar")
            and self.odom is not None
            and self.goal_xy is not None
            and not self._goal_reached):
        self._replan_requested.clear()
        return
    # 타이머 주기마다 무조건 재계획. reactive flag는 의미 명시 목적으로 clear.
    self._replan_requested.clear()
    self._plan_path()
```

**연산 영향:**
- `Event.set()`: O(1), tick 블로킹 없음. 50Hz 최악 호출 시에도 무시 가능.
- `replan_period` 단축(3.0→1.0s): A* ~8ms/call × 1Hz = CPU 0.8%(일반) / 2.4%(최악 3회 재시도).
  replan 스레드(`_cb_group_replan`)는 tick 스레드와 독립 → tick latency 영향 없음.
- **`replan_period` 하한**: A* 최악 ~50ms 가정 시 0.5s 이하는 콜백 중첩 위험.
  1.0s가 안전 최솟값.

---

### Fix 4. `grid_fuser.py` — 발행 전 1-cell morphological dilation 추가

**파일:** `px4_nav_perception/px4_nav_perception/nodes/grid_fuser.py`
**파라미터:** `px4_nav_perception/config/grid_fuser.yaml`

**원인:**
log-odds 가장자리 셀(`prob_hit=0.80`, `occupied_thresh=0.55`)은 LiDAR 빔이 비스듬히 닿아
occupied 확정이 느리고 드론 이동 시 miss 누적으로 oscillation 발생.
실제 장애물 끝 1-2셀(0.2~0.4m)이 FREE 로 발행 → A*가 완전하지 않은 지도에서 경로 생성.

**중복 적용 방지:**
`publish_dilation_cells=1` 적용 시 `astar_min_inflation_cells`는 **3 유지** (올리지 않음).
동시 적용 시 실질 마진 = (1+3)×0.2 = 0.8m → 2m 통로 유효폭 0.4m → fallback 급증.

**수정:**

```python
# grid_fuser.py 상단 import에 추가
try:
    from scipy.ndimage import binary_dilation as _scipy_dilation
    from scipy.ndimage import generate_binary_structure as _gen_struct
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# __init__ (파라미터 선언 후)
self.publish_dilation_cells = int(
    self.get_parameter('publish_dilation_cells').value)
self._dilation_struct = None
if self.publish_dilation_cells > 0 and _SCIPY_AVAILABLE:
    # rank=2: 8-connected (대각 포함).
    # 코너 장애물에서 LiDAR 빔이 대각으로 입사할 때 가장자리 FREE 셀을 커버.
    # rank=1 (4-connected, 초안)은 대각 방향 코너를 놓칠 수 있음.
    self._dilation_struct = _gen_struct(2, 2)

# _export_int_grid() 내 occ_mask 생성 직후
if (self.publish_dilation_cells > 0
        and _SCIPY_AVAILABLE
        and self._dilation_struct is not None):
    occ_mask = _scipy_dilation(
        occ_mask,
        structure=self._dilation_struct,
        iterations=self.publish_dilation_cells,
    )
```

```yaml
# grid_fuser.yaml
publish_dilation_cells: 1   # 발행 전 occupied 셀 팽창 반경 (셀 단위). 0=비활성
```

**연산 영향:**
- 221×221 배열, 8-connected dilation: ~3ms/call (4-connected ~2ms 대비 +1ms).
- grid_fuser 발행률 3.52Hz 기준 추가 CPU: ~3.5ms/s. 완전 무시 가능.
- `_dilation_struct` `__init__`에서 사전 생성 → 매 호출 비용 없음.
- scipy 미설치 시 `_SCIPY_AVAILABLE=False` → 자동 비활성.

---

### Fix 5. `params.yaml` — 타이밍·lookahead 파라미터 조정

**파일:** `px4_rl_offboard_mavros/config/params.yaml`

```yaml
# Before → After (근거 포함)

replan_period: 3.0 → 1.0
# 근거: Fix 3 reactive replan이 최대 replan_period만큼 대기.
# 1.0s × 0.8m/s = 0.8m 전진 내 경로 갱신 → 장애물 0.93m 전방 감지 후 충분히 여유.
# A* ~8ms/call × 1Hz = CPU 0.8%(일반). replan 스레드 독립, tick 차단 없음.
# 하한: 0.5s 이하는 A* 최악 ~50ms 기준 콜백 중첩 위험 → 1.0s 유지.

hybrid_lookahead_dist_m: 1.5 → 1.0
# 근거: Fix 2 start_idx+1 반환과 시너지.
# 1.0m lookahead는 좁은 통로 직전 코너 방향을 더 정확히 지향.
# Fix 1 세그먼트 투영으로 wp_idx 추종 정확도 향상 → 1.0m도 안정적.

smooth_bspline_output_points: 50 (변경 없음)
# 근거: 초안 80점 증가는 Fix 1이 threshold 방식이었을 때의 정합 로직.
# Fix 1이 세그먼트 투영으로 교체되면 점 밀도가 wp 전진에 영향을 주지 않음.
# 80점 증가는 _smooth_path_bspline 내 clearance 체크 1.6배 증가만 유발 → 50 유지.

wp_advance_search_window: 10  # Fix 1 추가 파라미터
# 50점 경로 기준 ~1.0m 전방까지 탐색. 일반적으로 충분.
```

---

### 적용 순서 및 검증 방법

| 순서 | Fix | 검증 방법 |
|------|-----|-----------|
| 1 | Fix 1 (`_path_advance_wp` 세그먼트 투영) | RViz `/planner/path` wp_idx 점프 없음 + 측방 드리프트 시 동결 없음 확인 |
| 2 | Fix 2 (`start_idx+1` 반환) | 3-wp 경로 구간에서 `goal_rel` 방향이 코너를 향하는지 `rqt_plot` 확인. wp_idx=1 시 goal을 향하는지 확인 |
| 3 | Fix 5 params 적용 | `replan_period: 1.0s` 적용 후 replan log 1.0s 간격 확인 |
| 4 | Fix 3 (reactive replan) | d_front < 0.93m 시 최대 1.0s 내 `_tick_replan` 실행 log 확인 |
| 5 | Fix 4 (grid dilation 8-connected) | RViz `/local_grid` 장애물 가장자리 1셀 확장 + 코너 대각 셀 포함 확인 |

**전체 통합 검증:** Fix 1~5 모두 적용 후 동일 코스 재시뮬레이션.
기대 지표:
- emergency brake 발생 횟수: 9회 → 3회 이하
- cmd_vel 방향 역전: 3회 → 0회
- 목표 도달 시간: 단축 또는 유지

---

## 2026-04-16: `params.yaml` — 미사용 RL 트리거 파라미터 제거

### `px4_rl_offboard_mavros/config/params.yaml`

**제거된 파라미터:**
- `rl_trigger_front_enter: 2.5`
- `rl_trigger_front_exit: 5.0`
- `rl_min_dwell_sec: 5.0`

**배경:** `hybrid` 모드에서는 PPO가 항상 실행되므로 RL 진입/해제 트리거 로직 자체가 없음.
코드(`rl_offboard_node.py`) `params` 선언 목록에도 포함되어 있지 않아 노드가 로드조차 하지 않던 dead params.

---

## 2026-04-16: `params.yaml` — `smooth_bspline_skip_if_few_wps` 4 → 3

### `px4_rl_offboard_mavros/config/params.yaml`

```yaml
# Before
smooth_bspline_skip_if_few_wps: 4

# After
smooth_bspline_skip_if_few_wps: 3
```

**배경:** `shortcut_path` 이후 경로가 3개 wp(start → routing_point → goal)로 압축되는 경우가 빈번함.
기존 skip 조건 `< 4`에서는 이 3-wp 경로에 B-spline이 적용되지 않아 routing_point에서 sharp corner가 유지됨.
코너 직전·직후 구간에서 PPO가 급격한 방향 전환을 수행해야 하므로 heading 오차가 누적됨.

**수정:** 3-wp 경로부터 B-spline smoothing 적용 (2-wp 직선 경로는 여전히 skip).
routing_point 코너가 부드럽게 변환되어 드론이 A* 경로를 보다 자연스럽게 추종 가능.

---

## 2026-04-16: A* 경로 벽 여유 부족 및 B-spline clearance 손실 수정

### `px4_rl_offboard_mavros/config/params.yaml`

**원인 분석 (R 코스 emergency brake 반복):**

`astar_min_inflation_cells=2` (0.4m) < `emergency_brake_dist=0.5m` 구조적 불일치.
A*가 생성한 경로 자체가 이미 emergency brake 발동 구역(0.5m 이내) 안에 위치하므로,
드론이 계획 경로를 정확히 따라도 `d_front ≈ 0.4~0.5m`가 측정되어 브레이크가 구조적으로 반복.

로그 근거: 전체 emergency brake의 d_front 값이 0.44~0.50m 범위에 집중 (0.5m 한계에서 촉발).

두 번째 요인: `smooth_bspline_s=0.1` (근사 모드)로 B-spline이 경유점을 정확히 통과하지 않고
코너 내측을 단축(shortcut)하여 A* inflation 보장(0.4m)보다 벽에 더 가까운 경로 생성 가능.

세 번째 요인: `min_inflation reduced 2→1` fallback 발생 시(로그에서 16회+) 경로 여유가 0.2m로 감소.

**수정 (params.yaml 3개 항목):**

```yaml
# ① astar_min_inflation_cells: 2 → 3
# 경로-벽 여유 0.4m → 0.6m
# 0.6m > emergency_brake_dist(0.5m) → 계획 경로가 브레이크 구역 밖
# 좁은 S 코스 2m 복도: 2m - 2×0.6m = 0.8m 유효폭, fallback 3→2→1 자동 적용
astar_min_inflation_cells: 3

# ② shortcut_los_clearance_cells: 2 → 3
# astar_min_inflation_cells와 일치: shortcut LOS도 동일 clearance 적용
shortcut_los_clearance_cells: 3

# ③ smooth_bspline_s: 0.1 → 0.0
# 근사(approximation) → 보간(interpolation): 모든 경유점을 정확히 통과
# A* inflation 보장을 B-spline smoothing 후에도 유지
smooth_bspline_s: 0.0
```

**예상 효과:**
- emergency brake 발생 빈도 대폭 감소 (구조적 불일치 제거)
- min_inflation fallback 빈도는 증가할 수 있으나 narrow passage 처리 로직이 자동 대응
- B-spline이 경유점을 이탈하지 않아 waypoint 추종 정확도 향상

---

## 2026-04-15: hybrid 모드 PPO lookahead 거리 분리 파라미터화 (`hybrid_lookahead_dist_m`)

### `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py` + `config/params.yaml`

**배경:** R 코스(ㄹ-pattern) 미로에서 드론이 x≈5~6m, y≈6.5m 지점에 반복 정체.
inner_r_h1 (y=6m 수평벽, x∈[2,7]) 옆면을 따라 이동 중 3m lookahead 포인트가
갭(x>7m) 통과 후 남쪽으로 꺾인 경로 위를 가리켜 PPO에 남향 velocity 성분을 부여.
결과적으로 드론이 벽 방향(남쪽)으로 접근 → emergency brake 반복(약 0.5~1.5s 주기).

**원인 구조:**
```
드론 at (5.5, 6.5), 벽 at y=6.0, A* 경로: → (7m, 6.5m) 갭 → (7.5m, 3m) 남쪽 코너
3m lookahead 포인트: 갭 통과 후 경로상 3m 지점 ≈ (7.x, 4.x)
goal_rel 방향 = SE (동남) → PPO가 남향 속도 성분 출력 → 벽 접근 → brake
```

**수정:**

1. **`hybrid_lookahead_dist_m` 파라미터 추가** (기본값 1.5m):
   - lookahead 거리를 코드 하드코딩에서 분리하여 params.yaml에서 조정 가능
   - 1.5m로 줄이면 lookahead 포인트가 갭 진입 전 경로상(≈동쪽)에 머물러
     남향 velocity 성분이 억제됨

2. **`norm_scale`을 학습값(3.0)으로 고정** (`LOOKAHEAD_NORM_SCALE = 3.0`):
   - lookahead 거리와 정규화 스케일을 분리
   - policy는 학습 시 goal_rel을 3.0m 기준으로 정규화된 값을 봤으므로
     lookahead 거리를 줄여도 norm_scale 유지 → policy 입력 분포 변화 없음
   - 1.5m lookahead + norm_scale=3.0 → goal_rel 크기 ≤ 0.5 (방향만 영향)

```python
# Before (하드코딩)
LOOKAHEAD_DIST = 3.0
wp = self._path_lookahead_wp_at_dist(px, py, LOOKAHEAD_DIST)
norm_scale = LOOKAHEAD_DIST   # ← lookahead와 norm이 연동

# After (분리)
LOOKAHEAD_NORM_SCALE = 3.0   # 학습값 고정
lookahead_dist = float(getattr(self, "hybrid_lookahead_dist_m", 1.5))
wp = self._path_lookahead_wp_at_dist(px, py, lookahead_dist)
norm_scale = LOOKAHEAD_NORM_SCALE   # ← norm은 항상 3.0
```

```yaml
# params.yaml 추가
hybrid_lookahead_dist_m: 1.5   # 3.0=기존, 1.5=R 코스 벽 방향 성분 억제
```

**튜닝 가이드:**
- `1.5`: R 코스 좁은 통로 진입 직전 벽 방향 velocity 성분 억제 (현재 설정)
- `3.0`: 기존 동작 (goal_rel 크기 ≤1.0 보장, 통로 너머 꺾임 허용)
- norm_scale은 항상 3.0으로 고정 → 값을 바꿔도 policy 입력 분포에 영향 없음

---

## 2026-04-15: `params.yaml` — `astar_min_inflation_cells` 2 → 1 하향

### `px4_rl_offboard_mavros/config/params.yaml`

- **`astar_min_inflation_cells`: 2 → 1** (벽 여유 0.4m → 0.2m)

**배경:** `sim_20260415_155832` 로그 분석: obstacle_course.world S-course의 좁은 통로(물리 폭 ~1.7m)에서 `min_inflation reduced 2→1` 경고가 집중 발생. inflation=2(0.4m)로는 유효 통과 폭이 0.9m로 경로 생성 실패가 빈번하여 매 replan마다 inflation=2 북쪽 우회 / inflation=1 남쪽 최단 경로가 교번 → 드론이 잘못된 방향으로 1~2회 replan분 이동하는 문제.

**근거:** PPO의 emergency brake(`emergency_brake_dist=0.5m`)와 `use_override=True` 자기감지 필터가 최종 충돌 방어를 담당하므로 A* inflation을 낮춰도 안전성 유지 가능. inflation=1(0.2m)로 고정하면 통로 경로 생성 안정성이 높아지고 fallback 교번이 사라짐.

---

## 2026-04-15: A* shortcut LOS에 inflation clearance 적용 (`shortcut_los_clearance_cells` dead param 수정)

### `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py` + `config/params.yaml`

**원인:** `shortcut_los_clearance_cells` 파라미터가 선언되어 있었지만 `_los_clear`에서 실제로 사용되지 않아 dead param 상태였음.

```python
# 기존: _los_clear가 raw self._grid_data만 체크 (clearance 파라미터 무시)
for ci, cj in self._bresenham_cells(...):
    if self._is_occ(ci, cj):   # raw grid, inflation 없음
        return False
```

A*는 `astar_min_inflation_cells=2`(0.4m) inflation grid로 안전한 경로를 생성하지만,
`_shortcut_path`는 raw grid LOS로 이 경로를 3 wps로 과도하게 단축했음. 단축된 경로는
inflation 마진 내(벽 0.4m 이내)를 통과 → emergency brake 반복 → 실제 이동거리 증가.

**수정:**
- `_los_clear`에 `grid: Optional[np.ndarray] = None` 파라미터 추가. `None`이면 raw grid(기존 동작 유지), 팽창 그리드 전달 시 그 grid 기준으로 체크.
- `_shortcut_path`에서 `shortcut_los_clearance_cells > 0`이면 `_inflate_grid_temp(clearance)`를 1회 계산 후 전달.
- `_paths_same_homotopy`의 `_los_clear` 호출은 grid=None 유지 (homotopy 판별 의미 변경 없도록).
- `params.yaml`: `shortcut_los_clearance_cells: 0 → 2` (`astar_min_inflation_cells=2`와 일치).

**효과:** shortcut 경로도 A*와 동일한 0.4m 벽 clearance 기준으로 LOS 체크 → 벽 근접 단축 차단 → 경로 wp 수 증가 → distance 기반 lookahead와 함께 routing point가 올바르게 PPO target으로 사용됨.

---

## 2026-04-15: `rl_offboard_node.py` — hybrid 모드 lookahead를 distance 기반으로 교체

### `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**배경:** `sim_20260415_135755` 로그 분석: obstacle_course.world S-course 진입 후
`(-3.5, 2.5)` 지점(inner_s_v1 0.5m 전방)에서 30초 이상 정체. PPO-only는 성공.

**원인:** A* 경로가 LOS 단축 후 3 wps로 압축될 때 step-count 기반 lookahead
(`path_lookahead_steps=12`)가 즉시 마지막 wp(goal)을 가리킴.

```
wp[0]=start → wp[1]=routing_point(y>4 갭) → wp[2]=goal(-8,-8)
wp_idx: 0 → (dist<0.5 → advance) → 1
lookahead: min(1+12, 2) = 2 = goal   ← routing point 완전 무시
```

PPO는 goal 방향(좌하↙)으로 heading하지만 inner_s_v1 벽이 0.5m 앞 → emergency brake 반복.
routing point(wp[1], y>4 갭)는 한 번도 PPO target으로 사용되지 않음.

**수정:** `_path_lookahead_wp_at_dist(px, py, dist)` 메서드 추가. 드론 현재 위치로부터
경로상 `dist`m 앞 지점을 세그먼트 보간으로 계산. `_build_obs`에서 이 메서드를 사용.

```
동일 3 wps 경로, 드론 at (-3.5, 2.5):
  seg: (-3.5,2.5) → (-3.0,5.x), 길이≈2.7m → 3m에 못 미침
  seg: (-3.0,5.x) → (-8.1,-8.3), 길이≈...  → 3m 지점 interpolation
  target ≈ (-3.0+t*..., 5.x+t*...) = routing point 방향 ✓
```

- `_path_lookahead_wp()` (step 기반): emergency brake 회전 방향 결정 전용으로 유지
- `_path_lookahead_wp_at_dist()` (distance 기반): `_build_obs` PPO obs 생성 전용

---

## 2026-04-15: hybrid 모드 cmd_vel 전송률 저하 및 경로 인덱스 정체 수정

### 배경

`sim_20260415_112151` 로그 분석:
- 목표 20 Hz → 실측 **12.8 Hz** (평균 간격 78.3 ms, 36% tick 손실)
- 100 ms 초과 gap 600+건, 최대 400 ms (목표의 8×)
- 원인: `use_topo_diversity: true` → `_astar_diverse` 내부에서 `ThreadPoolExecutor`로 3개 A* 제출 →
  Python GIL로 인해 실제 직렬 3회 실행 → cmd_vel tick 스레드를 상시 차단

별도 문제: `_path_advance_wp`가 현재 wp에서 0.5 m 이내 진입 시에만 idx를 1씩 증가시키는 방식.
lookahead(12칸 앞)를 추종하는 드론은 현재 wp를 비스듬히 지나치면 0.5 m 조건 미충족 → idx 고정 →
`_path_lookahead_wp()` = path[0+12] (1초 전 출발점 기준 3.6 m 앞 고정점) →
드론이 그 지점을 지나치면 PPO target이 뒤를 향하게 됨. obstacle_course 미로 코너에서 특히 심각.

---

### [수정 9] `params.yaml` — `use_topo_diversity: false` (GIL 병목 완화)

**파일:** `px4_rl_offboard_mavros/config/params.yaml`

**분석:**
- `_astar_diverse`는 `topo_inflation_radii_cells=[0,1,2]`에 `astar_min_inflation_cells=2`를 더해
  실제 inflation=[2,3,4]로 3회 A* 실행.
- 최종 경로는 `min(candidates, key=_plen)` → 항상 inflation=2 경로(최단) 선택.
- obstacle_course.world의 S/ㄹ 미로는 경로가 구조적으로 강제되어 homotopy 다양성 실이득 없음.
- noise 방어는 `astar_min_inflation_cells=2` 가 담당하며 diversity 설정과 독립적.
- `use_topo_diversity: false` 시 단일 A* 실행 → GIL 점유 ~1/3 → cmd_vel 12.8 Hz → 17~18 Hz 개선 예상.
- narrow passage fallback (inflation 2→1→0 순차 재시도)은 `false` 코드 경로에도 동일하게 구현됨.

```yaml
# Before
use_topo_diversity: true

# After
use_topo_diversity: false
```

**상태:** ✅ 적용 완료 (2026-04-15)

---

### [수정 10] `rl_offboard_node.py` — `_path_advance_wp` greedy projection 전진

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**원인:** 기존 0.5 m threshold 방식은 lookahead 12칸 앞을 쫓는 드론이 현재 wp를
코너에서 비스듬히 지나치면 idx가 멈춤. 1초 주기 replan으로 어느 정도 보정되지만,
replan 사이 window 내에서 lookahead target이 드론 뒤를 향하는 상황이 발생.

**수정:** "다음 wp가 현재 wp보다 드론에 더 가까우면 = 현재 wp를 이미 지나쳤다"는 판단으로
while 루프에서 여러 칸을 한 번에 건너뜀 (greedy projection).

```python
# Before
wp = self._path_xy[self._path_wp_idx]
dist = math.hypot(wp[0] - px, wp[1] - py)
if dist < 0.5 and self._path_wp_idx < len(self._path_xy) - 1:
    self._path_wp_idx += 1

# After
while self._path_wp_idx < n - 1:
    wp_cur  = self._path_xy[self._path_wp_idx]
    wp_next = self._path_xy[self._path_wp_idx + 1]
    d_cur  = math.hypot(wp_cur[0]  - px, wp_cur[1]  - py)
    d_next = math.hypot(wp_next[0] - px, wp_next[1] - py)
    if d_next <= d_cur:
        self._path_wp_idx += 1
    else:
        break
```

B-spline으로 부드럽게 생성된 경로에서 U턴 같은 역방향 급변 없으므로 과도 전진 위험 없음.

**상태:** ✅ 적용 완료 (2026-04-15)

---

## 2026-04-15: `rl_offboard_node.py` — emergency brake 회전 방향 진동 억제 (direction lock)

### `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**배경:** `sim_20260415_131308` 로그: emergency brake가 106초 내 ~60회 발생하며 `rotating left/right`가 1~2초 내 반복 반전되는 진동 현상 확인.

**원인:** brake 활성 중 매 tick마다 A* lookahead wp의 body-frame 상대각을 재계산하여 회전 방향 결정.
드론이 회전하면 wp의 상대각이 0° 기준으로 flip → `rotate_left ↔ right` 진동 유발.
`_emergency_brake_active` 플래그로 로그는 1회 출력되었지만, 방향 결정 자체는 매 tick 재실행.

**수정:** `self._brake_rotate_left: Optional[bool]` 추가. brake 진입 시(`_brake_rotate_left is None`) 1회만 방향 결정(A* wp 방향 우선, fallback: d_left/d_right). 이후 brake 해제(`d_front > emergency_brake_dist`) 전까지 저장된 방향 재사용(`src=locked`).

```python
if self._brake_rotate_left is None:
    # 진입 시 1회 결정
    ...
    self._brake_rotate_left = rotate_left
else:
    rotate_left = self._brake_rotate_left   # 고정 유지
    rotate_reason = "locked"
# brake 해제 시
self._brake_rotate_left = None
```

**추가 수정:** d_front_min 체크를 `use_override=True`로 변경 → `lidar_range_min_override=0.35m` 적용.
body 자기감지(0.24~0.31m)를 필터링하여 spurious brake 억제. [수정 3]의 설계 의도(`emergency_brake_dist=0.5m`은 body감지 제거 전제) 복원.
※ [수정 5]의 `use_override=False` 의도(실제 근접 장애물 raw 감지)는 d_left/d_right/d_rear 계산에는 그대로 유지.

---

## 2026-04-15: `rl_offboard_node.py` — emergency brake 회전 방향을 A* 경로 방향 우선으로 변경

### `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**변경 위치:** `_tick_control` 내 emergency brake 블록 (1206~1215번 줄)

**기존 동작:** `d_left >= d_right` (좌우 LiDAR 거리 비교)로 회전 방향 결정.

**문제:** 좌우 공간이 비슷하거나 A* 경로가 공간이 좁은 쪽을 가리킬 때 경로와 반대 방향으로 회전,
emergency brake 해제 후 경로 복귀에 추가 시간 소요.

**수정:** `_path_lookahead_wp()`가 유효하면 해당 wp의 월드 각도를 현재 yaw 기준 상대각으로 변환하여
회전 방향 결정. A* 경로가 없는 경우(ppo 모드, 경로 미생성)에만 기존 `d_left/d_right` fallback 유지.

```python
# 회전 방향 결정: A* 경로 lookahead wp가 있으면 그 방향 우선, 없으면 d_left/d_right fallback
rotate_reason = "lidar"
rotate_left = (d_left >= d_right)
wp_la = self._path_lookahead_wp()
if wp_la is not None:
    wp_angle_world = math.atan2(wp_la[1] - py, wp_la[0] - px)
    angle_diff = (wp_angle_world - yaw + math.pi) % (2 * math.pi) - math.pi
    rotate_left = (angle_diff >= 0.0)
    rotate_reason = "astar"
```

로그에 `(src=astar)` / `(src=lidar)` 출력으로 어느 기준이 적용됐는지 확인 가능.

---

## 2026-04-15: PPO hybrid 모드 경로 추종 불량 버그 수정

### 배경

`sim_20260415_112151` 로그 분석: A* 경로는 장애물을 올바르게 우회하여 생성되지만
PPO가 그 방향으로 제어하지 않고 현재 위치→goal 직선 방향으로 진행하는 경향.
드론이 goal 3.8m 전방에서 122초 중 47초 이상 정체(x 이동 std=0.08m)하며 목표 도달 실패.

**로그 분석 요약 (RL 구간 기준):**

| 구간 | 진행량 | 평균 속도 | 평균 \|yr\| |
|------|--------|-----------|-------------|
| t=14~34s | +7.0m | 0.59 m/s | 0.42 |
| t=34~54s | +2.7m | 0.44 m/s | 0.66 (급증) |
| t=74s~   | **+0.0m** | 0.23 m/s | **0.71** (정체) |

정체 구간: full yaw_rate 포화(≥0.95) 46%, x 이동 범위 0.34m, goal까지 3.8~4.1m 고정.

---

### [수정 7] `rl_offboard_node.py` — `_tick_hybrid_mode` goal_bias 제거

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**원인:** `_tick_hybrid_mode`에 `_tick_ppo_mode`와 동일한 `goal_bias` 로직이 그대로 복사되어 있었음.

```python
# 기존 (버그): hybrid 모드에도 goal_bias 적용
if d_min_sectors > self.goal_bias_dist:   # 2.5m 이상 = 장애물 없음
    yaw_rate *= self.goal_bias_yaw_scale  # × 0.25
```

`goal_bias`는 PPO-only 모드에서 최종 goal을 직선 추종할 때 열린 공간에서 드론이
빙글빙글 도는 것을 막기 위해 설계됨. PPO-only에서는 장애물이 없으면 큰 yaw 수정이
불필요하다는 가정이 성립함.

hybrid 모드에서는 PPO가 받는 목표가 최종 goal이 아니라 **A* lookahead waypoint**이므로
장애물 우회 경로에서 waypoint가 현재 heading과 크게 다른 방향에 있을 수 있음.
열린 공간(d_min > 2.5m)에서도 충분한 yaw_rate가 허용되어야 경로를 따를 수 있는데,
goal_bias가 yaw_rate를 25%로 감쇠시켜 경로 추종 불가 상태를 유발했음.

로그 확인: 전체 RL 명령의 17%가 goal_bias에 의해 감쇠 (0.20~0.28 rad/s 구간).
장애물이 없는 초기 구간(t=34~54s)에서 진행량 급감(+1.5m/10s)과 일치.

**수정:** hybrid 모드의 `goal_bias` 블록 2줄 제거. 경계 회전(`boundary_turn`) 로직은 유지.

**상태:** ✅ 적용 완료 (2026-04-15)

```python
# 수정 후: goal_bias 미적용, 경계 회전만 유지
# hybrid 모드: goal_bias 미적용.
# PPO-only 모드의 goal_bias는 직선 goal 추종 전제로 설계됨.
# hybrid 모드에서는 A* waypoint가 장애물 우회를 위해 현재 heading과 크게 다를 수 있으므로
# 열린 공간에서도 충분한 yaw_rate를 허용해야 경로를 따를 수 있음.
if (self._is_heading_outward_near_boundary(px, py, yaw)
        and d_min_sectors < self.boundary_turn_clearance):
    ...
```

---

### [수정 8] `params.yaml` — `path_lookahead_steps` 학습 환경 일치 조정

**파일:** `px4_rl_offboard_mavros/config/params.yaml`

**원인:** `path_lookahead_steps: 3`으로 설정되어 학습 환경(12 steps × 0.25m = 3.0m)과
불일치. `_build_obs`의 `LOOKAHEAD_DIST = 3.0`(하드코딩)으로 나누므로 goal_rel 신호 크기가
`0.9/3.0 = 0.30`으로 학습 기대치(≈1.0)의 3분의 1 수준. PPO가 방향 전환을 충분히 하지
않아 사실상 직진에 가까운 출력을 냄.

또한 장애물 근방에서 0.9m lookahead는 waypoint가 벽 안쪽에 위치할 수 있어 PPO가
혼란스러운 목표를 받는 원인이 됨 (stuck 구간 벽 충돌 루프와 연관).

```yaml
# Before
path_lookahead_steps: 3

# After
path_lookahead_steps: 12  # 학습 조건 일치: 12 × 0.25m = 3.0m = LOOKAHEAD_DIST
```

**상태:** ✅ 적용 완료 (2026-04-15)

---

## 2026-04-15: `rl_offboard_node` / `grid_fuser` 정리 및 버그 수정

### `px4_rl_offboard_mavros/rl_offboard_node.py`

- **`_build_obs` 런타임 오류:** 정의되지 않은 `use_override`를 참조하던 코드를 제거하고, RL 관측에서는 기존 의도대로 `lidar_range_min_override`만 `rmin`에 반영하도록 정리함.
- **`_dist_in_sector`:** `use_override` 인자가 본문에서 무시되던 문제를 `if use_override and override > rmin:`으로 수정(비상 제동은 `use_override=False`로 동작).  
  - 참고: 동일 주제는 아래 [수정 5] 항목과 연속된 작업임.
- **스핀 탈출 로직:** PPO/`hybrid` 모드에 중복되던 요레이트 포화 감지·후진 탈출을 `_escape_if_spinning()`으로 통합하고, `self._yaw_sat_count`를 `__init__`에서 초기화.
- **주석:** B-spline 안전 검증 위의 장문 설명 블록을 삭제(로직 변경 없음).

### `px4_nav_perception/nodes/grid_fuser.py`

- **미사용 코드 제거:** `apply_tf_2d`, `_lerp_pose_2d`, `_apply_pose_2d`, `_update_cell_free`, `_update_cell_occ`(어디에서도 호출되지 않음).
- **구독:** `single_cb` 제거 후 `LaserScan` 구독을 `process_scan`에 직접 연결.
- **Inflation 관련:** publish 경로에서 사용되지 않던 `inflation_radius` / `inflation_unknown_as_occ` / 오프셋 전처리 / `_inflate_grid` 및 로그 제거. 맵 팽창은 `rl_offboard_node`의 A* 경로 쪽에서 처리하는 구조와 일치.
- **주석:** TF·log-odds·ray 처리 부분의 과한 이력 주석을 짧게 정리.

### `px4_nav_perception/config/grid_fuser.yaml`

- 위 inflation 미사용에 맞춰 `inflation_radius` 관련 주석 블록 삭제.

---

## 2026-04-15: 근거리 벽 미감지 및 B-spline 경로 안전 마진 버그 수정 + replan_period 단축

### 배경

`sim_20260414_143250` 로그 분석: 드론이 벽 0.5m 이내 접근 시 `/local_grid`가 벽을 FREE(0)로
출력하여 A* 경로가 벽을 관통 → emergency brake → 회전 → 재계획 무한반복 현상.

### 수정 내역

#### [수정 3] `grid_fuser.py` — log-odds 업데이트 순서 역전 (핵심 버그)

**파일:** `px4_nav_perception/px4_nav_perception/nodes/grid_fuser.py`

**원인:** `process_scan()`에서 hit 업데이트 → clamp_max(3.0) → miss 업데이트 순으로 처리됨.
근거리 장애물은 동일 스캔에서 같은 셀에 다수의 hit 빔(예: 40개, l_hit 합계 87.89)과
측면 free-ray miss 빔(34개, l_miss 합계 -6.82)이 동시 적용되는데, 순서 탓에
`min(87.89, 3.0) = 3.0` → `3.0 + (−6.82) = −3.82` → clamp_min(-2.0) → prob=0.12 → **FREE 출력**.

**수정:** miss(free) 먼저 → clamp → hit(occupied) 나중.
`max(−6.82, −2.0) = −2.0` → `−2.0 + 87.89 = 85.89` → clamp_max(3.0) → prob=0.95 → **OCCUPIED 출력**.

```python
# Before (버그): hit → clamp → miss
np.add.at(self.grid_log, (ey_arr[valid_occ], ex_arr[valid_occ]), self.l_hit)
np.clip(...)
np.add.at(self.grid_log, (cy_all[in_bounds], cx_all[in_bounds]), self.l_miss)
np.clip(...)

# After (수정): miss → clamp → hit
np.add.at(self.grid_log, (cy_all[in_bounds], cx_all[in_bounds]), self.l_miss)
np.clip(...)
np.add.at(self.grid_log, (ey_arr[valid_occ], ex_arr[valid_occ]), self.l_hit)
np.clip(...)
```

---

#### [수정 4] `rl_offboard_node.py` — `_smooth_path_bspline` inflation 안전 검증 추가

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**원인:** B-spline 스무딩 후 안전 검증이 raw occupied 셀(`_is_occ`, threshold=50)만 확인.
A*는 `astar_min_inflation_cells=2`(0.4m) 여유로 벽을 피하지만, smooth path는
inflation margin 내부를 통과할 수 있어 드론이 0.4m 이내로 접근 → emergency brake 유발.

**수정:** `_inflate_grid_temp(astar_min_inflation_cells)` 호출 후 팽창된 그리드 기준 검증.
smooth path 점이 inflation zone 내에 있으면 원본 A* 경로 반환(스무딩 포기).

---

#### [수정 5] `rl_offboard_node.py` — `_dist_in_sector` `use_override` dead code 수정

**파일:** `px4_rl_offboard_mavros/px4_rl_offboard_mavros/rl_offboard_node.py`

**원인:** `use_override` 파라미터가 함수 시그니처에 정의되어 있으나 함수 본문에서 참조되지 않음.
`_min_dist_in_sector(use_override=False)` 호출(비상 제동용)도 실제로는 항상 override 적용.

**수정:** `if use_override and override > rmin:` 조건 추가.
비상 제동 시 `scan.range_min`(0.12m) 기준 사용 → 드론 자기감지 빔도 포함되지만
`arm_mask`로 이미 필터링된 상태이므로 실용적 영향 없음.

---

#### [수정 6] `grid_fuser.yaml` — `occupied_thresh` 0.51 → 0.55

**파일:** `px4_nav_perception/config/grid_fuser.yaml`

**원인:** `occupied_thresh=0.51` → log-odds 기준 0.040. 경계 셀이 단 1회 `l_miss(−0.200)`만
받아도 threshold 아래 탈락(0.040 − 0.200 = −0.160 → FREE). 짧은 스캔 주기에서 occupied↔free 진동.

**수정:** `occupied_thresh=0.55` → log-odds 기준 0.201.
1회 l_miss 후 0.201 − 0.200 = 0.001 > 0 → OCCUPIED 유지. clamp_max=3.0 셀의 안정성 향상.

| 항목 | 이전 | 이후 |
|------|------|------|
| `occupied_thresh` | 0.51 (log-odds 0.040) | 0.55 (log-odds 0.201) |
| 1회 l_miss 후 상태 | 0.040−0.200=−0.160 → **FREE** | 3.0−0.200=2.8 → **OCCUPIED** |
| clamp_max 셀 소거 필요 miss 횟수 | ≈15회 | ≈14회 |

**상태:** ✅ 적용 완료 (2026-04-15)

---

## 2026-04-14: `obstacle_course_light.world` 벽 두께·길이 조정 및 `goals.yaml` 정합

### 배경

- 벽 **두께**를 0.3 m → **0.6 m**으로 키운 뒤, 동일 pose·길이로 두면 **XY 평면에서 박스 AABB가 겹침** (코너, 오른쪽 세로 외벽–`inner_r_h2`/`inner_r_h4`, S코스 상·하단 가로–`inner_s_v3`/`inner_s_v4` 등).
- `px4_nav_planner/config/goals.yaml`의 `home_xyz` / `goal_R` / `goal_L`는 **스폰 기준 ENU**이므로, 월드 수정 시 **목표점이 벽 박스 안에 들어가지 않도록** 함께 확인.

### 수정 파일

| 파일 | 내용 |
|------|------|
| `Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/obstacle_course_light.world` | 두께 0.6 m 유지, **겹침 제거용 길이만** 축소 |
| `px4_nav_planner/config/goals.yaml` | 위 월드와 목표 정합 시 함께 점검하라는 **주석** 추가 (좌표 값은 변경 없음) |

### 월드 (`obstacle_course_light.world`) 수치

**두께 (이전 변경과 동일):** 모든 `<box><size>`의 얇은 변 **0.6 m** (예: `0.6×18×4`, `8×0.6×4` 패턴).

**길이 축소 (겹침 제거):**

| 모델 | 변경 (긴 변) | 비고 |
|------|----------------|------|
| `r_outer_top` | 8 → **7.3** (x) | `r_outer_right`와 NE 코너 겹침 제거 |
| `inner_r_h2`, `inner_r_h4` | 5 → **4.4** (x) | 오른쪽 세로 외벽과 겹침 제거 |
| `s_outer_top` | 8 → **7.3** (x) | `s_outer_left`와 NW 코너 겹침 제거 |
| `inner_s_v3`, `inner_s_v4` | 8 → **7.3** (y) | 각각 상·하 가로 외벽과 겹침 제거 |

`r_outer_bottom`, `inner_r_h1`/`inner_r_h3`, 세로 외벽 `r_outer_right`/`s_outer_left`, `inner_s_v1`/`inner_s_v2` 등은 **겹침 분석상 추가 축소 불필요**로 길이 유지.

### `goals.yaml` 좌표 (변경 없음, 검증 기준)

`spawn_enu` 기준 상대 좌표로, 축소 후 AABB와 **겹치지 않음**: home `(0, 8)`, goal_R `(8, -8.2)`, goal_L `(-8, -8.2)`.

**상태:** ✅ 반영 완료

---

## [계획] 2026-04-14: grid_fuser free-ray endpoint 덮어쓰기 버그 수정

### 버그 요약

`process_scan`의 파라메트릭 ray sampling에서 `t_vals`의 마지막 값(0.95)이 endpoint 셀과
동일한 그리드 셀로 반올림(int truncation)되는 경우, occupied 업데이트 직후 같은 셀이
free 업데이트되어 log-odds가 순 감소한다. 스캔이 반복될수록 장애물 셀이 clamp_min으로
수렴해 A*가 해당 장애물을 영원히 인식하지 못한다.

### 수치 분석

**버그 발생 조건:**

```
int(sx_g + 0.95 × dx) == int(sx_g + dx)
⟺ 0.05 × |dx| < 1
⟺ |dx| < 20 셀
⟺ 거리 < 20 × 0.2m = 4.0m  (resolution=0.2m 기준)
```

**스캔 1회당 log-odds 변화 (버그 발생 시):**

```
Δ = l_hit + l_miss = +0.405 + (−0.847) = −0.442
```

순 업데이트가 음수 → 반복 스캔으로 log-odds가 clamp_min=-2.0에 수렴.
`occupied_thresh`(log-odds 기준 +0.080)에 영원히 도달 불가.

**거리별 발생 여부 (resolution=0.2m, 예시: dx=-9셀=1.8m):**

| 거리  | \|dx\| 셀 | t=0.95 위치              | endpoint 셀 | 버그? |
|-------|-----------|--------------------------|-------------|-------|
| 1.8m  | 9         | int(90 + 0.95×(−9))=81   | 81          | ✅ BUG |
| 2.0m  | 10        | int(90 + 0.95×(−10))=80  | 80          | ✅ BUG |
| 3.8m  | 19        | int(90 + 0.95×(−19))=72  | 71          | ✅ BUG |
| 4.0m  | 20        | int(90 + 0.95×(−20))=71  | 70          | ✓ 안전 |
| 5.0m  | 25        | int(90 + 0.95×(−25))=66  | 65          | ✓ 안전 |

→ **4m 이내 장애물만 감지 안 되는 현상의 근본 원인.**

### 수정 방법

두 가지 마스크를 `in_bounds`에 추가한다. **파일:** `px4_nav_perception/px4_nav_perception/nodes/grid_fuser.py`

**수정 1 — `is_endpoint`: endpoint 셀을 free-ray에서 명시적 제외**

```python
# [수정 1] endpoint 셀을 free-ray 업데이트에서 제외
# 원인: |dx| < 20셀(4m) 구간에서 int(sx_g + 0.95*dx) == int(sx_g + dx) 가 성립,
#       occupied 셀에 l_miss 가 동시 적용되어 net = l_hit + l_miss = -0.44 < 0
#       → 4m 이내 장애물의 log-odds 가 스캔마다 감소 → 영구 미감지
is_endpoint = (cx_all == ex_arr[:, None]) & (cy_all == ey_arr[:, None])
```

**수정 2 — `is_dup`: beam 내 연속 중복 셀 제거**

`max_steps`는 `r_cap`(최대 거리) 기준 고정(=34)이므로, 짧은 beam(예: 9셀/1.8m)에서는
t 스텝 하나의 이동량 ≈ 0.26셀 → 같은 셀에 3~4회 `l_miss` 누적 → free evidence 3~4배 과다.

```python
# [수정 2] beam 내 연속 중복 셀 제거
cx_prev = np.concatenate([cx_all[:, :1], cx_all[:, :-1]], axis=1)
cy_prev = np.concatenate([cy_all[:, :1], cy_all[:, :-1]], axis=1)
is_dup = (cx_all == cx_prev) & (cy_all == cy_prev)
is_dup[:, 0] = False  # 첫 스텝은 비교 대상 없음, 항상 유효
```

**최종 `in_bounds`:**

```python
in_bounds = (
    (cx_all >= 0) & (cx_all < self.width)
    & (cy_all >= 0) & (cy_all < self.height)
    & ~is_endpoint
    & ~is_dup
)
```

### 왜 `is_endpoint` 마스크인가

| 방법 | 설명 | 단점 |
|------|------|------|
| `t_max` 축소 (0.95 → 더 작게) | t_max를 낮춰 endpoint 셀 도달 방지 | 거리에 따라 여전히 같은 셀에 truncation될 수 있어 완전 보장 안 됨 |
| **`is_endpoint` 마스크** | free-ray 결과 배열과 endpoint 셀 좌표를 직접 비교해 제외 | 브로드캐스트 연산 추가 (메모리 O(n_beams × max_steps)), 성능 미미한 증가 |

→ `is_endpoint` 방식이 모든 방향·거리·해상도 조합에서 정확히 동작함.

**상태:** ✅ 적용 완료 (2026-04-14)
---

## 참고: `grid_fuser` 파라미터 (`px4_nav_perception/config/grid_fuser.yaml`)

LiDAR `/scan`을 `odom` 격자에 누적해 `/local_grid`(`nav_msgs/OccupancyGrid`)로 내보내는 노드. `use_log_odds: true`일 때는 셀마다 log-odds를 유지하고, publish 시점에 **sigmoid**로 점유 확률 `prob`를 만든 뒤 `occupied_thresh`와 비교해 `100`(점유) / `0`(자유) / `-1`(미탐색)으로 export한다. (`occupied_thresh`는 **log-odds 숫자가 아니라 prob**와 비교.)

### 격자·좌표

| 파라미터 | 역할 | 값을 키우면 | 값을 줄이면 |
|----------|------|-------------|-------------|
| `resolution` | 셀 한 변 길이(m). | 셀 수(=width×height) 동일이면 **맵 범위가 커짐**; 같은 공간이면 **더 잘게 쪼개서** 얇은 장애물 표현에 유리, **CPU·메모리 부담 증가**. | 맵이 더 **거칠어짐**; 얇은 벽이 **빔 한 칸 사이로 빠지기 쉬움**. |
| `width`, `height` | 셀 개수. | 격자 **넓어짐**(맵 커버 범위↑, `origin`·`resolution`과 함께 결정). | **좁아짐**; 드론이 밖으로 나가면 **맵 밖** 처리. |
| `frame_id` | 격자·TF 기준 프레임(보통 `odom`). | — | — |
| `robot_frame` | 롤링 윈도우 시 로봇 위치 조회용(보통 `base_link`). | `rolling_window: false`면 **원점 고정**이라 영향 작음. | |
| `rolling_window` | `true`면 로봇 중심으로 **원점 이동**. | (토글) 고정 맵 대신 **주변만** 유지. | `false`면 **월드 고정** origin. |
| `origin_x`, `origin_y` | `rolling_window: false`일 때 격자 **남서 모서리**의 월드 좌표(m). | `resolution`·`width`/`height`와 함께 **커버 사각형**이 평행이동. | |

### 토픽·스캔·거리

| 파라미터 | 역할 | 값을 키우면 | 값을 줄이면 |
|----------|------|-------------|-------------|
| `lidar_topic` | 구독할 `LaserScan` 이름. | — | — |
| `publish_rate` | `/local_grid`를 퍼블리시하는 Hz(타이머). | 격자 **갱신이 자주** 나감, **CPU 부담**·`decay` 적용 횟수↑. | **느린 맵**; 오래된 점유가 화면에 남음. |
| `max_range` | 스캔에서 쓸 **최대 거리 상한**(m). Gazebo `range_max`보다 작게 두는 경우 많음. | **더 먼** 장애물까지 반영 시도. **끝을 occupied로 안 찍는** 구간(`max_range - epsilon`)도 길어짐. | **가까운** 쪽만 사용; 먼 쪽은 잘림. |
| `max_range_no_hit_epsilon` | 빔 거리가 `max_range - epsilon` **이상**이면 “끝에 막힘이 없다”고 보고 **끝 셀을 occupied로 찍지 않음**. | **더 넓은** 거리 구간이 **max로 처리**되어 끝 장애물 표시가 **줄어듦**(가짜 벽 방지 쪽). | **좁은** 구간만 max 처리 → **먼** 막힘이 **끝 점유로** 잡힐 수 있음(노이즈). |

### Log-odds (`use_log_odds: true`일 때만 의미 있음)

`l_hit = log(p_hit/(1-p_hit))`, `l_miss = log(p_miss/(1-p_miss))`. 히트 셀에는 `l_hit` **가산**, 빔이 지나간 자유 셀에는 `l_miss` **가산**(음수). 코드는 `prob_hit ∈ (0.5, 1)`, `prob_miss ∈ (0, 0.5)`만 허용.

| 파라미터 | 역할 | 값을 키우면 | 값을 줄이면 |
|----------|------|-------------|-------------|
| `prob_hit` | **히트** 한 번이 점유 쪽으로 주는 log-odds 증분의 **강도**에 해당하는 확률 파라미터. | **한 번 맞을 때** occupied로 가까워짐이 **빨라짐**. 너무 높이면 **노이즈 한 방**에도 점유. | **히트**가 **여러 번** 쌓여야 점유로 보임; **얇은 벽**이 더 안 잡힐 수 있음. |
| `prob_miss` | **자유 빔**이 지나간 셀에 쌓이는 **음수** 증분의 강도. 0.5에 가까울수록 `l_miss`가 0에 가깝다. | **자유 갱신이 약해짐** → 점유가 **덜 지워짐**. 얇은 벽이 **다른 빔의 자유 샘플**에 지워지는 문제 완화에 유리할 수 있음. | **자유 갱신이 강해짐** → **고스트 점유**를 빨리 지우지만, **얇은 벽 칸**도 **자유 빔**에 잘 지워질 수 있음. |
| `clamp_min`, `clamp_max` | 셀 log-odds의 **하한·상한**. | 범위가 넓을수록 **극단적 확신** 허용. | **포화**가 빨라지거나 느려짐; `clamp_max`를 낮추면 **강한 점유**가 어려워질 수 있음. |
| `occupied_thresh` | export 시 **점유 확률** `prob`가 이 값 **이상**이면 셀 `100`. (**log-odds와 직접 비교 아님.**) **`use_log_odds: true`이면 반드시 > 0.5** (중립 log-odds=0 → prob=0.5; thresh≤0.5면 **미관측 셀 전부 occupied**로 나가 맵이 검게 보이고 A*가 망가짐). | **더 쉽게** occupied(검은 칸)로 나옴; **고스트 장애물** 증가 가능. | **더 어렵게** occupied; **얇은 벽**이 안 보일 수 있음. |
| `decay_rate` | 매 `publish_grid`마다 log-odds를 **0(중립)** 쪽으로 곱하는 비율(0이면 비활성). | **오래된 지도**가 빨리 사라짐; **지나간 벽**도 잘 안 남음. | **0**에 가까우면 **증거가 오래 유지**; 지도가 **끈적**해질 수 있음. |

### Export 상수

| 파라미터 | 역할 | 값을 키우면 | 값을 줄이면 |
|----------|------|-------------|-------------|
| `occupied_value`, `free_value`, `unknown_value` | `OccupancyGrid.data`에 넣는 **int8** 값. RViz Map `map` 색상표는 보통 100→검정, 0→밝음, -1→회색. | **플래너 `occ_threshold`**(예: 50)와 **일관**되게만 맞추면 됨; 임의 변경 시 A*·시각화와 불일치. | |

### 튜닝 감각 (짧게)

- **벽이 안 잡힘:** `prob_hit`↑, `occupied_thresh` 소폭↓, `decay_rate`↓, 필요 시 `resolution`↓(셀 작게).
- **빔이 벽을 잘 안 맞추는데 다른 방향 자유가 벽을 지움:** `prob_miss`↑(0.5에 가깝게).
- **max_range 근처만 보이는 장애물이 안 잡힘:** `max_range`·`max_range_no_hit_epsilon` 관계 확인.

---

#### bag 녹화 명령 (추가 topic 포함)

```bash
ros2 bag record \
  /clock \
  /debug/policy_action_raw \
  /mavros/setpoint_velocity/cmd_vel_unstamped \
  /mavros/local_position/odom \
  /mavros/local_position/pose \
  /mavros/local_position/velocity_local \
  /mavros/state \
  /mavros/imu/data \
  /scan \
  /local_grid \
  /planner/goal \
  /planner/homing_done \
  /tf /tf_static
```

**`/clock` 추가 목적:** Gazebo sim real-time factor 추적.
`replan_period=1.0s`(sim time) 기준 A* 호출 빈도가 실제로는 sim 부하에 따라 달라짐.
`/clock` timestamp와 wall time을 비교하면 RTF = Δsim / Δwall 를 후분석으로 계산 가능.

```python
# RTF 후분석 예시 (python3)
import sqlite3, struct
conn = sqlite3.connect('bag.db3')
c = conn.cursor()
c.execute("""SELECT m.timestamp, data FROM messages m JOIN topics t ON m.topic_id=t.id
             WHERE t.name='/clock' ORDER BY m.timestamp""")
rows = c.fetchall()
# wall_ts = m.timestamp (nanosec, 시스템 clock)
# sim_ts  = rosgraph_msgs/Clock.clock (sec@4 + nanosec@8)
wall_prev = rows[0][0]; sim_prev = struct.unpack_from('<I', rows[0][1], 4)[0] * 1e9 + struct.unpack_from('<I', rows[0][1], 8)[0]
for (wall_ts, d) in rows[100::100]:
    sim_sec = struct.unpack_from('<I', d, 4)[0]; sim_nsec = struct.unpack_from('<I', d, 8)[0]
    sim_ts = sim_sec * 1e9 + sim_nsec
    rtf = (sim_ts - sim_prev) / max(1, wall_ts - wall_prev)
    print(f"RTF={rtf:.3f}")
    wall_prev, sim_prev = wall_ts, sim_ts
```

RTF < 0.8 구간이 잦으면 A* 부하 외 Gazebo 렌더링·물리 연산 병목을 의심.

### 제한사항 및 주의사항

| 항목 | 내용 |
|------|------|
| 발행 조건 | `nav_mode="ppo"` 또는 `"hybrid"`이고 `_infer()` 호출 시에만 발행 |
| 미발행 구간 | `nav_mode="astar"`, `sensors_ok()=False`, goal 미수신, emergency brake 발동 시 topic 끊김 |
| 단위 주의 | `linear.x`, `angular.z` 모두 **[-1,1] normalized**. m/s, rad/s 아님 |
| scaled 값 환산 | `v_cmd = v_min + (linear.x + 1.0) * 0.5 * (v_max - v_min)` |
| | `yaw_rate = angular.z * max_yaw_rate` |
| F1 판단 기준 | raw action이 saturation(-1 또는 1)에 자주 걸리거나 방향이 이상한 경우 |
| F2 판단 기준 | raw action은 정상인데 `/mavros/setpoint_velocity/cmd_vel_unstamped`가 다른 경우 |
| 실제 PX4 추종 오차 | `/mavros/local_position/odom` twist와 cmd_vel 비교로 확인 (별도 topic 불필요) |

### 의사결정

- `/debug/policy_action_raw` 1개만 추가 (`/debug/applied_action` 추가 안 함)
- `Twist` 대신 `TwistStamped` 사용 (타임스탬프 필수)
- bag 녹화 시 `/tf`, `/tf_static` 반드시 포함 (이전 누락 재발 방지)

**상태:** 계획 (미적용)

---

## [적용 완료] 2026-04-12: nav_mode 아키텍처 리팩토링

### 배경
Phase 3에서 `use_hybrid=true` + PPO 항상 실행으로 변경 후, obstacle_course 테스트에서
드론이 goal에 도달하지 못하고 방황하는 문제 발생.
원인: PPO가 최종 goal 기준으로 학습됐으나 배포 시 A* lookahead wp 기반으로 obs 입력 → obs 분포 불일치.
→ 학습 환경 수정 + nav_mode 3분할 구조로 리팩토링.

### 변경 내용

#### 1. `rl_offboard_node.py`
| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 제어 분기 파라미터 | `use_hybrid: bool` | `nav_mode: "hybrid"\|"astar"\|"ppo"` |
| policy 로딩 | 항상 로드 | `nav_mode="astar"` 시 불필요, 생략 |
| policy 파일 결정 | `policy_ts.pt` 고정 | `nav_mode`별 default 파일 자동 선택 |
| `_tick()` 하단 | 단일 분기 블록 | `_tick_hybrid_mode()` / `_tick_astar_mode()` / `_tick_ppo_mode()` 분리 |
| 삭제된 상태 변수 | `_prev_use_rl`, `_t_rl_entered`, `_t_rl_fallback_until` | 제거 |
| 삭제된 파라미터 | `rl_trigger_front_enter`, `rl_trigger_front_exit`, `rl_min_dwell_sec` | 제거 |

**nav_mode별 동작:**
- `"hybrid"`: A* 경로 생성 → lookahead wp를 obs 입력으로 PPO 항상 실행. `policy_hybrid_ts.pt` / `obs_norm_hybrid.json` 사용 (재학습 후 배포).
- `"astar"` : A* 단독. PPO 없음. policy 로드 불필요. 비교 실험용.
- `"ppo"`   : PPO 단독. 최종 goal 방향 / world_size 정규화. `policy_ts.pt` / `obs_norm.json` 사용 (기존 가중치). 비교 실험용.

**policy 파일 결정 로직:**
```
policy_path 파라미터 비어 있을 때:
  nav_mode="hybrid" → models/policy_hybrid_ts.pt, models/obs_norm_hybrid.json
  nav_mode="ppo"    → models/policy_ts.pt,        models/obs_norm.json
  nav_mode="astar"  → 로드 없음
```

#### 2. `config/` 파일 정리
| 파일 | 변경 |
|------|------|
| `params.yaml` | `use_hybrid: true` → `nav_mode: "hybrid"` |
| `params_astar.yaml` | 신규 생성 (`nav_mode: "astar"`) |
| `params_ppo.yaml`   | 신규 생성 (`nav_mode: "ppo"`) |
| `params_rl_only.yaml`    | 삭제 (→ `params_ppo.yaml`로 대체) |
| `params_global_only.yaml` | 삭제 (→ `params_astar.yaml`로 대체) |

#### 3. `rl/env_2d_nav.py` — 학습 환경 obs 수정
| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| goal_rel 기준 | 최종 goal / world_size(20m) | A* lookahead wp / lookahead_dist(3m) |
| A* 경로 | 없음 | `reset()` 시 1회 계산 (`_compute_astar_path`) |
| wp 전진 | 없음 | `step()` 에서 `_advance_wp()` 호출 |

**재학습 명령:**
```bash
cd /home/akswosla/PX4-Autopilot/rl && source .venv/bin/activate
python3 train_ppo_2d.py \
  --total_timesteps 10000000 \
  --learning_rate 1e-4 \
  --clip_range 0.2 \
  --n_epochs 10 \
  --eval_freq 500000 \
  --eval_runs 200 \
  --eval_obstacles 0 5 10 \
  --save_best \
  --do_eval
```

**재학습 후 배포:**
```bash
# export
python3 export_policy.py \
  --model ./models_ppo_2d/best_model \
  --vecnorm ./models_ppo_2d/vecnormalize.pkl \
  --output_dir ./export_deploy

# ROS2 패키지에 복사 (이름 변경 필수)
cp export_deploy/policy_hybrid_ts.pt  /home/dev/PX4-Autopilot/px4_ws/src/px4_rl_offboard_mavros/models/policy_hybrid_ts.pt
cp export_deploy/obs_norm_hybrid.json /home/dev/PX4-Autopilot/px4_ws/src/px4_rl_offboard_mavros/models/obs_norm_hybrid.json
```

## [적용 계획] 2026-04-12: 수정 #1~9 적용 순서 및 방법

### 의존성 분석

```
#2 (arm mask) ──────────────────────────────────────────▶ #3 (brake dist 0.5m)
                                                              ↑ lidar_range_min_override=0.35로 front body 자기감지 제거 후 상향 가능
                                                              (arm mask는 emergency brake와 무관; ±59°암은 ±30° brake 섹터 밖)
#4 (grid 0.2m, array.array, inflation 분리) ─────────────▶ #9 (MultiThreadedExecutor)
                                                              ↑ A* 실행 시간 단축 후 threading 적용이 효과적
#9 (path_lock / RLock) ──────────────────────────────────▶ #1 (PPO always on)
                                                              ↑ lock 구조 먼저 확보 후 제어 로직 변경
```

독립 적용 가능: #5, #6, #7, #8 (다른 수정과 충돌 없음)

---

### 적용 단계

#### Phase 1 — 인프라 기반 정비 (위험도 낮음, 행동 변화 없음) ✅ 2026-04-12 적용 완료
**목표:** 그리드와 타이밍 기반 수정. 이후 모든 테스트의 전제 조건.

| 순서 | 번호 | 내용 | 파일 |
|------|------|------|------|
| 1-A | #4a | `msg.data = array.array('b', ...)` 직렬화 교체 | `grid_fuser.py` |
| 1-B | #4b | 해상도 0.1m → 0.2m, 441×441 → 221×221, origin ±22.1 | `grid_fuser.yaml` |
| 1-C | #4c | publish 시 inflation 제거 (`_inflate_grid` 호출 주석 처리) | `grid_fuser.py` |
| 1-D | #5  | `time.time()` → `get_clock().now()`, odom/scan timeout sim clock화 | `rl_offboard_node.py`, `params.yaml` |
| 1-E | #4d | `process_scan` Python 이중 루프 → numpy 벡터화 (Bresenham 제거) | `grid_fuser.py` |

**Phase 1 적용 결과 (2026-04-12):**
| 항목 | 적용 전 | 적용 후 |
|------|---------|---------|
| `/local_grid` | 2.88 Hz | ~3.8 Hz |
| `/scan` | 7.4 Hz | ~8.0 Hz |

`/local_grid` 5Hz 미달 원인: `np.add.at` (free cell 업데이트)가 중복 인덱스 처리를
내부 순차 루프로 실행. `np.bincount`로 교체 시 추가 개선 가능하나,
현재 3.8~4.0 Hz는 `replan_period: 2.0s` 기준 grid 지연 최대 263ms → 실용상 허용 범위.

---

#### Phase 2 — 안전 계층 수정 (행동 변화 있음, 격리 테스트) ✅ 2026-04-12 적용 완료
**목표:** LiDAR 자기감지 제거 후 emergency brake 임계값 상향. 반드시 #2 먼저.

| 순서 | 번호 | 내용 | 파일 |
|------|------|------|------|
| 2-A | #2  | `_filter_arm_angles()` 구현, `_cb_scan`에서 `_filtered_ranges` 캐싱, `lidar_range_min_override: 0.35 유지` (arm mask는 ±59°/±123°만, 정면 body 자기감지 0.24~0.31m는 min_override가 담당) | `rl_offboard_node.py`, `params.yaml` |
| 2-B | #3  | emergency brake 후퇴 추가, `emergency_brake_dist: 0.2 → 0.5`, brake count 카운터 | `rl_offboard_node.py`, `params.yaml` |

**검증 방법:**
```bash
# 제자리 회전 명령으로 자기감지 빔 필터 확인
ros2 topic pub /mavros/setpoint_velocity/cmd_vel_unstamped \
  geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}" --rate 20

# RViz2 /scan 시각화에서 45°/135°/225°/315° 방향 빔이 max_range(10m)로 표시되는지 확인
```

**검증 결과 (2026-04-12):**
- RViz2 `/scan` 시각화: rotor arm 방향(±59°, ±123°) 빔이 10m로 마스킹 정상 확인 ✅
- 제자리 회전(yaw_rate 0.5 rad/s) 중 `emergency_brake` 미발동 확인 ✅

---

#### Phase 3 — 제어 동작 변경 (핵심, 단계적 적용) ✅ 2026-04-12 적용 완료
**목표:** 가장 큰 행동 변화. #8 먼저(빠른 효과, 낮은 위험) → #1(근본 구조 변경).

| 순서 | 번호 | 내용 | 파일 |
|------|------|------|------|
| 3-A | #8  | heading alignment 완전 정지 → 오차 비례 감속 | `rl_offboard_node.py` |
| 3-B | #1  | `_build_obs` lookahead goal, `_tick` binary switch 제거, `_tick_replan` 단순화 | `rl_offboard_node.py` |

**#8 먼저 적용하는 이유:** #1보다 변경 범위가 좁고 효과가 명확함
(88~93초 정지 → 1초 이내 정렬). #1 이전에 제어 흐름의 기본 안전성 확인.

**검증 방법:**
```bash
# 드론이 waypoint 방향 전환 시 제자리 회전 없이 부드럽게 이동하는지 확인
# cmd_vel 발행 주파수 확인 (목표 20Hz, 허용 18Hz 이상)
# - setpoint_rate_hz: 20.0 → 이론값 20Hz
# - PX4 OFFBOARD 최소 요구: 2Hz / 권장: 10Hz 이상
# - 18Hz 미만: A* 재계획(replan_period: 2.0s)이 메인 타이머 블로킹 의심 → Phase 5(#9) 필요
# - 15Hz 미만: PX4가 OFFBOARD 모드 이탈 위험
ros2 topic hz /mavros/setpoint_velocity/cmd_vel_unstamped
```

**검증 결과 (2026-04-12):**

`/mavros/setpoint_velocity/cmd_vel_unstamped` 발행 주기:
```
average rate: ~15 Hz  (목표 20Hz 미달)
```
- **원인 1 (초기 스파이크):** torch JIT 첫 호출 시 0.481s 컴파일 지연 → 노드 기동 직후 일시적 드롭
- **원인 2 (지속적 미달):** CPU inference 오버헤드 + SingleThreadedExecutor에서 scan 콜백(10Hz)·A* 재계획(0.5Hz)이 메인 타이머(20Hz)를 간헐적으로 블로킹
- 15Hz는 PX4 OFFBOARD 모드 이탈 임계값(2Hz)보다 충분히 높아 즉각적 위험 없음
- **근본 해결책:**
  - `__init__`에서 dummy inference로 torch JIT warmup (초기 스파이크 제거) — 미적용
  - Phase 5 (#9 MultiThreadedExecutor) ✅ 2026-04-12 적용 완료 → 재측정 필요

scan/odom 발행 주기:
```
scan:  ~10 Hz
odom:  ~22 Hz (stamp는 100ms 단위 갱신)
```

---

#### Phase 4 — 데이터 정합 개선 (미세 조정, 위험도 낮음) ✅ 2026-04-12 적용 완료
**목표:** 타임스탬프 및 관측 정확도 개선. 제어 동작에 미치는 영향 작음.

| 순서 | 번호 | 내용 | 파일 |
|------|------|------|------|
| 4-A | #6  | TF 조회: scan stamp 1차(20ms) → ExtrapolationException 시 최신 TF fallback | `grid_fuser.py` |
| 4-B | #7  | `_build_obs` 내 scan-odom dead reckoning (dt > 0.01s 시 velocity 보정) | `rl_offboard_node.py` |

**검증 방법:**
```bash
# 4-A: lookup_tf() 실패는 로그 없이 None 반환 → 로그로 직접 확인 불가
# ghost obstacle(스캔-그리드 정렬 오차로 잘못 생긴 점유 셀) 감소는 RViz2로 정성적 확인
#   /local_grid 시각화 후 실제 장애물 없는 구역에 점유 셀이 나타나는지 관찰
ros2 topic hz /local_grid   # grid 발행 주기 확인 (5Hz 근접)

# 4-B: scan/odom 타임스탬프 차이 확인 (dt > 0.01s 시 dead reckoning 보정 적용)
ros2 topic echo /mavros/local_position/odom --field header.stamp --once
ros2 topic echo /scan --field header.stamp --once
# colcon build 불필요 (Python 파일만 수정, symlink-install)
```

**검증 결과 (2026-04-12):**

4-A (`/local_grid` 발행 주기):
```
average rate: ~3.4 Hz  (목표 5Hz 미달)
min: -0.605s  max: 1.904s  std dev: ~0.20s
```
- 3.4Hz 미달 원인: `lookup_tf()` fallback timeout(0.1s)이 `grid_fuser.py`의 SingleThreadedExecutor 내에서 scan 콜백을 간헐적으로 블로킹
- min 음수(-0.605s): Gazebo sim clock 역행 또는 리셋 시 타임스탬프 역전 (동작 이상 아님)
- max 1.904s: TF lookup 연속 실패로 인한 블로킹 추정
- **`grid_fuser.py`는 별도 노드 → `rl_offboard_node.py`의 Phase 5와 무관**
- **추가 수정 (2026-04-12):** fallback `lookup_tf()` timeout `0.1s → 0.0s` — `rclpy.time.Time()`으로 최신 TF 조회 시 버퍼에 있으면 즉시 반환, 없으면 즉시 실패. 블로킹 불필요. scan 콜백(100ms 주기)에서 0.1s 블로킹이 콜백 큐 누적의 근본 원인이었음 → `/local_grid` 3.4Hz 개선 기대

4-B (scan/odom 타임스탬프 특성):
```
odom stamp: 100ms 단위 갱신 (22Hz 수신이나 stamp 해상도 100ms, 동일 stamp 2~3회 반복)
scan stamp: 100ms 간격 (10Hz), 1ms 해상도, 메시지마다 고유 stamp
```
- `dt = scan_t - odom_t` → odom stamp 해상도 100ms로 인해 거의 항상 `abs(dt) > 0.01s` 조건 충족
- → **dead reckoning 보정이 매 tick마다 정상 활성화됨** ✅

---

#### Phase 5 — 스레딩 (마지막, 모든 로직 안정화 후) ✅ 2026-04-12 적용 완료
**목표:** Phase 1~4의 모든 수정이 안정화된 후 threading 적용.
로직 버그가 남아있는 상태에서 threading 추가 시 디버깅 난이도 급증.

| 순서 | 번호 | 내용 | 파일 |
|------|------|------|------|
| 5-A | #9  | `threading.RLock()` 추가, `MultiThreadedExecutor(num_threads=3)`, callback_group 분리 | `rl_offboard_node.py` |

**적용 내용:**
- `MutuallyExclusiveCallbackGroup` × 2: `_cb_group_tick` (20Hz), `_cb_group_replan` (A* 재계획)
- `ReentrantCallbackGroup` × 1: `_cb_group_sensor` (odom/scan/grid/goal 콜백)
- `threading.RLock()` (`_path_lock`): `_path_xy` / `_path_wp_idx` 쓰기(`_plan_path`) ↔ 읽기(`_tick`) race condition 방지
- `_path_current_wp`, `_path_lookahead_wp`, `_path_advance_wp`: `_path_lock` 추가
- `_plan_path` path 쓰기 2곳: `with self._path_lock:` 추가, 죽은 코드(`_t_rl_fallback_until`) 제거
- `main()`: `rclpy.spin(node)` → `MultiThreadedExecutor(num_threads=3).spin()`
- `_build_obs` 외부 coarse lock 없음 (fine-grained `_path_lock`만 사용, A*와 PPO inference 병렬 실행 허용)

**검증 방법:**
```bash
# A* 실행 중 cmd_vel 주파수 확인 (Phase 3 ~15Hz → 20Hz 근접 목표)
ros2 topic hz /mavros/setpoint_velocity/cmd_vel_unstamped
# /local_grid 발행 주기 확인 (Phase 5와 무관 — grid_fuser.py 별도 개선 필요)
ros2 topic hz /local_grid
# 데드락 발생 여부: 노드가 응답 없이 멈추지 않는지 확인 (수분 이상 연속 실행)
```

---

### 전체 적용 우선순위 요약

```
Phase 1 (#4, #5)  →  Phase 2 (#2, #3)  →  Phase 3 (#8, #1)  →  Phase 4 (#6, #7)  →  Phase 5 (#9)
  인프라               안전                   제어 동작               데이터 정합              스레딩
  위험도: 낮음          위험도: 중간             위험도: 높음              위험도: 낮음              위험도: 중간
  즉시 적용             Phase 1 후              Phase 2 후              Phase 3 후              Phase 4 후
```

### 각 Phase 후 공통 확인 사항
- `ros2 topic hz /mavros/setpoint_velocity/cmd_vel_unstamped` → 20Hz 근접 유지
- `ros2 topic hz /local_grid` → 5Hz 근접 유지 (Phase 1 이후)
- RViz2 A* 경로가 장애물을 회피하는지 시각 확인
- 드론이 목표 지점까지 도달하는지 E2E 테스트

---

> **TODO (Phase 1~9 완료 후):** `np.add.at` → `np.bincount` 교체
> ```python
> flat_idx = cy_all[in_bounds] * self.width + cx_all[in_bounds]
> counts = np.bincount(flat_idx, minlength=self.height * self.width)
> self.grid_log += counts.reshape(self.height, self.width).astype(np.float32) * self.l_miss
> np.clip(self.grid_log, self.clamp_min, self.clamp_max, out=self.grid_log)
> ```

**검증 방법:**
```bash
# grid publish 실제 주기 확인 (5Hz에 근접해야 함)
ros2 topic hz /local_grid

# RViz2에서 그리드 셀 크기가 0.2m로 변경됐는지 확인
# odom/scan timeout 로그가 사라지는지 확인
```
---

## [계획] 2026-04-11: bag 분석 9가지 문제점 수정 계획

### #1 — 권한 분리 (보완 적용): PPO 항상 실행 + lookahead goal 입력

**변경 파일:** `rl_offboard_node.py`

**배경:**
- 현재 구조는 `_tick()`에서 `use_rl` flag로 global planner / RL을 binary switch → RL이 꺼진 구간에서 PPO 정책이 완전히 비활성
- `_build_obs()`에서 goal 방향을 final goal 기준으로 계산 → RL이 켜진 순간 먼 goal을 보고 방향 오류 발생 가능
- `_tick_replan()`에서 `_prev_use_rl` 조건으로 재계획 억제 → 불필요한 상태 의존

**변경 내용:**

| 함수 | 변경 내용 |
|------|-----------|
| `_build_obs()` | goal 입력을 final goal 대신 A* lookahead waypoint로 교체. `LOOKAHEAD_DIST=3.0m`로 정규화. `use_hybrid`가 False이거나 path 없을 때는 기존 final goal 사용. |
| `_tick()` | binary `use_rl` switch 제거. emergency brake 이후 PPO를 항상 실행. global planner의 waypoint 진행(`_path_advance_wp`)도 항상 실행. spinning trap 감지 및 boundary turn 로직 유지. |
| `_tick_replan()` | `_prev_use_rl` 조건 제거. `use_hybrid + odom + goal + not goal_reached` 조건만으로 단순화. |

**예상 효과:**
- RL/global planner 전환 경계에서 goal 방향 불연속 제거
- 재계획 로직 단순화 → 상태 버그 감소
- PPO가 항상 실행되므로 장애물 회피 반응성 유지

**2단계 (재학습 필요 시):**
PPO 학습 환경에서 goal을 final goal → lookahead waypoint로 교체하여 재학습.
단거리 목표 추종 + 장애물 회피를 동시에 최적화한 정책 획득 가능.

**상태:** ✅ 적용 완료 (Phase 3-B, 2026-04-12)

---

### #2 — LiDAR 자기 암 감지 필터: 거리 기반 → 각도 마스크 전환

**변경 파일:** `rl_offboard_node.py`, `params.yaml`

#### 현재 문제 구조

```
lidar_range_min_override: 0.35  → _build_obs (PPO 관측) 에만 적용
emergency_brake: use_override=False → 자기감지 빔(~0.25m) 그대로 진입
  → emergency_brake_dist 를 0.25m 이상으로 올리면 매 tick 발동
  → 현재 0.2m 로 고정: 실제 0.2m 앞 장애물은 감지 불가
```

거리 기반 필터의 근본 한계: 0.25m 자기감지 빔과 0.21m 실제 장애물을 거리만으로 구분 불가능.

#### 연구자 관점 검토 및 보완

**① 좌표계 검증 필요**
Iris 암 방향(45°/135°/225°/315°)은 드론 body frame 기준.
rplidar는 `xyz="0 0 0.25" rpy="0 0 0"`으로 마운트되어 body_frame과 정렬.
따라서 scan 의 `angle=0` = 드론 전방(+x) → 암 각도를 그대로 사용 가능.
단, 실제 시뮬레이션에서 scan angle=0 이 전방인지 **첫 실행 시 RViz2로 검증** 필요.

**② 성능: 원본 코드의 이중 루프 → numpy 벡터화로 대체**
원본 구현(`for i: for arm:` 이중 루프 + per-beam atan2)은 180빔 × 4암 = 720회 반복.
20Hz tick에서 매 스캔마다 실행되므로 벡터화가 필수.

```python
def _filter_arm_angles(self, ranges: np.ndarray) -> np.ndarray:
    """로터 암 방향 빔을 max_range로 대체 (numpy 벡터화)."""
    if not hasattr(self, 'arm_mask_angles_deg') or self.scan is None:
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
```

**③ 마스크 적용 위치: scan 수신 직후 1회만**
`_build_obs`와 `_dist_in_sector` 양쪽에서 각각 필터를 적용하면 중복 연산.
대신 scan 콜백에서 `self._filtered_ranges`를 캐싱하고 이후 모든 함수가 공유:

```python
def _cb_scan(self, msg):
    self.scan = msg
    raw = np.array(msg.ranges, dtype=np.float32)
    self._filtered_ranges = self._filter_arm_angles(raw)  # 1회만 실행
```

`_dist_in_sector`, `_build_obs` 내부에서 `self.scan.ranges` 대신
`self._filtered_ranges` 사용 → 일관성 확보 + 연산 절감.

**④ arm_mask_half_width_deg 범위 권고**
2° 빔 해상도(180빔) 기준: ±10° = 빔 10개 마스크, ±15° = 빔 15개 마스크.
너무 넓으면 실제 대각선 장애물 blind spot 발생.
Iris 암 폭 ~3cm, 25cm 거리 → 실제 각도 폭 ≈ ±7°.
**±10° 권장** (원안 12°는 허용 범위이나 10°가 더 보수적).

**⑤ emergency_brake_dist 재설정**
암 마스크 적용 후 자기감지 빔이 제거되므로:
- `emergency_brake_dist: 0.2 → 0.5m` 상향 가능 (원안 0.5~0.7m 중 보수적 하한 채택)
- 0.5m는 10Hz LiDAR + 최대 forward 속도 기준 1프레임(0.1s) 내 정지에 충분한 여유

**⑥ lidar_range_min_override 역할 변경**
암 마스크 도입 후 자기감지 제거 역할은 마스크가 담당.
`lidar_range_min_override`는 LiDAR 물리적 최소 측정 거리(~0.2m) 이하
노이즈 필터로만 유지 (`0.35 → 0.2` 로 축소 또는 scan.range_min 그대로 사용).

#### 변경 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 자기감지 필터 방식 | 거리 임계값 (`range_min_override=0.35`) | 각도 마스크 (±10°, 4방향) |
| 필터 적용 범위 | `_build_obs` 전용 | scan 콜백 캐싱 → 전체 일관 적용 |
| `emergency_brake_dist` | 0.2m (자기감지 회피 위해 고정) | 0.5m (안전 여유 확보) |
| `lidar_range_min_override` | 0.35m (전방향 자기감지 방어) | 0.35m 유지 (empty world 테스트에서 0.2로 낮추면 front body 자기감지 0.24~0.31m 재발 확인) |
| 성능 | 이중 루프 O(N×M) | numpy 벡터화 O(N) |

#### params.yaml 추가 파라미터

```yaml
arm_mask_angles_deg: [59.4, 123.0, 237.0, 301.0]  # Iris 암 방향 (iris.sdf 실측, atan2 기준)
                                                   # rotor_0(0.13,-0.22)=-59.4° → 301°
                                                   # rotor_1(-0.13,0.20)=123°
                                                   # rotor_2(0.13,0.22)=59.4°
                                                   # rotor_3(-0.13,-0.20)=-123° → 237°
arm_mask_half_width_deg: 12.0                      # ±12° (암 거리 0.25m 기준 실폭 ±7° + 여유)
emergency_brake_dist: 0.5                            # 0.2 → 0.5m (lidar_range_min_override=0.35로 front 자기감지 제거 후 상향)
lidar_range_min_override: 0.35                       # 유지 (0.2로 낮추면 front body 자기감지 0.24~0.31m 재발, 2026-04-14 확인)
```

**주의:** 첫 적용 시 RViz2 `/scan` 시각화에서 마스크된 빔이 max_range(10m)로
치환되는지 확인. 예상치 못한 각도에 암이 있으면 `arm_mask_angles_deg` 조정 필요.

**상태:** ✅ 적용 완료 (Phase 2-A, 2026-04-12)

---

### #3 — Emergency Brake 후퇴 속도 추가

**변경 파일:** `rl_offboard_node.py`

#### 현재 문제

```python
# 현재: 제자리 회전만 (장애물에서 멀어지지 않음)
self._publish(0.0, 0.0, self._vz_correction(), yaw_rate)
```

emergency brake 발동 시 회전만 하고 후퇴하지 않으므로, 드론이 장애물에 매우 근접한
채로 제자리 회전 → 회전 중 측면 빔이 장애물에 닿아 재발동 → brake 루프 탈출 불가.

#### 제안 수정

```python
retreat_speed = 0.4  # m/s
vx_r = -retreat_speed * math.cos(yaw)
vy_r = -retreat_speed * math.sin(yaw)
self._publish(vx_r, vy_r, 0.0, yaw_rate)
```

이미 **#1 `_tick()` 코드에 반영**되어 있음. #3은 해당 변경의 근거 문서화.

#### 연구자 관점 검토 및 보완

**① 후퇴 방향의 타당성**
`-cos(yaw), -sin(yaw)` 는 현재 heading의 반대 방향(정후방) 후퇴.
emergency brake 트리거 조건은 `d_front_min <= emergency_brake_dist` (정면 ±half_deg).
즉 장애물이 정면에 있을 때 발동 → 정후방 후퇴는 올바른 방향.
단, 측면 장애물로 인해 발동된 경우(드문 케이스)는 정후방이 최선이 아닐 수 있으나,
그 경우 yaw_rate로 회전하면서 측면 빔이 감소하므로 허용 범위.

**② vz 고도 보정 제거 여부**
원안: `vz=0.0` (고도 보정 제거).
그러나 brake가 수초 지속되면 고도가 ±0.5m 이상 드리프트 가능.
**보완: `self._vz_correction()` 유지 권장.**

```python
self._publish(vx_r, vy_r, self._vz_correction(), yaw_rate)
```

비상 상황에서도 고도 유지가 바닥/천장 충돌을 막는 2차 안전장치 역할을 함.

**③ retreat_speed 0.4 m/s 적절성**
`emergency_brake_dist=0.5m` (#2 반영) 기준으로:
0.5m / 0.4 m·s⁻¹ = 1.25초 이내 안전거리 확보.
LiDAR 10Hz 기준 12~13프레임 내 brake 해제 예상 → 적절.
단, 좁은 통로에서 후퇴가 반대편 장애물로 이어질 수 있으므로
후방 거리 체크를 추가하면 더 안전:

```python
d_rear = self._dist_in_sector(
    180.0 - self.emergency_brake_deg,
    180.0 + self.emergency_brake_deg, "min", use_override=True)
retreat_speed = 0.4 if d_rear > 0.8 else 0.0  # 후방도 막혔으면 제자리 회전만
```

이 체크는 선택적 개선 (params에 `rear_clearance_min: 0.8` 추가 필요).

**④ spinning trap과의 관계**
`_yaw_sat_count > 20` 트랩 탈출 로직이 emergency brake 블록 내에서는 적용되지 않음.
brake가 장시간 지속되면 사실상 무한 회전과 같은 효과.
**보완: brake 연속 발동 횟수 카운터를 도입해 일정 이상이면 후퇴만(yaw_rate=0) 전환:**

```python
self._brake_count = getattr(self, '_brake_count', 0) + 1
if self._brake_count > 30:  # 1.5초(30tick×0.05s) 이상 brake 지속
    yaw_rate = 0.0  # 회전 중단, 후퇴만
```

#### 변경 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| brake 시 이동 | 제자리 회전만 (vx=vy=0) | 정후방 후퇴 0.4 m/s + 회전 |
| 고도 보정 | `_vz_correction()` 적용 | `_vz_correction()` 유지 (원안 제거 → 복원) |
| 후방 장애물 체크 | 없음 | 선택적: `d_rear > 0.8m` 확인 후 후퇴 |
| 장시간 brake 처리 | 없음 | 30틱 초과 시 yaw_rate=0 (후퇴만) |

**상태:** ✅ 적용 완료 (Phase 2-B, 2026-04-12)

---

### #4 — grid_fuser CPU 병목 제거: 직렬화 + 해상도 + inflation 분리

**변경 파일:** `grid_fuser.py`, `grid_fuser.yaml`

#### 현재 문제 구조

5Hz 설정 → 실제 2.88Hz (42% 드롭). 3가지 중첩 원인:

| 원인 | 예상 소요 시간 |
|------|--------------|
| `grid_now.flatten().tolist()` — 194,481개 Python int 객체 생성 | ~15~30ms |
| inflation 연산 (매 publish) — 셀 전체 BFS/반경 탐색 | ~5~15ms |
| 194,481 셀 전체 log-odds 연산 + 타입 변환 | ~5ms |

200ms publish 주기 중 25~50ms를 직렬화+inflation이 소비 → publish_rate 실현 불가.

#### a) `.tolist()` → `array.array` 직접 변환

```python
# 기존: Python 루프로 194,481개 int 객체 생성
msg.data = grid_now.flatten().tolist()

# 개선: numpy buffer → array.array 직접 변환 (Python int 객체 생성 없음)
import array
msg.data = array.array('b', grid_now.flatten().astype(np.int8).tobytes())
```

**연구자 검토:**
ROS2 Python에서 OccupancyGrid `msg.data`(int8[])는 `array.array`, `bytes`,
`bytearray`, numpy array 등 buffer protocol 구현체를 모두 허용.
`array.array('b', ...)` 는 C 연속 메모리를 직접 참조하므로
`.tolist()` 대비 **~10~50배 빠름** (194,481개 → 0개 Python 객체 생성).

대안으로 numpy array 직접 할당도 가능하나, `array.array`가 직렬화 안정성 더 높음:
```python
# numpy 직접 (rosidl_runtime_py 버전에 따라 동작 여부 다름)
msg.data = grid_now.astype(np.int8).flatten()
# → 동작하지 않는 환경이 있으므로 array.array 권장
```

#### b) 그리드 해상도 0.1m → 0.2m

```yaml
# 기존: 441×441 = 194,481 셀
resolution: 0.10
width: 441
height: 441
origin_x: -22.0
origin_y: -22.0

# 개선: 221×221 = 48,841 셀 (75% 감소)
resolution: 0.20
width: 221
height: 221
origin_x: -22.1   # ← 대칭: 221×0.2=44.2m, -22.1 ~ +22.1m
origin_y: -22.1
```

**연구자 검토:**

*적절성*: 드론 크기 ~0.5m, `inflation_radius=0.5m` 기준 0.2m 해상도에서
inflation = 2.5셀. A* 경로 여유 셀 수는 절반으로 줄지만 장애물 회피에 충분.
실제 운용 구역 ±10m × ±10m 기준 셀 수: 100m² / (0.2m)² = 2,500셀.

*origin 보정*: 원안의 `origin_x: -22.0` 유지 시
221×0.2 = 44.2m → x 범위 [-22.0, 22.2m] (비대칭 0.2m).
`origin_x: -22.1`로 수정하면 [-22.1, 22.1m] 완전 대칭. **권장 적용.**

*A* 호환성*: `rl_offboard_node.py`의 `_world_to_grid`는
`OccupancyGrid.info.resolution`을 메시지에서 직접 읽으므로 코드 변경 없이 자동 적응.

*주의*: `grid_fuser.yaml`의 `max_range: 10.0` (LiDAR 최대 거리)는
해상도와 독립적으로 유지.

#### c) inflation을 publish 타이머에서 분리

```python
def publish_grid(self):
    self._apply_decay()
    grid_now = self._export_int_grid()
    # grid_now = self._inflate_grid(grid_now)  ← 제거: A* 호출 시점에 적용
    msg.data = array.array('b', grid_now.flatten().astype(np.int8).tobytes())
```

**연구자 검토:**
`rl_offboard_node.py`의 `_astar_with_inflation(radius)`이 이미
A* 호출 시 inflated grid를 생성하므로, grid_fuser에서 publish 시 inflation은
**중복 연산**이었음. 제거 시 publish 타이밍은 개선되고 A* 결과는 동일.

`grid_fuser.yaml`의 `inflation_radius: 0.5` 파라미터는 이후
grid_fuser 내부에서 미사용 상태가 됨 → 혼동 방지를 위해 주석 처리 또는 제거 권장:
```yaml
# inflation_radius: 0.5  # ← 미사용: inflation은 A* 호출 시점에 rl_offboard_node가 적용
```

#### 변경 요약

| 항목 | 변경 전 | 변경 후 | 효과 |
|------|---------|---------|------|
| 직렬화 | `.tolist()` (Python 객체 194,481개) | `array.array('b', .tobytes())` | ~15~30ms 절감 |
| 해상도 | 0.1m, 441×441 = 194,481셀 | 0.2m, 221×221 = 48,841셀 | 연산 75% 감소 |
| origin | -22.0 / -22.0 | -22.1 / -22.1 | 대칭 범위 보정 |
| inflation | publish마다 실행 | 제거 (A* 시점 전담) | ~5~15ms 절감 |
| inflation_radius yaml | 0.5 (활성) | 주석 처리 (미사용) | 혼동 방지 |
| 예상 publish 실현율 | 5Hz 설정 → 2.88Hz | 5Hz 설정 → ~5Hz | A* 맵 신선도 향상 |

**상태:** ✅ 적용 완료 (Phase 1, 2026-04-12)

---

### #5 — scan_timeout을 sim clock 기반으로 교체

**변경 파일:** `rl_offboard_node.py`, `params.yaml`

#### 현재 문제 구조

```python
# wall clock 기반: Gazebo 실시간 비율 저하 시 scan 주기가 길어져 timeout 오탐
self.t_last_scan = time.time()          # wall clock
now = time.time()
if (now - self.t_last_scan) > self.scan_timeout_sec:
    return False  # → _publish() 호출, cmd_vel 제어 중단
```

Gazebo가 50% 실시간 비율로 동작하면:
- LiDAR sim time 주기 100ms → wall clock 200ms
- `scan_timeout_sec=0.5s` 기준에서는 통과하지만 타이트하게 설정된 경우 오탐 발생
- `_sensors_ok()` → False → `_tick()` 내 제어 로직 스킵 → cmd_vel이 20Hz 설정에도 ~7Hz로 저하

#### 개선 코드

```python
def _cb_odom(self, msg: Odometry):
    self.odom = msg
    self.t_last_odom = self.get_clock().now()   # sim clock

def _cb_scan(self, msg: LaserScan):
    self.scan = msg
    self.t_last_scan = self.get_clock().now()   # sim clock

def _sensors_ok(self) -> bool:
    now = self.get_clock().now()
    odom_age = (now - self.t_last_odom).nanoseconds / 1e9
    scan_age = (now - self.t_last_scan).nanoseconds / 1e9
    if odom_age > self.odom_timeout_sec:
        return False
    if scan_age > self.scan_timeout_sec:
        return False
    ...
```

#### 연구자 관점 검토 및 보완

**① use_sim_time 의존성 명시**
`self.get_clock().now()`는 노드에 `use_sim_time=True`가 설정된 경우에만
sim clock을 반환. `False` 또는 미설정이면 wall clock (기존과 동일).
2026-04-10 수정에서 `rl_offboard.launch.py`에 `use_sim_time: True` 추가 완료 → 전제 충족.
실기체 배포 시에는 `use_sim_time: False`로 전환하면 wall clock으로 자동 복귀.

**② 초기화 — None 체크 또는 초기값 설정 필수**
`t_last_odom`, `t_last_scan`을 `None`으로 초기화하면 첫 `_sensors_ok()` 호출 시
`(now - None)` → `TypeError`. 두 가지 처리 방법:

```python
# 방법 A: 생성자에서 현재 시각으로 초기화 (권장)
self.t_last_odom = self.get_clock().now()
self.t_last_scan = self.get_clock().now()

# 방법 B: _sensors_ok에서 None 체크
if self.t_last_scan is None:
    return False
```

방법 A 권장: 노드 시작 직후 `_sensors_ok()=True`가 되어 pre_setpoint 전송이
즉시 시작되나, 어차피 `_pre_setpoint_count` 조건이 먼저 처리되므로 문제 없음.

**③ odom_timeout_sec 파라미터 추가 필요**
기존 코드에는 `scan_timeout_sec`만 존재. odom 체크를 추가하려면
`params.yaml`에 `odom_timeout_sec` 파라미터를 명시해야 함.
odom은 22Hz로 발행되므로 timeout을 스캔(10Hz)보다 타이트하게 설정 가능:

```yaml
scan_timeout_sec: 0.5   # 기존 유지 (10Hz → 5× 여유)
odom_timeout_sec: 0.3   # 신규 (22Hz → ~6× 여유)
```

**④ sim 일시정지(pause) 시 동작**
Gazebo가 pause되면 `/clock`도 멈춤 → `get_clock().now()` 고정
→ `scan_age`가 증가하지 않음 → `_sensors_ok()=True` 유지.
이는 **올바른 동작**: 시뮬레이션이 멈춘 것이지 센서가 실패한 게 아님.
wall clock 기반이었으면 pause 동안 scan_age가 증가해 timeout 오탐 발생.

**⑤ Duration 연산 타입 확인**
`(now - self.t_last_scan)` 에서 두 값 모두 `rclpy.time.Time`이면
결과는 `rclpy.duration.Duration`. `.nanoseconds` 속성으로 정수 ns 획득 후 `/1e9` → 초.
`time.time()` 기반 float 빼기와 혼용하지 않도록 주의 (기존 코드에서 혼용 사례 있을 수 있음).

#### 변경 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 시간 기준 | `time.time()` (wall clock) | `self.get_clock().now()` (sim clock) |
| odom timeout 체크 | 없음 | `odom_timeout_sec: 0.3` 추가 |
| Gazebo 50% 속도 시 | scan timeout 오탐 가능 | sim time 기준 → 오탐 없음 |
| Gazebo pause 시 | timeout 오탐 → 제어 중단 | clock 동결 → 정상 유지 |
| 초기화 | `None` 또는 `0.0` (위험) | `get_clock().now()` 로 초기화 |

#### params.yaml 추가 파라미터

```yaml
odom_timeout_sec: 0.3   # 신규 (22Hz odom 기준 ~6× 여유)
scan_timeout_sec: 0.5   # 기존 유지
```

**상태:** ✅ 적용 완료 (Phase 1-D, 2026-04-12)

---

### #6 — grid_fuser TF 타임스탬프 오프셋 처리

**변경 파일:** `grid_fuser.py`

#### 현재 문제 구조

```
scan.header.stamp (Gazebo 발행) ─── ~110ms 선행 ───▶ 최신 TF 타임스탬프
                                                        ↑
                          lookup_tf(None) 사용 (기존 수정)
```

`None` 사용으로 scan 드롭 문제는 해결됐으나 위치 오차가 남음:
- 속도 1.5 m/s × 110ms = **0.165m 위치 오차** → 0.2m 해상도 기준 최대 ~1셀

#### 제안 방법 검토

**방법 A: scan stamp 1차 조회 + 최신 TF fallback**
**방법 B: lookup_transform timeout 50ms → 200ms 증가**

#### 연구자 관점 검토

**① 방법 B(200ms 블로킹) 부적합 이유**
scan 콜백 주기 = 1/10Hz = **100ms**. timeout 200ms는 콜백 주기보다 길어
TF를 기다리는 동안 다음 scan 콜백이 쌓임 → 콜백 큐 누적 → CPU 과부하.
기존에 `None` 방식으로 전환한 이유가 바로 이 블로킹 문제였으므로 역행.
**방법 B는 적용하지 않음.**

**② 방법 A 구현 시 예외 처리 필수**
`lookup_transform(stamp)` 에서 stamp가 TF 버퍼보다 미래이면
`tf2_ros.ExtrapolationException` 예외를 발생(None 반환이 아님).
`lookup_tf` 래퍼가 이미 try/except로 None을 반환하는지 확인 필요.
반환하지 않으면 수정 필요:

```python
def lookup_tf(self, target, source, stamp, timeout_sec=0.02):
    try:
        return self.tf_buffer.lookup_transform(
            target, source,
            stamp if stamp is not None else rclpy.time.Time(),
            rclpy.duration.Duration(seconds=timeout_sec)
        )
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
        return None
```

**③ 방법 A의 1차 조회 timeout은 20ms 이하로 제한**
scan stamp 기반 1차 조회가 실패할 경우 fallback까지 최대 대기 시간:
`timeout_1차 + 처리 시간 < 콜백 주기 100ms`
→ **1차 timeout: 20ms** (여유 80ms), fallback은 즉시(0ms).

```python
def process_scan(self, scan: LaserScan):
    sensor_frame = scan.header.frame_id
    # 1차: scan stamp 기준 (정확, 단기 대기)
    tf_msg = self.lookup_tf(self.frame_id, sensor_frame,
                             scan.header.stamp, timeout_sec=0.02)
    # 2차 fallback: 최신 TF (기존 방식)
    if tf_msg is None:
        tf_msg = self.lookup_tf(self.frame_id, sensor_frame, None)
    if tf_msg is None:
        return
```

**④ 0.2m 해상도(#4 적용 후) 기준 실질 영향**
오차 0.165m < 0.2m(1셀) → TF 불일치에 의한 ghost obstacle은
1셀 범위 내에서 발생. log-odds decay(#4 yaml)로 수회 scan 후 자연 소멸.
즉, #4 해상도 변경 후에는 이 오차의 체감 영향이 감소함.

**⑤ scan stamp 미래 여부 사전 확인 (선택적 보완)**
1차 조회 실패 원인이 "미래 stamp"인지 확인하면 fallback 결정을 더 빠르게 할 수 있음:

```python
# 선택적: TF 버퍼 최신 시각과 비교
try:
    latest = self.tf_buffer.get_latest_common_time(
        self.frame_id, sensor_frame)
    use_stamp = (scan.header.stamp
                 if scan.header.stamp <= latest
                 else None)
except tf2_ros.LookupException:
    use_stamp = None
tf_msg = self.lookup_tf(self.frame_id, sensor_frame, use_stamp)
```

단, `get_latest_common_time` 자체도 예외를 던질 수 있어 구현 복잡도 증가.
기본 구현은 **방법 A(단순 try-fallback)** 으로 충분.

#### 변경 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| TF 조회 방식 | `None` (최신 TF 고정) | scan stamp 1차 (20ms timeout) → None fallback |
| 1차 조회 실패 처리 | 해당 없음 | ExtrapolationException 캐치 후 None 반환 |
| 블로킹 timeout | 50ms (→ None으로 우회) | 1차 20ms + fallback 0ms = 최대 20ms |
| 위치 오차 | 최대 0.165m (최신 TF 기준) | scan stamp 성공 시 ~0ms / fallback 시 0.165m |
| 방법 B 채택 여부 | — | **미채택** (200ms 블로킹 = 콜백 큐 누적 위험) |

**상태:** ✅ 적용 완료 (Phase 4-A, 2026-04-12)

---

### #7 — scan-odom 시간 정합: dead reckoning 보정

**변경 파일:** `rl_offboard_node.py` (`_build_obs()` 내부)

#### 현재 문제 구조

`_build_obs()`에서 `px, py, yaw`를 최신 odom에서 읽지만,
scan은 다른 타임스탬프로 취득됨:
- odom 주기: 22Hz → ~45ms
- scan 주기: 10Hz → ~100ms
- 최대 dt: ~45ms, 속도 1.5m/s → 위치 오차 최대 **0.068m**

PPO 관측의 `goal_rel` (2차원 벡터)이 실제 scan 시점이 아닌
odom 시점의 위치 기준으로 계산되어 방향 오류 발생.

#### 제안 코드 (dead reckoning 보정)

```python
px, py = float(self.odom.pose.pose.position.x), float(self.odom.pose.pose.position.y)
yaw = quat_to_yaw(self.odom.pose.pose.orientation)

# scan-odom 시간 정합: odom이 오래됐으면 velocity로 보정
if self.odom.twist is not None:
    scan_t = rclpy.time.Time.from_msg(self.scan.header.stamp).nanoseconds / 1e9
    odom_t = rclpy.time.Time.from_msg(self.odom.header.stamp).nanoseconds / 1e9
    dt = scan_t - odom_t
    if abs(dt) > 0.05:
        vx_b = self.odom.twist.twist.linear.x
        vy_b = self.odom.twist.twist.linear.y
        wz   = self.odom.twist.twist.angular.z
        vx_w = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
        vy_w = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
        px  += vx_w * dt
        py  += vy_w * dt
        yaw += wz   * dt
```

#### 연구자 관점 검토 및 보완

**① twist 좌표계 — body frame 변환 올바름**
MAVROS `/mavros/local_position/odom`의 `twist`는 child_frame_id(`base_link`)
기준 body frame 속도. 코드의 회전 변환:
```
vx_w = vx_b·cos(yaw) - vy_b·sin(yaw)
vy_w = vx_b·sin(yaw) + vy_b·cos(yaw)
```
ENU body→world 2D 회전으로 **올바름**.

**② `abs(dt) > 0.05` 임계값 — 거의 발동 안 됨**
odom 22Hz → 두 odom 사이 최대 간격 = 45ms.
scan과 가장 가까운 odom의 dt < 45ms < 50ms → **임계값 0.05s 미달로 보정이 적용되지 않음**.
보정이 실질적으로 동작하려면 임계값을 낮춰야 함:

```python
# 권장: 10ms 이상이면 보정 (odom 22Hz 기준 ≈ 절반 주기)
if abs(dt) > 0.01:
    ...
```

또는 임계값 없이 항상 적용 (`if True:` 또는 조건 제거):
dt가 작을 때 보정값도 작으므로 항상 적용해도 해롭지 않음.

**③ `self.odom.twist is not None` 체크 불필요**
ROS2 `nav_msgs/Odometry`에서 `twist`는 `TwistWithCovariance` 필드로
항상 초기화됨(None이 될 수 없음). 체크는 무해하나 의미 없음.
대신 scan stamp 유효성을 체크하는 것이 더 실용적:

```python
# scan stamp 유효성 체크 추가
if self.scan.header.stamp.sec == 0:
    pass  # invalid stamp → 보정 건너뜀
else:
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
        yaw += wz * dt
```

**④ dt 부호 처리**
- `dt > 0`: scan이 odom보다 미래 → 드론이 움직였을 위치 예측 (전진 dead-reckoning) ✓
- `dt < 0`: scan이 odom보다 과거 → 위치 역산 (드문 케이스, 수 ms 수준) ✓
- 양쪽 모두 `px += vx_w * dt`로 올바르게 처리됨.

**⑤ PPO 관측 내 실질 영향 평가**
66차원 관측 중 `goal_rel`은 2차원(3%). 최대 보정량:
- dt=45ms, v=1.5m/s → Δpos=0.068m
- goal_dist=5m 기준 각도 오차 ≈ `arctan(0.068/5)` ≈ **0.78°**
영향이 매우 작아 PPO 출력에 유의미한 차이가 없을 수 있으나,
좁은 통로에서 goal_rel 방향이 벽 쪽을 가리키는 상황에서는 의미 있음.
구현 비용이 낮으므로 적용 권장.

#### 최종 보완 코드

```python
px, py = float(self.odom.pose.pose.position.x), float(self.odom.pose.pose.position.y)
yaw = quat_to_yaw(self.odom.pose.pose.orientation)

# scan-odom 시간 정합 (dead reckoning)
if self.scan is not None and self.scan.header.stamp.sec != 0:
    scan_t = rclpy.time.Time.from_msg(self.scan.header.stamp).nanoseconds / 1e9
    odom_t = rclpy.time.Time.from_msg(self.odom.header.stamp).nanoseconds / 1e9
    dt = scan_t - odom_t
    if abs(dt) > 0.01:   # 10ms 이상 차이 시 보정 (원안 50ms → 10ms)
        vx_b = self.odom.twist.twist.linear.x
        vy_b = self.odom.twist.twist.linear.y
        wz   = self.odom.twist.twist.angular.z
        vx_w = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
        vy_w = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
        px  += vx_w * dt
        py  += vy_w * dt
        yaw += wz * dt
```

#### 변경 요약

| 항목 | 원안 | 보완 후 |
|------|------|---------|
| 보정 임계값 | `abs(dt) > 0.05` (거의 미발동) | `abs(dt) > 0.01` (22Hz odom 기준 실질 동작) |
| twist None 체크 | 있음 (불필요) | 제거, scan stamp 유효성 체크로 대체 |
| 보정 방향 타당성 | body→world 변환 올바름 | 동일 유지 |
| 예상 보정량 | dt<45ms, v<1.5m/s → 최대 0.068m | 동일 |

**상태:** ✅ 적용 완료 (Phase 4-B, 2026-04-12)

---

### #8 — Heading Alignment 완전 정지 제거: 오차 비례 감속

**변경 파일:** `rl_offboard_node.py` (`_tick_global()` 내부)

#### 현재 문제 구조

```python
# 기존: heading 오차 > 0.26rad (~15°) → 완전 정지 + 회전만
if abs(yaw_err) > tol:
    yaw_rate = max(-planner_yaw_limit, min(planner_yaw_limit, 2.0 * yaw_err))
    return 0.0, 0.0, self._vz_correction(), yaw_rate
```

bag 분석에서 이 분기가 **88~93초** 연속 정지를 유발. 원인:
1. A* waypoint 방향으로 yaw 정렬 시도
2. 정렬 완료 전 waypoint 갱신 → 새 yaw_err 발생
3. 반복 → 사실상 무한 제자리 회전

#### 제안 수정 (오차 비례 감속)

```python
if abs(yaw_err) > tol:
    yaw_rate = max(-planner_yaw_limit, min(planner_yaw_limit, 2.0 * yaw_err))
    align_scale = max(0.0, 1.0 - abs(yaw_err) / (math.pi / 2))
    v_cmd = min(self.v_max, 0.8 * gnorm) * align_scale
    vx = v_cmd * math.cos(target_yaw)
    vy = v_cmd * math.sin(target_yaw)
    return vx, vy, self._vz_correction(), yaw_rate
```

#### 연구자 관점 검토 및 보완

**① align_scale 선형 보간 분석**
```
yaw_err = 0°  → align_scale = 1.00  (정속)
yaw_err = 15° → align_scale = 0.83  (83%)
yaw_err = 30° → align_scale = 0.67  (67%)
yaw_err = 45° → align_scale = 0.50  (50%)
yaw_err = 90° → align_scale = 0.00  (완전 정지)
yaw_err > 90° → 0.0으로 클램프
```
선형 감속은 구현 단순성과 예측 가능성 면에서 실용적. 대안으로
코사인 기반 `align_scale = max(0.0, math.cos(abs(yaw_err)))`도 유효하나
물리적 의미 차이가 작으므로 선형 유지 권장.

**② 멀티로터에서 비정렬 이동의 타당성**
고정익과 달리 멀티로터는 yaw 방향과 무관하게 임의 방향으로 병진 가능.
`vx = v_cmd * math.cos(target_yaw)`: body 방향이 아닌 **world frame 목표 방향**으로 이동.
PPO emergency brake가 실제 LiDAR 빔을 기반으로 작동하므로
비정렬 상태에서 이동해도 장애물 안전성은 유지됨. **타당함.**

**③ `gnorm` 의미 명확화 및 이중 감속 주의**
`gnorm`이 waypoint까지의 거리라면 `planner_slowdown_dist` 로직이
이미 `_tick_global`의 다른 분기에서 감속을 적용할 수 있음.
`0.8 * gnorm` 추가 감속이 이중으로 적용되는지 코드 구현 시 확인 필요.
만약 `gnorm`이 unit vector의 크기(norm)이면 항상 1.0 근처 → 이중 감속 없음.

권장 확인:
```python
# gnorm 출력으로 스케일 확인
self.get_logger().debug(f"gnorm={gnorm:.3f}, align_scale={align_scale:.3f}")
```

**④ yaw_rate는 항상 planner_max_yaw_rate(0.3 rad/s)에 포화**
비례 제어 `2.0 * yaw_err`에서 yaw_err > 0.15 rad이면 즉시 상한 0.3 rad/s 포화.
따라서 실제 yaw_rate는 0.3 rad/s 일정 → heading_align_tol(0.26 rad) 도달 시간:
0.26 rad / 0.3 rad·s⁻¹ = **0.87초** (기존 수십 초 → 1초 이내).
단, waypoint 갱신 주기(`replan_period: 2.0s`)보다 짧으므로 정렬 완료 후 전진 가능.

**⑤ heading_align_tol 재검토**
현재 `0.26 rad (~15°)`. 비례 감속 도입 후 이 임계값의 역할이 바뀜:
- 기존: 정지/이동 경계 (15° 이상 → 완전 정지)
- 변경 후: 감속 시작 경계 (15° 이상 → align_scale에 따라 감속)

align_scale은 15°에서 83%로 거의 정속 → 임계값을 0.52 rad(30°)로 올려
더 넓은 범위에서 정속 유지도 가능. 단, 현재 값 유지해도 기능 동작에 문제 없음.

#### 변경 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| heading 오차 > tol 시 | 완전 정지 (vx=vy=0) | 오차 비례 감속 (최대 83%~0%) |
| 이동 방향 | 정지 | `target_yaw` world frame 방향 |
| 완전 정지 조건 | `abs(yaw_err) > 0.26 rad` (15°) | `abs(yaw_err) >= π/2` (90°) |
| 정렬 예상 소요 시간 | 수십~90초 (반복 재정렬) | **< 1초** (0.87초 이내) |
| 비례 감속 기준 | 해당 없음 | `1 - yaw_err/(π/2)` 선형 |

**상태:** ✅ 적용 완료 (Phase 3-A, 2026-04-12)

---

### #9 — A* 블로킹 제거: MultiThreadedExecutor + path_lock

**변경 파일:** `rl_offboard_node.py`

#### 현재 문제 구조

```
SingleThreadedExecutor
├── _tick (20Hz, ~5ms)
├── _tick_replan → _plan_path → A* (~50~200ms)   ← 블로킹!
└── _cb_scan, _cb_odom ...
```

`_tick_replan`에서 A*가 실행되는 동안 executor가 블로킹 → `_tick` 지연
→ cmd_vel이 20Hz 설정임에도 A* 실행 주기마다 드롭 발생.

#### 제안 구조

```
MultiThreadedExecutor (num_threads=3)
├── Thread 1: _ctrl_cb_group   → _tick (20Hz, 항상 즉시 실행)
├── Thread 2: _replan_cb_group → _tick_replan → A* (블로킹 격리)
└── Thread 3: default group    → _cb_scan, _cb_odom, _cb_state
```

#### 연구자 관점 검토 및 보완

**① 센서 콜백 그룹 누락 — 명시 필요**
제안 코드는 `_ctrl_cb_group`, `_replan_cb_group`만 명시.
센서 구독(`_cb_scan`, `_cb_odom`, `_cb_state`)은 callback_group 미지정 시
노드의 default MutuallyExclusiveCallbackGroup에 할당됨.
MultiThreadedExecutor에서 각 그룹은 독립 스레드에서 실행 가능하므로
3개 그룹(ctrl / replan / sensor)이 실질적으로 분리됨. **현 설계 유효.**

단, 더 명확한 의도 전달을 위해 센서 그룹도 명시 권장:
```python
self._sensor_cb_group = MutuallyExclusiveCallbackGroup()
# create_subscription 시 callback_group=self._sensor_cb_group 지정
```

**② ⚠ _build_obs의 중첩 lock — 데드락 위험**
#1 `_build_obs()` 코드에 아래 패턴이 있음:
```python
# _build_obs 내부 (#1 코드)
with self._path_lock:              # ① lock 획득
    wp = self._path_lookahead_wp() # ② _path_lookahead_wp 내부에서도 lock 획득 시도
```

`_path_lookahead_wp()`는 함수 내부에서 `with self._path_lock:` 을 다시 획득.
Python `threading.Lock()`은 재진입 불가(non-reentrant) → **데드락**.

수정: `_build_obs`에서 outer `with` 제거 (함수 자체가 lock 처리):
```python
# 수정 후
wp = self._path_lookahead_wp()   # 함수 내부 lock으로 충분
target_xy = np.array(wp, dtype=np.float32) if wp else self.goal_xy
```

**③ `threading.RLock()` 대안**
데드락을 구조적으로 방지하려면 `threading.Lock()` 대신 `threading.RLock()`(재진입 가능 lock) 사용:
```python
self._path_lock = threading.RLock()  # Lock → RLock
```
같은 스레드 내에서 중첩 획득을 허용하면서, 다른 스레드는 여전히 블로킹.
`_build_obs`의 outer lock 패턴을 유지하고 싶을 때 유용.

**권장: `RLock` 사용 + `_build_obs`의 outer lock 제거** (이중 방어).

**④ `_path_advance_wp` 내부 lock 구조 검토**
제안 코드의 `_path_advance_wp`는 lock 내부에서 `math.hypot`, 조건 분기, 인덱스 갱신까지 수행.
`_tick`(20Hz)에서 매 tick 호출 → lock 보유 시간은 수 μs 수준으로 짧아 허용 가능.
`_plan_path`의 lock 보유 구간도 `_path_xy`, `_path_wp_idx` 할당만이므로 적절.

**⑤ num_threads=3 적절성**
| 스레드 | 역할 | 특성 |
|--------|------|------|
| 1 | _tick (ctrl) | 20Hz, 짧은 실행 (~5ms), 지연 불허 |
| 2 | _tick_replan (A*) | 2Hz, 긴 실행 (~50~200ms), 지연 허용 |
| 3 | 센서 콜백 | 비동기 I/O, 짧음 |

`num_threads=4` 이상은 WSL2에서 컨텍스트 스위칭 오버헤드가 이득보다 클 수 있음.
**3스레드가 최적.**

**⑥ #4 해상도 변경(0.2m)과의 선행 조건**
A* 셀 수 194,481 → 48,841 (75% 감소) → 실행 시간 단축.
#9 적용 시 A*가 격리되므로 블로킹 자체는 해결되나, 실행 시간이 길수록
replan_cb_group 스레드가 오래 점유됨. #4 선행 적용 권장.

#### 변경 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| Executor | SingleThreadedExecutor | MultiThreadedExecutor(num_threads=3) |
| A* 실행 중 cmd_vel | 드롭 (블로킹) | 정상 20Hz 유지 |
| path_lock | 없음 | `threading.RLock()` (재진입 안전) |
| _build_obs outer lock | `with self._path_lock` (데드락 위험) | 제거 (함수 자체 lock 사용) |
| 센서 콜백 그룹 | default (암묵적) | `_sensor_cb_group` 명시 (선택) |
| #4 선행 조건 | — | 해상도 0.2m 전환 후 적용 권장 |

**상태:** ✅ 적용 완료 (Phase 5, 2026-04-12)

---

## 2026-04-11: LiDAR motion distortion 경감 — 빔 수 축소 + global planner yaw rate 제한

### 배경 (ros2 bag 분석 결과)
23분 비행 bag 분석에서 다음이 확인됨:
- LiDAR 설정 10Hz → 실제 수신 7.4Hz (26% 드롭): CPU가 이미 한계
- 그리드 설정 5Hz → 실제 3.5Hz: grid_fuser도 CPU 병목
- 드론 회전 시 200ms+ LiDAR 갭 519회 (호버 152회 대비 3.4×)
- 회전 시 최대 yaw rate 1.0 rad/s × 스캔 주기 100ms = 5.7° 회전
  → 3m 거리 장애물 위치 오차 ≈ 0.30m (3셀): 실제로 뚫린 공간이 occupied로 오표시됨

분석 보고서의 "LiDAR 20Hz 증가" 제안은 현재 CPU가 10Hz 설정에서도 26%를 드롭하고 있어
적용 시 오히려 드롭률이 60%+로 악화되므로 채택하지 않음.
"IMU 초기 추정값 통합(SLAM용)" 제안은 이 프로젝트가 SLAM 없이
PX4 EKF2 odom을 직접 사용하는 구조라 해당 없음.

### 수정 내용

| 파일 | 변경 내용 |
|------|-----------|
| `rplidar/model.sdf` | `samples: 360 → 180`. 빔 수를 절반으로 줄여 `process_scan` 연산 부하 절감 → 실제 LiDAR 수신율 10Hz에 근접. 2° 간격으로도 장애물 회피에 충분하며, `obs_beams: 64`로 다운샘플링하므로 정책 관측값 영향 없음. |
| `rl_offboard_node.py` | `_tick_global`: 헤딩 정렬 yaw rate를 `planner_max_yaw_rate` 파라미터로 별도 제한. RL의 `max_yaw_rate(1.0 rad/s)`는 재학습 없이 변경 불가하므로 global planner 전용 상한을 분리. |
| `params.yaml` | `planner_max_yaw_rate: 0.3` 추가. 0.3 rad/s × 100ms = 1.7° 회전 → 3m 장애물 오차 ≈ 0.09m(1셀 이내)로 감소. |

### 적용 후 예상 효과
| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| LiDAR CPU 부하 | 360빔 → 7.4Hz 실수신 | 180빔 → ~10Hz 실수신 |
| Global planner 회전 중 장애물 오차 | 0.30m (3셀) | 0.09m (1셀 이내) |
| RL 모드 yaw rate | 1.0 rad/s (변경 없음) | 1.0 rad/s (변경 없음) |

---

## 2026-04-10: 시뮬레이션 환경 점검 및 조치 — use_sim_time 통일 + URDF collision 수정

### 문제 상황
드론 회전 시 /scan 장애물 인식 오차 원인 점검. 5가지 항목 조사 결과:

1. **use_sim_time 미통일**: `rl_offboard.launch.py`, `goal_pub.launch.py`의 노드들에 `use_sim_time` 미설정.
   시뮬레이션 시간(/clock)과 시스템 시간이 혼용되어 TF 타임스탬프 불일치 가능성.

2. **base_link collision 상향 노출**: `iris_rplidar_depth.urdf`의 base_link collision box가
   `z=0` 중심, 높이 10cm → z ∈ [-5, +5] cm 범위. LiDAR(z=25cm)의 수평 빔이
   본체 collision 위를 지나는 구조는 직접적 차폐는 아니나, Gazebo 물리 엔진에서
   self-occlusion 영향 가능성 있음.

3. **scan_time/time_increment**: Gazebo에서 `time_increment=0` (모든 빔 동시 발사) 확인 → 정상, 변경 없음.

4. **rplidar_link 마운팅**: `xyz="0 0 0.25" rpy="0 0 0"` — 물리적으로 정상, 변경 없음.

5. **/tf, /tf_static bag 미포함**: 이전 bag 녹화 명령에 누락. 재생 시 TF 복원 불가.

### 수정 내용

| 파일 | 변경 내용 |
|------|-----------|
| `rl_offboard.launch.py` | `goal_pub`, `rl_node`, `rl_node_planner_only`, `rl_node_rl_only` 모든 노드에 `use_sim_time: True` 추가 |
| `goal_pub.launch.py` | `course_goal_publisher_node`, `astar_apf_planner2_node`에 `use_sim_time: True` 추가 |
| `iris_rplidar_depth.urdf` | `base_link` collision origin `z=0 → z=-0.05`. 본체 collision이 z ∈ [-10, 0] cm로 아래 방향만 확장 → LiDAR 방향(위) collision 없음 |

### 미조치 항목 (확인 필요)

| 항목 | 내용 |
|------|------|
| `/clock` 토픽 | `ros2 topic hz /clock` 으로 Gazebo 발행 여부 확인 필요. 미발행 시 `use_sim_time: True`가 오히려 타임아웃 유발 → `False`로 복원 필요 |
| bag `/tf`, `/tf_static` | 다음 녹화부터 `ros2 bag record ... /tf /tf_static` 추가 필요 |

---

## 2026-04-09: OccupancyGrid 버그 수정 + log-odds 리팩터 + 코드 정리

### 문제 상황

**1. l_miss 부호 버그 → 전체 그리드 occupied 수렴**
`l_miss = log(prob_miss / (1-prob_miss))`로 계산 시 `prob_miss=0.40` → `l_miss=-0.405` (음수).
`_update_cell_free`에서 `grid_log -= l_miss` = `grid_log + 0.405` → free 빔이 오히려 log-odds 증가.
hit/miss 빔 모두 log-odds를 올려 그리드 전체가 occupied(검정)로 수렴.

**2. float32 overflow**
`np.exp(self.grid_log)` 에서 float32 한계(~88) 초과 시 `RuntimeWarning: overflow in exp` 발생.

**3. Gazebo scan stamp 110ms 선행 → TF 큐 누적**
`scan.header.stamp` 기준 TF 조회 시 Gazebo에서 scan stamp가 최신 TF보다 ~110ms 앞서있어
`timeout=0.2s` 블로킹이 콜백 큐를 누적시킴. 9Hz scan 대비 처리 속도 저하 → stamp 격차 무한 증가.

**4. 미사용 파라미터 잔존**
`plan_unknown_block_radius_m=0.0`(비활성 고정)으로 인해 실행되지 않는 도넛 차단 로직이 코드에 잔존.

**5. Homing 중 yaw 강제 변경**
`_publish_setpoint`에서 `orientation.z=1.0, w=0.0` (yaw=180°) 하드코딩 → homing 이동 중 드론이 불필요하게 회전.

---

### 수정 내용

| 파일 | 변경 내용 |
|------|-----------|
| `grid_fuser.py` | **l_hit/l_miss 공식 통일**: 둘 다 `log(p/(1-p))` 동일 공식 사용. `l_hit>0` (prob_hit>0.5), `l_miss<0` (prob_miss<0.5). update 시 둘 다 `+=` 로 통일 → 부호 역전 버그 구조적 제거. |
| `grid_fuser.py` | **파라미터 범위 검증**: `prob_hit ∈ (0.5,1.0)`, `prob_miss ∈ (0.0,0.5)` 벗어나면 노드 시작 시 `ValueError`. 조용한 오동작 방지. |
| `grid_fuser.py` | **float32 overflow 방지**: `np.exp(grid_log)` → `np.exp(np.clip(grid_log, -87.0, 87.0))`. |
| `grid_fuser.py` | **TF 조회 방식 확정**: `scan.header.stamp` 기준 조회는 Gazebo에서 콜백 큐 누적을 유발함을 확인. `None`(최신 TF) 유지. 저속 드론 위치 오차 < 9cm(1셀 이내)로 허용 가능. |
| `grid_fuser.py` | **헬퍼 메서드 추가**: `_extract_pose_2d`, `_lerp_pose_2d`, `_apply_pose_2d` — 향후 per-beam 보간 활성화 시 재사용 가능하도록 분리. (`time_increment=0`인 Gazebo에서는 단일 TF로 동작.) |
| `rl_offboard_node.py` | **미사용 코드 제거**: `plan_unknown_block_radius_m`, `plan_unknown_block_inner_margin_m` 파라미터 선언 및 도넛 차단 로직 삭제. `radius=0.0`으로 항상 비활성이었음. |
| `params.yaml` | 위와 동일한 파라미터 2개 제거. |
| `course_goal_publisher_node.py` | **Homing yaw 유지**: `_publish_setpoint`에서 `orientation.z=1.0,w=0.0`(yaw=180° 고정) → 현재 드론 orientation 유지. pose 없을 시 identity(w=1.0) fallback. |

---

### log-odds 파라미터 해석 (정리)

```
l_hit  = log(prob_hit  / (1-prob_hit))  # 양수: hit 시 log-odds 증가
l_miss = log(prob_miss / (1-prob_miss)) # 음수: miss 시 log-odds 감소

prob_hit  ↑ → l_hit  ↑  → 점령 빠름   (유효 범위: 0.5 초과)
prob_hit  ↓ → l_hit  ↓  → 점령 느림
prob_miss ↑ → l_miss → 0 → 복구 느림  (유효 범위: 0.5 미만)
prob_miss ↓ → l_miss ↓  → 복구 빠름
```

### 수정 전/후 동작 비교

| 상황 | 수정 전 | 수정 후 |
|------|---------|---------|
| 드론 정지 중 | 전체 occupied (l_miss 부호 버그) | 올바른 free/occ 구분 |
| `np.exp` overflow | RuntimeWarning 발생 | clip으로 억제 |
| scan stamp 기반 TF | 콜백 큐 누적, stamp 격차 무한 증가 | None(최신 TF)으로 확정 |
| homing 이동 중 | yaw=180° 강제 회전 | 현재 yaw 유지 |
| 미사용 도넛 차단 로직 | 코드 잔존 | 완전 제거 |

---

## 2026-04-08: OccupancyGrid 오표시 보정 — TF 타임스탬프 + log-odds 활성화

### 문제 상황
드론 회전 중 LiDAR 스캔이 진행되면 각 빔이 서로 다른 드론 자세에서 측정되지만,
`grid_fuser.process_scan`이 스캔 전체에 단일 TF를 적용해 빔 위치가 왜곡됨(motion distortion).
이로 인해 실제로 뚫린 공간이 occupied로 오표시되어 A*가 해당 방향으로 경로를 생성하지 않음.

추가로, LiDAR 스캔 타임스탬프가 최신 TF보다 ~110ms 앞서는 경우
`"extrapolation into the future"` 오류로 모든 스캔이 버려져
그리드가 영구적으로 비어 있는 문제도 확인됨 (RViz2 검은 영역이 뜰 때/안 뜰 때 차이).

### 수정 내용

| 파일 | 변경 내용 |
|------|-----------|
| `grid_fuser.py` | `process_scan`: TF 조회 시 `scan.header.stamp` → `None` (최신 TF 사용). `scan.header.stamp`가 TF보다 ~110ms 앞서 "extrapolation into the future" 오류 방지. 저속 드론에서 위치 오차 < 9cm(1셀 이내)로 무시 가능. |
| `grid_fuser.yaml` | `use_log_odds: false → true`. 단일 빔 히트로 즉시 occupied 확정 방지 — 여러 번 일관된 관측이 있어야 occupied 확정, miss 관측 시 free로 복구 |
| `grid_fuser.yaml` | `prob_hit: 0.65 → 0.60` — 단일 히트 신뢰도 하향 |
| `grid_fuser.yaml` | `prob_miss: 0.35 → 0.40` — miss 시 free 복구 속도 향상 |
| `grid_fuser.yaml` | `clamp_max: 3.5 → 2.0` — occupied 확정에 필요한 누적 히트 수 감소 (~3회) |
| `grid_fuser.yaml` | `decay_rate: 0.0 → 0.02` — 매 publish마다 log-odds를 0 방향으로 소폭 감쇠, 오래된 오표시 자동 소멸 |

### 비고
Gazebo에서 `time_increment=0` (모든 빔 동시 발사) 확인 → per-beam 보간 불필요.
실제 motion distortion 오차는 TF 110ms 시간 오차에서 발생하며, 저속 드론에서 1셀 이내로 허용 가능.
상세 분석 및 후속 수정 → 2026-04-09 항목 참조.

---

## 2026-04-07: 시뮬레이션 충돌 수정 — start cell occupied 직선 fallback + RL exit 임계값

### 문제 상황 (로그 분석)
PPO 학습 성능(weighted_score=0.95)과 달리 실제 시뮬레이션에서 드론이 벽에 충돌.
다음 3단계 연쇄 실패 구조가 확인됨:

```
1. 드론이 장애물 1m 이내 접근
   → inflation_radius=1.0m 에 의해 드론 현재 셀 val=100 (occupied)

2. _plan_path의 elif self._is_occ(*s_ij) 분기 진입
   → A* 포기, 직선 fallback: self._path_xy = [(gx, gy)]  ← 장애물 관통 경로

3. 직선 경로가 장애물로 드론을 안내
   → 0.3~4초 만에 d_front < 2.5 → RL 재진입
   → RL 해제 → 또 직선 fallback → 루프 반복
   → 최종: d_front=0.73 → Emergency brake 실패 → 충돌
```

추가로 `rl_trigger_front_exit: 5.0`이 너무 높아, 장애물 밀집 구역에서 RL이 d_front=10.0 (max range)
에서만 해제되므로 exit → 즉시 재진입 사이클이 반복됨.

### 수정 내용

| 파일 | 변경 내용 |
|------|-----------|
| `rl_offboard_node.py` | `_plan_path`: start/goal occupied 시 A* 포기 제거. `_astar_with_inflation(radius=0)`으로 start/goal 강제 해제 후 우회 경로 탐색 |
| `rl_offboard_node.py` | `_plan_path`: `use_topo_diversity=False` 경로도 `_astar()`가 아닌 `_astar_with_inflation(radius=0)` 사용 — start/goal force-clear 통일 |
| `params.yaml` | `rl_trigger_front_exit` 변경 없음 (5.0 유지) — 장애물 밀집 구역에서 RL이 유지되는 것은 의도된 동작 |

### 수정 전/후 동작 비교
| 상황 | 수정 전 | 수정 후 |
|------|---------|---------|
| start cell occupied | 직선 fallback (장애물 관통) | A* force-clear 후 우회 경로 탐색 |
| RL exit 조건 | d_front > 5.0m | d_front > 5.0m (동일 유지) |
| exit 후 A* 성공 시 | 우회 경로 추종 | 우회 경로 추종 (동일) |
| exit 후 A* 실패 시 | 직선 fallback (충돌 유발) | 직선 fallback (force-clear로 실패 빈도 대폭 감소) |

### 설계 의도 메모
`rl_trigger_front_exit: 5.0`은 전방이 완전히 열렸을 때(`d_front=10.00 = max_range`)만 RL을 해제한다.
장애물 밀집 구역에서 RL이 오래 유지되는 것은 의도된 동작 — RL이 밀집 장애물 회피에 적합하고,
global planner는 장거리 방향 안내 역할. exit 임계값을 낮추면 global이 3.5m 전방 clear일 때 넘겨받지만
측면/후방 장애물로 인해 즉시 RL 재진입 → switching 빈도만 증가하고 실익 없음.

---

## 2026-04-05: 설계 메모 — B-spline 스무딩 vs 논문 궤적, 실제 스플라인 추종 시 변경 포인트

> 코드 변경 없음. `rl_offboard_node.py` 동작·향후 구현 참고용 정리.

### 참고 논문
- Zhou et al., *Robust Real-time UAV Replanning Using Guided Gradient-based Optimization and Topological Paths* ([arXiv:1912.12644 / ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1912.12644))

### 논문의 uniform B-spline 궤적 vs 현재 노드의 B-spline

| 구분 | 논문 (Fast-Planner 계열) | 현재 `rl_offboard_node.py` |
|------|---------------------------|----------------------------|
| 역할 | **시간 구조를 가진 3D 궤적**의 최적화 변수(제어점), ESDF·동역학 비용 | A* 경로를 **`scipy.interpolate.splprep` / `splev`**로 **2D 기하 스무딩** 후 점열로 저장 |
| 매듭·표현 | 균일 매듭(uniform) 등 **궤적 최적화용** 표준 설정 | `splprep`가 데이터 적합용으로 매듭·부드러움 `s` 결정 (논문의 “uniform knot 궤적”과 동일하다고 보기 어려움) |
| 실행 | 생성 스플라인이 **추종·실행 궤적의 중심** | 샘플된 `_path_xy`를 **웨이포인트**(폴리라인)처럼 인덱스·lookahead로 추종 |

### 현재 노드가 “스플라인 궤적 추종”이 아닌 이유
- `_smooth_path_bspline`이 곡선 계수(`tck`)를 **보관하지 않고**, `smooth_bspline_output_points`개로 **이산 샘플**만 `_path_xy`에 넣음.
- `_tick_global`은 `_path_wp_idx`, `path_lookahead_steps`로 **점 리스트**만 추종(목표 방향 헤딩 정렬 후 속도). 곡선 위 투영, 호장(arc length), 접선 기반 연속 추종 법칙 없음.
- 하이브리드에서 **RL 모드**일 때는 경로 기반 추종 없이 정책 출력만 사용.

### 실제 스플라인(연속 곡선) 추종으로 바꿀 때 검토할 코드 위치

1. **곡선 보관**  
   - `_smooth_path_bspline`: `tck`( 및 필요 시 `u` 범위)을 노드 상태로 저장.  
   - `_path_xy`는 RViz/`Path` 발행용으로 샘플 재생성 가능.

2. **상태 변수**  
   - `_path_wp_idx` 대신 **진행 파라미터** `u ∈ [0,1]` 또는 누적 **호장 `s`** 유지. 재계획 시 새 곡선에 맞게 리셋·재투영.

3. **매 제어 주기 기하 연산** (신규 헬퍼로 두는 것이 일반적)  
   - 기체 위치에서 곡선에 **진행 방향을 고려한 투영**(전역 최근접만 쓰면 `u` 점프 가능).  
   - **Lookahead**: `u` 또는 `s`를 앞당긴 점에서 `splev`로 위치·**접선(der=1)** 평가.  
   - 속도·요 레이트: 접선 방향 추종만으로도 1차 근사 가능; **Pure Pursuit / Stanley** 등 횡편차 제어를 쓰면 추종 품질 향상.

4. **기존 함수 정리**  
   - `_path_advance_wp`, `_path_current_wp`, `_path_lookahead_wp`: 스플라인 모드에서는 폐기하거나 폴리라인 폴백 전용으로 분리.  
   - `_tick_global`: 위 기하·추종 법칙으로 재작성(글로벌 분기).

5. **시간 매개 궤적(논문 수준)**  
   - 위는 **기하 스플라인 추종**(2D + 기존 고도 홀드)에 가깝다. 속도·가속 **시간 프로파일**까지 스플라인상에 두려면 `u`↔`t` 매핑·명령 궤적 생성이 **추가**로 필요.

6. **안전**  
   - 스플라인 추종만으로 장애물 회피는 자동 보장 안 됨. 기존 LiDAR 감속·비상 제동·재계획(`_plan_path`)은 유지·튜닝.

---

## 2026-04-05: A* 병렬화 — 공유 상태 제거 및 ThreadPoolExecutor 도입

### 문제 상황
`_astar_diverse`가 서로 다른 팽창 반경(inflation radius)으로 여러 A* 경로를 탐색할 때
`_astar_with_inflation`이 `self._grid_data`를 임시로 교체하는 패턴을 사용:

```python
# 변경 전 — 병렬 실행 불가
orig = self._grid_data
try:
    self._grid_data = inflated   # 공유 상태 덮어쓰기
    return self._astar(...)
finally:
    self._grid_data = orig
```

여러 스레드가 동시에 `self._grid_data`를 교체하면 race condition이 발생해
잘못된 그리드에서 경로를 탐색하거나 크래시가 발생할 수 있었음.
결과적으로 `_astar_diverse`는 순차 실행 외에 선택지가 없었음.

### 수정 내용

| 함수 | 변경 내용 |
|------|-----------|
| `_is_plan_blocked(i, j, grid=None)` | `grid` 인자 추가. `None`이면 기존대로 `self._grid_data` 사용 |
| `_astar_neighbors(i, j, grid=None)` | `grid`를 `_is_plan_blocked`에 전달 |
| `_astar(start_ij, goal_ij, grid=None)` | `grid`를 `_astar_neighbors`에 전달. 함수 전체가 `self._grid_data` 무수정 |
| `_astar_with_inflation(...)` | `self._grid_data` 교체/복원 로직 제거. 팽창 그리드를 `_astar(grid=inflated)`로 직접 전달 |
| `_astar_diverse(...)` | `for` 순차 루프 → `ThreadPoolExecutor`로 교체. 각 radius를 독립 스레드에서 병렬 실행 |

### 변경 후 구조

```python
# _astar_with_inflation — self._grid_data 무수정
inflated = self._inflate_grid_temp(radius_cells)
return self._astar(start_ij, goal_ij, grid=inflated)   # 로컬 그리드 전달

# _astar_diverse — 병렬 실행
with ThreadPoolExecutor(max_workers=len(radii)) as executor:
    future_to_radius = {
        executor.submit(self._astar_with_inflation, start_ij, goal_ij, r): r
        for r in radii
    }
```

### 병렬성 주의사항
- Python GIL로 인해 순수 Python 코드(heapq, dict)는 진정한 CPU 병렬 실행이 아님
- 단, numpy 연산(`_inflate_grid_temp` 내 `binary_dilation`, `copy` 등)은 GIL을 부분 해제해 실질적인 중첩 실행 효과 있음
- **가장 중요한 효과**: `self._grid_data` 공유 상태 의존 제거 → race condition 없이 안전한 동시 실행 가능
- 진정한 CPU 병렬화가 필요하면 `ProcessPoolExecutor` + standalone 함수로 추가 전환 가능

### 기대 효과
- 팽창 반경 수(기본 3개)만큼의 A* 탐색을 동시에 시작 → 전체 경로 탐색 레이턴시 단축
- 기존 API 완전 호환 (인자 없이 호출 시 동작 동일)

---

## 2026-03-31: Homing timeout 근본 수정 — velocity setpoint 충돌 제거

### 문제 상황
- Homing이 20초마다 timeout → 드론이 y≈7에서 미션 시작 (목표 y=16)
- y=7에서 시작하면 A*가 inner_s_v1 우회 경로로 lower gap(남쪽 우회) 선택
- 드론이 최단 경로(upper gap 통과) 대신 35m 이상 우회

### 원인
`rl_offboard_node._tick()`에서 homing 대기 중(`wait_for_homing=True, _homing_done=False`)에도
`_publish(0,0,0,0)`으로 velocity setpoint를 20Hz로 전송.
`course_goal_publisher`는 동시에 position setpoint를 20Hz로 전송.
PX4는 두 setpoint를 번갈아 적용 → 드론이 위치 명령과 속도 0 명령을 교대로 받아 제자리에서 진동
→ 20초 내 homing 목표 도달 불가 → timeout.

### 수정
| 파일 | 변경 내용 |
|------|-----------|
| `px4_rl_offboard_mavros/rl_offboard_node.py` | `_tick()` 최상단에 homing 대기 중 즉시 return 추가 (velocity setpoint 전송 없음) |
| `px4_rl_offboard_mavros/rl_offboard_node.py` | 기존 `wait_for_homing and not _homing_done` 블록(dead code) 제거 |

### 기대 효과
- Homing 중 course_goal_publisher의 position setpoint만 PX4에 전달 → 드론이 (0,16)으로 정상 이동
- y=16에서 미션 시작 → A*가 inner_s_v1 upper gap(y>8) 통과하는 최단 경로 선택

---

## 2026-03-31: RL 오탐 근본 수정 — lidar_range_min_override + rl_min_dwell_sec 단축

### 문제 상황
- `lidar_sector_agg: "median"` 으로 변경했음에도 d_front=0.25m 오탐 지속
- 완전 개방 공간(x=-2.45, y=12.65)에서도 d_front=0.25 → RL 진입
- RL 10초 동안 드론이 잘못된 방향으로 이동 → inner_s_v1/v3 통로 진입 실패

### 원인 (확정)
- **LiDAR 자기 감지**: Iris 드론 body/frame이 LiDAR에서 0.24~0.31m에 위치
- Gazebo에서 자기 기체를 LiDAR가 감지 → `scan.range_min`(0.1m)으로는 필터링 안 됨
- 로터 암(±59°, ±123°)은 emergency brake 섹터(±30°) 밖이므로 brake 오발동과 무관
  - 암 자기감지는 PPO policy obs 오인식 및 RL 진입 섹터(rl_front_deg=90°) 오판에 영향
- median 집계여도 front 빔 다수가 0.25m이면 median=0.25

### 수정
| 파일 | 변경 내용 |
|------|-----------|
| `px4_rl_offboard_mavros/rl_offboard_node.py` | `_dist_in_sector`: `use_override=True` 파라미터 추가; emergency brake 호출 3곳은 `use_override=False` |
| `px4_rl_offboard_mavros/rl_offboard_node.py` | `_min_dist_in_sector`: `use_override=False` 고정 (비상 제동 전용) |
| `px4_rl_offboard_mavros/rl_offboard_node.py` | `__init__`: `lidar_range_min_override: 0.0` 파라미터 선언 추가 |
| `px4_rl_offboard_mavros/config/params.yaml` | `lidar_range_min_override: 0.35` 설정 (로터 암 0.25m보다 큰 값) |
| `px4_rl_offboard_mavros/config/params.yaml` | `rl_min_dwell_sec: 10.0 → 5.0` (오탐 시 잘못된 방향 이동 시간 단축) |

---

## 2026-03-31: A* 경로 실패 + RL 오탐 수정 — unknown block 비활성화 + lidar_sector_agg median

### 문제 상황
- A*가 반복 실패 (plan_unknown_block_radius_m 3.0으로 줄여도 동일)
- global planner → RL 전환이 d_front=0.80m로 오탐 (실제 3m 이내 장애물 없음)

### 원인 1: A* 차단 링 구조 문제
`plan_unknown_block_radius_m`는 `inner < d <= outer` 범위 unknown 셀을 blocked 처리.
**어떤 반지름이든** 이 링이 미탐색 구역(그림자, 초기 grid)으로 완전히 채워지면 A*가 내부 1.2m 원 밖으로 확장 불가 → 경로 탐색 실패.
RL이 local 회피를 담당하므로 이 기능 자체가 불필요.

### 원인 2: lidar_sector_agg "mean" 오탐
`_dist_in_sector`에서 `ranges > range_min` 필터 통과 후 mean 계산.
Gazebo에서 range_min=0.0일 경우 드론 기체/지면 클리핑으로 생기는 짧은 빔 값이 필터링 없이 포함.
mean은 소수의 이상 빔에도 전체 평균이 크게 떨어져 RL 오진입 유발.

### 수정
| 파일 | 변경 내용 |
|------|-----------|
| `px4_rl_offboard_mavros/config/params.yaml` | `plan_unknown_block_radius_m: 3.0 → 0.0` (기능 비활성화) |
| `px4_rl_offboard_mavros/config/params.yaml` | `lidar_sector_agg: "mean" → "median"` |

---

## 2026-03-31: A* 경로 실패 수정 — plan_unknown_block_radius_m 8.0 → 3.0

### 문제 상황
- A*가 초반부터 반복 실패하고 직선 fallback만 사용됨
- 로그: `Path fallback (straight): A* failed to find path` 연속 출력
- 드론이 장애물 사이 통로로 진입하지 못하고 직선으로만 이동

### 원인
`plan_unknown_block_radius_m: 8.0` 설정으로 인해, 로봇 주변 1.2~8.0m 범위의 unknown(-1) 셀이 A* blocked 처리됨.
- 고정 그리드(`rolling_window: false`) 초기에는 대부분 셀이 unknown
- 드론이 새로운 구역으로 이동할 때마다 전방 셀이 아직 LiDAR 미스캔 → unknown → blocked
- 결과: A*가 경로를 못 찾고 → 직선 fallback → 장애물 충돌 → RL 진입 반복

### 수정
| 파일 | 변경 내용 |
|------|-----------|
| `px4_rl_offboard_mavros/config/params.yaml` | `plan_unknown_block_radius_m: 8.0 → 3.0` |

### 근거
- LiDAR `max_range: 10m` 대비 8m는 과도하게 커서 미탐색 전방 셀 전부 차단
- 3.0m로 줄이면 바로 근처(1.2~3m)만 안전하게 막고, 그 너머는 A* 통과 가능
- 여전히 실패 시: `plan_unknown_block_radius_m: 0.0` (기능 비활성화) 시도

---

## 2026-03-31 (진행 중): Course L 횡보 문제 진단 - 우선진단 1, 3

### 문제 상황
- **목표**: Course L 미션에서 드론이 `inner_s_v1` ↔ `inner_s_v3` 통로(x≈-8, 음의 x 방향)를 통과
- **실제 동작**: `inner_s_v1` 장애물을 넘지 못하고 **x+ 방향(우회로)으로만 왕복** → 목표 미달성
- **중요 발견**: A* 경로는 생성되는데도 따르지 않음 → 문제는 **경로 계획이 아닌 제어**

### 추가된 디버그 로그 (rl_offboard_node.py)

#### 우선진단 1: Lookahead Waypoint 추적
**위치**: `_tick_global()` 함수 (line ~627 이후)

**추가 코드**:
```python
# A* 경로의 어느 WP를 목표로 삼고 있는지 추적
curr_wp = self._path_current_wp()
self.get_logger().debug(
    f"[GLOBAL_WP] curr_idx={self._path_wp_idx}, "
    f"curr_wp={curr_wp}, lookahead_wp={wp}, "
    f"robot=({px:.2f},{py:.2f}), path_len={len(self._path_xy)}"
)
```

**로그 해석**:
- ✅ `lookahead_wp` x ∈ (-10, -6) → 통로 올바르게 가리킴
- ❌ `lookahead_wp` x > 0 → 우회로를 목표로 함 (경로 오류)

#### 우선진단 3: Global Planner 명령값 로깅
**위치**: `_tick()` 함수에서 global planner 호출 직후 (line ~773 이후)

**추가 코드**:
```python
# Global planner가 내린 속도 명령 확인
self.get_logger().debug(
    f"[GLOBAL_CMD] vx={vx:.2f}, vy={vy:.2f}, yaw_rate={yaw_rate:.2f}, "
    f"d_front={d_front:.2f}, d_min_sectors={d_min_sectors:.2f}"
)
```

#### 추가: 디버그 로그 주기 제어 (1초)
- `RLOffboardNode.__init__`에 `self._t_last_debug_log = 0.0` 추가
- `_tick_global()` 시그니처를 `(..., now: float)`로 변경
- `[GLOBAL_WP]` / `[GLOBAL_CMD]` 로그를 `if now - self._t_last_debug_log >= 1.0:`로 감싸 1초마다만 출력
- 로그 출력 시 `self._t_last_debug_log = now`로 갱신

이 변경으로 20Hz 루프 내부에서 디버그 메시지 출력 부하를 줄이고, 로그 분석 시 과도한 중복 줄을 방지할 수 있습니다.

**로그 해석**:
- ✅ `vx < 0, |vy| << |vx|` → 음의 x(통로) 방향 진입 명령
- ❌ `vx > 0` 또는 `vx ≈ 0` → 우회로 또는 정체 명령

### params.yaml 관련 주요 파라미터

| 파라미터 | 값 | 역할 |
|---------|-----|------|
| `path_lookahead_steps` | 3 | 현재 WP로부터 N칸 앞을 목표로 설정 |
| `rl_trigger_front_enter` | 2.0 | RL 전환 임계값(m) |
| `rl_trigger_front_exit` | 4.0 | Global 복귀 임계값(m) |
| `rl_min_dwell_sec` | 10.0 | RL 최소 유지 시간(s) |
| `planner_slowdown_dist` | 2.0 | 장애물 근접 시 감속 임계값(m) |

### 실행 방법

```bash
# 터미널 1: 노드 실행 (DEBUG 레벨 활성화)
ros2 run px4_rl_offboard_mavros rl_offboard_node --ros-args --log-level rl_offboard_node:=DEBUG 2>&1 | tee ~/diagnosis.log

# 터미널 2: 실시간 로그 모니터링
tail -f ~/diagnosis.log | grep -E "\[GLOBAL_WP\]|\[GLOBAL_CMD\]|→ RL|→ global"
```

### 로그 분석 (추후)

```bash
# Lookahead WP 추적
grep "\[GLOBAL_WP\]" ~/diagnosis.log | head -30

# Global planner 명령값
grep "\[GLOBAL_CMD\]" ~/diagnosis.log | head -30

# 모드 전환 통합 보기
grep -E "\[GLOBAL_WP\]|\[GLOBAL_CMD\]|→ RL|→ global" ~/diagnosis.log | head -50
```

### 해결 후 정리 사항
- [ ] rl_offboard_node.py: `[GLOBAL_WP]`, `[GLOBAL_CMD]` 로그 삭제
- [ ] CHANGELOG.md: 본 섹션 삭제 (정상 동작 섹션으로 대체)

---
```

**확인 사항**:
- `inner_s_v1` (x=-8, y∈[-8,8]) 영역이 **occupied(검은색 또는 100)**으로 표시되는가?
- Grid bounds가 설정한 대로인가? (origin -22, -22; width/height 441 → x,y ∈ [-22, 22])
- A* 경로가 **회피 우회로**(예: x > -8)로만 그려지는가?

#### STEP 3: Start/Goal Cell 위치 상세 로그 추가
현재 코드에서 `[ASTAR]` 로그는 start_xy → ij 변환만 보여줌.  
**다음을 추가 로그로 확인**:

```python
# _plan_path() 내에 추가
self.get_logger().info(
    f"[ASTAR_DETAIL] start_ij={s_ij}, cell_val={int(self._grid_data[s_ij]) if s_ij else 'None'}, "
    f"goal_ij={g_ij}, cell_val={int(self._grid_data[g_ij]) if g_ij else 'None'}, "
    f"grid_info: {self._grid_w}x{self._grid_h}, res={self._grid_res}"
)
```

이를 통해:
- Start cell이 정말 occupied가 아닌지 확인
- Goal cell이 grid 안에 있는지 확인

#### STEP 4: 관찰 기반 원인 특정

**Case A: "Path fallback" 로그 반복**
```
Path fallback (straight): A* failed to find path
→ 원인: A*가 경로를 못 찾음
```
**해결책**:
- Occupancy Grid 해상도 확인 (0.1 m): 너무 크면 0.05 m로 줄이기
- `occ_threshold` 확인 (`grid_fuser.yaml`): 점유 기준치가 너무 높으면(예: 100) 50으로 낮추기
- `plan_unknown_block_radius_m` 확인: Unknown 막힘이 과도하면 0으로 설정

**Case B: "A* succeeded" 로그는 나오는데 실제로 틀린 경로**
```
A* succeeded: 3 wps, start_ij=(100, 150), goal_ij=(50, 50)
→ waypoint가 3개뿐? 직선에 가깝다면 grid 막힘 많음
```
**해결책**:
- RViz grid 시각화: inner_s_v1 위치가 실제 장애물과 다른가?
- Grid 좌표 변환 검증:
  ```
  inner_s_v1: x=-8 → ij_j = floor((-8 - (-22.0)) / 0.1) = floor(14/0.1) = 140
  이것이 실제인지 로그로 확인
  ```

**Case C: "global planner → RL → global planner" 반복**
```
로그가:
  global planner → RL (d_front=1.8)
  RL → global planner (d_front=3.2)
  global planner → RL (d_front=1.5)  ← 반복!
→ 원인: 경로는 맞는데 제어가 불안정 (히스테리시스 범위 문제)
```
**해결책**:
- `rl_trigger_front_enter`, `rl_trigger_front_exit` 간격 벌리기  
  현재: 2.0 m → 3.0 m, 변경: 1.5 m → 4.0 m
- `rl_min_dwell_sec` 값 확인 (기본 5 s): 너무 짧으면 10 s로 늘리기

#### STEP 5: 로그 파일 분석 (Post-run)
```bash
# astar_debug.log에서 ASTAR 관련만 추출
grep "\[ASTAR\]" ~/astar_debug.log

# 모드 전환 기록
grep "global planner\|RL" ~/astar_debug.log | head -50

# 실패 패턴 분석
grep "Path fallback\|A* failed" ~/astar_debug.log | wc -l
```

### 권장 즉각적 조치 (3가지)

1. **Grid 시각화 우선**
   ```bash
   rviz2 &
   # Add → OccupancyGrid (/local_grid)
   → 맵이 실제와 맞는지 육안 확인 (이게 가장 빨름!)
   ```

2. **[ASTAR] 로그 확인**
   ```bash
   ros2 launch ... 2>&1 | grep "\[ASTAR\]" -A 2
   ```
   경로가 왜 실패하는지 직접 메시지로 확인.

3. **Fallback인지 A* 성공인지 판단**
   ```bash
   ros2 launch ... 2>&1 | grep "fallback\|succeeded" | tail -10
   ```
   Fallback 반복 → grid 문제  
   Succeeded 반복 → 제어/히스테리시스 문제

---

## 2026-03-31: A* 경로 생성 디버깅 (start cell occupied 체크 + grid 좌표축 검증)

### 요약
A* 알고리즘이 초기에 최단경로를 제대로 생성하지 못하는 문제 해결:
1. **Start cell occupied 체크 추가** — goal과 마찬가지로 start cell도 occupied 여부를 체크하여 fallback 처리
2. **Grid 좌표축 검증** — `_xy_to_ij()` 및 `_ij_to_xy()` 변환 검증 로그 추가
3. **디버그 로그 강화** — 실제 start/goal 좌표, cell index, grid 경계 정보 출력

### 세부 내용

#### Start Cell Occupied 체크
```python
# 이전: goal cell만 checked
elif self._is_occ(*g_ij):
    ...

# 수정: start cell도 체크
elif self._is_occ(*s_ij):
    ...
elif self._is_occ(*g_ij):
    ...
```
**배경**: Start cell이 occupied일 경우 A* 첫 번째 iteration에서 이웃이 없어 경로 탐색 실패. 직선 경로(fallback)만 사용하게 됨.

#### Grid 콜백 디버그 로그
`_cb_grid()` 수신 시 grid 정보 로그:
- Grid 크기 (WxH), 해상도, origin
- X/Y 경계(bounds) — 좌표 변환 검증용

#### A* 경로 계획 로그 강화
`_plan_path()` 호출 시:
- `[ASTAR]` 디버그 레벨: start/goal 좌표 → cell index 변환 결과
- `[WARN]` fallback 원인: grid 경계, occupied 여부(cell value 포함), A* 실패
- `[INFO]` 성공: 경로 길이, start_ij, goal_ij

### 변경 파일
| 파일 | 내용 |
|------|------|
| `px4_rl_offboard_mavros/rl_offboard_node.py` | `_cb_grid()`: grid 크기/origin/bounds 로그 추가; `_plan_path()`: start cell occupied 체크 추가, 상세 디버그/경고 로그 추가 |

### 큐링 팁
로그 필터링:
```bash
# 디버그 레벨 활성화 (ROS2)
ros2 run ... --ros-args --log-level rl_offboard_node:=DEBUG

# grep로 필터링
ros2 launch ... 2>&1 | grep "\[ASTAR\]"  # 경로 계획 시도
ros2 launch ... 2>&1 | grep "\[GRID\]"   # grid 정보
```

---

## 2026-03-31: hybrid RL dwell + grid_fuser 고정 그리드 + LiDAR 구간 집계

> 최초 드래프트에서 `grid_fuser`만 세션의 `Today` 메타(2026-03-30)로 표기되어 있었음 → **실제 정리일(2026-03-31)** 로 통일.

### hybrid RL 최소 유지 시간 (`rl_min_dwell_sec`)

#### 요약
- **`use_hybrid: true`** 일 때, **global planner → RL** 로 넘어간 뒤 **`rl_min_dwell_sec`**(기본 **5 s**) 안에는 전방 거리가 exit 조건을 만족해도 **RL을 유지**해 **RL ↔ global 빠른 전환**을 줄임.
- 진입 순간을 `_t_rl_entered`로 기록; dwell이 끝난 뒤에만 기존 히스테리시스대로 global 복귀. **`rl_min_dwell_sec: 0`** 이면 이 동작을 끄고 예전과 동일.

#### 변경 파일
| 파일 | 내용 |
|------|------|
| `px4_rl_offboard_mavros/rl_offboard_node.py` | 파라미터 `rl_min_dwell_sec`; `use_rl_raw` 계산 후 dwell 적용; 상태 `_t_rl_entered` |
| `px4_rl_offboard_mavros/config/params.yaml` | `rl_min_dwell_sec: 5.0` 및 주석 |

### grid_fuser 고정 그리드 (hybrid A* goal outside grid 완화)

#### 요약
- `grid_fuser`를 **롤링 윈도우 비활성화**(`rolling_window: false`)로 바꿔, `/local_grid`가 **odom 기준 고정 영역**을 덮도록 함. 기체 중심으로만 움직이던 40 m 창 때문에 goal이 그리드 밖으로 나가 **`Path fallback (straight): goal … outside grid`** 가 반복되던 문제를 줄이기 위함.
- 그리드 범위는 `px4_rl_offboard_mavros/config/params.yaml`의 **`map_x/y_min/max` ±22 m**와 맞춤. `resolution` 0.1 m일 때 좌표 **+22**가 셀 인덱스 440이 되므로 **`width`/`height` 441** (440이면 경계 goal이 그리드 밖 처리될 수 있음).
- 원점은 해당 월드의 남서쪽에 대응하도록 **`origin_x` / `origin_y`: -22** (스폰·goal이 동일 범위 안이라는 전제).
- 이 설정은 **`px4_nav_perception/launch/s3_grid_fuser.launch.py`** 로 올리는 `grid_fuser`에 `config/grid_fuser.yaml`을 넘기는 현재 구성에 적용됨.

#### 변경 파일
| 파일 | 내용 |
|------|------|
| `px4_nav_perception/config/grid_fuser.yaml` | `rolling_window: false`; `origin_x/y`: -22; `width`/`height`: 441; `max_range_no_hit_epsilon`; 주석 정리 |

#### 참고
- 고정 맵에서 스캔이 닿지 않은 셀은 **unknown**으로 남을 수 있음. A*·`occ_threshold` 동작에 따라 경로 품질이 달라질 수 있으므로, 문제가 있으면 threshold·인플레이션·재관측을 별도 튜닝.
- **후속 (`grid_fuser`)**: 레이가 **유효 최대거리 근처**(오픈)일 때도 끝점을 **occupied**로 찍던 동작을 수정. `max_range_no_hit_epsilon` 이내면 끝 셀을 **free**로만 두어, 뚫린 구역이 unknown/가짜 장애물 고리로 보이는 현상을 줄임.
- 이전 2026-03-29 항목의 “미변경”에 있던 `grid_fuser` 튜닝 항목은 **본 변경으로 일부 반영됨**.

### LiDAR 정면 섹터 집계 (`lidar_sector_agg`, RL/global·감속용 `d_front`)

#### 요약
- **정면(및 좌·우 섹터)** 거리를 셀 때 구간 내 모든 빔에 대해 **`min`만** 쓰면, 지면·메시 스치는 **한두 빔**만 짧아도 `d_front`가 과소가 되어 **RL 과진입**이 나기 쉬움.
- **`lidar_sector_agg`**: `min` | `median`(중앙값) | **`mean`(평균, 기본값)** — RL↔global 히스테리시스, 전환 로그의 `d_front`, `d_min_sectors`(planner 감속·goal bias 등)에 적용.
- **비상 제동(`emergency_brake`)** 은 집계와 무관하게 **항상 구간 `min`** 유지.
- **후속**: 시뮬에서 `median`이 체감과 잘 안 맞아 **기본 집계를 `mean`으로 변경** (노드 기본값·`params.yaml`).

#### 변경 파일
| 파일 | 내용 |
|------|------|
| `px4_rl_offboard_mavros/rl_offboard_node.py` | `_dist_in_sector(..., agg)`; `lidar_sector_agg` 파라미터(기본 `mean`); 비상은 `min`, 모드 전환은 설정 집계 |
| `px4_rl_offboard_mavros/config/params.yaml` | `lidar_sector_agg: "mean"` 및 주석 |

### A* unknown 국소 막힘 (`plan_unknown_block_radius_m`)

#### 요약
- 미리 장애물 지도를 주지 않는 전제에서, **전역 unknown=막힘** 대신 **기체 주변의 일정 구간 unknown만** A*에서 막힘으로 취급.
- **시작·목표 격자 셀**은 `occ_threshold` 이상일 때만 막고, unknown만으로는 막지 않음.
- **도넛 구간**: 로봇~셀 거리가 **`plan_unknown_block_inner_margin_m` 초과**이고 **`plan_unknown_block_radius_m` 이하**일 때만 unknown 막힘. 그 안쪽은 막지 않아 **시작 격자 이웃이 전부 unknown일 때 A*가 한 칸도 못 나가는 문제**를 피함.
- `plan_unknown_block_radius_m: 0` 이면 기존과 동일(unknown 통과).

#### 변경 파일
| 파일 | 내용 |
|------|------|
| `px4_rl_offboard_mavros/rl_offboard_node.py` | `_is_plan_blocked`; A* 인접 셀 판정; `_plan_path`에서 계획 컨텍스트 try/finally |
| `px4_rl_offboard_mavros/config/params.yaml` | `plan_unknown_block_radius_m`, `plan_unknown_block_inner_margin_m` |

---

## 2026-03-29: obstacle_course_light 2배 월드 (SITL + goal + 맵 경계)

### 요약
- Gazebo: `sitl_gazebo-classic/worlds/obstacle_course_light_2x.world` (XY 2배, 장애물 높이 z=4 유지)
- **SITL 월드**: 이 저장소에서는 `px4_ws` 쪽만 갱신. 실행 시 `PX4_SITL_WORLD`(또는 `sitl_run.sh` 인자)로 **`…/obstacle_course_light_2x.world` 절대경로** 지정 필요 (예: `start_px4.sh` 기본값을 해당 파일로 바꾸기)
- Navigator: homing/goal 상대 좌표를 2배에 맞춤
- RL offboard: 맵 경계 ±10 m → ±22 m (`enable_map_bounds` 가 2배 코스에서 너무 일찍 걸리지 않도록)

### 변경 파일 (px4_ws / 이번 커밋)
| 파일 | 내용 |
|------|------|
| `px4_nav_planner/config/goals.yaml` | `home_xyz`, `goal_R`, `goal_L` 의 x,y 를 light 대비 2배 (z=2.5 유지) |
| `px4_rl_offboard_mavros/config/params.yaml` | `map_x/y_min/max`: ±10 → ±22 |

### 관련 (PX4-Autopilot 트리, 별도 반영)
- `Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/obstacle_course_light_2x.world` — 2배 코스 SDF

### 미변경 (필수 아님·별도 검토)
- `world_size: 20.0` (정책/obs_norm과 학습 스케일 일치용). 코스만 키운 경우 **관측 분포 차이** 가능 → 장기적으로는 `world_size` 40 + 재학습·export 검토.
- `grid_fuser` 고정 그리드·범위는 **2026-03-31** 항목 참고. inflation·planner 등은 여전히 별도 튜닝 가능.

---

## 2025-03-16: 완전 Body Frame Obs (goal_rel body + heading 제거)

### 요약
- **obs_dim**: 68 → 66
- **goal_rel**: world frame → body frame (전방=x+, 좌측=y+)
- **heading**: 제거 (cos/sin yaw)
- **이전 모델/VecNormalize 재사용 불가** — 재학습 후 export 필요

### 변경 파일 및 내용

#### 1. `rl/env_2d_nav.py` (학습 환경)
- `observation_space`: shape (68,) → (66,)
- `_get_obs()`:
  - goal_rel: `(goal-pos)/world_size` (world) → `R(-yaw)@(goal-pos)/world_size` (body)
  - heading (cos, sin yaw) 제거
- docstring: State(observation) 설명 갱신

#### 2. `rl/export_policy.py` (정책 내보내기)
- `obs_ordering`: heading 제거, goal_rel env_norm을 body frame으로 수정
- `expected_obs_dim`: `obs_beams + 4` → `obs_beams + 2`
- `env_norm_example()`: raw_obs `n+4` → `n+2`, heading 처리 제거
- `verify_full_pipeline`: raw_obs 생성 시 heading 제거
- 안내 출력: ROS2 inference flow 문구 갱신

#### 3. `px4_rl_offboard_mavros/rl_offboard_node.py` (배포 노드)
- `obs_dim` 검사: `obs_beams + 4` → `obs_beams + 2`
- `_build_obs()`:
  - goal_rel: world → body (`R(-yaw) @ (goal-pos)/world_size`)
  - heading 제거
- docstring: obs 구성 설명 갱신

#### 4. `px4_rl_offboard_mavros/models/obs_norm.json`
- `obs_dim`: 68 → 66
- `ordering`: heading 제거, goal_rel env_norm body frame으로 수정
- `mean`, `std`: 66개로 축소 (lidar 64 + goal_rel 2)
- **주의**: 현재 mean/std는 placeholder. 재학습 후 `export_policy.py`로 덮어써야 함.

#### 5. `px4_rl_offboard_mavros/models/policy_ts.pt`
- **반드시 재학습 후 export로 교체 필요.** 기존 68-dim 정책은 사용 불가.

### 재학습 및 배포 절차
1. `rl/env_2d_nav.py` 사용해 PPO 학습 (train_ppo_2d.py 등)
2. `rl/export_policy.py`로 새 policy_ts.pt, obs_norm.json 생성
3. `px4_rl_offboard_mavros/models/`에 복사 후 노드 실행

---

