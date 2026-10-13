# PX4 SIM 장애물 회피 벤치마크 결과 (astar / hybrid / ppo)

> **출처·한계**: 원본 CSV(`px4_sim_worlds/benchmark/YYYYMMDD_N.csv`)와 벤치마크 스크립트는 2026-10-13 `deploy_to_px4.sh` 의 `rsync --delete` 로 PX4 트리에서 삭제되어 복구하지 못했다.
> 아래 수치는 당시 분석 세션이 남긴 메모리 요약(`~/.claude/projects/-home-akswosla-PX4-Autopilot/memory/obstacle-avoidance-benchmark.md`)에서 옮긴 것이다. **trial 단위 원본 데이터는 없으므로** 새로 통계 검정을 하거나 수치를 재계산할 수 없다.
> 남아 있는 원본: rosbag 6개 (`~/PX4-Autopilot/px4_ws/src/px4_sim_worlds/benchmark/bags/2026081{9,20}_*`).

## 실험 환경

- 월드: `Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/dynamic_obstacle_benchmark.world`. home=(0,−9,4) → goal=(0,9,4), x=0 직선 18 m 코스, 20×20 m 경계벽.
- 판정(`trial_monitor.py`):
  - COLLISION 은 contact 센서(`/drone_bumper`)로 판정하고, `reason` 에 WALL / CRASH / OBSTACLE:<name> 을 남긴다(2026-08-13 부터).
  - SUCCESS 는 goal 반경 도달, TIMEOUT 은 homing 후 제한시간 초과다.
  - `min_obstacle_dist` 는 정보용이며 판정에는 쓰지 않는다.
- CSV 컬럼: `algo, trial, outcome, reason, elapsed_sec, min_obstacle_dist, min_altitude, timestamp, entry_sim_t, phase_ok`.
- **짝비교(paired) 인프라** (2026-08-12 도입): trial N 은 세 알고리즘 모두 같은 진입위상(`entry_sim_t`, `phase_seeds.txt`)을 쓴다. 따라서 08-12 이전 run(0810·0811)은 이후 run 과 직접 비교할 수 없다(clearance 0.15→0.1 변경 포함).
- 공통 설정: 08-13_3 이후 모든 run 에서 `astar_min_inflation_cells=0`. astar 는 `emergency_brake_dist=0`(live-LiDAR fallback 없음)이며, 이는 의도된 ablation baseline 이다.

### 레이아웃

| 레이아웃 | 구성 | 해당 run |
|---|---|---|
| **L0 (원본)** | rows=4, 동적 10 + 정적 2 (`static2_center`, `static4_center`), 총 12 | 0813_3, 0814_1, 0816_2, 0817_1, 0818_1 (0810·0811 도 동적 10 + 정적 2 의 12 장애물 환경) |
| **L1** | 동적 9 + 정적 3. dt1b → `static1_center` 로 전환하고 dt1a/dt1c 범위를 확장 | 0815_1 |
| **L2** | 동적 9 + 정적 2. `static1_center` 를 삭제하고 dt1a [−9.6,−0.25] / dt1c [0.25,9.6] 로 확장 | 0816_1 |

## 성공률 요약

| CSV | 레이아웃 | n (algo당) | astar | hybrid | ppo | 비고 |
|---|---|:---:|:---:|:---:|:---:|---|
| 20260810_4 | (pre-gate) | 10 | 10% | 90% | 20% | 소표본 노이즈, 보고 제외 |
| 20260810_5 | (pre-gate) | 50 | 36% | 46% | 22% | |
| 20260811_1 | (pre-gate) | 100 | 45% | 45% | 26% | |
| 20260813_3 | L0 | 100 | 32% | 46% | 17% | 첫 paired run |
| 20260814_1 | L0 | 100 | 20% | 47% | 16% | 직전 run 대비 `grid_fuser` `decay_rate` 0.01→0.1 |
| 20260815_1 | L1 | 100 | 21% | 37% | 15% | 총 10h43m |
| 20260816_1 | L2 | 100 | 37% | 44% | 18% | 총 9h0m42s |
| 20260816_2 | L0 | 100 | 24% | 44% | 20% | 재현성 run, 총 4h0m17s (중간 RESUME 2회) |
| 20260817_1 | L0 | 100 | 25% | **51%** | 16% | `astar_min_inflation_cells` 0→1 변경 전 **"before" 기준선** |
| 20260818_1 | L0 | 100 | 24% | **50%** | 21% | 위와 동일 (기준선) |

→ paired run 7회(0813_3 ~ 0818_1) 모두 **hybrid 성공률 1위**.

## 성공 trial 의 평균 clearance / 소요시간

| CSV | clearance 평균 (m): hybrid / astar / ppo | 성공 평균 elapsed (s): astar / hybrid / ppo |
|---|---|---|
| 20260813_3 | 0.541 / 0.353 / 0.393 | — |
| 20260815_1 | 0.57 / 0.40 / 0.38 | 28.6 / 28.7 / 39.7 |
| 20260816_1 | 0.55 / 0.39 / 0.40 | — |
| 20260816_2 | 0.566 / 0.386 / 0.371 | 29.8 / 28.2 / 38.8 |

→ clearance 는 매 run **hybrid 가 최대**. 0813_3 기준 성공 trial 중 0.4 m 미만 비율은 astar 78%, ppo 62%.

## 실패 원인 (주요 충돌 장애물)

| CSV | astar | hybrid | ppo |
|---|---|---|---|
| 20260811_1 | dt1b 19, dt1a 7 (진입 직후 동적) | dt1b 16, dt3b 6 (분산) | static4_center 28/70, WALL 11, TIMEOUT 4 |
| 20260813_3 | static2_center 41/68 | dt4c 15/53, CRASH 1·TIMEOUT 1 | static4_center 26/83, WALL 16/83 |
| 20260814_1 | static2_center ~52%, dt1b ~26% (충돌 중 비율) | (기록 없음) | static4_center ~35–39%, WALL ~14–19% |
| 20260815_1 | static1_center 56/79 (전체 trial 의 56%) | static1_center 16, dt2c 15, dt3b 10, dt4c 8, dt1c 5 | static4_center 25, dt3b 17, WALL 10 |
| 20260816_1 | static2_center 38, dt1a/dt1c 8 | dt3b 17, dt2c 9, dt4c 7, dt1a/dt1c 10 | static4_center 29, dt3b 15, WALL 14, dt2c 13 |
| 20260816_2 | static2_center 40/76, dt1b 21 | dt1b 19, dt4c 10, static2_center 7, dt3b 6, dt2c 6, **WALL 0** | static4_center 20, WALL 20, dt1b 17 |

## 정리된 결론 (당시 분석 기준)

- **astar**:
  - 매 run "홈에서 가장 가까운 정적 장애물"(static1_center 또는 static2_center)에서 충돌이 집중된다.
  - live-LiDAR fallback 이 없고(`emergency_brake_dist=0`), 경로-장애물 여유가 grid_fuser dilation(~0.2–0.4 m)뿐인 구조 때문으로 해석했다.
- **ppo**:
  - `static4_center`(goal 직전 3 m, 경로 정중앙의 2.0 m 정적 장애물)에서 충돌이 반복된다. 같은 장애물에서 astar 는 1–7%, hybrid 는 1–2% 에 그친다. 전역 경로 없는 local minima 의 근거로 정리했다.
  - WALL 충돌이 매 run 재현되며, 원인은 미확정이다.
- **hybrid**:
  - 실패 원인이 특정 장애물에 몰리지 않고 분산된다. WALL 충돌은 거의 없다.
  - 성공률과 clearance 모두 매 run 1위다.
- **열린 항목**:
  - 0817_1/0818_1 에서 hybrid 가 50% 대로 오른 원인
  - ppo WALL 충돌 메커니즘
  - `astar_min_inflation_cells` 0→1 변경 후 비교(after run 결과는 기록 없음)
