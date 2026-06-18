# experiments — 실험 이력 · 학습 로그 인덱스

`env_2d_nav.py` / `train_ppo_2d.py` 기반 UAV 2D 내비게이션 RL 의 **최종 모델 채택에 영향을 준 핵심 실험** 과, 그 정량 근거가 되는 로그·텐서보드의 위치를 정리한다. MDP·관측·보상의 정식 정의는 [CONCEPTS.md](CONCEPTS.md) 참조.

> 보상 항별 ablation 의 raw 데이터(Case1~4·CK·E1·E2 모델·로그·CSV), 전체 학습 로그·텐서보드, 원본 개발 저널(`MODIFICATIONS_full.md`)은 모두 `_archive_ablation/` 에 보존되어 있으며 용량 때문에 git 추적에서 제외된다(`.gitignore`). 본 문서는 그중 설계 결정의 근거가 된 실험만 발췌·재구성한 것이다.

---

## 1. 최종 채택 모델 요약

| 축 | 채택 구성 | 근거 |
|----|-----------|------|
| **보상** | **C2 = BFS-progress + success bonus + collision penalty** (3 항, minimal) | §2 ablation 전반 — 추가 shaping 항은 marginal/negative |
| **관측** | 64-beam LiDAR(min 다운샘플, [-1,1]) + body-frame goal 상대 2 = **66-dim**. hybrid 는 goal 대신 **A\* 1 m lookahead waypoint** 주입 | §2.4 — A\* lookahead 가 PPO 부분관측 보완 → 수렴 속도·고밀도 충돌 안정성 우위(채택) |
| **역학** | dt=**0.05** s (배포 20 Hz setpoint 정합), PPO yaw steering(unicycle), max_steps=1800 | §2.5 — 배포 동역학 일치 |
| **학습** | 비대칭 AdaptiveEntropy(base **0.004** / boost **0.03**) + entropy 천장 가드, γ=**0.9983** | §2.6 — std 폭주 억제 + 빠른 수렴 동시 확보 |

> 위 표의 **C2** 는 보상 *구성* 이름(C0~C7 ablation 의 C2 = BFS+success+collision)이다. 학습 *run* 파일명은 이 문서에서 `hybrid_*` / `PPOonly_*` 로 통일했다(§4 참조).

### 배포 산출물

`export_policy.py` 가 best_model 을 TorchScript + obs 정규화로 변환한다.

| rl/ export 디렉터리 | 변환 모델 | 배포 위치 (`px4_rl_offboard_mavros/models/`) | 설명 |
|---------------------|-----------|----------------------------------------------|------|
| `export_hybrid_dt05/` | `policy_c2dt05_ts.pt` | `policy_hybrid_ts.pt` / `obs_norm_hybrid.json` | **현행 배포 후보** — C2 hybrid, dt=0.05 정합 |
| `export_ppo_dt05/` | `policy_ppo_c2dt05_ts.pt` | `policy_ppo_ts.pt` / `obs_norm_ppo.json` | PPO-only dt=0.05 (A\*/BFS 제거 비교용) |

> 배포 모드는 `rl_offboard_node.py` 가 `policy_mode`(hybrid/ppo)로 선택한다([config/params.yaml](../px4_ws/src/px4_rl_offboard_mavros/config/params.yaml)). dt=0.3 10M reference 모델(구 `export_C2_10M` / `export_C2_PPOonly`)은 배포 동역학과 불일치하여 제거했고, 해당 학습 로그만 아카이브에 남는다(§2.4).

---

## 2. 핵심 실험 (최종 설계의 정량 근거)

### 2.1 보상 항 정의 (A0–A6, D1)

| 항 | 변수 | 의미 |
|----|------|------|
| **D1** | `collision_penalty=500` | 충돌 페널티 — 모든 config 공통 base infra |
| **A0** | `w_progress=50` | **BFS-progress**: 격자 최단경로 거리의 스텝별 차분(우회 압력) |
| **A1** | `w_time_pressure` | 시간 압박 (−w·d_goal/L) |
| **A2** | `w_spin` | 스핀 페널티 (−w·\|yaw_rate\|) |
| **A3** | `w_no_progress(+soft)` | no-progress terminal + 누적 ramp |
| **A4** | `w_timeout` | timeout terminal |
| **A5** | `success_reward=800` | 성공 보너스 (sparse anchor) |
| **A6** | `w_danger`+`w_close` | 안전거리 내부 지수형+선형 근접 페널티 |

### 2.2 누적 보상 ablation (8 config × 3 seed × 4 M)

각 항을 누적 추가(C0=전부 OFF → C7=full)하며 isolated 기여도 측정. best_model weighted_score(3 seed mean):

| Config | 구성 | weighted | Δ(직전 대비) |
|--------|------|:--------:|:----:|
| C0 | D1 only | 0.380 | — |
| **C1** | **+A0(BFS)** | **0.911** | **+0.531 (압도적)** |
| C2 | +A5(success) | 0.928 | +0.017 |
| C3 | +A3(no-prog) | 0.927 | −0.001 |
| C4 | +A4(timeout) | 0.929 | +0.002 |
| C5 | +A6(obstacle) | **0.899** | **−0.030 (negative)** |
| C6 | +A1(time) | 0.917 | +0.018 (A6 부작용 완화) |
| C7 | +A2(spin), full | 0.924 | −0.010 |

**결론**: BFS-progress(A0)가 dominant contributor. 나머지 shaping 항(A1·A3·A4·A6)은 marginal 또는 negative — 특히 **A6 근접 페널티는 obs_0 에서 LiDAR oscillation 으로 no_progress 1%→16% 유발**. → 보상을 **C2(BFS+success+collision)** 로 최소화 결정.

### 2.3 보상 설계 정당화 (KIMST short paper ablation)

**(a) BFS vs Euclidean vs Chikhaoui 원식 재현 (CK)** — final eval weighted(3 seed):

| Config | progress 신호 | weighted |
|--------|---------------|:--------:|
| C1 (ours) | BFS-progress | **0.920** |
| E1 | Euclidean(연속) | 0.911 |
| E2 | Euclidean(격자) | 0.914 |
| **CK** | **Chikhaoui 2022 원식** | **0.086 (붕괴)** |

→ CK 는 모든 장애물 시나리오에서 reached ≤22%, collision 78~100% — **perverse incentive**(proximity penalty 누적 회피 위해 빠른 충돌 선호). 본 연구의 dense obstacle 분포(Chikhaoui 환경의 ~5000× 밀도)에서 원식 부적합. BFS≈Euclidean 이나, BFS 는 우회 시 신호 일관성이 높음.

**(b) 5-case isolated reward ablation** — final eval weighted(3 seed):

| Case | 구성 | weighted | ours 대비 |
|------|------|:--------:|:--------:|
| 1 | collision+A6+A1 (no BFS) | 0.733 | −18.7 %p |
| 2 | BFS+collision+A6+A1 (full Chikhaoui-like) | 0.901 | −1.9 %p |
| 3 | BFS+collision+A6 | 0.910 | −1.0 %p |
| 4 | BFS+collision+A1 | 0.889 | −3.1 %p |
| **5 (ours)** | **BFS+collision** | **0.920** | **0 (최고)** |

→ **① BFS 도입 +23 %p (dominant)**, **② A6 LiDAR 근접 페널티 −1 %p**(obs_0 oscillation), **③ A1 시간 압박 −3.1 %p**(장애물 환경 무모한 직진 → collision +7~8 %p). 최소 보상이 5-case 중 최고.

### 2.4 hybrid vs PPO-only (A\* 전역 경로 prior 효과, 10 M)

논문의 핵심 비교 — A\* 전역 경로를 PPO 관측에 주입한 하이브리드가 PPO 단독의 부분관측 한계를 보완하는지. 동일 보상·평가 프로토콜에서 weighted_score 와 고난이도(d15/d20) 충돌률 추이:

| step | weighted: `hybrid_dt3` / `PPOonly_dt3` | 충돌률(d15·d20 평균): hybrid / PPO-only |
|------|:---:|:---:|
| 1M | **0.80** / 0.51 | **34 %** / 64 % |
| 2M | **0.90** / 0.77 | **22 %** / 42 % |
| 3M | **0.91** / 0.84 | **21 %** / 32 % |
| 4M | **0.93** / 0.86 | **15 %** / 29 % |
| 최종(10M) | **0.9472** / 0.9412 | **9.5 %** / 12 % |

→ **하이브리드 우위**: ① 초반(1M~4M) **수렴 속도** 압도적(1M weighted 0.80 vs 0.51), ② 고밀도 **충돌률** 전 구간 낮음, ③ 최종 weighted 는 근소 우위(+0.6 %p). A\* 전역 경로 obs 가 센서 범위 밖 장애물에 대해서도 우회 방향을 조기 제공해 표본 효율·안전성을 높인 결과(CONCEPTS.md §3.1). 3D Gazebo 에서도 hybrid 81.3 % vs A\* 46.9 % vs PPO 25 %(보고서 Table 2). (로그: `_archive_ablation/logs/hybrid_dt3.log`, `PPOonly_dt3.log`)

> hybrid_dt3 는 4M(C2_seed42 from-scratch) + 6M(이어학습) = 10M 구조이며, 위 1M~4M 수렴 비교는 두 run 모두 from-scratch 구간의 직접 대조다.

### 2.5 학습 env 를 배포 동역학과 정합 (dt=0.05 재설계)

- **문제**: 배포 hybrid 는 yaw 를 A\* 기하로 산출하고 PPO yaw 를 버리는데, 학습 env 는 PPO yaw 를 적분 → v_cmd 가 배포와 다른 동역학에서 학습됨.
- **1차 시도(방식 C, v-only)**: yaw 를 A\* 기하로 override 하고 PPO 는 속도만 제어 → **학습 붕괴**(collision 74% 고착). 원인: 충돌은 "방향"이 결정하는데 PPO 에 조향 레버가 없어 회피 불가.
- **채택**: 학습 env 를 **C2 로 원복(PPO steering 복원)**, 단 배포 정합 위해 **① obs 를 A\* 1 m distance-lookahead, ② dt 0.3→0.05(20 Hz), max_steps→1800, γ→0.9983** 유지. 회피 정합은 배포 노드 수정으로 해결(`rl_offboard_node.py` 가 PPO yaw 사용하도록 변경). (로그: `hybrid_dt05.log`, `PPOonly_dt05.log`)

### 2.6 entropy 계수 분석 → 비대칭 base/boost + 천장 가드

dt=0.05 환경에서 ent_coef 의 dt-scale(sqrt) 적용 여부 통제 비교(10 M, 그 외 동일):

| | UNSCALED (base 0.01/boost 0.03) | SQRT×0.408 (base 0.004/boost 0.012) |
|---|---|---|
| std (1M→10M) | 0.26 → **5.02 (폭주)** | 0.14 → 0.43 (안정) |
| eval weighted | **4M 에 0.93 수렴** | 10M 에 0.82 (미수렴) |

→ sqrt scaling 은 std 발산은 막으나 **샘플효율 3배 희생**. 배포는 deterministic(평균) 추론이라 UNSCALED 의 std 폭주는 무해. 근본 원인은 **PUSH(boost)와 HOLD(base)에 필요한 크기가 다른데 한 스칼라로 묶인 것**. → **수정**: ① base 0.01→**0.004**(boost 0.03 유지, 비대칭), ② `EntCoefScheduleCallback` 에 **entropy 천장 가드**(entropy_loss<−2.5 면 floor 0.001 로 하드캡) 추가. (run: `hybrid_dt05`=unscaled, `hybrid_dt05_entsqrt`=sqrt, `hybrid_dt05_AB`=최종 A+B 수정. 비교 그림: `_archive_ablation/results/compare_hybrid_training.png`)

---

## 3. 실험 인덱스 (run → 로그·텐서보드)

번호는 시간순. 로그는 `_archive_ablation/logs/<run>.log`, 텐서보드는 `_archive_ablation/tensorboard/<run>/` 에 있다.

| # | run | 내용 | 참조 |
|---|-----|------|:---:|
| 1 | `hybrid_dt3` | C2 hybrid(A\*+BFS) 10M — reference (weighted=0.947) | §2.4 |
| 2 | `PPOonly_dt3` | A\*/BFS prior 제거 PPO-only baseline (0.941) | §2.4 |
| 3 | `hybrid_dt05` | dt=0.05 재설계, entropy unscaled (base 0.01) | §2.5·§2.6 |
| 4 | `hybrid_dt05_entsqrt` | dt-scale(sqrt) entropy 통제 비교 (base 0.004/boost 0.012) | §2.6 |
| 5 | `PPOonly_dt05` | PPO-only dt=0.05 | §2.5 |
| 6 | `hybrid_dt05_AB` | A+B 수정(비대칭 base 0.004/boost 0.03 + 천장 가드) | §2.6 |

- **reward ablation seeds**: §2.2~§2.3 채택(C2) 3-seed run 은 `tensorboard/ablation/hybrid_seed{7,42,123}/` + `logs/hybrid_seed{7,42,123}.log`. Case1~4·CK·E1·E2 의 seed 로그도 `logs/` 에 동봉.
- **결과 표·그림**: `_archive_ablation/results/` (`ablation_comparison.md`, `ablation_norm_true_results.csv`, `compare_{hybrid,ppoonly}_training.png`).

---

## 4. 개발 타임라인 (C2 도달까지, §1–§29 압축)

C2 최소 보상에 이르기까지의 누적 튜닝 이력. 각 항목은 단일 묶음 변경이라 isolated 기여는 §2 ablation 에서 사후 분리됨.

| 시기 | 핵심 변경 | 동기 |
|------|-----------|------|
| 04-04 | best_model 가중점수 저장, 커리큘럼 단계 점프(순차→한번에), 코드 리팩토링 | 평가 일관성 |
| 04-11 | **A\* lookahead waypoint 기반 obs 재학습** | short-horizon planning 신호 |
| 04 중 | collision_penalty↑, 커리큘럼 장애물 수·stress mix 조정, course_l_s(S코스) 맵 도입 | 고밀도/구조적 코스 대응 |
| 04 중 | 평가 시나리오 정비(start/goal 최소거리, course_l_s 범위, no_progress 판정) | 평가 신뢰성 |
| 04 중 | BFS cell_size 버그 수정, `_lidar_ranges` 벡터화, BFS-progress(우회 압력) 도입 | 학습 신호·속도 |
| 04 중 | course_l_s spawn 3단계 커리큘럼, eval env 버그 2종 수정 | 학습 분포 |
| 04 중 | obs=0 domain gap 해소(stage4 low-obstacle 혼합), Adaptive ent_coef 도입 | entropy 안정화 |
| 04 말 | base_coef 하향 + best_model 10M 추가학습 → **C2 reference(weighted=0.947) 확립** | reference 모델 |

이후 §2 의 ablation·정합·entropy 실험으로 C2 의 정당성과 배포 정합을 확립.

---

## 5. 부록 — 코드 toggle

현재 코드의 ablation toggle 은 **env 생성자 kwarg** 로 노출되고, 학습 스크립트 CLI 는 entropy·이어학습 인자만 노출한다. 보상 항별 ablation(§2.2~§2.3)은 `_archive_ablation/` 에 보존된 당시 스크립트 변형으로 재현된다(현행 `train_ppo_2d.py` 에는 해당 CLI 플래그가 없음).

| toggle / 인자 | 위치 | 용도 |
|---------------|------|------|
| `use_astar_lookahead` (kwarg) | `env_2d_nav.py`·`make_env` | A\* lookahead obs on/off — hybrid vs PPO-only (§2.4) |
| `w_progress`·`success_reward`·`collision_penalty` (kwarg) | `env_2d_nav.py` | 보상 항별 가중치 (§2.2~§2.3) |
| `hybrid_lookahead_dist_m=1.0` (kwarg) | `env_2d_nav.py` | 배포 obs lookahead 정합 (§2.5) |
| `--ent_base_coef/--ent_ceil_thr/--ent_floor_coef` (CLI) | `train_ppo_2d*.py` | 비대칭 entropy + 천장 가드 (§2.6) |
| `--resume_model/--additional_timesteps` (CLI) | `train_ppo_2d*.py` | best_model 이어 학습/fine-tune |
| `export_policy.py` | — | best_model → TorchScript + obs_norm.json 배포 변환 |
