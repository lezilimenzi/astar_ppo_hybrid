# UAV 하이브리드 항법 (A* + PPO) — 연구 가설·모델 설계·이론

이 문서는 `env_2d_nav.py` / `train_ppo_2d.py` 기준으로 **이 연구가 무엇을 증명하려 했는지**(연구 가설), **그에 맞춰 모델·MDP를 어떻게 설계했는지**, 그리고 그 설계를 뒷받침하는 **이론적 근거**를 정리한다. 정량 실험 이력은 [experiments.md](experiments.md), 원본 개발 저널은 `_archive_ablation/MODIFICATIONS_full.md` 를 참조한다.

핵심 주장은 **A\* 전역 경로 계획과 PPO 기반 경로 추종을 결합한 계층적 하이브리드 항법** 이 PPO 단독·A\* 단독보다 우월하다는 것이다(이민재, *A\* 전역 경로 계획과 PPO 기반 경로 추종을 결합한 UAV 하이브리드 항법 프레임워크*). 출발점은 Chikhaoui et al. (2022) 의 PPO 도시 항법이며, 본 연구는 (1) A\* 전역 경로를 PPO 관측에 주입해 **부분 관측 한계**를 보완하고, (2) 행동 공간을 **연속**으로 확장하며, (3) 보상을 본 항법 문제에 맞게 **BFS 진행 보상**으로 재설계한다.

**수식 표기**: LaTeX·백슬래시(`\`) 없이 유니코드(Δ, Φ, ψ, γ, σ, ε, Â, ∈, ≤, ≈ 등)와 아래첨자 표기만 사용한다.

> 본 문서의 2D 정량 비교(§1.1)는 학습 로그 `hybrid_dt3`(A\*+PPO) vs `PPOonly_dt3`(PPO 단독)에 근거하며, 이 2D 실험은 dt=0.3 설정에서 수행됐다. 현행 코드 기본값은 배포(20 Hz) 정합을 위해 dt=0.05 로 이후 갱신됐다(§5.1).

---

## 1. 연구 질문과 가설 — 무엇을 증명하려 했는가

전통적 A\* 격자 경로계획은 **실시간 회피 능력이 부족**하고 UAV 동역학을 반영하지 못한다. 반대로 PPO 같은 RL 은 반응적 회피는 가능하나 **전역 경로 정보가 없어 부분 관측(POMDP)** 에 의존한다. 본 연구는 두 방식을 계층적으로 결합해 상호 한계를 보완할 수 있는지를 묻고, 세 가설로 분해해 검증했다.

| # | 가설 | 검증 결과 | 근거 |
|---|------|-----------|------|
| **H1 (핵심)** | **하이브리드 우월성 · 부분관측 보완** — A\* 전역 경로를 PPO 관측에 경유점으로 주입하면 PPO 단독의 부분관측 한계를 보완해, **PPO 단독·A\* 단독 대비** 목표 도달률·수렴 속도·회피 안정성이 향상된다. | **입증** | 2D: 수렴(1M weighted 0.80 vs 0.51)·충돌(1M d15/20 평균 34% vs 64%) 전 구간 우위(§1.1). 3D Gazebo: hybrid **81.3%** vs A\* 46.9% vs PPO 25%(§1.2) |
| **H2** | **연속 행동 공간** — Chikhaoui 의 이산 행동(장애물 증가 시 성공률 저하·연속 기동 미표현)을 **연속 (v, ω)** 로 확장하면 UAV 의 연속적 기동을 더 잘 표현한다. | **입증(설계 채택)** | PPO 가우시안 연속 정책으로 전 시나리오 수렴(§3.3·§4.3) |
| **H3** | **BFS 진행 보상 재설계** — Chikhaoui 보상에서 항법과 무관한 **배터리 항을 제거**하고, 장애물 **우회**와 경유점 **진행**을 반영하는 **BFS 격자 최단경로 진행 보상**을 도입한다. | **입증** | BFS 진행 보상이 우회 시 단조 신호 제공 → 신호 대 잡음비 향상(§3.4, experiments.md §2.2~§2.3) |

### 1.1 2D 결과 비교 — hybrid vs PPO-only (10M)

동일 보상·평가 프로토콜(7 시나리오, 1M 마다 평가)에서 학습한 두 모델의 **가중 성공률(weighted score)** 과 **고난이도(d15/d20) 충돌률** 추이:

| step | weighted: hybrid | PPO-only | 충돌률(d15/20 평균): hybrid | PPO-only |
|------|:----:|:----:|:----:|:----:|
| 1M | **0.80** | 0.51 | **34 %** | 64 % |
| 2M | **0.90** | 0.77 | **22 %** | 42 % |
| 3M | **0.91** | 0.84 | **21 %** | 32 % |
| 4M | **0.93** | 0.86 | **15 %** | 29 % |
| 최종(10M) | **0.947** | 0.940 | **9.5 %** | 12 % |

→ 하이브리드는 특히 **초반(1M~4M) 수렴 속도가 뚜렷이 빠르고**, 고밀도 환경 **충돌률이 전 구간 낮다**. 최종 가중 점수는 근소 우위지만 **표본 효율과 안전성에서 일관된 우위** — A\* 전역 경로가 관측에 들어와 센서 범위 밖 장애물에 대해서도 우회 방향을 조기에 제공하기 때문(§3.1). (`hybrid_dt3.log`, `PPOonly_dt3.log`)

### 1.2 3D Gazebo SITL 결과 (보고서 Table 2)

사전 지도 정보 없이 1초 주기로 A\* 재계획. 시가지 모사 맵(좁은 통로·군 부대) × 목표 2개 × 8회 = 32회:

| 모델 | 목표 도달률 | 성공 시 평균 소요 |
|------|:----:|:----:|
| A\* 단독 | 46.9 % (15/32) | 82.8 s |
| PPO 단독 | 25.0 % (8/32) | 110.0 s |
| **Hybrid** | **81.3 % (26/32)** | **74.2 s** |

→ 하이브리드가 도달률·속도 모두 최고. A\* 단독은 새로 감지된 장애물에 대한 실시간 회피가 약하고, PPO 단독은 전역 경로 부재로 국소 최소에 빠져 도달률이 낮다(§3.2).

---

## 2. 설계 철학 — 가설에서 모델 설계로

각 설계 결정은 위 가설과 그것을 뒷받침하는 이론·실험에 1:1로 대응한다.

| 설계 결정 | 무엇을 | 어느 가설 | 이론 근거 | 실험 근거 |
|-----------|--------|:--------:|-----------|-----------|
| **A\* 전역 경로 → PPO 관측** | A\* 경로상 lookahead 경유점의 body-frame 상대 위치를 obs 에 포함 | H1 | POMDP 부분관측 보강(§3.1) | 수렴·충돌 우위(§1.1), 3D 81.3%(§1.2) |
| **계층 구조 (A\* 판단 / PPO 제어)** | A\* 전역 경로, PPO 추종·국소 회피로 역할 분담 | H1 | A\* 한계 + 계층 하이브리드(§3.2) | A\*·PPO 단독 3D 열위(§1.2) |
| **연속 (v, ω) 행동** | PPO 가우시안 연속 정책, 이산화 없음 | H2 | PPO 연속 최적화, 이산 행동 한계(§3.3) | 전 시나리오 수렴(§4.3) |
| **BFS 진행 보상(− 배터리 항)** | 유클리드가 아닌 격자 최단경로 거리 차분 | H3 | PBRS 측지 포텐셜(§3.4) | 우회 시 단조 신호(§4.4) |
| **64빔 LiDAR min 다운샘플 + body-frame 목표** | 360빔→64 섹터 최소거리, 목표는 기체 프레임 상대 | H1 | 연산↓·기준계 불변, LaserScan 해상도 정합(§4.2) | min=회피 보수적(§4.2) |
| **2D OccupancyGrid 인지** | LiDAR+Odometry → 점유 격자 → A\* | H1 | 사전 지도 불요, 부분 관측 누적 보강 | 사전지도 없는 환경 검증(§1.2) |

---

## 3. 이론적 배경

설계를 뒷받침하는 핵심 이론을 본 환경의 표기로 정리한다.

### 3.1 부분 관측성(POMDP)과 전역 경로 관측 보강 — H1 의 이론적 핵심

LiDAR 는 유한한 `max_range`(10 m) 까지만 보므로, 정책이 받는 관측 s_t 는 환경 전체 상태(전역 장애물 배치·목표까지의 위상)를 다 담지 못한다. 즉 문제는 엄밀히는 **부분 관측 마르코프 결정 과정(POMDP)** 이고, 순간 관측만으로는 **충분 통계(sufficient statistic)** 가 되지 못한다(Sutton & Barto). PPO 단독이 전역 경로 정보 부족으로 국소 최소(막다른 길·우회 실패)에 빠지는 근본 원인이다.

**A\* 전역 경로의 관측 주입**은 이 부분 관측을 보강한다. A\* 가 OccupancyGrid 위에서 산출한 전역 최단 경로의 **다음 경유점 상대 위치**를 obs 에 넣으면, 센서 범위 밖 장애물 배치까지 압축 반영한 **전역 방향 신호**가 매 스텝 정책에 제공된다. 이는 관측을 충분 통계에 가깝게 만들어(belief 를 요약한 privileged 특징) 정책 학습을 쉽게 한다. §1.1 에서 하이브리드가 PPO 단독보다 **초반 수렴이 빠르고 충돌이 적은** 이유가 바로 이 관측 보강이다.

### 3.2 전통적 A\*의 한계와 계층적 하이브리드 — 인지·판단·제어

격자 A\* 는 전역 최단 경로를 효율적으로 주지만 두 한계가 있다: (1) **실시간 회피 부재** — 정적 스냅샷 위에서 계획하므로 재계획 주기 안에 새로 감지된 장애물에 반응하지 못하고, (2) **동역학 미반영** — UAV 의 속도·요각속도 제약을 무시해 경로가 매끄럽게 추종 불가능할 수 있다(Chai et al. 2025 맥락). 반대로 RL 은 반응적·동역학 인지 제어는 되지만 전역 시야가 없다.

본 하이브리드는 이를 **계층 분담**으로 흡수한다.

- **인지(Perception)**: 360° LiDAR 를 64채널 다운샘플 + Odometry 결합 → **2D OccupancyGrid** 생성.
- **판단(Decision)**: **A\*** 가 격자 위에서 목표까지 전역 경로(경유점열) 산출 — *전역 시야*.
- **제어(Control)**: **PPO** 정책이 다음 경유점을 추종하며 LiDAR 로 **국소 회피**, (v, ω) 결정 → MAVROS → PX4 속도 제어. *반응·동역학*.

전역(A\*)이 큰 그림을, 지역(PPO)이 미세 회피를 맡아 서로의 사각을 덮는다. 정책 신경망은 모드 스위치를 직접 입력받지 않으며(관측 차원 동일), 두 계층은 동일 저수준 인터페이스(`cmd_vel`)를 공유한다(§6).

### 3.3 PPO 연속 제어 — 클립 대리목표·GAE, 연속 vs 이산 행동

연속 제어 정책 최적화로 **PPO**(Schulman et al. 2017)를 쓴다.

- **클립 대리목표**: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) 에 대해
  L^CLIP = E[ min( r_t · Â_t , clip(r_t, 1−ε, 1+ε) · Â_t ) ]
  로 확률비를 ε(clip_range 0.2) 안에 가두어 한 업데이트 폭을 제한 → **근사 신뢰영역**. `target_kl` 로 근사 KL 초과 시 추가 epoch 를 조기 중단.
- **GAE**(Schulman et al. 2016): Â_t = Σ (γλ)^l δ_{t+l}, δ_t = r_t + γV(s_{t+1}) − V(s_t). λ(0.95)로 어드밴티지 **편향–분산** 조절.
- **연속 vs 이산 행동(H2)**: Chikhaoui 는 행동을 이산화해 거친 기동만 가능했고 장애물이 늘수록 성공률이 떨어졌다. 본 연구는 PPO 의 강점인 **연속 공간 직접 최적화**(가우시안 정책, 이산화 없음)를 그대로 살려 (v, ω) 를 연속 출력 → 매끄러운·실현 가능한 기동을 표현한다.

### 3.4 BFS 진행 보상 — 포텐셜 기반 측지 shaping (H3)

희소한 성공 보상만으로는 PPO 표본 효율이 낮다. Chikhaoui 의 구조화 보상을 출발점으로 하되, 본 항법과 무관한 **배터리 항을 제거**하고 **BFS 진행 보상**을 도입했다. 그 정당성은 두 이론으로 뒷받침된다.

**(a) 포텐셜 기반 shaping(PBRS, Ng·Harada·Russell 1999).** 포텐셜 Φ(s) 에 대해 shaping 항 F(s,s') = γ·Φ(s') − Φ(s) 를 더해도 **최적 정책은 불변**이다. 포텐셜을 Φ(s) = −w_p · b(s) (b = 목표까지 BFS 최단경로 거리)로 두면

  γ·Φ(s') − Φ(s) = w_p · (b_{t−1} − γ·b_t) ≈ w_p · (b_{t−1} − b_t)   (γ ≈ 1)

로 본 보상의 진행 항과 일치한다. 즉 BFS 진행 보상은 **거의 포텐셜 기반**이라 해(최적 정책)를 (거의) 바꾸지 않으면서 매 스텝 dense gradient 만 공급한다 — sparse 성공만 줄 때보다 학습이 빠른 이유.

**(b) 측지 cost-to-go 근사.** PBRS 에서 학습을 가장 빠르게 하는 포텐셜은 진짜 cost-to-go 다. **유클리드 거리**는 장애물을 무시해 음의 기울기가 벽을 관통 → 우회 시 거리가 일시 증가(음수 보상)하는 **국소 최소**를 만든다. **BFS 격자 거리**는 free-space 그래프의 **측지 거리**라 기울기가 항상 통과 가능한 경로를 따라가 우회 중에도 단조 감소한다. 개활지에선 둘이 동급, **장애물이 빽빽할수록 BFS 가 우월**(experiments.md §2.3).

---

## 4. 마르코프 의사결정 과정 (MDP)

### 4.1 정식화

- **시간**: 이산 시간 스텝 **t = 0, 1, 2, …**, 한 스텝 길이는 **Δt** (`dt`, 기본 **0.05 s** — 배포 setpoint 20 Hz 정합, experiments.md §2.5).
- **상태 공간 S**: §4.2 의 관측 벡터 **s_t** ∈ **ℝ⁶⁶** 로 근사(센서·목표 정보만 사용 — 완전한 내부 상태는 위치·자세·장애물이지만 정책 입장에서는 **s_t** 가 주어진 **Markov 관측**으로 간주).
- **행동 공간 A**: §4.3 의 **a_t** ∈ **[-1, 1]²** (정규화 행동). 물리 제약(매핑 후): 선속도 **v ∈ [0.3, 0.8] m/s**, 요 각속도 **ω ∈ [-1.0, 1.0] rad/s** (§4.3).
- **전이 P(s_{t+1} | s_t, a_t)**: 2D kinematic 모델 + 장애물·경계 충돌 처리(결정적. 학습 시 시드로 장애물·초기 조건 재현).
- **할인율 γ**: `train_ppo_2d.py` 기본 **0.9983** (= 0.99^(1/6), dt 0.3→0.05 보정으로 실시간 discount horizon ~30 s 보존, §3.3).
- **보상 r_t**: §4.4.

**에피소드 종료** (`env_2d_nav.py` 와 동일, 최종 C2 모델):

- **종료(terminated)**: 충돌 또는 목표 반경 `goal_radius` 내 도달(success).
- **절단(truncated)**: **timeout** — `step_count` 가 **`max_steps` 이상**이면 절단. 기본 `max_steps` = **1800** (dt=0.05 기준 ~90 s).

> 과거 버전에는 no-progress 절단 + 시간·근접·스핀·timeout 페널티가 있었으나, **C2 최소 보상**에서는 env 코드에서 모두 제거되었다. 근거는 PBRS(§3.1)와 ablation(experiments.md §2 — BFS-progress 외 항은 marginal/negative).

### 4.2 상태 공간 — 에이전트가 매 순간 무엇을 보는가

#### 관측 벡터

**s_t** = ( **d̃₁, d̃₂, …, d̃₆₄, dx_body, dy_body** )ᵀ  ∈ **ℝ⁶⁶**

**차원 분해**: LiDAR 다운샘플 **`obs_beams`** 개(기본 **64**) + body-frame 목표 상대 **`goal_rel`** **2**개 = **64 + 2 = 66**. 코드에서는 `obs_dim = self.obs_beams + 2`, `observation_space.shape == (66,)`, 관측은 `np.concatenate([lidar_norm, goal_rel], axis=0)` 순서로 앞에서부터 **64개 거리 정규화** 뒤 **dx_body, dy_body** 이다.

(전치는 "세로 벡터" 의미로만 표기; 구현은 1차원 concatenate.)

##### (1) target waypoint까지의 상대 위치 (거리·방향 정보)

관측의 target **w** 는 모드에 따라 다르다(`_get_obs`):

- **hybrid (기본, `use_astar_lookahead=True`)**: A\* 경로상 **1 m 앞 lookahead waypoint**(`hybrid_lookahead_dist_m=1.0`). 정규화 거리 **L = `lookahead_dist`(3.0 m)** — 배포 `_build_obs` 의 `LOOKAHEAD_NORM_SCALE=3.0` 와 정합.
- **PPO-only (`use_astar_lookahead=False`)**: target = **최종 goal g**, 정규화 거리 **L = `world_size`(20 m)**.

- 기체 위치 **p_t**, 요각 **ψ_t**. **상대 변위(월드)**: **δ^w_t** = **w** − **p_t**.
- **기체 프레임으로 회전** (전방 x, 좌측 y; 구현은 **R(−ψ)** 로 월드 변위를 body로 변환):

**[ dx_body ; dy_body ]** = (1 / L) · **R(−ψ_t)** · **δ^w_t**

**R(θ)** = | cos θ   −sin θ |
        | sin θ    cos θ |

이후 각 성분을 **[-1, 1]** 로 **클립**. 위 **dx_body, dy_body** 두 개가 **s_t** 의 **마지막 2차원**(절 (2)의 **d̃₁…d̃₆₄** 가 앞의 **64차원**)이다. A\* lookahead 를 쓰면 정책이 **장애물 우회 경로 방향**을 직접 관측하므로 short-horizon planning 신호가 된다(기여도는 experiments.md §2.4 — 작음).

- **거리**: **d_t** = ‖**δ^w_t**‖ 는 **s_t** 에 직접 넣지 않지만, 회전은 거리 보존이므로 ‖**R(−ψ) δ^w**‖ = ‖**δ^w**‖ 이고, √(dx_body² + dy_body²) 로 스케일만 복원하면 된다.
- **방향(상대 각)**: **φ_t** = atan2(dy_body, dx_body) 가 기체 전방 기준 목표 방위.

##### (2) LiDAR 레이저 스캔 — 차원·360° 다운샘플

1. **Ray casting**: `num_beams = 360`, 시야 `fov = 2π`. 각도는 **[-π, π]** 를 균등 분할 후 기체 요각으로 회전. 각 빔은 거리 `max_range` 까지 ray marching (`step` 0.2 m); 장애물·맵 경계에 맞으면 해당 거리, 없으면 `max_range`.
2. **다운샘플** (`_downsample_lidar`): 길이 360인 벡터를 **정확히 `obs_beams = 64` 개 구간**으로 나눈다. `numpy.array_split(np.arange(360), 64)` 에 대응하는 인덱스 집합 **I₁, …, I₆₄** 에 대해:
   - **`downsample_mode="min"` (기본)**: **r̃_k** = min_{i ∈ I_k} **r_i** — 섹터 내 **최소 거리**만 취해, **가장 가까운 장애물** 보수적 표현(회피에 유리).
   - **`"avg"`**: 섹터 내 평균.
3. **다운샘플을 두는 이유** (구현은 `num_beams`→`obs_beams`):
   - **차원·연산**: 정책 입력 벡터 길이와 스텝당 전처리·메모리를 줄여 **학습·추론 속도**를 맞추기 위함.
   - **정보량**: 인접 방향 빔은 거리값이 **강하게 상관**되어, 국소 회피에 필요한 "가까운 장애물" 정보는 섹터별 요약(특히 **`min`**)으로도 대부분 보존된다.
   - **배포·해상도**: 360° / 64 ≈ **5.6°**/빔으로, 실제 `LaserScan` 과 비슷한 **각 해상도**로 맞추기 쉽다. 시뮬에선 ray를 촘촘히 쏴 장애물 형상을 반영하고, **정책에 주는 관측만** 저차원으로 줄인다.

##### (3) 정규화 및 max range clipping

원시 다운샘플 거리 **r_k** ∈ [0, **r_max**], **r_max** = `max_range` (10 m).

**d̃_k** = 2 · clip(**r_k** / **r_max**, 0, 1) − 1  ∈ [-1, 1].

**보상 계산** 시 `_sanitize_lidar`: NaN/Inf → **r_max**, [0, **r_max**] 클립 후 `min_r = min_k r_k` 로 danger 항에 사용.

##### (4) Body frame 사용 이유 (Markov·불변성)

- 정책 입력이 **기준계 불변**에 가깝게 됨: 동일한 상대 기하라면 절대 헤딩 **ψ** 에 덜 의존 → 관측 분포가 집중되어 학습·전이가 쉬워짐.
- 목표 방향이 항상 "내 전방 기준 좌/우·앞" 으로 주어져 **학습 난이도** 완화.
- 배포 시 ROS `odom` yaw로 동일 변환을 적용하면 sim-to-real 관측 정렬이 쉬움.

### 4.3 행동 공간 — 물리적 명령과 PPO 연속 제어

#### 정규화 행동 (정책 출력)

PPO는 **연속 행동**에 대해 가우시안 정책(평균·로그표준편차)을 학습하며, Stable-Baselines3 구현에서 네트워크 출력을 **tanh** 등으로 **유한 구간**에 맵한다. 본 환경과의 인터페이스는:

**a_t** = (**a_v**, **a_ω**) ∈ **[-1, 1]²**.

이는 PPO의 강점인 **연속 스페이스에서의 직접 최적화**(이산화 없음)를 유지하면서, 이후 **물리 한계**를 명시적으로 거는 구조다. Chikhaoui 의 **이산 행동**(장애물 증가 시 성공률 저하·연속 기동 미표현)을 연속 (v, ω) 로 확장한 것이 본 연구의 행동공간 설계다(H2, §3.3).

#### 물리 제어 매핑

**v_t** = **v_min** + ((**a_v** + 1) / 2) · (**v_max** − **v_min**),  **ω_t** = **a_ω** · **ω_max**.

**상태 갱신**(충돌 처리 전, 구현 순서와 동일):

**ψ_{t+1}** = wrap(**ψ_t** + **ω_t** · Δt),  **p_{t+1}** = **p_t** + **v_t** · Δt · ( cos **ψ_{t+1}** , sin **ψ_{t+1}** )ᵀ

(경로 상 세그먼트 충돌 검사로 실제 **p_{t+1}** 는 장애물 표면에서 잘릴 수 있음.)

#### 제어 파라미터 및 범위 (`env_2d_nav.py` 기본값)

| 기호 | 변수명 | 기본값 | 의미 |
|------|--------|--------|------|
| **v_min** | `v_min` | 0.3 m/s | 최소 전진 속도 (멈춤 방지) |
| **v_max** | `v_max` | 0.8 m/s | 최대 전진 속도 |
| **ω_max** | `yaw_rate_max` | 1.0 rad/s | 최대 요 각속도 |
| **Δt** | `dt` | 0.05 s | 시뮬레이션 스텝 (배포 20 Hz 정합) |
| **r_max** | `max_range` | 10 m | LiDAR 최대 거리 |

#### 실제 제어와의 연결

- **학습**: 위 kinematic 업데이트만 사용(Gazebo 미포함).
- **배포** (`rl_offboard_node`): 동일 **s_t** 구성 `/scan`·`odom`·목표로 생성, 동일 (**v**, **ω**) 매핑 후 `cmd_vel` 로 PX4/MAVROS에 전달. 즉 **정책의 입출력 의미**는 sim·real 공통.

### 4.4 보상 함수 설계 — 학습 목적함수

#### 논문과의 관계

Chikhaoui et al. (2022) 는 도시 환경 PPO 내비게이션에 **구조화된 보상(충돌·진행·목표·배터리 등)** 을 둔다. 본 연구는 이를 출발점으로 하되 **(1) 항법과 무관한 배터리 항을 제거**하고 **(2) 장애물 우회·경유점 진행을 위한 BFS 진행 보상을 도입**해 재설계했다(H3). 이어 다른 shaping 항(시간 압박·근접·스핀·no-progress·timeout)의 isolated 기여를 ablation 으로 측정한 결과(experiments.md §2) 대부분 marginal 또는 negative 임을 확인하고 **충돌 + BFS-progress + success 세 항(C2)** 으로 정리했다. 이론적 근거는 PBRS(§3.4): BFS-progress 만이 (거의) 포텐셜 기반이라 정책을 편향시키지 않고, 나머지 일반 shaping 은 편향을 유발한다.

#### 최종 C2 보상 (`_compute_reward`)

세 항만 사용한다. **r_g** = `goal_radius`.

- **충돌**: **r_t** = **−R_coll** (`collision_penalty=500`).
- **그 외**: **r_t** = **w_p** · (**b_{t−1} − b_t**) + (**R_success** if **d_t** ≤ **r_g** else 0)
  - **b_t**: 목표까지의 **BFS 격자 최단경로 거리**(`prev_bfs_dist`, `curr_bfs_dist`). **유클리드가 아닌 BFS** 를 쓰는 이유: 장애물을 우회할 때 직선 거리는 일시 증가하지만 BFS 거리는 단조 감소 → **우회 방향으로 일관된 학습 신호**(측지 cost-to-go, §3.4).
  - **w_p** = `w_progress=50` (BFS 거리 1 m 감소 시 +50), **R_success** = `success_reward=800`.

| 항 | 변수 | 역할 |
|----|------|------|
| BFS-progress | `w_progress=50` | 목표 방향 **우회 진행** 유도 (dense, dominant, ≈PBRS) |
| Success bonus | `success_reward=800` | 희소 **성공** anchor (sparse) |
| Collision | `collision_penalty=500` | **충돌** terminal 페널티 (안전) |

> `w_danger`·`w_close`·`w_spin`·`w_time_pressure`·`w_no_progress`·`w_timeout` 항은 이전 ablation 버전에 있었으나 **현재 env 코드에서는 제거**되어 C2 세 항만 구현된다. 항별 정량 기여·제외 근거는 experiments.md §2.2~§2.3.

#### 왜 dense shaping(BFS-progress)인가 — PBRS 관점 재정리

- **sparse 목표만으론 신호가 희소**하여 PPO 샘플 효율이 떨어진다. BFS-progress 는 매 스텝 우회 방향 gradient 를 제공해 **신호 대 잡음비**를 높인다(ablation 에서 C0→C1 +0.53 의 압도적 기여). 포텐셜 기반이라 **이 가속이 최적 정책을 바꾸지 않는다**(§3.4).
- 반대로 **포텐셜이 아닌 항을 더할수록 정책 편향·perverse incentive** 가 생긴다(근접 페널티의 obs_0 oscillation, 시간 압박의 무모한 직진, Chikhaoui 원식의 빠른 충돌 선호). 따라서 BFS 진행 보상 중심의 간결한 보상이 본 항법 문제에 적합하다는 것이 실험적·이론적 결론(H3).

---

## 5. 실험 설정 (Experimental setup)

본 절은 논문식 **Experimental Setup** 에 대응하며, 구현 기준 파일은 `env_2d_nav.py`(`UAVNav2DEnv`), `train_ppo_2d.py` 이다. 보상·커리큘럼·맵 세부 튜닝 이력은 [experiments.md](experiments.md) 를 참고한다.

### 5.1 시나리오 구성 (맵, 시작/목표, 난이도)

#### 워크스페이스·센서·제어 (공통)

| 항목 | 코드 기본값 | 비고 |
|------|-------------|------|
| 맵(작업 영역) | `world_size = 20` (m), **축정렬 사각 장애물** 리스트 | 좌표 **\[0, world_size\]²** , 경계는 `step`에서 클립 |
| LiDAR (내부) | `num_beams = 360`, 전방위 `fov = 2π`, `max_range = 10` m, ray step 0.2 m | 관측은 **64빔** `obs_beams`, 다운샘플 `min` (섹터별 최소거리) |
| 목표 판정 | `goal_radius` 는 **커리큘럼 단계**에 따라 2.0 → 1.5 → 1.0 → 0.5 m | `_apply_curriculum_level` |
| 에피소드 상한 | `max_steps = 1800` (timeout ~90 s, `dt=0.05`) | no-progress 절단은 C2 에서 제거 |
| 속도·요각속도 | `v_min=0.3`, `v_max=0.8` m/s, `yaw_rate_max=1.0` rad/s, `dt=0.05` s | 행동 **[-1,1]²** 선형 매핑 (§4.3) |

#### 커리큘럼 단계 (난이도 축)

`use_curriculum=True` 일 때 `curriculum_levels` (goal_radius, min_clearance, 장애물 개수). 5 단계(0~4):

| 단계 | goal_radius | min_clearance | 장애물 수 |
|------|-------------|---------------|-----------|
| 0 | 2.0 m | 기본값 `1.0` m | **0~3** 랜덤 |
| 1 | 2.0 m | 동일 | **4~6** 랜덤 |
| 2 | 1.5 m | 동일 | 고정 **8** |
| 3 | 1.0 m | 동일 | **8~12** 랜덤 |
| 4 | 0.5 m | 동일 | **12~20** 랜덤 (`rng.integers(lo, hi+1)`) |

진행 규칙: 최근 **100** 에피소드 윈도우에서 성공률이 **60% / 75% / 90%** 를 넘으면, **해당하는 최고 단계로 한 번에 점프** (순차 해금이 아님). 커리큘럼은 단계적 난이도 분포(stage 0~4)로, 완전 랜덤 정책이 초반 성공 경험을 확보하고 고밀도 환경까지 점진적으로 일반화하도록 돕는다.

#### 단계별 시나리오 혼합 (학습 분포)

커리큘럼이 켜져 있고 `eval_scenario is None` 일 때, 각 리셋에서 난수 한 번으로 아래 중 하나를 샘플한다. `_stage_mix_thresholds` 는 **누적 확률 경계** (normal → obstacle_density_15 → 20 → narrow_passage → course_l_s).

| 커리큘럼 단계 | normal (랜덤 박스) | density 15 | density 20 | narrow | course_l_s |
|---------------|-------------------|-------------|-------------|--------|------------|
| 0 | 100% | 0% | 0% | 0% | 0% |
| 1 | 90% | 0% | 0% | 5% | 5% |
| 2 | 70% | 10% | 0% | 10% | 10% |
| 3 | 45% | 15% | 10% | 15% | 15% |
| 4 | 5% | 15% | 35% | 15% | 30% |

- **normal**: `_spawn_obstacles()` 후 시작·목표는 여유·충돌 검사; 목표는 시작 기준 최소 **5** m (`min_dist=5.0`).
- **course_l_s 혼합**: 위 `_stage_mix_thresholds` 의 커리큘럼 단계별 비율로만 섞인다(별도 CLI 플래그 없음). 커리큘럼이 꺼진(`use_curriculum=False`) 평가/노이즈 모드에서는 `_reset_normal_scenario` 만 사용한다.

#### 평가·스트레스 시나리오 (`eval_scenario`)

| 이름 | 맵/장애물 | 시작·목표 |
|------|-----------|-----------|
| **obstacle_density_15 / 20** | 월드 내 **15 또는 20**개 직사각 장애물, 크기 **`obstacle_size_range=(1.5, 3.0)`** m | 시작은 자유 격자; 목표는 **선택 장애물 너머("뒤")** 방향으로 배치, BFS 도달 가능 쌍. 실패 시 fallback: 시작 기준 최소 **10** m |
| **narrow_passage** | **ㄱ/ㄴ 자 L-통로(닫힌 관)**: L 통로만 자유공간으로 두고 나머지를 벽 5개로 감쌈, 통로 폭 **1.8~2.4** m. `flip_x/flip_y` 로 4방향 랜덤화 | 통로 **양 끝점**(세로 다리 끝 ↔ 가로 다리 끝)을 시작·목표로 — 통로 안에서 **90° 코너를 꺾어** 반대쪽 끝까지 가야 도달; 충돌·BFS(`cell_size=0.2`) 도달 검증 |
| **course_l_s** | Gazebo `obstacle_course_light.world` 의 S 코스 박스를 `COURSE_L_S_OFFSET_XY=(12,10)` 평행이동해 **\[0,20\]²** 에 배치 | 스폰 `x∈[15,19]`, `y∈[12,19]`; 목표 `x∈[2.6,4.0]`, `y∈[1.0,4.0]`; ‖goal−start‖ ≥ **10** m, LiDAR `min_clearance`·BFS |

`eval_sensor_noise=True` 일 때만 학습용 커리큘럼을 끄고 `yaw_drift_std=0.05` 로 요우 노이즈 평가.

### 5.2 학습 절차·하이퍼파라미터 (`train_ppo_2d.py`)

#### 알고리즘·벡터 환경

- **PPO** (Stable-Baselines3), `MlpPolicy`, 병렬 환경 **`n_envs`** (기본 **8**): `SubprocVecEnv` 또는 `DummyVecEnv`.
- **`VecNormalize`**: 학습 시 `norm_obs=True`, `norm_reward=True`, `clip_reward=10.0`. 종료 시 **`models_ppo_2d/vecnormalize.pkl`** 저장.
- **`Monitor`** 로 에피소드 길이·보상 래핑.

#### `TrainConfig` 기본값 (CLI로 덮어쓰기 가능)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `total_timesteps` | 6e6 (`--total_timesteps`) | 전체 환경 스텝 수 상한; 학습이 이 만큼 진행되면 종료 |
| `n_envs` | 8 | 동시에 도는 병렬 환경 개수; 수집량·속도와 분산에 영향 |
| `n_steps` | 2048 | **환경 하나**당 롤아웃 길이; 한 PPO iteration당 총 샘플은 대략 `n_steps × n_envs` |
| `batch_size` | 256 | 한 번의 최적화에서 쓰는 미니배치 크기 |
| `n_epochs` | 10 (`--n_epochs`) | 같은 롤아웃 데이터로 정책·가치망을 몇 번 에폭 돌릴지 |
| `learning_rate` | 1e-4 (`--learning_rate`) | Adam 등 옵티마이저의 학습률 |
| `gamma` | **0.9983** (= 0.99^(1/6)) | 할인율; dt 0.3→0.05 보정으로 실시간 horizon 보존 (§3.3) |
| `gae_lambda` | 0.95 | GAE(일반화 어드밴티지)의 편향–분산 트레이드오프 파라미터 (§3.3) |
| `clip_range` | 0.2 (`--clip_range`) | PPO에서 확률비 클리핑 폭; 정책 업데이트 크기 상한 (§3.3) |
| `target_kl` | None (`--target_kl` 없으면 KL 조기 중단 비활성) | 설정 시 근사 KL이 이 값을 넘으면 해당 롤아웃의 추가 에폭을 중단 |
| `ent_coef` | **AdaptiveEntropy** (`EntCoefScheduleCallback`): 비대칭 base **0.004** / boost **0.03** + entropy 천장 가드(ceil −2.5 시 floor 0.001 하드캡) | 탐험 강도. entropy 부족 시 boost(push), 충분 시 base(hold), 과탐험 시 floor 로 std 폭주 차단 (§3.4, experiments.md §2.6) |
| `vf_coef` | 0.5 | 가치 함수 손실에 곱하는 가중치 (정책 손실 대비) |
| `max_grad_norm` | 0.5 | 그래디언트 클리핑(전역 노름 상한); 학습 폭주 완화 |
| 시드 | `seed=42` (`--seed`) | 난수 시드; 환경·PPO 초기화 등 재현성용 |

> critic 표현력: A\* lookahead obs 실험 구간에서 explained_variance 가 낮아 advantage 품질이 떨어지자(§3.3), `policy_kwargs` 의 `net_arch` 를 **비대칭**(`vf=[256,256,128]` > `pi=[64,64]`)으로 키워 value function 을 안정화했다(experiments.md §2.4 맥락).

#### 평가·체크포인트 (학습 스크립트)

- **주기 평가** `--eval_freq`>0: 전역 step마다 obs **0,5,10** + 시나리오 **obstacle_density_15/20, narrow_passage, course_l_s** 에 대해 `evaluate_model`; TensorBoard `eval/*`; 옵션 **`--save_best`** → 가중 **`EVAL_SCORE_WEIGHTS`** 로 `best_model.zip`.
- **학습 후** `--do_eval`: 동일 지표 요약 출력.
- **체크포인트** `checkpoint_freq_timesteps=500_000` (글로벌 step 기준; 서브프로세스 시 env별로 나눠 저장 주기 산출).
- **`save_best` 가중 점수** (`EVAL_SCORE_WEIGHTS`, 합=1.0): `obs_0, obs_5, obs_10` 각 **0.08**, `obstacle_density_15`·`obstacle_density_20` 각 **0.21**, `narrow_passage` **0.16**, `course_l_s` **0.18**.

선택(이어 학습/fine-tune): **`--resume_model` / `--additional_timesteps` / `--vecnorm_path`**.

### 5.3 학습 결과

정량 결과(성공률, stress test, best_model 시점 등)는 실험마다 다르므로 **[experiments.md](experiments.md) 의 실험 결과 표(§1·§2)**를 기준으로 인용한다. 본 문서는 MDP·코드 정의를 고정하고, 숫자는 해당 로그를 따라가면 재현·비교가 가능하다. 가설별 핵심 수치는 §1 표를 참조.

### 5.4 Sim 전이: 정책 가중치·관측 정규화

- **정책 가중치**: SB3가 저장한 **`ppo_uav2d_final.zip`**, **`best_model.zip`** 등 (내부에 Actor–Critic MLP 파라미터). 배포 시 같은 아키텍처(관측 **66**차원, 행동 **2**차원)로 로드한다.
- **VecNormalize**: 학습 중 누적된 **관측 평균·분산**(및 학습 시 보상 정규화 통계)이 **`vecnormalize.pkl`** 에 저장된다. 평가 루틴(`evaluate_model`)과 동일하게 **배포에서도 관측 정규화를 쓸 경우** `VecNormalize.load(..., training=False, norm_reward=False)` 로 래핑된 env에 맞춰 **같은 pickle** 을 로드해야 관측 분포가 일치한다. 관측만 쓰고 보상 정규화는 추론에 불필요하다.
- **관측 정합**: 시뮬/실기 ROS 파이프라인에서는 §4.2와 같이 **360° 또는 유사 스캔 → 64빔 `min` 다운샘플 → \[-1,1\] 정규화** 와 **body frame 목표 상대 (정규화 후 클립)** 를 동일하게 구성한다. §4.2·§4.3 및 (선택) **하이브리드 `rl_offboard_node`** 에서 `cmd_vel` 로 동일 **v, ω** 매핑을 쓰는 구조가 sim-to-real 정합의 전제다. (노드 경로는 본 `rl/` 레포 바깥일 수 있음.)

---

## 6. 이산 A\* 경로와 연속 PPO 제어의 결합

A\* 는 **격자 위 이산 경로**(셀 경유점열)를 산출하고 PPO 는 **연속 (v, ω)** 를 낸다. 하이브리드의 핵심은 이 이산–연속 간극을 매끄럽게 잇는 것이며, 본 연구는 **모드 전환 없이** 이를 해결한다.

- **연속 관측으로 가교**: A\* 전역 경로의 다음 lookahead 경유점을 **body-frame 상대 위치(연속값)** 로 관측에 넣는다(§4.2). 정책은 이 경유점을 향해 **연속 제어로 추종**하면서 LiDAR 로 국소 회피한다 → "하나의 신경망이 이산·연속 행동을 동시에 학습" 하는 문제로 환원되지 않는다(정책 출력은 항상 연속, A\* 는 obs 힌트일 뿐).
- **주기적 재계획(replan)**: 배포에서 OccupancyGrid 는 LiDAR 스캔으로 점진 갱신되므로, A\* 를 **주기적으로(보고서 기준 1 s)** 재계획해 새로 감지된 장애물을 전역 경로에 반영한다. 학습 env 도 동일 주기로 재계획해 sim–real 거동을 정합한다.
- **계층적 역할 분담**: 전역(A\*, 판단)이 큰 그림을, 지역(PPO, 제어)이 미세 회피·동역학을 맡아 서로의 사각을 덮는다(§3.2). 정책 네트워크는 모드 스위치를 입력받지 않으며, 두 계층은 동일 저수준 인터페이스(`cmd_vel`)를 공유한다.

요약: A\* 는 MDP 밖의 **상위 planner** 로서 전역 경로를 obs 로 제공하고, PPO 는 그 경유점을 연속 제어로 추종한다. §1 의 결과는 이 결합(하이브리드)이 A\*·PPO 단독을 모두 능가함을 보인다.

---

## 7. 문제 정의·재현성·설계 근거 (체크리스트)

1. **문제 정의**: "연속 속도 제어로 정적 장애물과 경계 내에서 waypoint까지 도달" 이며, 관측·행동·종료·보상이 §4 의 수식·코드와 1:1로 대응한다.
2. **재현성**: `seed`, `world_size`, `num_beams`, `obs_beams`, `downsample_mode`, `max_range`, 보상 계수, `dt`, 차량 역학 매핑이 코드에 고정되어 있고, 동일 시드·동일 SB3 버전이면 rollout 분포가 재현 가능하다(VecNormalize 통계는 학습 종속). PPO·병렬 env·평가 주기는 §5.2 참고.
3. **설계 근거(가설 추적성)**: 모든 핵심 설계 결정이 §1 의 가설과 §3 의 이론으로 추적된다 — A\* 전역 경로의 관측 주입은 부분관측 보완(H1, §3.1·§3.2), 연속 (v, ω) 행동은 PPO 연속 제어(H2, §3.3), BFS 진행 보상은 포텐셜 기반 측지 shaping(H3, §3.4). A\* 와 PPO 는 이산–연속 간극을 연속 관측으로 잇는 계층 구조로 결합된다(§6).

---

## 8. 부록 — 구현·학습 매핑

| 코드 위치 | 내용 |
|-----------|------|
| `UAVNav2DEnv._get_obs` | **s_t** 구성 (§4.2) |
| `UAVNav2DEnv._downsample_lidar` | 360→64, `min`/`avg` (§4.2) |
| `UAVNav2DEnv._compute_bfs_dist_map` / `_get_bfs_dist` | BFS 측지 거리 b_t (§3.4·§4.4) |
| `UAVNav2DEnv._compute_astar_path` / `_get_lookahead_wp*` | A\* 전역 경로 + lookahead 경유점 obs (§3.1·§6) |
| `UAVNav2DEnv.step` | 전이 + `terminated`/`truncated` (§4.1·§4.3) |
| `UAVNav2DEnv._compute_reward` | C2 보상 (§4.4) |
| `train_ppo_2d.py` | PPO, γ, VecNormalize, 평가 루틴 (§5.2) |
| `EntCoefScheduleCallback` | 학습 안정화용 entropy 스케줄 (experiments.md §2.6) |

---

## 9. 참고 문헌

- 이민재, *A\* 전역 경로 계획과 PPO 기반 경로 추종을 결합한 UAV 하이브리드 항법 프레임워크*. (본 문서가 정리하는 연구 — 하이브리드 우월성·연속 행동·BFS 보상 재설계 H1~H3.)
- Chikhaoui, K., Ghazzai, H. & Massoud, Y., *PPO-based Reinforcement Learning for UAV Navigation in Urban Environments*, IEEE MWSCAS 2022. (보상 구조의 **출발 틀**; 본 연구는 배터리 항 제거 + BFS 진행 보상 재설계 및 연속 행동 확장 — H2·H3.)
- Sutton, R. S. & Barto, A. G., *Reinforcement Learning: An Introduction*, MIT Press, 2018. (MDP·부분 관측 정식화 — §3.1·§4.1.)
- Chai, Y. et al., *UAVRM-A\*: A Complex Network and 3D Radio Map-Based Algorithm for Optimizing Cellular-Connected UAV Path Planning*, Sensors 25(13), 2025. (전통 A\* 경로계획의 한계·맥락 — §3.2.)
- Ng, A. Y., Harada, D. & Russell, S., *Policy Invariance under Reward Transformations*, ICML 1999. (포텐셜 기반 shaping 정책 불변성 — BFS-progress 정당화, §3.4.)
- Schulman, J. et al., *Proximal Policy Optimization Algorithms*, 2017 / *Generalized Advantage Estimation (GAE)*, ICLR 2016. (클립 대리목표·신뢰영역·어드밴티지 — §3.3.)
