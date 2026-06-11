# UAV 2D 내비게이션 RL — 개념 및 MDP 정의

이 문서는 `env_2d_nav.py` 기준으로 **마르코프 의사결정 과정(MDP)** 을 정식화하고, 상태·행동·보상 설계 근거와 **재현성**을 정리한다. 논지는 Chikhaoui et al. (2022), *PPO-based Reinforcement Learning for UAV Navigation in Urban Environments* 를 출발점으로 하되, 본 구현에서의 **fine-tuning·shaping** 을 명시한다.

**수식 표기**: LaTeX·백슬래시(`\`) 없이 유니코드(Δ, ψ, γ, ∈ 등)와 아래첨자 표기만 사용한다.

---

## 1. 이 설계가 답해야 할 질문

| 질문 | 이 문서에서의 답 |
|------|------------------|
| **문제 정의가 명확한가?** | §2에서 이산 시간 MDP **(상태공간 S, 행동공간 A, 전이 P, 보상 r, 할인율 γ)** 로 기술. 종료 조건(success / collision / timeout / no-progress)과 목표(goal 반경 내 도달)를 코드와 동일하게 정의. |
| **재현 가능한가?** | §3.1·§5·§6에 시나리오·학습 하이퍼파라미터, 시드, 관측·보상 공식을 고정. `env_2d_nav.UAVNav2DEnv` 의 `__init__` 인자와 일치. |
| **왜 이렇게 설계했는가?** | §2.2.1~2.2.3에서 body frame·다운샘플·보상 각 항의 역할을 분리해 서술. §4에서 A*–PPO 혼용 시 **불연속성 완화** 전략을 설명. |

---

## 2. 마르코프 의사결정 과정 (MDP)

### 2.1 정식화

- **시간**: 이산 시간 스텝 **t = 0, 1, 2, …**, 한 스텝의 시뮬레이션 길이는 **Δt** (`dt`, 기본 0.3 s).
- **상태 공간 S**: §2.2.1의 관측 벡터 **s_t** ∈ **ℝ⁶⁶** 로 근사(센서·목표 정보만 사용 — 완전한 내부 상태는 위치·자세·장애물이지만 정책 입장에서는 **s_t** 가 주어진 **Markov 관측**으로 간주).
- **행동 공간 A**: §2.2.2의 **a_t** ∈ **[-1, 1]²**.
- **전이 P(s_{t+1} | s_t, a_t)**: 2D kinematic 모델 + 장애물·경계 충돌 처리(결정적. 학습 시 시드로 장애물·초기 조건 재현).
- **할인율 γ**: 학습 스크립트 `train_ppo_2d.py` 에서 기본 0.99 (PPO 설정).
- **보상 r_t**: §2.2.3.

**에피소드 종료** (`env_2d_nav.py` 와 동일):

- **종료(terminated)**: 충돌 또는 목표 반경 `goal_radius` 내 도달(success).
- **절단(truncated)** (Gymnasium `truncated` 플래그; 보상은 §2.2.3 “절단 시 추가 항” 참고):
  - **timeout**: 매 `step` 시작 시 증가하는 `step_count` 가 **`max_steps` 이상**이면 절단. 기본 `max_steps` = **500** (`env_2d_nav.py`).
  - **no-progress**: 이전 스텝 끝의 목표 거리 `prev_dist` 대비, 이번 스텝 끝 거리 `dist_to_goal` 이 **`prev_dist − ε` 미만으로 줄지 않으면** `no_progress_count` 를 1 올리고, **그보다 더 줄어들면** 0으로 리셋. **`no_progress_count` ≥ `no_progress_steps`** 이면 절단. 기본 **`no_progress_steps` = 35**, **`ε` = 1e‑3 m**는 코드 상 `_progress_eps` 고정값. (`terminated` 가 이미 참이면 no-progress 절단은 걸리지 않음.)

---

## 2.2.1 상태 공간 — 에이전트가 매 순간 무엇을 보는가

### 관측 벡터

**s_t** = ( **d̃₁, d̃₂, …, d̃₆₄, dx_body, dy_body** )ᵀ  ∈ **ℝ⁶⁶**

**차원 분해**: LiDAR 다운샘플 **`obs_beams`** 개(기본 **64**) + body-frame 목표 상대 **`goal_rel`** **2**개 = **64 + 2 = 66**. 코드에서는 `obs_dim = self.obs_beams + 2`, `observation_space.shape == (66,)`, 관측은 `np.concatenate([lidar_norm, goal_rel], axis=0)` 순서로 앞에서부터 **64개 거리 정규화** 뒤 **dx_body, dy_body** 이다.

(전치는 “세로 벡터” 의미로만 표기; 구현은 1차원 concatenate.)

### (1) 전역 waypoint까지의 상대 위치 (거리·방향 정보)

- 월드 프레임 목표 **g**, 기체 위치 **p_t**, 요각 **ψ_t**.
- **상대 변위(월드)**: **δ^w_t** = **g** − **p_t**.
- **기체 프레임으로 회전** (전방 x, 좌측 y; 구현은 **R(−ψ)** 로 월드 변위를 body로 변환):

**[ dx_body ; dy_body ]** = (1 / L_world) · **R(−ψ_t)** · **δ^w_t**

**R(θ)** = | cos θ   −sin θ |
        | sin θ    cos θ |

이후 각 성분을 **[-1, 1]** 로 **클립**. **L_world** 는 `world_size`(기본 20 m). 위 **dx_body, dy_body** 두 개가 **s_t** 의 **마지막 2차원**(절 (2)의 **d̃₁…d̃₆₄** 가 앞의 **64차원**)이다.

- **거리**: **d_t** = ‖**δ^w_t**‖ 는 **s_t** 에 직접 넣지 않지만, 회전은 거리 보존이므로 ‖**R(−ψ) δ^w**‖ = ‖**δ^w**‖ 이고, √(dx_body² + dy_body²) 로 스케일만 복원하면 된다.
- **방향(상대 각)**: **φ_t** = atan2(dy_body, dx_body) 가 기체 전방 기준 목표 방위.

### (2) LiDAR 레이저 스캔 — 차원·360° 다운샘플

1. **Ray casting**: `num_beams = 360`, 시야 `fov = 2π`. 각도는 **[-π, π]** 를 균등 분할 후 기체 요각으로 회전. 각 빔은 거리 `max_range` 까지 ray marching (`step` 0.2 m); 장애물·맵 경계에 맞으면 해당 거리, 없으면 `max_range`.
2. **다운샘플** (`_downsample_lidar`): 길이 360인 벡터를 **정확히 `obs_beams = 64` 개 구간**으로 나눈다. `numpy.array_split(np.arange(360), 64)` 에 대응하는 인덱스 집합 **I₁, …, I₆₄** 에 대해:
   - **`downsample_mode="min"` (기본)**: **r̃_k** = min_{i ∈ I_k} **r_i** — 섹터 내 **최소 거리**만 취해, **가장 가까운 장애물** 보수적 표현(회피에 유리).
   - **`"avg"`**: 섹터 내 평균.
3. **다운샘플을 두는 이유** (구현은 `num_beams`→`obs_beams`):
   - **차원·연산**: 정책 입력 벡터 길이와 스텝당 전처리·메모리를 줄여 **학습·추론 속도**를 맞추기 위함(360개를 그대로 넣을 필요가 없다고 가정).
   - **정보량**: 인접 방향 빔은 거리값이 **강하게 상관**되어, 국소 회피에 필요한 “가까운 장애물” 정보는 섹터별 요약(특히 **`min`**)으로도 대부분 보존된다.
   - **배포·해상도**: 360° / 64 ≈ **5.6°**/빔으로, 실제 `LaserScan` 과 비슷한 **각 해상도**로 맞추기 쉽다. 시뮬에선 ray를 촘촘히 쏴 장애물 형상을 반영하고, **정책에 주는 관측만** 저차원으로 줄인다.

### (3) 정규화 및 max range clipping

원시 다운샘플 거리 **r_k** ∈ [0, **r_max**], **r_max** = `max_range` (10 m).

**d̃_k** = 2 · clip(**r_k** / **r_max**, 0, 1) − 1  ∈ [-1, 1].

**보상 계산** 시 `_sanitize_lidar`: NaN/Inf → **r_max**, [0, **r_max**] 클립 후 `min_r = min_k r_k` 로 danger 항에 사용.

### (4) Body frame 사용 이유 (Markov·불변성)

- 정책 입력이 **기준계 불변**에 가깝게 됨: 동일한 상대 기하라면 절대 헤딩 **ψ** 에 덜 의존.
- 목표 방향이 항상 “내 전방 기준 좌/우·앞” 으로 주어져 **학습 난이도** 완화.
- 배포 시 ROS `odom` yaw로 동일 변환을 적용하면 sim-to-real 관측 정렬이 쉬움.

---

## 2.2.2 행동 공간 — 물리적 명령과 PPO 연속 제어

### 정규화 행동 (정책 출력)

PPO는 **연속 행동**에 대해 가우시안 정책(평균·로그표준편차)을 학습하며, Stable-Baselines3 구현에서 네트워크 출력을 **tanh** 등으로 **유한 구간**에 맵한다. 본 환경과의 인터페이스는:

**a_t** = (**a_v**, **a_ω**) ∈ **[-1, 1]²**.

이는 PPO의 강점인 **연속 스페이스에서의 직접 최적화**(이산화 없음)를 유지하면서, 이후 **물리 한계**를 명시적으로 거는 구조다.

### 물리 제어 매핑

**v_t** = **v_min** + ((**a_v** + 1) / 2) · (**v_max** − **v_min**),  **ω_t** = **a_ω** · **ω_max**.

**상태 갱신**(충돌 처리 전, 구현 순서와 동일):

**ψ_{t+1}** = wrap(**ψ_t** + **ω_t** · Δt),  **p_{t+1}** = **p_t** + **v_t** · Δt · ( cos **ψ_{t+1}** , sin **ψ_{t+1}** )ᵀ

(경로 상 세그먼트 충돌 검사로 실제 **p_{t+1}** 는 장애물 표면에서 잘릴 수 있음.)

### 제어 파라미터 및 범위 (`env_2d_nav.py` 기본값)

| 기호 | 변수명 | 기본값 | 의미 |
|------|--------|--------|------|
| **v_min** | `v_min` | 0.15 m/s | 최소 전진 속도 |
| **v_max** | `v_max` | 0.8 m/s | 최대 전진 속도 |
| **ω_max** | `yaw_rate_max` | 1.0 rad/s | 최대 요 각속도 |
| **Δt** | `dt` | 0.3 s | 시뮬레이션 스텝 |
| **r_max** | `max_range` | 10 m | LiDAR 최대 거리 |

### 실제 제어와의 연결

- **학습**: 위 kinematic 업데이트만 사용(Gazebo 미포함).
- **배포** (`rl_offboard_node`): 동일 **s_t** 구성 `/scan`·`odom`·목표로 생성, 동일 (**v**, **ω**) 매핑 후 `cmd_vel` 로 PX4/MAVROS에 전달. 즉 **정책의 입출력 의미**는 sim·real 공통.

---

## 2.2.3 보상 함수 설계 — 학습 목적함수

### 논문과의 관계

Chikhaoui et al. (2022) 는 도시 환경 PPO 내비게이션에 **구조화된 보상(충돌·진행·목표 등)** 을 둔다. 본 코드 주석은 해당 논문의 보상 식(Eq.(10))을 **개념적 참조**로 명시하되, (i) 배터리/충전소 의사결정 제거, (ii) **denser shaping**(progress, danger, spin), (iii) stress 시나리오 대비 **충돌·근접 페널티 스케일** 은 실험(MODIFICATIONS 로그)에 따라 fine-tune 되었다.

즉, **sparse 목표만으론 신호가 희소**하여 PPO 샘플 효율이 떨어지므로, 이론적 골격은 논문·표준 내비게이션 RL과 맞추고 **가중치는 실험적으로 조정**한 형태로 이해하면 된다.

### 표기 (`_compute_reward` 인자·코드 변수와 대응)

- **d_t**: 스텝 종료 시점 목표까지 유클리드 거리 (`dist_to_goal`). 이전 스텝 종료 거리는 **d_{t−1}** (`prev_dist`).
- **Δd_t** = **d_{t−1}** − **d_t** (= 코드의 `progress`). 양수면 목표에 가까워짐.
- **m_t**: 고해상도 LiDAR에 `_sanitize_lidar` 적용 후 최소 거리 `min_r` (관측용 `obs_beams` 다운샘플 **이전**).
- **d_safe** = `safe_dist`, **λ** = `danger_decay`.
- **ω**: 명령 요각속도 `yaw_rate_cmd` (= 행동 `a_ω` 에 `ω_max` 곱한 값).
- **L_world**: `world_size`(0 나눗셈 방지 시 `max(world_size, ε)`).
- **R_success** = `success_reward`, **R_coll** = `collision_penalty`, **w_p** = `w_progress`, **w_d** = `w_danger`, **w_close** = `w_close`, **w_spin** = `w_spin`, **w_np** = `w_no_progress`, **w_to** = `w_timeout`.
- 시간항 계수 **c_time** = **0.02** (코드 리터럴).

### 스칼라 보상 하나로 정리 (`_compute_reward`)

**r_g** = `goal_radius`. **근접 항**(코드: `min_r >= d_safe` 이면 둘 다 0):

- **P_danger**(**m_t**) = **w_d** · ( exp((**d_safe** − **m_t**) / **λ**) − 1 ) **if** **m_t** < **d_safe** **else** 0.
- **P_close**(**m_t**) = **w_close** · (**d_safe** − **m_t**) **if** **m_t** < **d_safe** **else** 0.
`UAVNav2DEnv._compute_reward` 와 동일 (**if** 표기). **`truncated` ∧ ¬`terminated`** 인 스텝은 progress·시간·장애물·스핀·성공 shaping 을 쓰지 않고 **−w_np** 또는 **−w_to** 만 준다(no_progress 트리거이면 **w_np**, 아니면 timeout → **w_to**).

**r_t** = **−R_coll** if **collision** **else** (
    **−w_np** if (**truncated** and not **terminated** and no_progress 트리거) else
    **−w_to** if (**truncated** and not **terminated**) else (
        **w_p** · **Δd_t**
        − 2 · max(0, −**Δd_t**)
        − **c_time** · (**d_t** / **L_world**)
        − **P_danger**(**m_t**) − **P_close**(**m_t**)
        − **w_spin** · |**ω**|
        + (**R_success** if **d_t** ≤ **r_g** else 0)
    )
).

### 각 구성 요소·스케일 요약

| 항 | 역할 | 스케일 느낌 |
|----|------|-------------|
| **w_p** **Δd_t** | 목표 방향 **진행** 유도 (dense) | 1 m 가까워지면 +**w_p** 단위 기여 (스케일은 **Δt** 와 결합) |
| −2·max(0, −**Δd_t**) | **후퇴·정체** 억제 | 헤매기 비용 |
| 시간항 | 장기적 효율·timeout 유도 | 작은 계수 0.02 |
| Danger / Close | **충돌 전** 안전 여유 | **w_d**, **λ**, **w_close** 는 stress test collision 감소를 위해 키운 이력(MODIFICATIONS 참고) |
| Spin | 제자리 회전 남발 억제 | **w_spin** × 절댓값(**ω**) |
| **R_success** | 희소 **성공** 신호 강화 | sparse + dense 혼합 |
| **R_coll** | **충돌** 강한 패널티 | stress에서 회피 배우기 |
| **w_np**, **w_to** | 절단 스텝 **단독** 패널티(shaping 없음) | no-progress → **−w_np**, timeout → **−w_to** |

### 중간 보상 신호를 사용하는 이유

- **이론적 최적 정책**은 여전히 누적 보상을 최대화하는 같은 MDP이지만, **학습 초기**에 성공까지 도달하기 어렵다. Progress·danger shaping 은 **신호 대 잡음비**를 높여 PPO가 gradient를 받기 쉽게 한다.
- 논문식 구조를 완전히 버리지 않고, **penalty·bonus 의 상대 크기**로 “안전 vs 진행” 트레이드오프를 튜닝한 것이 본 프로젝트의 practical choice다.

---

## 3. Experimental setup (실험 설정)

본 절은 논문식 **Experimental Setup** 에 대응하며, 구현 기준 파일은 `env_2d_nav.py`(`UAVNav2DEnv`), `train_ppo_2d.py` 이다. 보상·커리큘럼·맵 세부 튜닝 이력은 `MODIFICATIONS.md` 를 참고한다.

### 3.1 시나리오 구성 (맵, 시작/목표, 난이도)

#### 워크스페이스·센서·제어 (공통)

| 항목 | 코드 기본값 | 비고 |
|------|-------------|------|
| 맵(작업 영역) | `world_size = 20` (m), **축정렬 사각 장애물** 리스트 | 좌표 **\[0, world_size\]²** , 경계는 `step`에서 클립 |
| LiDAR (내부) | `num_beams = 360`, 전방위 `fov = 2π`, `max_range = 10` m, ray step 0.2 m | 관측은 **64빔** `obs_beams`, 다운샘플 `min` (섹터별 최소거리) |
| 목표 판정 | `goal_radius` 는 **커리큘럼 단계**에 따라 2.0 → 1.5 → 1.0 → 0.5 m | `_apply_curriculum_level` |
| 에피소드 상한 | `max_steps = 500`, `no_progress_steps = 100` (BFS 거리 신기록 없으면 카운트 증가) | timeout ~150 s (`dt=0.3`) |
| 속도·요각속도 | `v_min=0.15`, `v_max=0.8` m/s, `yaw_rate_max=1.0` rad/s, `dt=0.3` s | 행동 **[-1,1]²** 선형 매핑 (§2.2.2) |

#### 커리큘럼 단계 (난이도 축)

`use_curriculum=True` 일 때 `curriculum_levels` (goal_radius, min_clearance, 장애물 개수):

| 단계 | goal_radius | min_clearance | 장애물 수 |
|------|-------------|---------------|-----------|
| 1 | 2.0 m | 기본값 `1.0` m | 고정 **4** |
| 2 | 1.5 m | 동일 | **8** |
| 3 | 1.0 m | 동일 | **12** |
| 4 | 0.5 m | 동일 | **에피소드마다 10~19** (`rng.integers(10, 21)`) |

진행 규칙: 최근 **100** 에피소드 윈도우에서 성공률이 **60% / 75% / 90%** 를 넘으면, **해당하는 최고 단계로 한 번에 점프** (순차 해금이 아님).

#### 단계별 시나리오 혼합 (학습 분포)

커리큘럼이 켜져 있고 `eval_scenario is None` 일 때, 각 리셋에서 난수 한 번으로 아래 중 하나를 샘플한다. `_stage_mix_thresholds` 는 **누적 확률 경계** (normal → obstacle_density_15 → 20 → narrow_passage → course_l_s).

| 커리큘럼 단계 | normal (랜덤 박스) | density 15 | density 20 | narrow | course_l_s |
|---------------|-------------------|-------------|-------------|--------|------------|
| 1 | 90% | 0% | 0% | 5% | 5% |
| 2 | 70% | 10% | 0% | 10% | 10% |
| 3 | 45% | 15% | 10% | 15% | 15% |
| 4 | 10% | 25% | 30% | 20% | 15% |

- **normal**: `_spawn_obstacles()` 후 시작·목표는 여유·충돌 검사; 목표는 시작 기준 최소 **5** m (`min_dist=5.0`).
- **course_l_s 단독 혼합**: 커리큘럼이 꺼진 경우에만 `course_l_mix`(기본 **0.12**, `train_ppo_2d.py` `--course_l_mix`) 확률로 S 코스; eval 시에는 0.

#### 평가·스트레스 시나리오 (`eval_scenario`)

| 이름 | 맵/장애물 | 시작·목표 |
|------|-----------|-----------|
| **obstacle_density_15 / 20** | 월드 내 **15 또는 20**개 직사각 장애물, 크기 **`obstacle_size_range=(1.5, 3.0)`** m | 시작은 자유 격자; 목표는 **선택 장애물 너머(“뒤”)** 방향으로 배치, BFS 도달 가능 쌍. 실패 시 fallback: 시작 기준 최소 **10** m |
| **narrow_passage** | `x≈8` 부근 두 벽 + **1.2~1.5** m 간격 통로 | 통로 **양쪽** 중 한쪽에 스폰·반대 편에 목표; ‖goal−start‖ ≥ **7** m, BFS 검증 |
| **course_l_s** | Gazebo `obstacle_course_light.world` 의 S 코스 박스를 `COURSE_L_S_OFFSET_XY=(12,10)` 평행이동해 **\[0,20\]²** 에 배치 | 스폰 `x∈[15,19]`, `y∈[12,19]`; 목표 `x∈[2.6,4.0]`, `y∈[1.0,4.0]`; ‖goal−start‖ ≥ **10** m, LiDAR `min_clearance`·BFS |

`eval_sensor_noise=True` 일 때만 학습용 커리큘럼을 끄고 `yaw_drift_std=0.05` 로 요우 노이즈 평가.

---

### 3.2 학습 절차·하이퍼파라미터 (`train_ppo_2d.py`)

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
| `gamma` | 0.99 | 할인율; 미래 보상의 현재 가치 비율 |
| `gae_lambda` | 0.95 | GAE(일반화 어드밴티지)의 편향–분산 트레이드오프 파라미터 |
| `clip_range` | 0.2 (`--clip_range`) | PPO에서 확률비 클리핑 폭; 정책 업데이트 크기 상한 |
| `target_kl` | None (`--target_kl` 없으면 KL 조기 중단 비활성) | 설정 시 근사 KL이 이 값을 넘으면 해당 롤아웃의 추가 에폭을 중단 |
| `ent_coef` | 스케줄: 학습 진행률에 따라 **0.02 → 0.01 → 0.005** (`EntCoefScheduleCallback`) | 정책 엔트로피 보너스 계수; 탐험(랜덤성) 강도—후반으로 갈수록 낮춤 |
| `vf_coef` | 0.5 | 가치 함수 손실에 곱하는 가중치 (정챁 손실 대비) |
| `max_grad_norm` | 0.5 | 그래디언트 클리핑(전역 노름 상한); 학습 폭주 완화 |
| 시드 | `seed=42` (`--seed`) | 난수 시드; 환경·PPO 초기화 등 재현성용 |

#### 평가·체크포인트 (학습 스크립트)

- **주기 평가** `--eval_freq`>0: 전역 step마다 obs **0,5,10** + 시나리오 **obstacle_density_15/20, narrow_passage, course_l_s** 에 대해 `evaluate_model`; TensorBoard `eval/*`; 옵션 **`--save_best`** → 가중 **`EVAL_SCORE_WEIGHTS`** 로 `best_model.zip`.
- **학습 후** `--do_eval`: 동일 지표 요약 출력.
- **체크포인트** `checkpoint_freq_timesteps=500_000` (글로벌 step 기준; 서브프로세스 시 env별로 나눠 저장 주기 산출).
- **`save_best` 가중 점수** (`EVAL_SCORE_WEIGHTS`, 합=1.0): `obs_0, obs_5, obs_10` 각 **0.08**, `obstacle_density_15`·`obstacle_density_20` 각 **0.21**, `narrow_passage` **0.16**, `course_l_s` **0.18**.

선택: **`--bc_init`**, **`--resume_model` / `--additional_timesteps` / `--vecnorm_path`**.

---

### 3.3 학습 결과

정량 결과(성공률, stress test, best_model 시점 등)는 실험마다 다르므로 **`MODIFICATIONS.md` 의 “실험 결과 요약” 표**를 기준으로 인용한다. 본 문서는 MDP·코드 정의를 고정하고, 숫자는 해당 로그를 따라가면 재현·비교가 가능하다.

---

### 3.4 Sim 전이: 정책 가중치·관측 정규화

- **정책 가중치**: SB3가 저장한 **`ppo_uav2d_final.zip`**, **`best_model.zip`** 등 (내부에 Actor–Critic MLP 파라미터). 배포 시 같은 아키텍처(관측 **66**차원, 행동 **2**차원)로 로드한다.
- **VecNormalize**: 학습 중 누적된 **관측 평균·분산**(및 학습 시 보상 정규화 통계)이 **`vecnormalize.pkl`** 에 저장된다. 평가 루틴(`evaluate_model`)과 동일하게 **배포에서도 관측 정규화를 쓸 경우** `VecNormalize.load(..., training=False, norm_reward=False)` 로 래핑된 env에 맞춰 **같은 pickle** 을 로드해야 분포가 일치한다. 관측만 쓰고 보상 정규화는 추론에 불필요하다.
- **관측 정합**: 시뮬/실기 ROS 파이프라인에서는 §2.2.1과 같이 **360° 또는 유사 스캔 → 64빔 `min` 다운샘플 → \[-1,1\] 정규화** 와 **body frame 목표 상대 (world_size 로 나눈 뒤 클립)** 를 동일하게 구성한다. `CONCEPTS.md` §2.2.1·§2.2.2 및 (선택) **하이브리드 `rl_offboard_node`** 에서 `cmd_vel` 로 동일 **v, ω** 매핑을 쓰는 구조가 sim-to-real 정합의 전제다. (노드 경로는 본 `rl/` 레포 바깥일 수 있음.)

---

## 4. A*와 PPO 사이의 “불연속성”은 어떻게 다루는가

- **학습 환경** (`env_2d_nav`): **A* 미사용**. MDP는 “LiDAR + 목표 상대 위치 → 연속 속도” 로 완결된다. 재현 가능한 RL 문제로 정의된다.
- **배포(선택) 하이브리드** (`rl_offboard_node`, `use_hybrid=True`):
  - **이산 전환**: 전방 LiDAR 거리 **d_front** 가 임계값 안쪽이면 **RL 로컬 회피**, 바깥이면 **전역 경로(A*) 추종**. 히스테리시스로 `enter` / `exit` 임계를 달리해 **채터링** 감소.
  - **연속성 측면**: 두 모드 모두 동일 저수준 인터페이스(`cmd_vel`)를 쓴다. RL 모드일 때도 관측은 같은 차원의 **s_t** (스캔 + 목표 상대). 즉 **정책 네트워크는 모드 스위치를 직접 입력받지 않는다**; 전환은 상위 로직이 수행한다.
  - **학습–배포 정합**: 정책은 “로컬 회피·목표 접근”에 특화해 학습되고, 하이브리드에서는 전역이 coarse 경로, RL이 미세 회피를 담당하는 **계층적 역할 분담**으로 불연속성을 **운영체계 설계**로 흡수한다.

요약: **MDP 자체는 연속 행동·부드러운 shaping** 으로 정의되고, A*는 그 MDP 밖의 **상위 planner** 로만 개입하므로 “하나의 신경망이 이산·연속을 동시에 학습”하는 문제로 환원하지 않는다.

---

## 5. 문제 정의·재현성·설계 근거 (체크리스트)

1. **문제 정의**: “연속 속도 제어로 정적 장애물과 경계 내에서 waypoint까지 도달”이며, 관측·행동·종료·보상이 §2–2.2.3 의 수식·코드와 1:1로 대응한다.
2. **재현성**: `seed`, `world_size`, `num_beams`, `obs_beams`, `downsample_mode`, `max_range`, 보상 계수, `dt`, 차량 역학 매핑이 코드에 고정되어 있고, 동일 시드·동일 SB3 버전이면 rollout 분포가 재현 가능하다(VecNormalize 통계는 학습 종속). PPO·병렬 env·평가 주기는 §3.2 참고.
3. **설계 근거**: 센서는 실제 RPLidar 파이프라인과 정렬; 보상은 논문형 구조 + shaping + 실험적 튜닝; A*는 학습 문제를 오염시키지 않고 배포 시 선택적 계층으로만 사용한다.

---

## 6. 부록 — 구현·학습 매핑

| 코드 위치 | 내용 |
|-----------|------|
| `UAVNav2DEnv._get_obs` | **s_t** 구성 |
| `UAVNav2DEnv._downsample_lidar` | 360→64, `min`/`avg` |
| `UAVNav2DEnv.step` | 전이 + `terminated`/`truncated` |
| `UAVNav2DEnv._compute_reward` | §2.2.3 과 동일 |
| `train_ppo_2d.py` | PPO, γ, VecNormalize, 평가 루틴 |

---

## 7. 참고 문헌

- Chikhaoui, Y. et al., *PPO-based Reinforcement Learning for UAV Navigation in Urban Environments*, 2022. (보상 구조의 **참고 틀**; 본 레포는 Eq.(10) literal 복제가 아니라 progress/danger 등 **확장 shaping**.)
