# PX4 RL Offboard Project

PX4-Autopilot 위에서 동작하는 강화학습(RL) 학습 코드와 ROS 2 offboard 제어 워크스페이스.

## 구성

- **`px4_ws/`** — ROS 2 (colcon) 워크스페이스의 소스 (`src/`)
  - `px4_rl_offboard_mavros` — MAVROS 기반 RL offboard 제어 노드
  - `px4_nav_perception` — 인지(perception) 패키지
  - `px4_nav_planner` — 경로 계획(planner) 패키지
- **`rl/`** — 강화학습 학습/평가 스크립트, 환경, 학습 로그/플롯

## 빌드 / 실행 메모

PX4-Autopilot 본체는 이 저장소에 포함하지 않습니다 (별도로 클론).
`px4_ws`는 PX4-Autopilot 환경 위에서 colcon으로 빌드합니다.

```bash
# 예시
cd px4_ws
colcon build
source install/setup.bash
```

## 저장소에 포함하지 않는 것 (`.gitignore`)

- 빌드 산출물: `px4_ws/build`, `px4_ws/install`, `px4_ws/log`
- 대용량 산출물: `rl/models_ppo_2d` (모델 체크포인트), `rl/_archive_ablation`
- 가상환경/캐시: `.venv`, `__pycache__`
