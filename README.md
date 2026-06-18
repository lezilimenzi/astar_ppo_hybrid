# PX4 RL Offboard Project

PX4-Autopilot 위에서 동작하는 강화학습(RL) 학습 코드와 ROS 2 offboard 제어 워크스페이스.

## 구성

- **`px4_ws/`** — ROS 2 (colcon) 워크스페이스의 소스 (`src/`)
  - `px4_rl_offboard_mavros` — MAVROS 기반 RL offboard 제어 노드
  - `px4_nav_perception` — 인지(perception) 패키지
  - `px4_nav_planner` — 경로 계획(planner) 패키지
- **`rl/`** — 강화학습 학습/평가 스크립트, 환경, 학습 로그/플롯
- **`sim/`** — 직접 만든 Gazebo Classic 커스텀 에셋(월드 2종·드론 모델·`start_px4.sh`). 상세: [`sim/README.md`](sim/README.md)

## 빠른 시작 (처음 받는 경우)

> PX4-Autopilot 본체는 이 저장소에 포함하지 않습니다. **공식 저장소를 별도 클론**한 뒤, 본 repo의 코드/에셋을 PX4 트리로 배포(`deploy_to_px4.sh`)해서 사용합니다. (docker는 `~/PX4-Autopilot` 만 마운트하므로.)

### 0. 사전 요구사항
- WSL2 + Ubuntu, Docker, git (GUI 필요 시 WSLg/GPU)

### 1. 저장소 클론 (두 개)
```bash
cd ~
git clone <THIS_REPO_URL> astar_ppo_hybrid
git clone https://github.com/PX4/PX4-Autopilot.git --recursive   # 공식, 본 연구는 v1.16.0-rc1 기준
```

### 2. PX4 SITL 빌드 (공식 절차)
```bash
cd ~/PX4-Autopilot
bash ./Tools/setup/ubuntu.sh            # 최초 1회 (PX4 빌드 의존성)
make px4_sitl gazebo-classic            # build/px4_sitl_default + gazebo plugin 생성
```

### 3. 저장소 → PX4 트리로 배포
```bash
cd ~/astar_ppo_hybrid
./deploy_to_px4.sh ~/PX4-Autopilot      # px4_ws/src + rl 소스 + sim(월드·모델·런처) 복사
```
> 이후 코드/월드를 고칠 땐 **이 repo를 정본으로 수정 → `deploy_to_px4.sh` 재실행**.

### 4. px4docker 컨테이너 진입
`~/.bashrc` 에 아래 [px4docker alias](#px4docker-alias-정의) 를 추가(절대경로는 본인 환경에 맞게)한 뒤:
```bash
px4docker          # px4-dev-jammy 컨테이너 진입 (없으면 생성)
```

### 5. 컨테이너 런타임 의존성 (최초 1회, 컨테이너 안)
ROS2 노드는 시스템 python으로 실행되므로 MAVROS·torch·scipy 가 필요합니다.
```bash
sudo apt-get update
sudo apt-get install -y ros-humble-mavros ros-humble-mavros-extras
ros2 run mavros install_geographiclib_datasets.sh    # GeographicLib 데이터셋
pip install torch scipy
```

### 6. ROS2 워크스페이스 빌드 (컨테이너 안)
```bash
cd /home/dev/PX4-Autopilot/px4_ws
colcon build --symlink-install
source install/setup.bash
```

### 7. RL 학습 환경 (학습할 때만, 컨테이너 안)
```bash
cd /home/dev/PX4-Autopilot/rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # gymnasium + Stable-Baselines3 + torch 등
```

### 8. 실행 (각 단계는 별도 컨테이너 셸 — `px4docker` 재진입)
```bash
# (셸 1) SITL + Gazebo
start_px4
# (셸 2) MAVROS 브리지
start_mavros
# (셸 1의 pxh 콘솔) 이륙
commander takeoff
# (셸 3) source 후 인지(grid) 스택
source /home/dev/PX4-Autopilot/px4_ws/install/setup.bash
ros2 launch px4_nav_perception s3_grid_fuser.launch.py
# (셸 4) source 후 항법(A*+PPO) 스택
source /home/dev/PX4-Autopilot/px4_ws/install/setup.bash
ros2 launch px4_rl_offboard_mavros rl_offboard.launch.py   # goal publisher + rl_offboard 동시 기동
```
> 다른 월드: `PX4_SITL_WORLD=.../worlds/military_airfield.world start_px4`. nav_mode(astar/ppo/hybrid)·목표는 `px4_ws/src/px4_rl_offboard_mavros/config/params.yaml`, `px4_nav_planner/config/goals.yaml` 에서 조정.

자세한 파이프라인 구조·좌표는 [`px4_ws/src/PIPELINE_OVERVIEW.md`](px4_ws/src/PIPELINE_OVERVIEW.md), sim 에셋은 [`sim/README.md`](sim/README.md) 참조.

## `px4docker` alias 정의

`px4_ws` colcon 빌드·offboard 제어·Gazebo 시뮬은 모두 이 Docker 컨테이너 안에서 실행합니다.

컨테이너 요점 (alias 정의 기준):

- 이미지 `osrf/ros:humble-desktop-full`, 컨테이너 이름 `px4-dev-jammy`
- 마운트: 호스트 `~/PX4-Autopilot` → 컨테이너 `/home/dev/PX4-Autopilot`
- `--network=host` + `DISPLAY`·`/tmp/.X11-unix`·`/dev/dri` 전달 (Gazebo GUI / GPU 가속)
- 최초 1회만 colcon·rosdep·vcstool 설치 (`/root/.setup_done` 마커로 이후 생략)

> `px4docker` 는 셸 alias 입니다. `~/.bashrc` 에 아래를 추가해 사용하세요 (호스트 절대경로 `/home/akswosla/...` 는 본인 환경에 맞게 수정).

```bash
alias px4docker='
  NAME=px4-dev-jammy
  WD=/home/dev/PX4-Autopilot

  if [ -n "$(docker ps -q -f name=^$NAME$)" ]; then
    # 실행 중이면: WD로 들어가도록 강제
    docker exec -it -w "$WD" "$NAME" bash -lc "cd \"$WD\" 2>/dev/null || cd /work; exec bash"
  elif [ -n "$(docker ps -aq -f name=^$NAME$)" ]; then
    # 꺼져 있으면 켠 뒤 WD로 exec
    docker start "$NAME" >/dev/null
    docker exec -it -w "$WD" "$NAME" bash -lc "cd \"$WD\" 2>/dev/null || cd /work; exec bash"
  else
    # 처음 생성
    xhost +local:root >/dev/null 2>&1 || true
    docker run -it \
      --network=host \
      --name "$NAME" \
      --dns 8.8.8.8 --dns 1.1.1.1 \
      -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 \
      -v /home/akswosla/PX4-Autopilot:/home/dev/PX4-Autopilot:rw \
      -v /home/akswosla/ros2_ws:/work:rw \
      -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
      --device /dev/dri \
      --shm-size=1g \
      --workdir "$WD" \
      osrf/ros:humble-desktop-full \
      bash -lc "
        set -e
        mkdir -p \"$WD\" /work
        # ---- 설치는 딱 1회만: 마커 파일이 없을 때만 실행 ----
        if [ ! -f /root/.setup_done ]; then
          apt-get update
          apt-get install -y python3-colcon-common-extensions python3-rosdep python3-vcstool git
          rosdep init || true
          rosdep update || true
          echo \"source /opt/ros/humble/setup.bash\" >> /root/.bashrc
          touch /root/.setup_done
        fi
        cd \"$WD\" 2>/dev/null || cd /work
        exec bash"
  fi
'
```

### RL 학습 (`rl/`) — 같은 `px4docker` 컨테이너 + 가상환경

`rl/` 학습·평가도 **동일한 `px4docker` 컨테이너 안에서** 실행하며, `rl/.venv` 가상환경을 활성화한 뒤 돌립니다.

```bash
px4docker          # 컨테이너 진입 (위와 동일)

# --- 컨테이너 내부 ---
cd /home/dev/PX4-Autopilot/rl
source .venv/bin/activate          # gymnasium + Stable-Baselines3 등 학습 의존성
python train_ppo_2d.py             # hybrid (A*+PPO)
# python train_ppo_2d_ppo_only.py  # PPO-only baseline
```

## 저장소에 포함하지 않는 것 (`.gitignore`)

- 빌드 산출물: `px4_ws/build`, `px4_ws/install`, `px4_ws/log`
- 대용량 산출물: `rl/models_ppo_2d` (모델 체크포인트), `rl/_archive_ablation`
- 가상환경/캐시: `.venv`, `__pycache__`
