#!/usr/bin/env bash
# deploy_to_px4.sh
# 이 저장소(astar_ppo_hybrid)의 코드/에셋을 클론한 PX4-Autopilot 트리로 배포한다.
#   - px4_ws/src         → $PX4/px4_ws/src        (ROS2 패키지, mirror)
#   - rl/ 소스(.py/.md/requirements.txt) → $PX4/rl (학습 코드, 추가 복사; venv·모델·로그는 보존)
#   - sim/ 에셋          → PX4 Gazebo worlds/models + start_px4.sh/start_mavros.sh
#
# docker(px4docker)는 ~/PX4-Autopilot 만 마운트하므로, 빌드/실행 전 이 스크립트로
# 저장소 내용을 PX4 트리에 반영해야 한다.
#
# 사용법:
#   ./deploy_to_px4.sh [PX4_DIR]
#   PX4_DIR=/path/to/PX4-Autopilot ./deploy_to_px4.sh
# 인자/환경변수가 없으면 흔한 위치를 자동 탐색한다.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- PX4 디렉터리 결정 ---
PX4="${1:-${PX4_DIR:-}}"
if [[ -z "$PX4" ]]; then
  for cand in /home/dev/PX4-Autopilot "$HOME/PX4-Autopilot" ./PX4-Autopilot ../PX4-Autopilot; do
    [[ -d "$cand/Tools/simulation/gazebo-classic/sitl_gazebo-classic" ]] && { PX4="$cand"; break; }
  done
fi
if [[ -z "$PX4" || ! -d "$PX4/Tools/simulation/gazebo-classic/sitl_gazebo-classic" ]]; then
  echo "[ERR] PX4-Autopilot 트리를 찾지 못했습니다.  사용: $0 /path/to/PX4-Autopilot"; exit 1
fi
echo "[INFO] PX4 = $PX4"

have_rsync() { command -v rsync >/dev/null 2>&1; }

# --- 1) px4_ws/src (ROS2 패키지) : mirror ---
mkdir -p "$PX4/px4_ws/src"
if have_rsync; then
  rsync -a --delete --exclude="__pycache__/" --exclude="*.pyc" \
    "$REPO/px4_ws/src/" "$PX4/px4_ws/src/"
else
  cp -r "$REPO/px4_ws/src/." "$PX4/px4_ws/src/"
fi
echo "[OK] px4_ws/src 배포"

# --- 2) rl/ 소스만 (venv·모델·로그·아카이브 보존, 삭제 없음) ---
mkdir -p "$PX4/rl"
for f in env_2d_nav.py env_2d_nav_ppo_only.py train_ppo_2d.py train_ppo_2d_ppo_only.py \
         export_policy.py requirements.txt CONCEPTS.md experiments.md; do
  [[ -f "$REPO/rl/$f" ]] && cp -f "$REPO/rl/$f" "$PX4/rl/$f"
done
echo "[OK] rl 소스 배포 (venv/모델/로그 미변경)"

# --- 3) sim 에셋 (worlds/models/launchers) ---
bash "$REPO/sim/install_sim_assets.sh" "$PX4"

echo
echo "[DONE] 배포 완료. 다음:"
echo "  컨테이너에서  cd $PX4/px4_ws && colcon build --symlink-install && source install/setup.bash"
