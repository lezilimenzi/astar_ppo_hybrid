#!/usr/bin/env bash
# 구 export 디렉토리 삭제 (C2 이전 버전). root 소유라 root 권한 필요:
#   sudo bash scripts/cleanup_old_exports.sh
#
# 삭제 대상:
#   export/         (2026-04-07 초기 PPO export, policy_ts.pt 96KB)
#   export_deploy/  (2026-04-13 hybrid export, policy_hybrid_ts.pt) — C2(export_C2_10M)가 대체
# 보존: export_C2_10M/, export_C2_PPOonly/  (C2 정식 배포 export)
set -u
cd "$(dirname "$0")/.."   # → rl/

for d in export export_deploy; do
  if [ -d "$d" ]; then
    echo "removing $d/ ..."
    ls -la "$d"
    rm -rf "$d"
    echo "  -> removed"
  else
    echo "$d/ : 이미 없음"
  fi
done

echo
echo "보존된 C2 export:"
ls -d export_C2_10M export_C2_PPOonly 2>/dev/null
