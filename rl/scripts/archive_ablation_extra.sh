#!/usr/bin/env bash
# §32 원복 3 (추가): ablation 결과/비교 산출물을 보류 폴더로 이동.
#   sudo bash scripts/archive_ablation_extra.sh
#
# _archive_ablation/ 가 root 소유(1차 스크립트가 sudo 로 생성)라 root 권한 필요.
# 결과 수치는 MODIFICATIONS.md 에 보존되어 있으므로 원본 CSV/MD 이동해도 무방.
# 주의: compare_*_training.png 는 C2 hybrid vs C2 PPOonly 학습곡선 비교 plot 임.
set -u
cd "$(dirname "$0")/.."   # → rl/

ARCHIVE="$PWD/_archive_ablation/results"
mkdir -p "$ARCHIVE"

for f in ablation_comparison.md \
         ablation_norm_true_results.csv  ablation_norm_true_results.md \
         ablation_norm_false_results.csv ablation_norm_false_results.md \
         compare_hybrid_training.png     compare_ppoonly_training.png; do
  mv -v "$f" "$ARCHIVE/" 2>/dev/null
done

echo
echo "완료. 보류 위치: $ARCHIVE"
ls -la "$ARCHIVE"
