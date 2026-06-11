#!/usr/bin/env bash
# §32 원복 3: ablation 실험 산출물을 보류 폴더(_archive_ablation/)로 이동(백업).
#
# 학습 산출물 디렉토리가 root 소유라 디렉토리 이동에는 root 권한이 필요합니다:
#     sudo bash scripts/archive_ablation.sh
#
# C2 / C2_PPO 관련 산출물은 이동 대상에서 제외(보존)합니다.
# 결과 CSV/MD(ablation_*.csv, ablation_*.md)도 그대로 유지합니다 (수치 보존).
set -u
cd "$(dirname "$0")/.."   # → rl/

ARCHIVE="$PWD/_archive_ablation"
mkdir -p "$ARCHIVE/models_ppo_2d/ablation" "$ARCHIVE/logs_ppo_2d/ablation" "$ARCHIVE/root_logs"

echo "[1/4] §30 ablation: C0/C1/C3~C7 seed 디렉토리 (C2_seed* 는 보존)"
for c in C0 C1 C3 C4 C5 C6 C7; do
  for s in 42 123 7; do
    mv -v "models_ppo_2d/ablation/${c}_seed${s}"     "$ARCHIVE/models_ppo_2d/ablation/" 2>/dev/null
    mv -v "logs_ppo_2d/ablation/${c}_seed${s}"       "$ARCHIVE/logs_ppo_2d/ablation/"   2>/dev/null
    mv -v "logs_ppo_2d/ablation/${c}_seed${s}.log"   "$ARCHIVE/logs_ppo_2d/ablation/"   2>/dev/null
  done
done
# ablation_master.log 는 C2 결과 포함 → 보존(이동 안 함)

echo "[2/4] §30a norm_false 디렉토리"
mv -v "models_ppo_2d/ablation_norm_false" "$ARCHIVE/models_ppo_2d/" 2>/dev/null
mv -v "logs_ppo_2d/ablation_norm_false"   "$ARCHIVE/logs_ppo_2d/"   2>/dev/null

echo "[3/4] §34/§35/§36/§37 실험 디렉토리 (E1/E2/CK/Case1~4)"
for name in E1 E2 CK Case1 Case2 Case3 Case4; do
  for s in 42 123 7; do
    mv -v "models_ppo_2d/${name}_seed${s}" "$ARCHIVE/models_ppo_2d/" 2>/dev/null
    mv -v "logs_ppo_2d/${name}_seed${s}"   "$ARCHIVE/logs_ppo_2d/"   2>/dev/null
  done
done

echo "[4/4] root-level stdout 로그 (C2_*.log 는 보존)"
for f in E1_seed42.log E1_seed123.log E1_seed7.log E1_master.log \
         E2_seed42.log E2_seed123.log E2_seed7.log E2_master.log \
         CK_seed42.log CK_seed123.log CK_seed7.log CK_master.log \
         Case1_seed42.log Case1_seed123.log Case1_seed7.log \
         Case2_seed42.log Case2_seed123.log Case2_seed7.log \
         Case3_seed42.log Case3_seed123.log Case3_seed7.log \
         Case4_seed42.log Case4_seed123.log Case4_seed7.log \
         Cases_master.log; do
  mv -v "$f" "$ARCHIVE/root_logs/" 2>/dev/null
done

echo
echo "완료. 보류 위치: $ARCHIVE"
echo "--- 보존된 C2 자산 확인 ---"
ls -d C2_10M.log C2_PPOonly_10M.log \
      models_ppo_2d/C2_10M models_ppo_2d/C2_PPOonly_10M \
      models_ppo_2d/ablation/C2_seed42 models_ppo_2d/ablation/C2_seed123 models_ppo_2d/ablation/C2_seed7 \
      logs_ppo_2d/C2_10M logs_ppo_2d/C2_PPOonly_10M \
      export_C2_10M export_C2_PPOonly 2>/dev/null
