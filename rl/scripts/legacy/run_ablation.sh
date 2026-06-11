#!/bin/bash
# ============================================================
# §30 A0-A6 isolated ablation: 8 config × 3 seed = 24 runs
# ============================================================
# 본 스크립트는 §30 ablation 의 Phase 2b 본 학습을 자동 실행한다.
# 각 run 은 fresh (resume 금지), 4M timesteps, 8-env 병렬.
# 결과는 ./logs_ppo_2d/ablation/{config}_seed{seed}/ 에 분리 저장.
#
# A0 (BFS-progress) 가 ablation 변수에 포함됨 — w_progress 의 isolated contribution 측정.
# C0 (BFS=0, 모두 OFF) → C1 (BFS=50, 인프라만) → ... → C7 (full) 누적 ablation.
#
# 사용:
#   chmod +x scripts/run_ablation.sh
#   nohup bash scripts/run_ablation.sh > ablation_master.log 2>&1 &
#
# 또는 tmux/screen 안에서 직접 실행:
#   tmux new -s ablation
#   bash scripts/run_ablation.sh
#
# 예상 소요: 24 run × ~70~80 분 ≈ 28~32 시간 (1.2~1.3 일)
# ============================================================

set -e  # 에러 발생 시 즉시 중단 (한 run 실패가 모두 망치는 것 방지)
set -o pipefail

# ============================================================
# 환경 설정
# ============================================================
# 스크립트 위치 기준으로 RL_DIR 자동 결정 (호스트 / 컨테이너 양쪽 호환)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"   # scripts/.. = rl/
VENV="${RL_DIR}/.venv/bin/python3"

cd "${RL_DIR}" || { echo "FATAL: rl 디렉토리 진입 실패"; exit 1; }
echo "Using RL_DIR=${RL_DIR}"

if [ ! -x "${VENV}" ]; then
    echo "FATAL: venv python3 not found at ${VENV}"
    exit 1
fi

# ============================================================
# Ablation 매트릭스 정의 (8 config 누적 ablation)
# ============================================================
# 변수 순서:    A0(w_progress)  A1(w_time_pressure)  A2(w_spin)  A3-term(w_no_progress)  A3-soft(w_no_progress_soft)  A4(w_timeout)  A5(success_reward)  A6-danger(w_danger)  A6-close(w_close)
#
# A0 = w_progress (BFS-progress, 본 연구 C1 인프라)
# A1 = w_time_pressure (시간 압박)
# A2 = w_spin (스핀 페널티)
# A3 = w_no_progress + w_no_progress_soft 통합 (no-progress mechanism)
# A4 = w_timeout (timeout terminal)
# A5 = success_reward (성공 보너스)
# A6 = w_danger + w_close 통합 (B1 obstacle potential)
#
# 누적 순서: A0 (인프라) → A5 (sparse anchor) → A3 (stuck 신호) → A4 (timeout) → A6 (obstacle) → A1 (시간 압박) → A2 (스핀, 마지막 trajectory 품질)

SEEDS=(42 123 7)
TIMESTEPS=4000000
LOG_BASE="${RL_DIR}/logs_ppo_2d/ablation"
SAVE_BASE="${RL_DIR}/models_ppo_2d/ablation"

declare -A CONFIGS=(
    # cfg     A0      A1     A2     A3-t     A3-s    A4       A5       A6-d   A6-c
    [C0]="0.0    0.0    0.0    0.0      0.0     0.0      0.0      0.0    0.0"     # base — 모두 OFF (충돌 페널티만)
    [C1]="50.0   0.0    0.0    0.0      0.0     0.0      0.0      0.0    0.0"     # +A0 (BFS-progress 인프라)
    [C2]="50.0   0.0    0.0    0.0      0.0     0.0      800.0    0.0    0.0"     # +A5 (success bonus)
    [C3]="50.0   0.0    0.0    200.0    1.0     0.0      800.0    0.0    0.0"     # +A3 (no-prog terminal+soft 통합)
    [C4]="50.0   0.0    0.0    200.0    1.0     300.0    800.0    0.0    0.0"     # +A4 (timeout)
    [C5]="50.0   0.0    0.0    200.0    1.0     300.0    800.0    6.0    10.0"    # +A6 (B1 obstacle 전체)
    [C6]="50.0   0.02   0.0    200.0    1.0     300.0    800.0    6.0    10.0"    # +A1 (시간 압박)
    [C7]="50.0   0.02   0.5    200.0    1.0     300.0    800.0    6.0    10.0"    # +A2 (스핀, full = ours)
)

CONFIG_ORDER=(C0 C1 C2 C3 C4 C5 C6 C7)

# ============================================================
# 디렉토리 준비
# ============================================================
mkdir -p "${LOG_BASE}"
mkdir -p "${SAVE_BASE}"

MASTER_LOG="${LOG_BASE}/ablation_master.log"
echo "===== §30 ablation start: $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${MASTER_LOG}"
echo "Total runs: 24 (8 config × 3 seed), timesteps=${TIMESTEPS}" | tee -a "${MASTER_LOG}"
echo "Estimated total time: 28~32 hours (1.2~1.3 days)" | tee -a "${MASTER_LOG}"
echo "Log base:  ${LOG_BASE}" | tee -a "${MASTER_LOG}"
echo "Save base: ${SAVE_BASE}" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

# ============================================================
# 21 run 순차 실행
# ============================================================
TOTAL_RUNS=24
CURRENT_RUN=0
RUN_START_GLOBAL=$(date +%s)

for cfg_name in "${CONFIG_ORDER[@]}"; do
    read -r w_prog w_time w_spin w_np w_npsoft w_to w_succ w_dang w_close <<< "${CONFIGS[$cfg_name]}"
    for seed in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        run_name="${cfg_name}_seed${seed}"
        run_log_dir="${LOG_BASE}/${run_name}"
        run_save_dir="${SAVE_BASE}/${run_name}"
        run_stdout_log="${LOG_BASE}/${run_name}.log"

        # 이미 완료된 run 은 skip (ablation_eval.json 존재 시)
        if [ -f "${run_save_dir}/ablation_eval.json" ]; then
            echo "[$(date '+%H:%M:%S')] [${CURRENT_RUN}/${TOTAL_RUNS}] SKIP ${run_name}: ablation_eval.json already exists" | tee -a "${MASTER_LOG}"
            continue
        fi

        run_start=$(date +%s)
        echo "" | tee -a "${MASTER_LOG}"
        echo "[$(date '+%H:%M:%S')] [${CURRENT_RUN}/${TOTAL_RUNS}] START ${run_name}" | tee -a "${MASTER_LOG}"
        echo "  weights: A0=${w_prog} A1=${w_time} A2=${w_spin} A3-t=${w_np} A3-s=${w_npsoft} A4=${w_to} A5=${w_succ} A6-d=${w_dang} A6-c=${w_close}" | tee -a "${MASTER_LOG}"

        mkdir -p "${run_log_dir}"
        mkdir -p "${run_save_dir}"

        "${VENV}" train_ppo_2d.py \
            --total_timesteps "${TIMESTEPS}" \
            --seed "${seed}" \
            --learning_rate 1e-4 \
            --clip_range 0.2 \
            --n_epochs 10 \
            --target_kl 0.1 \
            --eval_freq 1000000 \
            --eval_runs 100 \
            --save_best \
            --do_eval \
            --w_progress "${w_prog}" \
            --w_time_pressure "${w_time}" \
            --w_spin "${w_spin}" \
            --w_no_progress "${w_np}" \
            --w_no_progress_soft "${w_npsoft}" \
            --w_timeout "${w_to}" \
            --success_reward "${w_succ}" \
            --w_danger "${w_dang}" \
            --w_close "${w_close}" \
            --log_dir "${run_log_dir}" \
            --save_dir "${run_save_dir}" \
            > "${run_stdout_log}" 2>&1

        run_end=$(date +%s)
        run_duration=$((run_end - run_start))
        run_hours=$((run_duration / 3600))
        run_minutes=$(((run_duration % 3600) / 60))

        # ablation_eval.json 생성 확인
        if [ -f "${run_save_dir}/ablation_eval.json" ]; then
            weighted=$(grep -oP '"weighted_score":\s*\K[0-9.]+' "${run_save_dir}/ablation_eval.json" | head -1)
            echo "[$(date '+%H:%M:%S')] [${CURRENT_RUN}/${TOTAL_RUNS}] DONE  ${run_name}: weighted=${weighted}, duration=${run_hours}h${run_minutes}m" | tee -a "${MASTER_LOG}"
        else
            echo "[$(date '+%H:%M:%S')] [${CURRENT_RUN}/${TOTAL_RUNS}] FAIL  ${run_name}: ablation_eval.json not found! Check ${run_stdout_log}" | tee -a "${MASTER_LOG}"
        fi
    done
done

# ============================================================
# 완료 요약
# ============================================================
RUN_END_GLOBAL=$(date +%s)
TOTAL_DURATION=$((RUN_END_GLOBAL - RUN_START_GLOBAL))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo "" | tee -a "${MASTER_LOG}"
echo "===== §30 ablation end: $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${MASTER_LOG}"
echo "Total elapsed: ${TOTAL_HOURS}h${TOTAL_MINUTES}m" | tee -a "${MASTER_LOG}"

# 24 개 ablation_eval.json 존재 확인
COMPLETED=$(find "${SAVE_BASE}" -name "ablation_eval.json" 2>/dev/null | wc -l)
echo "Completed runs: ${COMPLETED}/24" | tee -a "${MASTER_LOG}"

if [ "${COMPLETED}" -eq 24 ]; then
    echo "✅ All 24 runs completed successfully." | tee -a "${MASTER_LOG}"
    echo "Next step: Phase 3 — scripts/aggregate_ablation.py 작성 및 실행" | tee -a "${MASTER_LOG}"
else
    echo "⚠ Only ${COMPLETED}/24 runs have ablation_eval.json. Check master log for FAIL entries." | tee -a "${MASTER_LOG}"
    exit 1
fi
