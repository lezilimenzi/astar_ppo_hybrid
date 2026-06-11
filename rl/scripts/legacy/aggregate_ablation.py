#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_ablation.py

§30 ablation 결과 자동 집계 스크립트.

입력: ./models_ppo_2d/ablation/{C0..C7}_seed{42,123,7}/ablation_eval.json
출력:
  - CSV: 모든 run × 모든 시나리오 × 모든 metric 의 long-format raw 데이터
  - Markdown: 학회 Table III 용 6 개 집계 표

사용:
  # norm_reward=False 환경 결과 집계
  python3 scripts/aggregate_ablation.py \\
      --input_dir ./models_ppo_2d/ablation_norm_false \\
      --output_csv ./ablation_norm_false_results.csv \\
      --output_md ./ablation_norm_false_results.md \\
      --label "norm_reward=False"

  # norm_reward=True 환경 결과 집계 (§30b 후)
  python3 scripts/aggregate_ablation.py \\
      --input_dir ./models_ppo_2d/ablation \\
      --output_csv ./ablation_norm_true_results.csv \\
      --output_md ./ablation_norm_true_results.md \\
      --label "norm_reward=True"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 설정
# ============================================================

CONFIGS: List[str] = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
SEEDS: List[int] = [42, 123, 7]

# 누적 ablation 의 각 단계가 추가하는 항 (학회 표 표기용)
CONFIG_ADDITIONS: Dict[str, str] = {
    "C0": "base (모든 항 OFF, 충돌 페널티만)",
    "C1": "+A0 (BFS-progress)",
    "C2": "+A5 (success bonus)",
    "C3": "+A3 (no-progress)",
    "C4": "+A4 (timeout)",
    "C5": "+A6 (B1 obstacle potential)",
    "C6": "+A1 (시간 압박)",
    "C7": "+A2 (스핀, full = ours)",
}

# 평가 시나리오 (학회 표 컬럼 순서)
SCENARIOS: List[str] = [
    "obs_0", "obs_5", "obs_10",
    "obstacle_density_15", "obstacle_density_20",
    "narrow_passage", "course_l_s",
]
SCENARIOS_DISPLAY: Dict[str, str] = {
    "obs_0": "obs_0",
    "obs_5": "obs_5",
    "obs_10": "obs_10",
    "obstacle_density_15": "d15",
    "obstacle_density_20": "d20",
    "narrow_passage": "narrow",
    "course_l_s": "course_l_s",
}

# 추출할 시나리오별 metric (key, format)
SCENARIO_METRICS: List[Tuple[str, str]] = [
    ("reached_rate", ".4f"),
    ("collision_rate", ".4f"),
    ("no_progress_rate", ".4f"),
    ("timeout_rate", ".4f"),
    ("mean_episode_length", ".1f"),
    ("mean_episode_reward", ".2f"),
]


# ============================================================
# 데이터 로드
# ============================================================

def load_ablation_eval(input_dir: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """모든 run 의 ablation_eval.json 을 로드.

    Returns:
        {config: {seed: ablation_eval_dict}}, 누락된 run 은 키 없음.
    """
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for cfg in CONFIGS:
        results[cfg] = {}
        for seed in SEEDS:
            path = os.path.join(input_dir, f"{cfg}_seed{seed}", "ablation_eval.json")
            if os.path.isfile(path):
                with open(path, "r") as f:
                    results[cfg][seed] = json.load(f)
    return results


def count_completed(results: Dict[str, Dict[int, Dict[str, Any]]]) -> int:
    return sum(len(seeds) for seeds in results.values())


# ============================================================
# CSV 생성 (long-format raw 데이터)
# ============================================================

def write_csv(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: str,
    label: str,
) -> None:
    """모든 run × 시나리오 × metric 의 long-format CSV.

    컬럼: env_label, config, seed, scenario, weighted_score,
          reached_rate, collision_rate, no_progress_rate, timeout_rate,
          mean_episode_length, mean_episode_reward, avg_distance,
          ablation_kwargs (JSON string)
    """
    rows: List[Dict[str, Any]] = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            data = results.get(cfg, {}).get(seed)
            if data is None:
                continue
            ws = data.get("weighted_score", float("nan"))
            akw = json.dumps(data.get("ablation_kwargs", {}), separators=(",", ":"))
            for sc in data.get("scenarios", {}):
                m = data["scenarios"][sc]
                rows.append({
                    "env_label": label,
                    "config": cfg,
                    "seed": seed,
                    "scenario": sc,
                    "weighted_score": ws,
                    "reached_rate": m.get("reached_rate"),
                    "collision_rate": m.get("collision_rate"),
                    "no_progress_rate": m.get("no_progress_rate"),
                    "timeout_rate": m.get("timeout_rate"),
                    "mean_episode_length": m.get("mean_episode_length"),
                    "mean_episode_reward": m.get("mean_episode_reward"),
                    "avg_distance": m.get("avg_distance"),
                    "ablation_kwargs": akw,
                })

    if not rows:
        print(f"[CSV] No completed runs found in input. Skipping CSV.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV] Wrote {len(rows)} rows to {output_path}")


# ============================================================
# Markdown 표 생성
# ============================================================

def _per_seed_weighted(data: Dict[int, Dict[str, Any]], seed: int) -> Optional[float]:
    if seed in data:
        return data[seed].get("weighted_score")
    return None


def _per_seed_scenario_metric(
    data: Dict[int, Dict[str, Any]],
    seed: int,
    scenario: str,
    metric: str,
) -> Optional[float]:
    d = data.get(seed)
    if d is None:
        return None
    sc = d.get("scenarios", {}).get(scenario)
    if sc is None:
        return None
    return sc.get(metric)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _fmt(v: Optional[float], fmt: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return format(v, fmt)


def write_markdown(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: str,
    label: str,
    input_dir: str,
) -> None:
    """학회 Table III 용 6 개 집계 표 markdown 생성."""
    completed = count_completed(results)
    total = len(CONFIGS) * len(SEEDS)

    lines: List[str] = []
    lines.append(f"# §30 Ablation 결과 — {label}\n")
    lines.append(f"- 입력: `{input_dir}`")
    lines.append(f"- Config: {', '.join(CONFIGS)} ({len(CONFIGS)} 개)")
    lines.append(f"- Seed: {SEEDS}")
    lines.append(f"- 완료된 run: **{completed} / {total}**\n")

    # ========================================================
    # (a) 종합 weighted_score (3 seed)
    # ========================================================
    lines.append("## (a) 종합 weighted_score (3 seed)\n")
    lines.append("| Config | 추가 항 | seed=42 | seed=123 | seed=7 | mean | std |")
    lines.append("|--------|--------|---------|----------|--------|------|-----|")

    overall_means: Dict[str, float] = {}
    for cfg in CONFIGS:
        data = results.get(cfg, {})
        ws_list = [v for v in (
            _per_seed_weighted(data, 42),
            _per_seed_weighted(data, 123),
            _per_seed_weighted(data, 7),
        ) if v is not None]
        mean, std = _mean_std(ws_list)
        if mean is not None:
            overall_means[cfg] = mean
        lines.append(
            f"| {cfg} | {CONFIG_ADDITIONS[cfg]} "
            f"| {_fmt(_per_seed_weighted(data, 42), '.4f')} "
            f"| {_fmt(_per_seed_weighted(data, 123), '.4f')} "
            f"| {_fmt(_per_seed_weighted(data, 7), '.4f')} "
            f"| **{_fmt(mean, '.4f')}** | {_fmt(std, '.4f')} |"
        )
    lines.append("")

    # ========================================================
    # (b) 단계별 marginal Δ weighted_score
    # ========================================================
    lines.append("## (b) 단계별 marginal Δ weighted_score (mean 기준)\n")
    lines.append("| 단계 | 추가 항 | mean weighted | Δ vs prev |")
    lines.append("|------|--------|---------------|-----------|")
    prev: Optional[float] = None
    for cfg in CONFIGS:
        m = overall_means.get(cfg)
        if m is None:
            lines.append(f"| {cfg} | {CONFIG_ADDITIONS[cfg]} | N/A | N/A |")
            continue
        delta_str = "—" if prev is None else f"{m - prev:+.4f}"
        lines.append(f"| {cfg} | {CONFIG_ADDITIONS[cfg]} | {m:.4f} | {delta_str} |")
        prev = m
    lines.append("")

    # ========================================================
    # (c)~(h) 시나리오별 metric (3 seed mean)
    # ========================================================
    metric_labels = [
        ("reached_rate", ".2f", "(c) 시나리오별 reached_rate"),
        ("collision_rate", ".2f", "(d) 시나리오별 collision_rate"),
        ("no_progress_rate", ".2f", "(e) 시나리오별 no_progress_rate"),
        ("timeout_rate", ".2f", "(f) 시나리오별 timeout_rate"),
        ("mean_episode_length", ".1f", "(g) 시나리오별 mean_episode_length (step)"),
        ("mean_episode_reward", ".1f", "(h) 시나리오별 mean_episode_reward"),
    ]

    for metric, fmt, title in metric_labels:
        lines.append(f"## {title} (3 seed mean)\n")
        header_cells = ["Config"] + [SCENARIOS_DISPLAY[sc] for sc in SCENARIOS]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["--------"] * len(header_cells)) + "|")
        for cfg in CONFIGS:
            data = results.get(cfg, {})
            cells = [cfg]
            for sc in SCENARIOS:
                vals = [v for v in (
                    _per_seed_scenario_metric(data, 42, sc, metric),
                    _per_seed_scenario_metric(data, 123, sc, metric),
                    _per_seed_scenario_metric(data, 7, sc, metric),
                ) if v is not None]
                mean, _ = _mean_std(vals)
                cells.append(_fmt(mean, fmt))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # ========================================================
    # (i) Ablation kwargs 검증 — 어떤 config 가 어떤 항 ON 인지
    # ========================================================
    lines.append("## (i) Ablation kwargs 검증\n")
    lines.append(
        "| Config | w_progress | w_time_pressure | w_spin | w_no_progress "
        "| w_no_progress_soft | w_timeout | success_reward | w_danger | w_close |"
    )
    lines.append("|--------|-----------|-----------------|--------|---------------|"
                 "--------------------|-----------|----------------|----------|---------|")
    kwarg_keys = [
        "w_progress", "w_time_pressure", "w_spin", "w_no_progress",
        "w_no_progress_soft", "w_timeout", "success_reward", "w_danger", "w_close",
    ]
    for cfg in CONFIGS:
        data = results.get(cfg, {})
        # seed=42 의 ablation_kwargs 를 대표로 표시 (모든 seed 동일해야 함)
        d42 = data.get(42, {})
        kwargs = d42.get("ablation_kwargs", {}) if isinstance(d42, dict) else {}
        cells = [cfg] + [_fmt(kwargs.get(k), ".2f") for k in kwarg_keys]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("---\n")
    lines.append(
        "**원본 데이터**: 각 run 의 `ablation_eval.json` 에서 직접 추출. "
        f"Aggregate 시점: {label}"
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[MD]  Wrote markdown to {output_path}")


# ============================================================
# 두 환경 비교 (선택, --compare_dir 옵션)
# ============================================================

def write_comparison_markdown(
    results_a: Dict[str, Dict[int, Dict[str, Any]]],
    results_b: Dict[str, Dict[int, Dict[str, Any]]],
    label_a: str,
    label_b: str,
    output_path: str,
) -> None:
    """두 환경 (예: norm=False vs norm=True) 의 weighted_score 비교 표."""
    lines: List[str] = []
    lines.append(f"# §30 Ablation 환경 비교: {label_a} vs {label_b}\n")
    lines.append(
        f"| Config | 추가 항 | {label_a} mean | {label_b} mean | Δ ({label_b} − {label_a}) |"
    )
    lines.append("|--------|--------|-----|-----|-----|")
    for cfg in CONFIGS:
        ws_a_list = [_per_seed_weighted(results_a.get(cfg, {}), s) for s in SEEDS]
        ws_b_list = [_per_seed_weighted(results_b.get(cfg, {}), s) for s in SEEDS]
        ws_a_list = [v for v in ws_a_list if v is not None]
        ws_b_list = [v for v in ws_b_list if v is not None]
        m_a, _ = _mean_std(ws_a_list)
        m_b, _ = _mean_std(ws_b_list)
        delta = (m_b - m_a) if (m_a is not None and m_b is not None) else None
        lines.append(
            f"| {cfg} | {CONFIG_ADDITIONS[cfg]} "
            f"| {_fmt(m_a, '.4f')} | {_fmt(m_b, '.4f')} "
            f"| {_fmt(delta, '+.4f')} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[MD]  Wrote comparison to {output_path}")


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="§30 ablation 결과 집계 스크립트")
    p.add_argument(
        "--input_dir", type=str, required=True,
        help="ablation 결과가 저장된 디렉토리 (예: ./models_ppo_2d/ablation_norm_false)",
    )
    p.add_argument(
        "--output_csv", type=str, default=None,
        help="CSV 출력 경로 (지정 안 하면 input_dir 옆에 자동 생성)",
    )
    p.add_argument(
        "--output_md", type=str, default=None,
        help="Markdown 출력 경로 (지정 안 하면 input_dir 옆에 자동 생성)",
    )
    p.add_argument(
        "--label", type=str, default="ablation",
        help="환경 레이블 (예: 'norm_reward=False', 'norm_reward=True')",
    )
    p.add_argument(
        "--compare_dir", type=str, default=None,
        help="비교할 다른 환경의 디렉토리 (지정 시 비교 markdown 추가 생성)",
    )
    p.add_argument(
        "--compare_label", type=str, default="other",
        help="--compare_dir 환경의 레이블",
    )
    p.add_argument(
        "--compare_md", type=str, default=None,
        help="비교 markdown 출력 경로",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    # 출력 경로 자동 결정
    base = os.path.basename(args.input_dir.rstrip("/"))
    parent = os.path.dirname(args.input_dir.rstrip("/")) or "."
    output_csv = args.output_csv or os.path.join(parent, f"{base}_results.csv")
    output_md = args.output_md or os.path.join(parent, f"{base}_results.md")

    print(f"[Aggregate] Loading from: {args.input_dir}")
    results = load_ablation_eval(args.input_dir)
    completed = count_completed(results)
    total = len(CONFIGS) * len(SEEDS)
    print(f"[Aggregate] Completed runs: {completed} / {total}")

    if completed == 0:
        print("[Aggregate] No ablation_eval.json found. Exiting.")
        return

    # CSV 생성
    write_csv(results, output_csv, args.label)

    # Markdown 생성
    write_markdown(results, output_md, args.label, args.input_dir)

    # 두 환경 비교 (옵션)
    if args.compare_dir:
        if not os.path.isdir(args.compare_dir):
            print(f"[Aggregate] Compare dir not found: {args.compare_dir}, skipping comparison.")
        else:
            print(f"[Aggregate] Loading compare dir: {args.compare_dir}")
            results_b = load_ablation_eval(args.compare_dir)
            compare_md = args.compare_md or os.path.join(parent, "ablation_comparison.md")
            write_comparison_markdown(
                results, results_b, args.label, args.compare_label, compare_md,
            )

    print("[Aggregate] Done.")


if __name__ == "__main__":
    main()
