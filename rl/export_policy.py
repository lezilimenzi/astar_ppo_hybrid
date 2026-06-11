#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_policy.py

Step 1: 정책 내보내기 (ROS2 + PX4 드론 배포용)

학습된 PPO 모델과 VecNormalize 통계를 로드하여:
1) obs_norm_hybrid.json: 관측 정규화(mean/std), action 스케일/스펙 저장
2) policy_hybrid_ts.pt: TorchScript 정책 (action만 반환하는 wrapper, deterministic=True)

사용:
  python3 export_policy.py \
    --model ./models_ppo_2d/ppo_uav2d_best_training \
    --vecnorm ./models_ppo_2d/vecnormalize.pkl \
    --output_dir ./export

주의:
- VecNormalize는 pickle.load() 대신 VecNormalize.load(path, venv)로 로드 (버전 안정성)
- dummy env 파라미터는 학습과 "동일"해야 함 (obs/action space 불일치 방지)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_2d_nav import UAVNav2DEnv


# -----------------------------
# Action-only wrapper
# (SB3 policy는 보통 action, value, log_prob 반환 → action만 필요)
# -----------------------------
class ActionOnlyWrapper(torch.nn.Module):
    """SB3 policy를 감싸서 action만 반환. deterministic=True 고정. TorchScript trace용."""

    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action, _, _ = self.policy(obs, deterministic=True)
        return action


def make_env(args: argparse.Namespace) -> DummyVecEnv:
    """
    학습과 동일한 파라미터로 env를 만드는 것이 원칙.
    (env 생성자 시그니처가 다르면 TypeError가 날 수 있어서 fallback 처리)
    """
    def _make() -> UAVNav2DEnv:
        # 최대한 학습과 동일하게 전달 (env_2d_nav.py의 __init__ 시그니처에 맞게 필요시 수정)
        kwargs: Dict[str, Any] = dict(
            obs_beams=args.obs_beams,
            max_range=args.max_range,
            world_size=args.world_size,
            v_min=args.v_min,
            v_max=args.v_max,
            yaw_rate_max=args.yaw_rate_max,
        )
        try:
            return UAVNav2DEnv(**kwargs)
        except TypeError:
            # fallback: 최소 인자만
            return UAVNav2DEnv(obs_beams=args.obs_beams, max_range=args.max_range)

    return DummyVecEnv([_make])


def export_norm_json(
    out_dir: str,
    args: argparse.Namespace,
    mean: np.ndarray,
    std: np.ndarray,
    clip_obs: float,
    epsilon: float,
    obs_dim: int,
    action_dim: int,
    filename: str = "obs_norm_hybrid.json",
) -> str:
    """obs_norm.json 저장"""

    # obs 구성 (body frame, no heading):
    #   [ lidar_0..lidar_{N-1}, goal_rel_x, goal_rel_y ]  (66 dim)
    # env norm:
    #   lidar: r/max_range*2 - 1  -> [-1,1]
    #   goal_rel: R(-yaw) @ (goal-pos) / world_size  (body frame, x=forward, y=left)
    obs_ordering = [
        {
            "name": "lidar",
            "indices": f"0:{args.obs_beams}",
            "unit": "m",
            "raw_range": [0.0, float(args.max_range)],
            "env_norm": "lidar/max_range*2-1 -> [-1,1]",
        },
        {
            "name": "goal_rel",
            "indices": f"{args.obs_beams}:{args.obs_beams+2}",
            "unit": "m/m",
            "raw_range": "[-1,1] (scaled)",
            "env_norm": "R(-yaw)@(goal-pos)/world_size (body frame)",
        },
    ]

    norm_dict = {
        "obs": {
            "obs_dim": int(obs_dim),
            "ordering": obs_ordering,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "clip_obs": float(clip_obs),
            "epsilon": float(epsilon),
            "note": "Apply env_norm first, then VecNormalize: (obs-mean)/std, then clip to [-clip_obs, clip_obs].",
        },
        "action": {
            "action_dim": int(action_dim),
            "action_range": [-1.0, 1.0],
            "action_type": "velocity",
            "scale": {
                "v_min": float(args.v_min),
                "v_max": float(args.v_max),
                "max_vxy": float(args.v_max),
                "max_yaw_rate": float(args.yaw_rate_max),
            },
            "formula": {
                "v_cmd": "v_min + (a_v + 1) / 2 * (v_max - v_min)",
                "yaw_rate_cmd": "a_yaw * max_yaw_rate",
            },
        },
        "env_params": {
            "obs_beams": int(args.obs_beams),
            "max_range": float(args.max_range),
            "world_size": float(args.world_size),
        },
    }

    norm_path = os.path.join(out_dir, filename)
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_dict, f, indent=2, ensure_ascii=False)

    return norm_path


def env_norm_example(raw_obs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    배포 파이프라인 점검용 "예시" env_norm.
    실제 env_2d_nav.py의 관측 생성 방식과 동일해야 함.
    obs: lidar (n) + goal_rel body (2) = n+2
    """
    n = args.obs_beams
    assert raw_obs.shape[0] == n + 2

    lidar = raw_obs[:n]
    goal = raw_obs[n:n+2]

    # lidar: [0,max_range] -> [-1,1]
    lidar_n = np.clip(lidar, 0.0, args.max_range) / float(args.max_range)
    lidar_n = lidar_n * 2.0 - 1.0

    # goal_rel body: 이미 R(-yaw)@(goal-pos)/world_size, clip
    goal_n = np.clip(goal, -1.0, 1.0)

    return np.concatenate([lidar_n, goal_n]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Export PPO policy for ROS2/PX4 deployment")
    p.add_argument("--model", type=str, required=True, help="PPO model path (.zip 생략 가능)")
    p.add_argument("--vecnorm", type=str, required=True, help="VecNormalize path (VecNormalize.save()로 저장된 파일)")
    p.add_argument("--output_dir", type=str, default="./export", help="Output directory")
    p.add_argument("--norm_filename", type=str, default="obs_norm_hybrid.json",
                   help="Output filename for obs/action normalization JSON")
    p.add_argument("--policy_filename", type=str, default="policy_hybrid_ts.pt",
                   help="Output filename for TorchScript policy (.pt)")

    # env 스펙(학습과 반드시 일치)
    p.add_argument("--obs_beams", type=int, default=64, help="Must match training env")
    p.add_argument("--max_range", type=float, default=10.0, help="Lidar max range (m)")
    p.add_argument("--world_size", type=float, default=20.0, help="World size for goal_rel scale")

    # action 스펙(학습과 반드시 일치)
    p.add_argument("--v_min", type=float, default=0.3, help="Min velocity (m/s)")
    p.add_argument("--v_max", type=float, default=0.8, help="Max velocity (m/s)")
    p.add_argument("--yaw_rate_max", type=float, default=1.0, help="Max yaw rate (rad/s)")

    # 검증 옵션
    p.add_argument("--verify_full_pipeline", action="store_true",
                   help="raw->env_norm->vecnorm->policy까지 예시 파이프라인 검증도 수행 (env_norm이 실제와 동일해야 의미 있음)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- (A) dummy env 생성 (학습과 동일 파라미터 권장) ----
    dummy_env = make_env(args)

    # ---- (B) VecNormalize 로드 (안정적인 방식) ----
    vecnorm = VecNormalize.load(args.vecnorm, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    obs_rms = vecnorm.obs_rms
    mean = np.array(obs_rms.mean, dtype=np.float64)
    var = np.array(obs_rms.var, dtype=np.float64)
    epsilon = float(getattr(vecnorm, "epsilon", 1e-8))
    std = np.sqrt(var + epsilon)
    clip_obs = float(getattr(vecnorm, "clip_obs", 10.0))

    obs_dim = int(mean.shape[0])

    # ---- (C) PPO 로드 ----
    load_path = args.model.rstrip("/")
    if load_path.lower().endswith(".zip"):
        load_path = load_path[:-4]

    model = PPO.load(load_path, env=dummy_env)

    # action_dim은 하드코딩 금지
    action_dim = int(np.prod(model.action_space.shape))
    model_obs_dim = int(np.prod(model.observation_space.shape))

    # ---- (D) 필수 일치 검사 ----
    if model_obs_dim != obs_dim:
        raise ValueError(f"obs_dim mismatch: model={model_obs_dim}, vecnorm={obs_dim}")

    # obs 구성(lidar N + goal_rel 2) 확인 (body frame, no heading)
    expected_obs_dim = int(args.obs_beams + 2)
    if obs_dim != expected_obs_dim:
        raise ValueError(
            f"Unexpected obs_dim={obs_dim}. Expected beams+2={expected_obs_dim}. "
            f"(env 관측 구성/VecNormalize 파일이 학습과 다를 수 있음)"
        )

    # ---- (E) obs_norm.json 저장 ----
    norm_path = export_norm_json(
        out_dir=args.output_dir,
        args=args,
        mean=mean.astype(np.float64),
        std=std.astype(np.float64),
        clip_obs=clip_obs,
        epsilon=epsilon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        filename=args.norm_filename,
    )
    print(f"Saved: {norm_path}")

    # ---- (F) TorchScript export ----
    model.policy.set_training_mode(False)
    policy = model.policy
    wrapper = ActionOnlyWrapper(policy).eval()

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, dummy_input)

    policy_path = os.path.join(args.output_dir, args.policy_filename)
    scripted.save(policy_path)
    print(f"Saved: {policy_path}")

    # ---- (G) 검증 1: wrapper vs TorchScript (trace 무결성) ----
    test_obs = torch.randn(1, obs_dim, dtype=torch.float32) * 0.5
    with torch.no_grad():
        orig_action = wrapper(test_obs)
        ts_action = scripted(test_obs)
    diff = (orig_action - ts_action).abs().max().item()
    print(f"Verification-1 (wrapper vs TorchScript): max diff = {diff:.8f} (should be ~0)")

    # ---- (H) 검증 2: 예시 full pipeline (raw->env_norm->vecnorm->policy) ----
    if args.verify_full_pipeline:
        # raw_obs 예시 생성 (lidar [0,max_range], goal_rel body [-1,1])
        n = args.obs_beams
        raw_lidar = np.random.uniform(0.0, args.max_range, size=(n,)).astype(np.float32)
        raw_goal = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        raw_obs = np.concatenate([raw_lidar, raw_goal], axis=0).astype(np.float32)

        # (1) env_norm (예시) -> (2) VecNormalize normalize -> clip -> (3) policy
        obs_env = env_norm_example(raw_obs, args)
        obs_vn = (obs_env - mean.astype(np.float32)) / std.astype(np.float32)
        obs_vn = np.clip(obs_vn, -clip_obs, clip_obs).astype(np.float32)

        obs_t = torch.from_numpy(obs_vn).unsqueeze(0)
        with torch.no_grad():
            a_wrap = wrapper(obs_t)
            a_ts = scripted(obs_t)
        diff2 = (a_wrap - a_ts).abs().max().item()
        print(f"Verification-2 (full pipeline example): max diff = {diff2:.8f} (should be ~0)")
        print("  NOTE: 이 검증은 env_norm_example이 '실제 env 관측 정규화'와 동일할 때만 의미가 큼")

    # ---- 안내 출력 ----
    print("\n=== Export complete ===")
    print("Artifacts:")
    print(f"  - {args.norm_filename}  : observation normalization & action scale/spec")
    print(f"  - {args.policy_filename}   : TorchScript policy (action only, deterministic)")

    print("\nROS2 inference flow (배포 시 권장 순서):")
    print("  1) raw obs build: [lidar_0..lidar_N, goal_rel_x, goal_rel_y] (body frame)")
    print("  2) apply env norm: lidar r/max_range*2-1, goal_rel=R(-yaw)@(goal-pos)/world_size")
    print(f"  3) apply VecNormalize: (obs-mean)/std, clip to [-{clip_obs}, {clip_obs}]")
    print("  4) policy_ts.pt(obs_tensor) -> action in [-1,1]")
    print("  5) scale:")
    print("     - v_cmd        = v_min + (a_v+1)/2*(v_max-v_min)")
    print("     - yaw_rate_cmd = a_yaw * max_yaw_rate")


if __name__ == "__main__":
    main()
