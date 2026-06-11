#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ppo_2d_ppo_only.py

train_ppo_2d.py 의 PPO-only ablation 변형.
- train_ppo_2d.py 와 학습 로직/하이퍼파라미터/콜백/평가 전부 동일.
- 유일한 차이: env 를 env_2d_nav_ppo_only.UAVNav2DEnv 로 import (A* lookahead 하이브리드 제거 baseline).
- 목적: hybrid(A*+PPO) vs ppo_only 의 학습 차이 비교.

논문 구조 기반 PPO 학습 + (선택) 평가 자동 실행 + (선택) 베스트 모델 저장.

핵심:
- PPO의 Advantage(GAE) + Trust Region(clip) 업데이트는 SB3 PPO가 수행
- V(s): critic value (rollout_buffer.values)
- A(s,a): rollout_buffer.advantages
- Q(s,a): 정의상 Q = V + A 로 추정치 로깅 가능

추가 기능:
1) 학습 종료 후 eval 자동 실행 (--do_eval)
2) 학습 중 주기적 eval로 best_model 저장 (--eval_freq, --save_best)
3) 학습 종료 시 직전보다 좋으면 ppo_uav2d_best_training.zip 저장

사용 예:
  # 기본 학습
  python3 train_ppo_2d_ppo_only.py

  # 학습 후 평가까지 한 번에
  python3 train_ppo_2d_ppo_only.py --do_eval --eval_runs 300 --eval_obstacles 0 5 10

  # 학습 중 주기적 eval + best_model 저장
  python3 train_ppo_2d_ppo_only.py --eval_freq 200000 --save_best --eval_runs 100 --eval_obstacles 8

  # 6M 학습 후 best_model로 3M 추가 학습 (fine-tune 보수 모드)
  python3 train_ppo_2d_ppo_only.py --resume_model ./models_ppo_2d/best_model --additional_timesteps 3000000 --learning_rate 5e-5 --clip_range 0.12 --target_kl 0.01 --n_epochs 5 --eval_freq 200000 --save_best
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, FloatSchedule

from env_2d_nav_ppo_only import UAVNav2DEnv
import torch


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    # Env

    # PPO
    n_envs: int = 8
    total_timesteps: int = 6_000_000
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.9983  # (§39-13) dt 0.3→0.05 보정: 0.99^(1/6)≈0.9983 → 실시간 discount horizon(~30s) 보존
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.03  # 0.08 → 0.01 (후반 수렴, scheduling 적용)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Misc
    seed: int = 42
    verbose: int = 1

    # Logging/Save
    log_dir: str = "./logs_ppo_2d"
    save_dir: str = "./models_ppo_2d"
    checkpoint_freq_timesteps: int = 500_000  # global timesteps 기준


# -----------------------------
# Utility: evaluation
# -----------------------------
@dataclass
class EvalMetrics:
    runs: int
    success: int
    collision: int
    no_progress: int
    timeout: int
    avg_steps: float
    avg_distance: float

    @property
    def success_rate(self) -> float:
        return self.success / max(self.runs, 1)

    @property
    def collision_rate(self) -> float:
        return self.collision / max(self.runs, 1)

    @property
    def no_progress_rate(self) -> float:
        return self.no_progress / max(self.runs, 1)

    @property
    def timeout_rate(self) -> float:
        return self.timeout / max(self.runs, 1)


def make_single_env(
    cfg: TrainConfig,
    num_obstacles: int,
    seed: int,
    eval_scenario: Optional[str] = None,
    eval_sensor_noise: bool = False,
) -> UAVNav2DEnv:
    env_kwargs = dict(
        num_obstacles=num_obstacles,
        seed=seed,
        eval_scenario=eval_scenario,
        eval_sensor_noise=eval_sensor_noise,
        use_curriculum=False,      # eval env는 커리큘럼 미사용: num_obstacles/goal_radius 고정
        curriculum_verbose=False,
    )
    env = UAVNav2DEnv(**env_kwargs)
    return env


def evaluate_model(
    model: PPO,
    cfg: TrainConfig,
    runs: int,
    num_obstacles: int,
    seed: int = 0,
    deterministic: bool = True,
    vecnorm_path: Optional[str] = None,
    print_every: int = 0,
    eval_scenario: Optional[str] = None,
    eval_sensor_noise: bool = False,
    print_label: Optional[str] = None,
) -> EvalMetrics:
    base_env = make_single_env(
        cfg,
        num_obstacles=num_obstacles,
        seed=seed,
        eval_scenario=eval_scenario,
        eval_sensor_noise=eval_sensor_noise,
    )
    if vecnorm_path:
        env = DummyVecEnv([lambda: base_env])
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        unwrapped = base_env  # for .pos access
    else:
        env = base_env
        unwrapped = base_env

    success = 0
    collision = 0
    no_progress = 0
    timeout = 0
    steps_list: List[int] = []
    dist_list: List[float] = []

    # VecEnv (DummyVecEnv) auto-resets when episode ends, so calling reset() each episode
    # would double-count: 300 episodes -> 600 resets. Only call reset() for first episode.
    use_vecenv = vecnorm_path is not None

    for ep in range(runs):
        if ep == 0 or not use_vecenv:
            # VecNormalize.reset() does not support seed; VecEnv.reset() may return (obs,) or (obs, info)
            try:
                result = env.reset(seed=seed + ep)
            except TypeError:
                result = env.reset()
            if isinstance(result, tuple) and len(result) >= 2:
                obs, info = result[0], result[1]
            else:
                obs = result
                info = {}

        done = False
        ep_steps = 0
        traveled = 0.0

        prev_pos = unwrapped.pos.copy()

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_result = env.step(action)
            # VecEnv returns (obs, rewards, dones, infos); Gymnasium returns (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                obs, rewards, dones, infos = step_result
                done = bool(dones[0])
                info = infos[0] if infos else {}

            ep_steps += 1
            traveled += float(np.linalg.norm(unwrapped.pos - prev_pos))
            prev_pos = unwrapped.pos.copy()

        reason = info.get("done_reason", "")
        if reason == "reached":
            success += 1
        elif reason == "collision":
            collision += 1
        elif reason == "no_progress":
            no_progress += 1
        else:
            timeout += 1

        steps_list.append(ep_steps)
        dist_list.append(traveled)

        # 주기적 통계 출력 (마지막 에피소드는 caller의 summary에서 출력하므로 제외)
        if print_every > 0 and (ep + 1) % print_every == 0 and (ep + 1) < runs:
            total = ep + 1
            reached_rate = success / total
            collision_rate = collision / total
            no_progress_rate = no_progress / total
            timeout_rate = timeout / total
            # 평가 시나리오/sensor_noise는 장애물 수와 무관 → 의미 있는 라벨 사용
            if print_label is not None:
                label = print_label
            elif eval_scenario:
                label = eval_scenario
            elif eval_sensor_noise:
                label = "sensor_noise"
            else:
                label = f"obs={num_obstacles}"
            print(
                f"[Eval {label}] episodes={total} "
                f"reached={success} ({reached_rate*100:.2f}%) "
                f"collision={collision} ({collision_rate*100:.2f}%) "
                f"no_progress={no_progress} ({no_progress_rate*100:.2f}%) "
                f"timeout={timeout} ({timeout_rate*100:.2f}%)"
            )

    metrics = EvalMetrics(
        runs=runs,
        success=success,
        collision=collision,
        no_progress=no_progress,
        timeout=timeout,
        avg_steps=float(np.mean(steps_list)) if steps_list else 0.0,
        avg_distance=float(np.mean(dist_list)) if dist_list else 0.0,
    )
    env.close()
    return metrics


def print_eval_summary(scenario: str, m: EvalMetrics, prefix: str = "") -> None:
    print(
        f"{prefix}[Eval {scenario}] "
        f"success={m.success}/{m.runs} ({m.success_rate*100:.2f}%) "
        f"collision={m.collision} ({m.collision_rate*100:.2f}%) "
        f"no_progress={m.no_progress} ({m.no_progress_rate*100:.2f}%) "
        f"timeout={m.timeout} ({m.timeout_rate*100:.2f}%) "
        f"avg_steps={m.avg_steps:.1f} avg_dist={m.avg_distance:.2f}"
    )


# -----------------------------
# Callbacks
# -----------------------------
class EntCoefScheduleCallback(BaseCallback):
    """
    ent_coef 적응형 조정:
    - base_coef: 정상 상태 ent_coef
    - boost_coef: entropy 부족 시 상향값
    - low_thr: entropy_loss 가 이보다 높으면(entropy 낮으면) boost 발동
    - high_thr: entropy_loss 가 이보다 낮으면(entropy 충분하면) base 복귀
    - cooldown: boost/복귀 후 최소 N rollout 동안 상태 유지 (진동 방지)
    - ceil_thr/floor_coef: (§40 B) entropy 천장 가드. entropy_loss < ceil_thr(과탐험)이면
      boost/cooldown 상태와 무관하게 ent_coef 를 floor_coef 로 하드캡해 std 런어웨이 차단.
    - fixed_ent_coef: 지정 시 adaptive 무시하고 고정값 유지 (--ent_coef 인자)
    """

    def __init__(
        self,
        total_timesteps: int,           # 미사용 (하위호환 유지)
        base_coef: float = 0.004,
        boost_coef: float = 0.03,
        low_thr: float = -0.6,
        high_thr: float = -0.9,
        cooldown: int = 10,
        ceil_thr: float = -2.5,
        floor_coef: float = 0.001,
        fixed_ent_coef: Optional[float] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.total_timesteps = int(total_timesteps)
        self.base_coef = base_coef
        self.boost_coef = boost_coef
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.cooldown = cooldown
        self.ceil_thr = ceil_thr
        self.floor_coef = floor_coef
        self.fixed_ent_coef = fixed_ent_coef
        self._boosted = False
        self._capped = False
        self._cooldown_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # --ent_coef 지정 시 고정값 유지
        if self.fixed_ent_coef is not None:
            self.model.ent_coef = self.fixed_ent_coef
            return

        entropy_loss = self.logger.name_to_value.get("train/entropy_loss")
        if entropy_loss is None:
            return  # 첫 rollout은 아직 기록 없음

        # (§40 B) entropy 천장 가드 (최우선): std 가 과도(entropy_loss < ceil_thr)하면
        # boost/cooldown 무시하고 floor 로 하드캡. base 로도 못 잡는 런어웨이를 차단.
        if entropy_loss < self.ceil_thr:
            if not self._capped and self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} < ceil {self.ceil_thr}"
                      f" → ent_coef ⤓ {self.floor_coef} (std 런어웨이 가드)")
            self.model.ent_coef = self.floor_coef
            self._capped = True
            self._boosted = False
            return
        if self._capped:
            # 가드 해제(entropy 정상 복귀) → base 로 복귀하고 토글 로직 재개
            self._capped = False
            self.model.ent_coef = self.base_coef
            self._cooldown_count = self.cooldown
            if self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} >= ceil {self.ceil_thr}"
                      f" → ent_coef ↩ base {self.base_coef} (가드 해제)")
            return

        if self._cooldown_count > 0:
            self._cooldown_count -= 1
            return

        if not self._boosted and entropy_loss > self.low_thr:
            # entropy 부족 → ent_coef 상향
            self.model.ent_coef = self.boost_coef
            self._boosted = True
            self._cooldown_count = self.cooldown
            if self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} > {self.low_thr}"
                      f" → ent_coef ↑ {self.boost_coef} (cooldown={self.cooldown})")
        elif self._boosted and entropy_loss < self.high_thr:
            # entropy 회복 → ent_coef 복귀
            self.model.ent_coef = self.base_coef
            self._boosted = False
            self._cooldown_count = self.cooldown
            if self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} < {self.high_thr}"
                      f" → ent_coef ↓ {self.base_coef} (cooldown={self.cooldown})")


class EpisodeStatsCallback(BaseCallback):
    """
    에피소드 결과 통계 + V/A/Q 추정치 로깅.
    """

    def __init__(self, print_every_episodes: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.print_every_episodes = int(print_every_episodes)
        self.ep_total = 0
        self.ep_success = 0
        self.ep_collision = 0
        self.ep_timeout = 0
        self.ep_no_progress = 0
        self._last_print_total = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue
            self.ep_total += 1
            reason = info.get("done_reason", "")
            if reason == "reached":
                self.ep_success += 1
            elif reason == "collision":
                self.ep_collision += 1
            elif reason == "no_progress":
                self.ep_no_progress += 1
            else:
                self.ep_timeout += 1  # max_steps 도달

        if (self.ep_total - self._last_print_total) >= self.print_every_episodes:
            self._last_print_total = self.ep_total
            total = max(self.ep_total, 1)
            reached_rate = self.ep_success / total
            collision_rate = self.ep_collision / total
            no_progress_rate = self.ep_no_progress / total
            timeout_rate = self.ep_timeout / total
            if self.verbose:
                print(
                    f"[episodes={self.ep_total}] "
                    f"reached={self.ep_success} ({reached_rate*100:.2f}%) "
                    f"collision={self.ep_collision} ({collision_rate*100:.2f}%) "
                    f"no_progress={self.ep_no_progress} ({no_progress_rate*100:.2f}%) "
                    f"timeout={self.ep_timeout} ({timeout_rate*100:.2f}%)"
                )
        return True

    def _on_rollout_end(self) -> None:
        # Advantage/Value/Q 추정치(정의 기반) 로깅
        try:
            rb = self.model.rollout_buffer
            adv = rb.advantages
            val = rb.values
            self.logger.record("debug/adv_mean", float(np.mean(adv)))
            self.logger.record("debug/v_mean", float(np.mean(val)))
            self.logger.record("debug/q_est_mean", float(np.mean(val + adv)))
        except Exception:
            pass


# best_model 가중치 (합 1.0): obs + density + narrow + Gazebo obstacle_course_light Course L(S)
EVAL_SCORE_WEIGHTS = {
    "obs_0": 0.08,
    "obs_5": 0.08,
    "obs_10": 0.08,
    "obstacle_density_15": 0.21,
    "obstacle_density_20": 0.21,
    "narrow_passage": 0.16,
    "course_l_s": 0.18,
}


class PeriodicEvalCallback(BaseCallback):
    """
    학습 중 주기적 평가 + (옵션) best_model 저장.

    - eval_freq: timesteps 단위 (전역 timesteps)
    - eval_runs: 평가 에피소드 수
    - save_best: 가중 점수(obs + stress + course_l_s) 기준으로 best_model.zip 저장
      score = EVAL_SCORE_WEIGHTS 합성 (TensorBoard eval/*_success_rate 동기)
    """

    def __init__(
        self,
        cfg: TrainConfig,
        eval_freq: int,
        eval_runs: int,
        eval_obstacles: List[int],
        save_path: str,
        save_best: bool = True,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.eval_freq = int(eval_freq)
        self.eval_runs = int(eval_runs)
        self.eval_obstacles = [int(x) for x in eval_obstacles]
        self.save_path = save_path
        self.save_best = bool(save_best)
        self.deterministic = bool(deterministic)

        self._last_eval_step = 0
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        vecnorm_path = os.path.join(self.save_path, "vecnormalize.pkl")
        os.makedirs(self.save_path, exist_ok=True)
        self.model.env.save(vecnorm_path)

        seed_base = self.cfg.seed + 10_000 + self.num_timesteps
        scores_dict: Dict[str, float] = {}

        # obs=0, 5, 10
        for n_obs in (0, 5, 10):
            m = evaluate_model(
                model=self.model,
                cfg=self.cfg,
                runs=self.eval_runs,
                num_obstacles=n_obs,
                seed=seed_base + n_obs,
                deterministic=self.deterministic,
                vecnorm_path=vecnorm_path,
                print_every=0,
            )
            key = f"obs_{n_obs}"
            scores_dict[key] = float(m.success_rate)
            if self.verbose:
                print_eval_summary(f"obs={n_obs}", m, prefix=f"[t={self.num_timesteps}] ")
            self.logger.record(f"eval/{key}_success_rate", scores_dict[key])
            self.logger.record(f"eval/{key}_collision_rate", float(m.collision_rate))

        # stress test + Gazebo S course (eval map)
        for scenario in ("obstacle_density_15", "obstacle_density_20", "narrow_passage", "course_l_s"):
            m = evaluate_model(
                model=self.model,
                cfg=self.cfg,
                runs=self.eval_runs,
                num_obstacles=0,
                seed=seed_base + hash(scenario) % 10000,
                deterministic=self.deterministic,
                vecnorm_path=vecnorm_path,
                print_every=0,
                eval_scenario=scenario,
                print_label=scenario,
            )
            scores_dict[scenario] = float(m.success_rate)
            if self.verbose:
                print_eval_summary(scenario, m, prefix=f"[t={self.num_timesteps}] ")
            self.logger.record(f"eval/{scenario}_success_rate", scores_dict[scenario])

        # 가중 점수
        weighted_score = sum(
            EVAL_SCORE_WEIGHTS.get(k, 0.0) * v for k, v in scores_dict.items()
        )
        self.logger.record("eval/weighted_score", weighted_score)
        self.logger.record("eval/mean_success_rate", float(np.mean(list(scores_dict.values()))))  # 호환용

        if self.save_best and weighted_score > self.best_score:
            self.best_score = weighted_score
            best_path = os.path.join(self.save_path, "best_model.zip")
            self.model.save(best_path)
            if self.verbose:
                print(f"[t={self.num_timesteps}] ✅ New best weighted_score={weighted_score:.4f} -> saved: {best_path}")

        return True


# -----------------------------
# Final evaluation helper
# -----------------------------
def _run_final_evaluation(
    model: PPO,
    cfg: TrainConfig,
    args: argparse.Namespace,
    vecnorm_path: str,
) -> None:
    """학습 종료 후 전체 시나리오 평가 + best_training 저장."""
    eval_seed = cfg.seed + 7777
    scores_dict: Dict[str, float] = {}

    if args.do_eval:
        print("\n=== Final Evaluation ===")
    for n_obs in (0, 5, 10):
        m = evaluate_model(
            model=model,
            cfg=cfg,
            runs=int(args.eval_runs),
            num_obstacles=int(n_obs),
            seed=eval_seed + int(n_obs),
            deterministic=True,
            vecnorm_path=vecnorm_path,
        )
        scores_dict[f"obs_{n_obs}"] = float(m.success_rate)
        if args.do_eval:
            print_eval_summary(f"obs={n_obs}", m)

    if args.do_eval:
        print("\n=== Stress Test Evaluation ===")
    for scenario in ("obstacle_density_15", "obstacle_density_20", "narrow_passage", "course_l_s"):
        m = evaluate_model(
            model=model,
            cfg=cfg,
            runs=int(args.eval_runs),
            num_obstacles=0,
            seed=eval_seed + hash(scenario) % 10000,
            deterministic=True,
            vecnorm_path=vecnorm_path,
            eval_scenario=scenario,
            print_every=0,
            print_label=scenario,
        )
        scores_dict[scenario] = float(m.success_rate)
        if args.do_eval:
            print_eval_summary(scenario, m)

    weighted_score = sum(EVAL_SCORE_WEIGHTS.get(k, 0.0) * v for k, v in scores_dict.items())

    best_training_path = os.path.join(cfg.save_dir, "ppo_uav2d_best_training.zip")
    meta_path = os.path.join(cfg.save_dir, "ppo_uav2d_best_training_meta.json")
    prev_best = -np.inf
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                prev_best = float(meta.get("weighted_score", -np.inf))
        except (json.JSONDecodeError, KeyError):
            pass

    if weighted_score > prev_best:
        model.save(best_training_path)
        with open(meta_path, "w") as f:
            json.dump({"weighted_score": weighted_score, "scores": scores_dict}, f, indent=2)
        print(f"New best training result (weighted_score={weighted_score:.4f}) -> saved: {best_training_path}")
    else:
        print(f"Training result (weighted_score={weighted_score:.4f}) <= prev best ({prev_best:.4f}), best_training not updated")

    # sensor_noise는 점수에 포함되지 않으므로 do_eval 시에만 추가 실행
    if args.do_eval:
        m = evaluate_model(
            model=model,
            cfg=cfg,
            runs=int(args.eval_runs),
            num_obstacles=2,
            seed=cfg.seed + 8888,
            deterministic=True,
            vecnorm_path=vecnorm_path,
            eval_sensor_noise=True,
            print_label="sensor_noise",
        )
        print_eval_summary("sensor_noise", m)
        print("=== End Evaluation ===")


# -----------------------------
# Env factory (train)
# -----------------------------
def make_env(cfg: TrainConfig, rank: int, seed: int):
    def _init():
        env = UAVNav2DEnv(
            seed=seed + rank,
            curriculum_verbose=(rank == 0),  # rank 0만 curriculum 로그 출력 (8회 중복 방지)
        )
        return Monitor(env)

    return _init


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # train core
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=None, help="학습률 (추가 학습 시 낮추면 fine-tuning)")
    p.add_argument("--clip_range", type=float, default=None, help="PPO clip_range (fine-tune 보수 모드: 0.1~0.15)")
    p.add_argument("--target_kl", type=float, default=None, help="PPO target_kl, KL early stop (fine-tune: ~0.01)")
    p.add_argument("--n_epochs", type=int, default=None, help="PPO n_epochs (흔들리면 줄이기)")
    p.add_argument("--verbose", type=int, default=1, help="0=로그 최소화, 1=기본")
    p.add_argument("--log_interval", type=int, default=10, help="debug/rollout/time/train 로그 출력 주기 (iteration 단위, 기본 10)")

    # eval after train
    p.add_argument("--do_eval", action="store_true", help="학습 종료 후 eval 자동 실행")
    p.add_argument("--eval_runs", type=int, default=300, help="평가 에피소드 수")
    p.add_argument("--eval_obstacles", type=int, nargs="+", default=[0, 5, 10], help="평가 장애물 개수 리스트")

    # periodic eval during training
    p.add_argument("--eval_freq", type=int, default=0, help="학습 중 주기 평가 timesteps (0이면 비활성)")
    p.add_argument("--save_best", action="store_true", help="주기 평가에서 best_model 저장")
    p.add_argument("--resume_model", type=str, default=None, help="Resume PPO model path")
    p.add_argument("--vecnorm_path", type=str, default=None, help="VecNormalize stats path")
    p.add_argument(
        "--additional_timesteps",
        type=int,
        default=None,
        help="(resume_model과 함께) 추가로 학습할 step 수. 이전 학습에서 이어서 학습.",
    )
    p.add_argument("--ent_coef", type=float, default=None, help="고정 ent_coef (지정 시 adaptive 무시)")
    # (§40 A) base 0.01→0.004: C2_dt05(base 0.01)는 dt=0.05 에서 base 만으로도 std 가
    # 5.0 까지 폭주(후반 explained_variance 0.93→0.70, clip_fraction 0.025). boost 0.03 은
    # 빠른 수렴에 유리하므로 유지하되, hold 단계인 base 를 entsqrt(0.004) 수준으로 낮춰 std~0.4 유지.
    p.add_argument("--ent_base_coef", type=float, default=0.004, help="adaptive ent_coef 기본값 (entropy 충분 시, hold). §40 A: 0.01→0.004")
    p.add_argument("--ent_boost_coef", type=float, default=0.03, help="adaptive ent_coef 상향값 (entropy 부족 시, push)")
    p.add_argument("--ent_low_thr", type=float, default=-0.6, help="entropy_loss 이보다 높으면 boost 발동")
    p.add_argument("--ent_high_thr", type=float, default=-0.9, help="entropy_loss 이보다 낮으면 base 복귀")
    p.add_argument("--ent_cooldown", type=int, default=10, help="boost/복귀 후 최소 유지 rollout 수")
    # (§40 B) entropy 천장 가드: base 로도 std 가 새면(과탐험) floor 로 하드캡해 런어웨이 차단.
    p.add_argument("--ent_ceil_thr", type=float, default=-2.5, help="entropy_loss 이보다 낮으면(≈std>0.65, 과탐험) ent_coef 를 floor 로 하드캡 (boost/cooldown 무시)")
    p.add_argument("--ent_floor_coef", type=float, default=0.001, help="천장 가드 발동 시 강제 ent_coef (std 런어웨이 차단)")

    # 출력 디렉토리 override (미지정 시 cfg 기본값: ./logs_ppo_2d, ./models_ppo_2d)
    # best_model.zip / ppo_uav2d_best_training.zip / checkpoint / vecnormalize 모두 save_dir 로 저장됨
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard 로그 디렉토리 (default: ./logs_ppo_2d)")
    p.add_argument("--save_dir", type=str, default=None, help="모델/체크포인트 저장 디렉토리 (default: ./models_ppo_2d)")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig()
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if args.clip_range is not None:
        cfg.clip_range = float(args.clip_range)
    if args.n_epochs is not None:
        cfg.n_epochs = int(args.n_epochs)
    cfg.n_envs = int(args.n_envs)
    cfg.seed = int(args.seed)

    # 출력 디렉토리 override (best_model/checkpoint/final/vecnormalize 모두 save_dir 기준)
    if args.log_dir is not None:
        cfg.log_dir = str(args.log_dir)
    if args.save_dir is not None:
        cfg.save_dir = str(args.save_dir)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    set_random_seed(cfg.seed)

    if args.additional_timesteps is not None and not args.resume_model:
        raise ValueError("--additional_timesteps requires --resume_model")

    if args.resume_model and not args.vecnorm_path:
        args.vecnorm_path = os.path.join(cfg.save_dir, "vecnormalize.pkl")

    # Vec env (train)
    if cfg.n_envs > 1:
        env = SubprocVecEnv([make_env(cfg, i, cfg.seed) for i in range(cfg.n_envs)])
    else:
        env = DummyVecEnv([make_env(cfg, 0, cfg.seed)])

    if args.vecnorm_path:
        env = VecNormalize.load(args.vecnorm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # checkpoint save_freq는 Subproc일 때 env step 단위 차이를 감안해 분할
    if cfg.n_envs > 1:
        save_freq = max(int(cfg.checkpoint_freq_timesteps // cfg.n_envs), 1)
    else:
        save_freq = int(cfg.checkpoint_freq_timesteps)

    # (dt 비례 ent_coef 스케일) ent_coef 기본값(init/base/boost)은 dt=0.3 에서 튜닝됨.
    # dt 가 작아지면 한 action 이 궤적에 미치는 영향이 ~dt 로 줄어 per-step advantage(std↓ 힘)가
    # 약해지는데, ent_coef(std↑ 힘)는 고정이라 균형이 깨져 policy std 가 발산함.
    # → init/base/boost 를 dt 비례로 축소해 균형 유지. adaptive 토글 로직(boost↔base)은 그대로.
    # 단 선형(dt/0.3)은 과보정(std 0.13 정지)이라 sqrt 사용: norm_reward=True 가 per-step
    # 보상 스케일을 일부 정규화해 실제 필요 축소가 선형보다 완만함. 목표 std ~0.5~1.0.
    # (C2_dt05.log 기준) dt-scale 비활성: C2_dt05.log 는 sqrt/linear dt-scale 없이
    # raw base=0.01 / boost=0.03 (init 0.03) 로 학습됨. entsqrt(×0.408)·v2(×0.167) 와
    # 구분하기 위해 _ent_scale=1.0 으로 고정해 C2_dt05.log 와 동일한 ent_coef 를 재현한다.
    _ENT_REF_DT = 0.3  # ent_coef 기본값이 튜닝된 기준 dt
    _env_dt = float(UAVNav2DEnv().dt)
    _ent_scale = 1.0  # C2_dt05.log 재현: dt-scale 미적용 (이전 sqrt: (_env_dt/_ENT_REF_DT)**0.5)
    if abs(_ent_scale - 1.0) > 1e-9:
        cfg.ent_coef *= _ent_scale
        args.ent_base_coef = float(args.ent_base_coef) * _ent_scale
        args.ent_boost_coef = float(args.ent_boost_coef) * _ent_scale
        print(f"[ent dt-scale] dt={_env_dt:g} (ref {_ENT_REF_DT:g}) → ×{_ent_scale:.3f}: "
              f"init={cfg.ent_coef:.5f} base={args.ent_base_coef:.5f} boost={args.ent_boost_coef:.5f}")

    total_ts = int(args.additional_timesteps) if args.additional_timesteps is not None else cfg.total_timesteps
    cb_list: List[BaseCallback] = [
        EntCoefScheduleCallback(
            total_timesteps=total_ts,
            base_coef=float(args.ent_base_coef),
            boost_coef=float(args.ent_boost_coef),
            low_thr=float(args.ent_low_thr),
            high_thr=float(args.ent_high_thr),
            cooldown=int(args.ent_cooldown),
            ceil_thr=float(args.ent_ceil_thr),
            floor_coef=float(args.ent_floor_coef),
            fixed_ent_coef=args.ent_coef,
            verbose=1,
        ),
        EpisodeStatsCallback(print_every_episodes=500, verbose=1),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=cfg.save_dir,
            name_prefix="ppo_uav2d",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]

    # periodic eval callback (optional)
    if args.eval_freq and args.eval_freq > 0:
        cb_list.append(
            PeriodicEvalCallback(
                cfg=cfg,
                eval_freq=int(args.eval_freq),
                eval_runs=int(args.eval_runs),
                eval_obstacles=[int(x) for x in args.eval_obstacles],
                save_path=cfg.save_dir,
                save_best=bool(args.save_best),
                deterministic=True,
                verbose=1,
            )
        )

    callbacks = CallbackList(cb_list)

    # PPO model
    # 비대칭 net_arch: critic(vf)을 policy(pi)보다 넓게 → explained_variance 개선 목적
    # pi: [64,64] 유지 (actor는 경량 유지), vf: [256,256,128] (value function 표현력 확대)
    policy_kwargs = dict(
        log_std_init=-2.0,
        net_arch=dict(pi=[64, 64], vf=[256, 256, 128]),
    )

    if args.resume_model:
        # SB3 load() adds .zip internally; strip it to avoid "path.zip.zip"
        load_path = args.resume_model.rstrip("/")
        if load_path.lower().endswith(".zip"):
            load_path = load_path[:-4]
        model = PPO.load(load_path, env=env, tensorboard_log=cfg.log_dir, verbose=cfg.verbose)
        print(f"Resumed PPO model from: {args.resume_model}")
        if args.learning_rate is not None:
            new_lr = float(args.learning_rate)
            model.learning_rate = new_lr
            model.lr_schedule = FloatSchedule(new_lr)  # 로드된 lr_schedule 덮어쓰기
            print(f"Learning rate overridden to {new_lr}")
        if args.clip_range is not None:
            model.clip_range = FloatSchedule(float(args.clip_range))
            print(f"clip_range overridden to {args.clip_range}")
        if args.target_kl is not None:
            model.target_kl = float(args.target_kl)
            print(f"target_kl overridden to {args.target_kl}")
        if args.n_epochs is not None:
            model.n_epochs = int(args.n_epochs)
            print(f"n_epochs overridden to {args.n_epochs}")
        if args.ent_coef is not None:
            model.ent_coef = float(args.ent_coef)
            print(f"ent_coef overridden to {args.ent_coef}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            target_kl=args.target_kl if args.target_kl is not None else None,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            verbose=cfg.verbose,
            tensorboard_log=cfg.log_dir,
            seed=cfg.seed,
            policy_kwargs=policy_kwargs,
        )

    print("=== Training start ===")
    total_display = (
        args.additional_timesteps
        if args.additional_timesteps is not None
        else cfg.total_timesteps
    )
    cr_val = args.clip_range if args.clip_range is not None else cfg.clip_range
    extra = []
    if args.target_kl is not None:
        extra.append(f"target_kl={args.target_kl}")
    if args.n_epochs is not None:
        extra.append(f"n_epochs={args.n_epochs}")
    extra_str = ", " + ", ".join(extra) if extra else ""
    print(
        f"n_envs={cfg.n_envs}, total_timesteps={total_display}, "
        f"n_steps={cfg.n_steps}, batch_size={cfg.batch_size}, clip_range={cr_val}{extra_str}"
    )
    if args.eval_freq and args.eval_freq > 0:
        print(f"Periodic eval: freq={args.eval_freq} runs={args.eval_runs} obstacles={args.eval_obstacles} save_best={args.save_best}")
    else:
        print("Periodic eval: OFF")

    reset_ts = True  # num_timesteps 초기화 → 진행률/평가 타이밍 정상
    if args.additional_timesteps is not None:
        print(f"Resume mode: training {total_ts} steps (num_timesteps 초기화)")

    model.learn(
        total_timesteps=total_ts,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_ts,
        log_interval=args.log_interval,
    )

    # save VecNormalize stats and final model
    vecnorm_path = os.path.join(cfg.save_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)
    final_path = os.path.join(cfg.save_dir, "ppo_uav2d_final.zip")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")
    print(f"Saved VecNormalize stats to: {vecnorm_path}")

    _run_final_evaluation(model, cfg, args, vecnorm_path)


if __name__ == "__main__":
    main()
