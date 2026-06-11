#!/usr/bin/env python3
"""Parse SB3 stdout logs (hybrid C2 vs PPO-only) and plot training-curve comparison."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PeriodicEvalCallback 가중치 (train_ppo_2d_ppo_only.py 와 동일)
EVAL_W = {
    "obs_0": 0.08, "obs_5": 0.08, "obs_10": 0.08,
    "obstacle_density_15": 0.21, "obstacle_density_20": 0.21,
    "narrow_passage": 0.16, "course_l_s": 0.18,
}
# eval 라벨 → weight 키 매핑
LABEL2KEY = {
    "obs=0": "obs_0", "obs=5": "obs_5", "obs=10": "obs_10",
    "obstacle_density_15": "obstacle_density_15",
    "obstacle_density_20": "obstacle_density_20",
    "narrow_passage": "narrow_passage", "course_l_s": "course_l_s",
}

ROW_RE = re.compile(r"^\|\s*([a-zA-Z_/]+)\s*\|\s*([-\d.eE+]+)\s*\|")
EVAL_RE = re.compile(
    r"^\[t=(\d+)\]\s*\[Eval ([a-z0-9_=]+)\]\s*success=\d+/\d+ \(([\d.]+)%\)"
    r"\s*collision=\d+ \(([\d.]+)%\)")


def parse(path):
    """rollout/train metrics(블록 단위) + periodic eval weighted_score 파싱."""
    rollout = {"ts": [], "ep_rew": [], "ep_len": [], "entropy": [],
               "exp_var": [], "std": []}
    cur = {}
    evals = {}   # t -> {key: success_rate}
    coll = {}    # t -> {key: collision_rate}

    def flush(block):
        if "total_timesteps" not in block:
            return
        ts = block["total_timesteps"]
        rollout["ts"].append(ts)
        rollout["ep_rew"].append(block.get("ep_rew_mean"))
        rollout["ep_len"].append(block.get("ep_len_mean"))
        rollout["entropy"].append(block.get("entropy_loss"))
        rollout["exp_var"].append(block.get("explained_variance"))
        rollout["std"].append(block.get("std"))

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = ROW_RE.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    cur[key] = float(val)
                except ValueError:
                    pass
                continue
            # 블록 경계(구분선)에서 flush
            if line.startswith("---") and cur:
                flush(cur)
                cur = {}
                continue
            em = EVAL_RE.match(line)
            if em:
                t = int(em.group(1))
                label = em.group(2)
                rate = float(em.group(3)) / 100.0
                crate = float(em.group(4)) / 100.0
                k = LABEL2KEY.get(label)
                if k:
                    evals.setdefault(t, {})[k] = rate
                    coll.setdefault(t, {})[k] = crate
    if cur:
        flush(cur)

    # weighted success/collision 재구성 (7개 시나리오가 모두 모인 t 만)
    et, ew, ec = [], [], []
    for t in sorted(evals):
        d, c = evals[t], coll.get(t, {})
        if all(k in d for k in EVAL_W) and all(k in c for k in EVAL_W):
            et.append(t)
            ew.append(sum(EVAL_W[k] * d[k] for k in EVAL_W))
            ec.append(sum(EVAL_W[k] * c[k] for k in EVAL_W))
    return rollout, (et, ew, ec)


LOGS = {
    "Hybrid (A*+PPO)": ("C2_dt05.log", "tab:blue"),
    "PPO-only": ("C2_PPOonly_dt05.log", "tab:orange"),
}

data = {name: parse(path) for name, (path, _) in LOGS.items()}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
panels = [
    ("ep_rew",  "ep_rew_mean (train rollout)", "reward"),
    ("ep_len",  "ep_len_mean (steps)",         "steps"),
    ("entropy", "entropy_loss",                "entropy_loss"),
    ("std",     "policy std",                  "std"),
]

# axes 배치: 0~3 = rollout 패널, 4 = collision trend, 5 = weighted_score
for ax, (key, title, ylab) in zip(axes.flat[:4], panels):
    for name, (_, color) in LOGS.items():
        roll, _ = data[name]
        xs = [t / 1e6 for t in roll["ts"]]
        ys = roll[key]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, label=name, lw=1.4)
    ax.set_title(title)
    ax.set_xlabel("timesteps (M)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# 패널 4: eval 충돌률 감소 추이 (weighted, periodic)
ax = axes.flat[4]
for name, (_, color) in LOGS.items():
    _, (et, ew, ec) = data[name]
    if et:
        ax.plot([t / 1e6 for t in et], [c * 100 for c in ec], color=color,
                marker="s", ms=4, label=name, lw=1.6)
ax.set_title("eval collision rate (weighted, periodic)")
ax.set_xlabel("timesteps (M)")
ax.set_ylabel("collision rate (%)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# 패널 5: eval weighted_score
ax = axes.flat[5]
for name, (_, color) in LOGS.items():
    _, (et, ew, ec) = data[name]
    if et:
        ax.plot([t / 1e6 for t in et], ew, color=color, marker="o",
                ms=4, label=name, lw=1.6)
ax.set_title("eval weighted_score (periodic)")
ax.set_xlabel("timesteps (M)")
ax.set_ylabel("weighted_score")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("C2 dt=0.05 10M: Hybrid (A*+PPO) vs PPO-only", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "C2_vs_PPOonly_dt05.png"
fig.savefig(out, dpi=130)
print(f"saved: {out}")

# 요약 출력
for name in LOGS:
    roll, (et, ew, ec) = data[name]
    last_rew = next((y for y in reversed(roll["ep_rew"]) if y is not None), None)
    if et:
        print(f"{name:18s} rollout pts={len(roll['ts'])} "
              f"last ep_rew={last_rew} eval pts={len(et)} "
              f"final weighted={ew[-1]:.4f} "
              f"collision {ec[0]*100:.1f}%→{ec[-1]*100:.1f}%")
    else:
        print(f"{name}: no eval")
