#!/usr/bin/env python3
"""
Parse PPO training stdout logs and plot training-progress curves:
  - episode reward (ep_rew_mean)
  - reached rate (incremental, per logging window)
  - collision rate (incremental, per logging window)

Two figures are produced:
  hybrid    = C2_seed42.log (0->4M, scratch) ++ C2_10M.log (4M->10M, resumed)
  ppo_only  = C2_PPOonly_10M.log (0->10M, scratch)
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RL = "/home/akswosla/PX4-Autopilot/rl"

re_episodes = re.compile(
    r"\[episodes=(\d+)\]\s+reached=(\d+).*?collision=(\d+)")
re_ts = re.compile(r"\|\s*total_timesteps\s*\|\s*([\d.eE+]+)\s*\|")
re_rew = re.compile(r"\|\s*ep_rew_mean\s*\|\s*([\d.eE+-]+)\s*\|")


def parse_log(path):
    """Return dict with reward curve and incremental reached/collision curve.

    reward curve : (ts_reward, ep_rew_mean) from rollout blocks
    rate curve   : (ep_no, reached_inc, collision_inc) from [episodes=] lines
    Also collects anchors (ep_no -> ts) so episode counts can be mapped to
    timesteps via interpolation.
    """
    ep_no, reached_cum, coll_cum = [], [], []
    ts_reward, ep_rew = [], []
    anchors_ep, anchors_ts = [], []   # (episode_count_seen_so_far -> total_timesteps)
    last_ep = 0
    pending_ts = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re_episodes.search(line)
            if m:
                last_ep = int(m.group(1))
                ep_no.append(last_ep)
                reached_cum.append(int(m.group(2)))
                coll_cum.append(int(m.group(3)))
                continue
            m = re_ts.search(line)
            if m:
                pending_ts = float(m.group(1))
                continue
            m = re_rew.search(line)
            if m and pending_ts is not None:
                ts_reward.append(pending_ts)
                ep_rew.append(float(m.group(1)))
                # anchor: episodes seen so far <-> this rollout's timesteps
                anchors_ep.append(last_ep)
                anchors_ts.append(pending_ts)
                pending_ts = None

    ep_no = np.array(ep_no, float)
    reached_cum = np.array(reached_cum, float)
    coll_cum = np.array(coll_cum, float)

    # incremental rates between consecutive logging markers
    d_ep = np.diff(ep_no, prepend=0.0)
    d_reached = np.diff(reached_cum, prepend=0.0)
    d_coll = np.diff(coll_cum, prepend=0.0)
    d_ep[d_ep <= 0] = np.nan
    reached_inc = d_reached / d_ep
    coll_inc = d_coll / d_ep

    return dict(
        ep_no=ep_no, reached_inc=reached_inc, coll_inc=coll_inc,
        ts_reward=np.array(ts_reward, float), ep_rew=np.array(ep_rew, float),
        anchors_ep=np.array(anchors_ep, float), anchors_ts=np.array(anchors_ts, float),
    )


def ep_to_ts(ep_no, anchors_ep, anchors_ts):
    """Map episode counts to timesteps by linear interpolation on anchors."""
    order = np.argsort(anchors_ep)
    xa, ya = anchors_ep[order], anchors_ts[order]
    return np.interp(ep_no, xa, ya)


def smooth(y, k=5):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    kern = np.ones(k) / k
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, kern, mode="valid")[: len(y)]


def build_hybrid():
    a = parse_log(f"{RL}/logs_ppo_2d/ablation/C2_seed42.log")   # 0..4M
    b = parse_log(f"{RL}/C2_10M.log")                            # resumed -> 4M..10M
    OFF_TS = 4_000_000.0
    off_ep = a["ep_no"][-1] if len(a["ep_no"]) else 0.0

    ts_rate_a = ep_to_ts(a["ep_no"], a["anchors_ep"], a["anchors_ts"])
    ts_rate_b = ep_to_ts(b["ep_no"], b["anchors_ep"], b["anchors_ts"]) + OFF_TS

    return dict(
        ts_reward=np.concatenate([a["ts_reward"], b["ts_reward"] + OFF_TS]),
        ep_rew=np.concatenate([a["ep_rew"], b["ep_rew"]]),
        ts_rate=np.concatenate([ts_rate_a, ts_rate_b]),
        reached=np.concatenate([a["reached_inc"], b["reached_inc"]]),
        coll=np.concatenate([a["coll_inc"], b["coll_inc"]]),
        split=OFF_TS,
    )


def build_single(path):
    d = parse_log(path)
    return dict(
        ts_reward=d["ts_reward"], ep_rew=d["ep_rew"],
        ts_rate=ep_to_ts(d["ep_no"], d["anchors_ep"], d["anchors_ts"]),
        reached=d["reached_inc"], coll=d["coll_inc"], split=None,
    )


def make_figure(data, title, color, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    xM = lambda a: np.asarray(a) / 1e6

    # 1) episode reward
    ax = axes[0]
    ax.plot(xM(data["ts_reward"]), data["ep_rew"], color=color, lw=1.6)
    ax.set_ylabel("Episode reward\n(ep_rew_mean)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # 2) reached rate (incremental)
    ax = axes[1]
    ax.plot(xM(data["ts_rate"]), data["reached"] * 100, color="0.75", lw=0.8, label="raw")
    ax.plot(xM(data["ts_rate"]), smooth(data["reached"]) * 100, color=color, lw=1.8, label="smoothed")
    ax.set_ylabel("Reached rate (%)\n(per window)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # 3) collision rate (incremental)
    ax = axes[2]
    ax.plot(xM(data["ts_rate"]), data["coll"] * 100, color="0.75", lw=0.8, label="raw")
    ax.plot(xM(data["ts_rate"]), smooth(data["coll"]) * 100, color=color, lw=1.8, label="smoothed")
    ax.set_ylabel("Collision rate (%)\n(per window)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Total timesteps (millions)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    if data["split"] is not None:
        for ax in axes:
            ax.axvline(data["split"] / 1e6, color="red", ls="--", lw=1, alpha=0.6)
        axes[0].text(data["split"] / 1e6, axes[0].get_ylim()[1],
                     " resume (4M)", color="red", fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    print("saved:", outpath)


if __name__ == "__main__":
    make_figure(build_hybrid(), "C2 Hybrid (PPO + BFS/A*) — 10M training",
                "#1f77b4", f"{RL}/compare_hybrid_training.png")
    make_figure(build_single(f"{RL}/C2_PPOonly_10M.log"),
                "C2 PPO-only baseline — 10M training",
                "#d62728", f"{RL}/compare_ppoonly_training.png")
