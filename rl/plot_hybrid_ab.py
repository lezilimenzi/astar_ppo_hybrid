#!/usr/bin/env python3
"""C2_dt05(hybrid, base0.01) vs C2_hybrid_dt05_AB(base0.004+ceiling) 비교 — §40 A+B 효과."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analyze_ent as A

LOGS = {
    "C2_dt05 (base 0.01)":            ("C2_dt05.log", "tab:red"),
    "C2_hybrid_AB (base 0.004+ceil)": ("C2_hybrid_dt05_AB.log", "tab:green"),
}
data = {name: A.parse(path) for name, (path, _) in LOGS.items()}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
roll_panels = [
    ("std",        "policy std  (← §40 핵심)", "std"),
    ("entropy_loss","entropy_loss",            "entropy_loss"),
    ("explained_variance", "explained_variance", "exp_var"),
    ("clip_fraction",      "clip_fraction",      "clip_frac"),
]
for ax, (key, title, ylab) in zip(axes.flat[:4], roll_panels):
    for name, (_, color) in LOGS.items():
        blocks, _, _ = data[name]
        xs = [b["total_timesteps"]/1e6 for b in blocks if key in b]
        ys = [b[key] for b in blocks if key in b]
        ax.plot(xs, ys, color=color, label=name, lw=1.4)
    ax.set_title(title); ax.set_xlabel("timesteps (M)"); ax.set_ylabel(ylab)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

# 패널 4: eval weighted_score
ax = axes.flat[4]
for name, (_, color) in LOGS.items():
    _, et, _ = data[name]
    ax.plot([t/1e6 for t,_,_,_,_ in et], [ws for _,ws,_,_,_ in et],
            color=color, marker="o", ms=4, label=name, lw=1.6)
ax.set_title("eval weighted_score (periodic)"); ax.set_xlabel("timesteps (M)")
ax.set_ylabel("weighted_score"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

# 패널 5: eval collision rate (weighted)
ax = axes.flat[5]
for name, (_, color) in LOGS.items():
    _, et, _ = data[name]
    ax.plot([t/1e6 for t,_,_,_,_ in et], [wc*100 for _,_,wc,_,_ in et],
            color=color, marker="s", ms=4, label=name, lw=1.6)
ax.set_title("eval collision rate (weighted)"); ax.set_xlabel("timesteps (M)")
ax.set_ylabel("collision (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle("HYBRID 10M: C2_dt05 (base 0.01) vs §40 A+B (base 0.004 + entropy ceiling)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.97])
out = "C2_hybrid_AB_vs_dt05.png"
fig.savefig(out, dpi=130)
print(f"saved: {out}")

for name in LOGS:
    blocks, et, _ = data[name]
    fstd = next(b["std"] for b in reversed(blocks) if "std" in b)
    fev  = next(b["explained_variance"] for b in reversed(blocks) if "explained_variance" in b)
    fcf  = next(b["clip_fraction"] for b in reversed(blocks) if "clip_fraction" in b)
    print(f"{name:34s} final std={fstd:.3f} expvar={fev:.3f} clipfr={fcf:.3f} "
          f"final wscore={et[-1][1]:.4f} coll={et[-1][2]*100:.1f}%")
