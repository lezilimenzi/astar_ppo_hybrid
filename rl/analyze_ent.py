#!/usr/bin/env python3
"""C2_dt05 (unscaled ent) vs C2_dt05_entsqrt (sqrt-scaled ent) 정밀 비교."""
import re

EVAL_W = {"obs_0":0.08,"obs_5":0.08,"obs_10":0.08,"obstacle_density_15":0.21,
          "obstacle_density_20":0.21,"narrow_passage":0.16,"course_l_s":0.18}
LABEL2KEY = {"obs=0":"obs_0","obs=5":"obs_5","obs=10":"obs_10",
             "obstacle_density_15":"obstacle_density_15",
             "obstacle_density_20":"obstacle_density_20",
             "narrow_passage":"narrow_passage","course_l_s":"course_l_s"}
ROW = re.compile(r"^\|\s*([a-zA-Z_/]+)\s*\|\s*([-\d.eE+]+)\s*\|")
EVAL = re.compile(r"^\[t=(\d+)\]\s*\[Eval ([a-z0-9_=]+)\]\s*success=\d+/\d+ \(([\d.]+)%\)\s*collision=\d+ \(([\d.]+)%\)")
TOG = re.compile(r"AdaptiveEnt\] entropy_loss=([-\d.]+) ([<>]) ([-\d.]+) . ent_coef (.) ([\d.eE-]+)")

def parse(path):
    blocks=[]; cur={}; evals={}; coll={}; toggles=[]; last_ts=0
    for line in open(path, errors="replace"):
        m=ROW.match(line)
        if m:
            try: cur[m.group(1)]=float(m.group(2))
            except: pass
            continue
        if line.startswith("---") and cur:
            if "total_timesteps" in cur:
                blocks.append(dict(cur)); last_ts=cur["total_timesteps"]
            cur={}; continue
        em=EVAL.match(line)
        if em:
            t=int(em.group(1)); k=LABEL2KEY.get(em.group(2))
            if k:
                evals.setdefault(t,{})[k]=float(em.group(3))/100
                coll.setdefault(t,{})[k]=float(em.group(4))/100
            continue
        tg=TOG.search(line)
        if tg:
            toggles.append((last_ts, float(tg.group(1)), tg.group(4), float(tg.group(5))))
    if cur and "total_timesteps" in cur: blocks.append(cur)
    et=[]
    for t in sorted(evals):
        d,c=evals[t],coll.get(t,{})
        if all(k in d for k in EVAL_W) and all(k in c for k in EVAL_W):
            et.append((t,
                       sum(EVAL_W[k]*d[k] for k in EVAL_W),
                       sum(EVAL_W[k]*c[k] for k in EVAL_W),
                       dict(d), dict(c)))
    return blocks, et, toggles

def at(blocks, ts, key):
    """ts 근처 블록의 key 값."""
    best=min(blocks, key=lambda b: abs(b["total_timesteps"]-ts))
    return best.get(key)

for name,path in [("UNSCALED  (C2_dt05: base0.01/boost0.03)","C2_dt05.log"),
                  ("SQRT×0.408(entsqrt: base0.004/boost0.012)","C2_dt05_entsqrt.log")]:
    blocks, et, toggles = parse(path)
    print("="*78); print(name); print("="*78)
    print("AdaptiveEnt 토글:")
    for ts,el,arrow,val in toggles:
        print(f"   ~{ts/1e6:.2f}M  entropy_loss={el:+.3f}  ent_coef{arrow} {val:g}")
    print(f"\n{'t(M)':>5} {'ent_loss':>9} {'std':>6} {'ep_rew':>8} {'ep_len':>7} {'kl':>7} {'clipfr':>7} {'expvar':>7}")
    for ts in [0.16e6,1e6,2e6,3e6,5e6,7e6,9e6,10e6]:
        print(f"{ts/1e6:5.1f} {at(blocks,ts,'entropy_loss'):9.3f} {at(blocks,ts,'std'):6.3f} "
              f"{at(blocks,ts,'ep_rew_mean'):8.0f} {at(blocks,ts,'ep_len_mean'):7.0f} "
              f"{at(blocks,ts,'approx_kl'):7.4f} {at(blocks,ts,'clip_fraction'):7.3f} "
              f"{at(blocks,ts,'explained_variance'):7.3f}")
    print(f"\n{'t(M)':>5} {'wscore':>7} {'coll%':>6}  per-scenario success")
    for t,ws,wc,d,c in et:
        key=" ".join(f"{k.split('_')[-1] if 'obs' in k else k[:4]}={d[k]*100:.0f}" for k in ["obs_10","obstacle_density_20","narrow_passage","course_l_s"])
        print(f"{t/1e6:5.1f} {ws:7.4f} {wc*100:6.1f}  {key}")
    print()
