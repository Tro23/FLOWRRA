"""
radius_study.py -- which local_radius should perception use? (CONFLICT_DESIGN.md, component 5)

    python radius_study.py                      # 50_, 60 fleets, radii 3..8, seeds 0 (train) and 1 (test)
    python radius_study.py 3,5,8 20             # chosen radii, collision within 20 steps

Offline, no training. Runs real 50_ traffic (shortest-path driver, every conflict
rule on) and, for each candidate radius, builds a density field identical to
core's except for its radius. For every sampled fleet it records what that field
SHOWS -- visible repulsion, slow memory and open cells, summed by hop ring -- and
whether the fleet COLLIDES within the next 5 steps. A logistic model per radius,
fitted on seed 0 and scored on seed 1 (AUC), says how much of the imminent
danger each radius can see. The model may ignore far rings, so a bigger radius
is never penalised for carrying more; if it scores no higher, the extra cells
carry nothing predictive. Also reports cost per field and input size.

A PROXY. The real judge is a training run (CONFLICT_DESIGN.md: wait share of
conflict choices and deliveries against cold_run24), and any radius other than 5
changes the network's input shape -- a cold start, no resuming checkpoints.
"""
import contextlib, io, random, sys, time, types
from collections import deque
import numpy as np
import drive_shortest_path as drv
from config_warehouse import CONFIG
from density_warehouse import WarehouseDensityField

RADII = [int(r) for r in (sys.argv[1].split(",") if len(sys.argv) > 1 else "3,4,5,6,7,8".split(","))]
ALL = {"conflict.path_warnings": True, "conflict.directional_braking": True, "conflict.corridor_entry": True,
       "conflict.priority": True, "conflict.node_aligned_moves": True}
# argv: radii (comma list), horizon K in steps
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EVERY, STEPS = 3, 300


def field_like(env, L):
    d = CONFIG["density"]
    f = WarehouseDensityField(
        max_vision_range=2 * L, falloff_radius=d["falloff_radius"], peer_severity=d["peer_severity"],
        projection_steps=d["projection_steps"], projection_falloff=d["projection_falloff"],
        memory_decay_factor=d["memory_decay_factor"], memory_floor=d["memory_floor"],
        memory_cap=d["memory_cap"], projection_max_branches=d.get("projection_max_branches", 6),
        project_stationary=d.get("project_stationary", True), kernel_metric=d.get("kernel_metric", "graph"),
        output_mode=d.get("output_mode", "affordance"), slow_channel=d.get("slow_channel", False),
        paths_channels=d.get("paths_channels", False), entropy_fix=d.get("entropy_fix", False),
        slow_decay_factor=d.get("slow_decay_factor", 0.987), slow_severity_scale=d.get("slow_severity_scale", 1.0),
        projection_mode=d.get("projection_mode", "intended"), grid_pos_dict=env.grid_pos_dict, graph=env.G)
    f.collapse_memory = env.density.collapse_memory      # the live memories, shared
    if hasattr(env.density, "slow_memory"):
        f.slow_memory = env.density.slow_memory
    return f


def run(seed):
    drv.apply_config(ALL); random.seed(seed); np.random.seed(seed)
    from core_warehouse import FLOWRRA
    args = types.SimpleNamespace(maps_dir="all_maps", scens_dir="all_scens_v4", map="50_5_5_10_5_2",
                                 agents=60, order_window="final")
    with contextlib.redirect_stdout(io.StringIO()):
        G, pos, miss, kw = drv.real_instance(args, seed); env = FLOWRRA(G, pos, miss, **kw)
    env.gnn = drv.ShortestPathDriver(env, "never", 0.5, seed)
    fields = {L: field_like(env, L) for L in RADII}
    offs = {L: np.argwhere(fields[L]._diamond_mask) - L for L in RADII}
    hop_cache, rows, dead, cost = {}, [], [], {L: [] for L in RADII}
    for t in range(STEPS):
        with contextlib.redirect_stdout(io.StringIO()):
            env.step(episode_step=1, total_episodes=1)
        dead.append(set(env._step_deadlocked))
        if t % EVERY:
            continue
        for n in env.get_active_nodes():
            c = tuple(int(v) for v in np.round(n.current_pos)); nid = env.grid_pos_dict.get(c)
            if nid is None:
                continue
            if nid not in hop_cache:
                d, q = {nid: 0}, deque([nid])
                while q:
                    u = q.popleft()
                    if d[u] >= max(RADII):
                        continue
                    for w in env.G.neighbors(u):
                        if w not in d:
                            d[w] = d[u] + 1; q.append(w)
                hop_cache[nid] = d
            hd = hop_cache[nid]
            feats = {}
            for L in RADII:
                t0 = time.perf_counter()
                v = fields[L].get_local_affordance(n.current_pos, env.nodes, env.frozen_nodes,
                                                   own_goal_pos=n.goal_pos, own_id=n.id)
                cost[L].append(time.perf_counter() - t0)
                k = len(offs[L]); ch = len(v) // k
                vol = v.reshape(ch, k)
                ring = np.array([hd.get(env.grid_pos_dict.get((c[0] + o[0], c[1] + o[1], c[2] + o[2])), 99)
                                 for o in offs[L]])
                vis = vol[0] > 0
                f = []
                for d_ in range(L + 1):
                    sel = vis & (ring == d_)
                    f += [float(sel.sum())] + [float(vol[j][sel].sum()) for j in range(1, ch)]
                feats[L] = (np.array(f), len(v))
            rows.append((t, n.id, feats))
    out = []
    for t, fid, feats in rows:
        y = any(fid in dead[s] for s in range(t + 1, min(len(dead), t + 1 + K)))
        out.append((feats, y))
    return out, {L: 1000 * np.mean(c) for L, c in cost.items()}


def auc(s, y):
    y = np.asarray(y, bool); r = np.argsort(np.argsort(s)) + 1
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def fit(X, y, lam=1e-2, it=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0; yy = y.astype(float)
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - yy) / len(yy) + lam * w); b -= lr * float(np.mean(p - yy))
    return lambda X2: ((X2 - mu) / sd) @ w + b


train, c0 = run(0)
test, c1 = run(1)
print(f"samples: train {len(train)} (collide within {K}: {sum(y for _, y in train)}), "
      f"test {len(test)} ({sum(y for _, y in test)})")
print(f"{'radius':>7}{'cells':>7}{'dims':>7}{'ms/field':>10}{'AUC test':>10}{'AUC train':>11}")
for L in RADII:
    Xtr = np.array([f[L][0] for f, _ in train]); ytr = np.array([y for _, y in train])
    Xte = np.array([f[L][0] for f, _ in test]); yte = np.array([y for _, y in test])
    m = fit(Xtr, ytr)
    dims = train[0][0][L][1]
    cells = (2 * L + 1) * (2 * L * L + 2 * L + 3) // 3     # the diamond of radius L
    print(f"{L:>7}{cells:>7}"
          f"{dims:>7}{(c0[L] + c1[L]) / 2:>10.2f}{auc(m(Xte), yte):>10.3f}{auc(m(Xtr), ytr):>11.3f}")
