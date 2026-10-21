"""
kiva_instances.py -- the exact lifelong instances Follower was evaluated on,
extracted from POGEMA itself, for FLOWRRA to run.

    pip install pogema            # 1.4.0 used here
    python kiva_instances.py      # all 6 agent counts x 10 seeds -> kiva_instances/
    python kiva_instances.py --validate --agents 64 --seeds 0,1

THE PROTOCOL (learn-to-follow, experiments/05-warehouse/05-warehouse.yaml):
    map wfi_warehouse (33 x 46), on_target restart (lifelong), 512 steps,
    collision_system soft, num_agents 32 64 96 128 160 192, seeds 0-9.
    Throughput = goals reached by all agents / episode length.

THE MAP IS OUR KIVA. wfi_warehouse is RHCR's kiva map, cell for cell (checked:
same orientation, same 1,278 open cells). Its symbols: '@' the 192 cells where
agents may start (RHCR's 'r' homes), '$' the 480 cells goals are drawn from
(RHCR's 'e' endpoints), '!' other open cells, '#' shelves. So the wfi string is
rebuilt here from kiva.map by symbol substitution, and the substitution is
checked cell by cell against the positions of 'r' and 'e'.

THE GOALS ARE FIXED IN ADVANCE. POGEMA gives every agent its own random stream,
seeded from the instance seed, and draws each next goal uniformly from the '$'
cells. So an agent's whole goal sequence is fixed before the episode starts,
whatever order it reaches them in time -- every method sees identical tasks.
This file calls POGEMA's own functions to produce the sequences (including a
quirk worth copying, not fixing: the "not the same cell again" redraw compares
against the agent's PADDED position, so it almost never fires), and
--validate proves it: it runs POGEMA with its built-in A* agents for the full
episode and checks every goal POGEMA assigned against the sequence.

OUTPUT: kiva_instances/kiva_n<agents>_s<seed>.json with starts and goal
sequences as FLOWRRA node ids (kiva_Nodes.csv: X = column, Y = row from the
bottom), plus the raw POGEMA (row, col) cells.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

AGENTS = [32, 64, 96, 128, 160, 192]
SEEDS = list(range(10))
STEPS = 512


def wfi_from_kiva(kiva_path: str):
    lines = [l.rstrip("\n") for l in open(kiva_path)]
    grid = lines[4:4 + int(lines[0].split(",")[0])]
    sub = {".": "!", "e": "$", "r": "@", "@": "#"}
    wfi = ["".join(sub[c] for c in row) for row in grid]
    for r, (a, b) in enumerate(zip(grid, wfi)):          # the substitution, checked
        for c, (x, y) in enumerate(zip(a, b)):
            assert (x == "r") == (y == "@") and (x == "e") == (y == "$"), (r, c)
    return grid, "\n".join(wfi)


def make_env(wfi: str, agents: int, seed: int):
    from pogema import GridConfig, pogema_v0
    gc = GridConfig(map=wfi, num_agents=agents, seed=seed, on_target="restart",
                    max_episode_steps=STEPS, collision_system="soft", observation_type="POMAPF")
    env = pogema_v0(grid_config=gc)
    # EXACTLY ONE reset. POGEMA shuffles its list of allowed start cells IN
    # PLACE on every reset, so the same seed gives different starts depending on
    # how many resets came before the episode. One reset after construction is
    # the convention here; whether it matches the published runs depends on the
    # evaluation toolbox's own reset sequence (see BENCHMARK.md).
    obs = env.reset(seed=seed)[0]
    return env, obs


def _core(env):
    e = env
    while not hasattr(e, "_generate_new_target") and hasattr(e, "env"):
        e = e.env
    return e


def sequences(wfi: str, agents: int, seed: int, length: int):
    """Starts and goal sequences, produced by POGEMA's own generator."""
    env, _obs = make_env(wfi, agents, seed)
    core = _core(env)
    R = core.grid_config.obs_radius
    starts = [(int(x) - R, int(y) - R) for x, y in core.grid.positions_xy]
    seqs = []
    for i in range(agents):
        g = tuple(int(v) for v in core.grid.finishes_xy[i])
        seq = [(g[0] - R, g[1] - R)]
        for _ in range(length - 1):
            core.grid.positions_xy[i] = g          # it has just reached its goal
            g = tuple(int(v) for v in core._generate_new_target(i))
            seq.append((g[0] - R, g[1] - R))
        seqs.append(seq)
    return starts, seqs


def validate(wfi: str, agents: int, seed: int, length: int) -> str:
    """Run POGEMA's A* agents for the full episode on a FRESH env and check
    every goal it assigned against the pre-generated sequences."""
    from pogema import BatchAStarAgent
    starts, seqs = sequences(wfi, agents, seed, length)
    env, obs = make_env(wfi, agents, seed)
    core = _core(env)
    R = core.grid_config.obs_radius
    assigned = [[(int(x) - R, int(y) - R)] for x, y in core.grid.finishes_xy]
    got_starts = [(int(x) - R, int(y) - R) for x, y in core.grid.positions_xy]
    algo = BatchAStarAgent()
    goals = 0
    for _ in range(STEPS):
        obs, rew, term, trunc, info = env.step(algo.act(obs))
        goals += int(np.sum(rew))
        for i, (x, y) in enumerate(core.grid.finishes_xy):
            g = (int(x) - R, int(y) - R)
            if g != assigned[i][-1]:
                assigned[i].append(g)
        if all(term) or all(trunc):
            break
    # A goal drawn twice in a row (1 in 480) does not visibly change, so the
    # watcher above cannot see the repeat: compare with repeats collapsed.
    def collapse(q):
        return [g for k, g in enumerate(q) if k == 0 or g != q[k - 1]]
    bad = sum(1 for i in range(agents)
              if assigned[i] != collapse(seqs[i])[:len(assigned[i])])
    return (f"n={agents} seed={seed}: starts match {got_starts == starts}; "
            f"{sum(len(a) for a in assigned)} goals assigned over the episode, "
            f"{bad} agents whose sequence differs; A* throughput {goals / STEPS:.3f}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--kiva", default="kiva.map")
    ap.add_argument("--nodes", default="all_maps/kiva_Nodes.csv")
    ap.add_argument("--out", default="kiva_instances")
    ap.add_argument("--agents", default=",".join(map(str, AGENTS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--length", type=int, default=300, help="goals per agent (plenty for 512 steps)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    grid, wfi = wfi_from_kiva(args.kiva)
    rows = len(grid)
    agents = [int(a) for a in args.agents.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.validate:
        for n in agents:
            for s in seeds:
                print(validate(wfi, n, s, args.length))
        return

    nodes = pd.read_csv(args.nodes)
    node_of = {(int(r.X), int(r.Y)): int(r.NodeId) for r in nodes.itertuples() if int(r.Z) == 0}
    to_node = lambda rc: node_of[(rc[1], rows - 1 - rc[0])]      # (row, col) -> NodeId
    os.makedirs(args.out, exist_ok=True)
    for n in agents:
        for s in seeds:
            starts, seqs = sequences(wfi, n, s, args.length)
            assert all(grid[r][c] == "r" for r, c in starts), "a start is not a home cell"
            assert all(grid[r][c] == "e" for q in seqs for r, c in q), "a goal is not an endpoint"
            json.dump({"map": "kiva", "protocol": "learn-to-follow 05-warehouse", "agents": n, "seed": s,
                       "episode_timesteps": STEPS, "starts": [to_node(p) for p in starts],
                       "goals": [[to_node(p) for p in q] for q in seqs],
                       "starts_rc": starts, "goals_rc": seqs},
                      open(os.path.join(args.out, f"kiva_n{n}_s{s}.json"), "w"))
        print(f"{n} agents: {len(seeds)} instances written")


if __name__ == "__main__":
    main()
