"""
test_profile_step.py

Runs profile_step.main() END TO END against stub modules.

WHY THIS EXISTS. profile_step.py has now shipped two crashes that fire only
AFTER map discovery and a 120k-node BFS precompute -- about ten minutes in:

  1. `import os` missing, used by the CSV setup.
  2. `block` referenced in the CSV metadata before it was assigned, because a
     patch inserted the CSV block above the line that defines it.

Neither is catchable by ast.parse (an undefined name is legal syntax) and
neither is caught by pyflakes (`--help` returns before reaching the code, so
even running the script proves nothing). Both are trivially caught by ACTUALLY
EXECUTING the function, which is what this does.

The stubs are deliberately dumb -- they are not a simulation. They exist only so
every line of main() runs: argparse, agent construction, the warmup loop, the
per-block driver read, the CSV writer, the profile, and the summary. If a name
is undefined or bound too late, this fails in under a second instead of ten
minutes.
"""

import os
import sys
import tempfile
import types

import numpy as np


# ---------------------------------------------------------------- stubs
class _StubNode:
    def __init__(self, i):
        self.id = f"f{i}"
        self.speed = 0.5 if i % 3 else 0.05      # mixed, so the histogram has buckets
        self.max_vision_range = 10
        self.current_pos = np.zeros(3)
        self.goal_pos = np.ones(3)

    def get_state_vector(self, nodes):
        return np.zeros(60, dtype=np.float32)

    def state_layout(self):
        return {"ray_distances": (6, 12), "goal_gradient": (48, 54),
                "rays_all": (6, 48), "_base_len": (0, 60)}


class _StubDensity:
    output_dim = 231

    def __init__(self):
        self.stamps_this_step = 1234
        self.stamp_early_outs = 567
        self.collapse_memory = {(1, 2, 0): 0.5}
        self._projection_fallbacks = 0
        self._kernel_manhattan_fallbacks = 0

    def describe(self):
        return "[StubDensity]"

    def get_local_affordance(self, *a, **k):
        return np.zeros(231, dtype=np.float32)


class _StubLoop:
    total_collisions = 3
    phantom_pairs_rejected = 7


class _StubEnv:
    def __init__(self, *a, **k):
        self.nodes = [_StubNode(i) for i in range(12)]
        self.density = _StubDensity()
        self.loop = _StubLoop()
        self.frozen_nodes = {"f0"}
        self.immobile_nodes = {"f0"}
        self.gnn = None
        self._steps = 0

    def get_active_nodes(self):
        return self.nodes[1:]

    def step(self, **k):
        self._steps += 1

    def is_episode_over(self):
        return False


class _StubAgent:
    def __init__(self, **k):
        self.loaded = None

    def load(self, path):
        self.loaded = path

    def epsilon_gaussian(self, *a, **k):
        return 0.0


class _StubCache:
    def __init__(self, *a, **k):
        pass

    def discover(self, only=None):
        return ["stub_map_a", "stub_map_b"]


def _stub_sample_instance(cache, name, k, rng):
    return {"G": None, "pos_dict": {i: 1 for i in range(500)},
            "missions": [], "gdm": {}, "goal_pool": ["g1"], "scen": "seed0"}


def install_stubs():
    """Put fake modules in sys.modules so main()'s deferred imports resolve."""
    torch = types.ModuleType("torch")
    torch.manual_seed = lambda *a, **k: None
    sys.modules["torch"] = torch

    runner = types.ModuleType("main_runner_warehouse")
    runner.MapCache = _StubCache
    runner.sample_instance = _stub_sample_instance
    sys.modules["main_runner_warehouse"] = runner

    core = types.ModuleType("core_warehouse")
    core.FLOWRRA = _StubEnv
    sys.modules["core_warehouse"] = core

    agent = types.ModuleType("agent_warehouse")
    agent.GNNAgent = _StubAgent
    sys.modules["agent_warehouse"] = agent


# ---------------------------------------------------------------- run
def main():
    install_stubs()
    import profile_step

    out = tempfile.mkdtemp(prefix="profstub_")
    sys.argv = [
        "profile_step.py",
        "--agents", "12",
        "--warmup", "6",
        "--steps", "2",
        "--block", "2",
        "--out", out,
        "--tag", "stubtest",
        "--rows", "5",
    ]

    print("=== executing profile_step.main() against stubs ===\n")
    try:
        profile_step.main()
    except SystemExit as e:
        print(f"\nFAIL: SystemExit({e.code})")
        return 1
    except Exception as e:
        print(f"\nFAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n=== checking artifacts ===")
    files = sorted(os.listdir(out))
    csvs = [f for f in files if f.startswith("warmup_")]
    fails = []
    if not csvs:
        fails.append("no warmup CSV written")
    if "runs_index.csv" not in files:
        fails.append("no runs_index.csv written")
    if "runs_summary.csv" not in files:
        fails.append("no runs_summary.csv written")

    if csvs:
        path = os.path.join(out, csvs[0])
        lines = open(path).read().splitlines()
        meta = [l for l in lines if l.startswith("#")]
        rows = [l for l in lines if l and not l.startswith("#")]
        print(f"  {csvs[0]}: {len(meta)} metadata lines, {len(rows)-1} data rows")
        if len(meta) < 5:
            fails.append("metadata header too short")
        if len(rows) < 2:
            fails.append("no data rows")
        header = rows[0].split(",")
        for col in ("ms_per_step", "stamps_per_obs", "mean_speed",
                    "min_speed", "mem_cells", "active"):
            if col not in header:
                fails.append(f"column {col} missing")
        try:
            import pandas as pd
            df = pd.read_csv(path, comment="#")
            print(f"  pandas reads it: {len(df)} rows x {len(df.columns)} cols")
        except ImportError:
            print("  (pandas not installed, skipped)")

    # runs_index.csv must hold EVERY BLOCK, and must ACCUMULATE across runs.
    if "runs_index.csv" in files:
        import csv as _csv
        with open(os.path.join(out, "runs_index.csv")) as fh:
            rows_a = list(_csv.DictReader(fh))
        print(f"  runs_index.csv after run 1: {len(rows_a)} block rows")

        sys.argv[sys.argv.index("--tag") + 1] = "stubtest2"
        profile_step.main()

        with open(os.path.join(out, "runs_index.csv")) as fh:
            rows_b = list(_csv.DictReader(fh))
        print(f"  runs_index.csv after run 2: {len(rows_b)} block rows")
        if len(rows_b) != 2 * len(rows_a):
            fails.append(f"index did not accumulate: {len(rows_a)} -> {len(rows_b)}")
        runs = {r["run_id"] for r in rows_b}
        tags = {r["tag"] for r in rows_b}
        print(f"  distinct run_ids={len(runs)} tags={sorted(tags)}")
        if len(tags) != 2:
            fails.append(f"runs not distinguishable by tag: {tags}")
        if len(runs) != 2:
            fails.append(f"run_ids collided: {runs}")
        for col in ("ms_per_step", "active", "frozen", "collisions",
                    "stamps_per_obs", "mean_speed", "mem_cells"):
            if col not in rows_b[0]:
                fails.append(f"index missing column {col}")

    print()
    if fails:
        print("FAILURES:\n  " + "\n  ".join(fails))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())