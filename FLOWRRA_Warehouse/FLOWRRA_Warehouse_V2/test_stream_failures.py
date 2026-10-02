"""
test_stream_failures.py -- the stream runs to the end with failures on.

Failures (errors.enabled) were off from the first stream run, so the rescue
path had never run in stream mode. When it did, it crashed: a rescuer carrying a
second, inherited order moved on to it after its first delivery, but that
order had left the pool meanwhile (delivered and retired -- in the stream an
order leaves the pool by more routes than being claimed), and the step raised
KeyError on goal_pool. Both places that build a rescuer's queue now keep an
order only while it still exists -- and when nothing is left to carry (the
orphans were delivered meanwhile, the rescuer's own goal a dock), the rescuer
carries on with what it was doing instead of raising IndexError. Seeded:
failure timing is random, and an unseeded version passed once by luck.
Found 2026-09-29, before cold_run26.
"""
import contextlib
import io
import random
import sys
import types

import numpy as np

import drive_shortest_path as drv

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def test_stream_with_failures_runs_to_the_end():
    RULES = {"conflict.path_warnings": True, "conflict.directional_braking": True,
             "conflict.corridor_entry": True, "conflict.priority": True,
             "conflict.node_aligned_moves": True, "conflict.yield_to_stopped": True}
    total_errors = total_rescues = 0
    for label, sets in (("rules off", {}), ("rules on", RULES)):
        for seed in (0, 1, 2, 3, 4, 5):
            random.seed(seed); np.random.seed(seed)      # failure timing is random
            drv.apply_config(dict(sets, **{"stream.enabled": True, "errors.enabled": True,
                                           "errors.prob_per_step": 0.05}))
            from core_warehouse import FLOWRRA
            with contextlib.redirect_stdout(io.StringIO()):
                G, grid, miss, kw = drv.synthetic_instance(types.SimpleNamespace(agents=24), seed)
                env = FLOWRRA(G, grid, miss, **kw)
            env.gnn = drv.ShortestPathDriver(env, "never", 0.5, seed)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(300):
                    env.step(episode_step=1, total_episodes=1)
            st = env.get_error_statistics()
            total_errors += st.get("errors_injected", 0)
            total_rescues += st.get("handovers_completed", 0)
            print(f"      {label}, seed {seed}: 300 steps, failures {st.get('errors_injected', 0)}, "
                  f"rescued orders {st.get('handovers_completed', 0)}, deliveries {st.get('stream_deliveries')}")
    drv.apply_config({})
    check("failures_were_injected", total_errors > 0, True)
    check("orders_were_rescued", total_rescues > 0, True)


if __name__ == "__main__":
    test_stream_with_failures_runs_to_the_end()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
