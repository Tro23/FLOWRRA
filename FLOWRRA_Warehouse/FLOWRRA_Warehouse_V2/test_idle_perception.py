"""
test_idle_perception.py

Idle-perception caching is a BEHAVIOUR change, not a compute saving, because
_build_adjacency() includes immobile fleets: the GAT attends over them, so an
ACTIVE fleet's embedding depends on its idle neighbours' 291 features. Reusing a
stale vector for an idle fleet therefore moves the active fleets' Q-values.

That makes three things worth asserting, and the first matters most:

  1. idle_mode="full" must be EXACTLY what the code did before this existed --
     byte-identical vectors, zero reuse. If the default drifts, every previous
     measurement becomes incomparable.

  2. "cached" must respect its refresh interval, and must DROP the cache the
     moment a fleet becomes mobile again -- otherwise a fleet that starts moving
     is served a vector from a position it has left.

  3. "skip" must reuse indefinitely while idle, which is the maximum-staleness
     arm and really an ablation.

No torch here: the helper is exercised directly against a stub environment, so
this runs in a second.
"""

import sys
import types

import numpy as np

def _load_shipped_methods():
    """
    Pull _perceive and _apply_lesion out of core_warehouse.py's SOURCE and
    compile them in isolation.

    Importing core_warehouse would drag in agent_warehouse and therefore torch,
    which turns a one-second test into an environment dependency. Parsing the
    file instead means this tests the code that actually ships -- not a copy of
    it that can silently drift, which is the whole reason for not pasting the
    logic in here.
    """
    import ast
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "core_warehouse.py")
    tree = ast.parse(open(path).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "FLOWRRA")
    wanted = {"_perceive", "_apply_lesion"}
    fns = [n for n in cls.body
           if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {f.name for f in fns}
    if missing:
        raise SystemExit(f"core_warehouse.FLOWRRA is missing {missing}")

    mod = ast.Module(body=fns, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"np": np}
    exec(compile(mod, "<core_warehouse:FLOWRRA>", "exec"), ns)
    return {name: ns[name] for name in wanted}


_SHIPPED = _load_shipped_methods()

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


class _Node:
    def __init__(self, i):
        self.id = f"f{i}"
        self.current_pos = np.array([float(i), 0.0, 0.0])
        self.goal_pos = np.array([9.0, 0.0, 0.0])
        self._sv = np.full(60, float(i), dtype=np.float32)

    def get_state_vector(self, nodes):
        # Changes every call, so a reused vector is detectable.
        self._sv = self._sv + 1.0
        return self._sv.copy()


class _Density:
    def __init__(self):
        self.calls = 0

    def get_local_affordance(self, *a, **k):
        self.calls += 1
        return np.zeros(231, dtype=np.float32)

    def action_entropy(self, affordance):
        # _perceive computes the local entropy scalar from the affordance and
        # writes it back onto the node BEFORE building the base vector, so the
        # stub has to answer this too.
        return 1.0


class _Env:
    """Minimal stand-in exposing exactly what _perceive touches."""

    def __init__(self, mode, refresh=10):
        self.nodes = [_Node(i) for i in range(6)]
        self.density = _Density()
        self.immobile_nodes = {"f0", "f1", "f2"}
        self.stopped_nodes = set()
        self.frozen_obstacle_severity = 1.0
        self.near_goal_radius = 3.0
        self.frozen_near_goal_radius = 3.0
        self.stopped_obstacle_severity = 1.4
        self.step_count = 0
        self._throughput_t = 1.0
        # _perceive passes these straight through to get_local_affordance.
        # Every time _perceive gains a dependency this stub has to follow -- the
        # cost of testing a real method against a fake environment, and still
        # cheaper than requiring torch to run a one-second test.
        self.static_obstacles = set()
        self.static_obstacle_severity = 3.0
        self._idle_mode = mode
        self._idle_refresh_every = refresh
        self._idle_perception_reused = 0
        self._idle_perception_computed = 0
        # _apply_lesion checks _lesion_names first (it resolves slices lazily,
        # because resolving them in __init__ read self.nodes before fleets were
        # spawned). Both attributes are needed for the stub to stand in.
        self._lesion_names = []
        self._lesion_slices = None

    _perceive = _SHIPPED["_perceive"]
    _apply_lesion = _SHIPPED["_apply_lesion"]


def test_full_mode_never_reuses():
    env = _Env("full")
    for step in range(5):
        env.step_count = step
        for n in env.nodes:
            env._perceive(n)
    check("full_reused", env._idle_perception_reused, 0)
    check("full_computed", env._idle_perception_computed, 30)
    check("full_affordance_calls", env.density.calls, 30)
    check("full_leaves_no_memo",
          any(hasattr(n, "_perception_memo") for n in env.nodes), False)


def test_cached_respects_interval():
    env = _Env("cached", refresh=3)
    for step in range(6):
        env.step_count = step
        for n in env.nodes:
            env._perceive(n)
    # 3 idle fleets, refreshed at steps 0 and 3 -> 2 computes each = 6;
    # 3 mobile fleets computed every step = 18. Total 24 computes, 12 reuses.
    check("cached_computed", env._idle_perception_computed, 24)
    check("cached_reused", env._idle_perception_reused, 12)
    check("cached_mobile_never_memoised",
          any(hasattr(n, "_perception_memo") for n in env.nodes[3:]), False)


def test_skip_reuses_indefinitely():
    env = _Env("skip")
    for step in range(8):
        env.step_count = step
        for n in env.nodes:
            env._perceive(n)
    # 3 idle fleets computed once each, then reused 7 times each.
    check("skip_computed", env._idle_perception_computed, 3 + 8 * 3)
    check("skip_reused", env._idle_perception_reused, 21)


def test_cache_dropped_when_fleet_becomes_mobile():
    """
    The dangerous case: a fleet is served a vector from a position it has left.
    """
    env = _Env("skip")
    env.step_count = 0
    for n in env.nodes:
        env._perceive(n)
    check("memo_present_while_idle", hasattr(env.nodes[0], "_perception_memo"), True)

    stale = env.nodes[0]._perception_memo[1].copy()
    env.immobile_nodes = set()          # f0 starts moving
    env.step_count = 1
    fresh = env._perceive(env.nodes[0])

    check("memo_dropped_on_mobile", hasattr(env.nodes[0], "_perception_memo"), False)
    check("fresh_vector_not_stale", bool(np.array_equal(fresh, stale)), False)


def test_reused_vector_is_the_same_object_content():
    """A reuse must return the cached CONTENT, not a fresh computation."""
    env = _Env("skip")
    env.step_count = 0
    first = env._perceive(env.nodes[0]).copy()
    env.step_count = 1
    second = env._perceive(env.nodes[0])
    check("reuse_returns_cached", bool(np.array_equal(first, second)), True)
    # And a mobile fleet must NOT be cached: its vector changes every call.
    env.step_count = 2
    a = env._perceive(env.nodes[5]).copy()
    b = env._perceive(env.nodes[5])
    check("mobile_not_cached", bool(np.array_equal(a, b)), False)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))