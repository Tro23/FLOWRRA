"""
test_state_layout.py

The lesion harness names state blocks by index range. If state_layout() ever
drifts from what get_state_vector() actually concatenates, the harness reports
confident numbers about whichever dimensions happen to sit at those offsets --
the exact failure mode that made hardcoding the offsets unacceptable.

So: verify the layout against the real vector, dimension by dimension, by
perturbing one input at a time and checking that only the named slice moves.
"""

import numpy as np
import networkx as nx
from collections import deque

from node_warehouse import FleetNode
from density_warehouse import WarehouseDensityField

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def build():
    G = nx.Graph()
    grid = {}
    for y in (0, 2):
        for x in range(6):
            grid[(x, y, 0)] = f"n_{x}_{y}"
            G.add_node(f"n_{x}_{y}")
        for x in range(5):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, 5):
        grid[(x, 1, 0)] = f"n_{x}_1"
        G.add_node(f"n_{x}_1")
        G.add_edge(f"n_{x}_0", f"n_{x}_1")
        G.add_edge(f"n_{x}_1", f"n_{x}_2")

    # build_spatial_indices, NOT a hand-built dict: is_structurally_valid now
    # bisects the aisle line and requires it sorted.
    from node_warehouse import build_spatial_indices
    aisle, _coords = build_spatial_indices(grid)

    dmap = dict(nx.single_source_shortest_path_length(G, "n_0_0"))
    return G, grid, aisle, dmap


def make_node(fid, pos, G, grid, aisle, dmap, direction=(1, 0, 0)):
    n = FleetNode(
        id=fid,
        current_pos=np.array(pos, dtype=np.float64),
        goal_pos=np.array([0.0, 0.0, 0.0]),
        grid_pos_dict=grid,
        aisle_index=aisle,
        goal_distance_map=dmap,
        current_goal_id="n_0_0",
    )
    n.G = G
    n.direction = np.array(direction, dtype=np.float32)
    return n


def test_layout_total_matches_vector():
    G, grid, aisle, dmap = build()
    a = make_node("a", (3.0, 0.0, 0.0), G, grid, aisle, dmap)
    b = make_node("b", (4.0, 0.0, 0.0), G, grid, aisle, dmap)
    v = a.get_state_vector([a, b])
    layout = a.state_layout()
    check("base_len_matches", int(layout["_base_len"][1]), int(len(v)))
    check("rays_all_is_42", layout["rays_all"][1] - layout["rays_all"][0], 42)
    check("ray_distances_at_6", layout["ray_distances"], (6, 12))
    check("goal_gradient_at_48", layout["goal_gradient"], (48, 54))
    check("situation_at_54", layout["situation_features"], (54, 60))


def test_blocks_are_contiguous_and_complete():
    G, grid, aisle, dmap = build()
    a = make_node("a", (3.0, 0.0, 0.0), G, grid, aisle, dmap)
    layout = a.state_layout()
    named = [(k, v) for k, v in layout.items()
             if not k.startswith("_") and k != "rays_all"]
    named.sort(key=lambda kv: kv[1][0])
    cursor, gaps = 0, 0
    for k, (lo, hi) in named:
        if lo != cursor:
            gaps += 1
            print(f"      gap/overlap before {k}: expected {cursor}, got {lo}")
        cursor = hi
    check("no_gaps_or_overlaps", gaps, 0)
    check("covers_whole_vector", cursor, layout["_base_len"][1])


def test_situation_features_land_where_named():
    """Perturb one named input; only its own slice may change."""
    G, grid, aisle, dmap = build()
    a = make_node("a", (3.0, 0.0, 0.0), G, grid, aisle, dmap)
    b = make_node("b", (4.0, 0.0, 0.0), G, grid, aisle, dmap)
    layout = a.state_layout()

    before = a.get_state_vector([a, b]).copy()
    a.sf_is_rescuer = 1.0
    a.sf_in_deadlock = 1.0
    after = a.get_state_vector([a, b])

    changed = set(np.flatnonzero(np.abs(after - before) > 1e-9).tolist())
    lo, hi = layout["situation_features"]
    check("situation_changed_only_its_own_slice",
          sorted(changed), [lo + 0, lo + 4])   # is_rescuer, in_deadlock


def test_goal_gradient_lands_where_named():
    G, grid, aisle, dmap = build()
    a = make_node("a", (3.0, 0.0, 0.0), G, grid, aisle, dmap)
    b = make_node("b", (4.0, 0.0, 0.0), G, grid, aisle, dmap)
    layout = a.state_layout()
    lo, hi = layout["goal_gradient"]
    v = a.get_state_vector([a, b])
    grad = a.get_goal_gradient()
    check("goal_gradient_slice_equals_function",
          np.allclose(v[lo:hi], grad), True)


def test_ray_block_equals_ray_function():
    G, grid, aisle, dmap = build()
    a = make_node("a", (3.0, 0.0, 0.0), G, grid, aisle, dmap)
    b = make_node("b", (4.0, 0.0, 0.0), G, grid, aisle, dmap)
    layout = a.state_layout()
    v = a.get_state_vector([a, b])
    rd, pv, pd = a.sense_6_axis_rays([a, b])
    lo, hi = layout["rays_all"]
    expected = np.concatenate([rd, pv.flatten(), pd.flatten()])
    check("rays_all_slice_equals_function",
          np.allclose(v[lo:hi], expected), True)


def test_lesion_zeroes_exactly_the_named_dims():
    """Simulate FLOWRRA._apply_lesion without needing torch."""
    G, grid, aisle, dmap = build()
    a = make_node("a", (3.0, 0.0, 0.0), G, grid, aisle, dmap)
    b = make_node("b", (4.0, 0.0, 0.0), G, grid, aisle, dmap)
    dens = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)

    base = a.get_state_vector([a, b])
    aff = dens.get_local_affordance(a.current_pos, [a, b], set(), own_id="a")
    full = np.concatenate([base, aff])

    layout = dict(a.state_layout())
    layout["density"] = (len(base), len(base) + dens.output_dim)

    for block in ("rays_all", "goal_gradient", "density", "situation_features"):
        lo, hi = layout[block]
        x = full.copy()
        x[lo:hi] = 0.0
        zeroed = set(np.flatnonzero(x == 0.0).tolist())
        target = set(range(lo, hi))
        # Some dims are legitimately already zero, so the test is that the
        # target is a SUBSET of the zeroed set and that nothing OUTSIDE the
        # target changed.
        changed = set(np.flatnonzero(np.abs(x - full) > 1e-12).tolist())
        check(f"lesion_{block}_subset", target.issubset(zeroed), True)
        check(f"lesion_{block}_no_collateral", changed.issubset(target), True)

    check("full_len", len(full), len(base) + dens.output_dim)
    check("density_dim_231", dens.output_dim, 231)


def test_lesion_resolution_against_shipped_code():
    """
    Exercises the ACTUAL _resolve_lesion_slices / _apply_lesion from
    core_warehouse.py, for every valid block name plus a bad one.

    WHY THIS EXISTS: the lesion hook was written on 2026-09-13 and first
    EXECUTED on 2026-09-14, when it raised AttributeError immediately -- it read
    self.nodes[0] inside __init__, before fleets are spawned. It survived a full
    session because the guard is `if self._lesion_names`, and every run until
    then used only the empty `control` arm.

    A code path gated behind a config flag is a code path that is not being
    tested. So this calls it directly rather than trusting that some run will.
    """
    import ast as _ast
    import os as _os

    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "core_warehouse.py")
    tree = _ast.parse(open(path).read())
    cls = next(n for n in tree.body
               if isinstance(n, _ast.ClassDef) and n.name == "FLOWRRA")
    wanted = {"_resolve_lesion_slices", "_apply_lesion"}
    fns = [n for n in cls.body
           if isinstance(n, _ast.FunctionDef) and n.name in wanted]
    missing = wanted - {f.name for f in fns}
    if missing:
        check("lesion_methods_present", sorted(missing), [])
        return
    mod = _ast.Module(body=fns, type_ignores=[])
    _ast.fix_missing_locations(mod)
    ns = {"np": np, "List": list, "Tuple": tuple, "Optional": object}
    exec(compile(mod, "<core_warehouse:FLOWRRA>", "exec"), ns)

    G, grid, aisle, dmap = build()
    a = make_node("a", (3.0, 0.0, 0.0), G, grid, aisle, dmap)
    dens = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)

    class _Stub:
        pass

    for block, width in (("rays_all", 42), ("goal_gradient", 6),
                         ("density", 231), ("situation_features", 6)):
        env = _Stub()
        env.nodes = [a]
        env.density = dens
        env._lesion_names = [block]
        env._lesion_slices = None
        env._resolve_lesion_slices = ns["_resolve_lesion_slices"].__get__(env)
        env._apply_lesion = ns["_apply_lesion"].__get__(env)

        vec = np.ones(291, dtype=np.float32)
        out = env._apply_lesion(vec.copy())
        lo, hi = env._lesion_slices[0]
        check(f"lesion_{block}_width", hi - lo, width)
        check(f"lesion_{block}_zeroed", int((out == 0.0).sum()), width)
        check(f"lesion_{block}_rest_intact", int((out == 1.0).sum()), 291 - width)

    # Unknown block name must raise, not silently do nothing.
    env = _Stub()
    env.nodes = [a]
    env.density = dens
    env._lesion_names = ["not_a_block"]
    env._lesion_slices = None
    env._resolve_lesion_slices = ns["_resolve_lesion_slices"].__get__(env)
    env._apply_lesion = ns["_apply_lesion"].__get__(env)
    try:
        env._apply_lesion(np.ones(291, dtype=np.float32))
        check("bad_block_raises", False, True)
    except KeyError:
        check("bad_block_raises", True, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))