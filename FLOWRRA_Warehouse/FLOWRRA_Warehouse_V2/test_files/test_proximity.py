"""
test_proximity.py -- correctness tests for proximity_warehouse.GraphProximity.

The graph is a two-aisle warehouse with a solid rack between the aisles:

    y=2   A---A---A---A---A---A        (x = 0..5)
          |                   |
    y=1   C       RACK        C
          |                   |
    y=0   A---A---A---A---A---A

Aisles connect ONLY at the two cross-aisles x=0 and x=5. So (2,0,0) and
(2,2,0) are Manhattan distance 2 apart and graph distance 6 apart. That is the
phantom pair the old metric could not see.
"""

import numpy as np
import networkx as nx

from proximity_warehouse import GraphProximity


def build_graph():
    G = nx.Graph()
    grid = {}

    def add(x, y):
        nid = f"n_{x}_{y}"
        grid[(x, y, 0)] = nid
        G.add_node(nid)
        return nid

    for y in (0, 2):
        for x in range(6):
            add(x, y)
        for x in range(5):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")

    for x in (0, 5):
        add(x, 1)
        G.add_edge(f"n_{x}_0", f"n_{x}_1")
        G.add_edge(f"n_{x}_1", f"n_{x}_2")

    return G, grid


class Fleet:
    def __init__(self, fid, pos):
        self.id = fid
        self.current_pos = np.array(pos, dtype=np.float64)


def prox(radius=8.0, metric="graph"):
    G, grid = build_graph()
    return GraphProximity(G, grid, search_radius=radius, metric=metric)


FAIL = []


def check(name, got, want, tol=1e-9):
    if isinstance(want, float) and not np.isfinite(want):
        ok = (got == want)
    elif isinstance(want, float):
        ok = np.isfinite(got) and abs(got - want) <= tol
    else:
        ok = (got == want)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------------------
def test_anchor_on_node():
    p = prox()
    a = p._anchor(np.array([2.0, 0.0, 0.0]))
    check("anchor_on_node.count", len(a), 1)
    check("anchor_on_node.id", a[0][0], "n_2_0")
    check("anchor_on_node.resid", a[0][1], 0.0)


def test_anchor_mid_edge():
    p = prox()
    a = dict(p._anchor(np.array([2.5, 0.0, 0.0])))
    check("anchor_mid.count", len(a), 2)
    check("anchor_mid.behind", a["n_2_0"], 0.5)
    check("anchor_mid.ahead", a["n_3_0"], 0.5)

    # np.round uses banker's rounding: round(3.5) == 4, not 3. The anchor must
    # still resolve to the same edge {3,4}, just approached from the other side.
    b = dict(p._anchor(np.array([3.5, 0.0, 0.0])))
    check("anchor_mid_bankers.count", len(b), 2)
    check("anchor_mid_bankers.behind", b["n_3_0"], 0.5)
    check("anchor_mid_bankers.ahead", b["n_4_0"], 0.5)


def test_basic_distances():
    p = prox()
    fleets = [
        Fleet("a", (2.0, 0.0, 0.0)),
        Fleet("b", (2.0, 0.0, 0.0)),   # stacked -> 0.0
        Fleet("c", (3.0, 0.0, 0.0)),   # adjacent -> 1.0
        Fleet("d", (4.5, 0.0, 0.0)),   # 2.5 from a
    ]
    p.refresh(fleets, excluded_ids=set())
    w = p._within["a"]
    check("stacked", w["b"], 0.0)
    check("adjacent", w["c"], 1.0)
    check("half_step", w["d"], 2.5)


def test_half_step_pair():
    p = prox()
    fleets = [Fleet("a", (2.5, 0.0, 0.0)), Fleet("b", (3.0, 0.0, 0.0))]
    p.refresh(fleets, excluded_ids=set())
    check("node_to_edge", p.nearest("a"), 0.5)


def test_same_edge_pair():
    """
    The one case the anchor-min formula gets wrong. Two fleets on the interior
    of edge n_2_0 -- n_3_0, 0.4 apart. Anchor-min would route out to an endpoint
    and back and report 1.0, hiding a collision. The special case must report 0.4.
    """
    p = prox()
    fleets = [Fleet("a", (2.3, 0.0, 0.0)), Fleet("b", (2.7, 0.0, 0.0))]
    p.refresh(fleets, excluded_ids=set())
    got = p.nearest("a")
    check("same_edge_pair", got, 0.4, tol=1e-9)
    # And it must be symmetric.
    check("same_edge_pair_sym", p.nearest("b"), 0.4, tol=1e-9)


def test_phantom_pair_rejected():
    """(2,0) and (2,2): Manhattan 2 (inside the warning band), graph 6."""
    p = prox()
    fleets = [Fleet("a", (2.0, 0.0, 0.0)), Fleet("b", (2.0, 2.0, 0.0))]
    p.refresh(fleets, excluded_ids=set())
    check("phantom_graph_distance", p.nearest("a"), 6.0)
    check("phantom_not_in_warning_band", len(p.pairs(radius=2.0)), 0)

    m = prox(metric="manhattan")
    m.refresh(fleets, excluded_ids=set())
    check("phantom_manhattan_distance", m.nearest("a"), 2.0)
    check("phantom_manhattan_fires", len(m.pairs(radius=2.0)), 1)


def test_real_pair_still_detected():
    """Same aisle, genuinely 1 cell apart. Must survive the change."""
    p = prox()
    fleets = [Fleet("a", (2.0, 0.0, 0.0)), Fleet("b", (3.0, 0.0, 0.0))]
    p.refresh(fleets, excluded_ids=set())
    pr = p.pairs(radius=2.0)
    check("real_pair_count", len(pr), 1)
    check("real_pair_dist", pr[0][2], 1.0)


def test_excluded_peer_invisible():
    p = prox()
    fleets = [Fleet("a", (2.0, 0.0, 0.0)), Fleet("b", (3.0, 0.0, 0.0))]
    p.refresh(fleets, excluded_ids={"b"})
    check("excluded_nearest", p.nearest("a"), float("inf"))
    check("excluded_pairs", len(p.pairs()), 0)
    # An excluded fleet still gets its OWN reading (sf_peer_proximity needs one
    # for every fleet), and sees the non-excluded peer.
    check("excluded_self_reading", p.nearest("b"), 1.0)


def test_censored_gap_never_infinite():
    p = prox(radius=4.0)
    fleets = [Fleet("a", (0.0, 0.0, 0.0)), Fleet("b", (5.0, 2.0, 0.0))]
    p.refresh(fleets, excluded_ids={"b"})
    check("nearest_is_inf", p.nearest("a"), float("inf"))
    check("censored_is_finite", p.nearest_censored("a"), 4.0)


def test_off_grid_fallback():
    """A position that is not on the graph at all must not disable braking."""
    p = prox()
    fleets = [Fleet("a", (2.0, 1.0, 0.0)), Fleet("b", (3.0, 0.0, 0.0))]  # (2,1,0) is RACK
    p.refresh(fleets, excluded_ids=set())
    check("off_grid_counted", p.off_grid_fleets, 1)
    check("off_grid_falls_back_to_manhattan", p.nearest("a"), 2.0)


def test_pairs_dedup():
    p = prox()
    fleets = [Fleet(c, (i, 0.0, 0.0)) for i, c in enumerate("abcd")]
    p.refresh(fleets, excluded_ids=set())
    pr = p.pairs(radius=1.0)
    keys = {(a, b) for a, b, _ in pr}
    check("pairs_dedup_count", len(pr), len(keys))
    check("pairs_adjacent_only", sorted(keys), [("a", "b"), ("b", "c"), ("c", "d")])


def test_manhattan_matches_old_code_exactly():
    """
    metric='manhattan' must reproduce np.sum(np.abs(...)) over the same
    exclusion set, for every pair, so published runs can be re-created.
    """
    rng = np.random.default_rng(0)
    fleets = []
    for i in range(12):
        x = float(rng.integers(0, 6))
        y = float(rng.choice([0.0, 2.0]))
        if rng.random() < 0.5:
            x += 0.5
        fleets.append(Fleet(f"f{i}", (x, y, 0.0)))
    excluded = {"f3", "f7"}

    m = prox(radius=1e9, metric="manhattan")
    m.refresh(fleets, excluded_ids=excluded)

    worst = 0.0
    for n in fleets:
        old = min(
            (float(np.sum(np.abs(n.current_pos - o.current_pos)))
             for o in fleets if o.id != n.id and o.id not in excluded),
            default=float("inf"),
        )
        new = m.nearest(n.id)
        worst = max(worst, abs(old - new) if np.isfinite(old) else 0.0)
    check("manhattan_reproduces_old", worst, 0.0)


def test_graph_never_shorter_than_manhattan():
    """
    Sanity invariant: on a unit-spaced grid, graph distance can never be LESS
    than Manhattan distance. If it ever is, the anchor arithmetic is wrong.
    """
    rng = np.random.default_rng(7)
    fleets = []
    for i in range(20):
        x = float(rng.integers(0, 6))
        y = float(rng.choice([0.0, 2.0]))
        if rng.random() < 0.5 and x < 5:
            x += float(rng.choice([0.25, 0.5, 0.75]))
        fleets.append(Fleet(f"g{i}", (x, y, 0.0)))

    g = prox(radius=1e9)
    g.refresh(fleets, excluded_ids=set())
    violations = 0
    for a, b, d in g.pairs():
        man = float(np.sum(np.abs(
            next(f for f in fleets if f.id == a).current_pos
            - next(f for f in fleets if f.id == b).current_pos)))
        if d < man - 1e-9:
            violations += 1
            print(f"      violation {a}-{b}: graph {d} < manhattan {man}")
    check("graph_ge_manhattan", violations, 0)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))