"""
test_loop_equivalence.py

INVARIANT UNDER TEST: the graph metric may only ever REJECT pairs that Manhattan
flags. It must never flag a pair Manhattan missed. If graph_only > 0 anywhere,
the anchor arithmetic is under-estimating and the change is unsafe.

FINDING FROM THIS TEST (2026-09-13). The original premise was "on an open grid
with no racks the two metrics are identical." That is FALSE, and the test caught
it: two fleets both mid-edge on PARALLEL tracks disagree even with no rack
anywhere.

    f0 at (6.5, 5, 0)   mid-edge on (6,5)--(7,5)
    f3 at (6.5, 3, 0)   mid-edge on (6,3)--(7,3)
    Manhattan = 2.0  -> inside the warning band, both fleets braked
    Graph     = 3.0  -> 0.5 back to a node, 2 across, 0.5 out again

There is no track at x=6.5 running in y. Manhattan cuts straight across the
middle of a cell where no rail exists. So this is a SECOND SPECIES of phantom
pair, unrelated to racks: it is created purely by treating continuous positions
as free points in 3-space.

It matters more than the rack species, because at base_speed 0.5 a fleet is
mid-edge roughly half the time, so roughly a QUARTER of all pairs have both
fleets mid-edge. The earlier phantom_pct audit measured rack phantoms only and
would not have seen this at all.
"""

import numpy as np
import networkx as nx

from loop_warehouse import WarehouseLoop
from proximity_warehouse import GraphProximity

COLLISION = 0.5
WARNING = 2.0


class Fleet:
    def __init__(self, fid, pos):
        self.id = fid
        self.current_pos = np.array(pos, dtype=np.float64)


def open_grid(n=8):
    G = nx.Graph()
    grid = {}
    for x in range(n):
        for y in range(n):
            grid[(x, y, 0)] = f"o_{x}_{y}"
            G.add_node(f"o_{x}_{y}")
    for x in range(n):
        for y in range(n):
            if x + 1 < n:
                G.add_edge(f"o_{x}_{y}", f"o_{x+1}_{y}")
            if y + 1 < n:
                G.add_edge(f"o_{x}_{y}", f"o_{x}_{y+1}")
    return G, grid


def racked_grid(n=8):
    """Same grid but every other column of horizontal links removed."""
    G, grid = open_grid(n)
    for x in range(1, n - 1, 2):
        for y in range(1, n - 1):
            for nb in (f"o_{x-1}_{y}", f"o_{x+1}_{y}"):
                if G.has_edge(f"o_{x}_{y}", nb):
                    G.remove_edge(f"o_{x}_{y}", nb)
    return G, grid


def classify(loop, nodes, frozen, prox):
    loop.check_integrity(nodes, 0, frozen, proximity=prox)
    return (frozenset(loop.deadlocked_nodes), frozenset(loop.warning_nodes), loop.current_integrity)


def run(graph_builder, label, trials=300, seed=11):
    G, grid = graph_builder()
    rng = np.random.default_rng(seed)
    disagreements = 0
    graph_only = 0     # graph flagged something Manhattan did not -- must be 0
    manhattan_only = 0
    for t in range(trials):
        k = int(rng.integers(2, 9))
        nodes = []
        for i in range(k):
            x = int(rng.integers(0, 8))
            y = int(rng.integers(0, 8))
            pos = [float(x), float(y), 0.0]
            if rng.random() < 0.4:
                ax = int(rng.integers(0, 2))
                if pos[ax] < 7:
                    pos[ax] += 0.5
            nodes.append(Fleet(f"f{i}", pos))
        frozen = {f"f{i}" for i in range(k) if rng.random() < 0.2}

        prox = GraphProximity(G, grid, search_radius=WARNING, metric="graph")
        prox.refresh(nodes, excluded_ids=frozen)

        a = classify(WarehouseLoop(COLLISION, WARNING), nodes, frozen, None)
        b = classify(WarehouseLoop(COLLISION, WARNING), nodes, frozen, prox)

        if a != b:
            disagreements += 1
            if (b[0] | b[1]) - (a[0] | a[1]):
                graph_only += 1
            if (a[0] | a[1]) - (b[0] | b[1]):
                manhattan_only += 1

    print(f"[{label}] trials={trials} disagreements={disagreements} "
          f"manhattan_only={manhattan_only} graph_only={graph_only}")
    return disagreements, manhattan_only, graph_only


if __name__ == "__main__":
    fails = []

    d, m, g = run(open_grid, "open grid")
    if g != 0:
        fails.append(f"open grid: graph flagged {g} pairs Manhattan did not (must be 0)")

    d2, m2, g2 = run(racked_grid, "racked grid")
    if g2 != 0:
        fails.append(f"racked grid: graph flagged {g2} pairs Manhattan did not (must be 0)")
    if m2 <= m:
        fails.append("racked grid rejected no MORE than the open grid -- racks not exercised")

    # The off-track species, isolated: two fleets mid-edge on parallel tracks,
    # no rack involved.
    G, grid = open_grid()
    pair = [Fleet("a", (6.5, 5.0, 0.0)), Fleet("b", (6.5, 3.0, 0.0))]
    px = GraphProximity(G, grid, search_radius=8.0, metric="graph")
    px.refresh(pair, excluded_ids=set())
    man = float(np.sum(np.abs(pair[0].current_pos - pair[1].current_pos)))
    got = px.nearest("a")
    print(f"[off-track species] manhattan={man} graph={got}")
    if not (abs(got - 3.0) < 1e-9 and abs(man - 2.0) < 1e-9):
        fails.append(f"off-track species: expected manhattan 2.0 / graph 3.0, got {man} / {got}")

    print()
    print("ALL PASS" if not fails else "FAILURES:\n  " + "\n  ".join(fails))