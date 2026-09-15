"""
test_conv_leakage.py

QUESTION: if the 231-dim density block becomes a 3D volume fed to a convolution,
does repulsion leak ACROSS A RACK -- from one aisle into the next -- where no
track connects them?

This is the phantom-pair problem again, one level down. We fixed it in braking,
in check_integrity, in the falloff kernel and in sf_peer_proximity. A 3D
convolution would reintroduce it INSIDE the network, because a 3x3x3 kernel
treats Euclidean neighbours as adjacent and knows nothing about tracks.

THE GEOMETRY

    y=2   B---B---B---B---B      aisle 2 (x = 0..6)
          |
    y=1   C                      cross-aisle, ONLY at x=0
          |
    y=0   A---A---A---A---A      aisle 0 (x = 0..6)

    observer at (1,0,0)
    SOURCE   at (2,0,0)   graph distance 1 from observer
    TARGET   at (2,2,0)   graph distance 6 from SOURCE (all the way round via x=0)
                          but EUCLIDEAN distance 2 -- two cells apart in y

    The cell between them, (2,1,0), is solid rack: not a graph node at all.

Both SOURCE and TARGET are inside the observer's depth-5 BFS mask, so both are
live cells in the affordance volume. A leak between them is a real error, not a
masked-out one.

THREE SCHEMES COMPARED

  plain      a uniform 3x3x3 box filter. Knows nothing about tracks.
  gated      partial convolution: invalid cells contribute nothing and each
             output is renormalised by how many VALID cells were in its
             receptive field. Validity propagates forward. The mask is a RULE
             the convolution obeys, not a channel it processes.
  graph      aggregates each cell only from its true graph neighbours. Cannot
             smear across a rack because there is no edge to smear along.

Depth matters: one 3x3x3 layer reaches 1 cell, two layers reach 2. A rack is
one cell thick, so two layers is the minimum interesting depth.
"""

import numpy as np
import networkx as nx

from density_warehouse import WarehouseDensityField

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------------- world
def build():
    G = nx.Graph()
    grid = {}
    for y in (0, 2):
        for x in range(7):
            grid[(x, y, 0)] = f"n_{x}_{y}"
            G.add_node(f"n_{x}_{y}")
        for x in range(6):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    grid[(0, 1, 0)] = "n_0_1"
    G.add_node("n_0_1")
    G.add_edge("n_0_0", "n_0_1")
    G.add_edge("n_0_1", "n_0_2")
    return G, grid


# ------------------------------------------------------------ convolutions
def plain_conv(vol, passes):
    """Uniform 3x3x3 box filter. No notion of tracks."""
    out = vol.copy()
    for _ in range(passes):
        acc = np.zeros_like(out)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    acc += np.roll(np.roll(np.roll(out, dx, 0), dy, 1), dz, 2)
        out = acc / 27.0
    return out


def gated_conv(vol, mask, passes):
    """
    Partial convolution. Invalid cells contribute nothing, and each output is
    renormalised by the number of VALID cells that fell in its window. Validity
    propagates: a cell stays invalid forever, so it can never act as a bridge.
    """
    out = vol * mask
    valid = mask.astype(np.float32).copy()
    for _ in range(passes):
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    def sh(a):
                        return np.roll(np.roll(np.roll(a, dx, 0), dy, 1), dz, 2)
                    acc += sh(out * valid)
                    cnt += sh(valid)
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1e-9), 0.0) * mask
    return out


def graph_conv(vol, coords_by_index, index_by_coord, G, grid, passes):
    """
    Aggregate each cell from its TRUE graph neighbours plus itself. Cannot smear
    across a rack: there is no edge to smear along, and it follows corridors
    around corners, which a 3x3x3 kernel cannot.
    """
    vals = {idx: float(vol[idx]) for idx in coords_by_index}
    nbrs = {}
    for idx, coord in coords_by_index.items():
        nid = grid.get(coord)
        out = [idx]
        if nid is not None:
            for nb in G.neighbors(nid):
                for c2, n2 in grid.items():
                    if n2 == nb and c2 in index_by_coord:
                        out.append(index_by_coord[c2])
        nbrs[idx] = out
    for _ in range(passes):
        vals = {idx: sum(vals[j] for j in nbrs[idx]) / len(nbrs[idx])
                for idx in vals}
    out = np.zeros_like(vol)
    for idx, v in vals.items():
        out[idx] = v
    return out


# -------------------------------------------------------------------- test
def run():
    G, grid = build()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    L = f.local_radius
    centre = (1, 0, 0)
    cidx = np.array(centre)

    def to_index(coord):
        return tuple(np.array(coord) - cidx + L)

    SRC = (2, 0, 0)
    TGT = (2, 2, 0)
    RACK = (2, 1, 0)

    print(f"observer {centre}   SOURCE {SRC}   TARGET {TGT}   rack cell {RACK}")
    print(f"  euclidean SOURCE->TARGET : {int(np.sum(np.abs(np.array(SRC)-np.array(TGT))))}")
    print(f"  graph     SOURCE->TARGET : "
          f"{nx.shortest_path_length(G, grid[SRC], grid[TGT])}")
    print(f"  rack cell is a graph node: {RACK in grid}")

    mask = f._structure_mask(cidx).astype(np.float32)
    live = {c: to_index(c) for c in grid if
            all(0 <= to_index(c)[d] < 2 * L + 1 for d in range(3))}
    live = {c: i for c, i in live.items() if mask[i] > 0}
    print(f"  cells inside the depth-{L} mask: {len(live)}")
    check("source_is_live", SRC in live, True)
    check("target_is_live", TGT in live, True)

    vol = np.zeros(f.grid_shape, dtype=np.float32)
    vol[to_index(SRC)] = 5.0          # one strong repulsion spike

    index_by_coord = dict(live)
    coords_by_index = {i: c for c, i in live.items()}

    print("\nleakage into TARGET, as a fraction of the SOURCE spike:")
    print(f"{'passes':>7}{'plain':>12}{'gated':>12}{'graph':>12}")
    worst_gated = 0.0
    worst_graph = 0.0
    for p in (1, 2, 3):
        a = plain_conv(vol, p)[to_index(TGT)] / 5.0
        b = gated_conv(vol, mask, p)[to_index(TGT)] / 5.0
        c = graph_conv(vol, coords_by_index, index_by_coord, G, grid, p)[to_index(TGT)] / 5.0
        worst_gated = max(worst_gated, abs(float(b)))
        worst_graph = max(worst_graph, abs(float(c)))
        print(f"{p:>7}{float(a):>12.5f}{float(b):>12.5f}{float(c):>12.5f}")

    leak2 = plain_conv(vol, 2)[to_index(TGT)] / 5.0
    print()
    check("plain_conv_leaks_across_rack", bool(leak2 > 1e-6), True)
    check("gated_conv_no_leak", worst_gated < 1e-9, True)
    check("graph_conv_no_leak", worst_graph < 1e-9, True)

    # Does the signal still travel where it SHOULD? A conv that leaks nowhere
    # because it moves nothing would pass the tests above and be useless.
    nxt = (3, 0, 0)
    print("\nsignal reaching (3,0,0) -- same aisle, genuinely 1 hop away:")
    for label, v in (("plain", plain_conv(vol, 1)),
                     ("gated", gated_conv(vol, mask, 1)),
                     ("graph", graph_conv(vol, coords_by_index, index_by_coord,
                                          G, grid, 1))):
        print(f"   {label:<6} {float(v[to_index(nxt)]) / 5.0:.5f}")
    ok = all(float(v[to_index(nxt)]) > 1e-6 for v in
             (plain_conv(vol, 1), gated_conv(vol, mask, 1),
              graph_conv(vol, coords_by_index, index_by_coord, G, grid, 1)))
    check("all_schemes_propagate_along_the_aisle", ok, True)

    # Cost per cell: how many contributions each scheme sums.
    deg = np.mean([len(list(G.neighbors(grid[c]))) + 1 for c in live])
    print(f"\nreceptive-field work per cell:")
    print(f"   plain / gated 3x3x3 : 27")
    print(f"   graph (mean degree) : {deg:.2f}   -> {27/deg:.1f}x cheaper")


if __name__ == "__main__":
    run()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))