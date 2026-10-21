"""
test_encoder.py -- the Phase-2 gated 3D convolution encoder.

Requires torch. The rest of the suite deliberately does not, so this is the one
file that will not run in a bare numpy environment.

WHAT IS BEING ASSERTED

  1. GATING HOLDS. Repulsion must not cross a rack, at any depth. This is the
     phantom-pair defect one level down: a plain 3x3x3 kernel treats Euclidean
     neighbours as adjacent and knows nothing about tracks.

  2. ATTENUATION IS RECOVERED. The bigger of the two problems. A 27-cell kernel
     on a ~2.9-degree graph divides by 27 while only ~3 of those cells are
     track, so a plain conv delivers 9x less signal to the cell that matters.
     Gating renormalises by the VALID count and gets it back.

  3. VALIDITY NEVER GROWS. In inpainting the mask expands as holes fill. Here a
     cell with no track must never become valid however many layers run -- there
     is no space beyond the aisle wall, not empty space.

  4. TRANSLATION INVARIANCE. The property a flat Linear cannot have: the same
     neighbourhood must encode the same way wherever it sits on the map. This is
     what makes one policy work on a 1,435-node map and a 120,000-node one.

  5. GRADIENTS FLOW AND STAY FINITE, including on malformed input.
"""

import numpy as np
import networkx as nx
import torch

from density_warehouse import WarehouseDensityField
from encoder_warehouse import GatedConv3d, DensityEncoder, FusedEncoder

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def racked(width=14):
    """Two aisles, cross-aisles only at the ends -- a rack between them."""
    G = nx.Graph()
    grid = {}
    for y in (0, 2):
        for x in range(width):
            grid[(x, y, 0)] = f"n_{x}_{y}"
            G.add_node(f"n_{x}_{y}")
        for x in range(width - 1):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, width - 1):
        grid[(x, 1, 0)] = f"n_{x}_1"
        G.add_node(f"n_{x}_1")
        G.add_edge(f"n_{x}_0", f"n_{x}_1")
        G.add_edge(f"n_{x}_1", f"n_{x}_2")
    return G, grid


def identity_gated(in_ch=1, out_ch=1):
    """A GatedConv3d whose weights are a uniform box filter, so its output is
    directly comparable with the numpy box filter in test_conv_leakage.py."""
    g = GatedConv3d(in_ch, out_ch, 3)
    with torch.no_grad():
        g.conv.weight.fill_(1.0)
        g.bias.fill_(0.0)
    return g


# ====================================================================== 1 & 2
def test_gating_blocks_the_rack_and_keeps_the_signal():
    G, grid = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    L = f.local_radius
    centre = np.array([1, 0, 0])

    def idx(c):
        return tuple(np.array(c) - centre + L)

    mask_np = f._structure_mask(centre).astype(np.float32)
    SRC, TGT, NEXT = (2, 0, 0), (2, 2, 0), (3, 0, 0)
    check("source_live", bool(mask_np[idx(SRC)] > 0), True)
    check("target_live", bool(mask_np[idx(TGT)] > 0), True)
    check("rack_cell_dead", bool(mask_np[idx((2, 1, 0))] == 0), True)

    vol = np.zeros(f.grid_shape, dtype=np.float32)
    vol[idx(SRC)] = 5.0
    x = torch.tensor(vol)[None, None]
    m = torch.tensor(mask_np)[None, None]

    g = identity_gated()
    gx, gm = x, m
    for _ in range(3):
        gx, gm = g(gx, gm)

    leak = float(gx[0, 0][idx(TGT)]) / 5.0
    signal = float(gx[0, 0][idx(NEXT)]) / 5.0
    print(f"      across the rack {leak:.6f}   along the aisle {signal:.6f}")
    check("no_leak_across_rack", abs(leak) < 1e-6, True)
    check("signal_survives_along_aisle", signal > 1e-3, True)

    # A PLAIN conv, same kernel, no gating: leaks and attenuates.
    px = torch.tensor(vol)[None, None]
    conv = torch.nn.Conv3d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0 / 27.0)
    for _ in range(3):
        px = conv(px)
    p_leak = float(px[0, 0][idx(TGT)]) / 5.0
    p_signal = float(px[0, 0][idx(NEXT)]) / 5.0
    print(f"      plain: across {p_leak:.6f}   along {p_signal:.6f}")
    check("plain_conv_leaks", p_leak > 1e-6, True)
    check("gating_recovers_signal", signal > p_signal, True)


# ====================================================================== 3
def test_validity_never_grows():
    G, grid = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    m0 = torch.tensor(f._structure_mask(np.array([1, 0, 0])).astype(np.float32))[None, None]
    x = torch.randn_like(m0)
    g = identity_gated()
    m = m0
    for _ in range(5):
        x, m = g(x, m)
    check("mask_identical_after_5_layers", bool(torch.equal(m, m0)), True)
    check("dead_cells_stay_zero",
          float(x[m0 == 0].abs().max()) < 1e-9, True)


# ====================================================================== 4
def test_translation_invariance():
    """
    The property a flat Linear cannot have. Two fleets standing mid-aisle at
    different x must produce the SAME embedding from the same neighbourhood --
    which is why one policy transfers between a 1,435-node map and a 120,000-node
    one.
    """
    G, grid = racked(width=20)
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    dm = f._diamond_mask
    enc = DensityEncoder(dm, out_dim=16).eval()

    def embed(cx):
        c = np.array([cx, 0, 0])
        mask = f._structure_mask(c).astype(np.float32)
        rep = np.zeros(f.grid_shape, dtype=np.float32)
        # identical local congestion: one peer, one cell to the +X side
        rep[tuple(np.array([cx + 1, 0, 0]) - c + f.local_radius)] = 0.7
        packed = np.concatenate([mask[dm], rep[dm]])
        with torch.no_grad():
            return enc(torch.tensor(packed)[None])

    a, b = embed(8), embed(11)
    d = float((a - b).abs().max())
    print(f"      max embedding difference across a 3-cell shift: {d:.2e}")
    check("same_neighbourhood_same_embedding", d < 1e-5, True)


# ====================================================================== 5
def test_gradients_finite_including_malformed_input():
    G, grid = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    dm = f._diamond_mask
    n = int(dm.sum())
    enc = FusedEncoder(base_dim=82, diamond_mask=dm)

    for label, x in (("real binary mask",
                      torch.cat([torch.randn(2, 5, 82),
                                 torch.cat([torch.ones(2, 5, n),
                                            torch.rand(2, 5, n)], -1)], -1)),
                     ("non-binary mask (malformed)",
                      torch.randn(2, 5, 82 + 2 * n)),
                     ("all-dead mask",
                      torch.cat([torch.randn(2, 5, 82),
                                 torch.zeros(2, 5, 2 * n)], -1))):
        enc.zero_grad()
        out = enc(x)
        check(f"output_finite__{label}", bool(torch.isfinite(out).all()), True)
        out.pow(2).mean().backward()
        bad = [nm for nm, p in enc.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
        check(f"grads_finite__{label}", bad, [])


def test_shapes_and_parameter_budget():
    G, grid = racked()
    f = WarehouseDensityField(max_vision_range=10, grid_pos_dict=grid, graph=G)
    dm = f._diamond_mask
    n = int(dm.sum())
    enc = FusedEncoder(base_dim=82, diamond_mask=dm)
    out = enc(torch.randn(3, 9, 82 + 2 * n))
    check("preserves_leading_dims", tuple(out.shape), (3, 9, 128))

    conv_params = sum(p.numel() for p in enc.parameters())
    flat_in = 82 + 231
    flat_params = flat_in * 128 + 128 + 128 * 128 + 128
    print(f"      flat Linear({flat_in},128) stack: {flat_params:,}")
    print(f"      fused conv encoder            : {conv_params:,}")
    check("conv_encoder_is_smaller", conv_params < flat_params, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))