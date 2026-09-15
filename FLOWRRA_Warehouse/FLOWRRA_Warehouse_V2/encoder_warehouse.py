"""
encoder_warehouse.py

The Phase-2 encoder: a GATED 3D convolution over the density volume, a separate
encoder for the base features, and LayerNorm on the fusion.

WHY THIS REPLACES Linear(313, 128)
----------------------------------
Three measured problems with the flat encoder, all of them in the INPUT rather
than the width:

1. THE DENSITY BLOCK IS A MASK. affordance = mask / (1 + R), so with R = 0 it is
   EXACTLY the structure mask. Measured 2026-09-14: at 0.25% occupancy **100%**
   of the 231 dims are exactly 0.0 or 1.0; at 7.8% it is still 96%. The
   congestion signal is roughly 5-9 effective dimensions out of 231.

2. IT DROWNS EVERYTHING ELSE. 231 of 313 input dims, so ~74% of the encoder's
   input width and ~74% of its pre-activation variance at init, spent on a
   near-constant binary stamp. The 60 ray dims are 19% and contribute like it.

3. SPACE IS THROWN AWAY. The 231 dims are a geodesic ball flattened into an
   unordered list, so the network has no idea cell 47 neighbours cell 48. Every
   spatial relationship has to be learned from scratch, by a dense layer, with
   no inductive bias at all.

WHAT THE VOLUME ACTUALLY IS
---------------------------
Not a cube and not a ball: a geodesic ball B(v, L) under the GRAPH metric, which
on a degree-~2.27 corridor graph is a thin skeleton of corridor arms radiating
from the observer. A line segment mid-aisle, a cross at a junction, a T at an
intersection, a stub at a dead end. About 35 live cells of 1,331 -- roughly 97%
of the dense array is padding.

The SHAPE is itself the signal: it says what kind of place the fleet is standing
in. A convolution is translation invariant, so a junction is a junction at cell
(5,4) and at (3000,88), and the same filters work on a 1,435-node map and a
120,000-node one. A flat Linear cannot have that property -- there, every cell
position gets its own weights.

WHY THE MASK GATES RATHER THAN BEING CONVOLVED
----------------------------------------------
A plain 3x3x3 kernel treats Euclidean neighbours as adjacent and knows nothing
about tracks, so it smears across racks -- the phantom-pair defect, now inside
the network. Measured in test_conv_leakage.py: 1.2% leaks across a rack at two
layers.

But the leak is the SMALL problem. The big one is attenuation: a 27-cell kernel
on a ~2.9-degree graph divides by 27 while only ~3 of those cells are track, so
it delivers **9x less signal** to the cell that genuinely matters (0.037 against
0.333). Gating normalises by the number of VALID cells actually in the receptive
field, and recovers all of it.

VALIDITY NEVER GROWS. In image inpainting the mask expands as the network fills
holes. Here a cell with no track must never become valid, however many layers
run -- there is no space beyond the aisle wall, not empty space. A convolution
that reaches past it is not blurring, it is inventing.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv3d(nn.Module):
    """
    Partial (gated) 3D convolution.

    Invalid cells contribute nothing, and each output is renormalised by how many
    VALID cells fell in its window. The mask is a RULE the convolution obeys, not
    a channel it processes -- so it cannot be learned away, and no amount of
    training will make the network convolve through a rack.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.kernel = kernel
        self.pad = kernel // 2
        # bias=False: the bias is added AFTER renormalisation, or it would be
        # divided by the valid count and become position-dependent.
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, padding=self.pad, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        # Fixed all-ones kernel that counts valid cells per window. Registered as
        # a buffer, never a parameter: it must not be learned.
        self.register_buffer(
            "count_kernel", torch.ones(1, 1, kernel, kernel, kernel))

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, C, D, H, W)
        mask : (B, 1, D, H, W), 1 where a track exists
        """
        out = self.conv(x * mask)
        with torch.no_grad():
            valid = F.conv3d(mask, self.count_kernel, padding=self.pad)
            live = valid > 0.5

        # torch.where, NOT a clamped divide.
        #
        # `out / valid.clamp(min=1e-8)` looks equivalent and is not: where the
        # window holds no valid cells, it divides by 1e-8 and turns any float
        # residue in the numerator into ~1e8, which becomes inf and then NaN
        # through the whole network on the first backward pass. With a truly
        # binary mask the numerator there is exactly 0 so it never fires -- which
        # is precisely the problem, because a guard that only holds for
        # well-formed input is not a guard. Caught by feeding a non-binary mask.
        out = torch.where(live, out / valid.clamp(min=1.0),
                          torch.zeros_like(out))
        out = out + self.bias.view(1, -1, 1, 1, 1)
        # Gate the output as well as the input: a cell with no track produces
        # nothing, so it can never act as a bridge for the next layer.
        out = out * mask
        # Validity is returned UNCHANGED, deliberately. See the module docstring.
        return out, mask


class DensityEncoder(nn.Module):
    """
    Scatter the 2-channel diamond back into a cube, convolve it with the mask
    gating, and pool to a fixed-width embedding.

    STORED AS A DIAMOND, PROCESSED AS A CUBE. The replay buffer holds
    2 x 231 = 462 numbers rather than 2 x 1331 = 2662: at buffer_capacity 15,000
    and 60 fleets that is 3.9 GB against 19.8 GB, and the dense cube is 97%
    padding anyway. The scatter is a fixed index assignment, so it costs nothing
    to undo.
    """

    def __init__(self, diamond_mask: np.ndarray, out_dim: int = 32,
                 widths: Sequence[int] = (8, 16, 32), kernel: int = 3):
        super().__init__()
        self.grid_shape = tuple(diamond_mask.shape)
        self.n_cells = int(diamond_mask.sum())
        flat_idx = np.flatnonzero(diamond_mask.ravel())
        self.register_buffer("flat_idx", torch.from_numpy(flat_idx).long())

        convs = []
        in_ch = 2
        for w in widths:
            convs.append(GatedConv3d(in_ch, w, kernel))
            in_ch = w
        self.convs = nn.ModuleList(convs)
        self.out = nn.Linear(widths[-1], out_dim)
        self.out_dim = out_dim

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """packed : (B, 2 * n_cells) -- mask channel then repulsion channel."""
        B = packed.shape[0]
        n = self.n_cells
        cube = packed.new_zeros(B, 2, int(np.prod(self.grid_shape)))
        cube[:, 0].index_copy_(1, self.flat_idx, packed[:, :n])
        cube[:, 1].index_copy_(1, self.flat_idx, packed[:, n:])
        cube = cube.view(B, 2, *self.grid_shape)

        mask = cube[:, :1]
        x = cube
        for conv in self.convs:
            x, mask = conv(x, mask)
            x = F.relu(x)

        # Masked mean over live cells only. A plain global pool would divide by
        # 1,331 while ~35 cells are real, reintroducing the same 9x attenuation
        # the gating exists to remove -- and it would make the embedding depend
        # on how much padding a given cell's neighbourhood happens to have.
        denom = mask.sum(dim=(2, 3, 4)).clamp(min=1e-8)
        pooled = x.sum(dim=(2, 3, 4)) / denom
        return self.out(pooled)


class FusedEncoder(nn.Module):
    """
    Separate encoders for the two halves, then LayerNorm on the fusion.

    SEPARATE ENCODERS ARE WHAT FIX DROWNING, NOT LAYERNORM. LayerNorm normalises
    AFTER a layer; the 462-versus-82 imbalance happens at that layer's INPUT.
    Compressing each half independently equalises their contributions by
    construction: the density block reaches the fusion as `density_dim` numbers
    however many cells it started with.

    LayerNorm still earns its place -- the base half now mixes features on
    genuinely different scales (entropy in [0,1], T in (0,1], ray distances,
    binary flags) and the fusion benefits from being renormalised.
    """

    def __init__(self, base_dim: int, diamond_mask: np.ndarray,
                 hidden_dim: int = 128, base_hidden: int = 96,
                 density_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.base_dim = base_dim
        self.density_encoder = DensityEncoder(diamond_mask, out_dim=density_dim)
        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, base_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(base_hidden + density_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        node_features : (..., base_dim + 2 * n_cells)

        Accepts any leading dimensions -- the GAT passes (batch, nodes, features)
        -- and flattens them for the convolution, which needs a real batch axis.
        """
        lead = node_features.shape[:-1]
        flat = node_features.reshape(-1, node_features.shape[-1])
        base = flat[:, :self.base_dim]
        packed = flat[:, self.base_dim:]
        h = torch.cat([self.base_encoder(base),
                       self.density_encoder(packed)], dim=-1)
        return self.fuse(h).reshape(*lead, self.hidden_dim)


def parameter_report(base_dim: int, n_cells: int, hidden_dim: int = 128) -> str:
    """Side-by-side count against the flat encoder it replaces."""
    flat_in = base_dim + n_cells
    flat = flat_in * hidden_dim + hidden_dim + hidden_dim * hidden_dim + hidden_dim
    mask = np.zeros((11, 11, 11), dtype=bool)
    mask.ravel()[:n_cells] = True
    enc = FusedEncoder(base_dim, mask, hidden_dim=hidden_dim)
    conv = sum(p.numel() for p in enc.parameters())
    return (f"flat Linear({flat_in},{hidden_dim}) stack : {flat:>8,} params\n"
            f"fused conv encoder                 : {conv:>8,} params\n"
            f"ratio                              : {flat / conv:>8.2f}x")