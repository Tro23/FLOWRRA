# FLOWRRA — quick reference

Scannable index of what changed and what the flags are called.
Full reasoning lives in `CHANGES_2026-09-13.md` and `DESIGN_PROPOSAL.md`.

---

## State vector — 313 dims (82 base + 231 density)

| block | range | dims | added |
|---|---|---|---|
| `self_direction` | [0, 3) | 3 | — |
| `self_displacement` | [3, 6) | 3 | — |
| `ray_distances` | [6, 12) | 6 | — |
| `peer_velocities` | [12, 30) | 18 | — |
| `peer_displacements` | [30, 48) | 18 | — |
| `ray_hit_waiting` | [48, 54) | 6 | **Phase 2** |
| `ray_hit_permanent` | [54, 60) | 6 | **Phase 2** |
| `ray_hit_unknown` | [60, 66) | 6 | **Phase 2** |
| `goal_gradient` | [66, 72) | 6 | — |
| `situation_features` | [72, 80) | 8 | **+2 Phase 2** |
| `gibbs_state` | [80, 82) | 2 | **Phase 2** |
| `density` | [82, 313) | 231 | — |

Convenience groups for lesioning, overlapping the above:
`rays_all` [6, 66) = 60 dims · `ray_semantics` [48, 66) = 18 dims

**Never hardcode the width.** Every consumer computes
`input_dim = len(get_state_vector) + density.output_dim`. A width change
invalidates saved checkpoints (`load_state_dict` is strict) — by design.

**`gnn.memory.push` needs no change.** It receives `node_features_array` built
by `_perceive`, so new dims flow through on their own.

---

## Flags — all default to pre-change behaviour

| flag | default | alternative |
|---|---|---|
| `proximity.metric` | `graph` | `manhattan` |
| `proximity.adjacency_metric` | **`manhattan`** | `graph` |
| `density.kernel_metric` | `graph` | `manhattan` |
| `density.projection_mode` | `intended` | `dead_reckoning` |
| `density.project_stationary` | `True` | `False` |
| `density.ray_transform` | **`clip25`** | `smooth` |
| `density.output_mode` | **`affordance`** | `channels` (462 dims) |
| `density.static_obstacle_severity` | `3.0` | — |
| `gnn.encoder` | **`flat`** | `conv` |
| `density.ray_softness` | `8.0` | — |
| `perception.idle_mode` | **`full`** | `cached`, `skip` |
| `perception.idle_refresh_every` | `10` | — |
| `waiting.enabled` | **`False`** | `True` |
| `waiting.max_wait_steps` | `12` | — |
| `waiting.mutual_wait_steps` | `3` | — |
| `waiting.block_threshold` | `2.0` | — |
| `lesion.zero_blocks` | `[]` | block names |
| `ablation.ray_origin_fallback` | `True` | `False` |

Bold = still on the old behaviour, waiting for the retrain.

---

## Changes in build order

### Day 1 — correctness, then compute (5138 → 407 ms/step)

1. **`proximity_warehouse.py`** (new) — one graph-distance index replacing four
   Manhattan copies: `sf_peer_proximity`, braking, safety reward,
   `check_integrity`. Found **two** phantom species — behind racks, and both
   fleets mid-edge on parallel tracks.
2. **Peer projection** → BFS descent on the peer's own `goal_distance_map`,
   replacing `current_pos + direction * k`, which projected through racks and
   aliased on half-cells.
3. **Falloff kernel** → graph hops. 1.49× faster, 3.27% less spurious repulsion.
4. **Tabu masking** → network's ranked legal action, not `random.choice` over
   non-tabu (which ignored structural validity; ~70% walked into racks).
5. **`is_structurally_valid`** → no re-sort, no copy, bisect instead of scan.
   **5.2×.** The aisle line was already sorted at build time and re-sorted
   6.2M times.
6. **`stamp` split** into `stamp_cell` (integer) + wrapper. Round trips down 25×.
7. **Rays** → graph walk capped at `ray_range=25`. **4.5×**, and the cap costs
   no information because the transform pins at 1.0 there anyway.
8. **Frozen-peer discount** scalarised. **20.9×** on that term; flipped the cost
   trend from +45% to −24%.
9. **Idle perception** (`idle_mode`) — 1.68× at 200 fleets, no directional
   behaviour cost at 25. Not adopted for the retrain.

### Day 2 Phase 1 — foundations, all flagged off

10. **`get_local_volume()`** — 2-channel `(2, 11, 11, 11)`: mask and repulsion
    **never multiplied**. The multiply made `0` mean both "no track" and
    "fully contested".
11. **`_build_adjacency_graph()`** — the GAT's attention graph on graph
    distance. Fourth and last phantom instance, and the only one that corrupted
    a **topology** rather than a number.
12. **`ray_transform="smooth"`** — `d/(d+c)`. Under `clip25`, 37 and 29 cells of
    clear aisle both read exactly 1.0.

### Day 2 Phase 2 — in progress

13. **Waiting** — `waiting_nodes`, `sf_is_waiting`, `sf_wait_steps`. Idle
    penalty exempted while blocked, **bounded**, cap depends on blocker type:
    long behind a mover, short in a mutual standoff, symmetry broken by
    remaining graph distance. Dead and parked fleets never register as blockers
    — they aren't blocking anything.
14. **Ray semantics** — 18 dims. Separates five cases that all previously read
    `direction == 0`.
15. **`static_obstacles`** — live set in core, currently empty. Humans and
    debris, one category, no transparency rule. Dimension reserved so adding
    them later costs no retrain.

16. **Gibbs state** — 2 dims. `sf_local_entropy` = Shannon entropy over the six
    neighbours' affordance, normalised by log(LIVE neighbours) so it measures
    DECISIVENESS not degree. `sf_throughput_t` = `1/(1+orders_left/steps_left)`.
    Affordance is now computed BEFORE the base vector in `_perceive`, since the
    entropy is read back inside `get_state_vector`.

17. **Unregistered obstacles wired** — hard veto in `get_valid_action_mask`
    (never drive into a human) plus severity 3.0 in the field, above stopped
    (1.4) and parked (1.0). No near-goal transparency: that discount exists to
    make a corpse invisible to its own rescuer, and nobody's goal is a person.
18. **`encoder_warehouse.py`** (new) — `GatedConv3d`, `DensityEncoder`,
    `FusedEncoder`. Measured: **zero** leak across a rack at any depth against
    0.0075 plain; **14x** more signal along the aisle (0.213 vs 0.015);
    translation invariance exactly 0.0; 43,560 params vs 56,704 flat.

**PHASE 2 COMPLETE.** Set `density.output_mode="channels"` AND
`gnn.encoder="conv"` together -- neither works alone.

**Phase 3:** edge-aware attention carrying path conflict.
**Phase 3:** edge-aware attention carrying path conflict.

---

## Key variables

**Core** — `waiting_nodes`, `_wait_steps`, `_wait_cap`, `static_obstacles`,
`proximity`, `adjacency_metric`, `_lesion_names`, `_idle_mode`

**FleetNode** — `sf_is_waiting`, `sf_wait_steps`, `sf_is_immobile`,
`sf_local_entropy`, `sf_throughput_t`, `get_gibbs_state()`,
`static_obstacles`, `ray_range`, `ray_transform`, `ray_softness`,
`ray_origin_fallback`, `state_layout()`

**Density** — `get_local_volume()`, `action_entropy()`, `kernel_metric`, `projection_mode`,
`_kernel_neighbourhood()`, `_cached_intended_path()`, `_fleet_cell()`

**Counters** — `waits_started`, `waits_capped`, `mutual_waits`,
`phantom_pairs_rejected`, `projection_fallbacks`, `kernel_manhattan_fallbacks`,
`ray_origin_recovered`, `ray_saturated_rate`, `ray_peer_hit_rate`,
`idle_perception_reuse_rate`

---

## Tests — 12 files

`test_proximity` · `test_projection` · `test_loop_equivalence` ·
`test_state_layout` · `test_kernel` · `test_structural_validity` · `test_rays` ·
`test_idle_perception` · `test_profile_step` · `test_conv_leakage` ·
`test_phase1` · `test_encoder` (the only one needing torch)

Run them after **every** file copy. Three restarts on day 1 were stale files;
`core_warehouse.py` and `config_warehouse.py` break each other silently.

---

## Findings to keep in mind

- **6 BFS dims carry the routing.** Lesion at 0.24%: −7.3pts completion, 24/30
  episodes directional, steps 258 → 712.
- **Their grip weakens with density.** Ratio −6.66 at 0.24%, −1.04 at 1.74%,
  **+3.53** at 4.18%.
- **Rays and the field are independent**, ρ ≈ −0.30, invariant across a 17×
  density change.
- **At 4.18% every lesion helps.** Six of six deltas favour blinding. Not an
  artifact — reverting all geometry changes moved completion 0.011.
- **The density block is a mask.** 100% pure structure at 0.25% occupancy, 96%
  at 7.8%. **93 distinct shapes** on a 344-cell map, one covering 49%.
  ~6.5 bits in 7,392.
- **A plain 3D conv loses 9× signal**, not just 1.2% leakage: 27-cell kernel on
  a ~2.9-degree graph. Gating is mandatory.
- **13,041 collapse events** at 4.18%, 25.7% of Tier-1 escapes moving nowhere,
  `[19,49]` colliding 314 times.