# FLOWRRA — the simultaneous step

**Status: built and validated,** behind `step.simultaneous`, default `False`.
Paired with the slow congestion channel (`density.slow_channel`) as one fix in
two parts: this aligns everything *within* a step, the slow channel carries
memory *across* steps. See "What building it found" at the end.

Every claim below carries the measurement or the line of code behind it.
Nothing here is proposed from argument alone.

---

## The principle

Every fleet takes **one step together**, and the system is judged over all of
them at once. Within that step:

- **Distinct conflicts resolve independently.** Five separate conflicts on the
  floor are five separate problems. Resolving one must not change the outcome
  of another, unless they are physically close enough to interact.
- **Entangled conflicts resolve jointly.** Fleets that affect each other are one
  problem, and are solved as one, at once.

Formally: build a graph whose nodes are fleets and whose edges are conflicts.
Each **connected component** is one cluster. Clusters are resolved jointly
inside, independently of each other, all in the same step.

This is the intent behind `check_integrity()` taking every fleet in one call.
The per-fleet loop that applies actions today is a cost-saving shortcut that
departs from it.

---

## What was measured

Same scenario, same seed, and every fleet given a **fixed, order-independent
action** (a step down its own goal gradient), so the only thing that differs
between the two runs is the order the fleets are processed in.

| after | fleets in a different place, forward vs reversed | same goals claimed? | collisions |
|---|---|---|---|
| 1 step | **2 / 40** | no | 1 vs 1 |
| 5 steps | 5 / 40 | no | 5 vs 5 |
| 40 steps | 5 / 40 | no | 40 vs 40 |

After a single step:

```
f13   forward  7.90   reversed  9.05
f29   forward 19.05   reversed 17.90
claimed only forward: n_19_0      claimed only reversed: n_9_6
```

**A simultaneous step would give 0 / 40 at every horizon.** Anything else means
the outcome depends on where a fleet happens to sit in `self.nodes`.

**Size, honestly -- corrected.** I first wrote that the divergence "settled at
5 of 40 and did not grow." That was one seed. Across seeds it ranged from 5 to
**12 of 40**, so the leak was larger than first claimed. The test
fixture has 35 unique goals for 40 fleets, so goal contention there is
artificial. The real scenarios have unique goals (`missions 50 | unique goals
50` on Seed0), so on `25_5_2_5_2_1` the leak runs through cell contention,
rescue-inherited goals and mutual waits, and is probably smaller. It is not
what holds completion near 0.87. It is a correctness problem, not the ceiling.

Reproduce with `measure_order_dependence.py`.

---

## Where the order leaks in

Verified in `core_warehouse.py`. The movement loop runs **L2600–3297, 697 lines**,
and interleaves movement, rewards, goal claiming, waiting, rescue servicing and
despawn in a single pass over `self.nodes`.

| site | reads | order-dependent? | effect |
|---|---|---|---|
| goal claim, `claimed_goals.add` (L3082) | set mutated **inside** the loop | **yes** | when two fleets want one goal, the one processed first claims it; the other sees it taken and retargets, which changes its gradient and its movement |
| mutual-wait test, `_blocker in self.waiting_nodes` | set mutated **inside** the loop | **yes** | whether a stand-off is mutual, and so capped at 3 steps rather than 12, depends on which fleet was processed first |
| recovery Tier 1, `for node in tier_nodes` | positions updated as it goes | **yes** | each fleet picks an escape cell seeing the previous fleets' *new* positions; the order is what stops two fleets taking one cell, so it is load-bearing but still arbitrary |

---

## Already consistent with the principle

| stage | how |
|---|---|
| action choice | `choose_actions()` is called **once** on one feature snapshot |
| braking | reads `proximity.nearest()`, refreshed at the start of the step |
| waiting: which peer is the blocker | `proximity.peers_within()`, the same snapshot |
| collision reward | `attribute_collisions_post_action`: judged in a second pass after every fleet has moved |
| warning splats | `warning_splat_per_pair`: one splat per real pair, local to that pair (see below) |

---

## The design: a two-phase step

**1. Propose.** Every fleet computes its next position from the same snapshot:
chosen action, braking, livelock and tabu overrides. Nothing is written back.

**2. Detect.** Build the conflict graph over the *proposals*, using the edges
defined in the next section.

**3. Partition.** Connected components, by union-find. Each component is one
cluster.

**4. Resolve.** A singleton — a fleet in no conflict — commits its proposal
unchanged. A multi-fleet component is solved jointly (next section but one).
Components never see each other.

**5. Commit.** Write every position at once.

**6. Judge.** Integrity, rewards and splats over the committed configuration.

---

## Conflict edges

Two fleets `i`, `j` are joined when their proposals interact:

| edge | condition |
|---|---|
| **same cell** | both proposals round to the same grid cell |
| **swap** | `i` proposes `j`'s current cell and `j` proposes `i`'s — a pass-through on one edge |
| **overlap** | proposed positions within `collision_threshold` (graph distance), which catches mid-edge overlaps the rounding misses |
| **goal contention** | both proposals would claim the same goal this step |

A convoy is **not** an edge. Two fleets on the same axis, same direction, gap
not closing, propose different cells and never trigger any of the four.

---

## Resolving a component

Joint assignment of fleets to cells: the "Hungarian Tier 1" item from the open
list, generalised from recovery to the whole step.

- **Candidates per fleet:** its proposed cell, its current cell (stay), and its
  other structurally valid neighbours.
- **Cost:** increase in graph distance to goal, plus a small penalty for
  deviating from the proposal, so the policy's choice is honoured wherever the
  geometry allows it.
- **Constraints:** at most one fleet per cell; no swap pairs. The assignment
  solver does not forbid swaps on its own, so a solution containing one is
  re-solved with one of the pair fixed in place.
- **Solver:** `scipy.optimize.linear_sum_assignment` on the fleet × candidate
  matrix. Components are small, so O(k³) is cheap.
- **Ties** are broken by something that does not depend on list order: the
  `arrival_order` edge feature, then distance to goal, then a stable hash of the
  fleet id. **Never** position in `self.nodes`.

---

## Where each concern moves

| concern | today | under the simultaneous step |
|---|---|---|
| action choice, braking, livelock, tabu | in the loop | phase 1, shaping the proposal |
| movement | applied fleet by fleet | phase 5, all at once |
| goal claims | first processed wins | goal contention is an edge; resolved in phase 4 |
| mutual waits | reads a set mutated in the loop | decided from the snapshot in phase 1 |
| rewards | in the loop, plus a post-action pass | phase 6 only |
| recovery Tier 1 | sequential escape | the same component resolver, applied to deadlocked clusters |
| rescue servicing, dwell, despawn | in the loop | after commit, as per-fleet state updates |

---

## Acceptance tests

Each is falsifiable, and the first one fails today.

1. **Order invariance.** Forward versus reversed fleet order gives identical
   positions, claimed goals, collisions and reward matrix at every horizon.
   Today: 2 / 40 fleets differ after one step. Required: 0.
2. **Independence.** Adding or removing a conflict far away leaves every other
   component's resolution unchanged.
3. **Joint correctness.** No committed configuration has two fleets on one cell,
   and no swap.
4. **Behaviour.** A single-variable A/B on `25_5_2_5_2_1` against the cold_run13
   baseline, same seed, paired episode by episode.

---

## Cost and risk

- **The loop is 697 lines** carrying six concerns at once. Separating movement
  from its side effects is the real work, not the solver.
- **Behaviour will change**, so it needs its own run and must not be bundled with
  any other change.
- **Compute is not the concern.** Most fleets are singletons and commit their
  proposal unchanged; components are small.
- **Flag-gated**, default off, so the current step remains reproducible.

---

## Related changes already made

Both are consistent with the principle, and both are in cold_run13.

**`density.warning_splat_per_pair` — conflicts get their own repulsion.** The
old code splatted at the midpoint of `list(warning_nodes)[0]` and `[1]`, the
first two elements of an unordered set, once per step however many conflicts
existed. With two genuine conflicts and one convoy placed on the map at once:

| mode | splats | landed at |
|---|---|---|
| old | 1 | **empty floor, near no pair** |
| per pair | 3 | conflict A, conflict B, the convoy |
| per pair, convoys skipped | 2 | conflict A, conflict B |

Under the old code, both real conflicts got nothing. Each splat now stays local
to its own pair, which is the "distinct conflicts don't affect each other" half
of the principle already in place for splats.

**`recovery.preempt_skip_convoys` — convoys are not conflicts.** On cold_run12,
742 of 769 recovery invocations (96%) were preemptive, with no collision, and
only 148 of those (20%) separated anyone. Reading the coordinates showed why:
two fleets on the same route about a cell apart, down the x=5 shaft and along
the corridor, permanently inside `warning_threshold` and re-triggering recovery
every step. `_is_following()` exempts a pair when both moved along the same axis
and direction and the gap did not shrink; head-on, crossing, a leader idling
with a follower closing, and every real collision still go to recovery. It
classified all six test geometries correctly, and on a hand-built convoy the
flag turns a teleport of both fleets into no intervention at all.


---

## What building it found

**Acceptance test 1 passes.** `measure_order_dependence.py --simultaneous`:

| seed | flag off | flag on |
|---|---|---|
| 3 | 12 of 40 diverge | **0** |
| 7 | 5 of 40 diverge | **0** |
| 11 | 9 of 40 diverge | **0** |

With the flag on, positions, claimed goals, collision counts and total reward are
identical whichever way round the fleet list is processed. With it off, the
two-phase code reproduces the original single loop **bit for bit** over 60 steps,
so the restructure itself changes nothing until the flag is turned on.

**The loop was split, not rewritten.** Movement never reads another fleet's
position, so phase A (decide and move) was already order-independent. Only four
values cross into phase B: `_moved`, `base_speed`, `old_dist`, `old_pos`.

**There were five leaks, not three.** The design listed goal claims, mutual waits
and recovery Tier 1. Building it found two more:

| leak | how it depended on order | fix under the flag |
|---|---|---|
| goal claims | first fleet processed claimed a contested goal | closer fleet wins, ties by id |
| mutual waits | read a set being filled during the same loop | read a snapshot |
| recovery escape order | escapes followed `self.nodes` | fixed order by fleet id |
| recovery winner | a *stable* sort by distance kept list order on ties | tie-break by id |
| tabu fallback | `random.choice` on the global generator, drawn fleet by fleet | a generator seeded by fleet and step (`zlib.crc32`, since `str` hashes change per process) |

Order-dependent random *samples* are distinguished from order-dependent
*decisions*. Epsilon exploration draws fleet by fleet too, but every fleet still
gets a fair independent draw, so it biases nothing and is left alone. Decisions
that favour whoever comes first are what the flag removes.

**A leftover in-loop refresh, removed unconditionally.** A proximity refresh that
fired once inside the loop, after the first fleet moved, meant every later fleet
braked against an index where fleet 0 had moved and they had not. Its output was
discarded anyway. It had been active in cold_run10, 12 and 13.

**A safety bug the refresh had been hiding.** Removing it changed trajectories,
and one new trajectory put a fleet into a collision beside an obstacle. Recovery
Tier 1 then **teleported it onto the obstacle**. Recovery bypasses the action
mask, takes candidates from `G.neighbors()` (obstacles are overlaid on the graph,
not removed), and relied on the soft affordance score to steer away. Tier 1 and
Tier 2 now reject obstacle cells outright, unconditionally, since for a human
this is exactly what the obstacle work exists to prevent. It cannot affect
earlier runs, which all had obstacles disabled. Counted as
`tier1_rejected_obstacle` and `tier2_rejected_obstacle`.

**The judge phase fixes the reward/splat lag.** Collisions and warnings are now
splatted after every fleet has moved and before `next_s(t)` is perceived, so a
fresh collision's penalty and its visible trace land in the same transition.

---

## Part two: the slow congestion channel

`density.slow_channel` adds a third density input beside mask and repulsion.

| | fast memory | slow memory |
|---|---|---|
| decay per step | 0.7 | 0.987 |
| half-life | 1.9 steps | 53 steps |
| a 1.5 splat after 10 steps | 0 (gone) | 1.316 |
| after 53 steps | 0 | 0.750, exactly half |

Verified: repulsion is **bit-identical** with and without the slow memory, so it
lives only in its own channel. The encoder infers its channel count from the
state width and rejects a mismatch loudly. Both memories decay independently —
the old `step_decay` began with an early return on an empty fast memory, which
would have frozen the slow one permanently on any quiet step.

**It changes the state size** from 83 + 462 to 83 + 693 = 776, so it needs a
**fresh cold run**; checkpoints trained without it cannot be loaded.

With both flags on, a full episode learns with finite losses, the step stays
order-invariant (0 of 40), and at episode end the slow channel still remembers
42 congested cells where the fast one remembers 6.


---

## Round three: the snapshot recovery was missing

**Recovery moved fleets and nobody re-took the snapshot.** The proximity index
is refreshed once at the top of the step, and the network's nearest-peer input
(`sf_peer_proximity`) is written from it, both *before* recovery runs. Recovery
teleports fleets and never refreshes the index, so perception, braking and the
waiting check all read positions from before the teleport. Present in every run
up to cold_run14.

Measured at the point the network reads it, for fleets recovery had just moved:

| | input says "peer on top of you" | truth | input matches truth |
|---|---|---|---|
| before the fix | 100% | 25% | 1% |
| `step.simultaneous` on | 17% | 17% | **100%** |

Fix, behind the flag: snapshot positions before recovery; if recovery moved
anyone, refresh the index and rewrite `sf_peer_proximity` before perception.
`sf_in_warning` and `sf_in_deadlock` are left alone on purpose -- whether they
mean "in conflict now" or "involved in one this step" is a design decision.

**The leftover refresh is restored, behind the flag** -- present when
`simultaneous` is off, absent when on. Flag-off now reproduces the true
cold_run13 code. Checked on five seeds at 50 fleets where the refresh
*changes the outcome* (the true original and a refresh-removed version
disagree in all five): the new code matches the true original in all five.

An earlier "bit-for-bit" check was weaker than it looked: in that scenario the
refresh had no effect at all, so it would have passed even if the restoration
were wrong. Only scenarios where the refresh matters can test it.

**A correction.** To restore the old timing, the first fleet's waiting check is
given neighbours read from the unrefreshed index. I first claimed this was
generally necessary. A mutation test showed otherwise: removing it still
matched in all five seeds. It matters only in a narrow case -- recovery
teleporting a fleet near the first fleet on a step that fleet stands still --
because waiting only consults its neighbours when the fleet did not move. It is
kept, since it reproduces the old read by construction rather than by luck.