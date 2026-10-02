# CONFLICT_DESIGN.md — establishing priority before conflicts, not after

**Goal.** Replace "freeze and rewind after a collapse" with rules that act before
fleets meet: warn only on real path conflicts, let convoys follow, stop head-on
meetings at corridor entrances, and decide who goes first by priority.

Evidence: STREAM_DESIGN.md, sections "cold_run24 results" and "Conflict rules:
evidence for path-based warnings". In brief: 129 collapses per episode, 74%
repeats (worst pair 59 times), every hold at the 30-step cap; the warning zone
(2.0 hops) is wider than a one-cell sidestep can clear; 57% of today's warnings
are convoys sitting 1.5–2 hops apart; the policy waits in ~70% of conflict
choices; throughput plateaus near 35% of the conflict-free ideal.

FLOWRRA principle kept throughout: goals come from the global map; **traffic is
perceived locally** (rays, density channels, neighbours' route intents).

---

## Components (each behind its own switch; all off = today's behaviour exactly)

### 1. Path-based warnings — `conflict.path_warnings`
For every pair within `conflict.radius` hops (default **3**), classify by their
next `conflict.route_horizon` cells (default 8): head-on (each route contains the
other's cell), following (one heads where the other is, the other moves away),
blocked by a stationary fleet, contested (a shared cell reached within one step
of each other), sequential, or no conflict.
- Within `conflict.floor` hops (default **1.0**): always a warning (safety floor:
  routes are predictions).
- Head-on, contested, blocked: warning, and the density splat lands on the
  contested cells only.
- Following at `conflict.follow_gap` hops or more (default **1.5**, the kinematic
  minimum): no warning. Closer: warning.
- Sequential, no conflict: nothing.
Feeds the loop's warning set, so integrity, risk steps and preemption follow.

### 2. Direction-aware braking — `conflict.directional_braking`
Brake on the nearest peer only when closing (the pair is not a steady-gap
convoy). Required for component 1: today's braking holds convoys at 1.5–2 hops.

### 3. Corridor entry — `conflict.corridor_entry`
Single-lane stretches are chains of two-neighbour cells, with a junction at each
end (static). A fleet about to step from a junction into a stretch waits at the
junction if a fleet inside the stretch is heading toward it -- detected through
its own rays (within `max_vision_range`, straight sections) or neighbours' route
intents (bends). Same-direction entry (a convoy) is allowed.
Vision: 10 edges covers ~93% of stretches whole; covering the longest needs ~25
(25_ map) or ~50 (50_ map).

### 4. Priority with aging — `conflict.priority`
Where fleets compete for the same cell or stretch, priority = hops to goal, raised
by time spent waiting, reset on delivery; ties broken at random (seeded). The
higher-priority fleet moves; others wait or step aside. Replaces "closest to goal
wins" applied only after a collapse. (PIBT-style; the strongest published
lifelong systems use a priority shield of this kind.)

### 5. Hop-limited perception — `density.hop_limited_stamps`
Stamp density only on cells within `local_radius` (5) **hops** of the fleet, not
within 5 grid steps, so walls do not create phantom neighbours (on 50_, 23% of
cells within 2 grid steps are more than twice as far in hops). Per-cell hop
neighbourhoods are static: computed once per map.

### Plus: the four measurement fixes (STREAM_DESIGN.md, "After cold_run24")
Count risk steps before the early exit; prevention metric (watched fleets collide
within k steps); randomised preemption test; efficiency in the episode line.

---

## Build order

1. Measurement fixes (so every later change is measured honestly).
2. Components 1 + 2 together (coupled), then 3, then 4, then 5.
3. After each: identity with the switch off (benchmark and stream configs), unit
   tests on hand-built geometries, then the shortest-path driver A/B.

## Pre-registered success criteria (driver A/B, 60 fleets, 50_ map, 300 steps, 3 seeds)

For components 1–4 together, against today's rules:
- **repeat offences** (collapses of a pair already collapsed) fall by at least half;
- **holds** fall by at least half;
- **deliveries** rise on at least 2 of 3 seeds;
- **collisions** do not rise by more than 10%.
Component 5 affects only the learned policy: judged in a training run by the wait
share of conflict choices (today ~70%) and deliveries, paired by episode against
cold_run24.
---

## Status

### Step 1 -- measurement fixes: built (2026-09-29)

| Fix | Where | How it reads |
|---|---|---|
| 1. Risk steps counted before the early exit | core `_policy_recovery_step` | `risk acted/total (rate)` is now a real rate; it read 100% by construction |
| 2. Prevention metric | core `_watch_open/_watch_resolve`; `recovery_policy.prevention_window` (5) | episode line `pre N/strict clear5 X/Y`; CSV `preempt_*` |
| 3. Randomised preemption test | `drive_shortest_path.py --recovery coin` | invoked vs declined opportunities, collisions within k, 25-step block bootstrap |
| 4. Efficiency | core `_stream_efficiency` | episode line `done D/T eff 36% of 403`; CSV `stream_efficiency`, `stream_ideal_*` |

**Criteria counters** (CSV `conflict_*`): collapses, repeat offences (= the
`REPEAT OFFENCE` lines printed), worst pair, holds, holds at the cap, hold steps.

**Fix 2 in detail.** Every preemptive recovery opens an event on the fleets it
moved. The event is judged at the start-of-step checks at ages 1..k (never age
0) and resolves at age k: *collided* if any of those fleets was in a collision
at any check, *clear* otherwise, *pending* if the episode ends first. It also
records how many of the fleets collided. Collisions with any fleet count.

**Fix 3 in detail.** An opportunity is a step with fleets in the warning band
and none colliding (collisions are always recovered anyway). At each one the
coin decides; the whole at-risk set is watched either way (intention to treat).
The estimate is one extra invocation against a background that invokes half the
time; `--recovery always` vs `never` on paired seeds gives the policy-level
comparison.

**Fix 4 in detail.** Each dock order records its conflict-free cycle, dock ->
goal -> nearest dock (hops). Ideal = fleets x steps x base_speed / mean cycle,
over this episode's dock orders, so it follows the order window. 60 fleets, 800
steps, 59.4 hops -> 404.

**Identity** (`check_identity.py`, learning agent, 80 steps, seeds 0 and 1,
benchmark and stream configs, against the code as it was): positions at every
step, rewards, every loss and the final weights byte-identical; every statistic
identical except the three fix 1 is meant to change (`risk_steps`,
`warning_steps`, `intervention_rate`). In the benchmark run, seed 0, the old
counter saw 20 of 80 risk steps.

**Tests:** `test_measurement_fixes.py` (45 checks) plus the existing suite,
17 files, all pass (18 with `test_recovery_charge.py`, below).

### Tools for every later step

- `drive_shortest_path.py --arm today --arm name:key=value,...` runs the
  pre-registered A/B (60 fleets, 50_, 300 steps, seeds 0,1,2 by default) and
  prints each criterion as PASS/FAIL against the first arm. An unknown config
  key is refused, so a typo cannot run the baseline twice under two labels.
- `check_identity.py --old <tree before> --new .` is the switch-off identity
  test.

### Fixed: recovery cost charged twice (2026-09-29)

`_policy_recovery_step` applied its cost-and-outcome block twice in a row. The
first copy is removed (no switch: it was a duplicate, not a choice).

| Recovery head's choice | Before | Now |
|---|---|---|
| Invoke with nobody at risk (wasted) | -4 | -4 |
| Invoke preemptively, strict success | -8 + 9 = +1 | -4 + 9 = +5 |
| Invoke preemptively, no strict success | -8 | -4 |
| Invoke during a collision (resolves it) | -8 + 12 = +4 | -4 + 6 = +2 |
| Forced fallback | +4 | +2 |

Before, asking for help with fleets at risk cost twice what asking for nothing
did, and cleaning up a collision paid better than preventing one. Calibration.md's
"two invocation costs in one step (-8.5)" was this duplicate.

Only the recovery head's reward changes (`safety_includes_recovery` is False, so
the movement heads never saw it). Under the shortest-path driver, which reads no
reward, all 42 run metrics and all 106 watch events are identical to step 1.
`test_recovery_charge.py` checks every path on both ledgers and fails on the old
code with exactly double.

**Consequence for comparisons:** runs from here on are not reward-comparable
with cold_run24 on the recovery head (`qval_recovery`, `loss_recovery`, the
integrity column). Every other head, and the environment itself, is unchanged.

### Components 1 + 2: built (2026-09-29)

`conflict_warehouse.py` (classifier), `conflict.path_warnings`,
`conflict.directional_braking`. Both off = identical behaviour (identity: 4/4
runs byte-identical, learning included).

- The loop's warned pairs are the classifier's; the preemptive filters, the
  splat and every post-action peek read the same pairs. (The filters used to
  re-derive pairs at 2 hops, which would have silently dropped every new
  warning at 2-3 hops from preemption.)
- Splat on the meeting cell (the blocker's cell for a blocked pair).
- Braking reads a snapshot taken before anyone moves: the move loop rewrites
  headings mid-step, so reading them live makes braking order-dependent.
- `test_conflict_rules.py`: corridor, crossing and rack geometries; a convoy at
  1.5 hops now moves 0.5/step (was 0.35), a head-on pair at 1.5 still brakes;
  forwards = reversed with both switches on.

**The 50_ map.** 6,300 cells, 10 floors (Z = 0, 6, ... 54) of 540 cells plus 900
shaft cells. 89.5% two-neighbour cells, 10.5% junctions, no dead ends. Each
floor: two six-lane end blocks (x 0-5, 54-59) joined only by five 46-cell
corridors (y = 0, 6, 12, 18, 24); shafts at x = 5, 6, 53, 54. 710 single-lane
stretches, none bent: 180 shafts (5 cells), 480 block lanes (5-6), 50 long
corridors (46). Vision 10 edges sees 93% of stretches whole; the long
corridors need 47.

**Where cold_run24 collapsed** (the 1,914 first-offence escapes that print
positions): shafts are 14% of cells, 27.5% of fleets involved; end blocks 49%
-> 59%; long corridors 36.5% -> 13%. 23% of sites are shaft mouths.

**Driver A/B** (60 fleets, 50_, 300 steps, seeds 0-2):

| | today | 1 + 2 |
|---|---|---|
| no preemption: deliveries / collisions / repeats / holds | 291 / 121 / 43 / 64 | 288 / 123 / 37 / 60 |
| preempt always: deliveries | 293 | **337** (all 3 seeds up) |
| preempt always: holds at the cap / all holds | 10,345 / 11,755 | **3,195** / 10,606 |
| preempt always: repeats / collisions | 741 / 8 | 676 / 10 |

1 and 2 change what is known, not who goes first: nothing moves when nothing
reads the warnings, and when preemption does, deliveries rise and capped holds
fall by two thirds. Halving repeats and holds is for 3 and 4.

### Components 3 + 4: built (2026-09-29)

`corridor_warehouse.py`; switches `conflict.corridor_entry`, `conflict.priority`,
`conflict.node_aligned_moves`. All off = identical (4/4 identity runs).

- **Corridor index**, static: every single-lane chain between junctions (710 on
  50_, all straight). Straight corridors are seen 10 edges deep, as rays would;
  bent ones through route intents within 3 hops.
- **Entry (3).** A fleet on a junction does not step in when the nearest fleet
  it can see inside is coming at it; convoys may follow. Two fleets at the two
  ends in sight of each other: priority picks one. A fleet waiting ON the
  junction stands on the one cell the oncoming fleet must cross, so when that
  fleet is 2 cells away the waiter pulls over (never onto the oncoming fleet's
  next cell) and comes back after.
- **Priority (4).** Key: hops to goal - 0.25 x steps made to wait (reset on
  delivery), then seniority in the corridor, then a fixed tiebreak. Facing
  inside a corridor: the loser and its followers back out to distinct pull-over
  cells beyond the junction behind, and are released once the winners are
  through (timeout as a backstop). Two fleets claiming the same cell: the
  lower waits a step. A fleet backing out always wins a claim.
- **Once per step, from start-of-step positions,** so order-independent
  (tested forwards = reversed with all switches on).
- **Node-aligned moves.** The rules exposed a movement bug: braking leaves
  fleets out of phase with the grid, so a fleet can jump over a junction at
  every step and never turn there (oscillating forever), or turn slightly off
  the node and be stranded (every move invalid). With the switch, a step never
  overshoots the next node and a turn snaps onto the node. Required by 3 and 4,
  and a substantial change on its own (below).

**Names (decided 2026-09-29).** The code before this design is
**version_unrefined**: a fixed reference for how far things have come. The
components are judged against **version_unrefined + node alignment**
(`aligned`), because they cannot work without it.

**Tests** (`test_corridor_rules.py`, hand-built corridor between two hubs,
shortest-path driver): 6 cells -- without the rules 2 collisions and neither
fleet delivers in 400 steps; with them no collision, both deliver in 51 steps.
30 cells (beyond sight) -- one meeting inside, the fleet farther from its goal
backs out and is released (not timed out), both deliver in 146 steps, no
collision. Suite: 20/20 files.

**Driver A/B** (50_, 60 fleets, 300 steps, seeds 0-2; totals; baseline `aligned`):

| no preemption | deliveries | collisions | repeats | holds |
|---|---|---|---|---|
| version_unrefined | 291 | 121 | 43 | 64 |
| aligned (baseline) | 354 | 172 | 62 | 93 |
| aligned + 1-2 | 353 | 166 | 55 | 94 |
| aligned + 1-4 | **423 (+19%)** | **117 (-32%)** | 46 (-26%) | 54 (-42%) |

| preempt always | deliveries | collisions | repeats | holds |
|---|---|---|---|---|
| version_unrefined | 293 | 8 | 741 | 11,755 |
| aligned (baseline) | 309 | 7 | 770 | 11,492 |
| aligned + 1-2 | 366 | 6 | 666 | 12,132 |
| aligned + 1-4 | **376 (+22%)** | 4 | 643 (-16%) | 7,958 (-31%) |

Pre-registered criteria, 1-4 vs aligned: deliveries up (3/3 seeds without
preemption, 2/3 with) and collisions not up -- **met**; repeat offences and
holds fall, but by 16-42%, not by half -- **not met**. Against
version_unrefined, 1-4 deliver 45% more without preemption (28% more with).

### Where the remaining collisions are (quick check, 2026-09-29)

All rules on, 50_, 2 seeds x 300 steps: 92 new collisions, 90 in the end
blocks, 2 in shafts, none in the long corridors. By kind: a moving fleet
driving INTO A STOPPED ONE at a junction 62%, head-on at a junction 16%,
crossing at a junction 11%, the rest rear-ends. Shafts and corridors are
solved; the open problem is mesh junctions, where the rules make fleets wait
ON junction cells and nothing stops others running into them. (Not a swap
problem, as first guessed.)

### Component 5: already in place -- and the radius stays 5 (2026-09-29)

**Hop-limited perception needed no new code.** Earlier work had made every
part of the field hop-based: the visible cells are a depth-L BFS (the
structure mask), peer stamps spread by hops from each source, trails follow
routes, and the conv encoder gates by the mask. Measured on 50_ (1,800
observations): removing every peer beyond the hop horizon changes **0**
visible cells; the mask never failed open. Phantom geometry exists -- 4.4% of
peers within 7 grid steps are more than twice as far by hops -- but none of
it reaches what the network sees. A `hop_limited_stamps` switch would change
nothing, so none was added.

**The radius** (`radius_study.py`): real 50_ traffic, all rules on; for each
radius a field identical to core's, features summed by hop ring, a logistic
model fitted on seed 0 and scored on seed 1 for "this fleet collides within
K steps".

| radius | cells | input dims | ms/field | AUC, K = 5 | AUC, K = 20 |
|---|---|---|---|---|---|
| 3 | 63 | 315 | 3.2 | 0.925 | 0.868 |
| 4 | 129 | 645 | 3.2 | 0.926 | -- |
| **5** | **231** | **1,155** | **3.3** | **0.927** | **0.888** |
| 6 | 377 | 1,885 | 3.9 | 0.927 | -- |
| 7 | 575 | 2,875 | 4.1 | 0.928 | -- |
| 8 | 833 | 4,165 | 4.4 | 0.927 | 0.889 |

Imminent danger is visible from 3 hops; seeing it coming 20 steps out needs
5; beyond 5 nothing is gained for 3.6x the input. **5 is the knee: keep it**
(and no cold start). A proxy, not the judge: the pre-registered test for 5 is
still a training run (wait share of conflict choices, deliveries, against
cold_run24).

### Yield to stopped fleets: built (2026-09-29)

`conflict.yield_to_stopped` (in `corridor_warehouse.py`). A fleet never moves
into a cell held by a fleet that stays put this step (told to wait, held,
waiting, idle, or unable to move): it queues behind it, and queues propagate
back. A wait is never imposed where it would close a loop of waits -- a fleet
waiting ON a junction for a corridor, and the fleet leaving that corridor
waiting for it -- so the rule cannot deadlock anything; such pairs are left to
recovery and counted. Test: a fleet held on a junction and another routed
through it -- 1 collision without the rule, none with it, both deliver.
Identity with the switch off: byte-identical.

**Driver A/B** (50_, 60 fleets, 300 steps, seeds 0-2, baseline `aligned`):

| no preemption | deliveries | collisions | repeats | holds |
|---|---|---|---|---|
| version_unrefined | 291 | 121 | 43 | 64 |
| aligned (baseline) | 354 | 172 | 62 | 93 |
| aligned + 1-4 | 423 | 117 | 46 | 54 |
| **aligned + 1-4 + yield** | **395 (+12%)** | **11 (-94%)** | **4 (-94%)** | **4 (-96%)** |

**All four pre-registered criteria met** (deliveries up on 3/3 seeds). Against
version_unrefined: +36% deliveries, -91% collisions, -91% repeats, -94% holds.
Queueing costs some throughput against 1-4 alone (423 -> 395): the price of
not crashing.

| preempt always | deliveries | collisions | repeats | holds |
|---|---|---|---|---|
| aligned (baseline) | 309 | 7 | 770 | 11,492 |
| aligned + 1-4 + yield | 387 (+25%) | 0 | 566 (-26%) | 6,230 (-46%) |

Two of four criteria: under constant preemption, recovery's own escalation
drives the repeats and holds. Seed 2 there is identical with and without the
rule -- checked: every wait order is applied (0 of 341 ignored), but 219 fall
on fleets recovery was already holding; that seed locks into recovery holds.

**What is left** (no preemption, 2 seeds x 300 steps): 9 collisions, down from
92 -- all in the end blocks; 7 into stopped fleets (the loop cases the guard
deliberately leaves), 1 head-on, 1 rear-end.

### For the next training run

Switch on all six: `path_warnings`, `directional_braking`, `corridor_entry`,
`priority`, `node_aligned_moves`, `yield_to_stopped` (the radius stays 5; see
component 5). Judge it on the wait share of conflict choices and deliveries,
paired by episode against cold_run24 -- with the amendment below.

**Amended 2026-09-29: judge the policy on what it PROPOSED.** The old measure
(`rwd_choice_*`, "~70% wait" in cold_run24) counts what fleets DID. With the
rules on, every wait they impose reads as the policy choosing to wait: on the
test warehouse with all six switches on, the shortest-path driver proposed
waiting 0% of the time among fleets at risk, and 74% of those decisions ended
as waits -- all 649 overrides the rules'. New columns, for every fleet at risk
at the start of a step: `choice_policy_*` (what the network proposed) and
`choice_exec_*` (what was executed), each wait / toward goal / away / sideways;
`choice_over_rule|hold|other` (who overrode it); and the two wait shares, also
in the episode line as `wait policy X% / done Y%`. The policy's measure is
`choice_policy_wait_share`. (Without the rules the two differ too -- 13%
proposed, 41% done -- through recovery holds and the older overrides.)

### cold_run25, episodes 1-15 (all six switches on; paired against cold_run24)

| episodes | deliveries 24 -> 25 | collisions 24 -> 25 | policy wait / executed |
|---|---|---|---|
| 1-5 | 109 -> 165 | 75 -> 48 | 24% / 70% |
| 6-10 | 126 -> 193 | 54 -> 31 | 19% / 70% |
| 11-15 | 104 -> 184 | 57 -> 22 | 16% / 74% |

Deliveries higher in 15/15 paired episodes (+67.5, t = 10.4); collisions lower
in 15/15 (-28.1, t = -6.2); collisions per 100 deliveries 55 -> 19. Best
episode so far 215 (cold_run24's best in 60: 176). Collisions fall as the
policy learns; the policy proposes waiting less and less on its own.

**Recovery holds are now the biggest override** -- 2,500-5,800 per episode,
three to five times the rules' 700-1,750 -- and episode 14 was a recovery
lockup (3,458 holds, 3,166 at the 30-step cap; 96% of at-risk decisions ended
as waits; deliveries 129). Preemption escalating into capped holds, as in the
always-preempt A/B. The strict preemptive success fell (component 1 widened
the warning set), so the +9 bonus is rarely paid; preemption count held steady
through episode 15 all the same. Next lever: recovery's hold escalation.

**Logging fixes after the run started** (take effect next run): the runner
did not write `corridor_*` statistics to the CSV (missing from its prefix
list) -- fixed; and it now logs the conflict switches at startup.

### Map capacity: is the ideal reachable? (2026-09-29)

Shortest-path driver, all six rules, no preemption unless noted; 800 steps,
order window +-3, seed 0.

| fleets | deliveries | of the no-conflict ideal | per fleet |
|---|---|---|---|
| 5 | 37 | 106% | 7.4 |
| 10 | 59 | 85% | 5.9 |
| 20 | 92 | 69% | 4.6 |
| 30 | 107 | 47% | 3.6 |
| 45 | 117 | 37% | 2.6 |
| 60 | 153 | 36% | 2.6 |
| 80 | 160 | 29% | 2.0 |
| 100 | 183 | 26% | 1.8 |
| 60, preempt always | 173 (19,043 holds) | 38% | 2.9 |

- **The ideal formula is sound**: with light traffic a 0.5-speed fleet
  reaches it (5 fleets: 106%).
- **The map saturates under shortest-path routing** between 30 and 45
  fleets: past that, extra fleets add little. At 60 fleets the realistic
  level for this routing is ~36-38% of the ideal, not 100%.
- **Holds are not needed for safety** (rules alone: 4 holds at 60 fleets),
  **but at saturation holding fleets back raises throughput** (+13% with
  preemption, like ramp metering). Waste is holds that lock up (cold_run25
  episode 14), not holding as such.
- cold_run25 at episodes 11-15 already runs at 40-46% of its ideal (window
  +-1 to +-2, so not the same ideal); the like-for-like check is its episodes
  with window +-3 (from ~30) against the 36-38% above.
- **What would raise the ceiling** is capacity, not conflict handling:
  shortest paths pile traffic onto the same shafts and corridors. Levers:
  one-way conventions (the five parallel corridors per floor and the
  adjacent shaft pairs at x = 5/6 and 53/54 allow up/down and east/west
  lanes); routing that spreads load (the learned policy reading density);
  metering at the docks instead of escalating holds mid-map.

