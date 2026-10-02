# STREAM_DESIGN.md — an order stream with despawn and respawn

The next phase: fleets leave the floor after delivering, re-enter for new work,
and episodes become a stream of orders instead of a fixed set of missions. First
run on map 50_5_5_10_5_2 with 60 fleets (scenarios `all_scens_v4`).

Reference for comparison: `config_benchmark_cold_run23.py` (stream off).

---

## The map

A ten-storey building. Full floors of 540 cells at Z = 0, 6, 12, … 54 (6,300
nodes, 6,910 edges, all two-way). Between floors, **20 single-lane lift
shafts**, 6 steps per floor. Z is vertical: only 1,080 edges run along Z, against
2,950 along X and 2,880 along Y.

Goals and starts are spread over all ten floors (13 of the 150 bank goals, and 14%
of scenario starts, sit inside lift shafts). Only 11% of scenario missions stay on
one floor, so most missions already ride the lifts.

## Decisions

| question | decision | why |
|---|---|---|
| Where fleets leave | **Both designs, one setting.** `exit_floors`: `"all"` (docks on every floor's perimeter) or `"ground"` (docks on the lowest floor only) | Both exist in real buildings: ramp-up multi-storey warehouses and edge conveyors or vertical lifts give every floor its own docks; single-dock-level buildings send everything through internal lifts. From a goal, the nearest exit averages 6.6 hops (max 28) with per-floor docks, and 37.7 hops (max 79) with ground-floor docks, nearly all through the 20 lift shafts. |
| How many docks | `exits_per_floor` (default 6), spread evenly along each floor's perimeter | Real buildings have a handful of docks per floor, not every perimeter cell. Fewer docks: realistic queueing; more: shorter exit legs. |
| Which first | *To be chosen* | Recommended: per-floor first (traffic like today's, so the stream mechanics are tested on familiar ground), then ground-floor docks (lift logistics). |
| Where they re-enter | **The exit cell they left from**, once it is free | "Respawn from there again." |
| New orders | **Dock orders from the goal bank**, joining the shared pool | The first 60 fleets start on scenario pairs (re-arrangements under way); every later order is a dock order. |
| Re-arrangement orders (rack A → rack B) | **Later**, as two-leg missions | Scenario files are their natural source. |
| Parked fleets | `frozen_obstacle_severity: 0.0` | The false alarm, and a safe fallback for any fleet that cannot leave. |
| Injected errors | **Off for the first run** | Broken fleets, rescues and recalls interact with despawning; they deserve their own test. |

---

## Mechanics

**Exits.** Full floors are detected automatically: levels holding at least half
the cells of the largest level, along the vertical axis (the axis with the fewest
edges, or set explicitly). A floor's perimeter is its cells at the minimum or
maximum of either horizontal axis (166 cells per floor on 50_). The
`exits_per_floor` docks are spread evenly along it, on every floor or on the
ground floor only (`exit_floors`). The existing design stores one distance map
per exit, which grows heavy with many docks, so **one multi-source distance
map** to the nearest dock is used instead, together with which dock is nearest
from each cell -- the same code serves either layout.

**Life cycle.** A fleet delivers (paid at the goal), then drives to the nearest
exit -- fully active traffic, as the existing despawn already does -- and leaves
the floor on reaching **any** exit cell. It re-enters at that cell at the start
of a later step (delay configurable, default 0; the cell must be free) under a
**new id** (`f12#2`: fleet 12's second life). Fourteen per-fleet dictionaries in
core, plus loop and recovery state, are keyed by id; a new id means nothing from
a previous life can leak into the next.

**Orders.** Each episode has a seeded order queue drawn from the goal bank, so
every run faces the **same sequence of work** even though different fleets pick
it up -- this preserves the controlled comparisons. Each re-entry adds the next
queued order to the shared pool, and the existing matching assigns fleets to
pool goals. A location can recur as a later order after an earlier one to it has
been delivered: claims are per order, not per location.

**Learning.** Leaving the floor ends a life: `next_active` is 0 on the step a
fleet reaches an exit (it is collected at the start of the next step). Removals
and re-entries both happen at the **start** of a step, before the "before" state
is built (core ~line 3079), and nothing is added or removed until the "after"
state (~line 4103) -- so every transition stays aligned fleet by fleet.

**Episode end.** Only the horizon. `is_episode_over()` would otherwise fire if
the floor were empty for an instant.

**Metrics.** Completion rate stops meaning anything. Instead, per episode:
deliveries, orders issued, exits, re-entries, fleets on the floor, mean order
cycle time, mean exit-leg time, and collisions per 100 deliveries. The episode
log line and the CSV (new `stream_` prefix) carry them.

---

## First-run settings

- Map 50_5_5_10_5_2, `--agent-sets 60`, `--scens-dir all_scens_v4`.
- Buffer about **15,000** (the same memory as 20,000 at 46 fleets). At ~800
  steps per episode that is a ~19-episode buffer; cold_run23's schedule (below
  0.05 from episode 30) still leaves 31 clean episodes.
- Value scales unchanged; this run is also the **calibration run** for the new
  map, fleet count and reward flow (CALIBRATION.md §1). Re-measure afterwards.
- Everything else as cold_run23.

## Tests before any run

1. **Identity:** stream off, benchmark config → learning and statistics
   identical to the last digit.
2. **Exits on 50_:** 10 floors, Z detected as vertical, the configured docks per
   floor spread along each perimeter, for both `exit_floors` settings.
3. **Life-cycle accounting** on a real map: exits = lives ended = re-entries
   (delay 0); ids unique; no fleet id reused.
4. **Alignment:** every stored transition has the same fleets, in the same order,
   before and after.
5. **Terminal:** `next_active` is 0 exactly for fleets leaving that step.
6. **Order determinism:** the same seed gives the same order sequence.
7. The full suite.

## Later

Two-leg re-arrangement orders; batteries and
charging stations (built on leave-and-return); injected errors with despawn;
instance difficulty for streams.

---

## Implementation and verification (built and tested before any run)

**Code.** `core_warehouse.py`: stream setup, per-floor docks and the single
nearest-dock map (`_init_stream_exits`), order queue (`_issue_order`), re-entry
under a new id (`_stream_maintenance`, `_new_fleet`), departure book-keeping
(`_stream_left`), terminal on leaving (`_stream_leaving_ids` feeding
`next_active_mask`), horizon-only episode end, `stream_` statistics. Every path is
gated by `CONFIG["stream"]["enabled"]`. `main_runner_warehouse.py`: passes the
whole goal bank and an order seed of `seed × 1000003 + episode` (independent of
the instance generator, so instance draws are unchanged); the episode line counts
deliveries of orders issued; a live `stream |` line; `stream_` columns in the CSV.

**A latent bug, found by testing.** The first long test reported "delivered 129
of 123 orders": every fleet claimed twice per life, because reaching its **dock**
counted as a delivery -- the arrival check did not exclude fleets driving out. The
original despawn code had the same flaw and had never run with despawn on. Fix: a
guard so fleets driving out never claim. (It also restarted the exit-leg timer,
which had made exit legs look like 3 steps; the true figure is about 24.)

**Tests, all passing:**

| test | result |
|---|---|
| Identity: stream off, benchmark config | loss, recovery loss, weights and all 217 statistics identical to the old code (re-run after the guard) |
| Docks on 50_ | Z detected as vertical; 10 floors; 60 docks; all 6,300 cells reach a dock |
| 300 steps, 60 fleets, real runner, shortest-path actions | 74 deliveries of 123 orders; claims = deliveries; no life claims twice; orders = 60 + 63 re-entries; exits (63) = re-entries + off floor |
| Alignment | every transition has the same fleets before and after |
| Ids | 123 distinct ids = 60 + 63; no id reappears after leaving |
| Terminal | 63 life-ending events = 63 exits, falling exactly on the leaving fleets |
| Order queue | same seed → same sequence; different seed → different; of 300 orders, none issued while its location was outstanding; 137 of 150 locations used |
| Learning with varying fleet counts | transitions of 54–60 fleets (5-step re-entry delay), aligned; learning ran with sensible head losses |
| Suite | all 16 pass under the benchmark config **and** under the stream config (`test_obstacles.py` now holds the stream off: it tests the original per-exit despawn) |

## First stream run

**Smoke test first:** `--episodes 1 --max-steps 100 --out smoke24`. At startup:
- `[Core] Stream: 10 floor(s) on axis Z, 60 dock(s) (6 per floor), 6300/6300 cells reach a dock.`
- `stream: enabled=True exit_floors=all docks/floor=6 respawn_delay=0 | errors=False | frozen_obstacle_severity=0.0`
- `input_dim=1239`
- (With `--episodes 1` the exploration line reads 0.010 throughout; that is expected.)

After the episode, a `stream | delivered N of M orders | exits … re-entries …`
line, and `stream_` columns in the CSV. Check that `all_scens_v4/50_5_5_10_5_2/`
holds the 25 scenario files and the goal bank.

**Launch:**

    python main_runner_warehouse.py --maps-dir all_maps --scens-dir all_scens_v4 \
      --maps 50_5_5_10_5_2 --episodes 60 --agent-sets 60 --max-steps 800 \
      --seed 0 --out cold_run24 2>&1 | tee cold_run24.log

Expect roughly 16 hours (cold_run23's 12, scaled to 60 fleets). Read the real
figure from the timestamps of consecutive episode lines. (Correction: `left …h`
in the episode line is NOT hours remaining -- it is the mean distance, in hops,
of unfinished fleets from their goals.)

**What to watch** (this is also the calibration run):
- deliveries per episode, and collisions per 100 deliveries, over the run;
- `stream_on_floor_mean` near 60, `stream_offfloor_at_end` near 0, and
  `stream_reentry_wait_mean` (queueing at docks);
- `stream_life_steps_mean` and `stream_exit_leg_steps_mean`;
- `qval_delivery` at or below 1.42 (one delivery per life);
- afterwards, re-measure the value scales from this run (CALIBRATION.md §1).

---

## Finding before launch: the lift shafts jam (recorded after the smoke test)

**Logging.** Stream runs now print `[Stream] Delivery #N (of M orders so far): …`,
`[Stream] Fleet … left the floor at dock … (life k ended after … steps; …)` and
`[Stream] Fleet … entered at dock …, heading for …`. Checked over 500 steps:
76 delivery lines for 76 deliveries, numbered 1–76 without a gap; one departure
line per exit (65) and one entry line per re-entry (65).

**The jam.** With shortest-path driving (no yielding -- a harsh test driver), the
stream **gridlocks**: deliveries per 50 steps went 36, 20, 6, 7, 3, 2, 0, 1, 1, 0;
the last delivery came at step 408; recovery fired 476 times, with collapse
groups of up to 36 fleets, and almost no collisions (fleets locked, not
crashing). At step 300: 17 of 60 fleets inside the single-lane two-way lift
shafts (about twice the shafts' share of the building), 6 shafts holding two
fleets (10 heading up, 7 down: head-on locks), 26 fleets waiting on others, and
45 of 60 needing another floor. Recovery moves concentrated 4× at or beside shaft
columns.

**Density test.** 40 fleets: deliveries per 50 steps 20, 20, 6, 10, 7, 9 -- 72 in
300 steps, against 74 with 60. Throughput is capped by the shafts (~0.2
deliveries per step) whatever the fleet count; at 60 they lock completely.

**Why the stream, not the map.** In fixed-mission runs the floor empties as fleets
retire (explore_90_1 ran 90 fleets here at 0.900 completion). A stream keeps 60
fleets moving forever, most of them needing the shafts.

**Launch was on hold** until the shaft question was decided. Decision: option 1,
refined -- a widening floor window for new orders.

## The lift shafts, and the order window

**What a shaft is on this map.** Each of the 20 shafts is a column of 55 cells
from floor 0 to floor 9: the 5 cells between each pair of floors connect only up
and down (all 900 have exactly two neighbours) -- a single-lane corridor with no
passing -- and the cell where the shaft meets each floor is a junction (3–6
connections). Head-on locks form in those 5-cell stretches. The floors
themselves are 88% single-lane corridor cells.

**The window.** A new order's goal lies within `w` floors of the dock's floor
(goals inside a shaft count as their nearest floor). `w` grows over the run:
`stream.order_floor_window = {start 1, end 3, reach_frac 0.5}` gives ±1 for
episodes 1–8, ±2 for 9–23, ±3 from 24 -- finished before exploration drops below
0.05 at episode 30, so the judged episodes see one steady mix of work
(cold_run23's lesson). The initial 60 missions still come from the scenario
files, unrestricted.

**Tested with the harsh shortest-path driver (300 steps):**

| setup | deliveries per 50 steps | total | waiting at end |
|---|---|---|---|
| unlimited orders, 60 fleets | 36, 20, 6, 7, 3, 2 | 74 | 26 |
| **±1, 60 fleets** | 35, 25, 23, 11, 6, 7 | **107** | 28 |
| ±3, 60 fleets | 34, 29, 13, 8, 8, 4 | 96 | 33 |
| ±1, 40 fleets | 20, 20, 10, 14, 7, 2 | 73 | 12 |

Every issued order respected its window (mean floor gap 0.55 at ±1, 1.43 at ±3).
The window gives 30–45% more deliveries and halves the fleets stuck in shafts,
but every setup still decays under a driver that never yields: floor corridors
lock too. That driver has reached its limit as a judge -- yielding (wait, follow,
reroute) is exactly what training must teach, and in a stream a jam must be
resolved rather than dissolving as fleets retire.

**Jam detector.** `stream_deliv_q1…q4`: deliveries in each quarter of the episode
(checked against a step-by-step recorder: 35/25/23/11, exact). Late quarters
collapsing toward zero mean a jam set in mid-episode. Shown live as
`by quarter a/b/c/d`.

**Logging.** Every delivery line now ends in 🔥 (stream, despawn and retirement
lines alike). Identity re-proven after these changes (loss, weights and all 217
statistics unchanged under the benchmark config); suites pass with the stream on.

## Launch, and when to stop

Launch as above (`--out cold_run24`). Watch the `stream |` line:
- **by quarter:** healthy if the last quarter stays a reasonable share of the
  first; a collapse to near zero in most episodes means jams are winning;
- deliveries per episode rising over the run; `waiting` and recovery counts;
- `orders within ±w floors` stepping 1 → 2 → 3 at episodes 9 and 24.

**Stop and reassess** if, after about 10 episodes, most episodes' last quarter is
near zero and not improving. Then the next step is structural: one-way shafts
(10 up, 10 down), or a jam-breaking rule in single-lane corridors.

---

## The docks sit inside corridors (found after cold_run24's first 5 episodes)

**The 0.987 run's first five episodes** (seed 0): deliveries 78, 80, 125, 109,
117, with shortest-path agreement rising 0.48 → 0.62 -- it is learning. The flow
is front-loaded (e.g. quarters 57/23/22/15: the first quarter carries the 60
initial missions); collisions 70–95 per episode; exit legs 74–104 steps for docks
about 7 hops away.

**Finding.** All 60 docks sit inside two-way corridors (every dock cell has exactly
two neighbours): a fleet on a dock blocks the corridor, and one re-entering
beside other traffic starts a head-on standoff. Catchments were uneven (one dock
nearest to 236 cells; median 43).

**Fix (behind switches).** `exits_per_floor: 12`; `reentry_gate`: re-enter only
where the dock and all its neighbours are free and no fleet driving out is within
`reentry_clear_hops` (3) of it; `reentry_any_dock_on_floor`: the least-busy such
dock on the floor, own dock first. New statistics: `stream_reentry_into_crowd`,
`stream_reentry_other_dock`, `stream_reentry_deferrals`.

**Tested honestly (harsh driver, 300 steps, three seeds):**

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| deliveries, old (6 docks, no gate) | 107 | 106 | 93 | 102 |
| deliveries, fix (12, gated) | 86 | 113 | 95 | 98 |
| exit leg (steps), old | 25.2 | 27.3 | 25.4 | 26.0 |
| exit leg (steps), fix | 21.4 | 22.6 | 21.7 | 21.9 |

The fix does what it targets -- exit legs ~16% shorter on every seed, no
re-entry into a crowd, no re-entry delays -- but does **not** raise throughput
(differences −21, +7, +2: noise). Under the old design only 3 of 93 re-entries
met a crowd. So the dock standoff was real but small; the mid-episode decay comes
from corridor and shaft jams. Kept as a correctness fix. The real run's long exit
legs most likely reflect the young policy's general inefficiency (lives of
200–300 steps against ~75 for the driver).

Identity re-proven; suites pass with the stream on.

**Decay comparisons must be re-baselined.** The 0.966 and 0.987 runs so far used
the old docks. On the new docks, run each decay setting for at least 5 episodes
(same seed), pair by episode, and compare deliveries, the last three quarters,
collisions and exit legs -- not completion rate.

## Slow-memory decay, measured (before the long run)

The slow channel is a hazard memory: warnings (0.6) and collisions (1.5) stamp
it with max(), and it decays by `slow_decay_factor` per step. Because the test
driver never reads the density channels, one run's traffic is identical under any
decay, so one recording (3,769 stamps, 300 steps, 60 fleets) was replayed under
each value. Per fleet-step: an *alarm* is a hazard ≥ 0.3 on its next 5 route
cells; *stale* if no other fleet is there and no hazard recurs there within 20
steps; *anticipates* = of recurring hazards with no traffic there now, the share
already shown.

| decay | half-life | alarms | stale | anticipates |
|---|---|---|---|---|
| 0.987 | 53 | 47% | 8% | 42% |
| **0.9786** | **32** | 46% | 6% | 41% |
| 0.9764 | 29 | 45% | 6% | 40% |
| 0.966 | 20 | 40% | 6% | 29% |
| 0.955 | 15 | 36% | 6% | 20% |
| 0.944 | 12 | 34% | 6% | 14% |

Shortening to ~30 steps cuts stale alarms by a quarter at almost no loss of
anticipation; shorter only loses anticipation. **Chosen: 0.9786.** (One instance,
harsh-driver traffic, threshold and horizon chosen by us; the shape is what
matters.)

---

## After cold_run24: measurement fixes (agreed at episode 44; the run continues untouched)

Found at episode 44: preemptive recoveries succeed only ~2% of the time
(typically 55–90 per episode, 0–5 successes), and the risk counter reads exactly
100% acted in every episode.

1. **Count risk steps before the early exit.** In `_policy_recovery_step`, a head
   choosing mode 0 returns *before* the risk step is counted, so only risk steps
   where the head acted were ever counted: `risk_steps_acted / risk_steps` is 100%
   by construction (cold_run22's logs show it too). Count every risk step first,
   so the intervention rate is real.
2. **Add a prevention metric.** Whether the watched fleets collide within k steps
   of a preemptive recovery, alongside the strict success (every moved fleet clear
   of the warning band one step later -- nearly impossible in single-lane
   corridors, so the +9 bonus almost never fires and preemption looks like pure
   cost to the head: its value fell from −0.03 to −0.22).
3. **Randomised test, then possibly redefine success.** With the test driver, at
   each risk step invoke or not at random (50/50) and compare collisions among the
   at-risk fleets over the next few steps -- a causal measure of what preemption
   prevents. If it helps, redefine success as "no collision among the watched
   fleets within k steps", so the head is rewarded for what preemption is for.
4. **Show efficiency in the episode line.** Stream completion is deliveries ÷
   (deliveries + orders in flight), and ~49 orders are always in flight at the end,
   so even a perfect policy would score only ~0.89 (150 deliveries → 0.75). Report
   deliveries against the conflict-free ideal instead (~403 per 800 steps with 60
   fleets on 50_: 119-step ideal cycle): at episode 40, 146 on average = 36%, best
   episode 176 = 44%.

---

## Pre-registered at episode 44: does the target-copy rhythm shape performance?

**Mechanism.** The target network is hard-copied every 1,000 learning steps
(`--target-sync 1000`); learning runs once per environment step from global step
64, so copies land at global steps 1,063, 2,063, … Episodes are 800 steps, so the
copy's position within an episode cycles with period 5: episodes ≡ 1 (mod 5) get
a copy at step 63, ≡ 2 at 263, ≡ 3 at 463, ≡ 4 at 663, and **multiples of 5 get
no copy at all** (targets fixed for ~1,200 consecutive steps).

**Evidence so far (post hoc -- hypothesis formed after seeing episode 40).**
Episodes 24–44 (window ±3): no-copy episodes 25, 30, 35, 40 delivered 154.2 on
average against 136.3 for copy episodes (+18.0; permutation p = 0.016). Against:
episodes 2–23 show the opposite (107.5 vs 128.3; confounded by the exploration
peak at episode 15 and window changes); and no dose-response (copy at step 63:
144.5; 263: 136.8; 463: 131.8; 663: 133.0 -- a late copy should disrupt less).
Neither the initial missions (|r| ≤ 0.2 with deliveries) nor the order queues
(expected cycle 59.4 ± 0.8 hops across all episodes) explain episode-to-episode
swings.

**Prediction for episodes 45–60 (not yet run).** The no-copy episodes 45, 50, 55
and 60 each beat the mean of the four copy episodes in their block (41–44 vs 45,
46–49 vs 50, 51–54 vs 55, 56–59 vs 60) in at least 3 of 4 blocks, with a mean
advantage of at least +10 deliveries. If not, the settled-phase gap was noise.
(Four blocks give limited power: 4 of 4 wins would occur by chance about 6% of
the time. A decisive test needs a designed run -- e.g. soft target updates,
which remove the rhythm, against hard copies.)

---

## cold_run24 results (60 episodes, complete)

60 fleets, map 50_5_5_10_5_2, order stream with per-floor docks (12 per floor,
gated), order window ±1 → ±3, slow decay 0.9786, per-fleet terminal, exploration
peak 0.30 at episode 15.

| phase | exploration | deliveries | efficiency | collisions / 100 deliveries | shortest-path agreement |
|---|---|---|---|---|---|
| 1–8 (±1) | 0.12 | 117 | 29% | 64 | 0.53 |
| 9–23 (±2) | 0.26 | 123 | 31% | 46 | 0.57 |
| 24–30 (±3) | 0.10 | 136 | 34% | 54 | 0.63 |
| 31–45 | 0.02 | 141 | 35% | 46 | 0.64 |
| 46–60 | 0.01 | 136 | 34% | 48 | 0.64 |

Episode 1 delivered 46; within three episodes, about 130. Exploration's peak
(episodes 10–16) dipped to about 100. From about episode 20: a plateau near 140
(35% of the conflict-free ideal of ~403), best episode 40 (176, 44%), with a mild
slide over episodes 46–60 (trend −0.54 per episode over 30–60; rolling mean ending
near 130).

**Improved:** shortest-path agreement 0.53 → 0.64; collisions per 100 deliveries
64 → ~47; the delivery value rose sensibly (0.07 → 0.21, far below the 1.42
ceiling). **Did not improve:** collisions per episode (~64 throughout); conflict
choices stayed dominated by waiting (~70%; follow ~7%, reroute ~9%); exit legs
lengthened late (48 → 58 steps); the recovery head's value kept falling (−0.02 →
−0.30) at ~64 invocations per episode.

**Pre-registered target-copy prediction: NOT MET.** No-copy episodes won 2 of 4
blocks, mean advantage −6.4 (required: 3 of 4 and ≥ +10). Over episodes 24–60,
no-copy 142.4 vs copy 137.1 -- within noise. The settled-phase gap at episode 44
was regression to the mean; hard target copies are not shaping performance.

**Reading.** The policy learned route-following and delivery fast, then settled
into a cautious equilibrium: in single-lane corridors, waiting is the safe learned
response, and nothing in the design teaches fleets who should go first. Conflicts
resolve slowly through waiting and recovery, and throughput levels off near 35%.
Next: the four measurement fixes, the randomised preemption test, then structural
levers (priority-based passage, coordinated single-lane access, congestion-aware
routing).

---

## Conflict rules: evidence for path-based warnings (measured after cold_run24)

**cold_run24's recovery record:** 7,756 collapses (~129 per episode); 83% resolved
by a Tier 1 one-cell sidestep; 74% were repeats of the same pair (median 4th
offence, worst 59); 46% involved 5+ fleets (largest 51); all 64,992 holds ran the
30-step cap. "Closest to goal wins" (recovery_warehouse._assign_yields, with
alternation for repeated pairs) is applied only after a collapse. The waiting
system has no priority rule: "waiting" only labels a fleet held still near a peer.
The warning zone is 2.0 hops (graph distance, via proximity.pairs; the config
comment saying "Manhattan" is stale), so a one-cell sidestep cannot clear it --
the re-meeting loop.

**Distances:** collisions (0.5), warnings (2.0), braking and waiting use hops. The
perception diamond (density channels) uses straight-line grid distance: on 50_,
23% of cells within 2 grid steps are more than twice as far in hops; on 25_, 18%
of cells within 5 are 10+ hops away (phantom closeness). Candidate: stamp only
cells within 5 hops (static per-cell neighbourhoods).

**Pair classification** (shortest-path driver, 60 fleets, 300 steps; each pair
within 3 hops classified by the next 5 route cells):

| | within 1 hop | 1–1.5 | 1.5–2 | 2–3 |
|---|---|---|---|---|
| following | 135 | 16 | 2,208 | 1,077 |
| head-on | 29 | 866 (1–2 combined) | | 844 |
| contested | 0 | 25 (1–2 combined) | | 430 |
| no conflict | 401 | 444 (1–2 combined) | | 679 |

Today's warnings (within 2 hops): ~4,100 pair-steps, 57% following, 22% head-on,
21% no conflict. Convoys sit at 1.5–2 hops.

**Kinematics:** collision at 0.5 hops, speed 0.5 cells per step, simultaneous
moves: a follower 1.0 hop behind a leader that stops reaches 0.5 (collision) in
one blind step; from 1.5 it ends at 1.0 and brakes next step. **Minimum safe
following gap: 1.5 hops.**

**Rule candidate:** hard floor -- warn within 1 hop regardless of paths; 1–3 hops:
warn and splat along contested cells only for head-on, contested or blocked
pairs; following allowed from 1.5 hops; no conflict → nothing. Effect on today's
warnings: following from 1.5 keeps 36% and adds 1,277 earlier (2–3 hop) conflict
warnings -- 67% of today's total, 81% real conflicts. (Following from 2.0 would
keep 89%, since convoys sit just inside 2.) Requires direction-aware braking.
Pair with a corridor-entry rule (wait at the junction if a fleet inside the
stretch is coming toward you) and priority by hops-to-goal with aging.