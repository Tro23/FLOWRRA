# Reward design: structure, priorities, normalisation

Status: **proposal**. No reward code changes until the priority order (section 4)
is decided.

## 1. Why this exists

cold_run17 added a per-head reward breakdown (`rwd_{all|learned}_{pos|neg}_{head}`).
It reconciles with the logged episode reward to within 0.001 in every episode,
and every point of every penalty reaches the gradient (the `all` and `learned`
views are identical). What it showed:

- Penalties are **61%** of all reward magnitude: 14,601 of penalty against 9,378
  of positive reward per episode, netting -5,223.
- The trained policy collides **more** than the untrained one at the same low
  exploration: 8.1 -> 17.4 collisions per episode (cold_run16 showed the same
  direction, 8.1 -> 10.9).
- At the moment a fleet chooses, advancing beats every safe alternative:

| choice | reward per step |
|---|---|
| push on toward the goal | **+1.5** (0.5 cells x 3.0) |
| wait, peer already blocking | 0 (idle exemption) |
| wait early, before the peer blocks | -0.5 (idle penalty) |
| detour away | -1.5 (moving away on the static map) |
| warning-zone penalty, at its worst | -1.2 |

The largest warning penalty (-1.2) is smaller than the reward for advancing
(+1.5). Only the -50 collision pushes back, and it is uncertain and a few steps
away. Braking softens this near a peer, but that means fleets close right in
before stopping.

No single constant is the cause. **The reward has no explicit priorities and no
common scale.** Adjusting one number against another is a patch; this document
proposes the structure instead.

## 2. The current structure

All five heads are summed with **equal weight (1.0)** to choose actions.

| head | term | value | fires when | depends on this fleet's action? |
|---|---|---|---|---|
| goal | progress | +/-3.0 per cell (+/-1.5 per step) | every step it moves | **yes, strongly** |
| | delivery | +100 | once, on arrival | yes |
| | final approach | +3.5 | near the goal | yes |
| | baseline | +0.1 | every step, every fleet | no |
| safety | warning zone | 0 to -1.2 | every step near a peer | yes, weakly |
| | collision | -50 | on a collision | yes |
| integrity | coherence | integrity - 1 | every step, every fleet | barely (global) |
| | recovery used / resolved / preemptive | -4 / +6 / +9 | recovery events | partly |
| rescue | pickup bonus | one-off | on a handover | yes |
| | delivery bonus | 0 (off) | -- | -- |
| time | idle | -0.5 | not moving, not blocked | yes |
| | overtime | up to -10 | last ~30 steps | no (the clock) |

Measured per episode, cold_run17 (46 fleets):

| head | positive | negative | structure |
|---|---|---|---|
| goal | 8,643 | -2,824 | two-sided |
| safety | 0.2 | -1,693 | penalty only |
| integrity | 697 | -7,038 | mostly penalty |
| rescue | 38 | 0 | reward only |
| time | 0 | -3,046 | penalty only |

The goal head's 8,643 splits into roughly **4,013 of delivery** (+100 x 40
arrivals) and **4,630 of shaping** (progress, baseline, final approach).

## 3. What is wrong

1. **Priorities are accidental.** With equal weights, the raw constants *are* the
   priorities. How much "do not crash" counts against "get there" was set by the
   fact that one number is 50 and another 100.
2. **Scale and frequency differ wildly.** Progress speaks at every decision of
   every moving fleet. Collisions happen about ten times an episode across 46
   fleets. The frequent voice wins every decision.
3. **The goal head mixes two different things.** Over half of it is shaping (how
   fast), not delivery (whether). The shaping part is what decides each step.
4. **Integrity is global, in two ways.** `integrity - 1` is one number for the
   whole warehouse, charged to every fleet every step. And recovery events
   (-4 per invocation, +6 per resolution, +9 per preemptive recovery) are
   collected into `_pending_integrity_reward` and then added in full to **every**
   fleet: one recovery anywhere charges every fleet in the warehouse. Neither
   depends on a fleet's own action, so neither steers its choices; both add
   unpredictable noise to learning (consistent with `loss_integrity` rising
   while nothing improves).

   *Correction:* this document first assumed the coherence term dominated
   integrity. The per-term instrument showed otherwise on the test fixture:
   recovery events -33,840 against coherence -1,534. Real-map proportions
   await cold_run18.
5. **The warning re-attribution undoes the wrong amount.** The step charges a
   warning from a fleet's closeness *before* its move; the post-action
   re-attribution undoes it using closeness recomputed *after*. So the undo does
   not cancel the charge: a fleet that stays in the zone pays its starting
   closeness regardless of its action, and one that escapes keeps its starting
   penalty. Measured on the fixture: the undo mismatched on 951 of 3,096
   fleet-steps (31%), and 46 escaping fleets kept a penalty. The magnitude is
   small (58 of 1,773 warning penalty, ~3%), because closeness changes little in
   one half-cell step, and the consequence is **delayed** one transition rather
   than lost. It is the same lag the collision fix removed for collisions.
   Collisions are unaffected: their undo subtracts the same constant it added.

6. **Progress is measured on the empty warehouse.** Detouring around a jam is
   charged as moving away from the goal (see the congestion-aware gradient idea).

In practice the effective ordering is **efficiency first, safety last**, the
reverse of what anyone would choose.

## 4. Proposed priorities (to accept or reshape)

**Safety > Delivery > Efficiency.**

- **Safety**: no collisions, no conflicts you are part of, few interventions.
- **Delivery**: every order reaches its goal, including orders taken over from a
  dead fleet.
- **Efficiency**: delivered with less: fewer steps, less idling, fewer detours,
  less recovery.

Proposed mapping of existing terms:

| priority | terms |
|---|---|
| safety | collision; warning zone (reframed, see 6); integrity **for conflicts this fleet is part of**; recovery events |
| delivery | arrival; rescue pickup; rescue delivery |
| efficiency | progress (as shaping, see 6); idle; overtime; final approach |
| candidate for removal | baseline (+0.1, identical for every action, steers nothing) |

## 5. Normalisation: options

The aim: put every head on a comparable scale **before** weighting, so a weight
means what it says.

**A. Per-head reward normalisation.** Divide each head's per-step reward by a
running estimate of its spread, then apply priority weights. Simple, and the
weights become meaningful. Caution: heads made of rare large events (collisions)
and heads of frequent small ones (progress) have very different distributions, so
the spread estimate must be robust.

**B. Per-head value normalisation (PopArt-style).** Normalise each head's value
targets adaptively while preserving its outputs. Handles scale drift, e.g. the
goal head growing into the +100 arrival reward, which is what `loss_goal` rising
looks like. More machinery.

**C. Safety as a constraint.** Choose the best action for delivery and efficiency
*among actions judged safe*, rather than trading safety off with a weight. Closest
to how the action masks already behave, and it cannot be outvoted. Needs a
reliable "safe" judgement from the safety head.

Tentative recommendation: **C for safety, A for delivery against efficiency.**
Safety should not be negotiable by a larger progress reward; delivery and
efficiency are genuinely a trade-off, and weights suit that.

## 6. Shaping, not goals

Progress (`(old_dist - new_dist) x 3.0`) is a **shaping** term: its purpose is to
speed learning, not to define what is good. Potential-based shaping,
`F = gamma * Phi(s') - Phi(s)`, provably leaves the optimal policy unchanged. The
current term is close to that form but omits `gamma`, uses the static map, and in
practice dominates greedy choices. Proposal: make it exactly potential-based,
compute `Phi` from the congestion-aware distance (linking the gradient design),
and keep it subordinate to safety.

The warning zone should charge **approach**, not presence: penalise closing speed
toward a peer, so convoys (closing speed zero) pay nothing and advancing into
someone's space costs more than it earns.

## 7. Measure before building

Now in place (measurement only, proven not to change behaviour):

- `rwd_term_{pos|neg}_{head}_{term}` -- every term inside every head, active
  fleets. Terms within a head sum to the head exactly.
- `rwd_spread_{mean|std}_{head}` -- per active fleet-step: the scale a
  normalisation would use.
- `rwd_warnundo_*` -- how often, and by how much, the warning undo misses.

cold_run18 supplies the real-map numbers to calibrate against.

## 8. Predictions (these can fail)

If this diagnosis is right, after the redesign:

- in the per-step table of section 1, a safe choice comes out ahead of advancing
  when a conflict is approaching;
- the trained policy collides **less** than the untrained one at the same low
  exploration, reversing cold_run16/17;
- the plateau after the first few episodes lifts.

If collisions still rise with training under a reward where yielding pays, the
diagnosis is wrong.

## 9. Decisions (made)

1. **Priority order: Safety > Delivery > Efficiency**, with the term mapping in
   section 4.
2. **Integrity: look at everyone, pay by your own actions.** Two separate
   choices:
   - *Perception* is holon-wide: every fleet should know the holon's state,
     alongside the cumulative density field and path awareness.
   - *Charges* are individual: a fleet pays only for the conflicts and
     recoveries it is part of.

   The current code is the exact inverse. No fleet perceives the holon's
   integrity (every situation feature is local), yet every fleet is charged for
   the whole warehouse: `integrity - 1` and every recovery event, in full. A
   fleet is paying for something it cannot observe, so it cannot learn to
   influence it.
3. **Safety: a weighted, normalised head.** The action masks already form a hard
   constraint layer (vetoing moves into occupied or blocked cells), so the design
   is a hybrid: masks for what is imminent, the weighted safety head for what is
   anticipated. Safeguard: once scales are calibrated, the per-step choice table
   must show a safe option outscoring advancing when a conflict approaches;
   otherwise safety's weight is raised.
4. **Baseline: dropped.** Efficiency would want its own baselines, and a per-step
   bonus rewards staying alive rather than finishing: +0.1 made hovering worth
   about +10 in value (0.1 / (1 - 0.99)).

## 10. Build plan

**Structure.** Three heads matching the priorities -- safety, delivery,
efficiency -- replacing the five. Terms regrouped as in section 4; baseline
removed. Changing the head count changes the network's output, so the first
run is a cold start.

**Attribution.** `current_integrity` is a step function of the whole
warehouse -- 0 if *any* collision exists, 0.5 if *any* warning, 1 only when
everything is clear -- so it is the holon's **worst** fleet's state, not a sum.
The per-fleet version charges each fleet its **own** state: -1 in a collision,
-0.5 in a warning, 0 if clear. (An earlier draft said the charges would "add up
to the holon's loss"; a worst-case measure does not add up, so that was wrong.)
Recovery events go to the fleets each event involves -- the fleets recovery
acted on, or the preemptive watch set for preemptive bonuses. A wasted
invocation, with nobody at risk, is a whole-holon decision and is split evenly
across active fleets.

**Perception.** Add the holon's integrity to every fleet's state -- the "look at
everyone" half. This changes the state width, so it rides with the cold start.

**Fixes carried in.** The warning undo uses the charge actually made; progress
becomes exact potential-based shaping (`gamma * Phi(s') - Phi(s)`).

**Normalisation: fixed scales, calibrated, not running.** A running estimate
would keep changing while old transitions sit in the replay buffer, stored at
the old scale -- the same reward meaning different things over time. Instead,
each head's scale is fixed from measured data.

**Calibration by shadow mode.** During cold_run18, compute the new three-head
reward alongside the current one and record its per-step spread, **without
learning from it**. That gives the fixed scales from real data, with no
behaviour change. cold_run19 then switches the new reward on.

## 11. Shadow mode (built; runs in cold_run18)

The three-head reward is computed every step alongside the live one and never
learned from -- proven: with flags off the code still reproduces the reference
run exactly. Columns `rwd_shadow_{pos|neg|mean|std|freq|typical}_{head}`.

Findings on the test fixture (crowded and untrained; real-map values come from
cold_run18):

- **Delivery reconciles exactly** with the live terms (2,025 both ways).
- **Attribution** cuts the combined safety + integrity penalty by about 24%
  (-38,158 -> -28,850). The fixture has most fleets in conflict at once; the
  real map should separate broadcast from attributed far more.
- **Exact potential-based shaping adds a large per-step term.** With potential
  `-m * distance`, `gamma * Phi(s') - Phi(s)` pays `m * (1 - gamma) * distance`
  on every step, even standing still: about 0.03 x distance. On the fixture it
  turned efficiency from net -1,417 to +172. Theory says the optimal policy is
  unchanged (the value function absorbs it), but it is large enough to be a
  decision: **keep exact potential-based shaping, or keep today's form**
  (`m * (old - new)`, which differs only by that term).
- **Spread alone would mis-scale the heads.** Per active fleet-step:

| head | speaks on | typical size when it does | std |
|---|---|---|---|
| safety | 65% | 9.0 | 8.1 |
| delivery | 0.4% | 96 | 6.3 |
| efficiency | 99.6% | 0.66 | 0.93 |

  Delivery is rare and large; its std hides its size. Safety's std is inflated
  by rare collision spikes, so normalising by it would shrink the everyday
  warning signal. Normalisation should use frequency and typical size, not std
  alone. cold_run18 records all three.

## 12. Built for cold_run19

All behind switches; with every switch off the code reproduces the reference run
exactly, and all 16 suites pass under both the legacy and the cold_run19 config.

| switch | cold_run19 | what it does |
|---|---|---|
| `reward_decomposition.mode` | `"priority"` | push the three priority heads instead of the five legacy heads |
| `perception.holon_integrity` | `True` | every fleet sees the holon's integrity (base features 83 -> 84) |
| `training.double_dqn` | `True` | online net chooses the bootstrap action, target net scores it (see TARGET_NETWORK.md) |

**How the priority reward is built.** Terms are still computed on the five legacy
heads -- every reward line and instrument is written against them -- then
regrouped per fleet by one shared builder (used by shadow mode too, so the two
cannot drift apart): safety = collision + corrected warning + own integrity
state + own share of recovery events; delivery = arrival + rescue pickup +
rescue delivery; efficiency = progress (today's form) + idle + overtime + final
approach. No baseline. Verified: what is pushed, un-scaled, equals the builder
exactly on every head.

**Value scales**, fixed from cold_run18 as each head's mean reward per active
fleet-step divided by (1 - gamma): safety 34.3, delivery 70.4, efficiency 17.5.

**Weights: 6 : 4 : 1**, derived from the priority order rather than chosen:

| check (normalised, weighted) | 3 : 2 : 1 | 6 : 4 : 1 |
|---|---|---|
| worst warning step vs one step of progress | 0.105 vs 0.086 -- barely | 0.21 vs 0.086 |
| one delivery vs a 25-cell route of progress | 2.84 vs 4.29 -- **fails** | 5.68 vs 4.29 |
| one collision vs one delivery | 4.38 vs 2.84 | 8.75 vs 5.68 |

With 6 : 4 : 1, yielding beats advancing whenever closeness to a peer exceeds
about 0.4 (with 3 : 2 : 1, only right at the worst closeness).

**Caught during testing.** The Double DQN extra forward pass overwrote the
network's cached `last_recovery_q`, which would have frozen the recovery head
for the whole run. Found because Double DQN with identical networks did not
reproduce the plain-DQN loss; fixed, and now equal to every printed digit.

**Prediction for cold_run19 (can fail):** at the same low exploration, the
trained policy should collide *less* than the untrained one -- reversing
cold_run16, 17 and 18 -- and the goal-loss spikes should not recur.

## 13. The recovery head (found in cold_run19_1)

cold_run19_1 stopped at episode 30 (a power cut), at the exploration peak.
Completion and collisions matched cold_run16 and cold_run18 within noise, and the
losses were calm. But **policy-invoked recoveries rose 2.4x** (39.2 vs ~16.6 per
episode) and **wasted invocations nearly doubled** (70.1 vs ~39), while forced
recoveries -- real deadlocks -- were unchanged.

**Cause.** The recovery head is a single holon-level decision, trained on the
*mean across active fleets* of the "integrity" reward column. Priority mode has
no such column, and the lookup silently fell back to column 0: the scaled safety
head. There, a wasted invocation's -4 is split across the holon and divided by
34.3 -- about -0.004 on average, against -4 before. Invoking recovery became
roughly a thousand times cheaper for the head that decides it.

**Fix -- the user's principle, applied at two levels.** Fleets are charged for
their own actions through the three reward heads; the holon's own decision is
judged by the holon's state. Priority mode now pushes a **fourth column**: the
legacy integrity head exactly (holon coherence plus every recovery event at full
size, identical on every fleet, unscaled). The learner accepts K or K+1 columns,
gives the extra one to the recovery head only, and the three reward heads never
see it. Legacy mode pushes K columns and is untouched.

**Verified.** Stored rewards are [N, 4]; the column carries full-size events
(-8.5 to +5 on the fixture); the reward heads still reconcile exactly. Adding
+100 to the fourth column alone left all three reward-head losses identical while
the recovery loss moved 5.95 -> 91.8. Double DQN with identical networks still
equals plain DQN. Legacy reproduces the reference run; all 16 suites pass under
both configurations.