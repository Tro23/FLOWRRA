# FLOWRRA — proposed design changes

**Scope: the existing dead-fleet scenarios. Same maps, same injection regime, same
metrics.** No continuous order stream, no battery, no respawn. The point is to
test whether these changes fare better on the benchmark you already have, where
you already know what the current design does.

Every item carries the measurement that motivated it. Nothing here is proposed
from argument alone.

---

## What today established, as the basis for all of it

| finding | evidence |
|---|---|
| The 6 BFS gradient dims carry the routing | lesion at 0.24%: completion −7.3pts, 24/30 episodes directional, steps 258 → 712 |
| Their grip weakens as density rises | completion ratio −6.66 at 0.24%, −1.04 at 1.74%, **+3.53** at 4.18% |
| Rays and the density field are independent channels | spearman −0.296 at 0.24%, −0.301 at 4.18% — invariant across a 17× density change and a 5.8× change in ray peer-hit rate |
| Yet lesioning **either** leaves the policy no worse at 4.18% | rays +0.78, density +1.19 on completion; both *reduce* collisions |
| Today's geometry changes are behaviour-neutral | reverting all three: completion 0.8278 → 0.8389, 0.16 sd |
| The orchestrator is doing the collision work | override rate 14%, 13,041 collapse events, 25.7% of Tier-1 escapes move nowhere |

**The puzzle that drives the redesign:** two independent information channels,
neither of which the policy appears to use. Independent *and* unused is stranger
than redundant, and it points at what sits between the features and the outcome
rather than at the features.

---

## 1. Waiting — the highest-priority change

**The problem.** Standing still is available (action 0, always structurally
valid) and actively punished. `idle_penalty = −0.5` fires on any non-moving step
while moving toward the goal pays `(old_dist − new_dist) × movement_multiplier`.
The policy is taught every step that waiting costs and pushing pays — at every
density, always.

A real multi-step hold exists (`_yield_until`) but **only Tier-3 recovery can
invoke it**, at one call site. Its own comment records that it replaced a
one-step version because that "let a loser immediately retry the same losing move
the very next step." The lesson was learned once, for recovery only. The policy
has no equivalent.

**What that looks like at 4.18%:** 13,041 collapse events, `[19,49]` colliding
314 separate times, `[27,55]` 256 times. Two fleets meet, neither holds, recovery
shoves both, integrity resets, they meet again.

**Proposed:**

- **Conditional idle penalty.** Free when blocked, penalised when clear. One
  config change in shape, not a new mechanism. This is also the concrete form of
  making `T` state-dependent (see §5).
- **A hold action with learned persistence.** The policy can choose a
  `_yield_until`-style hold rather than re-deciding every step. Needs a release
  signal — see §2.
- **Fix Tier 1 reporting failure as success.** `best_pos == current_pos` should
  fall through to Tier 2, not call `force_repair()` and re-fire next step.
  Standalone bug, worth fixing regardless.

---

## 2. The release signal — what tells a held fleet to move

A held fleet currently has no way to know its blocker has cleared. Nothing in the
state distinguishes "blocked now" from "blocked and clearing."

**Most of the ingredients already exist:**

- Peers' intended paths are in the state as of this session — the projection
  walks each peer's BFS descent and stamps its next 3 cells with decaying
  severity.
- `peer_velocities` reports the hit peer's direction per ray. A peer moving
  *away* along your ray is a clearing corridor; stationary or approaching is not.

**Proposed:** make the clearing signal explicit rather than implicit. Whether
that needs new dimensions or better use of the 18 `peer_velocities` dims is an
open question — and the ray lesion suggests the network may not be reading them
today.

---

## 3. The fused cell-level density field

**Why, and it is not compute.** A ray is the only channel that can detect an
obstacle nobody registered — humans, debris, a dropped pallet. Position reports
only cover the fleet registry; the graph only covers static structure. **The
fused field is the only place an unknown obstacle can exist at all**, and adding
one becomes a line of code rather than a new subsystem.

**Structure:**

- **Shared global field over cells**, stamped once per step, sampled per
  observer. Currently every observer rebuilds its own 11³ cube: ~303,000 stamp
  calls per step at 200 fleets, of which **99.9% are rejected by the early-out**.
- **Positions** cover registered fleets, including around blind corners — corner
  coverage comes from position reporting, not from rays.
- **Rays** cover unregistered obstacles and line-of-sight clearance.
- Exactly equivalent to today for the shared terms: active-peer stamps,
  projection trails and collapse memory have no observer argument. Self-exclusion
  and the near-goal discount apply as corrections on top.

**And the real reason this is a prerequisite, not an optimisation:** a shared
field is the only structure that has a **partition function**. Normalising stress
across cells gives `Z`, and without `Z` there is no distribution — no entropy, no
Gibbs measure, just an exponential with nothing behind it. Per-observer 11³ cubes
cannot express a floor-wide normaliser. Everything in §5 depends on this landing
first.

**Held in the background, dependent on this:**

- Collapse memory dies in 8 steps (×0.7, floor 0.05) — measured at **2–6 live
  cells** across a 200-step run. The field is almost entirely instantaneous and
  carries no memory of where things went wrong. `[19,49]` colliding 314 times
  says repeat-offender junctions are not being remembered.
- Memory and instant share a cell, so the network cannot tell "someone is here"
  from "something went wrong here recently."

---

## 4. Heterogeneous graph — obstacles and agents as distinct node types

**The problem.** `_build_adjacency` treats every fleet identically, so the GAT
attends over parked fleets and moving ones the same way. At block 200 of a
200-step run that was **160 of 200 fleets** — 80% of perception spent on fleets
that are not going anywhere, then masked out of the loss by `active_mask`.

**Proposed.** Two node types. An idle fleet must still be *perceived* — it is a
real obstacle — but it does not need to *perceive*. A cheap static embedding for
the obstacle class gives the same saving as idle caching **without the
staleness**, because the static class does not need 291 dims at all.

**What the measurement already says.** Idle caching bought 1.68× at 200 fleets
with no directional behaviour cost at 25 fleets (13 of 30 episodes diverged, 8
worse and 5 better — divergence, not degradation). But it was never tested at
density, and the heterogeneous version sidesteps the staleness question entirely.

---

## 5. A Gibbs objective: `⟨S,A⟩` entropy and T as throughput pressure

**The frame is the Gibbs measure**, and the choice of the exponential form is
deliberate rather than decorative:

```
P(cell) = exp(−E / T) / Z          F = E − T·S
```

**Why the exponential, and not a rational form.** `exp(−E/T)` is
scale-covariant: double every stress and halve T and the distribution is
unchanged. So one policy transfers across densities and floor sizes **without
retuning** — which is the scaling problem this whole line of work started from.
The system expands and contracts with the load. `1/(1+R)^β`, which was the
earlier proposal, has no such invariance; β was a cheap dodge that avoided
underflow by giving up exactly the property worth having.

**The numerical objection was to one substitution, not to the form.** Putting
integrity directly in the temperature slot — `exp(−S / k·C)` with `C` the global
tri-state {0.0, 0.5, 1.0} — fails twice: a singularity at `C = 0`, so one
collision anywhere sets the temperature for every fleet on a 120k-node map; and
underflow, since low coherence and high stress are the same situation, so
`exp(−10/0.05)` collapses to exactly zero in float32. That is the August
saturation defect returning — a neighbourhood of identical zeros with no
gradient, in a jam, which is the one moment the gradient is load-bearing.

**Both problems dissolve with a proper normaliser.** Computing `Z` over the
shared field means log-sum-exp: subtract the max before exponentiating, and
nothing ever underflows. The failure was an artifact of per-observer cubes with
no partition function, not of the Gibbs form. **This is why §3 comes first.**

**The objective:** `F = E − T·S`, minimising stress while paying for entropy at
exchange rate T.

**The state scalar decomposes:**

```
H(S,A) = H(S) + H(A|S)
```

Spatial entropy — how concentrated stress is on the floor — plus conditional
action entropy — given where fleets are, how undecided they are. Both terms, not
a choice between them. `H(A|S)` distinguishes a fleet undecided in a hotspot from
one undecided in an empty aisle; `H(A)` alone cannot.

Computable directly once the field is shared and cumulative: normalise stress
across cells to get a distribution, then `S = −Σ p log p`. **Measured, not
posited** — which is what was missing when we tried putting integrity in the
temperature slot.

**T is throughput pressure.** Full order queue → T low, push, accept risk. Slack
→ T high, hold, stay clean. Operationally meaningful, and the knob a continuous
system needs.

**And T already exists, hardcoded.** `idle_penalty` against
`movement_reward_multiplier` is a fixed exchange rate between safety and
progress. Today's log is what a badly-chosen T looks like. So §1 and §5 are the
same change at different altitudes — which makes this testable by sweeping one
number, the best property a piece of theory can have.

**Reading:** Jaynes 1957 for T as a Lagrange multiplier with no gases anywhere;
Haarnoja et al. 2017/2018 for `reward − T·entropy` with automatic temperature
tuning already working. Caveat: MaxEnt-RL's entropy is over the action
distribution, so the auto-tuning results do not transfer to the spatial term for
free.

---

## 6. Also queued, from earlier in the session

- **The target validity mask.** `next_a = next_q_sum.argmax(dim=2)` maxes over
  all 7 actions with no mask, so the target bootstraps from actions the agent
  could never take. ~70% of actions are invalid on a degree-2.27 graph:
  systematic, one-directional overestimation. A uniformly inflated target is
  perfectly learnable and teaches nothing — the exact shape of "losses converge,
  behaviour doesn't", twice now. **Traps:** use a finite sentinel (`−inf ×
  active_mask 0` is NaN, which kills the whole loss silently), and mask the
  *next* state, not the current one.
- **Ray transform** `d/(d+c)` replacing the saturating `min(d/25, 1)`. Saturation
  measured at 26% sparse, 3.8% dense.
- **Ray hit-type flag (6 dims)** — rays cannot currently distinguish a wall from
  a stopped fleet at the same distance.
- **Override rate bounds what the policy can learn.** 14% of actions rewritten at
  4.18%. Measurable by re-running a lesion with overrides disabled.
- **Learned dispatch.** `_hops` is a four-line sort. A value-head bid is the
  change that would make the GNN load-bearing in recovery.

---

## Suggested order

1. **Waiting** (§1) — highest evidence, smallest change, and your intuition is
   that it is the root cause of the collapse churn.
2. **Target validity mask** (§6) — must land before any retrain, cheap, and
   independently motivated.
3. **Fused field** (§3) — enables §5 and the unregistered-obstacle channel.
4. **Gibbs objective** (§5) — starts as a one-number sweep on the conditional
   idle penalty, which is already §1. The full form needs `Z` from §3.
5. **Heterogeneous graph** (§4) — largest change, best deferred until the field
   is shared.

**Decide the full state-vector set before the retrain.** Any dimension change
forces it, and you do not want to run 300 episodes twice.

**Keep the benchmark fixed** while all of this changes: same dead-fleet
scenarios, same maps, same metrics. One thing at a time, measured against
numbers you already have.
