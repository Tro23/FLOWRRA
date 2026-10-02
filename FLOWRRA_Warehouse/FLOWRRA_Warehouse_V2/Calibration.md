# CALIBRATION.md — how FLOWRRA's numbers were set, and how to set them again

Many numbers in `config_warehouse.py` are not free choices. Each was either
**measured** in a run, **derived** from a condition the system must satisfy, or
**computed** from other numbers by a formula. When the map, the fleet count, a
reward value or an event type changes, these numbers must be re-derived the same
way, not guessed again. This document records each one: its current value,
where it came from, and the recipe to redo it.

The numbers form a chain. Measured numbers feed derived ones: change a reward
value and the scales move; move the scales and the weights must be re-checked;
the recovery bound follows the holon scale by itself.

Related documents: REWARD_DESIGN.md (scales and weights in full),
TARGET_NETWORK.md (target updates, the recovery head's runaway),
PATH_AWARENESS.md (approach charge), SIMULTANEOUS_STEP.md (slow memory).

---

## Quick reference: when you change X, redo Y

| you change | redo |
|---|---|
| map, fleet count, or scenario mix | value scales (§1), then weight checks (§2); buffer size (§7) |
| any reward constant | value scales (§1), then weight checks (§2) |
| a recovery cost or bonus | holon scale (§1); the bound follows by itself (§3) |
| a new recovery event, or events that can fire more than twice a step | the bound's formula, by hand (§3); holon scale (§1) |
| gamma | every scale (§1) and the bound (§3) |
| state width (new perception channels) | buffer size (§7); train a fresh network |
| hard-copy interval K | tau = 2 / K (§4) |
| buffer size, episode count, or steps per episode | exploration schedule (§9) |

---

## 1. Value scales — measured

`reward_decomposition.scales`: safety **34.3**, delivery **70.4**, efficiency
**17.5**, holon **70.3**.

**What they do.** Each head's reward is divided by its scale before weighting,
so the heads' values sit on a comparable footing and the 6 : 4 : 1 weights mean
what they say. Without this, the raw constants become the priorities by
accident: progress speaks at every step, collisions rarely but loudly.

**How they were set.** cold_run18 ran in shadow mode: it trained on the old
reward while computing the three new heads alongside, without learning from
them. Each scale is that head's mean reward per active fleet-step, divided by
(1 − γ), which is the size of value the head carries if that typical reward keeps
arriving. The holon scale came the same way from the legacy integrity reward:
−0.703 per active fleet-step ÷ 0.01 = 70.3. Before it existed, the unscaled
recovery loss was about 100× the three heads' combined and trained the shared
layers almost alone (cold_run19).

**Recipe.**
1. Run on the new setup (map, fleet counts, reward values) long enough to see
   typical traffic. cold_run18 was a full run; 15–20 episodes gives a first
   estimate, to be confirmed on the full run.
2. From the CSV, take each head's mean reward per active fleet-step: the
   `rwd_shadow_mean_{head}` columns in shadow mode.
3. Scale = |mean| ÷ (1 − γ), averaged over the run's episodes, not one episode.
4. Pitfall (REWARD_DESIGN.md §11): scaling by spread alone mis-scales the heads.
   Use the mean per active fleet-step.

**If only gamma changes**, scale × (1 − γ_old) ÷ (1 − γ_new) is a first estimate
with no run needed. Re-measure anyway: any potential-based shaping term's
per-step reward itself depends on γ.

---

## 2. Priority weights — derived from decision checks

`reward_decomposition.weights`: **6 : 4 : 1** (safety : delivery : efficiency).

**How they were set.** Derived from the priority order, not chosen. Each check
compares two quantities, each normalised (divided by its head's scale) and
weighted:

| check | must hold | 3 : 2 : 1 | 6 : 4 : 1 |
|---|---|---|---|
| worst warning step vs one step of progress | warning larger | 0.105 vs 0.086, barely | 0.21 vs 0.086 |
| one delivery vs a 25-cell route of progress | delivery larger | 2.84 vs 4.29, **fails** | 5.68 vs 4.29 |
| one collision vs one delivery | collision larger | 4.38 vs 2.84 | 8.75 vs 5.68 |

With 6 : 4 : 1, yielding beats advancing whenever closeness to a peer exceeds
about 0.4 (with 3 : 2 : 1, only right at the worst closeness).

**Recipe.** After any change to the scales or reward constants, recompute both
sides of every check (event reward ÷ its head's scale × its weight). Pick the
smallest whole-number weights that pass all checks with margin. When a new
priority rule is introduced, add a check for it. The event values behind each
row are in REWARD_DESIGN.md.

---

## 3. Recovery value bound — computed by the runner, automatically

`training.recovery_value_bound: True` → **±55.5** today.

**Why it exists.** In cold_run21 the recovery head's value ran from −0.4 to
32,554 in about 20 episodes (TARGET_NETWORK.md). No value can legitimately
exceed the largest per-step reward divided by (1 − γ). Anything beyond that is
the bootstrap feeding on itself.

**Formula** (`_recovery_value_bound()` in main_runner_warehouse.py):

    per-step bound = (1 + 2 × (|invocation cost| + resolution bonus + preemptive bonus))
                     ÷ holon scale
    value bound    = per-step bound ÷ (1 − γ)

    today: (1 + 2 × (4 + 6 + 9)) ÷ 70.3 ÷ 0.01 = 55.5

The 1 is the holon's coherence range (−1 to 0). The 2 × allows every event twice
per step. That is generous on purpose. The test fixture once showed two
invocation costs in a single step (−8.5 unscaled); that was a duplicated
cost-and-outcome block in `_policy_recovery_step`, removed 2026-09-29, so one
of each event per step is now the real maximum and the 2 × is pure margin.

**What updates by itself.** Change a cost, a bonus, the holon scale or γ, and
the runner recomputes the bound at startup and logs it
(`recovery head: ... value_bound=`).

**What needs a hand edit.** A new kind of recovery event, or events that can
fire more than twice per step: extend the formula. Check against data: the
holon column's most extreme per-step value (unscaled) must fall inside the
per-step bound.

**Reading it in a run.** `qval_recovery` should live well inside ±bound. If it
sits pinned at the bound for long stretches, one of two things is happening:
- the bootstrap loop is still pushing, so the next lever is a slower target:
  a longer hard-copy interval (§4);
- legitimate values are being clipped because the per-step bound is too tight,
  so check the data as above.

**The same recipe works for any head.** The three movement heads need no bound
today (steady throughout cold_run21). If one ever runs away: its largest scaled
per-step reward ÷ (1 − γ).

---

## 4. Target network updates — computed

`training.target_tau: 0.002` (soft updates). Or `target_tau: 0` with
`--target-sync K` for a hard copy every K learning steps (the default K is
1,000).

**Equal average lag.** A hard copy every K steps leaves the target between 0 and
K steps behind, K/2 on average. A soft update of rate τ averages about 1/τ steps
behind. Setting them equal gives **τ = 2 / K**: K = 1,000 → τ = 0.002.

**What differs is the shape of the lag.** Hard copies freeze the target between
jumps. Soft updates move it a little every step, which removes the jumps and
kept the three movement heads steady in cold_run21. But it also gave the weakly
anchored recovery head no pause from chasing its own estimate. That is why the
recovery head now has Double DQN and the bound (§3).

**Choosing.** cold_run20 ran hard copies and cold_run21 soft. cold_run22 returns
to **hard copies** (`target_tau: 0`, K = 1,000): with this many changes at once,
the target mechanism stays the known one. Soft updates remain an option once the
recovery fixes have proven themselves. If the recovery head still pins against
its bound under hard copies, lengthen K for a slower target.

---

## 5. Approach charge size — fixed by construction

`reward_decomposition.approach_warning: True`.

    charge = warning_zone × closeness × min(1, share ÷ base_speed)

`share` is this fleet's part of the gap closed this step. Driving at a neighbour
at full speed costs exactly the old worst warning, so the checks in §2 hold
unchanged. The identity is built into the formula: changing `warning_zone` or
`base_speed` keeps it true. If the formula itself changes, rerun the six
geometry unit tests (PATH_AWARENESS.md) and recheck the first row of §2.

---

## 6. Slow congestion memory — set by half-life

`density.slow_decay_factor: 0.987`, a half-life of **53 steps** (the fast memory
is 0.7, a half-life of 1.9 steps).

    decay = 0.5 ^ (1 / h)        for a half-life of h steps

To change it, choose h (how long a congestion trace should take to fade to half)
and compute the decay. For example, h = 100 gives 0.99309.

---

## 7. Replay buffer — a memory budget

`training.buffer_capacity: 20,000` at 1,239 inputs (it was 30,000 at 777).

Memory grows roughly with capacity × state width × fleets per transition. Keep
that product near what the machine has already handled: 30,000 × 777 ≈ 23.3
million, and 20,000 × 1,239 ≈ 24.8 million, both at 46 fleets.

    new capacity ≈ old capacity × (old width × old fleets) ÷ (new width × new fleets)

For example, at 90 fleets and 1,239 inputs: about 20,000 × 46 ÷ 90 ≈ 10,000.
Confirm RAM on a smoke run before a full run.

---

## 8. Known gaps — not yet calibrated

- **Episode length.** `--max-steps 800` is passed on the command line, but the
  environment reads `training.max_steps_per_episode` (780). The two should
  agree; the fix is logged.
- **Standoff pressure.** Under the approach charge, only the idle penalty presses
  on a stopped face-to-face pair, roughly 7× weaker than the old presence charge.
  cold_run22's counters will show whether standoffs grow.
- **Identical instances.** Every run with `--seed 0` on this map and scenario
  set faces the same 60 problems in the same order, with the same fleet shuffle
  (the runner's generator is used only for these draws; replay matched cold_run22
  60/60, and claimed-goal fingerprints confirm cold_run19–21). Episode-by-episode
  paired comparisons across runs are therefore controlled.
- **Run-to-run noise.** There is no measured noise floor for the current
  configuration, so we cannot yet say how large a difference between two runs
  must be to be real. Two seeds of cold_run22's configuration would set it.

---

## 9. Exploration schedule — sized by the buffer

`exploration`: peak **0.30** at **T/4**, width **T/8**, minimum 0.01 (cold_run23).
Absent keys keep the historical 0.95 at T/2, width T/6.

**Why it is derived, not chosen.** The replay buffer holds one transition per
step, so its capacity in episodes is capacity ÷ steps per episode: 20,000 ÷ ~780
≈ 25 episodes. What the learner studies at episode t is the average behaviour
of episodes t−24 to t. cold_run22's 0.95 peak at episode 30 left the buffer
**63% random at episode 50**, with only 5 near-greedy episodes at the end.
Collisions surged exactly as fleets turned greedy among greedy neighbours while
the buffer still held the peak's random swarm (and in cold_run19 and 20 too).

**The rule.** Near-greedy (eps < 0.05) for at least one full buffer turnover
before the episodes you judge the run on, with a margin.

| schedule | eps ep1 | below 0.05 from | clean episodes at end | buffer random @50 | @60 | all moves random |
|---|---|---|---|---|---|---|
| 0.95, T/2, T/6 (cold_run22) | 0.024 | 56 | 5 | 63% | 28% | 40% |
| **0.30, T/4, T/8 (cold_run23)** | 0.061 | 30 | **31** | **2.8%** | 1.1% | 10% |
| 0.25, T/5, T/10 | 0.055 | 24 | 37 | 1.2% | 1.0% | 7% |

**Recipe.** Compute buffer episodes = capacity ÷ steps per episode. Choose the
peak and its position so eps < 0.05 at least one buffer-length (plus margin)
before the end. Compute the buffer's mean eps at the judged episodes; keep it
under a few percent. More episodes or a smaller buffer allow a later peak.

**The recovery head** explores at `exploration_scale` (0.25) × the fleet
epsilon, so its peak falls with the fleets' (0.24 → 0.075 here). Random
recovery invocations also disrupt the buffer (collapses, holds), which is the
reason to let it fall.

---

## Appendix: switches for cold_run22

Printed from the staged config.

| setting | value | purpose |
|---|---|---|
| `step.simultaneous` | True | all fleets move in one step |
| `density.slow_channel` | True | slow congestion memory channel |
| `density.slow_decay_factor` | 0.987 | half-life 53 steps (§6) |
| `density.warning_splat_per_pair` | True | warning splats per fleet pair |
| `density.paths_channels` | True | **new:** my route and their routes (channels 4 and 5) |
| `density.entropy_fix` | True | **new:** entropy reads the right channels |
| `perception.holon_integrity` | True | holon integrity as a perceived feature |
| `reward_decomposition.mode` | "priority" | three priority heads |
| `reward_decomposition.weights` | [6, 4, 1] | safety : delivery : efficiency (§2) |
| `reward_decomposition.scales` | 34.3 / 70.4 / 17.5 / 70.3 | value scales (§1) |
| `reward_decomposition.safety_includes_recovery` | **False** | off on purpose: recovery is judged in the holon column only |
| `reward_decomposition.approach_warning` | True | **new:** charge approach, not presence (§5) |
| `rewards.mask_active_before_actions` | True | reward masking fix |
| `rewards.attribute_collisions_post_action` | True | collision attribution fix |
| `recovery.preempt_skip_convoys` | True | convoys are not preempted |
| `recovery.escalate_on_recurrence` | True | recurring deadlocks escalate |
| `recovery.preempt_skip_held_pairs` | **False** | off on purpose: cold_run17 regression |
| `recovery_policy.enabled` | True | learned recovery head |
| `training.double_dqn` | True | Double DQN, three movement heads |
| `training.target_tau` | **0.0** | hard target copies every 1,000 learning steps, as cold_run20 (§4) |
| `training.recovery_double_dqn` | True | **new:** Double DQN, recovery head |
| `training.recovery_value_bound` | True | **new:** recovery target clamped to ±55.5 (§3) |
| `training.buffer_capacity` | 20,000 | **new:** sized for 1,239 inputs (§7) |
| `warehouse.arrival_radius` | 0.5 | arrival distance |

**Added for cold_run23:**

| setting | value | purpose |
|---|---|---|
| `training.per_fleet_terminal` | True | a fleet's bootstrap ends when it retires or stops (TARGET_NETWORK.md) |
| `exploration.eps_peak / eps_min / mu_frac / sigma_frac` | 0.30 / 0.01 / 0.25 / 0.125 | small, early exploration; 31 near-greedy episodes at the end (§9) |

Also from cold_run23: logging to stdout (episode lines no longer glue onto
prints), path statistics (`path_*` columns, PATH_AWARENESS.md), and a startup
line `learning: per_fleet_terminal=True | exploration ep1=0.061 peak=0.300 at ep
15 ... below 0.05 from ep 30`.

**Launch.** cold_run19's recorded command, with only `--out` changed:

    python main_runner_warehouse.py --maps-dir all_maps --scens-dir all_scens_v3 \
      --maps 25_5_2_5_2_1 --episodes 60 --agent-sets 46 --max-steps 800 \
      --seed 0 --out cold_run23 2>&1 | tee cold_run23.log

Use no `--resume`: the state width changed, so the network starts fresh.
cold_run19's command did not use `--cold-start` (which peaks exploration at
episode 0). Keep it that way for comparability with cold_run19–21.

**Smoke test first:** `--episodes 1 --max-steps 100 --out smoke22`. The log
should show `input_dim=1239`, and a `recovery head:` line with
`double_dqn=True value_bound=55.48 | target_tau=0.0` and all three new
switches `True`.