# COLD_RUN23_ANALYSIS.md — results, scorecard, and handoff

cold_run23: cold_run22's system plus two changes — the **per-fleet terminal**
(a fleet's bootstrap ends when it retires or stops) and a **buffer-sized
exploration schedule** (peak 0.30 at episode 15, width T/8, below 0.05 from
episode 30). Hard target copies every 1,000 learning steps. 60 episodes, 46
fleets, map 25_5_2_5_2_1, `--seed 0`: the identical 60 instances as cold_run19–22.
Configuration frozen as `config_benchmark_cold_run23.py`.

All comparisons are either paired on identical episodes, or adjusted for each
episode's intrinsic difficulty (`instance_difficulty.py`; slope +2.69 collisions
and −1.06 deliveries per unit, fitted on episodes 2–45 of cold_run20 and 22).

---

## Headline

| | cold_run20 | cold_run22 | **cold_run23** |
|---|---|---|---|
| collisions per episode, all 60 | 10.3 | 11.0 | **8.2** |
| collisions, late (50–60) | 13.8 | 17.1 | **9.9** |
| late collisions beyond difficulty | +3.8 | +7.1 | **−0.1** |
| completion, all 60 | 0.888 | 0.882 | 0.876 |
| delivered, late (50–60) | — | 38.1 | **40.2** |

- **The late decline is gone.** Difficulty-adjusted collisions stay flat across
  every phase (+0.4, −1.6, −1.3, −1.1, −0.1); cold_run20 and 22 climb to +3.8 and
  +7.1 late.
- **Paired against cold_run22:** fewer collisions on 37 of 55 differing episodes
  (p = 0.007); from episode 30 on, 22 of 27 (p = 0.001); late, 9 of 10
  (p = 0.011), with 2.1 more deliveries per late episode.
- **Completion did not rise:** about 40 of 46 in every phase. The late collapse
  is gone, but a ceiling of about 40 remains.

---

## Pre-registered predictions, scored

| prediction | result |
|---|---|
| Terminal fix: delivery value stays at or below ~1.42 | **Met.** Peak 0.995; cold_run22 reached 1.51 by ep 40 and 1.86 by ep 54. The value now settles smoothly below the ceiling. |
| Terminal fix: doorstep behaviour improves | **Partly.** Step-in rate 0.56 → 0.65 (cold_run22 ended near 0.52); share of doorstep steps moving away unchanged at 25–35%. |
| Schedule: no collision surge after episode 30 | **Met.** Flat residuals throughout. |
| Schedule: late collisions below cold_run22's 17.1 on identical episodes | **Met.** 9.9. |
| Transient addendum: bump in 33–48, or none | **No bump** (mean residual −0.9, max rolling +1.5). Leans toward the random-neighbour account; a small transient (+2, ~1.8 standard errors) is not excluded. |
| (cold_run22's) choices shift toward follow and reroute | **Met now** (not met in cold_run22): follow 15% → 23%, reroute 24% → 26%, wait 37% → 31%, proceed 25% → 20%. |

Also: the recovery head was the calmest of any run (value −0.18 to 0, bound
±55.5; 2–9 invocations per episode). Standoffs still grow late (103 → 175
pair-steps), as in cold_run22 — a standing watch item for the approach charge.

**Exploration level made no detectable difference** within 0.01–0.30: every band
delivered 0.6–1.5 fewer and collided 0.4–1.3 less than difficulty predicts, all
within about one standard error (bands are tangled with training stage). The
"noise unsticks fleets" hint from episode 42 does not hold across the run.

---

## What is settled

- **Per-fleet terminal:** a real bug, fixed; the delivery value now behaves.
- **Size the exploration schedule by the buffer:** near-greedy for at least one
  full buffer turnover before the judged episodes (CALIBRATION.md §9). This
  removed the late surge.
- **Recovery head:** Double DQN plus the value bound keep it stable.
- **Path awareness** helps the young policy (cold_run22, 10 of 10 early episodes).
- **Method:** identical instances across runs (`--seed 0`), difficulty-adjusted
  judgement (`instance_difficulty.py`), predictions recorded before the data.

## What remains

- **The completion ceiling, about 40 of 46.** The ~7.8 unfinished fleets per
  episode break down as: **4.35 moving without arriving** (they start ~11 hops
  out and end ~9 away); **2.55 killed by injected errors** (1.6 of them rescued;
  matches the 2.55 errors injected per episode -- by design, unavoidable); 0.7
  rescuers still busy with rescue work; about 0.2 yielding or waiting; livelock
  is almost nil (0.07). So the realistic ceiling is about 43 of 46, and the stuck
  fleets are almost the whole controllable gap. Neither learning fix touched
  them, which points at the environment. Top suspect: **parked fleets as a false
  alarm** — retired fleets stay on the floor, stamped at severity 1.0 (above an
  active fleet's 0.7), permanently, while being passable, cost-free and ignored
  by route gradients. A version of this was already patched once, near a fleet's
  own goal only.
- **Standoffs** grow late under the approach charge.
- **Doorstep dithering** (about 30% of doorstep steps move away) persists.

---

## Handoff: next phase, in order

1. **Benchmark frozen:** `config_benchmark_cold_run23.py`. Spawn and despawn will
   change what an instance is, so later runs stop being comparable episode by
   episode with 19–23; this is the fixed reference.
2. **Spawn and despawn** outside the layout, so retired fleets leave the floor.
   The machinery exists (exit nodes, distance maps, "left the floor"); it is
   off (`despawn=False`). Wake it up and review before designing new parts.
   `frozen_obstacle_severity` then acts on nothing; rename
   `stopped_obstacle_severity` to `broken_obstacle_severity`, keeping the old
   name working. Optional first: one run with `frozen_obstacle_severity: 0.0`
   on the identical instances, to confirm the false alarm causes the stuck fleets.
3. **Batteries and charging stations**, built on despawn's leave-and-return idea.
4. **Constant exploration** (about 0.05) in a longer run, with near-greedy
   evaluation episodes every few episodes (needs a small runner change): no
   handover by design.
5. **Targeted noise** for fleets that make no net progress, instead of global
   noise; plus a per-fleet end-of-episode instrument for stuck fleets.
6. **Protected failure memory** (per failure type) once obstacles return.
7. Carried forward: `path_` columns now reach the CSV (staged runner);
   `instance_difficulty.py` must learn the new instance semantics once spawn
   and despawn change them; log the instance difficulty every episode.