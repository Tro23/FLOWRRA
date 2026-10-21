# cold_run25 -- analysis (2026-09-30)

60 episodes, 60 fleets, 50_5_5_10_5_2, order stream, all six conflict switches
on, recovery charged once. Paired by episode against cold_run24 (same seeds,
same exploration schedule, same order sequences). Charts:
`cold_run25_overview.png`, `cold_run25_drivers.png`,
`cold_run25_report_capacity.png`; tables: `cold_run25_report.md`.

## 1. Headline

| | cold_run24 | cold_run25 | paired |
|---|---|---|---|
| deliveries per episode | 131.7 | **191.6** (+45%) | +60.0, t = 23.4, better in **60/60** |
| collisions per episode | 63.9 | **37.0** | -26.8, t = -12.0, better in 58/60 |
| collisions per 100 deliveries | 48 | **19** | |
| best episode | 176 (ep 40) | **235** (ep 23) | |
| last 20 episodes | 135.7 | 189.8 | |

## 2. Over the run

| episodes | order window | deliveries | of the ideal | collisions | preemptive | holds (at cap) | policy wait | executed wait | cold_run24 |
|---|---|---|---|---|---|---|---|---|---|
| 1-10 | 1.2 | 178.8 | 38% | 39.4 | 62.9 | 593 (154) | 21% | 70% | 117.7 |
| 11-20 | 2.0 | 187.4 | 42% | 25.2 | 66.5 | 822 (410) | 17% | 70% | 116.2 |
| 21-30 | 2.7 | 205.7 | 48% | 32.1 | 39.1 | 356 (92) | 15% | 61% | 140.0 |
| 31-40 | 3.0 | 198.2 | 48% | 41.6 | 30.3 | 326 (49) | 13% | 62% | 144.9 |
| 41-50 | 3.0 | 196.7 | 48% | 38.8 | 31.8 | 358 (49) | 12% | 61% | 138.1 |
| 51-60 | 3.0 | 183.0 | 44% | 45.2 | 32.5 | 448 (184) | 15% | 67% | 133.2 |

Efficiency by order window: +-1 (eps 1-8) 37.2%, +-2 (9-23) 43.2%, +-3 (24-60)
47.0% (sd 3.6).

## 3. Against the ceilings

- **The learned policy beats the rule-based controller.** On the +-3 window
  cold_run25 runs at 47% of the conflict-free ideal; the shortest-path driver
  with the same six rules, at the same 60 fleets and window, at 42% (three
  seeds). Episodes 24-50: 48%. That is routing the rules alone cannot do --
  the congestion-aware skill Follower is known for, shown on a much harder map.
- **The last block is back near the controller**: 44% (183 deliveries against
  its 176). See section 5.
- Nothing reaches the ideal in dense traffic on this map: the capacity sweep
  saturates between 30 and 45 fleets (BENCHMARK.md).

## 4. What costs deliveries

Correlation of episode-to-episode CHANGES with deliveries (the training trend
removed -- the strictest view of analyse_run.py; |r| > 0.26 is p < 0.05 at
n = 59, and consecutive episodes are not independent, so treat it as a hint):

| mechanism | r |
|---|---|
| actions overridden by recovery holds | **-0.74** |
| recovery holds | -0.71 |
| repeat offences | -0.64 |
| holds at the 30-step cap | -0.59 |
| actions overridden by the rules | -0.39 |
| preemptive recoveries | -0.38 |
| collisions | -0.17 (n.s.) |
| waits the policy proposed | -0.03 (n.s.) |

**Collisions themselves barely move deliveries; recovery's response to them
does** -- escalating holds, capped at 30 steps. The policy's own waiting does
not matter. The rules cost something too, less than holds.

## 5. The late decline (episodes 51-60)

Deliveries 197 -> 183, efficiency 48% -> 44%, with, at the same time:
holds at the cap 49 -> 184 per episode, executed waits 61% -> 67%,
collisions 39 -> 45. Exploration had been at its floor since episode ~30, so
it is not exploitation setting in (the same argument as for cold_run24's
dip at 55). It is recovery: more collisions -> more escalated holds -> more
fleets standing still.

## 6. The recovery head -- the effect flagged in episode 1

- Preemptive recoveries halved around episode 21 (~65 -> ~31) and stayed
  there. Its strict success -- every moved fleet clear of the (wider, path-
  based) warning set one step later -- is almost never met, so the +9 bonus
  is rarely paid and preemption looks like pure cost.
- The head's value slid all run: -0.02 -> -0.44 (cold_run24: -0.32). It
  trains -- `loss_recovery` is non-zero in 60/60 episodes, ~1e-4.
- Collisions per 100 deliveries bottomed at ~12 (episode 13) and doubled to
  ~25 by the end, as preemption fell.

## 7. Other signals

- Episode 14: a recovery lockup -- 3,458 holds, 3,166 at the cap, 96% of
  at-risk decisions ended as waits, 129 deliveries. The only one in 60.
- Values: delivery 0.07 -> 0.40 (cold_run24 0.21), safety -0.002 -> -0.11
  (cold_run24 -0.20), efficiency 0.06 -> 0.47. The efficiency loss rose
  fourfold over the run (0.0037 -> 0.0165), as in cold_run24 -- worth
  watching, not yet a problem.
- The policy's own share of waits fell 26% -> 12% -> 15%: with the rules
  handling waits, the network learned to move.

## 8. What next

1. **Recovery reward: price the collision, drop the "clear" bonus** (a switch;
   calibrate cost/C with the coin test). It addresses sections 5 and 6
   directly: the head stops treating preemption as pure cost only if
   collisions cost it something.
2. **Hold escalation.** Holds at the cap are the costliest override (section
   4). With the rules on, the escalation built for version_unrefined may be
   too aggressive: a lower cap, or no escalation while the rules are active,
   is worth an A/B.
3. **cold_run26** (45 fleets, rescues on) as planned. One change per run keeps
   it readable: rescues in cold_run26, the recovery reward in a run of its
   own -- or both in parallel on AWS, against this run.
