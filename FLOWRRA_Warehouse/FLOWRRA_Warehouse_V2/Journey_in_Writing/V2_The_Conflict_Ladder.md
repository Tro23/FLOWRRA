# FLOWRRA v2 — The Conflict Ladder

Rohit Tamidapati · 5 October 2026 · updated 23 October 2026

> **Milestone, 20 October 2026: the final deploy test.**
> - **The test:** FLOWRRA's decentralised fleet against the MAPF planners PIBT and RHCR (reimplemented), all on the same floor and inside the same safety boundary. Every robot is a point on the graph, as in MAPF; FLOWRRA's boundary around each one (a minimum gap, warning bubbles, a braking distance) acts as its footprint. The same failures and people come on the same days.
> - **Scale:** 1,920 arm-days, four maps (three from the 3D MAPF warehouse dataset (Wang, Veerapaneni, Wu, Li and Likhachev, ICAPS 2024), plus a Kiva-style layout made for this project), 20–80 robots.
> - **Results:**
>   - **ladder4 delivered the most of the book in both conditions** (98.9% clean, 87.6% with failures and people).
>   - At 60–80 robots with failures and people it delivered firmly more than every planner.
>   - No arm had a collision worth counting (at most 0.016 a day).
>   - The planners were faster per order and kept further from people.
>   - The route-off networks trail ladder4: the next step.
> - **What it means:** a decentralised, learned fleet holds its own against centralised planners when every robot keeps a safety distance, as real AGVs do. Moving from the usual MAPF rules (no two robots in one cell, no swaps) into that boundary costs PIBT two-thirds of its throughput. Details in entry 44.
> - **Wording corrected the same day (Rohit):** the three warehouse maps come from the 3D MAPF warehouse dataset (Wang, Veerapaneni, Wu, Li and Likhachev, "MAPF in 3D Warehouses: Dataset and Analysis", ICAPS 2024), not from us. And in every system, ours included, a robot is a point on the graph: what FLOWRRA adds is the safety boundary around each point, and every arm ran inside it.
>
> **Status, 11 October 2026.** Steps 1–3 are built and tested, and runs 2–4 are complete. **The v2 champion is run 2's checkpoint: the ladder and rung costs, with no Learner.** In a seeded re-run of the shock benchmark it beat run 4 (the bounded Learner) on all five pre-registered small-map measures. On the large map it matches RULES alone and beats RHCR-PIBT + naive by 14 points of recovered orders, losing 60% fewer rescuers. The Learner stays in the code, switched off. The benchmark is now seeded per instance, after the same checkpoint scored 65.5% and 70.6% in two unseeded runs. People now share the floor (debris that a human comes to clear), orders can come from a fixed order book (the same work every episode, whatever the fleet count), and **run 5 (warm_stream1)** put the champion back to work with both, still learning: 4 of 6 predictions passed. People finished every job in all 50 episodes and deliveries held, but learning on the job made the fleet faster and slightly less safe, so run 2's checkpoint stays the champion until a frozen, like-for-like test (`deploy_eval.py`; built, predictions written, running now) says otherwise. The build log at the end records each step and what its runs showed.

## Why

The late storms in teacher\_run1 were not crashes: they were two controllers fighting over the same fleets, and neither looked at paths. Collisions stayed under 1 per episode late in the run; recoveries reached 249 in episode 110.

What the 115 episodes showed:

- **The recovery head became a coin flip.** Its value errors grew to about 2, while a needless ask costs about 0.06. It asked for recovery 69–458 times an episode with nothing at risk, and its preemptions cleared fleets 3–5% of the time, down from 30–50%. aws\_run4 shows a milder version (errors about 0.4, 77 wasted asks) without any teacher.
- **Holds locked RULES out.** RULES' action applies only to a fleet that is not held. In storm episodes, hold overrides outnumbered RULES overrides 5–12 to 1; in healthy ones RULES led about 5 to 1.
- **The tiers do not look at paths.** Tier 1 moves a fleet one cell by density, Tier 2 rewinds positions, Tier 3 picks the fleet closest to its goal. The fleets then steer back along the same converging routes.
- **Fairness without commitment livelocked.** Between fleets 7 and 73 in episode 110 the winner changed 30 times in 31 decisions. Each win lasted one 5-step hold, so neither got through.
- **The margin teacher drifted the values.** It pushed down moves that are never executed, so nothing pulled them back up; `qval_safety` fell from 0.0 to −6.6 while real safety penalties shrank.
- **A padding bug** fed dummy fleets into the recovery head's view of the floor in training. Fixed in step 1 (masked pooling).

## Principles

One ladder of path-aware RULES moves replaces the position-based tiers, and each learned head does one job.

1. **Three jobs in three heads.** Safety, Delivery and Efficiency forecast consequences. A Learner head learns RULES' judgement and speaks only inside conflicts. When the ladder itself fails, a fixed rule steps in (L3); the learned recovery head is parked. They combine only at choice time.
2. **RULES are the tiers.** Every intervention is a RULES move, chosen from the routes the fleets intend, not from where they stand.
3. **Freedom first.** A fleet moves as its policy chooses until its move conflicts on a path. The ladder steps in early, not after a crash.
4. **Physics.** Every move happens at driving speed, retreats included: half a cell per step, so one hop takes two steps. No jumps.
5. **Commitment.** Once a fleet is granted passage, it keeps it until it has cleared the conflict.
6. **Pay by consequence.** A fleet pays for the rung its move made necessary, never for following a rule.
7. **Values stay honest.** Nothing but real rewards ever moves a Q-value.

## Definitions

Five objects, all built from what the code already computes.

| Term | Meaning | Already in the code |
| --- | --- | --- |
| Route πᵢ | The next H = 8 cells fleet *i* intends, one set per step (ties branch) | `density._intended_path`, used by path warnings |
| Conflict group | Fleets whose routes meet within the horizon, joined transitively | Union-find in the simultaneous step |
| Wait-for graph W | An arrow *i → j* when *j* holds, or is committed to, a cell on πᵢ before *i* reaches it | Pair labels in `conflict_warehouse.py` (`head_on`, `blocked`, `following`, `contested`) |
| Retreat distance rᵢ | Hops *i* must reverse along its own trail until it is off every route in its group | New: trail = last positions, kept per fleet |
| Priority key | hops to goal − 0.25 × steps waited, then seniority, then a fixed tie-break | `ConflictRules.key` |

The key fact behind the ladder is standard in deadlock theory: **a group whose W has no cycle resolves by waiting alone; only a cycle needs someone to retreat.** Waiting never resolves a cycle, so at least one fleet in it must physically back up.

## Classification

Every conflict group is sorted by its size and the shape of its wait-for graph; that alone decides the lowest rung that can work.

| Group | Path label | Wait-for graph | What resolves it | Rung |
| --- | --- | --- | --- | --- |
| 2 fleets | following | one arrow, gap steady or growing | Braking only | L0 |
| 2 fleets | following, gap closing | one arrow | Brake the follower | L0 |
| 2 fleets | contested (one shared cell) | a chain A → B | Ordering: the lower key waits | L0 |
| 2 fleets | blocked (one is stationary) | a chain A → B | Wait; if the blocker is dead, route around (yield to stopped) | L0 |
| 2 fleets | head-on, at a corridor mouth | would become A ⇄ B | Corridor-entry hold: the entering fleet waits outside | L0 |
| 2 fleets | head-on, already committed | a 2-cycle A ⇄ B | One fleet retreats | L1 or L2 |
| 3+ fleets | any, no cycle | a chain or tree | Ordering, leaders first (topological order) | L0 |
| 3+ fleets | any, with cycles | one or more cycles | One retreat per cycle, then ordering | L1 or L2 |

Two-fleet conflicts are the common case and are fully covered by rules that already exist, except the 2-cycle outside a corridor. Multi-fleet conflicts are new territory: today they fall straight to the position-based tiers.

For 3+ fleets with cycles, the fewest-retreats choice is a feedback-vertex-set problem, which is hard in general. Groups here hold 2–5 fleets, so trying every option is cheap; above 6 fleets, break each cycle at its cheapest member, one cycle at a time.

## The ladder

Four rungs, each tried only when the one above cannot resolve the group; ordinary traffic never leaves L0.

<div align="center"> 
  <img src="The_Ladder_Conflict_v2.png" alt="The conflict ladder: L0 RULES ordering, L1 retrace, L2 pull over, L3 safety net" width="750"/> <br></br>
</div>

In the drawing, "one hop per step" means one hop at a time: fleets drive half a cell per step, so each hop takes two steps.

The triggers between rungs are structural (a cycle, a blocked trail, no progress in T steps), never a learned head's guess. Only L3 uses today's tiers, and Tier 2 stays there as the last resort.

## Rules

Four rules decide every rung: who retreats, how long a grant lasts, how waiting is rewarded, and who outranks whom.

**Who retreats: the cheapest retreat, not the furthest from goal.** If side A backs out, the cost is about twice its retreat (out and back) plus the time for B to pass; B barely waits. A side is the fleet and every fleet following it, as in today's corridor back-out. So retreat the side with the smaller total:

```math
R(S) = \sum_{f \in S} r_f, \qquad \text{retreat } \arg\min_{S \in \{A, B\}} R(S)
```

Ties go by priority key, then the fixed tie-break. Example: A is one hop outside a junction, B is inside the corridor heading out. R(A) = 1, R(B) ≥ 2, so A steps back one hop and B comes out.

**Commitment.** A grant holds until the conflict it settled is over: no member is still head-on with the fleet that gave way, or racing it for the same cell. A convoy or a queue that remains is left to braking and the queue rules, which are built for it. *Corrected 6 October 2026: the first version held until the two routes shared no cell at all. In a harmless convoy they keep sharing cells, so retreating fleets stayed parked and blocked others; in retrace_run2's first episode, blocked pairs tripled.* No new decision is made on that pair while a grant stands. Today's corridor retreat orders already work this way; the recovery tiers' alternation rule does not, which is what looped 7 and 73.

**One grace check before a retreat.** A cycle is spotted from routes up to 8 cells ahead, so it starts as a standoff, not a crash. On first detection, L0 holds both sides and their policies get one chance to break it; if the cycle is still there at the next check, L1 acts. A fleet that backs off on its own pays no rung cost, so Safety learns that resolving early is free. Exception: a cycle first spotted with the fleets already adjacent goes straight to L1.

**Aging, per conflict.** Every step a fleet is made to wait raises its priority by 0.25 hops, but only within its current conflict: the count resets once the fleet has been out of every warned pair for 3 steps. Aging exists to stop starvation inside a conflict. Carried across a whole trip, as today (reset only on delivery), an old wait lets a fleet outrank one a cell from its goal in an unrelated conflict. Without it, the fleet that clears soonest goes first, which minimises total waiting.

**Precedence.** From highest to lowest:

1. L3 safety-net actions (rare, only after the ladder has failed)
2. Ladder orders: retreats, pull-over holds, corridor holds, ordering waits
3. The fleet's own policy

A recovery hold never silences a ladder order. Holds themselves become ladder orders, so there is one authority in a warning zone instead of two. This removes the `and not _held` lock-out in `core_warehouse.py`.

## Why it terminates

With commitment and aging, a conflict group of n fleets clears within n grants, and no fleet waits forever. A sketch, to be turned into tests:

1. **No cycle.** Order the group by W (leaders first). The leader's next cell is free of the group, so it moves. Each committed grant ends with that fleet past every route it met, so the group shrinks by at least one.
2. **Cycle.** At least one fleet per cycle retreats along its own trail. The trail is cells it occupied, so those cells were reachable, and reversing frees the cell the next fleet in the cycle needed. The cycle becomes a chain, which step 1 clears.
3. **No starvation.** A waiting fleet's key falls by 0.25 per step, so after a bounded wait it outranks every competitor it can still meet.
4. **No livelock.** A grant cannot be reversed mid-passage (commitment), so the 7/73 pattern, winner changing every decision, cannot occur.

Where the argument can fail, and what catches it:

- A retreat trail is blocked, because another fleet is now standing on it. Then try the other side of the cycle.
- Both sides are blocked, or a group fails to shrink within T steps. Then fall to L3 and log it.
- A dead (failed) fleet sits on a trail. Dead fleets are fixed obstacles and are never part of a cycle; they route the live fleets around.

## The three heads

Each head has one job and one training signal, and only the choice step combines them.

| Head | Its job | Trained by | Used where |
| --- | --- | --- | --- |
| Safety, Delivery, Efficiency | Forecast the consequences of a fleet's own move | TD on real rewards only | Choice: weighted sum 6 : 4 : 1, as today |
| Learner (new, step 3) | Learn RULES' judgement: what the ladder would do here | Supervised, on every L0–L2 decision | Choice only, and only for a fleet inside a conflict group: its preference is added with weight β. Silent in free traffic, where it was never taught. Never inside a Q-value or a target |
| Recovery | Decide when the ladder has failed | TD on the holon signal; frozen while the ladder runs | Parked: L3 is a fixed rule. If L3 proves common, it can learn when to fire L3 and which tier, acting only if Q(act) − Q(none) > δ |

**Safety pays by rung.** Costs are charged to the fleets the event involved, through the per-fleet ledger that already runs in shadow mode:

| Rung | What happened | Safety cost |
| --- | --- | --- |
| L0 | Ordering, braking, corridor hold | 0 (normal traffic) |
| L1 | Retraced 1–2 hops | c₁ per hop retreated |
| L2 | Retreat to a pull-over cell | c₁ per hop + c₂ |
| L3 | Safety net fired | c₃ per fleet involved, about one collision |

Calibrated against the existing safety terms (fatal collision −50, worst warning step −1.2, both raw): **c₁ = −2.4 per hop**, because one hop takes two steps at half speed, so giving way costs what lingering in the worst warning would for the same time; **c₂ = −6** extra for leaving a corridor for a bay; **c₃ = −50**, what a collision costs. Each escalation is charged once, when it happens. All four values are settings (`training.rung_costs`), open to a paired dry run. Paying the retreating fleet only would punish the fleet that solved the conflict, so every fleet in the group shares the cost. The share is weighted by path: fleets in the conflict group pay 1.0, fleets whose route heads into the conflict's cells pay 0.25, and everyone else pays 0. Distance alone would charge bystanders, such as a fleet parked nearby or driving away. A fleet that routes itself around a jam pays nothing, so the cost teaches good routing, not just escaping.

**Recovery head: frozen under the ladder, with two fixes ready if it returns.** If unfrozen, it acts only when "act" beats "none" by a margin δ larger than its value noise, so a coin flip defaults to doing nothing. And it gets its own small encoder over the fleets' raw features, so the fleet heads' drift cannot change what it sees. A stop-gradient alone would not do this: it only stops the recovery head from disturbing the shared trunk, not the trunk from disturbing it.

## What happens to the teacher

The margin teacher is retired; its idea continues as the Learner head: the ladder teaches, the Learner learns, and a paired test decides whether that head earns its place.

- **Why Safety alone is not enough.** With the ladder, Safety pays only for escalations, so it learns what to avoid. L0 ordering costs nothing, so Safety has no reason to learn it, and RULES would stay as permanent traffic control. That is most of RULES' everyday judgement, and the policy would never absorb it.
- **Why both, not one.** The Learner learns what RULES would do here: dense, fast, but capped at RULES' level. Safety learns what happens if I do this: slower, but it is how the policy can beat RULES, by avoiding conflicts upstream before the ladder ever acts. The Learner speaks only inside a conflict group, where it has been taught; in free traffic the three consequence heads decide alone, so it never interrupts the flow.
- **The bleeding, made measurable.** As the Learner's agreement rises, RULES overrides fall. The override rate across training is the number that shows RULES moving into the policy.
- **The margin switch** (`training.teacher_margin.enabled`) stays in the code, set to off, so teacher\_run1 can be reproduced. Its write-up records it as tried and replaced, with the reason: it pushed on values that no experience could correct.
- **The test.** First the ladder and Safety costs with the Learner off; then the same run with the Learner on. If L0 overrides do not fall, the Learner head is not doing its job.

**Ladder context: letting fleets see the ladder.** The Learner can only learn a back-off if a fleet can tell it is in a situation that needs one, and today its observation cannot show that waiting will not help. Step 3 adds four signals to every fleet's observation. All four are knowable from its neighbours' broadcasts, so nothing becomes central.

| Signal | What it tells the fleet | Values |
| --- | --- | --- |
| In a cycle | Waiting alone cannot resolve my conflict | 0 or 1 |
| Steps in this conflict | How long my group has been stuck | Count, divided by T |
| Group's rung | How far the ladder has escalated | L0–L3, one-hot |
| Cheaper side to back out | My side's retreat cost against the other side's | Ratio; below 1 means my side is cheaper |

With these, the reflex becomes learnable from a fleet's own view, the way a good driver reads the road: facing another fleet in a single lane, closer to the junction, so reverse now, before the ladder has to order it. A back-off that starts before any ladder order also costs nothing (the grace check), so Safety rewards the same reflex the Learner copies. The signals enlarge the network's input, which v2 can absorb because it trains from scratch.

Tracked: **self-initiated back-offs**, where a fleet reverses with no ladder order. As the reflex forms, they rise while RULES-ordered retreats fall.

**Pair relations on the attention connections.** The four signals summarise a fleet's situation; the pairwise picture they come from can go into the network too. Graph attention already carries 4 numbers on each neighbour connection (`edge_dim=4`). Step 3 adds each pair's path-awareness label to that connection, so every fleet sees, neighbour by neighbour, how that neighbour relates to it:

| Relation | What it tells the fleet |
| --- | --- |
| head-on | Each route runs into the other: one of us must give way |
| contested | We reach the same cell at about the same time |
| blocked | My route runs into a neighbour standing still, or theirs into me |
| following | Same direction, one behind the other |
| wait direction | Whether I wait for this neighbour, it waits for me, or neither |

As numbers, that is a one-hot over the four kinds plus the wait direction, a handful of extra values per connection. It stays decentralised: each label comes from two neighbours' broadcast routes, exactly what the orchestrator already computes for the wait-for graph.

## Build order, switches and tests

Three steps, each behind its own switch and each off by default, so a run with every switch off reproduces today exactly.

| Step | What lands | Switch | Its own test |
| --- | --- | --- | --- |
| 1 (done) | Masked pooling for the graph-level heads | none: a bug fix | `test_graph_pooling.py` |
| 2a (done) | Ladder: one authority (no hold lock-out), wait-for graph, retreat by cheapest side, grace check, commitment, retrace one hop at a time at driving speed | `conflict.ladder` (`shadow`, `retrace`) | `test_ladder_authority.py`, `test_ladder_shadow.py`, `test_ladder_retrace.py` |
| 2b | Recovery head frozen under the ladder (done); margin δ and its own encoder only if it is unfrozen later | `conflict.ladder` (freezes it) | `test_recovery_frozen.py` |
| 2c (done) | Safety pays by rung, weighted by path | `training.rung_costs` | `test_rung_costs.py` |
| 3 (done; off in the v2 champion, build log 21) | Learner head, silent outside conflict groups, and the ladder-context signal in every fleet's observation, plus pair relations on the attention connections | `training.learner` | `test_learner_head.py` |

The run plan:

1. Paired dry runs per switch (2 episodes, seed 0, same instances, switch off vs on), as with the teacher margin.
2. One long run with 2a, 2b and 2c on and the Learner off, on all\_scens\_v5, both maps, 25/40/60 fleets.
3. The same run with the Learner on. The difference between 2 and 3 is the Learner head's contribution.
4. The frozen shock benchmark, with every arm re-run on the new code. RULES alone is re-run too, because the ladder changes RULES.

## Measurements and predictions

Each prediction is written before the runs and judged against teacher\_run1 on the same metric; a miss gets reported, not reworded.

| Prediction | teacher\_run1 (late) | Pass if |
| --- | --- | --- |
| Endgame loops end | max repeats per pair: 70 | ≤ 3 in every episode |
| No flip-flopping | winner changes per pair: 30 in 31 | ≤ 1 per pair per conflict |
| The ladder resolves conflicts itself | recoveries: 51 per episode | L3 fires < 2 per episode |
| The recovery head stops coin-flipping | wasted asks: 69–458 per episode | < 10 per episode |
| Values stay honest | qval\_recovery −15, qval\_safety −6.6 | both within ±1 for the whole run |
| Delivery holds | completion 97–99% | ≥ 97% in every 20-episode block |
| Safety holds | collisions under 1 per episode | no worse |
| Bleeding (run 3 only) | not measured | L0 overrides fall across training, while the Learner's agreement rises; self-initiated back-offs rise as RULES-ordered retreats fall |
| The verdict | policy + RULES 67–70% vs RULES 85–87% | on the re-run shock benchmark, policy + RULES ≥ RULES alone |

New counters the CSV needs: rung per intervention (L0–L3), retreat hops, grant flips per pair, groups by size and cycle count, time-to-clear per group, self-initiated back-offs (a fleet reverses with no ladder order), the four ladder-context signals, and why each retrace order timed out (reached its cell or not, whether the other fleet progressed, the pair's last relationship).

## Decisions

All six open questions are decided; δ and β take their values from paired dry runs.

- [x] **Recovery head at L3: a fixed rule first.** If a group has not shrunk within T steps, collapse. The learned head stays in the code, switched off, and every L3 event is logged; if L3 proves common, that data will train it. Training on denser maps just to make L3 common would defeat the ladder.
- [x] **Lifts: retrace allowed.** Edges are two-way, so a fleet can back up or down the way it came instead of going out, waiting and coming back in. A shaft of single-lane cells is a corridor, so the entry rule orders fleets at its mouth.
- [x] **Trail length: 8 cells,** the route horizon. Simple and light to compute.
- [x] **T, in steps: steps per hop × (2 × the group's largest retreat + one route length) + 10.** Fleets drive half a cell per step, so a hop takes 2 steps, and a 1-hop retreat gives T = 2 × (2 + 8) + 10 = 30. The same formula sets each retrace order's budget. *Corrected 6 October 2026: the first version counted hops as steps, and in retrace_run1, 6 of 22 retrace orders timed out while the other fleet was still passing at half speed.*
- [x] **δ and β.** δ sits above the recovery head's measured value noise, so a coin flip defaults to doing nothing. β starts small and is set by a paired dry run.
- [x] **Stream mode.** A fleet driving to its exit dock is a normal group member at full weight; at zero weight it would learn to push through on its way out. Once it has left the floor it drops out, and the group re-checks its cycles.

## Build log

Each step was built behind its own switch, tested, and run against the step before it on the same seed and the same 10 episodes (both maps, 25/40/60 fleets). Ten cold episodes are noisy: identical setups have swung from 12 to 24 collisions between reruns, so only large or consistent changes are read as effects.

1. **Step 1, masked pooling.** The recovery head averaged dummy padding rows into its view of the floor during training, but not when acting. Fixed; a padded state now gives exactly the live answer (`test_graph_pooling.py`).
2. **Step 2a-i, one authority.** The learned recovery head was switched off and L3 became a fixed rule (for now: a warned pair lasting 30 steps with neither fleet closer to its goal). Holds no longer silence RULES, and aging resets per conflict. In a paired dry run at 60 fleets, an episode that had 96 recoveries and 1,529 holds had 3 L3 calls and 2 holds instead.
3. **Step 2b, recovery head frozen.** Under the ladder its loss is left out of training, so its drifting values can no longer reach the trunk the fleet heads share (`test_recovery_frozen.py`). pooling_run1 had shown it drifts even without the teacher.
4. **Step 2a-ii, the shadow graph** (shadow_run1, measuring only). 79 wait-for cycles in 10 episodes: 97% two fleets facing each other, 59% cleared on their own within one check, 85% on corridor cells. RULES' yield-to-stopped guard steps aside from loops it cannot order; in the two episodes where it did, cycles lived 29 and 31 steps.
5. **Step 2a-iii (2a-iv folded in), retrace.** retrace_run1: longest cycle 31 → 8 steps, loops left by the guard 8 → 1, L3 fires 2 → 0. But 6 of 22 orders timed out.
   - *Fix 1, the budget in steps* (retrace_run2). The budget had counted hops as steps; fleets drive half a cell per step. Timeouts halved in the calm episodes, but the first, most exploratory episode cascaded.
   - *Fix 2, release when the conflict is over* (retrace_run3). Orders had waited for the two routes to share no cell, which a harmless convoy never satisfies. Now an order ends once nobody is head-on with the fleet or racing it for a cell. Result: all 49 orders released, 0 timeouts, longest cycle 6 steps, completion 99.4%, the best of the series.
6. **Step 2c, Safety pays by rung** (rung_run1). Safety's values stayed stable (−0.002 to 0.013), and no collision ever involved a retreating fleet: reversing does not cause contact. Collisions fell 24 → 10 and retreats needed 49 → 18, within run-to-run noise so far. To watch: L2 corridor back-outs carry 75% of all charges, and completion dipped in two episodes.

7. **Step 3a, the Learner head** (built and tested; first run with run 3). A small head reads the shared trunk but never trains it: its input is detached, and its gradients are clipped on their own, so it cannot even shrink the trunk's updates through a shared clipping limit (a leak `test_learner_head.py` caught in the first build). No Q-value can move because of it. It learns, by plain supervision, the action RULES chose on every RULES-decided step. At choice time it nudges only fleets inside a conflict: score = Q + β × its log-probability, with β = 0.3 to start. Off, no head is built, so older checkpoints still load. Also counted: **self-initiated back-offs**. With the Learner off, ladder_run1's network agreed with RULES about 20% of the time; that is the baseline step 3 should raise.

8. **Step 3b, ladder context** (built and tested). Seven values appended to every fleet's state: in a wait-for cycle; time in conflict (steps ÷ T, capped at 1); its group's rung (one-hot L0–L3, all zero outside a conflict); and whether its side is the cheaper one to back out (its retreat ÷ both sides', below 0.5 meaning it is). They are computed at the start of a step for the choice, and again after the move, from the same verdicts the reward uses, so a stored next state never carries the previous step's context. The state grows by 7, so runs with it on start cold. `state_layout()` now names the block, and it also gained the holon value it had been missing.

9. **The Learner's strength, β** (paired dry runs at 0, 0.1, 0.3 and 1.0). The Learner reached the same 82–89% agreement with RULES at every β; β only sets how often it overrides the consequence heads. At 1.0 it changed 77% of conflicted choices and cut self back-offs to a third, so the policy mostly copies RULES. At 0.3 it changed about a third, matched 1.0 on completion and collisions, and kept far more of the back-off reflex. **β = 0.3.**
10. **Step 3c, pair relations** (built and tested). Six numbers join the four already on each attention connection (closeness, path conflict, swap head-on, arrival order): head-on, contested, blocked or following from the path verdicts, plus "I wait for this neighbour" and "it waits for me", which swap when seen from the other side. Edge width 4 → 10, so runs with it on start cold.
    - *Found while building it:* the stored **next** state kept only the bare adjacency mask, so every value target was computed on a graph with no pairwise numbers, while every choice used them. That is the same train-versus-act mismatch as the padding bug, and it predates this work. Fixed behind `gnn.next_state_edge_features`, off by default so run 2 stays reproducible. It stays off for run 3 (see the ablation below), so run 3's difference from run 2 is step 3a + 3b alone.
    - *Also found, by the smoke test:* RULES and ladder orders, applied after the policy's obstacle check, could drive a fleet onto an obstacle. Every order must now be a valid move, or the fleet holds a step (`ladder_rule_orders_vetoed`).

11. **The step-3 ablation, which chose run 3's configuration.** Each piece was added one at a time, with everything else fixed: the ladder with shadow and retrace, rung costs, next-state edges off. Same seed, 5 episodes on the small map at 60 fleets. Each episode was cut at **300 steps, 3/8 of the 800 used in the long runs**, so completion here measures how far fleets got in that time, not whether they would have finished, and none of these numbers compare with the long runs.

    | 5 episodes, 300 steps | base (run 2's config) | + Learner (3a) | + context (3b) | + pair relations (3c) |
    | --- | --- | --- | --- | --- |
    | completion, average | 90.3% | 93.0% | 92.7% | 86.7% |
    | collisions | 51 | 54 | 45 | 55 |
    | retreats the ladder had to order | 80 | 49 | 34 | 102 |
    | Learner agrees with RULES | – | 71% → 80% | 82–84% | 78–83% |
    | self back-offs | 1,430 | 682 | 807 | 1,022 |
    | qval_safety | −0.004 | −0.019 | −0.015 | −0.027 |

    The Learner (3a) cut the retreats the ladder had to order by 39%: fleets increasingly did what RULES would do before the ladder stepped in. The context (3b) gave the fewest collisions and retreats and the highest agreement, and self back-offs rose again (682 → 807), which is the design's intent: a fleet that can see it is in a cycle, and is the cheaper side, backs off on its own. Pair relations (3c) gave the lowest completion and tripled the retreats. A hypothesis, not tested: the pair relations largely re-encode what the existing edge features (path conflict, swap head-on, arrival order) and the ladder context already carry, so they add inputs without adding information, and a fresh network learns more slowly. With next-state edges on as well, 3c did worse still (86 collisions against 55 in a paired dry run), so the next-state mismatch alone does not explain it.

    **Decision:** run 3 uses 3a + 3b (`learner.enabled`, `ladder.context`), with `ladder.pair_relations` and `gnn.next_state_edge_features` off. Both stay in the code behind their switches, for a later experiment of their own against run 3. The caveat: 5 short episodes in the densest setting, though the pattern was consistent across completion, collisions and retreats.

12. **Run 2, ladder_run1: the ladder and rung costs, Learner off** (150 episodes, both maps, 25/40/60 fleets, 800 steps; paired with teacher_run1 on the same instances).

    | Prediction | Pass if | teacher_run1 | ladder_run1 | |
    | --- | --- | --- | --- | --- |
    | Endgame loops end | ≤ 3 repeats per pair, every episode | max 70 | 2 of 150 episodes over 3 (78, 107; max 8) | ❌ narrowly |
    | No flip-flopping | ≤ 1 winner change per pair | 30 in 31 | not instrumented | – |
    | The ladder resolves conflicts | L3 < 2 per episode | – | 0.15 per episode (4 episodes ≥ 2) | ✅ |
    | No coin-flipping | < 10 wasted asks | 130 per episode, 345 late | **0 in all 150** | ✅ |
    | Values stay honest | within ±1 | qval_safety → −0.78 (−7.0 at worst) | −0.029 to −0.003 | ✅ |
    | Delivery holds | ≥ 97% per 20-episode block | | lowest block 98.05% | ✅ |
    | Safety holds | collisions no worse, late | 0.50 per episode (131–150) | 0.50 | ✅ |

    Over all 150 episodes: completion 98.8% against 98.0%, collisions 1.05 against 1.45 per episode, recoveries 0.15 against 19.8 per episode. Retrace issued 440 orders with no timeouts and no collision involving a retreating fleet. teacher_run1's worst storm, episode 110 (249 recoveries, a pair repeating 70 times), played on the same instance: **0 recoveries, 0 repeats, 100% completion.** With the Learner off, the network agreed with RULES 17% of the time: the baseline run 3 should raise.
    - *An unplanned replicate.* The run was launched three times by mistake; two copies were stopped at episodes 125 and about 130. They played the same instances on different trajectories, and the copy kept for the record matches the stopped one closely (completion 98.70% against 98.61%, collisions 1.16 against 1.04 per episode over 125 episodes), so the result replicates. Their 25-episode blocks differed by up to 0.9 collisions and 0.5 L3 fires per episode: **the measured noise band** for judging run 3. All numbers come from the kept copy's CSV; the shared log interleaves the three copies.
    - *Weak spots.* "No room to retreat" happened 42 times in 19 episodes: a fleet's way back along its own trail was occupied, so nothing could act until L3. That is the design's L2 case (pull over off both routes), which exists today only inside corridors. Extending it everywhere comes after run 3, to keep run 3's comparison clean. Separately, episodes where every remaining order finished before a stranded order's rescue could start ended with the order unrescued (episode 125); that predates the ladder and needs its own look before the shock benchmark.
    - *Still to come:* the verdict, policy + RULES against RULES alone on the re-run shock benchmark, using this run's checkpoint.

13. **Run 3, learner_run1: run 2 plus the Learner (β = 0.3) and the ladder context** (150 episodes, same instances; judged against run 2 with the replicate's noise band of about 0.9 collisions and 0.5 L3 fires per episode per 25-episode block).

    | | run 2 | run 3 |
    | --- | --- | --- |
    | RULES overrides, whole run | 7,547 | **3,338 (−56%)** |
    | RULES overrides per episode, first → last block | 68 → 40 | 32 → 15 |
    | Learner agreement with RULES | – | 81% → 87% |
    | completion | 98.8% | 98.6% |
    | collisions per episode (last 20 episodes) | 1.05 (0.50) | 1.32 (0.35) |
    | L3 fires per episode | 0.15 | **0.66** |
    | episodes with a pair repeating more than 3 times | 2 (max 8) | **11 (max 13)** |
    | qval_safety | −0.023 | −0.013 → −0.057 |

    **The bleeding prediction passed:** RULES had to step in 56% less, falling across training, while the Learner's agreement rose. **But the Learner was far louder than the β sweep suggested.** It changed 57% of conflicted choices in the first 10 episodes and 82–89% later, against 34% in the 2-episode sweep: as it grows confident its log-probabilities spread apart, and β × log-probability outweighs the small gaps between the heads' values. β = 0.3 is gentle only while the Learner is unsure. A hypothesis for the rise in L3 and repeats: the Learner learns only from steps RULES decided, but nudges every conflicted fleet, including ones RULES left free; having learned mostly "wait", two fleets can both wait, make no progress, and reach L3. Safety's lower values are honest: L3 charges (−50 per fleet) rose fivefold, from −2,238 to −11,138. Zero retrace timeouts and zero collisions while retreating, as in run 2.

    **Decision:** run 2 goes to the shock benchmark as the best configuration so far. The nudge is redesigned to be **bounded**: the Learner's probability (0 to 1) instead of its log-probability, so its pull on a choice can never exceed β, with β calibrated against the heads' 6 : 4 : 1 weights the way those weights were calibrated. The run's log is unreliable (binary bytes, stops at episode 40); all numbers come from the CSV.

    *Where the nudge actually changes behaviour.* For a fleet RULES decide, RULES' order is applied last anyway, so the nudge only changes what the policy **proposes**: overrides fall, the executed move does not change. For a conflicted fleet RULES leave free, the nudge decides the move, and that is exactly where the Learner has never seen a label. So in run 3 the Learner improved its agreement where it changed nothing, and steered where it was guessing. A bounded nudge confines that guessing to near-ties until RULES are actually withdrawn. Built: `learner.nudge = "prob"` (`"logprob"` reproduces run 3), and every run now logs the heads' own best-versus-second-best gap in conflicted choices (`learner_qgap_p25/50/75`), so β can be set at the typical near-tie gap rather than by feel.

14. **The shock benchmark: the design's final prediction** (ladder_run1's checkpoint; all three arms re-run on the ladder code; held-out `all_scens_v2`, 30 instances per map: seeds 0–9 × 25/40/60 fleets; 3 failure waves of 3 vehicles at 30%, 55% and 75% of the episode; paired Wilcoxon tests per instance; figure and tables from `generate_benchmark_figures_v2.py`).

    | 30 instances per map (small / large) | RULES alone | Policy + RULES | RHCR-PIBT + naive | Policy vs RULES, paired p |
    | --- | --- | --- | --- | --- |
    | Stranded orders recovered | 93.3% / 86.5% | 74.9% / **87.1%** | 69.6% / 71.9% | < 0.001 / 0.708 |
    | Rescuers lost per episode | 0.20 / 1.00 | 1.43 / **0.93** | 2.73 / 2.53 | < 0.001 / 0.710 |
    | Orders delivered | 97.3% / 97.2% | 94.3% / **97.5%** | 92.1% / 92.8% | 0.001 / 0.669 |
    | Hops to reach a stranded order | 23.5 / 47.8 | 50.6 / 65.3 | 23.2 / 75.6 | < 0.001 / < 0.001 |
    | Collisions per episode | 0.10 / 0.00 | 1.03 / 0.03 | 0.07 / 0.00 | < 0.001 / 0.317 |
    | Distance travelled (cells) | 434 / 1,266 | 1,080 / 1,523 | 421 / 1,326 | < 0.001 / < 0.001 |
    | Replan downtime (s per episode) | 0.0 / 0.0 | 0.0 / 0.0 | 2.1 / 46.0 | – |

    **Verdict on "policy + RULES at least matches RULES alone": passes on the large map, fails on the small one.** On the large map (27,000 nodes) the two are indistinguishable on recovery, rescuer losses and completion (p ≈ 0.7): a match, not an improvement. On the small map (1,435 nodes) the policy recovers 18 points fewer stranded orders and loses seven times as many rescuers (p < 0.001). RULES itself barely moved with the ladder code (identical results in 81–100% of instances, depending on the metric; RHCR identical in all), so the earlier RULES numbers still stand. The ladder was active in both RULES-based arms, which rules out the frozen recovery head as a cause: on the small map, the v1 policy arm fired about 128 recovery-tier actions per episode (43 spatial escapes, 31 rewinds, 54 yields), against 1.5 for ladder_run1's (no rewinds at all), and the RULES arm now shows a few L3 calls (1.6 per episode) where the v1 harness showed almost none.

    **Progress across checkpoints** (the 24 instances per map that all three benchmarks share):

    | small / large | v1 policy | teacher_run1 | ladder_run1 | RULES alone | ladder_run1 vs v1, paired p |
    | --- | --- | --- | --- | --- | --- |
    | Stranded orders recovered | 70.0% / 66.8% | 56.5% / 48.9% | 73.5% / 85.4% | 91.6% / 85.2% | 0.251 / < 0.001 |
    | Rescuers lost per episode | 1.50 / 1.71 | 2.00 / 1.75 | 1.46 / 1.04 | 0.25 / 1.08 | 0.825 / 0.007 |
    | Orders delivered | 93.0% / 93.0% | 87.7% / 85.9% | 93.7% / 97.1% | 96.6% / 97.0% | 0.472 / < 0.001 |
    | Hops to reach a stranded order | 84.0 / 103.3 | 106.6 / 184.4 | 52.8 / 68.2 | 23.5 / 49.1 | < 0.001 / < 0.001 |

    On the large map, the ladder checkpoint closed the whole gap to RULES (recovery 66.8% → 85.4%). On both maps it cut the hops to reach a stranded order by about a third. On the small map, recovery did not improve significantly. Scored against the README's v1 predictions: "recovery hops fall toward RULES'" now holds in direction (84 → 53 and 103 → 68), though still above RULES' 23 and 49; "frozen policy + RULES at least matches RULES alone" holds on the large map only.

    **Why the small map fails, a hypothesis, not yet tested.** There the policy drives 2–3 times RULES' distance at every fleet count (745 against 370 cells at 25 fleets) and takes about twice the hops to reach a stranded order (51 against 23). Rescuer losses do not track collisions across instances (correlation −0.43), so the likelier link is exposure: a rescuer on the road twice as long is more often caught by the next failure wave. The detours predate the ladder (v1 took 84 hops); the ladder checkpoint shortened them but not enough. A possible contributor: L2 corridor back-outs carried 75% of the rung charges in training, which could teach fleets to avoid the small map's narrow aisles. Separating the two is the next diagnostic.

15. **Calibrating the bounded nudge, and next-state edges** (5-episode paired dry runs, small map, 60 fleets, each cut at 300 steps; all with the ladder, context and rung costs on, pair relations off).

    | 5 episodes | no Learner | unbounded, β 0.3 | bounded, β 0.3 | bounded, β 0.1 | β 0.1 + next-state edges |
    | --- | --- | --- | --- | --- | --- |
    | completion, average | 90.3% | 92.7% | 90.0% | 94.0% | 94.3% |
    | collisions | 51 | 45 | 41 | 49 | 38 |
    | L3 fires | 10 | 14 | 5 | 5 | 4 |
    | retreats the ladder had to order | 80 | 34 | 52 | 67 | 44 |
    | self back-off rate | 32% | 12% | 28% | 27% | 29% |
    | conflicted choices the Learner changed, by episode 5 | – | 63% | 31% | 14% | 10% |
    | Learner agrees with RULES | – | 83% | 80% | 80% | 85% |

    - *The bound works.* At β 0.3 the bounded nudge changed a quarter as many choices as the unbounded one, L3 fell to the lowest seen (5), and the back-off reflex returned to near the no-Learner level. But its share still climbed (3% → 31%) and completion slid every episode (95% → 87%), because the heads' own best-versus-second-best gap was shrinking (0.133 → 0.068) while the Learner grew confident.
    - *β from the heads' gap.* The gap's median was about 0.1, so **β = 0.1**. There the Learner's share held at 2–14%, completion stayed at 92–97%, and the heads' gap stopped shrinking (it settled near 0.1). A hint, from one pair of runs, that a loud nudge also flattens the heads' own preferences.
    - *Next-state edges, tested alone.* At β 0.1, turning them on gave the fewest collisions (38 against 49), the fewest L3 fires (4 against 5), fewer ordered retreats (44 against 67) and the highest agreement with RULES (85%). Per episode the ranges overlap, so no single number is decisive, but every measure points the same way. This reverses the earlier dry run where they hurt; that run also had the pair relations on, which points to the pair relations as the cause there.

    **Run 4, learner_run2: run 3 with the bounded nudge (β = 0.1) and next-state edge features on** (150 episodes, same instances; config header records every switch). It changes two things at once against run 3, and the evidence for each is the dry pairs above. **Predictions, written before the run:**

    | Prediction | Pass if | run 2 | run 3 |
    | --- | --- | --- | --- |
    | The Learner stays a voice | conflicted choices changed stay below 20% in every 25-episode block | – | 57% → 89% |
    | No return of stalemates | L3 fires ≤ 0.4 per episode in every 25-episode block (run 2's worst block: 0.36) | 0.15 overall | 0.66 overall (worst block 1.24) |
    | The bleeding holds | RULES overrides fall across training, below run 2's | 7,547 | 3,338 |
    | Delivery holds | ≥ 97% in every 20-episode block | 98.05% lowest | 98.03% lowest |
    | Values stay honest | qval_safety and qval_recovery within ±1 | −0.029 to −0.003 | −0.057 to −0.004 |
    | The verdict | re-run shock benchmark: policy + RULES ≥ RULES alone on the large map, and closer than 74.9% vs 93.3% on the small map | | |

    *Pair relations, re-tested under run 4's settings before it started* (bounded nudge at β 0.1, next-state edges on; 10 episodes, 800 steps, small map, 60 fleets). The earlier evidence against them came from runs with the unbounded nudge, so it might not carry over. The decision rule was fixed before either run finished: five measures, the majority wins. **Off won, 4 to 1:** collisions 64 against 82, L3 fires 13 against 18, ordered retreats 130 against 214, longest repeat of one pair 7 against 9; completion favoured on (96.7% against 95.8%). Off was also steadier: with pair relations on, qval_safety drifted down every episode (−0.032 → −0.061, including late episodes with almost no L3 or collisions) and the safety loss was twice as high. On improved fastest late (99.0% completion over its last five episodes), so a long run of its own may be worth it someday. Run 4 keeps pair relations off, as pre-registered.

    No prediction is made that run 4 closes the small-map gap. The Learner speaks only inside conflicts, while the small-map shortfall looks like routing in free traffic (detours of 2–3 times RULES' distance), which the Learner never touches. That gets its own diagnostic, run alongside.

16. **Run 4, learner_run2: scored** (150 episodes, same instances as runs 2 and 3).

    | Prediction | Pass if | run 2 | run 3 | run 4 | |
    | --- | --- | --- | --- | --- | --- |
    | The Learner stays a voice | changes < 20% of conflicted choices in every 25-episode block | – | 57% → 89% | 12, 25, 28, 30, 34, 28% | ❌ |
    | No return of stalemates | L3 ≤ 0.4 per episode in every 25-episode block | 0.15 (worst block 0.36) | 0.66 (worst 1.24) | 0.16, 0.24, 0.36, 0.28, **0.96**, 0.04 | ❌ narrowly: 5 of 6 blocks |
    | The bleeding holds | RULES overrides fall, below run 2's | 7,547 | 3,338 | 8,012 | ❌ |
    | Delivery holds | ≥ 97% in every 20-episode block | 98.05% lowest | 98.03% lowest | 98.11% lowest | ✅ |
    | Values stay honest | within ±1 | −0.029 to −0.003 | −0.057 to −0.004 | qval_safety −0.050 to −0.012, qval_recovery 0.07 to 0.25 | ✅ |
    | The verdict | benchmark: ≥ RULES on the large map, closer than 74.9% vs 93.3% on the small | | | large 82.6% vs 86.5% (p = 0.29), a match; small 71.2% vs 93.3%, further not closer (seeded, entry 20) | ❌ |

    Over the whole run, run 4 sat between runs 2 and 3 on every measure: completion 98.6%, collisions 1.19 per episode (run 2: 1.05, run 3: 1.32), L3 0.34 (0.15, 0.66), episodes with a pair repeating more than 3 times 7 (2, 11). Its last 25 episodes were the best of any run: L3 0.04 and collisions 0.36 per episode, RULES overrides 23.5 per episode against run 2's 40.0. *Why the Learner's share kept rising:* the heads' own best-versus-second-best gap shrank to 0.042 by the end, so a fixed β of 0.1 grew to 2.4 times the typical near-tie, and the bounded nudge decided more choices as training went on.

17. **Tying β to the heads' gap** (built: `learner.beta_mode = "gap"`, β = `beta_scale` × the running median of the heads' gap over the last 5,000 conflicted choices, with the fixed β until 200 are seen; `test_learner_head.py`). Paired against fixed β 0.1 (10 episodes, 800 steps, small map, 60 fleets, same seed). The rule was fixed before the run: adopted only if it wins at least 3 of 5 measures.

    | 10 episodes | fixed β 0.1 | gap-tied β |
    | --- | --- | --- |
    | completion | 95.8% | **96.3%** |
    | collisions | **64** | 82 |
    | L3 fires | **13** | 14 |
    | retreats the ladder had to order | **130** | 205 |
    | longest repeat of one pair | 7 | **5** |

    **Fixed won, 3 to 2; not adopted.** The gap-tied β did what it was built to do: the Learner's share held at 5–11% per episode (fixed: up to 28%) as the effective β eased from 0.131 to about 0.09. But fewer Learner changes came with more collisions and 58% more ordered retreats. The switch stays in the code, default `"fixed"`.

18. **β around 0.1, with a narrower exploration schedule** (dry runs without `--cold-start`, so most of each run exploits; run 4's settings, 10 episodes, same seed; scored with `five_measure_rule.py`, which applies the 5-measure rule to any two CSVs).

    | 10 episodes | β 0.08 | β 0.1 | β 0.117 (reference) | β 0.15 |
    | --- | --- | --- | --- | --- |
    | completion | 96.3% | 96.8% | 96.2% | 96.2% |
    | collisions | 74 | 72 | **51** | 85 |
    | L3 fires | 18 | 19 | **13** | 19 |
    | retreats the ladder had to order | 152 | 142 | **99** | 115 |
    | longest repeat of one pair | 11 | 4 | 5 | 8 |
    | Learner's share of conflicted choices | 5.9% | 9.8% | 11.3% | 15.6% (up to 40%) |
    | score against β 0.117 | 1–4 | 2–3 | – | 1–4 |

    β 0.117 led every other value on collisions, L3 fires and retreats. Not taken to a long run: with several candidates, a 3–2 win can come from luck, and the benchmark (entries 19–20) was answering a different question by then.

19. **The benchmark was noisy; it is now seeded.** Run 4's checkpoint at β 0.1, benchmarked twice with identical settings, recovered 65.5% and then 70.6% of stranded orders on the small map. The cause: the evaluation ε (0.015) takes its random moves from Python's `random`, which the benchmark never seeded, so each run made different random moves, and on the crowded small map a few different moves change an episode. **Fix** (`benchmark_all.py`): before every run, `random`, numpy and torch are seeded from (map, seed, fleet count), the same for every arm, so arms face the same draws until their choices differ. Verified: wherever the Learner was never consulted, the three β arms below produced identical rows. Also added: `--betas` (several Learner strengths on one checkpoint, no training) and `--no-seed` (the old behaviour). *What it means for earlier results:* RULES (ε = 0) and RHCR make no random moves, so their numbers stand; a single unseeded FLOWRRA result carries about ±5 points of small-map recovery. Run 2's 74.9% re-ran at 73.9% seeded. The seeded runs used **780 steps** (the config's value at launch) against 800 for entry 14's RULES and RHCR rows, which if anything favours those two.

20. **Voice or network?** Run 4's checkpoint did worse than run 2's on the small map. Was that the Learner's nudge at choice time, or what the network had learned? Test: run 4's checkpoint, seeded, with β set to 0.1, 0 and 0.117 at test time, no training. The rule was fixed before the run: if β 0 recovers at least 70% of stranded orders on the small map (halfway between run 4's 65.5% and run 2's 74.9%), the voice was the cause; below 70%, the network.

    | Small map, 30 instances (p against β 0.1) | β 0.1 (trained) | β 0 (muted) | β 0.117 |
    | --- | --- | --- | --- |
    | stranded orders recovered | **71.2%** | 64.2% | 64.1% |
    | orders delivered | **92.3%** | 90.6% | 91.3% |
    | collisions per episode | **2.17** | 3.20 (0.09) | 3.40 (0.02) |
    | rescuers lost per episode | **1.47** | 1.67 | 1.97 (0.02) |
    | safety-tier actions per episode | **3.9** | 7.4 | 7.6 (0.03) |
    | longest repeat of one pair | **0.47** | 1.07 (0.08) | 1.13 (0.03) |
    | choices the Learner changed, per instance | 134 | 0 | 238 |

    **β 0 recovered 64.2%: the network.** Run 4's network came to rely on the nudge it trained with; muting it, or raising it by 17%, both made it worse. On the large map the three arms were within a point (82.6%, 83.1%, 82.6%) and identical in 26 of 30 instances, because the Learner was consulted about 20 times per instance and changed about 3 choices, against about 600 and 130 on the small map: its influence scales with how dense the conflicts are. *The lesson:* a component that is on during training cannot be switched off at deployment for free.

21. **Run 2, seeded: the v2 champion.** The rule was fixed before the run: run 2 against run 4 at β 0.1, paired, on the small map (where they can differ), five measures; 3 or more wins confirm run 2. The large map is reported, not scored.

    | Small map, 30 paired instances | run 2 (ladder only) | run 4 (β 0.1) | p | |
    | --- | --- | --- | --- | --- |
    | stranded orders recovered | **73.9%** | 71.2% | 0.49 | run 2 |
    | orders delivered | **94.7%** | 92.3% | 0.009 | run 2 |
    | collisions per episode | **1.30** | 2.17 | 0.006 | run 2 |
    | rescuers lost per episode | **1.37** | 1.47 | 0.69 | run 2 |
    | longest repeat of one pair | **0.27** | 0.47 | 0.26 | run 2 |
    | *distance travelled (cells)* | *1,124* | *1,445* | *< 0.001* | |
    | *safety-tier actions per episode* | *2.3* | *3.9* | *0.02* | |

    **Run 2 wins 5 to 0;** completion and collisions are significant on their own. On the large map it also leads on 4 of 5 (recovery 85.7% against 82.6%, completion 97.2% against 96.4%, rescuers lost 1.00 against 1.10), with collisions tied.

    **The champion against the baselines** (run 2 seeded at 780 steps; RULES and RHCR from entry 14 at 800 steps; paired by instance):

    | small / large | Run 2 (ladder) | RULES alone | RHCR-PIBT + naive |
    | --- | --- | --- | --- |
    | Stranded orders recovered | 73.9% / **85.7%** | 93.3% / 86.5% | 69.6% / 71.9% |
    | Rescuers lost per episode | 1.37 / **1.00** | 0.20 / 1.00 | 2.73 / 2.53 |
    | Orders delivered | 94.7% / **97.2%** | 97.3% / 97.2% | 92.1% / 92.8% |
    | Collisions per episode | 1.30 / 0.07 | 0.10 / 0.00 | 0.07 / 0.00 |
    | Distance travelled (cells) | 1,124 / 1,519 | 434 / 1,266 | 421 / 1,326 |
    | Replan downtime (s per episode) | 0 / 0 | 0 / 0 | 2.1 / 46.0 |

    On the large map, run 2 and RULES are indistinguishable (recovery p = 0.91, rescuers lost p = 0.97, completion p = 0.92), and both clearly beat RHCR (recovery p = 0.001, rescuers lost and completion p < 0.001). On the small map, run 2 loses half as many rescuers as RHCR (p < 0.001) and delivers more (p = 0.009), with recovery level (p = 0.09); RULES alone stays well ahead of both.

    **Decision.** Run 2's configuration is the v2 champion: the ladder (shadow, retrace) and rung costs, with the Learner, ladder context, pair relations and next-state edges off. The Learner and the other switches stay in the code. If the Learner returns, it is on from the start of training, or runs in shadow (logging what it would choose, with no voice) during a warm phase; it is never switched on or off around a trained network. The small-map gap to RULES (73.9% against 93.3%) remains the open problem, and entry 14's hypothesis (exposure from detours: 2.6 times RULES' distance) still stands untested.

22. **People on the floor, and the order stream** (built and tested before run 5).
    - *Obstacles, "cleanup" mode* (`obstacles_warehouse.py`). Debris appears during the episode (about once every 200 steps, at most 2 at a time, never on a fleet, a goal, a dock or a dock's neighbours). Ten steps later a human enters at the edge of that floor, walks a shortest path to it, clears it in 5 steps and walks out to another edge cell. A human never steps onto a fleet: blocked, they wait, look for a way round after 2 blocked moves, and with no way round step aside after 6. Nothing is permanent: debris nobody can reach is cleared after 1,000 steps, and a human boxed in for 2,000 steps leaves. Humans have absolute priority; the ladder (fleet to fleet) never asks one to move. Obstacles cost no state dimensions: the fleets perceive them through the six `ray_hit_unknown` inputs reserved for this, through the density field's repulsion, and through the hard veto on moves. The old "wander" mode is kept and behaves exactly as before.
    - *Two bugs the integration test found, both older than this work, both invisible until obstacles were switched on:* (1) **the veto checked the wrong cell.** It rounded the half-step position, and Python rounds halves to even: from x = 2, half a step toward 3 is 2.5, and `round(2.5)` is 2, the fleet's own cell. From every even coordinate moving up, or odd one moving down, a fleet could drive halfway into a human. The veto now checks the next whole cell along the move (`test_obstacle_veto.py`). (2) **The livelock escape ignored the veto.** A fleet stalled for a while is handed to the goal gradient for a burst of steps; a fleet waiting behind a human counts as stalled, and the gradient points straight through them. Moves into an obstacle are now dropped from that choice; with none on the floor it picks exactly as before (checked on about 200,000 random gradients).
    - *Also:* a human on a dock now delays a fleet's re-entry there; the obstacle stream is seeded per episode (it used an empty config seed, so placements differed on every rerun); a dead-end test caught the floor detection treating rows as floors on single-floor maps. With all of it in place: no fleet ever on an obstacle, no human ever on a fleet, identical obstacle histories on identical seeds (`test_obstacles_integration.py`).
    - *Left out on purpose:* a retreat checks its way back for fleets, not people. A human on a retreating fleet's trail delays the retreat until they walk on; the retreat-timeout counter will show whether that ever matters.

    **Run 5, warm_stream1: the champion keeps learning on the job, with the order stream and people clearing debris** (50 episodes, warm from run 2's checkpoint, both maps, 25/40/60 fleets, 800 steps; every switch the network trained with is run 2's; new: stream with a fixed order book, obstacles, gentle exploration peaking at 0.05). **Predictions, written before the run** (first 25 episodes against the last 25):

    | Prediction | Pass if |
    | --- | --- |
    | Safety holds while it learns | fleet-to-fleet collisions per episode, last 25 no worse than first 25 |
    | The ladder still resolves | L3 ≤ 0.4 per episode in both 25-episode blocks (run 2's worst block: 0.36) |
    | Throughput holds | share of the order book delivered, last 25 at least 95% of the first 25 *(revised before the run; see below)* |
    | Values stay honest | qval_safety within ±1 throughout |
    | People always finish the job | in every episode, no debris expires and no human times out: the fleets always let them through |
    | Retreats are not blocked by people | retreat timeouts ≤ 1% of retreat orders |

    Reported, not predicted: order lead time (release to delivery), the step the whole book was delivered, near misses (a fleet next to a human), human waiting steps, reroutes and give-ways, collisions per 100 deliveries, and every comparison again within each map and fleet count (each episode's map and fleet count is drawn at random, and one small-map 60-fleet episode can outweigh several others, so a half can look worse from an unlucky draw). Fleet-to-human contact is zero by construction (the veto), and checked by the integration test, so it is not a prediction.

    **The shakedown, and a change of workload made before the run.** The predictions above were first written for the *closed loop*, where every robot always has an order, and pushed before a 3-episode shakedown. The shakedown ran: no crash, people finished every job (one waited 273 steps and stepped aside 13 times in a crowded episode, but cleaned up and left), retreat timeouts 2 of 541 orders. Large map, 40 fleets: 0 collisions, efficiency 58%. Small map: 40 fleets delivered 95 orders per 800 steps and 60 fleets 94, with 61 and 157 collisions and 9 and 65 L3 fires. Twenty more robots added no deliveries, only congestion: in a closed loop nobody parks, so the small map (about 290 cells per floor) stays full all episode, which run 2 never saw. That also showed a flaw in the measure: the closed loop gives each robot an order, so the work grows with the fleet, and efficiency (deliveries ÷ fleets × steps × 0.5 ÷ cycle length) assumes every robot drives conflict-free all the time, so it falls whenever robots are added. **The workload was therefore changed, before the run, to a fixed order book** (`stream.order_book`, `test_order_book.py`): 80 orders per episode whatever the fleet count, the ones the fleets start with counting toward it and the rest released steadily over the first 400 steps; after a delivery a fleet exits and re-enters only when an order is waiting, otherwise it waits at its dock; the episode ends when the book is delivered. That is the operator's question, "this many orders this shift: how long, with how many robots?", and the same work every episode. Only the throughput prediction changed, from efficiency to the share of the book delivered. The L3 prediction stays as written, although the shakedown suggests it will fail. A second 3-episode shakedown checked the order book before the run, with releases spread over 500 steps:

    | Shakedown 2 (order book of 80) | small map, 40 fleets | large map, 40 fleets | small map, 60 fleets |
    | --- | --- | --- | --- |
    | delivered | 71 (89%) | 76 (95%) | 77 (96%) |
    | lead time, mean / 90th percentile (steps) | 127 / 292 | 189 / 314 | 93 / 206 |
    | robots on the floor (busy) | 58% | 82% | 36% |
    | collisions (closed-loop shakedown) | 26 (61) | 1 (0) | 16 (157) |
    | L3 fires (closed-loop shakedown) | 7 (9) | 1 (1) | 0 (65) |

    With the same work whatever the fleet size, 60 fleets on the small map went from 157 collisions and 65 L3 fires to 16 and 0: robots with no order wait at the docks instead of crowding the floor. People again finished every job (no debris expired, nobody timed out). No book finished within 800 steps: an order released near step 500 needs up to about 300 steps (the 90th percentile lead time), so **the release window was shortened to 400 steps**, before the run. Efficiency (deliveries ÷ the conflict-free ideal) is left blank under an order book, because its ideal assumes every robot always has work; the book reports instead the share delivered, lead time, the share of robot-steps on the floor, and deliveries per 1,000 robot-steps (× 3.6 ÷ seconds per step gives orders per robot per hour). *For scale:* if one step is one second (0.5 m/s, 1 m cells), 800 steps is 13.3 minutes, and the shakedown's 95 deliveries are about 430 orders per hour.

23. **Run 5, warm_stream1: scored** (50 episodes, 5.2 hours, no errors; 10 robots failed per episode on average, people cleared 182 pieces of debris).

    | Prediction | Pass if | first 25 | last 25 | |
    | --- | --- | --- | --- | --- |
    | Safety holds while it learns | collisions per episode, last 25 no worse than first 25 | 12.6 | 13.4 | ❌ narrowly |
    | The ladder still resolves | L3 ≤ 0.4 per episode in both blocks | 2.96 | 3.36 | ❌ (the shakedown had suggested it) |
    | Throughput holds | share of the book delivered, last 25 ≥ 95% of first 25 | 92.5% | 90.6% (97.9% of first) | ✅ |
    | Values stay honest | qval_safety within ±1 | −0.045 → −0.164 | | ✅ (still drifting down) |
    | People always finish the job | no debris expires, no human times out, in every episode | 0, 0 | 0, 0 | ✅ all 50 episodes |
    | Retreats are not blocked by people | retreat timeouts ≤ 1% of orders | 4 of 2,364 (0.17%) | | ✅ |

    **4 of 6 pass.** *Within each map and fleet count* (averaged over the six combinations, reported, not scored): collisions 11.0 → 14.1 per episode, L3 2.5 → 3.8, lead time 145 → 134 steps, book delivered 92.4% → 90.5%. **Learning on the job made the fleet faster but slightly less safe:** orders arrived about 8% sooner (lead time down in 5 of 6 combinations) while collisions and L3 rose. A midway look at 20 episodes had suggested the opposite; the full run did not bear it out. qval_safety fell steadily across the run without levelling off, so the values are still adjusting to stream traffic. People: near misses rose from 18 to 22 steps per episode and human waiting from 8 to 14 steps, with 0 → 0.36 give-ways; no debris ever expired. Only 2 of 50 books were finished within 800 steps; about 1.3 orders per episode were still waiting at a failed robot for a rescuer at the end.

    **How many robots this much work needs** (the book allows at most 360 orders an hour, at one step ≈ one second):

    | orders per hour (share of the book) | 25 robots | 40 robots | 60 robots |
    | --- | --- | --- | --- |
    | small map | 317 (88%) | 320 (89%) | **338 (94%)** |
    | large map | 295 (82%) | **342 (95%)** | 345 (96%) |

    Forty robots carry this workload on the large map; the small map needs sixty, and with forty its aisles are the most crowded (more robots on the floor at once, the most collisions); twenty-five is too few on both.

    **Decision:** run 2's checkpoint stays the champion until a like-for-like test says otherwise. That test is next: both checkpoints frozen, on identical seeded instances.

24. **The frozen, like-for-like test (`deploy_eval.py`).** Each instance is a map, a fleet count and a seed. It uses the seed-th scenario file of the held-out set (`all_scens_v2`), its first 25, 40 or 60 fleets, and a seeded order book: 80 orders released over 400 steps, with 800 steps to deliver them. Before every arm runs an instance, every random stream is reset, including those behind failure waves, single failures, debris and people. Every arm therefore starts the same day; once their choices differ, details drift apart, as they would on a real floor. Nothing learns. Exploration is fixed at 0.015 for the networks and 0 for RULES.

    **Three arms:**
    - **champion:** run 2's checkpoint.
    - **warm:** run 5's checkpoint.
    - **RULES:** the same orchestrator with shortest-path driving. Its rules are what the ladder falls back on at L3.

    **Two conditions:**
    - **Full:** failures and people on, 5 seeds. That is 2 maps × 3 fleet counts × 5 seeds = 30 instances, 90 runs.
    - **Clean:** both off, 3 seeds: 54 runs.

    Arms are compared in pairs over shared instances with the Wilcoxon signed-rank test. Here orders per hour equals the share of the book × 360 (80 orders, 800 steps, one step ≈ one second), so it is not a separate measure.

    **Predictions, written before the runs:**

    | Prediction | Pass if |
    | --- | --- |
    | The test is repeatable | `--check` runs one instance twice and reports IDENTICAL |
    | A clean floor delivers the book | clean condition: every arm delivers at least 97% of the book (mean over instances) |
    | RULES is the safest driver | RULES has fewer collisions per episode than both networks, in both conditions; a tie fails (in the shock benchmark, RULES had 0.10 per episode on the small map against the champion's 1.03) |
    | Learning on the job traded safety for speed | full condition: warm's mean lead time is shorter than the champion's and its collisions are higher, as in run 5 |

    **Decision rule, champion against warm (full condition):** warm becomes the champion for order-stream work only if it wins at least 3 of 5 measures. The five are share of the book delivered, lead time, collisions, L3 fires, and near misses (a fleet next to a human).

    **Agreed in advance:** suppose RULES matches or beats both networks on share delivered and collisions in the full condition. Then the reading is that the architecture (rules, ladder, rescue) is what keeps the fleet whole, and the learned policy has not yet earned its place in deployment. That supports the view of FLOWRRA as a self-maintaining system and sets the policy a clear bar to clear.

    **Amended before any runs (11 October).** The check and the trace (entry 25) found a movement bug. Both runs will use the fixed code, and a fourth arm is added. No clean or full run had been made.
    - **The fix:** `node_warehouse.reach_position`, the line-end trap. It changes movement for every arm, the same way.
    - **The fourth arm, champion+route:** run 2's network with route by default. A robot with no other robot within the warning distance takes its shortest-path move, and the network decides only near other robots.

    The predictions above stand unchanged, and these are added:

    | Prediction | Pass if |
    | --- | --- |
    | Route by default delivers more | full condition: champion+route delivers a larger share of the book than the champion (mean over instances) |
    | ...at no cost in safety | full condition: champion+route has no more collisions per episode than the champion |

    **Added before the full run (11 October, after the clean run):** 15 fleets, in both conditions (5 seeds full, 3 clean), in separate runs and CSVs. With 15 robots the book is beyond the fleet's capacity, so the share delivered measures driving efficiency directly. A failure wave of 3 robots is a fifth of the fleet. Reported only; not part of the predictions, which stay on 25, 40 and 60 fleets.

    **Reported, not predicted:** champion+route against RULES on the five measures. That comparison is the real question. With routing taken away from the network, does it add anything over RULES where robots meet? If it wins none of the five, the network adds nothing yet, even in contention.

25. **The detour habit (traced; open).** The check run for entry 24 (one instance, the first 120 steps) found the champion slow on the same orders RULES finished quickly. RULES delivered all 25 starting orders by step 37; the champion had delivered 16 by step 120. Fleet 22's trip took RULES 5 steps and the champion 87. It is not new: in the shock benchmark the champion drove 1,080 cells to RULES' 434 on the small map. The training records show the habit directly. These are the network's own proposals when another robot was within 2 cells:

    | Network's proposals, another robot within 2 cells | toward its goal | wait | away from its goal |
    | --- | --- | --- | --- |
    | run 2, the champion (150 training episodes, 31,941 decisions) | 49.7% | 18.5% | 31.8% |
    | run 5, warm (50 stream episodes, 106,377 decisions) | 46.4% | 17.6% | 35.3% |

    One hop from its goal, it stepped in 69% of the time in run 2 and 36% in run 5. It waited 12% and 27% of the time, and stepped away 19% and 37%. Over all moves, it took the shortest-path direction 71% of the time in run 2 and 63% in run 5; RULES always does.

    **Suspected cause, in the reward (to be tested).** Per unit of reward, the safety head weighs three times as much as progress (6:4:1 on value-normalised heads). It charges a robot −0.5 for every step it spends within 2 cells of another, whoever moved (the integrity part, `coh` in `_priority_rewards`). It adds an approach charge when the robot closes the gap. The approach-only switch (`approach_warning`) replaced the warning term, not this one. When the neighbour is in the way, the three choices compare like this:
    - **Advance:** earns progress but keeps the charge and adds the approach charge.
    - **Wait:** keeps the charge and adds the idle penalty.
    - **Step back:** costs a little progress, which is won back later, and can end the charge.

    Near another robot, then, a step back is often the cheapest move, and on a crowded floor each step back tends to put the robot next to someone else. RULES ignores all of this: it drives the shortest path and leaves conflicts to the ladder.

    **The test:** `trace_fleet.py` replays the check instance. For every robot-step it records:
    - what the network proposed and what ran;
    - what decided it: rules order, ladder retreat, recovery hold, livelock escape, tabu swap, or its own choice;
    - its speed;
    - its Q-values for waiting, for the best move toward the goal, and for its choice.

    If the reward is the cause, the detours will be the network's own choices, made near other robots, by small Q margins.

    **What the trace found: the guess was mostly wrong.** The replay matched the check exactly (16 and 33 delivered, 3 and 0 collisions). The champion produced 2,796 robot-steps and RULES 1,292.
    - **Most detours happen in the clear.** The table below counts only steps where a move toward the goal was available. The reward near other robots adds about 9 points of stepping away, so it is a minor cause. 82% of the champion's away proposals came with no robot within 2 cells.

      | Champion's proposals, a move toward the goal available | toward | away | wait |
      | --- | --- | --- | --- |
      | no robot within 2 cells (2,201 steps) | 38% | 36% | 25% |
      | another robot within 2 cells (393 steps) | 35% | 45% | 18% |

      RULES proposed the move toward the goal 99.5% of the time.
    - **Far from the goal, its choice is close to random among the moves open to it.** In the clear, more than 10 hops from the goal, it proposed toward 32%, away 37% and wait 31%. Within 1–2 hops, toward rose to 55%. In the seeded shock benchmark, its gradient agreement was 0.63 on the small map and 0.84 on the large one, so the habit is not specific to the stream.
    - **Its Q-values are noisier than the differences between moves.** By its own numbers it is not torn: its chosen move beat the best move toward the goal by a median of 76% of the spread across its moves. But for the same robot, Q(wait) changed by a median 0.41 from one step to the next, more than the 0.29 median spread between moves. A third of its moves right after a move (34%) reversed it.
    - **Why the signal is weak (reasoning, not measured).** Progress is paid as a change in distance, so a step away and a step back cancel out. What separates toward from away is the delivery bonus arriving a little later. That is worth roughly 0.1 in Q units near the goal, and less further away, which is below the network's noise. The gap is larger near the goal, and so is the share of moves toward it.
    - **A bug, smaller but real: a trap at the end of a line.** The action mask and the goal gradient test the position a full step (0.5 cell) ahead. A robot braked to a fractional speed sits between nodes, for example 0.2 short of the last node on a line. From there a full step would pass the end of the line, so the move is marked impossible, although `apply_discrete_action` (NODE_ALIGNED) would stop on the node. The robot can only go back or wait, and the next move toward the goal brings it back to the same spot. This happened on 112 champion robot-steps (4%), to 9 robots. Fleet 22 spent about 40 of its 87 steps in this loop (steps 12–17, 22–26 and 47–81). The livelock escape reads the same gradient, so it cannot help: `trace_fleet` labels those steps "livelock escape", but the escape had no move to give. RULES rarely brakes (2.8% of its steps), so it rarely meets the trap. The mask's docstring promises that "valid for masking and valid when applied can never disagree"; turning NODE_ALIGNED on broke that promise for robots between nodes.

    **What it means.** The goal gradient was added so the network would not have to solve routing ("not what FLOWRRA is about") and could concentrate on contention. In practice the network still decides every move, and away from conflicts its ranking barely follows the route. Options, cheapest first:
    1. Fix the line-end check, so the mask and the gradient test the position the move would actually reach.
    2. Route by default: execute the shortest-path move unless another robot is within the warning distance, and let the network decide only there. A frozen checkpoint can be tested this way as an extra arm in `deploy_eval`, with no training.
    3. In training, a small cost for every step until delivery, so moving toward the goal is worth more than the noise everywhere.

    **Built (11 October): options 1 and 2.**
    - **The line-end fix.** `reach_position()` in `node_warehouse.py` is now the one computation the move, the action mask and the goal gradient share. `test_line_end.py` checks it in four ways:
      - Fleet 22's spot, rebuilt: from z = 11.65 the old mask refused +Z; the fixed one allows it, the gradient points +Z, and the move lands exactly on z = 12, from where +X reaches the goal.
      - On the half-cell lattice: every mask and gradient on a random sparse grid is identical to the old code.
      - With NODE_ALIGNED off: identical everywhere.
      - Off the lattice: 109 moves newly allowed and none newly refused, and every newly allowed move really moves the robot.

      The obstacle-veto and cleanup tests still pass.
    - **Route by default** (`deploy_eval.install_route_default`, `--route-arms`). It sits between the network's proposal and the env, so `core_warehouse.py` does not change. `trace_fleet.py` labels its decisions "route default". The share of moves it routed, and how often the network would have chosen differently, are written per run (`route_share`, `route_network_differed`).

    **The trace after the fix** (same instance, 120 steps, all four test files passing on Rohit's machine):

    | | delivered | collisions | first orders delivered by step 120 |
    | --- | --- | --- | --- |
    | champion | 16 | 1 (3 before the fix) | 16 of 25 |
    | champion+route | 31 | 0 | 24 of 25 |
    | RULES | 33 | 0 (unchanged) | 25 of 25 |

    - **The trap is gone.** No champion steps were stranded at a line end, against 112 before. Fleet 22 delivered at step 21 instead of 87. The champion still delivered 16, because cause 1 (near-random choices in the clear) is untouched.
    - **Route by default matches RULES wherever the robots are apart.** 24 of the 25 first orders were delivered at exactly the same step as under RULES.
    - **Where robots meet, the network backs off.** It decided 154 steps for 13 robots, each time with another robot within 2 cells: 79 moves away, 55 toward, 20 waits. RULES, in the same situation, moved toward 51 times of 54.
    - **Ping-pong at the 2-cell boundary.** The route moves a robot in and the network moves it straight back out: 25 times, for 8 robots. Fleet 5 shows the cost. RULES drove past the same robot and delivered at step 24. Under route by default, fleet 5 went back and forth between (4,9,6) and (5,9,6) from step 20 until the failure wave stopped it at step 52.
    - **Reading.** The first guess (the reward near other robots) was wrong for the clear floor but right where robots meet. The presence charge teaches the network never to stay within 2 cells of anyone. RULES passes within 2 cells all the time without a collision. A training change to try later: charge approach and collisions, not presence. The arm itself is not changed before the runs; the ping-pong is part of what they measure.

26. **deploy_eval, clean condition: scored.** No failures and no people. 2 maps × 25/40/60 fleets × 3 seeds gives 18 instances; with four arms that is 72 runs, 1.3 hours. The line-end fix is in place.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions per episode | L3 per episode | books finished in 800 steps |
    | --- | --- | --- | --- | --- | --- |
    | RULES | 99.2% | 71 / 151 | 0.44 | 2.2 | 13 of 18 |
    | champion+route | 98.7% | 75 / 162 | 4.3 | 2.3 | 13 of 18 |
    | champion | 96.7% | 120 / 260 | 10.9 | 4.4 | 2 of 18 |
    | warm | 96.6% | 123 / 261 | 14.2 | 4.2 | 3 of 18 |

    **Predictions, the clean parts:**

    | Prediction | Result | |
    | --- | --- | --- |
    | The test is repeatable | IDENTICAL, champion and champion+route | ✅ |
    | A clean floor delivers the book (every arm ≥ 97%) | champion 96.7%, warm 96.6%; champion+route 98.7%, RULES 99.2% | ❌ narrowly, for both networks |
    | RULES is the safest driver (clean half) | RULES 0.44 against the champion's 10.9 (p 0.001) and warm's 14.2 | ✅ so far; the full condition decides |
    | Learning on the job traded safety for speed; route by default; warm against the champion | full condition only | pending |

    **Paired comparisons** (Wilcoxon, 18 instances):
    - **champion+route against the champion:** it wins on every measure. Delivered 98.7% against 96.7% (p 0.017). Lead time 75 against 120 steps (p < 0.001). Collisions 4.3 against 10.9 (p 0.005). L3 2.3 against 4.4 (p 0.12).
    - **warm against the champion:** no measure differs (p 0.42 to 0.94). Warm's collisions were 14.2 against 10.9, not significant. Learning on the job changed nothing measurable on a clean floor.
    - **champion+route against RULES (the real question, reported):** delivered and L3 tie (p 0.57 and 0.90). RULES has shorter lead times, 71 against 75 steps (p 0.018), and fewer collisions, 0.44 against 4.3 (p 0.021).
    - **By map.** On the large map, champion+route matches RULES: delivered 100% against 99%, lead time 104 against 104 steps, collisions 0.67 against 0.33, L3 0.8 against 1.3. On the small map, where robots meet far more often, its lead time is 47 against 38 steps and its collisions 7.9 against 0.56 per episode; 71 of its 77 collisions were there.

    **Reading** (clean floor only; the full condition decides the pre-registered questions):
    - Taking routing away from the network closes most of the gap to RULES, and on the large map nearly all of it.
    - What is left is where robots meet. There the network's decisions cost collisions RULES does not have: on the small map, about 14 times as many. Its habit of backing away (entry 25) does not make it safer; RULES passes within 2 cells and collides less.
    - Warm performs the same as the champion.
    - Orders per hour are capped by the book (80 orders in 800 steps is 360 an hour), so on a clean floor every arm is near the cap (348 to 357). Lead time separates the arms better.

27. **deploy_eval, full condition: scored.** Failures and people on, as in run 5. 2 maps × 25/40/60 fleets × 5 seeds gives 30 instances; with four arms that is 120 runs, 3.0 hours. About 9 to 10 robots failed per shift in every arm. In all 120 runs people cleared every piece of debris and nobody timed out.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions per episode | L3 per episode | near misses with people (median) |
    | --- | --- | --- | --- | --- | --- |
    | RULES | 95.3% | 80 / 181 | 0.87 | 3.9 | 3 |
    | champion+route | 95.2% | 86 / 196 | 4.7 | 2.2 | 5 |
    | champion | 90.9% | 131 / 298 | 11.0 | 3.5 | 8 |
    | warm | 90.0% | 131 / 295 | 14.7 | 3.8 | 7 |

    **Predictions, scored:**

    | Prediction | Result | |
    | --- | --- | --- |
    | RULES is the safest driver (both conditions) | full: RULES 0.87 against the champion's 11.0 and warm's 14.7, both p < 0.001. Clean: 0.44 against 10.9 and 14.2 | ✅ |
    | Learning on the job traded safety for speed | warm's lead time 131.2 against 131.0 steps (p 0.69), so not shorter; its collisions 14.7 against 11.0 (p 0.10) | ❌ |
    | Route by default delivers more | 95.2% against 90.9% (p < 0.001; better in 25 of 30 instances) | ✅ |
    | ...at no cost in safety | collisions 4.7 against 11.0 (p 0.009) | ✅ |
    | Warm against the champion (3 of 5 measures to adopt) | warm wins 0 of 5, none significantly | run 2 stays the champion |

    With the clean run's results (entry 26), that makes 4 of 6 pass: the test repeats, RULES is the safest, and both route predictions. A clean floor delivering the book and learning on the job trading safety for speed both fail.

    **Agreed in advance: the condition is met.** RULES matches or beats both networks on share delivered (95.3% against 90.9% and 90.0%) and on collisions (0.87 against 11.0 and 14.7). The architecture (rules, ladder, rescue) is what keeps the fleet whole. The learned policy has not yet earned its place in deployment.

    **Reported: champion+route against RULES.** Route by default wins one of the five measures by mean, and not significantly. The network adds nothing measurable even where robots meet, and costs collisions there.
    - **Delivered:** tie, 95.2% against 95.3% (p 0.98).
    - **Lead time:** RULES shorter, 80 against 86 steps (p 0.010).
    - **Collisions:** RULES fewer, 0.87 against 4.7 (p < 0.001); route had fewer in only 3 of 30 instances.
    - **L3:** route fewer, 2.2 against 3.9 (p 0.25, not significant).
    - **Near misses:** tie, medians 5 and 3 (p 0.94). The means are pulled up by a few long standoffs next to people, which a trace should explain: one route instance had 498 near-miss steps, another 369.

    **Clean against full, on the same 18 days (seeds 0 to 2).**
    - **Delivered:** failures and people cost RULES 3.6 points of the book (99.2% to 95.6%) and route by default 3.4 (98.7% to 95.3%). The networks lost about 6 (96.7% to 90.7%, and 96.6% to 90.8%).
    - **Lead time:** rose by 8 to 12 steps for every arm.
    - **Collisions:** barely changed. They come from robots meeting each other, not from failures or people.
    - **Books finished within 800 steps:** 13 of 18 for RULES and for route by default on the clean floor. In full conditions, 1 of 30 for RULES and none for the others. About 12 orders per shift were still out at the end, 16 for the networks.

    **Reading.**
    - The strongest result is not about the network. The architecture keeps the fleet whole through about ten robot failures a shift and people on the floor: rules, ladder, rescue and the hard veto on people. Under RULES driving, that is less than one collision a shift, and every human job finished.
    - Route by default closes the gap to RULES on delivery and nearly on lead time. What the network still decides, where robots meet, costs collisions.
    - Warm performs the same as the champion. Fifty episodes of learning on the stream did not change frozen behaviour.
    - **The cost of a real floor (Rohit's note).** About ten robot failures a shift, about nine handovers, and people clearing debris cost RULES and route by default about 3.5 points of the book against a clean floor. There was no stop, no escalation and no human in the loop. What a conventional planner loses on the same days with the same failures is not measured yet; the lifelong RHCR comparison on these maps would give that baseline.

28. **15 fleets (reported only, as registered).** Clean: 2 maps × 3 seeds, 6 instances. Full: 2 maps × 5 seeds, 10 instances. Four arms each, no errors in either run.

    | Clean | book delivered | lead time (steps) | collisions | books finished |
    | --- | --- | --- | --- | --- |
    | RULES | 98.7% | 93 | 1.2 | 3 of 6 |
    | champion+route | 98.3% | 97 | 4.0 | 3 of 6 |
    | champion | 98.5% | 131 | 2.5 | 2 of 6 |
    | warm | 96.9% | 161 | 2.2 | 0 of 6 |

    | Full | book delivered | lead time (steps) | collisions | robots failed per shift (of 15) |
    | --- | --- | --- | --- | --- |
    | RULES | 69.6% | 99 | 1.2 | 10.7 |
    | champion+route | 72.6% | 106 | 2.9 | 9.5 |
    | champion | 57.8% | 133 | 3.2 | 9.5 |
    | warm | 51.1% | 128 | 1.9 | 9.4 |

    - **On a clean floor, 15 robots carry the whole book.** Every arm delivered 97 to 99%. The expectation in entry 24's amendment, that 15 would be beyond capacity, was wrong. Orders take longer: 93 steps under RULES, against 71 with 25 to 60 robots. With few meetings, the network's collision cost almost disappears (the champion's 2.5).
    - **In full conditions the failure model decides the result.** Failures are counted per shift, not per robot: up to 5 single failures plus up to 3 waves of 2 to 4 robots, whatever the fleet size. The config itself calls this a "training rate, deliberately unrealistic". That gives about 10 failures a shift at every fleet size (15 fleets 9.8, 25 fleets 7.9, 40 and 60 fleets about 10).
      - With 15 robots, that is two-thirds of the fleet. In 3 of the 10 shifts, 14 or 15 of the 15 failed.
      - The share delivered falls with the share of the fleet lost (Spearman −0.6 to −0.8 in every arm).
      - These shifts measure a fleet losing most of itself in 13 minutes.
    - **Even so, efficient driving held up.** RULES and route by default kept about 70% (72.6% and 69.6%, a tie, p 0.36), against 57.8% for the champion and 51.1% for warm. One small-map shift (seed 2, 9 to 13 robots failed) shows it: RULES 92.5%, route 90%, the champion 30%, warm 32.5%. When the fleet is decimated, how well the survivors drive is what is left.
    - **This also re-reads entry 27.** Ten failures in a 13-minute shift on 25 to 60 robots is a stress test (17 to 40% of the fleet), far above any real rate. **For the redesign:** express failures as a rate per robot-hour, so the stress scales with fleet size and can be set to realistic values as well as stress values. This goes with reissue, technician retrieval and the battery protocol.

29. **Design for approval (12 October): the intent guard, bubbles by kind, hops everywhere.** No code yet. Written after reading the current code (Rohit's `flowrra_code.zip`, identical to the working copies except an older `trace_fleet.py`).

    **What already exists.**
    - **Path awareness** (`conflict.path_warnings`, `conflict_warehouse.py`). Every pair within 3 hops is classified by its next 8 route cells: head-on, blocked, following, contested, sequential or none. It warns by kind: within 1 hop always; head-on, contested and blocked always; following only closer than `follow_gap` (1.5); sequential and none never. Each route is "what a fleet would broadcast to its neighbours". It is local intent-sharing, and nothing plans for anyone.
    - **Directional braking.** A steady convoy at 1.5 hops or more does not brake.
    - **Rays.** Each fleet casts six rays along the aisle axes, up to 25 cells, seeing peers and obstacles. Corridor entry already uses them ("as its ray would").

    **The hole.** Braking reads distance only, and routes are predictions; the config itself keeps the 1-hop floor "because routes are predictions". In the step a robot actually takes, a ladder retreat reverses it, the network steps away (a third of its moves right after a move were reversals, entry 25), and holds and tabu swaps change moves. A pair labelled "following" can therefore be head-on this very step. That is why a 1.25-hop bubble is unsafe under distance-only braking.

    **The proposal.**
    1. **Intent for this step is the actual move.** All moves are already final before anyone moves (`step.simultaneous` is on), so the step is split into three passes: decide every fleet's move, brake, then move.
    2. **Bubbles by kind, from path awareness.** The comfort braking ramp ends at a radius that depends on the pair: about 1.75 hops for meeting kinds (head-on, contested, blocked), about 1.25 for following, and the 1-hop floor for sequential and none. Unknown intent (no route, for example a dead radio) is treated as head-on. The rays still see such a robot.
    3. **The intent guard: the last, exact check.** For every pair within 3 hops, it takes both robots' actual moves at their braked speeds and computes where each would land, using `reach_position`. If the gap after the step would fall below the minimum gap of **1 hop** (Rohit's "one hop"; a collision stays at 0.5), both are slowed just enough to stop at the minimum, down to standing still if needed. It repeats until no pair is short, since slowing one robot can change another pair. It is local: each robot needs only its neighbours' broadcast next moves, the way VDA 5050 robots publish their planned nodes.
    4. **What it means.** A moving robot can no longer come within 1 hop of another moving robot, so collisions from movement go to about zero for every arm, the network included. The new safety measure becomes **guard interventions**: how often the safety layer had to step in, the simulator's version of a hardware safety stop. Collisions can still come from re-entries and recovery moves, and are still counted. This is the Simplex pattern from entry 23: a verified safety layer under a policy that only decides efficiency.

    **Hops everywhere (the audit).**
    - **Already in hops:** the collision and warning check; braking distance; the reward's closeness; the network's "robot nearby" input; path-warning radius, floor and follow gap; the attention graph; the density kernel. Rays are cells along an aisle, which are hops along that line.
    - **Not in hops, to convert:**
      - the reward's approach charge measures the closing gap in straight-line (Euclidean) distance;
      - the convoy tests (`_is_following`, `_following_snap`) measure the gap change in Manhattan distance;
      - recovery (its pair checks and Tier 1 to 3 placements) uses Manhattan distance. That is always on the safe side, since Manhattan never exceeds hops, but it is inconsistent.
      - The config comments still call the collision and warning thresholds "Manhattan".
    - Same-cell checks (pickup dwell, arrival) are fine as they are: within a cell the two agree.

    **The other two switches.**
    - **Reward:** drop the presence charge (−0.5 per step for being in any warned pair, which with path warnings reaches 3 hops for head-on routes). Keep the collision charge (−1) and the approach charge, the latter measured in hops.
    - **Route by default as a core switch** (`conflict.route_by_default`), so training and evaluation use the same rule.

    **Tests before any training.**
    - **The two-robot sweep.** Every half-hop distance from 0.5 to 3, head-on, following, crossing and reversing, every pair of actions, with ladder orders and holds. With the guard on, no step may end with two moving robots under 1 hop apart, and none at 0.5.
    - **Equivalence.** With the guard and bubbles off, the step must be identical to today's (the existing `test_loop_equivalence`, smoke and integration tests, plus a replay of the check instance).
    - **A short shakedown** to see how often the guard steps in.

30. **Built (12 October): the intent guard, bubbles by kind, hops everywhere, the presence switch, route by default in the core.** All approved in entry 29 (Rohit: "yes", "yes", "everything else as it has been", and the config comments corrected). Every new behaviour is a config switch, and **all of them off reproduces the step before them exactly**.
    - **`guard_warehouse.py` (new).** It holds:
      - `landing_position()`: where a move will put a fleet, through `reach_position`, the move's own computation.
      - `HopMetric`: the graph distance between continuous positions, the same anchors and same-edge rule as `GraphProximity`.
      - `crossed()`: two fleets on one line swapping order in a step.
      - `guard_speeds()`: per pair, slow the fleets that are closing the gap (never one that is backing away) by bisection, just enough to stop at the minimum gap. It repeats in rounds, and whatever is still short stops outright. Pairs are taken in a canonical order, so the result never depends on a fleet's place in the list.
    - **`test_intent_guard.py`, all pass here (no torch):**
      - landing equals the real move in 19,670 cases;
      - one aisle: every quarter-hop distance from 0.25 to 3.5, every pair of moves, speeds 0.1 to 1.1 (8,064 cases, the guard stepped in on 1,241);
      - a junction (11,760 cases), chains of three, and 2,000 random crowds of 4 to 8 fleets on a sparse grid;
      - no pair ever ended under 1 hop or drove through the other;
      - no needless braking, and a fleet backing away was never slowed;
      - the same result whatever the list order.

      The test caught one gap on the way: a fleet overtaking another from 0.25 apart, or the Tier-3 winner's 1.1 bump, can land as far apart as it started while driving straight through. `crossed()` closes that.
    - **`core_warehouse.py`:**
      - **The guard.** With `safety.intent_guard`, phase A decides every move first, the guard sets speeds, then everyone moves. The apply and motion accounting are the same as before. A move the guard stops outright becomes a wait, and the wait is recorded as the executed action.
      - **A live invariant check after every guarded step** (`guard_violations`, must stay 0). Every training run re-tests the guard.
      - **Bubbles by kind** (`_kind_brake`): head-on, contested, unknown and the mover of a blocked pair brake from 1.75 hops, a follower from 1.25, the rest only at the 1-hop floor.
      - **The presence switch** (`reward_decomposition.presence_charge`).
      - **Route by default** (`conflict.route_by_default`): the network decides only for fleets in the step's warning or deadlock set, which with path warnings means a warned pair.
      - **Hops for the approach charge and the convoy tests** (`safety.hops_everywhere`).
      - **New counters:** `guard_*` and `route_*`, in the statistics, the training CSV and a new `safety |` line in the training log.
    - **`recovery_warehouse.py`:** every fleet-to-fleet distance goes through `_gap()`, which is hops with `hops_everywhere` and otherwise exactly the old expression.
    - **`config_warehouse.py`:** RUN 6 header; new `safety` section; `route_by_default`; `presence_charge` False; `eps_peak` 0.30 (cold, as run 2). Four comments that said "Manhattan" now say hops: the collision and warning thresholds, the density falloff and the attention radius. Everything else is as run 5.
    - **`deploy_eval.py` and `trace_fleet.py`:** a `--safety config|on|off` option. `off` must reproduce the 11 October check exactly: champion 16 delivered with 1 collision, champion+route 31 and 0, RULES 33 and 0. `trace_fleet` labels "safety guard" and core "route default" decisions. **`main_runner_warehouse.py`:** the new CSV columns and log line.
    - **Run on Rohit's machine (they need torch), 12 October:**
      - `test_guard_integration.py`: real steps, 24 fleets, the order stream. The invariant must hold, the guard must engage, the same seed must give the same episode, and with the switches off nothing new may happen. **ALL PASS** (guard stepped in on 239 of 300 steps, 572 pairs too close, 711 slowed, 645 stopped, no fallback stops, route 28%, collisions 0).
      - The existing suite: **ALL PASS** (after the two stale tests below were fixed).
      - The `--safety off` check above: **reproduced exactly.** Champion 16 delivered with 1 collision, champion+route 31 and 0 (routed 89% of moves), RULES 33 and 0, both repeats IDENTICAL. With its switches off, the build changes nothing.
      - The `--safety on` check (same map, 25 fleets, seed 0, 120 steps, failures and people on): **passed.**
        - **Invariant 0 and repeat IDENTICAL** for champion (31 and 31 delivered, 0 and 0 collisions). Collisions 0 in both arms.
        - **Route share 98% for champion and 97% for RULES**, higher than the outside rule's 89%. I had predicted lower, and was wrong. The outside rule handed a robot to the network whenever anyone was within 2 cells. Path awareness hands it over only when the two routes actually meet, follow or block within the next 8 cells. Robots near each other but on separate routes count as clear. For RULES the route never disagreed with the rule's own move (`route_network_differed` 0).
        - **The guard never had to act on this map:** 14 candidate pairs checked over 120 steps for champion and 17 for RULES, none too close. Braking by kind kept everyone apart first. The brake duty cycle fell by more than half for champion (0.046 to 0.019) and by a third for RULES (0.039 to 0.026).
        - **Deliveries:** champion 31, the same as champion+route with safety off; lead time 23.2 steps against 21.5. RULES 30 against 33. With one instance of 120 steps this cannot be told from timing. With less braking, RULES' fleet 21 left the floor at step 16 instead of 18, re-entered first and took order 1510. From there the book went to different robots, and the step-52 failure wave (the same three robots) caught different orders. The 5-seed, 800-step frozen evaluation with the guard on for every arm is what measures it.
        - **What it means for the run.** On this map, at 25 fleets, the network chooses about 2% of moves; everything else is the route. Whether that is enough to learn from depends on how crowded the training floors are: read the route share at 40 and 60 fleets in the shakedown.
      - **The 3-episode shakedown** (`ladder_run2_shake`, cold, 9.8 minutes, no errors, losses small and finite). With 3 episodes the exploration schedule was squeezed: epsilon 0.19, 0.05, 0.01.

        | | Ep 1 | Ep 2 | Ep 3 |
        | --- | --- | --- | --- |
        | Map, robots | small, 40 | large, 40 | small, 60 |
        | Book delivered | 78 of 80 | 80 (all done at step 648) | 80 (all done at step 466) |
        | Collisions | 0 | 0 | 0 |
        | Guard: steps stepped in, robots stopped | 51, 46 | 0, 0 | 26, 30 |
        | Fallback stops, invariant violations | 0, 0 | 0, 0 | 0, 0 |
        | Route decided | 90% | 98% | 81% |
        | Lead time, mean (p90), steps | 43 (97) | 119 (257) | 31 (64) |
        | L3 | 1 | 0 | 3 |
        | Network proposes moving away near another robot | 63% | 69% | 46% |

        For scale, run 5 (warm, learning on the job) averaged about 13 collisions and 3.4 L3 fires per episode, with a lead time of about 140 steps, 91% of the book, and no book finished within 800 steps. This is 3 episodes on different scenario files, but the gap is far larger than the choice of scenario.

        **What it means.** A network with almost no training already delivers 99% of the book with no collisions, so the route, the bubbles and the guard do most of the work. Two of the proposed predictions (book ≥ 92%, L3 ≤ 3.4) would pass whether or not the network learns anything. What the network still controls is where robots meet: 2 to 19% of moves, a few hundred to a thousand decisions an episode. There, the cold network proposes moving away from its goal half the time or more. The guard and the rules catch that, at the cost of stops and time. So the question for the frozen evaluation becomes: does ladder_run2 beat this barely trained network? The shakedown saved it as `ladder_run2_shake/flowrra_curriculum.pth`. Revised predictions below, agreed before the run.
    - **Two older tests were stale, and are fixed.** `test_conflict_rules.py` and `test_corridor_rules.py` treated "restore the config file" as "every component off", which stopped being true when run 2's config switched the components on. They failed on that, not on the build: `no_classifier`, `no_verdicts`, `no_brake_exemptions` and `off_throttles_the_convoy` on Rohit's machine. Both now spell out "off" (`CONFLICT_OFF`, `SAFETY_OFF`) and isolate the component each test is about. A second stale assumption: they were written when failures were off, and the config now has robot failures, failure waves and people switched on. In the corridor test that showed as `on_both_deliver` (one fleet failed, so only M delivered) and `forwards_equals_reversed` (4 differences). Both tests now also switch failures and people off (`NO_FAILURES`), as when they were written. `test_guard_integration.py`'s requirement that the route decide over half the moves was a guess about its crowded floor (route 28% there); it now requires only that both the route and the network decide some.
    - **Note:** `benchmark_flowrra.py` and `benchmark_all.py` read the config, so with this config the shock benchmark runs with the guard on. Reproducing earlier benchmark numbers needs the safety switches off.

    **Predictions for run 6, ladder_run2: agreed 12 October, before the run** (revised after the shakedown showed that a barely trained network already meets the old throughput and L3 bars). 150 episodes, cold, started 12 October.

    *During the run.* A cold start opens at the exploration peak: the random-move rate is 0.30 in episode 1, about 0.19 by episode 50, 0.10 by 75 and 0.05 by 100, and 0.01 to 0.02 in the last 25 (a half-Gaussian of width 50 episodes; an earlier draft of this entry said it climbs to 0.30 around episode 37, which was wrong). So first-against-last comparisons mix learning with exploration. These are process checks; the frozen test below is the decision.

    | Prediction | Pass if |
    | --- | --- |
    | Safety by construction | `guard_violations` 0 in every episode, including the first ones at the exploration peak; collisions per episode below 0.5 on average (only re-entries and recovery moves can still cause one) |
    | The reward stops teaching retreat | among the network's own proposals near another robot (`choice_policy_*`), moves away from the goal fall below 20% in the last 25 episodes (shakedown: 46 to 69%; run 2: 32%; run 5: 35%) |
    | Nothing gets worse (no-harm, not a claim of learning) | in the last 25: book delivered at least 95% and L3 per episode no higher than run 5's 3.4 |

    Reported, not scored: how often the network's own choice in the clear would differ from the route (shakedown 55 to 61%). It should fall as the network learns from the routed moves, but it cannot reach 0, because the route takes one of several equally short moves.

    **Then frozen, on the same seeded days** (`deploy_eval`, full condition, 2 maps × 25/40/60 fleets × 5 seeds, `--safety on` for every arm, about 3 hours). Arms:
    - ladder_run2;
    - **shake3**, the 3-episode network (`ladder_run2_shake/flowrra_curriculum.pth`, kept for this);
    - the champion (ladder_run1);
    - RULES.

    Decision rule, 3 of 5 measures (delivered, lead time, guard stops, L3, near misses):
    - **ladder_run2 against shake3:** did training add anything over the safety layer?
    - **ladder_run2 against the champion:** did the new reward and conditions beat run 2's?
    - **RULES** is reported alongside both.

    This separates what the safety layer gives (shake3) from what learning adds on top.

31. **Run 6, ladder_run2: scored (12 October).** 150 episodes, cold, 12.2 hours, no errors. Both maps, 25/40/60 fleets, failures and people on, the order book (80 orders over 400 steps).

    | Prediction | Pass if | Result |
    | --- | --- | --- |
    | Safety by construction | `guard_violations` 0 in every episode; collisions below 0.5 per episode | ✅ 0 violations in all 150 episodes and no fallback stops; 2 collisions in 150 episodes (0.013 per episode; run 5: about 13 per episode) |
    | The reward stops teaching retreat | the network's "move away" proposals near another robot below 20% in the last 25 | ❌ 48% (first 25: 41%) |
    | Nothing gets worse | book ≥ 95% and L3 ≤ 3.4 in the last 25 | ❌ / ✅ book 93.4%; L3 2.8 |

    **1 pass, 1 fail, 1 split.** The book figure is partly the mix: the last 25 drew 15 episodes with 25 robots, the hardest setting under this failure model. Weighting every map and fleet count equally, deliveries were flat across the run (95.1% in episodes 1–50, 94.8% in 101–150). L3 rose for real, from about 1.0 to 2.5 per episode on the same equal weighting.

    **The safety layer worked; the learning did not.** By 25-episode block:

    | Block | 1 | 2 | 3 | 4 | 5 | 6 |
    | --- | --- | --- | --- | --- | --- | --- |
    | Exploration (start of block) | 0.30 | 0.27 | 0.19 | 0.10 | 0.05 | 0.02 |
    | L3 per episode | 1.08 | 0.84 | 1.44 | 1.48 | 2.20 | 2.80 |
    | Guard stops per episode | 75 | 73 | 86 | 70 | 117 | 95 |
    | Network differs from the route in the clear | 54% | 63% | 60% | 62% | 64% | 60% |
    | "Move away" proposals near another robot | 41% | 41% | 45% | 50% | 41% | 48% |
    | Delivery head, mean value | 0.29 | 0.96 | 1.40 | 1.68 | 1.82 | 1.99 |
    | Loss, delivery / efficiency | 0.012 / 0.003 | 0.019 / 0.007 | 0.023 / 0.014 | 0.024 / 0.016 | 0.022 / 0.015 | 0.027 / 0.016 |

    **Diagnosis (survived its first check, not yet proven): the values inflated through moves that experience never corrects.**
    - **The evidence.** The delivery head pays one delivery, 100 / 70.4 = 1.42 in its units, so its mean value should stay below 1.42 (cold_run22's ceiling). It climbed all run, through 1.42 around episode 70, to 2.13 at episode 150. The reward earned per step stayed flat throughout (delivery about 0.008 per fleet-step in every block). The losses rose with it; Rohit saw this as "gradients more intense than previous runs".
    - **The proposed mechanism.** It is the one the code documents for impossible moves (cold_run5, warm_run7). Route by default made a new set of never-corrected moves: in the clear, about 85% of moves, the route always takes the one shortest move. The other valid moves are never executed there, so no experience pulls their values back. The learning target takes the network's best move in the next state, which in the clear can be one of those drifted moves. Inflated targets raise the values, which raise the targets. Double DQN does not stop it, because the online and target networks share the drift.
    - **What fits.** The network's own choice in the clear disagreed with the route more after the first block, although nearly everything it learned from was route moves. Its choices where robots meet stayed close to random. As exploration fell, stand-offs that random moves used to break lasted longer (L3 up).
    - **What it means for the design.** Principle 7 (values stay honest) was not carried through to route by default. The target should value the policy actually run: the route's move where the route will drive, and the network's best move only where the network decides. That was missed in entry 29.
    - **Falsification check: the diagnosis survived it (12 October).** If ladder_run1 (run 2, no route by default) had also pushed the delivery value above 1.42, the diagnosis would be wrong. It did not:

      | Run | Route by default | Delivery value, first block → last block (max) | Delivery loss, first → last | Efficiency value, last block |
      | --- | --- | --- | --- | --- |
      | Run 2, ladder_run1 (cold, 150 episodes) | off | 0.22 → 0.91 (0.93), levelling off | 0.006 → 0.002 | 0.87 |
      | Run 5, warm_stream1 (warm, 50 episodes, order stream) | off | 0.89 → 0.90 (1.03) | 0.006 → 0.007 | 0.95 |
      | Run 6, ladder_run2 (cold, 150 episodes, order stream) | **on** | 0.29 → 1.99 (2.13), still rising | 0.012 → 0.027 | 3.35 |

      Run 2 is the like-for-like comparison: cold, 150 episodes, the same maps. Its values levelled off under the ceiling and its loss fell. Run 5 had the order stream too, so the stream (exits, each robot's own terminal) is not the cause. Of run 6's new switches, route by default is the one that leaves moves untried; the guard stores what was executed, and the presence switch changes only the safety reward. Proof would be a run with the target fixed whose values stay under the ceiling.

    **The harsh episodes.** The worst 8 by deliveries are almost all the large map with 25 robots. 11 to 15 failures in one shift, 40 to 60% of the fleet: episode 39 delivered 53 of 80 with 15 failures. The share of the fleet that failed tracks deliveries (r = −0.62). This is the stress model as designed (about 10 failures a shift whatever the fleet size, 2.5 per robot-hour on the floor), far above a real site; see Next item 5.

    **Predictions for the frozen tests, written before they run.** Every arm has the guard on.
    - Collisions near zero in every arm, RULES included.
    - ladder_run2 does **not** beat shake3 on 3 of 5 measures: training added little over the safety layer.
    - Against the champion, no prediction.

32. **Frozen test, clean condition, guard on for every arm: scored (12 October).** No failures and no people. The same 18 seeded days as entry 26 (2 maps × 25/40/60 fleets × 3 seeds), four arms, 72 runs.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions, all 18 days | L3 per day | guard stops per day | books finished | network differs from the route in the clear |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | RULES | 99.3% | 72 / 154 | 0 | 3.5 | 23 | 14 of 18 | (rules: 0%) |
    | champion | 99.4% | 75 / 161 | 0 | 3.3 | 143 | 16 of 18 | 14% |
    | ladder2 | 99.9% | 78 / 169 | 0 | 2.6 | 130 | 16 of 18 | 62% |
    | shake3 | 99.6% | 76 / 162 | 0 | 2.4 | 141 | 16 of 18 | 58% |

    0 invariant violations and 0 fallback stops in all 72 runs.

    **Prediction: ladder2 does not beat shake3: ✅ confirmed.** No measure differs (p 0.42 to 0.92). 150 episodes of training added nothing measurable over a network trained for 3.

    **The safety layer, against entry 26 on the same days** (guard off then, on now):
    - **Collisions are gone.** RULES went from 0.44 a day to 0. The champion with route by default went from 4.3 to 0, and without it from 10.9.
    - **Throughput did not move.** RULES delivered 99.2% then and 99.3% now, at a lead time of 71 then and 72 now. The champion with the route delivered 98.7% at 75 steps, and 99.4% at 75 now.
    - **The cost is stand-offs.** L3 rose: RULES 2.2 to 3.5, the champion with the route 2.3 to 3.3.

    **Supporting evidence for entry 31's diagnosis.** In the clear, the champion's own choice matches the route 86% of the time: trained without the route, it had to learn to head for its goal. ladder2's matches 38%, no better than shake3's 42%. Its values never learned the way to the goal, even though nearly all it trained on was route moves.

    **RULES is still the best driver where robots meet.**
    - **Lead time:** 72 steps against ladder2's 78 (p 0.003). On the small map it is 39 against 46 to 47 for every network. On the large map the champion ties it, 104.5 against 105.
    - **Guard stops:** 23 a day against the networks' 130 to 143 (against ladder2, p 0.016). Its moves rarely need the guard: on the small map 46 a day against 197 to 267.

    **Where the networks beat RULES.** Credit where it is due (Rohit's correction, 12 October):
    - **Fewer stand-offs.** L3 is 2.4 to 3.3 a day for the networks against 3.5 for RULES. RULES makes the same polite choice on both sides of a head-on meeting and waits for the ladder; the networks' choices break some stand-offs themselves.
    - **More finished books.** Every network finished the whole book on 16 of 18 days; RULES on 14.
    - **Slightly more delivered.** ladder2 delivered 99.9% against RULES' 99.3% (p 0.078, not yet firm).
    - This is the network acting at meetings, from its .pth file. It is shared by all three networks, so it is the contribution of the learned approach; what run 6's training failed to add on top of it is a fault of the training target (entry 31), not of the network.
    - Its instrumentation found that fault: the per-head values, the agreement with the route, and the choices near other robots.

    **For a run 7.** The champion is the better starting point: honest values, it heads for its goal, it finishes as many books, and its lead time is within 3 steps of RULES. What remains to learn is the meeting itself, and the target is RULES there: lead time on the small map, and moves that do not need the guard.

33. **Frozen test, full condition, guard on for every arm: scored (12 October).** Failures and people on. The same 30 seeded days as entry 27 (2 maps × 25/40/60 fleets × 5 seeds), four arms, 120 runs, about 9.5 failures per shift in every arm.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions, all 30 days | L3 per day | guard stops per day | near misses with people (median) | books finished |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | RULES | 96.2% | 81 / 186 | 0 | 3.6 | 28 | 4.5 | 4 of 30 |
    | champion | 95.9% | 82 / 191 | 0 | 2.2 | 123 | 4.5 | 2 of 30 |
    | ladder2 | 95.8% | 85 / 194 | 1 | 3.7 | 127 | 4.0 | 3 of 30 |
    | shake3 | 95.8% | 86 / 197 | 0 | 2.4 | 160 | 5.5 | 4 of 30 |

    **The safety layer under full stress, against entry 27 on the same days:**
    - **Collisions:** 1 in 120 shifts, against 0.87 per shift for RULES, 4.7 for the champion with the route and 11.0 without it.
    - **Deliveries went up for every arm:** RULES from 95.3% to 96.2%, the champion with the route from 95.2% to 95.9%.
    - **Lead time held or improved:** RULES 80 to 81, the champion with the route 86 to 82.

    **ladder2 against shake3:** no measure differs (p 0.16 to 0.86). The prediction is confirmed in both conditions.

    **The champion ties RULES under full stress:**

    | Measure | Champion | RULES | p |
    | --- | --- | --- | --- |
    | Delivered | 95.9% | 96.2% | 0.51 |
    | Lead time (steps) | 82 | 81 | 0.42 |
    | Near misses with people | 36.6 | 33.3 | 0.74 |
    | L3 per day | 2.2 | 3.6 | 0.051 |
    | Guard stops per day | 123 | 28 | < 0.001 |

    The champion has 40% fewer stand-offs, which is just short of firm, and on the large map it is slightly the faster (115 against 117 steps). RULES still needs the guard about four times less.

    **The one collision, and both of run 6's: a gap in the safety layer.** All three have the same cause:
    - A robot left with no order mid-floor is parked in place ("frozen, pool exhausted while retargeting").
    - Parked robots are outside both the guard and the collision check: the guard pairs only robots that are moving this step, and the loop's collision sweep skips parked robots.
    - Another robot drove onto the parked robot's cell. The collision registered only when the parked robot was recalled for a rescue. This is the sequence in all three cases: run 6, episodes with fleets 192#2/70#2 and 15#2/86#2; deploy, fleets 27#2/37#2.

    So the invariant held for every pair the guard checked, but parked robots were never among them. **The fix:** treat a parked robot as a stationary obstacle in the guard, in the move mask and in the collision check. In the stream, a robot with no work should arguably head for a dock rather than park in an aisle.

    **Traced in the code (Rohit's question: a mismatch in information flow between the modules?). Yes, one path is out of date.** Each part has its own idea of what "frozen" means:
    - **"Frozen" is a pre-stream state.** It was designed to mean "arrived at its goal and parked for good". On that meaning, the collision check, the proximity index (and so the guard and path awareness) and the GNN (`freeze_node`, "a static landmark") all treat a frozen robot as furniture: a density stamp, no hard block.
    - **The stream already removed it on the delivery path.** When a robot delivers, it now drives out at the nearest dock instead of parking. The comment there names the problem: a parked fleet "leaves only a static density stamp, forever, on a cell somebody wanted"; "this will never move again" is false of a parked robot that might be recalled; "despawning deletes the ambiguous case".
    - **The retarget path was never updated.** When a robot's order is taken by another robot while it is en route and nothing is left in the pool, it still freezes where it stands, mid-aisle: 131 times in the full test.
    - **Rescue dispatch can recall a frozen robot** ("RECALLED from retirement"). The furniture comes back to life, and whatever drove onto its cell in the meantime is in a collision.
    - **The fix, for the run 7 design:** in the stream, the retarget path does what the delivery path does and sends the robot out at the nearest dock. No new state, no change to the guard; the ambiguous case goes away.

34. **Design for run 7, ladder_run3 (13 October): two fixes, then a warm run. Approved in part; open points marked.**

    **Fix 1: honest values (approved).** Value the policy that actually runs.
    - **The rule.** When a lesson asks "how good is the next state?", it uses, for each robot in that next state:
      - the network's best move, if the network actually drove that robot (the route did not take it, and no rule, ladder hold or guard stop changed its move);
      - otherwise the move that was actually made.
    - **Plumbing.** The move made in the next state is known only when the next step has run. So each step's transition is held one step, completed with the next step's real moves, matched by robot ID (robots leave and enter between steps), and then stored. When a step ends the episode, it is stored at once. After the last step the runner flushes the held transition.
    - **Storage and learning.** The replay buffer gains one field per robot, "the move made next, or −1 if the network chose". The learning step uses that move wherever it is set, after the masked double-DQN choice.
    - **Switch.** `training.honest_target`; off is exactly today's code.

    **Fix 2: no frozen robots in the stream (approved).**
    - **Where.** Three places still freeze a robot mid-aisle: "pool exhausted while retargeting", and two silent ones where a robot's goal is taken as it arrives (the in-loop retarget and the deferred retarget).
    - **The change.** In the stream they send the robot out at the nearest dock, as the delivery path already does, without counting a delivery. A robot with no reachable dock still parks, and is counted (`stream_parked_no_exit`); no-work exits are counted too.
    - **Switch.** `stream.no_work_exit`; off is exactly today's code.

    **Tests before any training.**
    - `test_honest_target.py`:
      - switch off: targets bit-identical to today's;
      - a route-driven robot is valued at the route's move, a network-driven robot at the network's best, a guard-stopped robot at waiting;
      - robots are matched across a roster change;
      - every transition is stored exactly once.
    - `test_stream_no_work.py`: the pool runs out mid-stream; the robot drives to a dock and leaves, is never frozen, and counts no delivery.
    - The existing suite, and `deploy_eval --check` with both switches off, which must reproduce the 12 October check exactly (champion 31 delivered, 0 collisions; RULES 30, 0).

    **Decisions (Rohit, 13 October).**
    - **A. Start from the champion (ladder_run1).** Yes.
    - **B. A short dry run first,** to see whether the losses or values blow up. Warm (no `--cold-start`), so the exploration follows the warm Gaussian and the losses can be read against it. Then the run.
    - **C. Exploration peak 0.05, as run 5.** Yes. Rohit asked whether to raise the floor of 0.01: every time exploration neared it, L3 rose. Answer:
      - `recovery_policy.exploration_scale` is not the cause. It scales only the recovery head's exploration, and that head is frozen under the ladder.
      - Fleet exploration is per robot, everywhere, with no exception near conflicts.
      - The observation is right, though. Near 0.01 the network's choices at a meeting become fully deterministic, so two robots can mirror each other. RULES, fully deterministic, has the most L3 of all (entries 32 and 33), and run 6's L3 rose as exploration fell.
      - Randomness breaking stand-offs is a crutch: every arm is evaluated at 0.015. The dry run keeps run 5's floor, and its L3 against exploration shows whether the network has learned to break ties itself. If not, the floor or the Learner (below) is the next lever.
    - **D. Failures at run 5's level.** Yes, so run 5 is the like-for-like comparison.
    - **E. Book 80 for training.** Yes. Rohit's addition, a top-up, is an open point below.

    **Open point 1: learn RULES' and the guard's judgement directly? (Rohit.)** Rohit's idea: optimise the network towards what RULES and the guard recommend at meetings, since they are now known to be better there.
    - **The project already has this, built to respect principle 7: the Learner head (step 3a).** A small head reads the shared trunk but never trains it, and learns by plain supervision the move RULES chose; no Q-value moves. At choice time it nudges only robots inside a conflict: score = Q + β × its log-probability.
    - **Its first form broke principle 7.** The margin teacher pushed Q-values towards RULES, and the values drifted (entry "What happens to the teacher").
    - **The Learner head lost before, and the reason is now gone.** It lost to the champion in run 4 (5–0 on the small map, entry 21). The note at the time: the Learner speaks only inside conflicts, while the small-map gap looked like routing in free traffic. Route by default now handles free traffic, and the remaining gap to RULES is exactly at meetings (lead time, guard stops), which is where the Learner speaks.
    - **The guard as a second teacher** is new: a guard stop marks a proposal as unsafe. To consider after the first test.
    - **It complements Fix 1, it does not replace it.** The Q heads still bootstrap, so without Fix 1 their values inflate whether or not the Learner is on.
    - **Proposal:** dry run B, the same as dry run A with the Learner on (β 0.1), paired with A, before choosing run 7's settings.
    - **Decided (Rohit, 13 October): held until the end.** The Learner would also need to learn from the guard and the route to be a fair test, so it comes after run 7. Dry run B is dropped from this plan.

    **Open point 2: a top-up book (Rohit).** When the fleet finishes its 80 orders early, give it more work to use the rest of the shift. Proposed rule, to confirm: whenever everything released so far has been delivered before step 600, release 10 more orders. Orders delivered in 800 steps then measures capacity, uncapped (today 80 orders cap every arm at 360 an hour). Proposed use: the capacity test in `deploy_eval`, in place of the fixed 120- and 160-order days; training keeps the fixed 80. A switch, off by default.
    - **Decided (Rohit, 13 October): approved as worded,** for the capacity test.

    **Idea for after run 7 (Rohit, 13 October): work from the future.** Now that each transition waits a step, why not decide the present from the future, since path awareness, the route and the guard already look ahead? Recorded with the answer given:
    - **At decision time, FLOWRRA already decides from predicted futures.**
      - The route reads the distance to the destination.
      - Path awareness reads each robot's next 8 route cells.
      - The guard lets every robot declare its move first, computes where all of them would land, and only then lets anyone move (decide, guard, move).
      - That is look-ahead, or predictive control. Only predictions are available at that point, never the future itself.
    - **In learning, information from the actual future flowing back is legitimate and standard.** Every value is learned from what came after (bootstrapping). The honest target goes one step further: it waits to see what the system actually did. "Retrocausality" is fair as a metaphor for this, if the claims use the technical name: hindsight credit assignment. Decisions are made from predicted futures; learning is done from actual futures.
    - **Two natural extensions, both using the hold:**
      1. **n-step hindsight.** Hold a transition n steps, use the n real rewards and the moves actually made, and only then bootstrap. Less reliance on estimated values; more variance under exploration.
      2. **A Learner trained on what the whole system executed** (route, rules, ladder and guard together), not on RULES alone. The labels are the corrected moves, so the Learner learns to propose moves that need no correcting: the "fight" between the nudge and the values (run 4) should not arise. Imitation is capped at the system's level; the Q heads are what can go beyond it.
    - **Rohit's sharpening, the same evening: untangling a jam from the destinations backwards.** When robots bunch up, consider every joint state they could pass through. Choose the moves that untangle them without contact, so that each can still reach its destination. That is working backwards from the future to the present. Recorded with the answer given:
      - **It is a two-point boundary problem.** Where the robots are now and where each must end up fix the path between, as in physics' least action.
      - **The "future" is a present fact.** The destinations are known now. So this is backward reasoning from known goals, not physical retrocausality, and that is what keeps the language defensible.
      - **FLOWRRA already does it for one robot.** The goal-distance maps are computed backwards from each destination to every cell, and the route follows them downhill.
      - **For several robots in a jam, it is local space-time planning (windowed MAPF).** It has the same shape as wave function collapse in procedural generation:
        - every robot's possible positions over the next few steps start in superposition;
        - the most constrained one is fixed first;
        - the constraints (one robot per cell per step, no swaps, progress towards each destination) propagate;
        - the first step of the consistent plan is the move taken now.
      - **Ways to learn it:**
        - Solve each L2/L3 jam exactly offline; jams are small, a few robots over a few steps. Train a head on the solutions: imitation of a destination-consistent teacher.
        - A generative model over short joint trajectories, conditioned on the destinations. Diffusion planners (Janner et al., 2022) start from noise over whole trajectories and settle into one consistent with the start and the goal, close to the "collapse" picture; recent work extends this to several robots.
        - The guard stays underneath either way.
      - **Rohit's refinement: construct the second boundary by learning, then collapse.**
        - Only the goal's direction is known (the distance map's downhill move), not the robot's whole future path. So inside the small zone, choose for each robot the cell where it should leave the zone, and solve the zone's moves between where the robots are and those cells.
        - This is a horizon with a terminal cost (model-predictive control), or search with learned values at the leaves (as in AlphaZero).
        - **Step 1, the boundary.** Assign each robot an exit cell of the zone, no two the same, minimising the moves inside the zone plus the cost-to-go from the exit cell. The distance maps already give that cost-to-go exactly when there is no traffic. The network's values are what can price traffic beyond the zone, so this is where learning enters.
        - **Step 2, the collapse.** An exact space-time solve inside the zone, between the two fixed boundaries.
        - The ladder's L2 retrace is a crude version of step 1: one retreat cell, chosen by rule.
        - **Why run 7 comes first:** a learned boundary is only as good as the values behind it, so honest values are a precondition.


    **Run 7, step by step** (ticked off here as each is done):
    1. [x] Build Fix 1 and its test (13 October). Changed: `agent_warehouse.py` (the optional 15th field in push, padding with −1, sampling, and the bootstrap in learn), `core_warehouse.py` (the hold, `_honest_complete_pending`, `flush_pending_transition`, counters `honest_*`), `main_runner_warehouse.py` (the flush after each episode, `honest_` CSV columns, the log line), `config_warehouse.py` (`training.honest_target`, False). New: `test_honest_target.py`. Checked here without torch: the hold-and-complete rule on a stand-in (route move kept, network move −1, guard stop 0, a fleet that left −1, counts only active fleets) and the storage layout (14 fields with it off, 15 with it on, padding −1). The torch parts run on Rohit's machine (step 3).
    2. [x] Build Fix 2 and its test (13 October). `core_warehouse.py`: `_exit_without_work`, used at the three freeze sites ("pool exhausted while retargeting", the in-loop and the deferred retarget) and at a fourth path found while building. In stream maintenance, a fleet whose order was delivered by someone else and that finds nothing to retarget to used to drive on to the old goal cell and only freeze on arrival; it now heads for a dock at once. The delivery path's own no-dock park is counted too. Counters `stream_no_work_exits` and `stream_parked_no_exit`. `config_warehouse.py`: `stream.no_work_exit`, False. New: `test_stream_no_work.py`. Checked here without torch: the helper on a stand-in (sent out with the dock as its target and its exit leg timed; no dock: parks and is counted; switch off or outside the stream: nothing happens).
    3. [x] Rohit's machine: both new tests, the existing suite, and the `--check` with both switches off. All done, 13 October.
       - [x] `test_honest_target.py` ALL PASS, after one fix to the test itself: its hand-built tensors were on the CPU while the agent was on the GPU. `test_stream_no_work.py` ALL PASS.
       - [x] `--check` with both switches off (`deploy_check5`): reproduces 12 October exactly. Champion 31 delivered, 0 collisions, lead time 23.16; RULES 30, 0, 19.5; the repeat IDENTICAL. The new columns read 0. The log still shows "Fleet 25#2 frozen, pool exhausted", as it should with the switch off.
       - [x] The existing suite: all 13 pass (test_teacher_margin, test_phase_3, test_graph_pooling, test_learner_head, test_stream_failures, test_smoke_integration, test_line_end, test_intent_guard, test_conflict_rules, test_corridor_rules, test_ladder_retrace, test_ladder_authority, test_guard_integration). First pass: `test_teacher_margin.py` and `test_graph_pooling.py` failed `sample_returns_16`. Both are stale checks on the number of parts in a sampled batch: Fix 1 appends a 17th, last, so every index they read is unchanged. Both now expect 17, plus a check that the 17th is None when no transition carried the field. I missed these two when I searched for `sample()` callers: I checked which indexes they read, not the length they assert.
    4. [x] (config ready: RUN 7 header, `honest_target` and `no_work_exit` on, `eps_peak` 0.05; command in the header) Dry run A, **done 13 October: stable**: 10 episodes, warm from the champion, peak 0.05 / floor 0.01, run 5's failures, book 80, both fixes on. Read the delivery value (below 1.42, not climbing), the losses against exploration, and L3 against exploration.
       **Dry run A, results** (`ladder_run3_dry`, 10 episodes, warm from the champion, 43.8 minutes, no errors):
       - **Values honest and settling.** Delivery value by episode: 1.04, 0.82, 0.78, 0.83, 0.88, 0.83, 0.76, 0.74, 0.74, 0.74. Below 1.42 throughout, levelling off. Run 5, warm from the same champion without the honest target: 1.03 to 0.86. Run 6 climbed every episode.
       - **Efficiency value rising, but plausibly honest:** 0.56 to 1.34. With the route, every routed step makes progress: the efficiency reward is 0.042 per fleet-step here against run 5's 0.018, over lives of about 97 steps. A discounted return of 1.3 to 1.8 is in range. To watch in the run: it should level off.
       - **Losses falling.** Delivery 0.015 to 0.009; efficiency 0.008 to 0.006; safety about 0.001.
       - **Fix 1 working.** 70% to 97% of next moves valued as actually made. 800 transitions stored per 800-step episode, one of them by the final flush.
       - **Fix 2 working.** 31 robots sent out with no order and 0 parked; not one "frozen" or "recalled from retirement" line in the log.
       - **Safety.** 0 collisions and 0 invariant violations in all 10 episodes.
       - **L3 does not rise as exploration falls.** 3.0 per episode at exploration 0.03 to 0.05 (episodes 1–5); 2.4 at 0.01 (episodes 6–10). Episode 4, small map with 25 robots and 12 failures, had 10.
       - **Delivered 95.3%** (73 to 78 of 80).
       - **Reported:** "move away" near another robot 37%, the champion's level, with no change expected in 10 episodes.
       - **An observation to watch.** In the clear the network now disagrees with the route on 43% of moves; the frozen champion in the tests disagreed on 15%, and run 6's network on 52%. The honest target corrects the values of the moves the route makes, but the moves it never makes in the clear keep stale values. They no longer enter any target, so they cannot inflate anything. With the route on they are never executed, so this costs nothing on the floor; it matters only if the route were switched off. The Learner idea (the destination note) would address it.
    4b. [x] **Fix 3, explore everywhere, value the policy that runs: agreed and built (13 October).** Changed:
       - `agent_warehouse.py`: `last_explore_mask`.
       - `core_warehouse.py`: in the route, an exploring fleet in the clear is left to explore. It is recorded as eligible, with the route's move, and labelled with that move in the honest target. Counters `route_explore_on` and `route_explore_kept`.
       - `config_warehouse.py`: `conflict.route_explore` True for run 7; dry run B in the header.
       - `deploy_eval.py`: forced off in `apply_safety_mode` for every mode, which covers `trace_fleet` too; shown in the `[deploy] safety` line.
       - New: `test_route_explore.py`. `test_honest_target.py` pins it off, so it keeps testing Fix 1 alone.
       - Checked here on a stand-in: the explorer kept its move, the others were routed; labels route-driven 2, explorer 4 (the route's move), guard stop 0, network −1; switch off, the explorer was routed.
       - Rohit's machine, 13 October: `test_route_explore.py`, `test_honest_target.py` and `test_guard_integration.py` all pass; switches read `True True True 0.05`; dry run B started. Original proposal:
       - **The finding (Rohit's question).** Disagreement with the route in the clear jumps as soon as learning starts and then holds, whatever the exploration rate. The frozen champion disagreed on 23% of clear moves on the small map and 7% on the large. In the dry run: 35% in episode 1, about 50–55% on the small map and 35–44% on the large, flat while exploration fell from 0.047 to 0.010.
       - **The cause.** In the clear only the route's move is ever executed, even when the network had chosen to explore. The route overrides exploring robots too. So only that move's value is corrected; as it settles (the delivery value fell from about 1.0 to 0.74), the other moves keep stale values and some now rank above it.
       - **The fix.**
         - A robot whose move this step is an exploring move is not overridden by the route, so it explores in the clear at the Gaussian's rate (0.01 to 0.05). The guard keeps it safe.
         - Its lesson is valued honestly: the honest target labels a next-state robot that would have been routed with the route's move, the policy that runs there the rest of the time, even when it explored.
         - The stale moves are then corrected by real experience (principle 7), not by a margin or penalty.
       - **Switch and scope.** `conflict.route_explore`, off by default. It applies only in training: `deploy_eval` and `trace_fleet` force it off, because deployment never explores through the route.
       - **To check in a second dry run:** disagreement falls across episodes instead of holding, the delivery value stays under 1.42, and L3 and guard stops do not rise much.
       **Dry run B, results** (`ladder_run3_dryB`, all three fixes, 43.9 minutes, no errors). The same 10 scenarios in the same order as dry run A (same seed), so every episode is a paired comparison.

       | Mean over 10 episodes | A (fixes 1–2) | B (fixes 1–3) |
       | --- | --- | --- |
       | Network disagrees with the route in the clear | 43% | 40% |
       | Delivery value / efficiency value | 0.815 / 1.10 | 0.810 / 1.09 |
       | Delivery / efficiency loss | 0.011 / 0.007 | 0.010 / 0.007 |
       | L3 per episode | 2.7 | 1.2 |
       | Guard stops per episode | 52 | 87 |
       | Delivered (of 80) / lead time (steps) | 76.2 / 79.9 | 76.7 / 84.5 |
       | Collisions, violations, frozen robots | 0, 0, 0 | 0, 0, 0 |

       - **Values and losses unchanged:** the honest target holds with Fix 3 on.
       - **Disagreement moves the right way, slowly, and only on the large map.**
         - Part of the 3 points is the measure itself. A counted exploring robots that the route overrode (about 2–3 points at exploration 0.03–0.05); B does not.
         - The clean comparison is episodes 6–8 (the same large-map scenarios, exploration 0.01): 36–37% in A against 30–32% in B. Within B, the large map falls 36%, 32%, 31%, 30%.
         - The small map does not improve: episode 9, 55% in A against 57% in B.
         - At exploration 0.01 only 45–160 exploring moves an episode are corrected in the clear.
         - The per-episode swings follow the map (small map about 43–57%, large about 30–40%; Rohit: "the sudden map change"), so comparisons are made within a map.
       - **The cost, in training only:**
         - Guard stops rise, though very unevenly: up by 252 in episode 1 and 223 in episode 7, down by 155 in episode 9.
         - Lead time is 4.5 steps longer, from the detours in the clear.
         - Deployment forces route-explore off, so the frozen tests carry neither cost.
       - **Possibly a bonus, not yet firm:** L3 2.7 to 1.2 and recovery collapses 27 to 12, but episode 4 alone (10 to 2) carries most of it.
    5. [x] ~~Dry run B with the Learner on~~: held until after run 7 (decided 13 October).
    6. [x] Agree run 7's settings and predictions; then the run (50 episodes, about 4 hours). Done; scored in entry 35.
       **Agreed 13 October, before the run.**
       - **Settings:** all three fixes on; warm from the champion; exploration peak 0.05 and floor **0.02**. The floor was raised from 0.01 after dry run B, so route-explore corrects the clear about twice as fast. Training only: `deploy_eval` uses its own 0.015.
       - `recovery_policy.exploration_scale` stays 0.25 (Rohit asked). It has no effect in this run: with the ladder on, the learned recovery head is never consulted (`choose_recovery` is skipped; L3 is a fixed rule).
       - **Exploration by episode:** 0.026 at 1, 0.05 at about 13, 0.024 at 25, 0.02 from 40.

       | Prediction | Pass if |
       | --- | --- |
       | Values stay honest | delivery value below 1.42 in every episode |
       | Learning settles | delivery loss in the last 10 episodes no higher than in the first 10, and the efficiency value levels off (last-10 mean within 10% of episodes 31–40) |
       | Safety | 0 invariant violations, 0 collisions, 0 robots parked mid-floor (`stream_parked_no_exit` 0) |
       | Nothing gets worse | book ≥ 95% and L3 ≤ 3.4 per episode, every map and fleet size weighted equally |
       | The stale values get corrected | on the large map, disagreement with the route in the clear is lower in the last 10 episodes than in the first 10 (fleet sizes weighted equally) |

       Reported, not scored: the "move away" share near another robot; disagreement on the small map; a per-setting comparison with run 5 (same start, same length). Then the frozen tests decide: ladder_run3 against the champion on 3 of 5 measures, clean and full, guard on for every arm, route-explore off; RULES reported alongside.
       **Run 7, midpoint check (25 of 50 episodes, about 2 hours in; not a score).**
       - **Healthy.** No errors, 0 collisions, 0 invariant violations, 0 frozen or recalled robots, 69 no-work exits. The delivery value settled at about 0.80 (episode 1: 1.02) and the delivery loss is flat at about 0.010.
       - **Paired with run 5,** which had the same start and the same episodes (the same seed gives the same sequence of settings):
         - Lead time is far shorter on the small map: 40 robots 53 steps against 139, 60 robots 37 against 106, 25 robots 64 against 163. It is shorter on the large map too: 134 against 159.
         - Collisions are 0 against up to 33 per episode in run 5.
         - L3 is about 1.0 per episode against 2.3 to 5.4.
         - This is mostly the system (guard, route, fixes), not yet learning.
       - **To watch:**
         1. **The efficiency value is still rising** (0.58, 1.34 at episode 10, 1.68 at 25), and its loss is creeping up (0.007 to 0.010). Rough estimate of the honest level from the rewards: about 1.0 on the small map and about 2.5 on the large, where lives are about 155 steps and the route doubles the per-step progress. So this may be a slow climb towards a higher honest level; the "levels off" prediction decides.
         2. **Disagreement on the large map is flat** at about 31–40%, not falling. Fix 3 removed the early jump but is not pulling it lower. That prediction may fail.
         3. **The book is 93.6%** with settings weighted equally, against the prediction's 95%. The one large-map, 25-robot episode delivered 85%, and route-explore detours cost time in training only.
    7. [ ] Frozen tests, clean and full, guard on for every arm (ladder_run3, champion, RULES); then the capacity test with the top-up book (approved).

35. **Run 7, ladder_run3: scored (13 October).** 50 episodes, warm from the champion, all three fixes on, exploration 0.05 peak and 0.02 floor; 4.1 hours, no errors.

    | Prediction | Pass if | Result |
    | --- | --- | --- |
    | Values stay honest | delivery value below 1.42 in every episode | ✅ highest 1.075 (run 6: 2.13) |
    | Learning settles | delivery loss last 10 ≤ first 10; efficiency value levels off (last 10 within 10% of 31–40) | ❌ delivery loss 0.010 → 0.016; efficiency value +14.5% (1.80 → 2.06) |
    | Safety | 0 violations, 0 collisions, 0 parked mid-floor | ✅ all 0 over 50 episodes; also 0 fallback stops, 0 frozen, 0 recalled; 130 no-work exits |
    | Nothing gets worse | book ≥ 95% and L3 ≤ 3.4, settings weighted equally | ❌ / ✅ book 93.5% (large map with 25 robots 86% over 5 episodes); L3 1.48 |
    | The stale values get corrected | large-map disagreement lower in the last 10 than the first 10 | ❌ 33% → 44%: it rose in the second half, at exploration 0.02 |

    **2 pass, 2 fail, 1 split.**

    **The honest target did its main job.** About 90% of next moves were valued as actually made (78% to 100% per episode). The delivery value never came near the ceiling: 0.74 to 0.80 through episode 30, then a step to about 1.0 over episodes 31–36, flat after that, ending at 1.07. The delivery loss stepped with it, from 0.010 to about 0.016, then held.

    **But the values are not fully settled.**
    - **The step in the delivery value coincides with a jam.** Episodes 30–31 (small map, 25 robots) had 354 and 362 guard stops and L3 11 and 6, with the network driving 36–41% of moves. There the bootstrap is still Q-learning's best move, including moves the rules or the guard overrode and so never ran.
    - **The efficiency value rose all run** (0.58 to 2.16), its loss flat at about 0.011. The per-step reward did not change (0.038 to 0.039), but the second half drew more large-map episodes (13 against 11), where the honest level is higher (about 2.5 against about 1.0 on the small map). It looks like part mix, part drift, and the two cannot be separated here.
    - **Disagreement in the clear rose on the large map** (31–35% early, 37–49% late). Exploration at 0.02 corrects too few of the moves the route never makes.
    - **One residual cause fits all three:** values of moves that experience does not correct. In the clear they are the moves the route never makes; in contention, the moves the rules or the guard override, which the bootstrap's maximum can still pick.
    - **Clarified with Rohit (13 October): two separate pieces, often run together.**
      1. **SARSA keeps every value honest.** Each head's value of a state and move is learned from the move actually made next, whoever chose it. It answers "how good is what really happens?" It does not teach the network which move the route or the guard would make: moves never made still have no data.
      2. **Alignment to the route and the guard is imitation: the Learner head**, trained on the moves the whole system executed. It answers "what would the system do here?" It is a separate head, so it moves no Q-value (principle 7).
      - Together, the network's choice is consequences (honest Q) plus the system's judgement (the Learner). The Learner proposes moves that need no correcting, and the Q heads are what can go beyond the system.
      - The rewards were not the problem in run 6; the bootstrap targets were.
      - **Rohit: can the two be weighed together, so that they agree in the end?** Yes. The principled form is KL-regularised (soft) reinforcement learning, with the Learner as the prior policy:
        - **The Learner** learns π_sys(a|s), what the whole system does (route, rules, ladder, guard), by supervision.
        - **The Q heads** learn by (expected) SARSA under the policy actually run.
        - **The choice:** π(a|s) ∝ π_sys(a|s)^β · exp(Q(s,a)), that is, score = Q + β · log π_sys. This is exactly the form of the Learner's existing nudge.
        - Because the bootstrap uses the policy actually run, nudge included, the values become consistent with the combined policy. At the fixed point, the policy that is valued is the policy that runs.
        - **β is the one dial.** Large β copies the system; β near 0 trusts the values. A schedule that starts high and lowers β as the values prove honest lets the network graduate from imitation to its own judgement.
        - **Principle 7 holds:** Q-values move only from real rewards through moves actually made, and imitation lives in its own head and enters only at choice time.
        - **Avoid** pushing the Q-values towards the system's moves with a margin (DQfD): that was the margin teacher, retired when it drifted the values.
        - **A variant:** weight the imitation by the honest advantage (as AWAC/AWR do). Imitate the system where its move is at least as good as the alternatives; learn where it is not.
        - **Order agreed (Rohit, 13 October):** the frozen tests first; then SARSA everywhere, as one rule; then the KL-regularised Learner (retrained on the whole system's moves, with a β schedule).
        - **Run 4, read in this light:** the Learner was trained on RULES alone, with a fixed β of 0.1 and inflating values, and the network came to rely on the nudge. Trained on the whole system, with honest values and a falling β, it becomes the designed combination.
    - **A candidate fix, for after the frozen tests:** bootstrap from the move actually made for every robot, network-driven ones included. That is SARSA, the honest target applied everywhere, with no maximum left. At exploration 0.02 it values practically the same policy. It would not by itself fix the disagreement in the clear; that needs more exploration there or the Learner (the destination note).

    **Paired with run 5:** the same starting network and, by the seed, the same 50 episodes in the same order.

    | | run 7 | run 5 | p | run 7 better in |
    | --- | --- | --- | --- | --- |
    | Book delivered | 94.1% | 91.5% | 0.005 | 31 of 50 |
    | Lead time (steps) | 86 | 135 | < 0.001 | 48 of 50 |
    | Collisions per episode | 0 | 13.0 | < 0.001 | 44 of 50 (6 ties at 0) |
    | L3 per episode | 1.48 | 3.16 | < 0.001 | 31 of 50 |

    On the small map the lead time is 54 against 136 steps with 40 robots, 38 against 106 with 60, and 65 against 155 with 25. Most of this is the system (the guard, route by default, the fixes), and both runs learned on the job. The frozen tests separate the network's own contribution.

    **Next: the frozen tests**, clean and full, guard on for every arm, route-explore forced off, no-work exit on for every arm (the config's setting). Arms: ladder_run3, the champion and RULES. The decision rule, agreed before: ladder_run3 against the champion on 3 of 5 measures (delivered, lead time, guard stops, L3, near misses); RULES reported alongside. No winner is predicted. The champion's numbers will differ a little from entries 32–33, because the no-work exit is new for every arm.

36. **Frozen test for run 7, clean condition: scored (13 October).** No failures and no people. The same 18 seeded days as entries 26 and 32 (2 maps × 25/40/60 fleets × 3 seeds), three arms, 54 runs, 33 minutes. The `[deploy]` line confirmed: guard, bubbles, hops and the route on; route explore off.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions, all 18 days | L3 per day | guard stops per day | books finished | network differs from the route in the clear |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | RULES | 99.3% | 72 / 154 | 0 | 3.5 | 23 | 14 of 18 | (rules: 0%) |
    | champion | 99.4% | 75 / 161 | 0 | 3.3 | 143 | 16 of 18 | 14% |
    | ladder3 | 99.9% | 78 / 170 | 0 | 2.9 | 130 | 17 of 18 | 50% |

    0 invariant violations and 0 fallback stops in all 54 runs.

    **The days replayed exactly.** The champion's and RULES' rows are identical, day by day, to entry 32's. No robot ran out of work in the clean condition (0 no-work exits, 0 parked, no "frozen" line in the log), so the new switch never acted here. In the full condition it will: entry 33 had 131 freezes. Their count in the full test, now as no-work exits, also confirms the switch is on there.

    **ladder3 against the champion: no measure differs.** This is not the decision; the full test is.

    | Measure | ladder3 | champion | p | ladder3 better on |
    | --- | --- | --- | --- | --- |
    | Delivered | 99.9% | 99.4% | 0.29 | 2 days, worse on 1, 15 ties |
    | Lead time (steps) | 78 | 75 | 0.20 | 6 of 18 |
    | Guard stops per day | 130 | 143 | 0.73 | 5 of 18 (lower mean, but more stops on 10 days) |
    | L3 per day | 2.9 | 3.3 | 0.80 | 6 of 18 (more on 9; lower mean from a few bad champion days) |
    | Near misses with people | – | – | – | (no people in clean) |

    By the mean, ladder3 is ahead on three of the four measured (delivered, guard stops, L3) and behind on lead time. None is firm.

    **By map, the picture splits:**
    - **Small map**, where robots meet most and the network drives most: ladder3 needs the guard less (221 stops a day against 267) and has fewer stand-offs (L3 4.2 against 5.4). It delivers more (99.9% against 98.7%) and finishes 8 books against 7. It is slower: 50 steps against 46 (RULES: 39).
    - **Large map:** lead time ties (106, 105 and 105 for ladder3, the champion and RULES). ladder3 needs the guard twice as often (39 stops a day against 18), with L3 at 1.7 against 1.1.
    - **Disagreement in the clear** is 55% on the small map and 44% on the large one, against the champion's 22% and 7%. It is harmless while the route drives in the clear, but those stale values are next door to the meetings where the network does drive. That may be part of the extra guard stops on the large map; it is not shown.

    **Run 7's network against run 6's, on the same days** (valid, because the days replayed exactly): they are indistinguishable. Delivered 99.9% against 99.9% (p 0.56), lead time 78 against 78 (p 0.85), guard stops 130 against 130 (p 0.78), L3 2.9 against 2.6 (p 0.76). Only the disagreement differs: 50% against 62% (p < 0.001, lower on 16 of 18 days).
    - So the honest target did what it was built to do: values that stay honest (entry 35) and less disagreement. But in the clean condition it did not change what happens on the floor.
    - Every network trained under the guard (shake3, ladder2, ladder3) lands in the same place: 99.6–99.9% delivered, lead time 76–78, guard stops 130–141, L3 2.4–2.9. The champion is at 99.4%, 75, 143 and 3.3. Training under the guard shifts the network the same small way every time, slightly more delivered and fewer stand-offs for about 3 steps of lead time, and none of it is firm on 18 days.
    - In clean conditions, most of the outcome is set by the safety system: the route in the clear and the guard at meetings.

    **Against RULES.** RULES is still the faster driver where robots meet (lead time 72 against 78, p 0.030, faster on 14 of 18 days). It needs the guard far less (23 stops a day against 130, p 0.006).

    **Where the network beats RULES:**
    - **More books finished:** 17 of 18, the most of any arm on these days with the guard on (RULES 14, the champion 16, ladder2 and shake3 16).
    - **More delivered:** 99.9% against 99.3% (p 0.074, not yet firm). On the large map RULES leaves 1.4% of the book undelivered and finishes 5 books of 9; ladder3 delivers it all and finishes 9.
    - **Fewer stand-offs:** L3 2.9 against 3.5 (not firm).

    **Does run 7's training mix explain the map split? (Rohit's question; checked from run 7's metrics and log.)** The map and fleet count of each episode are random draws from the seed, not a set rotation.
    - **Maps were used evenly:** 26 small and 24 large episodes.
    - **Exploration was spread nearly evenly:** a mean of 0.030 on the small map and 0.028 on the large. One exception: the five peak episodes (12–16, 0.046–0.050) all fell on the small map, and the large map had 11 of the 20 episodes at the 0.020 floor.
    - **Fleet counts were not even:** 13 episodes with 25 robots, 17 with 40 and 20 with 60. The large map with 25 robots had only 5.

    | Training episodes | 25 robots | 40 robots | 60 robots |
    | --- | --- | --- | --- |
    | Small map | 8 | 8 | 10 |
    | Large map | 5 | 9 | 10 |

    **The mix does not explain the test.** By cell, on the clean test:
    - **Large map, 40 robots:** all of ladder3's extra guard stops are here. It had 50, 56 and 221 stops on the three days; the champion had 0 on each. Yet this was the large map's most-trained cell: 9 episodes, 7 of them at higher exploration.
    - **Large map, 25 robots:** the least-trained cell (5 episodes) is ladder3's best. It had 9 stops against the champion's 55, and the fastest lead time of all three arms (123 steps against 132 for the champion and 129 for RULES).
    - **Small map, 60 robots:** results swing from day to day for every arm (champion 495, 60, 75 stops; ladder3 256, 482, 1), so 3 days cannot separate them.
    - **Disagreement in the clear** is flat across fleet counts on the large map (42–46%), so it does not single out 40 robots either.
    - The whole schedule spans only 0.020–0.050: about 2 to 5 moves in 100. Even a perfect spread changes few moves.

    **For the SARSA run (open):** a balanced rotation of the six map × fleet-count cells, so that each gets the same number of episodes and the same exploration. It is not needed to explain this result, but it removes a question from the next one.

    **What is being tested (clarified with Rohit, 13 October).** Every arm runs with the route in the clear: ladder3, the champion and RULES alike. The `route_default` column is True in every row. Since entry 30 the route lives inside the env, and `--safety on` switches it on for everyone. The older `--route-arms` wrapper (the "champion+route" arm of entries 26–28) was not used here.
    - **Each arm is the .pth plus the same system:** the route in the clear, the ladder at stand-offs, the guard on every move. Only the decision-maker differs, so the differences between arms are the networks' (and the rules') own.
    - **ladder3 is deployed as it was trained.** Runs 6 and 7 trained with the route on, so its job is the meetings. The route decides about 96% of moves on the large map and 70–75% on the small; the network decides the rest, all at meetings.
    - **The champion was trained without the route,** and the route helps it (entry 26: lead time 120 → 75 before the guard). For RULES it changes nothing (0% disagreement).
    - **Route off with the guard on** would measure the network's own driving in the clear: something ladder3 was never trained to do, and the gap the SARSA and Learner work is meant to close. Not run; a possible diagnostic for later.

    **Next: the full test** (failures and people), the same three arms and 30 seeded days, about 1.5 hours. It decides ladder3 against the champion on 3 of 5 measures, with RULES reported.

37. **Frozen test for run 7, full condition: scored (13 October). ladder3 wins 4 of 5 and becomes the v2 champion.** Failures and people on. The same 30 seeded days as entries 27 and 33 (2 maps × 25/40/60 fleets × 5 seeds), three arms, 90 runs, 87 minutes. About 9.5 failures per shift in every arm. People cleared all debris and nobody timed out.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions, all 30 days | L3 per day | guard stops per day | near misses with people, mean (median) | books finished | network differs from the route in the clear |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | RULES | 95.1% | 83 / 189 | 0 | 4.3 | 28 | 12.7 (4) | 1 of 30 | (rules: 0%) |
    | champion | 93.8% | 84 / 189 | 0 | 2.5 | 139 | 40.0 (5.5) | 1 of 30 | 16% |
    | ladder3 | 95.0% | 87 / 197 | 0 | 1.8 | 86 | 8.1 (3) | 2 of 30 | 49% |

    0 collisions, 0 invariant violations, 0 fallback stops and 0 robots parked mid-floor in all 90 shifts.

    **The decision (the rule fixed before the run: the higher mean wins a measure, 3 of 5 to win):**

    | Measure | ladder3 | champion | Winner | p | ladder3 better on |
    | --- | --- | --- | --- | --- | --- |
    | Delivered | 95.0% | 93.8% | ladder3 | 0.082 | 19 of 30 days, worse on 6 |
    | Lead time (steps) | 86.6 | 83.6 | champion | 0.31 | 13, worse on 17 |
    | Guard stops per day | 86 | 139 | ladder3 | 0.12 | 17, worse on 11 |
    | L3 per day | 1.8 | 2.5 | ladder3 | 0.16 | 16, worse on 8 |
    | Near misses with people | 8.1 | 40.0 | ladder3 | 0.055 | 14, worse on 7 |

    **ladder3 wins 4 to 1** and becomes the v2 champion: the starting point for the SARSA run.
    - **No single measure is firm.** Near misses (p 0.055) and delivered (p 0.082) come closest.
    - **The champion's near-miss mean is driven by two very bad days** (397 and 394). Without each arm's two worst days it is 14.6 against 5.7, and the medians are 5.5 against 3. The direction holds; the size of the gap is inflated.
    - **The same pattern held in both conditions.** In the clean test (entry 36), delivered, guard stops and L3 also favoured ladder3 by the mean, and lead time favoured the champion. Two independent sets of days pointing the same way is support that a single test cannot give.
    - **One caveat:** ladder3 is the only arm that trained with the no-work exit, which every arm now runs with (see below).

    **By map:**
    - **Small map:** ladder3 needs the guard about half as often as the champion (133 stops a day against 243). It has fewer stand-offs (L3 2.8 against 3.5), far fewer near misses (9.0 against 68.6) and delivers a little more (96.2% against 95.6%). It is slower: 55.6 steps against 50.3, and RULES' 46.4.
    - **Large map:** ladder3 delivers more (93.7% against 92.1%; RULES 93.5%). It has fewer stand-offs (0.9 against 1.5; RULES 3.0) and fewer near misses (7.3 against 11.3). Lead time ties (118, 117 and 119). Guard stops tie (39 against 36).
    - **The clean test's weak spot did not repeat.** On the large map with 40 robots, ladder3 needed the guard less than the champion here (46 stops against 85), the reverse of the clean test. It was most likely day-to-day noise on 3 days.

    **Did run 7's training change the floor?** In the clean test, run 7's and run 6's networks were indistinguishable (entry 36). Under full stress, they differ against the common reference, the champion:
    - Run 6's ladder2 was worse than the champion on stand-offs (L3 3.7 against 2.2) and level on guard stops (127 against 123; entry 33).
    - Run 7's ladder3 is better on both (1.8 against 2.5, 86 against 139).
    - This is indirect: the two networks never ran on the same days, because the no-work exit changed the days. But it is the first sign, on the floor, that the honest target made the trained network better where it matters most, in a crowded and failing floor.

    **Against RULES (reported, not scored).**
    - **ladder3 has fewer stand-offs:** L3 1.8 against 4.3 a day (p 0.010, fewer on 19 of 30 days). This is the first measure on which a network beats RULES firmly under full stress. The champion's lead was just short in entry 33 (p 0.051).
    - **Fewer near misses with people:** 8.1 against 12.7 (p 0.068).
    - **Delivered ties:** 95.0% against 95.1% (p 0.86). On the large map ladder3 delivers a little more (93.7% against 93.5%) and is slightly faster (118 against 119 steps).
    - **RULES is still the faster driver where robots meet:** 83 steps against 87 overall (p 0.135, not firm), and 46 against 56 on the small map.
    - **RULES needs the guard about 3 times less:** 28 stops a day against 86 (p 0.007).

    **The no-work exit, on the floor (entries 33–34, Fix 2).** Every shift with no exit replayed entry 33's day exactly: 5 of 5 for RULES, 1 of 1 for the champion. So every change below comes from the exits, about 2.2 to 2.9 per shift.
    - **The safety hole is closed:** 0 collisions in 90 shifts, against 1 in 120 in entry 33, which came from a frozen robot. 0 robots parked mid-floor.
    - **People are safer around RULES:** near misses fell from 33.3 to 12.7 (not firm, p 0.17). For the champion they were unchanged (36.6 to 40.0).
    - **The cost is throughput, for every arm:**
      - RULES delivers about 1 order a shift less (77.0 to 76.1, p 0.002) and the champion 1.7 less (76.7 to 75.1, p < 0.001).
      - Fewer robots are on the floor (18.6 to 17.9 on average) and more are off it at the end (29.2 to 30.5). A robot that exits waits about 120 steps before the stream brings it back.
      - A frozen robot, by contrast, stayed on the floor and could be recalled. For RULES, handovers fell slightly (9.4 to 9.2, p 0.014).
    - **Open, for later; not built:** keep the safety and recover the throughput. For example, give a robot heading out a newly released order on its way to the dock, or let it wait at a safe spot that the guard treats as an obstacle. To be designed with the capacity test.

    **Next (agreed order):** design SARSA everywhere, as one rule, for approval before any code; then the KL-regularised Learner. The SARSA run starts warm from ladder3.

38. **Design for run 8: SARSA everywhere (13 October). For approval; no code yet.**

    **What is left of Q-learning after the honest target.** Since run 7, a lesson asks "how good is the next state?" and answers it, for each robot in that state, with:
    - **the move actually made,** if anything other than the network drove the robot: the route, a rule, a ladder hold or the guard;
    - **the route's move,** if the robot explored in the clear (Fix 3);
    - **the network's best move by today's values (the max),** if the network drove it. This is the last place the max is used. In run 7 it covered 11–22% of next moves on the small map (honest share 78–89%) and 1–7% on the large (93–99%).

    **Why that last max still matters (entry 35).** The network's best move by today's values can be a move that would never have run: one the guard would have stopped, or a rule overridden. Experience never corrects those values. Run 7 shows the signs:
    - **The delivery value stepped up and stayed up.** It went from about 0.80 to 1.0 over episodes 30–36, right after the small-map jams of episodes 30–31 (354 and 362 guard stops).
    - **The efficiency value rose in every episode,** 0.58 to 2.16, on both maps alike: 2.12 on the small map and 2.16 on the large by episodes 48–50. Yet the rewards suggest the small map's honest level is lower, because lives there are shorter. A climb that ignores the map points to the bootstrap more than to the episode mix.
    - **Neither is proven.** Run 8 is the test.

    **The rule, in one line: every robot in the next state is valued at the move it actually made there.** Only the network-driven case changes: from "the network's best move" to "the move the network actually made", its exploring moves included.

    | The robot in the next state was driven by | Run 7 (honest target) | Run 8 (SARSA) |
    | --- | --- | --- |
    | The route, a rule, a ladder hold, the guard | the move made | the move made |
    | Explored in the clear (Fix 3) | the route's move | the route's move |
    | The network | its best move (the max) | **the move it made** |

    - **This values exactly the system as deployed.** In deployment the route always drives the clear, so the route's move is the right label there. The network also explores in deployment (0.015 against 0.02–0.05 in training), so the move it made is the deployed policy, give or take that small difference.
    - **No max is left over moves that run.** Three cases keep today's form, by design:
      - a robot that left the floor: its value chain ends there, so nothing reaches the loss;
      - a step that ends the episode: no bootstrap at all;
      - the one transition per episode cut off at the step budget, which has no next step to read. It bootstraps from the best move, as is standard for a time limit: 1 transition in 800.

    **Why not expected SARSA.** Expected SARSA averages over what the policy might do: mostly its best move, plus the exploration. The "best move" part is the max again, over moves that may never run. Plain SARSA uses only what happened.

    **What it will not fix, said before the run:** disagreement with the route in the clear. SARSA keeps the values honest; it does not teach the network what the route would do. That is the Learner's job, next. So disagreement is reported, and not predicted to fall.

    **One known cost.** Lessons stay in memory for up to about 25 episodes (20,000 steps). So "the move actually made" was made by the network of up to 25 episodes earlier, and SARSA values a recent mix of the network's own behaviour. That can lag; it cannot inflate. Warm, at exploration 0.02–0.05, the network changes slowly.

    **Plumbing.**
    - `core_warehouse.py`: in `_honest_complete_pending`, the network-driven case writes the move made instead of −1 when the switch is on. A new counter, `honest_sarsa`, counts the network-driven moves labelled this way; `honest_net` still counts how often the network drove. A new share, `honest_labelled_share`, is labelled next moves over all active ones; it should read about 100%.
    - `agent_warehouse.py`: no change. `learn()` already uses every label ≥ 0.
    - `main_runner_warehouse.py`: the log line shows SARSA on and the labelled share. The CSV columns come through `honest_`.
    - `config_warehouse.py`: `training.sarsa`, False by default. It acts only with `honest_target` on, and prints a warning if set without it. A RUN 8 header.
    - **Off is exactly run 7's code.**

    **Tests before any training: `test_sarsa.py`.**
    1. **Switch off:** labels bit-identical to run 7's rule (network-driven −1).
    2. **Switch on, in real FLOWRRA steps:** 24 robots, the stream, the route, the guard and route-explore on, with exploration raised so every case occurs. Every active next robot is labelled with the move it made, and an explorer in the clear with the route's move. No active robot reads −1, and the counters match.
    3. **`learn()`:** with every label given, the target is exactly the reward plus γ times the target network's value of the labelled move. Changing the online network's preferences does not change the target: no max is left.
    4. **Then the three recent tests and the suite of 13.**

    **Decisions for Rohit.**
    - **A. Start from the champion or from ladder3?**
      - **From the champion (recommended).** With `--seed 0`, run 8 is run 7 with one switch changed: the same start and the same 50 episodes in the same order. Every episode pairs with run 7, and the frozen test (run 8 against ladder3) becomes a clean comparison of SARSA on the floor.
      - **From ladder3.** It builds on the new champion, 100 episodes in all. But if it wins, the cause could be SARSA or just 50 more episodes. And ladder3 starts with run 7's possibly inflated values (efficiency 2.16), which SARSA would first have to unlearn.
    - **B. Dry run first.** 10 episodes from the same start as the run, about 45 minutes. Check: values below 1.42, a labelled share of about 100%, losses not climbing, 0 collisions. Then agree the predictions, and run 50 episodes (about 4 hours).
    - **C. Keep the episode draw as it is:** same seed, no rotation. Entry 36's check found the map mix does not explain the results, and the same seed keeps runs 5, 7 and 8 paired episode by episode. A balanced rotation can come later, on its own.
    - Exploration, failures, the book and the three fixes stay as in run 7.

    **Predictions (proposed; to agree after the dry run).**

    | Prediction | Pass if |
    | --- | --- |
    | Every value labelled | at least 99.5% of active next moves labelled with a real move, in every episode |
    | Values stay honest | delivery value below 1.42 in every episode |
    | Learning settles (run 7 failed this) | delivery loss in the last 10 episodes no higher than in the first 10; efficiency value levels off (last-10 mean within 10% of episodes 31–40) |
    | The climb was the bootstrap | from the champion: efficiency value in the last 10 below run 7's 2.06, and no lasting step in the delivery value after the jam episodes 30–31. From ladder3: the efficiency value falls from 2.16 and levels off |
    | Safety | 0 violations, 0 collisions, 0 robots parked mid-floor |
    | Nothing gets worse | paired with run 7 episode by episode, neither the book delivered nor L3 is significantly worse (p < 0.05) |

    Reported, not scored: disagreement in the clear (not expected to fall) and the network-driven share. Then the frozen tests, clean and full: run 8 against ladder3, 3 of 5, with RULES reported.

    **SARSA, one level deeper (Rohit's question).**
    - **Deciding, in the core, every step:**
      1. The network scores every move for every robot (the three heads, weighted 6 : 4 : 1) and proposes one per robot, exploring at ε.
      2. The route replaces the proposal for robots in the clear.
      3. Rules and the ladder may override at stand-offs.
      4. The guard checks every pair and slows or stops.
      5. The robots move, and each earns three rewards (safety, delivery, efficiency).
    - **The hold.** The lesson for step t is (state, move made, rewards, next state, ?). The "?" is the move each robot makes in the next state, unknown until step t+1's decision chain has run. Then it is filled in, robot by robot, matched by ID, and the lesson is stored. This is unchanged from run 7; SARSA only changes what is written for network-driven robots.
    - **Learning, in the agent, every step once memory holds 64 lessons:**
      1. Sample 64 lessons.
      2. For each robot and each head: the target is the reward plus γ (0.99) times the target network's value of the next state at the move written in the "?". The same move is used for all three heads, so their weighted sum is the value of one real policy.
      3. The guess is the online network's value of the move made.
      4. The loss is the gap (smooth L1), over active robots only. The gradient moves the three heads and the shared trunk. The target network follows slowly.
    - **Why SARSA cannot inflate.** Q-learning's target uses the highest of the next state's values, including moves never tried there. Their errors never shrink, and the max keeps choosing the most overestimated one; through bootstrapping, the error compounds. SARSA's target uses Q(s′, a′) for a move a′ that really happened at s′. That (s′, a′) is itself the start of a stored lesson, so the value being borrowed is itself being trained. Every link in the chain was experienced. Its fixed point is the true value of the behaviour that runs.
    - **How the policy still improves.** SARSA evaluates; the choice improves. The network picks moves with high values. Exploration occasionally tries others, and if one does better, its value rises, it is picked more, and SARSA re-evaluates the new behaviour. Evaluate, improve, repeat (generalised policy iteration). It is slower than Q-learning's assumption of perfect play from here on, but every step of it is real.
    - **What SARSA learns that Q-learning does not:** the consequences of the whole system acting after the network, including the guard's stops, the ladder's holds and its own exploration. Near other robots, a move that the guard will stop next step is valued as what really follows: waiting. This is the cliff-walk lesson in RL textbooks. SARSA learns the path that is safe given its own occasional slip; Q-learning learns the edge and pays for it in exploration.

    **Rohit's idea: the Learner as the router, its moves going into SARSA, and SARSA's output into the three heads.** Mostly right, and close to the agreed KL design. One correction.
    - **The correction: SARSA is not a box in the network.** It is the rule for the answer key, the "?" in each lesson. It produces no output in the forward pass and adds no layer. The Learner and the three Q heads sit side by side on the shared trunk; they meet at decision time (score = Q + β × log π_Learner), not inside the network.
    - **"The Learner's moves go into SARSA": yes, through what actually happened.** When the Learner's nudge shapes a choice, the move made carries it, and SARSA writes the move made. The values become the values of the combined policy: network, Learner, route, rules, guard. This is the KL design's fixed point: the policy that is valued is the policy that runs.
    - **"SARSA's output goes into the three heads": yes, as their training targets.** One target per head per robot: that head's reward plus γ times that head's value at the move made next.
    - **"The Learner as the router": yes.** That is the Learner trained on the whole system's moves, extended to the clear:
      - Today it learns only from rules and holds (`teacher_sources`) and nudges only robots in a conflict.
      - As the router, it also learns from the route's moves in the clear and the guard's stops, and nudges everywhere.
      - Once it routes as well as the route, the hard-coded route can be switched off. The route-off frozen test (entry 36) then measures whether the network has learned to drive on its own.
    - **"Clean learning right into the network": clean as long as the Learner does not train the trunk.** Today it reads the trunk but never trains it (`h.detach()`). If imitation trained the trunk, it would move the Q-values without any reward, against principle 7. Keep it detached.
    - **One trap to avoid:** the Learner should learn only from moves the system decided (the teacher mask), never from the network's own. Otherwise it learns to imitate itself, and the nudge feeds on its own output.
    - **Order unchanged:** SARSA first, alone, so its effect is measured on its own; then the Learner as the router, with a β schedule.

    **Rohit's follow-up: make a′ a learned outcome, a′ = f(route, the network's move), optimised so that a′ comes close to the route.** The aim is right: blend the route's knowledge with the network's judgement, and pull the network towards the route. The lever is in the wrong place.
    - **a′ grades the move before it; it does not steer the next choice.** In the target r + γ·Q(s′, a′), a′ decides how good it was to arrive at s′. What the network chooses at s′ comes from its ranking of Q(s′, ·), learned from the lessons that start at s′. Writing the route's move into a′ would change how earlier moves are valued, not make the network choose the route.
    - **And it would bring back a value nothing corrects.** At a meeting, the route's move is usually the one the guard stops (it is the shortest path, straight at the other robot). If a′ is that move, the target borrows Q(s′, route's move) for a move that did not happen there. It is never corrected, which is run 6's inflation again, and principle 7 is broken. Any rule for a′ other than "what happened" reopens this.
    - **Where "close to the route" belongs: in the choice, through the Learner.** The Learner is trained, by its own loss, to match the route and the rest of the system: that is the "optimising function". The blend happens at decision time: score = Q + β·log π_Learner. SARSA then records the move that came out of the blend, after the route, rules and guard, as a′. So a′ is, in effect, f(route, network). The blend comes first, in the choice; the recording second, in the target.
    - **The guess stays Q(s, move made):** it is the thing being trained.
    - **Pulling the Q-values themselves towards the route** (a margin or a penalty) is the margin teacher, retired when it drifted the values.
    - **A legitimate later variant:** expected SARSA under the blended policy. The target averages Q(s′, ·) over softmax(Q + β·log π_Learner), the soft, KL-regularised target. It holds only when the blend is what actually runs. While the route, rules and guard still override at meetings, it is not, and plain SARSA is the honest one. Once the Learner drives the clear and the route is off, the two converge.
    - **Rohit, clarifying: "a′ as the move to make, not the move made".** a′ is the next step's move either way. "The move to make" is a decision, so it belongs in the deciding loop. That is where Rohit's zone idea fits: five robots in a warning zone, each given a safe exit point inside the density, then collapsed through the ladder. This is entry 34's boundary-then-collapse planner, a better decider. "The move made" is the record, and belongs in the learning loop: SARSA. If the planner decides well, the guard has nothing to stop, so the move made is the move to make, and SARSA learns the value of following the planner. The Learner can later imitate it. SARSA is the foundation that planner needs.

    **Decided (Rohit, 13 October): A, start from the champion; B, a dry run first; C, keep the same seed, no rotation.** Run 8 is named `ladder_run4`.

    **Run 8, step by step** (ticked off as each is done):
    1. [x] **Build (13 October).**
       - `core_warehouse.py`:
         - the switch `training.sarsa`, read at start-up, with a warning if it is set without the honest target;
         - in `_honest_complete_pending`, the network-driven case writes the move made when the switch is on;
         - counter `honest_sarsa`; stats `honest_sarsa_on`, `honest_sarsa` and `honest_labelled_share`.
       - `main_runner_warehouse.py`: with SARSA on, the safety line reads "SARSA: x% of next moves valued as actually made (y% the network's own)". The CSV columns come through the `honest_` prefix.
       - `agent_warehouse.py`: a comment only. `learn()` is unchanged.
       - `config_warehouse.py`: `training.sarsa` True, and the RUN 8 header with the dry run and run commands (`--resume ladder_run1/...`, `--seed 0`, `--out ladder_run4_dry` / `ladder_run4`). Run 7 moved to EARLIER RUNS.
       - New: `test_sarsa.py`. The earlier tests `test_honest_target.py` and `test_route_explore.py` now pin `sarsa` off, so each keeps testing its own fix. With SARSA on in the config, their "network-driven reads −1" checks would otherwise fail.
       - **Checked here without torch, on a stand-in of `_honest_complete_pending`.** Six robots plus one that left the floor: routed, explored in the clear, network-driven, guard-stopped, network-driven, and network-driven but inactive.
         - SARSA off: [2, 5, −1, 0, −1, −1, −1]. SARSA on: [2, 5, 5, 0, 1, 6, −1].
         - Counters: system-driven 3, network-driven 2, labelled by SARSA 2, so the labelled share is 1.0.
         - The robot that left reads −1 either way. The inactive one is labelled but not counted, and its bootstrap is masked in `learn()`.
    2. [x] **Rohit's machine, 13 October:** `test_sarsa.py`, `test_honest_target.py`, `test_route_explore.py`, `test_guard_integration.py` and the suite all pass. The switches read `True True True True {'eps_peak': 0.05, 'eps_min': 0.02, 'mu_frac': 0.25, 'sigma_frac': 0.125}`.
    3. [x] **Dry run, done 13 October: stable, passes every check** (`ladder_run4_dry`, 10 episodes, warm from the champion, 44.3 minutes, no errors). The same 10 scenarios in the same order as dry run B and as run 7's first 10. Neither is an exact pair: dry run B's exploration floor was 0.01 against 0.02 here, and run 7's first 10 episodes sat inside a 50-episode schedule.
       - **SARSA is live:** 100.0% of next moves valued as made in every episode. The network's own share was 11–23% on the small map and 1–4% on the large.
       - **Safety:** 0 collisions, 0 violations, 0 fallback stops, 0 parked; 28 no-work exits; no frozen or recalled robots; no tracebacks.
       - **Values honest.** The delivery value fell from 1.06 to 0.74, far below 1.42. It ran a little above dry run B's early on (9 of 10 episodes, p 0.004) and met it by episodes 9–10 (0.742 against 0.744). Per head, SARSA can be higher: the max picks the best move by the weighted sum, where safety counts 6 times, and that move can be worse for delivery than the one actually made.
       - **The efficiency value was lower than dry run B's in all 10 episodes** (p 0.002), by 3–12%: 1.30 against 1.36 at episode 10 (run 7: 1.34). But it climbed at nearly the same rate (0.45 to 1.30, against 0.47 to 1.36).
         - So most of the early climb is common to both. That fits an honest adjustment: the champion learned its values without the route, and with the route every step makes progress, so the true efficiency return is higher than the champion believed.
         - The leftover max adds a little on top: the 3–12%.
         - The open question is the run's second half, where run 7 kept climbing to 2.16.
       - **Losses:** the same as dry run B (delivery 0.009–0.012, efficiency 0.005–0.008, safety about 0.001).
       - **The floor:** no paired measure differs. Delivered 76.9 against 76.7; L3 1.4 against 1.2; guard stops 85 against 87; lead time 87.5 against 84.5 (p 0.19).
       - **Disagreement in the clear:** 36% against 40% (p 0.105), but 37% more exploring moves were kept in the clear (235 against 171 an episode, from the higher floor). Not credited to SARSA.
    4. [x] Agree the predictions; then the run (`ladder_run4`, 50 episodes, about 4 hours). **Agreed by Rohit, 13 October; run 8 started.** Final wording, set after the dry run:

       | Prediction | Pass if |
       | --- | --- |
       | Every value labelled | at least 99.5% of active next moves labelled with a real move, in every episode |
       | Values stay honest | delivery value below 1.42 in every episode |
       | Learning settles (run 7 failed this) | delivery loss in the last 10 episodes no higher than in the first 10; efficiency value levels off (last-10 mean within 10% of the mean of episodes 31–40) |
       | The late climb was the bootstrap | efficiency value in the last 10 at least 10% below run 7's (≤ 1.85 against 2.06); and the delivery value's mean over episodes 36–50 less than 0.10 above its mean over 21–29 (run 7: +0.20, the step after the jams) |
       | Safety | 0 violations, 0 collisions, 0 robots parked mid-floor |
       | Nothing gets worse | paired with run 7 episode by episode, neither the book delivered nor L3 is significantly worse (p < 0.05) |

       - The bootstrap threshold is set at 10% below run 7, not just "below": the dry run already sits 3–12% under dry run B. A small gap would mean the late climb was mostly honest, and the prediction should then fail.
       - Reported, not scored: disagreement in the clear, the network's own share, and lead time.
       - No winner is predicted on the floor; the frozen tests decide that.
    5. [x] Score it, paired with run 7 episode by episode: entry 39.
    6. [ ] Frozen tests, clean and full: ladder4 against ladder3, 3 of 5; RULES reported.

39. **Run 8, ladder_run4 (SARSA everywhere): scored (13 October).** 50 episodes, warm from the champion, run 7's settings with one switch changed (`training.sarsa`); 4.05 hours, no errors. The same 50 scenarios in the same order as run 7, so every episode is a paired comparison.

    | Prediction | Pass if | Result |
    | --- | --- | --- |
    | Every value labelled | ≥ 99.5% of active next moves labelled, every episode | ✅ 100% in all 50 episodes |
    | Values stay honest | delivery value below 1.42 in every episode | ✅ highest 1.04 (run 7: 1.07) |
    | Learning settles | delivery loss last 10 ≤ first 10; efficiency value last 10 within 10% of 31–40 | ❌ delivery loss 0.0105 → 0.0113, narrowly higher (run 7: 0.0100 → 0.0159); efficiency value +17.7% (run 7: +14.5%) |
    | The late climb was the bootstrap | efficiency value last 10 ≥ 10% below run 7's; delivery value step 21–29 → 36–50 under 0.10 | ✅ / ❌ **delivery step −0.001 (run 7: +0.210)**; efficiency only 1.8% below run 7 (2.02 against 2.06) |
    | Safety | 0 violations, 0 collisions, 0 parked | ✅ all 0; 0 fallback stops; 118 no-work exits; no frozen or recalled robots |
    | Nothing gets worse | book delivered and L3 not significantly worse than run 7, paired | ✅ delivered 93.8% against 94.1% (p 0.96); L3 1.96 against 1.48 (p 0.18) |

    **4 pass, 1 fail, 1 split.**

    **The finding: the leftover max was the delivery head's problem, and SARSA removed it.**
    - **Run 7's delivery value** stepped from 0.82 (episodes 21–29) to 1.03 (36–50), right after the small-map jams of episodes 30–31. It never came back down.
    - **Run 8's,** on the same episodes, stayed flat: 0.81 and 0.81. It ended at 0.80 against run 7's 1.07.
    - **The delivery loss stayed flat** at about 0.011, where run 7's rose 60% to 0.016.
    - This is entry 35's hypothesis confirmed for this head, by a clean paired test: the same start and the same episodes, with one switch changed.

    **The efficiency climb is not the max.**
    - Both runs climbed almost identically: run 8 from 0.49 to 2.06, run 7 from 0.58 to 2.16. The last-10 means are 2.02 against 2.06.
    - The gap was largest mid-run (5–7% over episodes 15–35) and closed late. The prediction was set to fail if the gap was small, and it failed.
    - So the efficiency climb is common to both runs. Two readings remain:
      - **Honest:** under the route every step makes progress, so the true efficiency return is higher than the champion's values (learned without the route) said. The memory, about the last 25 episodes, also held more large-map episodes late on, where lives are longer.
      - **A slower drift** that SARSA does not remove: bootstrapping through a network that generalises across states, a lagging target network, and replayed labels from older behaviour.
    - **A direct check would separate them:** compare the efficiency value of the moves made against the discounted efficiency return actually earned afterwards. Not built; a candidate measurement. A rough version from the existing columns follows (Rohit's question): it points to the honest reading.
    - **The efficiency loss is higher under SARSA** (0.0148 against 0.0114 in the last 10). Some of that is expected: a SARSA target follows one real next move, exploration included, so it is noisier than a max. But it rose with the value all run.

    **Rohit: do the rewards explain the efficiency climb? Yes. The value is catching up to a level the rewards set, not inflating past it.** Checked from the metrics CSV, both runs, without new code.
    - **The rewards themselves did not change over the run.** The efficiency reward the network learns from, per active robot-step, is about 0.054 on the small map (0.054 in the first 25 episodes, 0.053 in the last 25). On the large map it is about 0.071 (0.069 to 0.073). Run 7 is the same.
    - **A rough honest level from those rewards:** the reward rate times the discounted steps a robot has left in its life (γ 0.99, lives about 62 steps on the small map and 138 on the large), averaged over the roughly 25 episodes that the memory holds.
      - It is about 1.3 on the small map and 3.2 on the large.
      - Over the memory, it sits at about 2.3–2.8 from episode 5 on, with no trend.
    - **The learned value started far below that level and climbed towards it all run,** while the level stayed flat:

      | Episode | 1 | 10 | 25 | 40 | 50 |
      | --- | --- | --- | --- | --- | --- |
      | Level from rewards (memory-weighted) | 1.6 | 2.7 | 2.6 | 2.6 | 2.7 |
      | Learned efficiency value, run 8 | 0.49 | 1.32 | 1.54 | 1.88 | 2.06 |
      | Learned efficiency value, run 7 | 0.58 | 1.34 | 1.68 | 1.92 | 2.16 |

    - **So the "climb" is the value filling up.**
      - The champion learned its efficiency values without the route, when robots detoured and progress per step was lower: about 0.5.
      - Under the route, nearly every step makes progress, so the true value is higher.
      - With γ 0.99 a value sums about 100 future steps, and each lesson passes information back only one step, so it fills slowly. At episode 50 it is still below the level.
      - That is why "levels off" failed: it had not arrived.
      - That is also why SARSA barely changed it. The efficiency reward is earned nearly every step by whichever move runs, so the max has little to inflate.
    - **The same check on the delivery head agrees with the main finding.** The level its rewards set (one delivery, discounted by the steps until it) is about 0.62–0.67 from episode 10 on, flat. Run 8's delivery value is flat at about 0.80. Run 7's climbed to 1.02–1.08, after the jams.
    - **Caveats:** a rough estimate. It assumes a constant reward rate within a life, a uniform position in the life, the mean life length, and no end-of-episode cut-off; the levels could be off by perhaps a quarter. The trends (flat rewards, rising value) do not depend on those assumptions.
    - **A prediction for the next run from these weights:** the efficiency value should level off at about 2.5–2.8. Well above about 3 would mean drift.
    - **Rohit: what does efficiency do once safety and delivery have settled? Each head fills towards its own level at its own pace; efficiency is the last.** Run 8, 10-episode windows, against the level the rewards set (same rough method):

      | Episodes | Delivery: learned (slope) | Safety: learned / level | Efficiency: learned (slope) / level |
      | --- | --- | --- | --- |
      | 1–10 | 0.83 (−0.021 per episode) | −0.043 / −0.131 | 1.04 (+0.067) / 2.42 |
      | 11–20 | 0.79 (+0.011) | −0.053 / −0.093 | 1.36 (+0.007) / 2.50 |
      | 21–30 | 0.81 (−0.003) | −0.068 / −0.085 | 1.57 (+0.025) / 2.54 |
      | 31–40 | 0.83 (+0.002) | −0.085 / −0.084 | 1.72 (+0.018) / 2.40 |
      | 41–50 | 0.80 (+0.001) | −0.094 / −0.081 | 2.02 (+0.016) / 2.68 |

      - **Delivery settled first,** by about episode 11. It started near its level.
      - **Safety arrived by episodes 31–40,** at −0.085 against a level of −0.084. It started too optimistic (−0.04) and filled downwards. In the last 10 it is slightly past the level (−0.094), within the method's accuracy.
      - **Efficiency, once both had settled, is still climbing, but more slowly.** The gap to its level shrinks: 1.38, 1.13, 0.97, 0.68, 0.66.
      - **A fit to the gap from episode 10 on:** it halves about every 35 episodes. With these settings the value would come within 10% of its level (about 2.55) around cumulative episode 90, and within 5% around 125.
      - **Why efficiency is last:** it had the furthest to go (0.5 against about 2.5), and its value sums about 100 steps of small rewards. Delivery's is one event a few dozen steps away, and safety's is small.
      - **For the next warm run** (the Learner, from whichever network wins the frozen tests): about 50 more episodes should bring efficiency close to flat, near 2.5–2.8.
    - **Clarified with Rohit:** the climb follows training time, not the exploration floor. It ran from episode 1, at the highest exploration, and was steady throughout. The head weights stay 6 : 4 : 1. What grows is the efficiency head's accuracy: the robots' efficiency per step did not change; the network learned what that efficiency is worth. As its values become accurate, the differences between moves in that head sharpen, and with them its say in close choices. That is the network learning correctly, not the focus shifting.

    **Disagreement with the route in the clear fell, against the prediction.** It was 38.5% against run 7's 42.7% (p < 0.001), lower in 43 of 50 episodes. On the large map it went from 25% to 37% (first 10 to last 10), against run 7's 33% to 44%: about 7 points lower throughout, though still rising within the run.
    - The exploration schedule is identical here, so unlike dry run B, this is a fair pair.
    - **A plausible reading, not shown:** the max at meetings inflated the values of moves that lead into meetings, and that leaked one step back into the clear. Without the max, the route's move ranks first more often.
    - **Predicted wrongly, in the good direction.**

    **The floor in training: no paired difference from run 7.**
    - Delivered: 75.0 against 75.3 a shift (p 0.92).
    - Lead time: 85.5 against 86.2 (p 0.87).
    - Guard stops: 86 against 69 (p 0.69).
    - L3: 1.96 against 1.48. Worse on 23 episodes, better on 12; not firm (p 0.18).
    - Near misses with people: 78 against 54 (p 0.47).
    - **The two weakest episodes had more robot failures than run 7 had on the same days:** episode 29 (large map, 25 robots) had 15 failures against 8 and delivered 55 against 73; episode 8 had 16 against 11. This is the known order leak (item 3): failures draw from the shared `random`, so once choices differ, so do the failures. Training episodes pair only until then. The frozen tests seed every stream before each arm, so they do not have this problem.

    **The network's own share:** 15.7% of next moves on the small map, 4.8% on the large.

    **Next: the frozen tests**, ladder4 against ladder3 (the v2 champion), with RULES reported. The rule: 3 of 5 measures (delivered, lead time, guard stops, L3, near misses), clean and full. No winner predicted.
    - **Proposed, to save time (agreed):** run the clean test with all three arms (about 33 minutes). Nothing in deployment changed since entry 36: the SARSA switch acts only in training. So ladder3's and RULES' rows should be identical, day by day, to entry 36's. If they are, run the full test with ladder4 alone (about 35 minutes instead of 90), paired with entry 37's ladder3 and RULES rows on the same 30 days.


40. **Frozen test for run 8, clean condition: scored (13 October).** No failures and no people. The same 18 seeded days as entries 26, 32 and 36, three arms, 54 runs, about 32 minutes. The `[deploy]` line confirmed: guard, bubbles, hops and the route on; route explore off.

    **The days replayed exactly.** ladder3's and RULES' rows are identical, day by day, to entry 36's (18 of 18 each). Nothing in deployment changed with SARSA. So the full test can run ladder4 alone, paired with entry 37's ladder3 and RULES rows on the same 30 days.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions | L3 per day | guard stops per day | books finished | network differs from the route in the clear |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | RULES | 99.3% | 72 / 154 | 0 | 3.5 | 23 | 14 of 18 | (rules: 0%) |
    | ladder3 | 99.9% (1,439 orders) | 78 / 170 | 0 | 2.9 | 130 | 17 of 18 | 50% |
    | ladder4 | 99.9% (1,438 orders) | 76 / 166 | 0 | 1.9 | 78 | 17 of 18 | 41% |

    0 invariant violations, 0 fallback stops and 0 robots parked in all 54 runs.

    **ladder4 against ladder3: no measure differs firmly.** This is not the decision; the full test is.

    | Measure | ladder4 | ladder3 | Higher mean | p | ladder4 better on |
    | --- | --- | --- | --- | --- | --- |
    | Delivered | 99.86% | 99.93% | ladder3, by 1 order in 1,440 | 0.66 | 1 day, worse on 1 |
    | Lead time (steps) | 76.2 | 77.9 | ladder4 | 0.42 | 10 of 18, worse on 8 |
    | Guard stops per day | 78 | 130 | ladder4 (−40%) | 0.26 | 9, worse on 7 |
    | L3 per day | 1.9 | 2.9 | ladder4 | 0.33 | 6, worse on 5 |
    | Near misses with people | – | – | – | – | (no people in clean) |

    - By the mean, ladder4 is ahead on 3 of the 4 measured; delivered goes to ladder3 by a single order.
    - **Disagreement with the route in the clear is lower on every day:** 41% against 50%, 18 of 18 (p < 0.001). This is training's finding (entry 39) holding in the frozen test.

    **By map:**
    - **Large map:** ladder4 needs the guard half as often (18 stops a day against 39) and has half the stand-offs (L3 0.8 against 1.7). Lead time ties (105.4, 105.7; RULES 105.0).
      - ladder3's clean weak spot, 40 robots, is gone: 28 stops a day against 109.
    - **Small map:** fewer guard stops (137 against 221), fewer stand-offs (3.1 against 4.2), faster (47.1 against 50.2 steps; RULES 39.1).
      - **With 60 robots it nearly matches RULES:** lead time 33.7 against 32.2 (ladder3 43.8).
      - **A weak spot, 25 robots:** guard stops 264 against 227 and L3 7.0 against 2.7. One day carries most of it (seed 1: L3 12 and 2 orders undelivered); on another, guard stops were 212 against 44.

    **Against RULES:**
    - **RULES is still faster** (72 steps against 76, p 0.043) and **still needs the guard less** (23 stops against 78, p 0.010).
    - **ladder4 has closed part of both gaps** relative to ladder3: lead time 6 steps → 4, guard stops 130 → 78.
    - **ladder4 finishes more books** (17 of 18 against 14) and delivers more (99.9% against 99.3%, p 0.16). It has fewer stand-offs (1.9 against 3.5, p 0.53).

    **Next: the full test, all three arms (decided by Rohit, 13 October)**, about 90 minutes. It costs 55 minutes more than ladder4 alone, and buys two things:
    - a self-contained comparison: every arm runs through the same days in the same session;
    - a second replay check, this time with failures and people. ladder3's and RULES' rows should match entry 37's day by day.


41. **Frozen test for run 8, full condition: scored (13 October). By the agreed rule ladder4 wins 3 to 2, but the one firm difference goes against it: near misses with people.** Failures and people on. The same 30 seeded days as entries 27, 33 and 37, three arms, 90 runs, about 90 minutes. The `[deploy]` line confirmed the safety settings and route explore off.

    **The days replayed exactly, with failures and people too.** ladder3's and RULES' rows are identical to entry 37's on all 30 days. With entry 40, that makes 48 of 48 days replayed exactly for each, across both conditions. The shortcut would have given the same rows, and running all three arms gave a self-contained file plus this check.

    | | book delivered | lead time, mean / 90th pct (steps) | collisions, all 30 days | L3 per day | guard stops per day | near misses with people, mean (median) | books finished | network differs from the route in the clear |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | RULES | 95.1% | 83 / 189 | 0 | 4.3 | 28 | 12.7 (4) | 1 of 30 | (rules: 0%) |
    | ladder3 | 95.0% | 87 / 197 | 0 | 1.8 | 86 | 8.1 (3) | 2 of 30 | 49% |
    | ladder4 | 95.5% | 86 / 192 | 0 | 1.2 | 87 | 14.1 (5.5) | 0 of 30 | 39% |

    0 collisions, 0 invariant violations, 0 fallback stops and 0 robots parked in all 90 shifts. About 9.4 failures per shift in every arm.

    **The decision (the rule fixed before the run: the higher mean wins a measure, 3 of 5 to win):**

    | Measure | ladder4 | ladder3 | Winner | p | ladder4 better on |
    | --- | --- | --- | --- | --- | --- |
    | Delivered | 95.5% | 95.0% | ladder4 | 0.52 | 13 of 30 days, worse on 12 |
    | Lead time (steps) | 85.8 | 86.6 | ladder4 | 0.67 | 16, worse on 14 |
    | Guard stops per day | 86.7 | 85.9 | ladder3, by 0.8 a day | 0.67 | 12, worse on 15 |
    | L3 per day | 1.2 | 1.8 | ladder4 | 0.61 | 10, worse on 11 |
    | Near misses with people | 14.1 | 8.1 | **ladder3, firmly** | **0.007** | 7, worse on 16 |

    **By the rule, ladder4 wins 3 to 2.** The three it wins are not firm. One of the two it loses is: near misses with people, the only firm difference in the test.

    **The near misses, looked at more closely:**
    - **It is broad, not one bad day.** ladder4 was worse on 16 days and better on 7. Three days carry 116 of the 178 extra near-miss steps; without them it is 9.2 against 6.9.
    - **Part of it is exposure.** On ladder4's days more people came onto the floor (3.6 against 3.2 a shift) and they moved more (237 steps against 217), because more debris was dropped (3.6 against 3.4). Once choices differ, those details drift. Per step a person moved, it is 0.059 near misses against ladder3's 0.037 and RULES' 0.054.
    - **ladder4 is at RULES' level** (14.1 against 12.7, p 0.99; medians 5.5 against 4). ladder3 was the one unusually low.
    - **Where:** mostly the small map, with 40 robots (18.2 against 4.8) and 60 (22.6 against 12.8).
    - **Not explained yet.** In the clear the route drives either way, so the difference must come from moves the network makes near meetings while people are close. To look at with `trace_fleet` on the worst day (small map, 40 robots, seed 4: 78 against 15).

    **Where ladder4 is better:**
    - **The fewest stand-offs of any arm:** L3 1.2 a day. Against RULES' 4.3 that is firm (p 0.001, fewer on 21 of 30 days), and stronger than ladder3's 1.8 (p 0.010).
    - **The most delivered of all three arms** (95.5%). On the large map it is 94.7%, against 93.7% and 93.5%.
    - **Small map with 60 robots: faster than RULES,** 34.7 steps against 36.1 (ladder3 45.7). It needs the guard less than half as often as ladder3 there (46 stops against 103).
    - **The gap to RULES on lead time narrowed** to 3.0 steps (ladder3: 3.8; p 0.15).
    - **Disagreement with the route in the clear is lower on every day:** 39% against 49%, 30 of 30.

    **Where it is not:**
    - Guard stops: level with ladder3, and still 3 times RULES' (p 0.001).
    - It finished no full book (ladder3 2, RULES 1). Under full stress a finished book is rare for every arm.
    - The small map with 25 robots is still its weakest setting: guard stops 201 against 180.

    **Reading.** Clean and full together:
    - ladder4 is ahead of ladder3 by the mean on most measures in both conditions. None of those is firm.
    - Its stand-offs are the lowest of any network so far.
    - The one firm result is a regression on near misses with people, to RULES' level.

    **To decide (Rohit):**
    - **The rule promotes ladder4.** Changing a rule after seeing the result would undo what the rule is for.
    - **Proposed for every future comparison, fixed now, before the next test:** a safety gate. A challenger is not promoted if it is firmly worse (p < 0.05) on collisions or on near misses with people, whatever its score on the 5 measures. Applied to this test, it would have kept ladder3.
    - **The champion (what would be deployed) and the next run's starting point can differ.** Whichever is champion, the next run is better started from ladder4: its values are honest (delivery 0.80 against ladder3's inflated 1.07), so SARSA has nothing to unlearn.

    **Rohit's checks (14 October): the typical day, and like-for-like days.** "We are at the end of tuning, so the increments will be small. Look at the mean and the mode, and at like-for-like days: the same debris, the same number of robots failing."

    **The typical day.** For the counts the mode is 0 on many days (L3: 13 days each; near misses: 10 each), and lead time hardly repeats, so the median stands for "a typical day". ladder4 / ladder3 (RULES):

    | Measure | Mean | Median | Mode | ladder4 better on |
    | --- | --- | --- | --- | --- |
    | Delivered (orders) | 76.4 / 76.0 (76.1) | 76 / 76 (76) | 76 / 75 | 13 days, worse on 12 |
    | Lead time (steps) | 85.8 / 86.6 (82.9) | 78.2 / 78.3 (69.8) | – | 16, worse on 14 |
    | Guard stops | 86.7 / 85.9 (28.1) | 32.5 / 36.0 (0) | 0 / 0 | 12, worse on 15 |
    | L3 | 1.2 / 1.8 (4.3) | 1 / 1 (2) | 0 / 0 | 10, worse on 11 |
    | Near misses with people | 14.1 / 8.1 (12.7) | 5.5 / 3.0 (4.0) | 0 / 0 | 7, worse on 16 |

    - **On a typical day the two are nearly twins.** The medians tie on delivered, lead time and L3. ladder4 is a little lower on guard stops (32.5 against 36); ladder3 is lower on near misses (3 against 5.5).
    - **ladder4's lead in the means comes from fewer bad days.** Fewer stand-off pile-ups (the L3 mean, 1.2 against 1.8, with equal medians) and a slightly shorter tail of lead times (90th percentile 192 against 197).

    **Like-for-like days.** Robot failures, debris and people differ between arms on the same seeded day: the known order leak (item 3). Once choices differ, the shared random draws differ. So days were matched after the fact:
    - **The same number of failures:** 8 of 30 days. Near misses 12.9 against 7.9 (p 0.094).
    - **The same debris:** 14 of 30 days. Near misses 13.4 against 9.8, medians 1.5 against 3.0, ladder4 worse on 5 and better on 4 (p 0.41). On these days ladder4 also delivered more (76.2 against 75.2, better on 7, worse on 3).
    - **The same failures and the same debris:** 4 days, too few to read.
    - **Near misses per unit of exposure:**
      - per step a person moved: 0.056 against 0.045 (p 0.38);
      - per person entering: 3.9 against 2.5 (p 0.047);
      - per piece of debris: 3.9 against 2.3 (p 0.019).
      - RULES is at 0.067, 3.5 and 3.3, so ladder4 stays at RULES' level on every one.
    - **A regression of the daily near-miss gap on the exposure differences:** exposure explains about 1 of the 5.9 extra near-miss steps; about 5 are left. With 30 days and three factors, this is a hint, not a measurement.

    **What it adds up to.**
    - The near-miss gap shrinks and loses firmness on two like-for-like views (same debris; per person-step), but keeps it on two others.
    - It is weaker evidence than the headline p 0.007, and it does not vanish.
    - ladder4 is at RULES' level on every view.
    - The other measures: ladder4's edge is fewer bad days, not a better typical day.

    **Making like-for-like days automatic.** Fix the order leak by seeding every draw from the episode, the step and the robot (or the debris spot): failures, debris and people's choices. Then every arm faces the same failures and the same debris on the same day, as far as its own robots allow. It is already planned with the per robot-hour failures (item 5). Done before the next frozen test, it would make "same day" mean the same events.

    **Decided (Rohit, 14 October): ladder4 (run 8, `ladder_run4`) is the new v2 champion.**
    - It wins the agreed rule in the full test (3 to 2), and leads by the mean in the clean test (3 of the 4 measured, entry 40).
    - **In the clean test it also leads on the typical day:** median lead time 70.5 against 71.4, guard stops 32 against 47, L3 0.5 against 1.0; delivered ties at 80. RULES: 64.1, 0 and 1.5.
    - **Its values are honest,** so it is the starting point for the next training run.
    - **Carried with the decision:**
      1. The safety gate for every comparison from now on: no promotion if firmly worse on collisions or on near misses with people. Fixed now, before the next test.
      2. Trace the worst near-miss day (small map, 40 robots, seed 4) with `trace_fleet`.
      3. Fix the order leak (seed failure, debris and people draws from the episode, step and robot) before the next frozen test, so that every arm faces the same events on the same day.


42. **Design for run 9: the Learner as the router (14 October). For approval; no code yet.** Built on entries 34 (the destination note), 35 (the KL design) and 38 (SARSA, and Rohit's "Learner as the router").

    **What it is for.** The Learner is a learned copy of what the whole system does (π_sys): the route in the clear, the rules and the ladder at stand-offs, the guard's stops at close calls. It has two jobs:
    1. **Now, at meetings:** propose moves that will not need correcting. ladder4 still needs the guard about 3 times as often as RULES (87 stops a day against 28); the guard stops are the target.
    2. **Later, in the clear:** learn the route well enough that the hard-coded route can be switched off, and the network drives everywhere on its own. The route-off frozen test (entry 36) is the graduation exam.

    **What already exists (step 3a, entry 17; run 4).**
    - A small head (Linear, ReLU, Linear: 7 move scores) reads the shared trunk with `h.detach()`, so it never trains the trunk (principle 7).
    - It learns by cross-entropy, over valid moves, on fleets the teacher mask marks. Today that means RULES' orders and, optionally, ladder holds (`teacher_margin.sources`).
    - At choice time it nudges only fleets in a conflict (the warning or deadlock set): score = Q + β × its probability ("prob") or log-probability.
    - β is fixed, or "gap": the running median of the Q heads' top-two gap at meetings × a scale.
    - **Run 4's lesson:** trained on RULES alone, with a fixed β and inflating values, the network came to lean on the nudge, and lost to run 2 five measures to nothing (entry 21).

    **What changes.**
    1. **Teacher: the whole system, never the network's own moves.** The Learner learns from every fleet whose move the system decided:
       - **route:** routed fleets in the clear, labelled with the route's move;
       - **rules:** RULES' orders at stand-offs, including retrace and corridor back-outs;
       - **holds:** ladder holds;
       - **guard:** fleets the guard stopped, labelled "wait".
       - **Not taught:** a network move that ran unchanged, so the Learner never learns to imitate itself. Also not taught: an explorer in the clear (its executed move is noise).
       - These are its own sources (`training.learner.sources`), separate from the retired margin teacher's.
    2. **It learns before it speaks: a shadow warm-up.** For the first 5 episodes β = 0. The Learner trains and the gaps are collected, but it has no voice. This keeps entry 21's rule: the Learner is never switched on around a trained network.
    3. **β falls on a schedule, tied to the network's own confidence.** β = scale × the running median top-two gap at meetings, with the "prob" nudge. Since a probability lies between 0 and 1, scale reads directly:
       - "a fully confident Learner can overturn a Q preference up to scale × a typical near-tie gap";
       - scale 2.0 lets it overturn most preferences at meetings; 0.5 lets it decide only close calls.
       - **Schedule:** 2.0 → 0.5, linearly over episodes 6–50. The final 0.5 is what deployment uses.
       - The network graduates from imitation to its own judgement as its values (honest under SARSA) take over.
    4. **The Q heads keep SARSA unchanged.** The move made already carries the Learner's influence (entry 38). So the values become the values of the combined policy (Q, the Learner, the route, the rules, the guard): the KL design's fixed point.
    5. **The nudge stays at meetings in run 9, and the route stays on.** In the clear the route overrides any nudge, so the Learner only learns there. Switching the route off is a later phase, once the Learner agrees with the route on about 98% of clear moves.

    **One level deeper: what happens at one meeting.** Robot A is closing on robot B; both are in the warning set, so the network drives A.
    - **The Q heads** score A's moves. "Advance" leads "wait" by a small gap, say 0.3 (a near-tie).
    - **The Learner,** having seen the guard stop hundreds of robots in states like this, gives "wait" 0.8 and "advance" 0.1.
    - **With β = 1.0 × the median gap** (say 0.4), the nudge adds 0.32 to "wait" and 0.04 to "advance". The 0.3 gap is closed and A waits.
    - **Without the Learner,** A advances, the guard stops it, and the step is the same wait, but with a guard stop counted.
    - **SARSA records "wait" as the move made** either way, so the values learn that "here the system waits" (honest).
    - **As β falls,** A waits only when Q itself is nearly indifferent. If the values have learned that advancing is genuinely better (B is about to move away), Q's gap grows past the nudge and A advances.
    - **In the clear,** robot C is routed; the Learner learns "route's move here" and has no voice. Its agreement with the route is the readiness gauge for switching the route off.

    **A trade-off to expect, said before the run.** At meetings the Learner sees only the system's interventions: stops, holds, back-offs. So it leans towards waiting.
    - **Expected:** fewer guard stops.
    - **Possible:** more stand-offs (L3) and slower lead times. RULES' polite symmetry is what gives RULES the most L3 (entries 32–33).
    - The falling β exists to hand stand-off breaking back to the network's own values.
    - **A later variant:** advantage-weighted imitation (AWR). Also teach the network's own accepted moves, weighted by how much better they turned out than average. The Learner then copies only what worked.

    **Plumbing.**
    - **`core_warehouse.py`:** the teacher mask gains the "route" and "guard" sources. Plus live agreement per source: at each step, the Learner's top move from this step's forward pass against the move the system made, tallied for route, rules, holds and guard.
    - **`agent_warehouse.py`:**
      - The Learner gets its own optimizer, and the main one excludes it. A warm start from ladder4, which has no Learner, then loads the main optimizer's state exactly. `load()` accepts missing Learner weights only, logs that, and starts the Learner fresh.
      - Its initial weights are drawn from a separate random stream, so building it changes no other random draw.
      - The β scale is set per episode by the runner.
    - **`main_runner_warehouse.py`:** the schedule (warm-up, start, end); CSV columns `learner_agree_{route,rules,holds,guard}`, `learner_changed` share, `beta_eff`.
    - **`benchmark_flowrra.build_agent` (used by `deploy_eval`):** build each arm with or without the Learner according to its own checkpoint, not the config. Today a mixed comparison (ladder5 against ladder4) would fail to load one or the other. A Learner arm uses the schedule's final scale.
    - **`config_warehouse.py`:** `training.learner` enabled, nudge "prob", beta_mode "gap", `sources` [route, rules, holds, guard], `warmup_episodes` 5, `scale_start` 2.0, `scale_end` 0.5; a RUN 9 header.

    **Tests before any training (`test_learner_router.py`).**
    1. The teacher mask marks routed, rules, holds and guard-stopped fleets. It does not mark network-unchanged fleets or explorers.
    2. A Learner-only step changes only the Learner's weights: the trunk and the Q heads are bit-identical (principle 7).
    3. A warm start from a checkpoint without the Learner loads; the main optimizer's state is the checkpoint's.
    4. **Shadow is silent:** with β = 0, choices and Q updates over real FLOWRRA steps are identical to the Learner switched off.
    5. A uniform Learner changes no choice; the nudge reaches only conflicted, network-driven fleets.
    6. The β schedule's values per episode.
    7. `deploy_eval` loads a Learner arm and a non-Learner arm side by side.
    8. Then `test_learner_head.py` (updated where needed), the SARSA tests and the suite.

    **Decisions for Rohit.**
    - **A. Start from ladder_run1 (as run 8) or from ladder4?**
      - **From ladder_run1, recommended for the measurement.** With `--seed 0`, run 9 is run 8 with the Learner added: the same start and the same 50 episodes. The frozen test (ladder5 against ladder4) is then a clean A/B of the Learner, both 50 episodes from the same start.
      - **From ladder4: the lineage.** 100 episodes in all, and entry 39's prediction (efficiency levelling off at about 2.5–2.8) becomes testable. But a win could be the Learner or simply 50 more episodes.
    - **B. Teacher sources:** route, rules, holds and guard; not the network's own moves (recommended).
    - **C. Nudge and schedule:** "prob" with gap-tied β; scale 2.0 → 0.5 over episodes 6–50 after a 5-episode shadow; deployment at 0.5 (recommended). "logprob" is the exact KL form, but unbounded: a move the Learner rates at 0.1% is pushed down by 7β, a veto.
    - **D. Scope:** nudge at meetings only, route on; route-off is a later phase.

    **Then:** build and tests (on Rohit's machine), a dry run (10 episodes), predictions, the run (50 episodes, about 4 hours), the frozen tests under the safety gate (entry 41).

    **Revised the same night (Rohit, 14 October): the Learner as the router in the clear, not at meetings.** Rohit's points:
    - The problem was always in the clear more than in the waiting: the detour habit (entry 25), and the stale values of moves the route never makes (entries 31, 34–35).
    - At meetings the network has reached RULES. ladder4 ties RULES on delivered (95.5% against 95.1%) and near misses, and has firmly fewer stand-offs (L3 1.2 against 4.3).
    - L3 below about 5 a day is the FLOWRRA idea working: a spatial or temporal collapse of a jam before anyone collides. So why teach RULES at meetings?

    **Agreed. The original version would have spent the Learner on the network's strongest skill.**
    - Copying the system's interventions at meetings leans towards waiting (the trade-off above), and risks the stand-off breaking that beats RULES.
    - The remaining meeting gap, guard stops (87 against 28), is the guard doing its job safely. It costs a little time, not safety.
    - So the meeting layer is, for now, complete: safety by construction, the ladder, and the network's judgement. Open items are elsewhere: the clear, near misses with people (entry 41), the order leak, and the realistic conditions (item 5).

    **Is the route "better" in the clear?** In an empty aisle, the route's move (the downhill step on the goal's distance map) is the shortest path: it cannot be beaten locally.
    - **The network can beat it only by anticipation:** among equally short moves, pick the one leading into less traffic, or avoid an aisle that will jam before the robot enters the warning zone.
    - **The network already has the route's information.** Its inputs include the goal gradient (6 dimensions, `node.get_goal_gradient()`), whose downhill direction is exactly the route's move. So its 40% disagreement in the clear is a value problem, not missing information. The route drives there, so the network's moves there never run, and their values are never corrected.
    - **The Learner as router fixes both.** It hands the clear to the network gradually, and the network's own moves there finally get real experience.

    **The revised design (run 9).**
    - **The Learner learns the route's move for every fleet in the clear,** whether or not the route executed it. The route is a function and can always be computed, so this is distillation of the route, never the network's own moves. Nothing is taught at meetings.
    - **Handover in one run:**
      1. **Warm-up, route on:** the Learner learns in shadow (β = 0) for at least 5 episodes, and until it agrees with the route on at least 98% of clear moves. Since the gradient is in its inputs, this should come quickly.
      2. **Then route off for the rest of training.** In the clear: score = Q + β × π_Learner. At meetings: Q alone, as today, with the rules, the ladder and the guard after it.
    - **β:** gap-tied, measured in the clear; scale 3.0 → 1.0 over the episodes after the switch. Deployment uses 1.0. With scale 3.0 the Learner wins almost every clear choice, so the switch should barely change behaviour.
    - **The Q heads keep SARSA.** Now the clear is experienced: the network's own moves there run, guided by the Learner, and are valued as made. As β falls, a move Q prefers over the route gets tried more often and valued honestly. If it is worse, Q learns that; if it is better (traffic-aware), it stays.
    - **`route_explore` becomes moot after the switch:** exploration happens in the clear at ε, as everywhere.
    - **The guard stays on everywhere,** so the hand-over is safe by construction.

    **The graduation exam.** A frozen test of ladder5 with the route off (Learner at scale 1.0) against ladder4 with the route on, and RULES. 3 of 5, plus the safety gate. Matching ladder4 is already the research result: the network drives the whole floor itself.

    **Risks, said before:**
    - If the Q heads' clear values do not correct, the falling β brings the detours back: lead time rises (entry 26: the champion without the route, 120 steps against 75). Watched every episode; the end scale is the knob.
    - Near misses with people: in the clear the network now drives near people too. The safety gate covers it.

    **Plumbing changes from the first version:**
    - **Teacher:** the route's move for every fleet in the clear (the route's eligibility test), recorded as the Learner's own label. With the route off, the label is computed but not executed.
    - **Nudge region:** the clear (eligible fleets), not the conflict set.
    - **The route switch:** a training-time switch flipped by the runner when the warm-up conditions hold. It is logged, with the episode of the switch in the CSV.
    - **`deploy_eval`:** route on or off per arm (for example `ladder5=path:noroute`), and each arm built with or without the Learner according to its own checkpoint.
    - **Tests:**
      - the label is the route's move in the clear only, including with the route off;
      - the nudge reaches only fleets in the clear;
      - a uniform Learner changes nothing; the shadow is silent;
      - the switch fires at the right episode;
      - deployment runs a route-off Learner arm beside a route-on arm;
      - plus the earlier items (Learner-only step, warm-start loading, the separate optimizer).

    **Decisions (revised), for Rohit:**
    - **A. Start from ladder4** (recommended now). Run 9 tests a new way of operating, route off. Starting from the champion keeps its meeting judgement, and entry 39's efficiency prediction (levelling at about 2.5–2.8) becomes testable.
    - **B. Teacher:** the route's move in the clear only.
    - **C. Nudge:** in the clear only, "prob", gap-tied; scale 3.0 → 1.0 after the switch; deployment at 1.0.
    - **D. Hand-over:** route on for at least 5 episodes and until agreement ≥ 98%, then off.
    - **E. Graduation:** ladder5 (route off) against ladder4 (route on), 3 of 5, safety gate.
    - The meeting Learner (sources rules, holds, guard) stays in the code, off.

    **Agreed (Rohit, 14 October): all decisions A–E, with one change to C: the teacher disappears in the end.**
    - **β falls to 0, not to 1.0.** The scale goes from 3.0 to 0 after the switch. The last stretch of training runs at β = 0: pure network, route off, no nudge. That is exactly how it will be deployed and tested, so training ends in the deployed condition.
    - **A longer run:** about 80 episodes, about 6.5 hours at run 8's 4.9 minutes an episode.
      - Episodes 1 to at least 5: route on, Learner in shadow, until it agrees with the route on 98% of clear moves.
      - Then the switch, and the scale from 3.0 to 0, linearly over about 60 episodes.
      - Then about 15 episodes at 0.
      - The exact lengths are set after the dry run.
    - **The dry run** compresses the same shape into about 12 episodes: a warm-up of at least 3, then 3.0 → 0. It checks how fast the Learner reaches 98%, how smooth the switch is, and what happens to lead time as β reaches 0.
    - **Whether to start at 3.0 or 2.0** is read from the dry run: how many of the network's disagreements in the clear the Learner overturns at each scale (`learner_changed`). If 2.0 already overturns nearly all, start there.
    - **The graduation test deploys ladder5 at β = 0, route off:** the network alone. The Learner head is then a training scaffold only.
    - **A longer run also tests entry 39's prediction better:** the efficiency value levelling at about 2.5–2.8, by about 130 episodes in total.

    **Rohit: does the route act after RULES have untangled the warning zone through path awareness?** Not after: beside. Within one step, in the code's order:
    1. **Path awareness and proximity draw the line** between the clear and contention. The warning and deadlock sets hold robots whose routes meet within the conflict radius, a follower too close, or anyone within 1 hop.
    2. **The network proposes** a move for every robot (`choose_actions`).
    3. **The route replaces the proposal for robots in the clear** (`_apply_route_default`).
    4. **The rules and the ladder act on robots in contention** (RULES' orders, holds, retrace, corridor back-outs), in the per-robot action loop.
    5. **The guard checks every intended move together,** and slows or stops (`_run_intent_guard`).
    6. **The robots move.**
    - Steps 3 and 4 touch different robots, so their order does not matter. What matters is that the guard comes last for everyone.
    - After the switch in run 9, step 3 becomes the Learner's nudge inside step 2 (in the clear only); steps 1, 4, 5 and 6 are unchanged.

    **Run 9, step by step** (ticked off as each is done):
    1. [x] **Build (14 October).**
       - **`agent_warehouse.py`:**
         - **Isolated start:** the Learner head is built inside `torch.random.fork_rng`, seeded `_LEARNER_INIT_SEED`. Building it consumes no global draw, so every other layer and every later random draw match a Learner-free agent.
         - **Its own Adam:** the main optimizer covers the other parameters in the same order as a Learner-free network, so ladder4's optimizer state loads exactly. Gradients were already clipped separately (test_learner_head).
         - **`load()`:** a checkpoint without Learner weights loads everything else exactly, and only the Learner's keys may be missing; any other mismatch still fails. A one-optimizer Learner checkpoint (run 4) loads with fresh moments and a message.
         - **`checkpoint_has_learner()`;** and `save()` stores the Learner's optimizer.
         - **The nudge** runs in "gap" mode even at β 0, so gaps and counts keep coming in during shadow. Nothing is added when β is 0.
         - **`choose_actions` publishes** the heads' own top move and the Learner's probabilities for the core's measures, reset every call.
         - **The buffer** gains an optional 16th field, the router's label; `sample()` returns an 18th element. `learn()` teaches from the labels when the batch has them, and otherwise from the teacher mask as before. The meetings path is numerically unchanged.
       - **`core_warehouse.py`:**
         - With `training.learner` enabled and mode "router", `_router_clear_moves` runs before the choice. It is the route's own eligibility test, and returns the clear set, the route's move and the route's ties.
         - The nudge region passed to the choice is the clear set; the label (below) goes on every push.
         - `_router_measure` counts, per clear fleet-step and up to ties: the Learner's agreement with the route, the heads' own disagreement, the chosen move's disagreement, and the share with ties.
         - Stats: `router_on`, `router_route_on`, `router_clear_fleet_steps`, `router_learner_agree`, `router_q_differs`, `router_choice_differs`, `router_tie_share`.
       - **`main_runner_warehouse.py`:**
         - `router_plan` and `router_scale`. Before each episode the route is set on or off and the scale set. After it comes the switch decision, which is logged.
         - A "router" line per episode, and CSV columns `router_*`, `router_scale`, `router_switch_episode`.
       - **`benchmark_flowrra.build_agent` (deploy_eval's builder)** builds each arm with or without the Learner according to its checkpoint. A router Learner is deployed at `scale_end` (0).
       - **`deploy_eval.py`:** an arm written `label=path:noroute` runs with the route off, for that arm only; the setting is restored after the env is built. Rows get `route_env_off` and the `router_*` measures.
       - **`config_warehouse.py`:** `training.learner` enabled, mode "router", the router block (switch_agreement 0.98, warmup_min 3, warmup_frac 0.0625, tail_frac 0.1875, scale 3.0 → 0.0), beta 0.0, beta_mode "gap", nudge "prob". A RUN 9 header with the dry run, the run and the graduation arms.
       - **One improvement found while building: ties.** The route's move is the first of the moves with the best goal gradient. Two directions that each bring the goal a hop closer are equally short, and on this map that is common.
         - So the label is a bit mask of all of the route's equally-short moves (bit a = move a). The Learner learns them with a soft target, uniform over the tied moves, and agreement counts any of them.
         - So the Learner stays neutral among equally short moves, and the heads, which can see traffic, choose between them: the anticipation the router exists for.
         - A strict single-move label would also have taught an arbitrary tie-break by index. That could have kept agreement below 98% and the switch from ever firing.
       - **Tests:** new `test_learner_router.py`.
         - `test_honest_target.py`, `test_route_explore.py` and `test_sarsa.py` pin the Learner off in their env parts, so each keeps testing its own fix.
         - The sample-length checks in `test_honest_target.py`, `test_graph_pooling.py` and `test_teacher_margin.py` now expect 18, with the 18th None when no label is stored.
       - **Checked here without torch, on stand-ins:**
         - **The schedule:** 80 episodes switching at 6 gives 0, 0, 3.0, 2.95, 2.29, 1.53, 0.76, 0.05, then 0 from 65. 12 switching at 4 gives 3.0 at 4, falling by 0.5 a step to 0 at 10.
         - **The padding of a 16-field transition:** the label is padded with −1; 14 and 15 fields are unchanged.
         - **The clear set** equals `_apply_route_default`'s eligibility and moves (6 robots: busy, immobile, no gradient, masked move, a tie).
         - **The tie bits** come out as 0b10100 for moves 2 and 4.
         - **The measures** are counted up to ties: Learner 2 of 3, heads 1 of 3, chosen 1 of 3.
         - **No undefined names** in any edited file (pyflakes).
    2. [x] **Rohit's machine (done 14 October: tests ALL PASS, switch check, smoke test):**
       - **The tests:** `test_learner_router.py`, `test_learner_head.py`, `test_sarsa.py`, `test_honest_target.py`, `test_route_explore.py`, `test_guard_integration.py` and the suite.
         - **First pass, 14 October:** everything passed except two checks in `test_learner_router.py`'s shadow test.
         - Every agent-level check passed, including the trunk and Q heads bit-identical to a Learner-free agent's after 5 updates. So did the route-on and route-off checks in real steps, and the deploy builder.
         - The switch check read `True router gap {...}` as expected.
         - **What failed, and why it was the test:** the shadow test compared two 90-step runs that learned every step. Its own control showed the GPU does not repeat a learning run bit for bit: the Learner-free run twice gave identical moves but weights that differ after learning (atomic adds in the backward pass). Over 90 learning steps, one flipped near-tie is enough to send two identical policies down different days. The run with the Learner in shadow diverged at about step 25 (move agreement 0.70), which measures that chaos, not the Learner.
         - **Reworked into two halves:**
           - (a) Choices: the same 60 steps without learning (forward passes, which do repeat) must give identical moves.
           - (b) Updates: then 20 updates on the now-identical memories, with the same draws. The trunk and Q heads must stay as close to the Learner-free agent as a copy of that agent stays to itself.
       - **Second pass, 14 October:** `test_learner_router.py` ALL PASS with the reworked shadow test.
       - **The switch check:** done, `True router gap {...}`.
       - **A deploy smoke test:** ladder4 with `:noroute`, one instance, with `--check`. It confirms the per-arm route-off plumbing, and gives the first look at ladder4 driving the clear itself.
       - **Smoke test, done 14 October** (`check9.log`, `deploy_check9.csv`: small map, 25 robots, seed 0, a clean day).
         - **The plumbing works.** The log says "route OFF for this arm"; the `--check` repeat is IDENTICAL; no errors. The row reads `route_env_off` True, and the router measures are filled in. `router_learner_agree` reads 0 because ladder4 has no Learner head, as expected.
         - **The first look at ladder4 driving the clear itself,** against the same day in clean8, where the route drove the clear:

           | Same day (small map, 25 robots, seed 0) | ladder4, route off | ladder4, route on | ladder3, route on | RULES |
           | --- | --- | --- | --- | --- |
           | Delivered | 80 of 80 | 80 | 80 | 80 |
           | Book finished at step | 763 of 800 | 626 | 615 | 478 |
           | Lead time, mean / p90 (steps) | 183 / 360 | 65 / 158 | 62 / 183 | 47 / 99 |
           | Guard stops | 697 | 182 | 208 | 108 |
           | Wait-for cycles seen | 218 | 93 | 88 | 12 |
           | Longest cycle (steps) | 118 | 64 | 89 | 70 |
           | Retreats ordered | 110 | 40 | 40 | 9 |
           | L3 fires | 9 | 6 | 4 | 2 |
           | Collisions / guard violations | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
           | Runtime | 147 s | 39 s | 37 s | 18 s |

         - **Clear moves:** 12,238 robot-steps in the clear. The heads' own move differs from the route's on 60% of them, and the chosen move on 60% (the deploy exploration of 0.015 barely shows). The route had tied moves on 6.5%.
         - **Reading.** This is the detour habit in full view: the "before" picture the router run has to change. Lead time nearly triples, and the guard, the ladder and retreats work two to three times as hard, because robots that wander in the clear arrive at meetings that should never have happened. Safety by construction held with the network driving every move: no collision and no guard violation. The day still finished, 37 steps before the end of the shift.
         - **The bar for graduation, on this day:** ladder5 with the route off should land near ladder4 with the route on (lead about 65, guard stops about 180), and the heads' disagreement in the clear should fall well below 60%. Some of it should stay, by design: where the heads see traffic the route cannot, a different move can be the better one. One day is a glimpse, not a result; the frozen tests decide.
         - **The 60% is not the same measure as clean8's 45%.** Clean8 counts disagreement in states the route had kept tidy. Here the network makes its own states, which are more crowded and further from what the route would have produced.
    3. [ ] **Dry run** (`ladder_run5_dry`, 12 episodes, about 1 hour). Read:
       - how fast the Learner reaches 98%, and when the switch fires;
       - lead time, guard stops and L3 before and after the switch, and as the scale reaches 0;
       - the heads' and the chosen moves' disagreement in the clear; the tie share;
       - the values (SARSA) and safety.

       **Dry run 1, stopped after 7 episodes on 14 October: the switch could not fire.** Warm from ladder4; episodes are numbered from 1. The CSV holds episodes 1–5 (the runner saves every 5); 6 and 7 are read from the log. No errors, 0 collisions, 74–78 of 80 delivered every episode, losses normal.

       | Ep | Map | Robots | Learner, live (clear moves) | Learner, training batches | Heads differ |
       | --- | --- | --- | --- | --- | --- |
       | 1 | small | 40 | 63.0% of 4,299 | 85.4% (loss 0.40) | 51% |
       | 2 | large | 40 | 96.3% of 12,628 | 94.8% (0.20) | 44% |
       | 3 | small | 60 | 83.7% of 3,524 | 94.2% (0.20) | 51% |
       | 4 | small | 25 | 81.7% of 5,304 | 92.8% (0.23) | 53% |
       | 5 | small | 40 | 84.0% of 4,546 | 91.7% (0.26) | 48% |
       | 6 | large | 40 | 95.6% of 12,723 | – | 47% |
       | 7 | large | 40 | 97.5% of 13,620 | – | 36% |

       - **The small map levelled off at 82–84%** over three episodes, on days of different difficulty (episodes 4 and 5 had failure waves of 8 and 9 robots).
       - **The large map was still climbing slowly** (96.3%, 95.6%, 97.5%) and might have crossed 98% at episode 8. Stopping was still right: the small map was the binding one, and with fade_end at 10 a switch at 8 would have left a fade of two episodes and one small-map episode with the route off.
       - **It is not the memory mix.** The memory holds 20,000 steps, about 25 episodes, so every episode was still in it. Three small-map episodes in a row (3–5) did not raise live small-map agreement, and agreement on the training batches *fell* (94.2% → 91.7%, loss 0.20 → 0.26) as small-map examples filled the memory. Allowing for the large map's share, the Learner fits only about 87% of the small-map examples it trains on, and gets about 83% on fresh small-map days. On the large map, live agreement matches or beats training. Balancing the maps in its batches would not help.
       - **The most likely cause (a hypothesis, not yet tested): what the trunk passes on.** The Learner reads only the trunk's summary of each robot (`h.detach()`), which is built for the value heads. The route's move comes from six goal-gradient numbers that are in every robot's input, but after the trunk's message passing each robot's summary mixes in its neighbours. The small map has 1,435 free cells across 5 floors against 27,000 across 10 for the large map, so with 25–60 robots, even a robot in the clear has more neighbours in the graph. The trunk was also shaped by heads that disagree with the route on about half of clear moves, so it has no reason to keep the route's ranking sharp.
       - **The heads' disagreement on the large map fell to 36% by episode 7** while still in shadow, so run-8-style training with the route on keeps improving it there. The small map stayed at 48–53%.
       - **98% was set before the Learner's natural level was known.** The data says about 97% on the large map and about 83% on the small one.
       - **Options weighed.**
         - Giving the Learner the six gradient numbers directly would take it near 100%, but it would then be a copy of the route (a pure planner), and the nudge might as well be the route itself. Not taken.
         - A per-map scoreboard for the switch (each map's latest episode must pass) and a measure splitting the nudge's changes into "onto the route's move" and "elsewhere" were proposed. With an 80% bar and this map order, the scoreboard switches at the same episode as today's rule in both the dry run (after episode 3) and the 80-episode run (after episode 5). It matters when more maps are added.
       - **Decided (Rohit, 14 October): keep it simple.** `switch_agreement` 0.98 → **0.80**, with today's rule unchanged; no scoreboard and no new measure. In `config_warehouse.py` the RUN 9 header now says 80% and names the rerun `ladder_run5_dry2`.
         - **The Learner keeps learning after the switch.** The route's move is computed for every robot in the clear on every step, whether or not the route drives (`if self.router:` in the core, with no condition on the route), and goes into memory as the label. Route off only means the route stops driving. After the switch the Learner learns in the states the network makes, which are the ones it will face. The smoke test confirmed it: with the route off, 12,238 clear robot-steps were labelled and measured.
         - **The nudge's effect is read from the existing router line.** After the switch, the gap between "heads alone differ" and "chosen moves differ" is the nudge's net pull towards the route. If chosen moves on the small map fall from about 50% towards the Learner's error rate (about 17%), it is working. If they stay near the heads', or the small map gets worse, it is misleading, and the fade starts at 2.0.
    3b. [x] **Dry run 2** (`ladder_run5_dry2`, 12 episodes, 80% bar; the switch should fire after episode 3, the fade runs from episode 4 (3.0) to episode 10 (0), and the small map has three episodes with the route off: 4, 5 and 9). Read as in step 3.

       **Done 14 October, 71.2 minutes, no errors.** The switch fired after episode 3 (small map, 82.9%), and the route was off from episode 4. The scale went 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, then 0 for episodes 10–12. The small map had four episodes with the route off, not three: 4, 5, 9 and 12 (I had missed episode 12). "Run 8" below is the same episode of run 8: the same orders, map and fleet count, with the route on and no Learner. Run 8 started from ladder3 with its own exploration schedule, and failure draws differ, so it is the same day with the route on, not an exact pair.

       | Ep | Map, robots | Scale | Learner, live | Heads differ → chosen differ | Delivered (run 8) | Lead time (run 8) | Guard stops (run 8) |
       | --- | --- | --- | --- | --- | --- | --- | --- |
       | 1 | small 40 | shadow | 65.3% | 47% → 48% | 74 (75) | 54 (55) | 171 (163) |
       | 2 | large 40 | shadow | 95.7% | 42% → 43% | 75 (79) | 122 (125) | 0 (137) |
       | 3 | small 60 | shadow | 82.9% | 52% → 53% | 78 (78) | 33 (34) | 8 (19) |
       | 4 | small 25 | 3.0 | 71.1% | 63% → 48% | 74 (73) | 165 (71) | 158 (308) |
       | 5 | small 40 | 2.5 | 66.9% | 59% → 48% | 69 (75) | 136 (46) | 782 (2) |
       | 6 | large 40 | 2.0 | 88.9% | 55% → 37% | 67 (75) | 172 (139) | 107 (1) |
       | 7 | large 40 | 1.5 | 92.3% | 49% → 36% | 73 (75) | 193 (145) | 58 (316) |
       | 8 | large 40 | 1.0 | 91.4% | 50% → 41% | 75 (69) | 176 (116) | 244 (3) |
       | 9 | small 60 | 0.5 | 74.0% | 58% → 54% | 77 (80) | 140 (39) | 515 (93) |
       | 10 | large 60 | 0 | 93.9% | 42% → 43% | 76 (73) | 126 (70) | 45 (42) |
       | 11 | large 25 | 0 | 94.3% | 41% → 41% | 46 (73) | 220 (161) | 12 (0) |
       | 12 | small 60 | 0 | 74.3% | 53% → 53% | 78 (79) | 153 (47) | 669 (33) |

       - **The heads are learning from their own moves.** With the route off, their disagreement with the route in the clear fell on both maps. Large map: 55% → 49% → 50% at 40 robots (episodes 6–8), then 42% and 41% (episodes 10 and 11). Small map: 63% → 59% → 58% → 53%; episodes 9 and 12 are both 60 robots, 58% → 53%. These are measured in the network's own states, with the fleet count and the scale changing alongside, so the trend is the evidence, not any single number. The frozen ladder4 differed on 60% in the smoke test (small map, 25 robots).
       - **The Learner keeps learning in the network's states,** as expected. Small map: 71% → 67% → 74% → 74%. Large: 89% → 92% → 91% → 94% → 94%. On its training batches it stays at 85–90%.
       - **The nudge was weak:** it closed 10–18 points of disagreement at scales 1.0–3.0 (small map 63% → 48%; large 55% → 37%), and 4 points at 0.5.
       - **The network alone is still far from the route.** Over the route-off episodes 4–12: mean lead time 165 against run 8's 93 on the same days, guard stops 288 against 89, delivered 70.6 against 74.7, reward 7,674 against 17,436. On the small map deliveries held (77–78 of 80 at 60 robots), but lead time was about 3× and guard stops 5–20× those of the same day with the route.
       - **Episode 11** (large map, 25 robots): 46 of 80 delivered against 73 in run 8. Failures took 10 of 25 robots (9 rescued), but run 8 had the same failure load (11 errors, 8 in waves) on that day. The difference is speed: at the end of the shift, 13 of the 25 robots were still driving to their goals (run 8: 7), and lead time was 220 against 161. On the large map with few robots, every detour is long, and with 10 down there was no slack.
       - **Safety held:** 0 collisions and 0 guard violations in all 12 episodes. Near misses with people over episodes 4–12 averaged 20.6 against run 8's 18.7 on the same days; episode 4 had 92 against 31.
       - **Values stable:** safety −0.10 to −0.12 (as at the warm start), delivery 0.87–1.10, efficiency 1.73–1.91. Losses did not climb, apart from efficiency rising from 0.010 to about 0.014, as expected now that the network's own moves are valued. SARSA labelled 100% of next moves.
       - **Episodes took 5.5–8.6 minutes with the route off** (about 4.5 with it on). The 80-episode run comes to about 8–9 hours, not 6.5.
       - **The bar (Rohit's question: 0.9 or 0.93?).** With the current rule and these two maps, any bar between about 0.85 and 0.95 switches on the first large-map episode after the warm-up (the large map sits at about 96% in shadow, the small at about 83%). In the 80-episode run, 0.8 switches after episode 5 (small map), and 0.9 or 0.93 after episode 6 (large), one episode later. The bar is not a lever on the Learner's readiness: its small-map level does not rise in shadow, and it rises after the switch, in the network's own states.
       - **The efficiency value after the switch (Rohit's question).** In run 8 the value climbed because it was filling towards a higher level that the route's efficient driving set (entry 39). With the route off, the network drives less efficiently, so the rewards now set a lower level. The same rough check, rebuilt from the CSV (the efficiency reward per robot-step on the floor, times the discounted steps left in a robot's life, γ 0.99):

         | | Reward rate per robot-step | Life (steps) | Rough level | Learned value |
         | --- | --- | --- | --- | --- |
         | Small map, route on (run 8, same days; dry run 2 episodes 1 and 3) | 0.026–0.036 | 42–78 | 0.5–1.0 | – |
         | Small map, route off (episodes 4, 5, 9, 12) | 0.002–0.010 | 157–201 | 0.1–0.5 | 1.73–1.91 |
         | Large map, route on (run 8, same days; dry run 2 episode 2) | 0.039–0.058 | 96–168 | 1.5–2.8 | – |
         | Large map, route off (episodes 6–8, 10, 11) | 0.011–0.022 | 191–245 | 0.65–1.4 | 1.75–1.88 |

         - The rate fell to about a third (small map) or under half (large), and lives grew 1.5–4× longer. The learned value only dipped (about 2.0 → 1.75–1.9), so it now sits **above** the level the rewards set: the opposite of run 8, where it sat below and climbed.
         - The memory holds about 25 episodes, so in the 80-episode run it fills with route-off experience, and an honest value should come down towards the new level. Then, if the network gets faster, the level rises again and the value should follow it up.
         - Caveat: this rebuild counts every robot on the floor in the rate, so its absolute levels come out lower than entry 39's (run 8 by this method: about 0.7 small, 2.25 large, against 1.3 and 3.2 there). Only comparisons made with one method mean anything; the drop between route on and route off does not depend on the denominator.
       - **The bar: kept at 0.8 (Rohit, 14 October).** One more episode with the network driving rather than one more in shadow; the Learner improves after the switch, not before. In the 80-episode run the warm-up minimum is 5 episodes, so the switch comes after episode 5 at the earliest (small map, about 84% in both dry runs), and the route is off from episode 6: 75 episodes with the network driving, the fade to episode 65, and 15 at 0.
       - **Rohit's idea: ramp the scale up, then fade it** (for example 1 → 3 → 0, like the exploration Gaussian), because the Learner is weakest right after the switch and is still learning. Sound reasoning: trust the teacher more as it gets better. **Not for this run:**
         - **Even at its weakest, the nudge pulled towards the route, never away from it.** Episode 4 (Learner 71% on the small map): heads 63% → chosen 48%. Episode 5 (67%): 59% → 48%. A 70% Learner is still right far more often than the heads alone (wrong on 55–63%), so a strong nudge at the start helps on average. A ramp would fix a problem the dry run did not show.
         - **The early episodes are where the network is worst** (episode 5: 19 recoveries, reward −5,012), so that is where the help matters most.
         - **The effect would be small in an 80-episode run.** The Learner adapted to the network's states within 3–5 episodes per map, while the fade falls by only about 0.05 per episode, so a ramp would change roughly the first 5–10 of 75 episodes.
         - **It would need code, a test and another dry run.**
         - **Kept as a candidate.** If the first route-off episodes show the nudge misleading (chosen-move disagreement above the heads', or near misses jumping), try it in the next run.
    4. [ ] Predictions; then the run (`ladder_run5`, 80 episodes, **about 8–9 hours**: episodes with the route off take 5.5–8.6 minutes).

       **Predictions, agreed by Rohit on 14 October** (item 6 added for his question about the efficiency value). "Last 15" is episodes 66–80, the network alone. "First 10" is the first 10 episodes with the route off.

       | # | Prediction | Measure | Fails if |
       | --- | --- | --- | --- |
       | 1 | The heads learn the clear | heads-alone disagreement in the clear (`router_q_differs`), last 15, per map | above 35% on the large map or 45% on the small (the dry run ended at about 41% and 53%) |
       | 2 | The network alone gets faster | mean lead time, last 15 against first 10, per map | less than 25% lower on either map |
       | 3 | Safety holds | collisions, guard violations, near misses with people | any collision or violation, or near misses above about 21 per episode on average |
       | 4 | The safety value stays steady | `qval_safety`; all losses | outside −0.15 to −0.05, or any loss climbing over the run |
       | 5 | The Learner improves in the network's own states | `router_learner_agree` over the last 10 episodes of the fade | below 80% on the small map or 95% on the large |
       | 6 | The efficiency value stays honest | `qval_efficiency` against the efficiency reward rate | it keeps climbing while the reward rate stays low (drift); expected instead: it falls in the early route-off episodes, then moves with the reward rate |
       **Started 14 October, 03:53 IST. Check at 20 episodes** (CSV episodes 1–20; run 8's same episode in brackets: the same orders, map and fleet count, route on, its own exploration schedule).
       - **The switch fired after episode 5** (small map, 81.1%); the route has been off from episode 6, with the scale falling 3.00 → 2.29 by episode 20. No errors, 0 collisions, 0 guard violations. Exploration is at its peak (0.050 at episode 20) and falls from here. Episodes with the route off take about 6.7 minutes, so the run should end around 12:30–13:00 IST.
       - **The two maps tell different stories.**

         | Large map, route off | Ep 6 | 7 | 8 | 10 | 11 | 17 | 18 | 20 |
         | --- | --- | --- | --- | --- | --- | --- | --- | --- |
         | Robots | 40 | 40 | 40 | 60 | 25 | 40 | 60 | 40 |
         | Heads differ | 55% | 53% | 45% | 49% | 43% | 40% | 38% | **36%** |
         | Chosen differ | 32% | 31% | 27% | 30% | 27% | 25% | 25% | 23% |
         | Learner, live | 87% | 90% | 91% | 92% | 93% | 92% | 94% | 93% |
         | Lead time (run 8) | 181 (139) | 182 (145) | 181 (116) | 99 (70) | 186 (161) | 160 (147) | 108 (88) | 153 (139) |
         | Delivered (run 8) | 69 (75) | 71 (75) | 72 (69) | 76 (73) | 66 (73) | 75 (75) | 78 (78) | 77 (72) |

         | Small map, route off | Ep 9 | 12 | 13 | 14 | 15 | 16 | 19 |
         | --- | --- | --- | --- | --- | --- | --- | --- |
         | Robots | 60 | 60 | 60 | 40 | 40 | 60 | 25 |
         | Heads differ | 58% | 57% | 58% | 53% | 57% | 57% | 55% |
         | Chosen differ | 47% | 47% | 46% | 45% | 47% | 47% | 45% |
         | Learner, live | 70% | 72% | 72% | 70% | 71% | 72% | 75% |
         | Lead time (run 8) | 133 (39) | 124 (47) | 103 (35) | 176 (54) | 176 (78) | 108 (36) | 197 (62) |
         | Guard stops (run 8) | 554 (93) | 313 (33) | 532 (2) | 451 (58) | 572 (176) | 405 (1) | 138 (171) |

         - **Large map: the heads are learning fast.** At 40 robots their disagreement with the route fell 55% → 36% (episodes 6 → 20), already at prediction 1's line. By episodes 17–20, lead time is within about 10% of the same day with the route (160 against 147, 153 against 139; 108 against 88 at 60 robots), and deliveries match or beat it (75, 78, 77 against 75, 78, 72). The first clear evidence that handing the clear to the network can work.
         - **Small map: flat.** Heads 53–58%, the Learner about 71%, lead time about 3× the route's, guard stops several times higher. No trend yet over 7 episodes.
         - **Possible reasons, not yet separable:**
           - exploration rising to its peak (0.033 → 0.050) through these episodes; random moves cost more on a crowded floor, and the states get messier;
           - harder credit: on the large map a detour costs the robot its own distance (a clean efficiency signal), while on the small map it mostly costs others, in crowding and guard stops;
           - the weaker nudge (Learner about 71%);
           - and part of the disagreement may be legitimate: under the route, the heads already differed from it on about 45–51% of small-map clear moves (episodes 1–5 in shadow), so the small map's natural level may sit near prediction 1's 45%.
       - **Values (Rohit: "watch the q values").** All moving the way honest values should. The losses are flat (safety 0.0011–0.0016, delivery 0.009–0.011, efficiency 0.013–0.016).
         - **Safety:** −0.10 → −0.13. The safety charges per robot-step on the small map are 2–3× those under the route (rough level −0.11 to −0.34 against −0.03 to −0.09), from the extra close approaches, so a more negative value is honest. It is inside prediction 4's band. If the small map stays rough it may approach −0.15, and that would be the value following real charges, not instability; to be judged together with the safety rewards when scoring.
         - **Efficiency:** 2.07 → 1.69, falling as prediction 6 expected. On the large map the efficiency reward rate is recovering (0.026–0.028 per robot-step in episodes 6–8, 0.031–0.036 in 17–20; under the route 0.045–0.056). On the small map it stays low (0.006–0.011).
         - **Delivery:** 0.78–0.98, no trend.
       - **Near misses with people,** episodes 6–20: 16.6 per episode, under prediction 3's 21.

       **Check at 55 episodes** (CSV episodes 1–55, saved at 55; Rohit was on a hotspot with a laptop low on battery, so this is a safety copy). The scale was 0.51 at episode 55, exploration at its floor (0.020) from about episode 40. 0 collisions and 0 guard violations in all 55 episodes.

       | Period (route off) | Map | Episodes | Heads differ | Lead time | Guard stops | L3 | Near misses | Delivered |
       | --- | --- | --- | --- | --- | --- | --- | --- | --- |
       | 6–25 | large | 10 | 42.9% | 151 | 156 | 2.1 | 7.4 | 73.6 |
       | 26–40 | large | 7 | 35.8% | 154 | 15 | 0.4 | 2.6 | 68.6 |
       | 41–55 | large | 8 | **33.1%** | 122 | 42 | 1.5 | 18.9 | 75.8 |
       | 6–25 | small | 10 | 56.5% | 147 | 444 | 5.1 | 43.5 | 74.0 |
       | 26–40 | small | 8 | 56.5% | 147 | 506 | 5.0 | 44.9 | 75.1 |
       | 41–55 | small | 7 | 55.6% | 143 | **659** | 6.9 | 116.7 | 73.0 |

       - **Large map: learned, then levelled off at about 31–35%** from about episode 24. Against the same days in run 8 (route on), lead time runs about 15–20% behind (for example 108 against 93, 118 against 100, 121 against 106 at 60 robots in episodes 43–45). Earlier it was about 30% behind, and the nudge has faded from 2.8 to 0.5 meanwhile, so the network now carries more of the job itself. Deliveries are on par (episodes 32 and 39, 25 robots, were bad days: 42 and 63).
       - **Small map: no learning in 50 episodes.** Heads 53–62%, flat, including episodes 40–55 at the exploration floor, so exploration was not the cause. The Learner sits at 70–77%. Lead time is flat at about 3× the route's (60 robots: 96–126 against 36–51). As the nudge fades, the chosen moves drift further from the route (45–47% → 51–55%), and guard stops rise (444 → 506 → 659 per episode). Near misses rose too, mostly from one episode (55: 566; episodes 42–53 averaged about 42).
       - **Near misses overall,** episodes 6–55: mean 37.1, median 14. Run 8, episodes 6–50: mean 84.2, median 10. Both are dominated by a few spike days. Prediction 3's line (about 21 per episode, from the dry run's 12 episodes) is above that, so it is failing on the mean.
       - **Values:**
         - **Safety:** −0.128 → −0.151 (episodes 49–50) → −0.145. At prediction 4's edge, as foreseen at 20 episodes. On the small map the safety charges per robot-step stay about 5–10× the large map's (−0.003 to −0.004 against −0.0003 to −0.0007).
         - **Efficiency:** falling towards the level the rewards set, from above, as prediction 6 expected. Learned 1.55 at episode 30 against a memory-weighted rough level of 1.06; 1.25 at 55 against 1.10. On the large map the efficiency reward rate is recovering (0.030 → 0.034 per robot-step); on the small map it fell (0.008 → 0.0045).
         - **Delivery:** flat, 0.88–0.98.
         - **Losses:** delivery 0.0104 → 0.0085 and safety about 0.0012, both flat or falling. Efficiency creeping up: 0.0141, 0.0135, 0.0133, 0.0145, 0.0156 over 10-episode windows. To watch for prediction 4.
       - **Reading so far.** The handover works where a detour's cost lands on the robot itself (the large map) and does not where it lands on others through crowding (the small map). The last 15 episodes, at scale 0, will show how the small map holds up with no help at all.
       - **Episodes 56–58 (from the log):** 56, small map, 25 robots: 75 delivered, heads 58%. 57, large, 60 robots: **80 of 80, the first full book with the network driving the clear**, heads 31%. 58, large, 60 robots: 75 delivered, heads **28%**, the large map's lowest yet (the Learner there at 96%).
       - **The small map: what next (Rohit, 14 October).**
         - **Two ideas weighed and held:**
           - *A small-map-only run.* The small map had about 25 route-off episodes here with no trend, so more of the same signal extends a flat line. One network serves both maps, so training on one alone risks losing the large map's gains. A small-map-heavy mix may make sense once the fix is known.
           - *Keeping the nudge above 0.* This run already shows it would not fix the map: lead time stayed at 147, 147 and 143 while the scale fell from 2.9 to 0.5. Only guard stops moved (444 → 659), and the Learner itself sits at about 74% there.
         - **Agreed order:** finish the run; score the six predictions; the graduation frozen tests (which also give ladder5 against ladder4 with the route, map by map); a diagnostic; then the fix and a new run.
         - **Candidate causes,** each fitting the two-map split:
           1. *A blurry picture.* The trunk mixes each robot's summary with its neighbours', and on a crowded map the robot's own goal signal is diluted. The Learner's ceiling (about 74% small, 96% large) hints at it. Fix: feed each robot's own inputs straight to the heads beside the trunk's summary, with the new weights starting at zero, so the warm start is exact.
           2. *The wrong lesson (credit).* On the small map a detour's cost lands mostly on others (crowding, guard stops), which the detouring robot's own reward barely sees. Fix: charge a robot for the waiting it causes others.
           3. *Blind value targets* (Rohit's question about edge features). The 4 edge features (closeness, conflict, head-on, arrival order) are used when choosing, but `gnn.next_state_edge_features` is off. So the stored next state, from which SARSA's value target is computed, keeps only the bare neighbour mask: when the network learns what follows a move, it cannot see the pairwise picture of that next state. On the small map nearly every consequence is about neighbours; on the large map robots in the clear have few, so it matters little. Fix: a config switch, with no new weights and an exact warm start, at the cost of more replay memory (edge features stored twice). Tested alone in entry 15: fewer collisions, L3 fires and retreats and the highest RULES agreement, though the per-episode ranges overlapped. Pair relations stay off (the likely cause of harm in the earlier combined test).
         - **How to tell them apart.** The diagnostic (frozen ladder5, mistakes in the clear split by neighbour count on each map) tests cause 1. Causes 2 and 3 act through learning, so they need a paired dry run against dry run 2: same seed, small-map-heavy, one change at a time.
    4b. [x] **Run 9 finished, 14 October, 12:52 IST: 80 episodes in 539 minutes, no errors.** 0 collisions and 0 guard violations in all 80 episodes.

       **Scored against the six predictions** ("last 15" = episodes 66–80, scale 0; "first 10" = episodes 6–15, the first 10 with the route off):

       | # | Prediction | Result | |
       | --- | --- | --- | --- |
       | 1 | Heads differ, last 15: large ≤ 35%, small ≤ 45% | large **24.4%**, small 50.7% | ❌ (large ✅, small ❌) |
       | 2 | Lead time, last 15 ≥ 25% below first 10, per map | large −22.1% (164 → 129), small **+21.7%** (143 → 174) | ❌ |
       | 3 | No collision or violation; near misses ≤ about 21 per episode | 0 and 0; near misses mean 41.4 (median 12) over episodes 6–80, 23.0 in the last 15 | ❌ on near misses (run 8: mean 84, median 10) |
       | 4 | Safety value within −0.15 to −0.05; no loss climbing | safety value −0.169 at its lowest, −0.166 at the end; efficiency loss 0.014 → 0.017, safety loss 0.0011 → 0.0017 | ❌ |
       | 5 | Learner over episodes 56–65: small ≥ 80%, large ≥ 95% | small 74.9%, large 95.2% | ❌ (large ✅, small ❌) |
       | 6 | Efficiency value honest: falls, then moves with the rewards | fell from 2.07 to 1.04 and met the rough level the rewards set (1.05 at episode 80; it was 1.55 against 1.06 at episode 30) | ✅ |

       **One of six passed cleanly; on the large map, 1, 5 and 6 passed.** The run did half of what it set out to do.

       - **Large map: the network learned the clear.** Heads 55% → 24% (last 15). With no help at all (scale 0), it held the level it had with the nudge: lead time about 113 at 60 robots and 146 at 40, the same as in episodes 41–55. It is about 15–20% behind the route on the same days, and deliveries are on par. Episode 57 was the first full book with the network driving (80 of 80, 60 robots); episode 73 delivered 79. Prediction 2 missed by 3 points on this map, and partly from the fleet mix: at 40 robots, 181 → 146 (−19%); at 60, about 105 → 113.
       - **Small map: the last bit of nudge was holding it together (Rohit's reading).** The heads' disagreement improved only once the nudge was gone (about 56% → 51%), presumably because their own moves now all ran and were corrected. But the floor got rougher. At 60 robots, the like-for-like count:

         | Small map, 60 robots | Scale ≥ 0.66 (10 episodes) | Scale ≤ 0.2 (episodes 61, 68, 78) |
         | --- | --- | --- |
         | Lead time | 115 | 155 |
         | Guard stops | 468 | 964 |
         | L3 | 6.4 | 10 |

         At 25 robots, lead time 167 → 185 at scale 0. So a small nudge (0.5 and above) mattered on the crowded map, while the large map did not need it. Earlier (episodes 6–55) the strength between 2.9 and 0.5 barely changed lead time; it is the step to zero that hurt. Even with the nudge, the small map stayed at about 3× the route's lead time.
       - **The safety value drifted.** It kept falling after episode 40 (−0.137 → −0.166) while the safety charges per robot-step stayed flat on both maps (small about −0.0039, large about −0.0005). By the rough method, its level sits near −0.11, so by the end the value was about 0.05 more pessimistic than the charges explain, and the gap was widening. It errs on the cautious side, but it is what prediction 4 was there to catch. A possible link to cause 3 above (blind value targets): the safety head's targets value the next state without its edge features (closeness, conflict), so they cannot see the danger they are estimating.
       - **Near misses** are spike-driven: a few days carry most of the mean (episode 55: 566, 64: 558, 61: 210, 21: 190), and the median is 12. Below run 8's mean of 84, above the line.

       **Next (Rohit, 14 October):**
       - **Graduation frozen tests,** clean and full, as agreed.
       - **For the next run (Rohit):** keep a floor on the scale (`scale_end` about 0.5), from the small map's evidence above. And test `next_state_edge_features` in a dry run, which he expects to be a must.
       - **Planning notes:**
         - The clean test can carry an extra arm, ladder5 at scale 0.5 with the route off, to measure the floor frozen before committing to it.
         - The full test was to come after the order-leak fix (carried item 3, entry 41).
    5. [ ] Score it; the graduation frozen tests: ladder5 `:noroute` against ladder4 and RULES, clean and full, 3 of 5 plus the safety gate.
       **Running since 14 October, 13:30 IST** (four processes side by side; the machine has 16 cores):
       - `deploy_eval_clean9.csv` (seeds 0–2, failures and people off) and `deploy_eval_full9.csv` (seeds 0–4, both on): ladder5 `:noroute`, ladder4 and RULES in the same invocation, `--safety on`. Rohit's call: same code version and same days, so no replay check is needed. Comparing these ladder4 and RULES rows with clean8 and full8 also checks, for free, that the router build left them unchanged.
       - `deploy_eval_clean9_floor.csv` and `deploy_eval_full9_floor.csv`: ladder5 alone, labelled `ladder5f`, with `router.scale_end` 0.5 (the floor). The config was changed after the first two runs had built their agents: deploy_eval reads the scale once, at start-up (`[deploy] arm` lines, two per run, since RULES builds no network). Same seeded days, so these compare directly with the rows above.
       - **The order leak stays in this full test,** as in full7 and full8: failures, debris and people draw from the shared random, so arms meet different events on the same seeded day once their choices differ. Its fix goes into run 10's build, so that run 10's graduation is like-for-like.
       - The config now has `scale_end` 0.5 on Rohit's machine (and 0.0 in mine), pending the floor arm's result.

       **Scored 14 October: ladder5 does not graduate; ladder4 stays the champion.** No collisions in any run of any arm. Rohit also ran a 15-robot clean test (`deploy_eval_clean9_k15.csv`, 2 maps × 3 seeds, all three arms), to check whether the policy had overfitted to the trained fleet counts (25, 40, 60).
       - **Replay check, free:** the ladder4 and RULES rows match clean8 and full8 exactly, 36 of 36 and 60 of 60 (steps run, delivered, lead time, guard stops, L3, collisions; in full also near misses and failures). The router build left them unchanged.
       - **Overall, ladder5 (route off) against ladder4 (route on), paired:**

         | | Clean, 18 days: ladder5 / ladder4 (p) | Full, 30 days: ladder5 / ladder4 (p) |
         | --- | --- | --- |
         | Book filled | 0.979 / 0.999 (0.066) | 0.923 / 0.955 (0.000) |
         | Lead time | 140 / 76 (0.000) | 150 / 86 (0.000) |
         | L3 | 8.6 / 1.9 (0.085) | 4.5 / 1.2 (0.000) |
         | Guard stops | 581 / 78 (0.004) | 441 / 87 (0.000) |
         | Near misses | – | 54 / 14 (0.179) |

         Worse on every measure in both conditions, so it fails 3 of 5 and the safety gate's spirit (near misses about 4×, not firm at p 0.18). Against RULES the same picture.
       - **By map (means, medians in brackets):**

         | | Large map, clean | Large map, full | Small map, clean | Small map, full |
         | --- | --- | --- | --- | --- |
         | Lead time: ladder5 / ladder5f / ladder4 / RULES | 123 / 120 / 105 / 105 | 141 / 132 / 119 / 119 | 157 / 148 / 47 / 39 | 159 / 156 / 53 / 46 |
         | Delivered: ladder5 / ladder4 | 79.4 / 80.0 | 74.2 / 75.7 | 77.2 / 79.8 | 73.5 / 77.0 |
         | Guard stops: ladder5 / ladder4 | 46 (5) / 18 (0) | 52 (30) / 46 (18) | 1,116 / 137 | 830 / 128 |
         | L3: ladder5 / ladder4 | 0.8 / 0.8 | 1.1 / 0.7 | 16.3 / 3.1 | 7.8 / 1.7 |
         | Near misses: ladder5 / ladder4 | – | 6.5 (0) / 9.4 (3) | – | 102 (28) / 19 (8) |

         - **Large map, the network alone: about 16–18% slower than the route, deliveries within 1–1.5, the same L3, and fewer near misses with people (6.5 against 9.4).** A policy driving every move in the clear with no planner, close to the planner on the large map: a real result (Rohit: "just a policy to be there is quite something"), and the first time in v2.
         - **Small map: not learned.** About 3× the route's lead time, 6–8× the guard stops, 5× the near misses (medians 28 against 8).
       - **The floor (ladder5f, scale 0.5), paired against ladder5 on the same days:** lead time slightly lower (clean −9 steps small map and −3 large; full −3 small and −9 large, lower on 10 of 15 large-map days), L3 mixed (full small map +3.8, higher on 12 of 15), near misses about the same. A small help, not a fix, and smaller than the training episodes suggested (entry 42, 4b).
       - **The 15-robot test (Rohit's overfit check), ladder5 against ladder4, clean:**

         | Robots | Large map: lead ratio | Large map: heads differ | Small map: lead ratio | Small map: heads differ |
         | --- | --- | --- | --- | --- |
         | 15 (not trained) | **1.46** (189 / 129) | 26% | 2.89 | 56% |
         | 25 | 1.05 | 24% | 2.54 | 51% |
         | 40 | 1.21 | 25% | 4.04 | 51% |
         | 60 | 1.28 | 25% | 3.90 | 52% |

         On the large map at 15 robots the gap widens to 46%, but the heads' disagreement is the same as at the trained counts (26%). So the policy does not behave differently on the untrained count; the same habit costs more. With fewer robots nearly the whole trip is in the clear, driven by the network, and trips are longer, while at 60 more of each trip is at meetings, which the rules and the ladder decide. Mild evidence for overfitting at most; 3 days per map. The cheap guard against it is adding 15 to the training fleet counts.
       - **Rohit's reading (14 October): ladder4 is undefeated.** He doubts ladder5 as the start of run 10 ("there might be a few unlearning steps involved").
         - *For ladder5:* on the small map without the route it is no worse than ladder4 (smoke test: ladder4 alone 183 lead at 25 robots on the seed-0 day, ladder5 159 on the same day here), and it is far better on the large map.
         - *Against it:* the drifted safety value (−0.166) and its route-off value levels come with it.
         - The dry runs can settle it with data (entry 43).


43. **Draft for run 10 (`ladder_run6`): the small map (14 October). For discussion once the frozen tests are scored; no code yet.**

    **Where run 9 left us.** On the large map the network drives the clear itself, about 15–20% behind the route, with no help. On the small map it does not: the heads still differ from the route on about half of clear moves, lead time is about 3× the route's, and removing the last of the nudge hurt (entry 42, 4b). The safety value drifted pessimistic while its charges stayed flat. Three candidate causes (entry 42, the 55-episode check): a blurry picture (cause 1), the wrong lesson about crowding (cause 2), blind value targets (cause 3).

    **Proposed shape of run 10:**
    - **Warm from ladder5.** Its large-map driving and its Learner are already trained.
    - **Route off from the first episode, no warm-up.** The network has already driven 75 episodes without the route; a warm-up would refill the memory with route-made states. Needs a small router option (for example `start_off`), since today's runner always starts with a warm-up. The dry runs need it too.
    - **Fleet counts:** consider adding 15 to the training sets (15, 25, 40, 60), after the 15-robot test (entry 42, step 5).
    - **The scale held at a floor, no fade.** In run 9 the strength between 2.9 and 0.5 barely changed the small map, while the step to 0 hurt it. So a constant 0.5, also used at deployment. Pending the floor arm (entry 42, step 5): if ladder5 at 0.5 is no better than at 0 frozen, keep 0.
    - **`next_state_edge_features` on (cause 3),** if the paired dry run below supports it. It may also explain the safety value's drift.
    - **The order-leak fix,** for run 10's graduation to be like-for-like: failure, debris and people draws seeded from the episode, the step and the robot (or the debris spot). Every arm then meets the same events on the same day, as far as its own robots allow.
    - **Map mix:** both maps as now. Weighting towards the small map only once a fix is known to work.

    **Before the run:**
    1. **The diagnostic for cause 1** (frozen, minutes). A new small script replays a few days with ladder5 on each map. For every robot-step in the clear it records the robot's neighbours in the network's graph, the distance to the nearest robot, and whether the Learner and the heads matched the route. If mistakes rise with neighbour count on *both* maps, the picture is blurry, and the fix is a direct path from each robot's own inputs to the heads, with its new weights starting at zero so the warm start stays exact. If the small map is worse at the same neighbour count, it is not the picture.
    2. **Paired dry runs for cause 3,** run side by side: 12 episodes each, the same seed, route off from the start, small map only (the question is the small map, and 12 episodes of it give a clear signal). A: from ladder5, edges off. B: from ladder5, edges on. **C (added after the frozen tests, for Rohit's doubt about starting from ladder5): from ladder4, edges off.** A against B isolates the edges; A against C isolates the starting point. Every episode is a pair (same orders, fleets and seed). Scale 0 in all three: ladder4 has no trained Learner, so a nudge would not be the same across arms, and the floor helped only a little frozen. Read: the heads' disagreement, lead time, guard stops, L3, and the safety value against its charges. A decision rule to agree before they run.
    3. **Cause 2** (charging a robot for the waiting it causes others) only if causes 1 and 3 do not account for the small map. It changes the reward, so it is the biggest step.

    **Checks for the build:**
    - **Next-state edges were last used in run 4** (learner_run2), before SARSA and the router. The core path is still there (`core_warehouse.py`, the next state gets `_build_edge_features` when the switch is on), and the buffer pads 3-D adjacency. A short test should confirm the 18-element sample with a 3-D next adjacency.
    - **Memory:** each stored next state grows from the bare mask (N × N) to N × N × 5, about 58 KB more per transition at 60 robots, so up to about 1.2 GB for the 20,000-transition memory. Check free memory first.

    **Revised 14 October, after the frozen tests (Rohit, and a defect found while building).**
    - **Rohit's arms:** C starts from ladder4 *with* the next-state fix, since the question is whether run 10 should start there. So A (ladder5, old storage) against B (ladder5, fixed) isolates the fix, and B against C (ladder4, fixed) isolates the starting point.
    - **Rohit's schedule, "get aggressive":** the scale falls from 5 to 1 instead of 3 to 0, the same for all three arms, so it does not confound the comparisons. All three use the runner's usual warm-up (route on until the Learner reaches 80% on the small map, at least 3 episodes): ladder4 has no Learner, so it needs the warm-up to train one, and the `start_off` router option is not needed after all. A trade-off to watch: a stronger nudge means fewer of the heads' own moves run and get corrected, and in run 9 the small map's heads improved only once the nudge was gone.
    - **A second defect in the stored next state, found while checking the edge path: its attention graph is built on the wrong distance.**
      - The choice in s builds its graph on GRAPH distance: neighbours within 10 hops along the floor, immobile robots excluded (`_build_adjacency_graph`, the phantom-pair fix of 13 September).
      - The stored s', from which every value target is computed, still used the MANHATTAN builder (`_build_adjacency`): straight-line distance through racks and floors, all robots included. Robots behind a rack, or on the floor above, were neighbours in s' but not in s. That fix had called itself "the fourth and last instance of the phantom-pair defect"; this is a fifth, in storage.
      - It is in every run so far. It cannot touch frozen tests, which do not learn.
      - Like the missing edge features, it should matter most where robots crowd and floors stack close: the small map (5 floors in 12 units of height, 1,435 cells).
    - **So arm B and C's change is "make the stored next state the state the choice saw":** both the choice's graph and its edge features. Both correct the same defect, so they are tested together and adopted together if they help.

    **Built (14 October):**
    - **`core_warehouse.py`:** `gnn.next_state_adjacency`, "manhattan" (default: exactly the old storage) or "choice" (the choice's builder, with the proximity index refreshed on the post-move floor first; idempotent with the post-move refresh, and the next step refreshes again). An unknown value fails at start-up.
    - **`config_warehouse.py`:** the new switch, defaulting to "manhattan".
    - **`main_runner_warehouse.py`:**
      - The `edges` start-up line reports both next-state switches.
      - A new `--set KEY.PATH=VALUE` option (repeatable) overrides a config value for one run, so runs side by side can differ without editing the config file between launches. The key must exist, so a typo fails at start-up; the value is read as a Python literal (True, 5.0) or kept as text. Each override is logged with the file's value.
    - **`test_next_state.py`** (new), on real FLOWRRA steps:
      1. With the old storage, the stored next adjacency is 2-D and equals the Manhattan builder.
      2. With "choice", it equals the graph builder.
      3. With both on, it is 3-D with 1 + E channels, channel 0 is the graph builder, and the edge channels are not all zero.
      4. Learning with both on: `sample()` gives 18 elements and a 4-D next adjacency with 1 + E channels, and every loss is finite.
      5. For information: how often the two builders' neighbour sets differ, and the memory added.
    - **`diag_clear.py`** (new), the diagnostic for cause 1: a frozen replay, route off. Per robot in the clear, it records neighbours in the choice's graph and under Manhattan, the graph distance to the nearest robot, the route's tie count, and whether the Learner and the heads matched the route; per step, the share of robots whose two neighbour sets differ. It prints agreement by neighbour count and by nearest distance per map, plus the size of the next-state mismatch per map. Clean conditions as deploy_eval.
    - Checked here without torch: every edited file parses, no undefined names (pyflakes), and the `--set` parsing on a stand-in (nested keys, literals, text, a typo caught).
    - **Rohit's machine, 14 October:** `test_next_state.py` and `test_learner_router.py` both ALL PASS.
    - **Rohit's question: is everything in hops now?** Checked every distance in the decision code. Every decision is in hops:
      - the attention graph at choice, and with the fix the stored next state;
      - the warning set and collision sweep (graph proximity index);
      - the guard (hop metric);
      - the opening-or-closing tests (hops, `hops_everywhere`);
      - L3's goal distance (BFS);
      - the edge features' closeness (graph distance).
      Conflict, head-on and arrival order come from the routes along the graph. Directions are unit vectors of the last move, compared by dot product, which is right in space.
      Straight-line distance remains only where it is harmless: on-the-pickup-cell checks (half a hop, where the two agree), measurement-only stats (`sep_*`, the phantom-pair counter), and the fallback for a position off the graph.

    **The diagnostic, done 14 October** (`diag_small.log`, `diag_large.log`; ladder5 frozen, route off, clean, seed 0, 25/40/60 robots; 32,593 clear robot-steps on the small map, 42,925 on the large). "Neighbours" = robots within 10 hops in the choice's attention graph. The robot itself is in the clear by the route's test, so none is close; these are the ones its picture is mixed with.

    | Neighbours | Small map: share / Learner / heads | Large map: share / Learner / heads |
    | --- | --- | --- |
    | 0 | 8.7% / **97.5%** / 61.8% | 77.7% / **99.8%** / 80.0% |
    | 1 | 8.0% / 89.8% / 49.8% | 18.7% / 83.2% / 63.3% |
    | 2 | 8.6% / 80.6% / 45.8% | 3.2% / 78.6% / 57.7% |
    | 3–4 | 17.3% / 77.6% / 45.4% | 0.4% / 85.5% / 45.2% (166 robot-steps) |
    | 5–7 | 22.9% / 71.2% / 45.8% | – |
    | 8+ | 34.6% / 67.3% / 43.9% | – |
    | All | 75.5% / 46.8% | 96.0% / 76.0% |

    (Learner and heads: share of clear moves on one of the route's equally short moves.) By the graph distance to the nearest robot the picture is the same: the Learner falls from 87.9% at 8–10 hops to 68.0% within 4 on the small map, and from 86.1% to 77.2% on the large.

    - **Cause 1, the blurry picture, is real, and on both maps.**
      - A robot alone reads the route almost perfectly on either map: the Learner at 97.5% (small) and 99.8% (large).
      - With neighbours it falls on both: to about 80% with 1–2 on the large map, to 67% with 8+ on the small.
      - So the trunk's summary of a robot loses its own route direction as neighbours are mixed in.
      - The small map's lower total is mostly the mix: 91% of its clear robot-steps have neighbours within 10 hops (35% have 8 or more), against 22% on the large map. Its robots average 7.3 neighbours in the graph; the large map's 0.2.
    - **But the heads have a second, small-map problem that the picture does not explain.** Alone, with a clean picture, the heads follow the route on 61.8% of moves on the small map against 80.0% on the large. The Learner shows the information is there (97.5%), so it is the heads' *values* that are off on the small map: the learning-target side. That is where the next-state defect and the crowding cost (cause 2) act.
    - **The next-state defect is about four times larger on the small map.**
      - Small map: 28.4% of robots had a different neighbour set in the stored s' than at the choice (on average +0.52 phantom neighbours, 7.85 against 7.33). Among robots in the clear, **39.3%**.
      - Large map: 9.3% of robots, +0.10.
      - Part of the difference is immobile robots, which the choice's graph excludes and the Manhattan builder kept.
    - **Dry runs A, B and C, done 14 October** (small map only, 12 episodes, seed 0, same scenarios in the same order; warm-up with the route on, switch after episode 3 in all three at 84–86%, then the scale 5.0 → 1.0 by episode 10 and 1.0 for 11–12; about 2 hours each, side by side, no errors). Means over the route-off episodes 4–12, then the paired count:

      | | A: ladder5, old storage | B: ladder5, next-state fix | C: ladder4, next-state fix |
      | --- | --- | --- | --- |
      | Heads differ from the route | 57.1% | **54.5%** | 53.1% |
      | Chosen moves differ | 45.8% | 44.5% | 43.9% |
      | Lead time | 143 | **133** | 139 |
      | Guard stops | 546 | **496** | 517 |
      | L3 | 6.2 | 4.8 | 5.8 |
      | Near misses | 157 | **29** | 52 |
      | Delivered | 75 | 76 | 76 |
      | Learner, live | 73.0% | 74.3% | 72.2% |
      | Safety value / loss | −0.246 / 0.0025 | −0.229 / **0.0017** | −0.141 / 0.0014 |
      | Efficiency value | 0.655 (falling to 0.56) | 0.751 (to 0.70) | 1.402 (barely moving) |
      | Last 3 (scale 1.0): lead / guard / near misses | 126 / 424 / 147 | **110 / 341 / 11** | 122 / 433 / 34 |

      - **B against A (the next-state fix): a consistent improvement, modest in size.**
        - The heads' disagreement with the route was lower in **9 of 9** paired episodes (−2.6 points; a sign test gives p ≈ 0.004), and the chosen moves in 7 of 9.
        - Lead time was 7% lower (13% in the last three, 110 against 126), lower on 6 of 9. Guard stops 9% lower (6 of 9). L3 lower on 7 of 9. Near misses lower on 7 of 9; A's mean is carried by spike days (549, 422).
        - The safety loss was a third lower (0.0017 against 0.0025) and the efficiency loss lower at the end (0.0073 against 0.0085): more consistent targets learn more cleanly.
        - The safety value was less pessimistic from the first episode (−0.255 against −0.311).
        - Each of these except the heads is 6–7 of 9, not firm alone; together, and with the heads at 9 of 9, it is clear. Failure draws differ between arms (the order leak), which adds noise to every paired comparison.
        - **It does not solve the small map.** The heads still differ on about half of clear moves (49% in the last three), and lead time is still about 2.5× the route's. That fits the diagnostic: the bigger share of the problem is the blurry picture, which this fix does not touch.
      - **B against C (the starting point, both fixed): ladder5 is at least as good.**
        - C's heads were slightly lower (6 of 9, −1.4 points). But B had fewer guard stops (7 of 9), fewer near misses (7 of 9) and slightly lower lead time. C's Learner, trained from scratch in the warm-up, was lower on 9 of 9.
        - **Rohit's "unlearning" concern points the other way.** ladder4's values belong to the route-on world: its efficiency value stayed at about 1.4 through 12 episodes, while A and B, starting from ladder5, sit at 0.6–0.75 for the same small-map days. ladder4 is the one carrying values to unlearn, as run 9's efficiency value did over 80 episodes (2.07 → 1.04).
      - **The aggressive schedule (5 → 1)** pulled the chosen moves about as far as 3 → 0 did in dry run 2: about 16 points at the start in both (A: 62.5% → 43.6% at scale 5; dry run 2: 63% → 48% at scale 3). The nudge saturates, since the Learner is about 72% right on this map and unsure where it is wrong. No harm seen either; B's best episodes came at the floor of 1.0.
    - **So two problems, two fixes:**
      - The learning targets: the next-state fix, being tested now in dry runs A/B/C.
      - The picture: a direct path from each robot's own encoding to the heads (and the Learner) beside the trunk's mixed summary. Concatenate the node encoder's output, from before message passing, to the trunk's output, with the new input weights of each head starting at zero, so the warm start stays exact. Decided after the dry runs.
    - **Decided (Rohit, 14 October): fix the blur before the long run** ("it's best to do a fixed long run in the end"). Then run 10 from ladder5, with the next-state fix, the 5 → 1 schedule and a floor of 1.0 (also at deploy), both maps, fleet counts 15/25/40/60, 80 episodes, and the order-leak fix before its graduation.

    **Built: the self path (`gnn.self_skip`), 14 October.**
    - **`agent_warehouse.py`:**
      - **The new maps.** With the switch on, each Q head gets a bias-free linear map from the fleet's own encoding (the node encoder's output, before any message passing) into its first layer: the head's first layer computes `W·h + U·h0 + b` instead of `W·h + b`. The Learner gets the same, reading `h0.detach()`, so it still never trains the trunk.
      - **Same network at the start:** the maps start at exactly zero, so a warm start is the same network bit for bit.
      - **Nothing else changes:** they are registered last and built from their own random stream (seed 2027), so every other parameter, their order and every later global random draw are as without them.
      - **Own optimizer:** the maps have their own Adam, so the main and the Learner's optimizers cover exactly the same parameters as before, and ladder5's states load exactly. They are clipped on their own (the Q heads' maps together, the Learner's apart), so at zero the first update of the main path is the same as without them.
      - **Loading:** `load()` lets a checkpoint without the self path fill everything else exactly, and anything beyond that still fails loudly, including a self-path checkpoint loaded into a plain agent. `save()` stores the maps' optimizer, and `checkpoint_has_self_skip()` tells the two kinds of checkpoint apart.
    - **`main_runner_warehouse.py`:** passes `gnn.self_skip` and reports it on the `edges` line. **`benchmark_flowrra.build_agent`:** builds each deploy arm with or without the self path, from its checkpoint. **`config_warehouse.py`:** `gnn.self_skip`, default False.
    - **`test_self_skip.py`** (new):
      1. Building it draws nothing from the global generator, and the other layers are identical.
      2. At zero, the Q-values, per-head values and Learner logits are identical.
      3. A warm start from a plain checkpoint loads every other weight and both optimizers' states exactly, with the self path at zero and the output identical.
      4. The first update leaves the main path identical, and both kinds of map leave zero.
      5. A self-path checkpoint round-trips, and fails loudly in a plain agent.
      6. In real FLOWRRA steps with the conv encoder, it learns with finite losses and moves off zero.
      7. deploy's builder builds each kind correctly.
    - **The dry run:** D is B plus `--set gnn.self_skip=True`, run beside a repeat of B (B2, unchanged) as a yardstick for run-to-run noise: two identical runs on one GPU are not bit for bit the same. D against B and B2, by episode.
    - **Rohit's machine, 14 October:** `test_self_skip.py`, `test_learner_router.py`, `test_learner_head.py` and `test_next_state.py` all ALL PASS. D and B2 started at 22:28 IST. The logs confirmed:
      - D: five `--set` lines, `self_skip=True`, and "has no self path: everything else loaded exactly; the self path starts at zero".
      - B2: four `--set` lines, `self_skip=False`, and no `[Agent]` line (ladder5 matches exactly).

    **Results, dry runs D and B2 (15 October).** Small map only, 12 episodes, seed 0, warm from ladder5, the next-state fix on, scale 5 → 1. Means over the route-off episodes (4–12), with D's paired wins against B2:

      | Measure | D (self path) | B2 (repeat of B) | B | D better than B2 |
      | --- | --- | --- | --- | --- |
      | Lead time | **65.1** | 132.7 | 133 | 9 of 9 |
      | Guard stops | **188** | 406 | 496 | 7 of 9 |
      | Learner, live (router line) | **98.3%** | 73.6% | 74.3% | 9 of 9 |
      | Learner, training batches | 99.5% | 81.3% | – | 9 of 9 |
      | Chosen moves differ from the route | **28.0%** | 42.4% | 44.5% | 9 of 9 |
      | Heads alone differ from the route | 49.6% | 51.9% | 54.5% | 6 of 9 |
      | Delivered (of 80) | 77.3 | 75.0 | 76 | 7 of 9 |
      | L3 | 4.8 (one day of 20, ep 11) | 3.8 | 4.8 | 4 of 9 |
      | Near misses | 51 (one day of 174, ep 12) | 54 | 29 | 5 of 9 |
      | Collisions | 0 | 0 | 0 | – |
      | Safety value / loss | −0.249 / 0.0019 | −0.230 / 0.0017 | −0.229 / 0.0017 | value lower 9 of 9 |
      | At scale 1.0 (eps 10–12), lead time | 47 / 120 / 50 | 84 / 158 / 95 | – | 3 of 3 |

      - **The noise yardstick holds:** B2 and B agree on lead time (133 against 133) and the Learner (74%), with the heads about 3 points apart. Every D gap above is several times that.
      - **The blur was the limit.** The Learner was already 92–98% right in the shadow episodes (1–3), against its 72–78% ceiling in every earlier small-map run. With its own encoding beside the mixed summary, it reads the route's move in crowds.
      - **An accurate Learner, through the nudge, halves the small map's lead time and guard stops,** into the route's own range: the route-on warm-up days (1–3) ran at 29–70 in both arms, and D's route-off days at 41–76, with one day of 120 (fleet sizes differ by day, so this is a range, not a paired figure). The gain shows most at the floor of 1.0 (episodes 10–12).
      - **The heads alone barely moved** (49.6% against 51.9%, 6 of 9). The Q heads have the new path too, but their targets are still value targets; 12 episodes did not teach them the route. So the gain is carried by the Learner's nudge, and **the floor of 1.0 at deploy is essential**: with the nudge at 0, D would drive about as B does.
      - **Safety:** no collisions; near misses and L3 level on average, each with one spike day. The safety value sat slightly lower in D on all 9 days (−0.249 against −0.230), a small drift to watch in run 10 (run 9's P4 failed on the same drift). The efficiency value was higher in D (0.86–0.98 against 0.73–0.85) and its loss lower.
      - **Decided (15 October): adopt the self path.** Run 10 keeps the `--set gnn.self_skip=True` line.

    **Next steps (written 14 October; step 1 done 15 October, above):**
    1. ~~**Read D and B2.**~~ Done: adopt the self path.
    2. **Agree the predictions, then run 10** (`ladder_run6`), about 9 hours, with the self path.

       **Run 10 started 15 October, 00:56 IST (19:26 UTC on the machine), 80 episodes; the six predictions below agreed by Rohit as drafted.** 80 rather than 60: the schedule scales with length (warm-up 6.25%, floor tail 18.75%), so 80 keeps run 9's shape (5 warm-up, fade, last 15 at the floor), a tail of 15 sees each of the 8 map-and-fleet pairs about twice, and the heads need time. The log confirmed five `--set` lines, `next_state_edge_features=True | next_state_adjacency=choice | self_skip=True` on the edges line, and "has no self path: everything else loaded exactly; the self path starts at zero".

       **Mid-run check, episode 50 (15 October, 05:00 IST; CSV only).** Means by window, route off from episode 6, run 9 (ladder_run5) at the same episodes. Two cautions: run 10 is at a higher scale at the same episode (5 → 1 against 3 → 0; about 2 against 0.8 at episode 50), and it includes the 15-robot days, which deliver less, so deliveries and the large map's lead time are not like for like. No collisions in either run.

       | Small map | 6–20 r9 / r10 | 21–35 r9 / r10 | 36–50 r9 / r10 |
       | --- | --- | --- | --- |
       | Learner, live | 72% / **97%** | 75% / **99%** | 74% / **99%** |
       | Heads alone differ | 56% / 48% | 56% / 46% | 56% / **44%** |
       | Chosen moves differ | 46% / 23% | 47% / 20% | 50% / 21% |
       | Lead time | 145 / **68** | 145 / **96** | 144 / **82** |
       | Guard stops | 424 / 140 | 511 / 118 | 688 / 151 |
       | L3 | 5.7 / 1.6 | 4.8 / 2.1 | 6.2 / 4.3 |
       | Near misses, mean (median) | 27 (14) / 95 (13) | 55 (34) / 9 (7) | 42 (20) / **166 (96)** |

       - **Large map:** Learner 98–99%; heads 30% / 26% / 31% (run 9: 45% / 36% / 34%); chosen 10–11%; lead time 143 / 117 / 136 against run 9's 156 / 146 / 131; guard stops lower; near misses level (median 2.5–3.5 against 0–1.5).
       - **Provisional reading of the predictions:** 1 (Learner) on track; 2 (chosen moves) on track, but the scale is still about 2; 3 (small-map lead time 40% below run 9) on track at about −43%, close to the line; 4 (safety) **at risk on small-map near misses**, rising late (episodes 46, 47 and 49: 291, 96, 638; episode 14: 440); 5 (safety value) −0.141, in its band, but its loss is creeping (0.0007 → 0.0016); 6 (heads ≤ 45% on the small map) at 44%, on the line, falling slowly (48 → 46 → 44).
       - **To trace after the run:** the small map's near-miss days (with the carried item, the worst near-miss day), to see whether the robots are approaching people more or the spikes are failure-heavy days.
       - **Like for like, added the same morning.** The CSV's `stream_fleets` column is the fleet size (the log's `k=` and the CSV's `agents` are the robots still on the floor at the end). Over episodes 6–50, route off, by fleet size:
         - **Small map:** lead time 170 → 76 (25 robots), 160 → 59 (40), 116 → 45 (60, one day); guard stops 334 → 180, 796 → 207; deliveries level or higher. Near misses: the median is lower at every fleet size (53 → 29 at 25 robots, 32 → 5 at 40), the mean higher, carried by spike days.
         - **Large map:** lead time lower at every shared fleet size (175 → 166, 161 → 131, 113 → 94), deliveries level, guard stops lower at 40.
         - **The spike days** are 49 (25 robots, 638, one person), 14 (40, 440), 46 (25, 291, one person) and 47 (25, 96): all on the small map, with 5–9 robots failed in waves. The late cluster (46, 47, 49) is why the episode 36–50 window looks bad. Scored like for like on episodes 36–50 against run 9's last 15 (`report_run.py`), near misses are above on both mean and median (116 / 15.5 against 23 / 11), so prediction 4 is failing at mid-run; prediction 3 fails by a hair on its large-map clause (133 against 129). Both are judged at episode 80.
         - **The efficiency value rose** to about 1.8–1.9 (run 9 fell to about 1.4 by this point), with its loss climbing like run 9's. Faster deliveries raise the honest level, so this may be honest; to judge at the end.
       - **`report_run.py`** (new, CSV only, seconds): tables and graphs for a training run against a baseline, run 10's six predictions scored (all fleets and like for like), and the graduation scoring (3 of 5 plus the safety gate, paired by day, by map). Checked against full9: it reproduces entry 42's verdict (ladder5 lost to ladder4 on every measure).

       **Predictions (proposed and agreed 15 October, before the run).** "Last 15" is episodes 66–80, route off at the floor of 1.0; run 9's figures are in entry 42.

       | # | Prediction | Fails if |
       | --- | --- | --- |
       | 1 | The Learner reads the route on both maps | live agreement in the last 15 below 95% on the small map or 97% on the large |
       | 2 | The chosen moves follow the route | chosen moves differ on more than 35% of small-map or 20% of large-map clear moves (last 15) |
       | 3 | The small map is fixed | small-map lead time in the last 15 is not at least 40% below run 9's last 15; the large map's is worse than run 9's |
       | 4 | Safety holds | any collision trend, or near misses or L3 in the last 15 above run 9's last 15 (mean and median) |
       | 5 | The safety value is steady | the safety value leaves −0.10 to −0.20, or its loss climbs over the last 30 episodes (D sat slightly lower than B2) |
       | 6 | The heads learn the clear with the self path (the open question, allowed to fail) | small-map heads still differ on more than 45% of clear moves in the last 15 |

       Command:

       ```
       nohup python -u main_runner_warehouse.py --seed 0 --maps 25_5_2_5_2_1,50_20_5_10_5_2 --scens-dir all_scens_v5 --episodes 80 --agent-sets 15,25,40,60 --max-steps 800 --resume ladder_run5/flowrra_curriculum.pth --set training.learner.router.scale_start=5.0 --set training.learner.router.scale_end=1.0 --set gnn.next_state_adjacency=choice --set gnn.next_state_edge_features=True --set gnn.self_skip=True --out ladder_run6 > ladder_run6.log 2>&1 &
       ```

       The 15-robot set follows the overfit check (entry 42, step 5). The warm-up switches after episode 5, since ladder5's Learner passes 80% on the small map.
    **Rohit's run sheet for the end of run 10 (written 15 October, 05:00 IST; credits are out, so these run without me).** All on his machine, in ~/flowrra_aws.
       1. **Check run 10 finished:** `tail -3 ladder_run6.log` shows Ep 0080, and `ls -la ladder_run6/` shows `flowrra_curriculum.pth` with a fresh time.
       2. **Score the training run** (seconds):
          ```
          python report_run.py train --csv ladder_run6/curriculum_metrics_ladder_run6.csv --base ladder_run5/curriculum_metrics_ladder_run5.csv --score-run10 --out report_run6
          ```
          Gives `report_run6.txt` (all tables and the six predictions, scored twice: all fleets, and like for like without the 15-robot days), `report_run6_training.png` and `report_run6_by_fleet.png`. Adjust the CSV paths if they sit elsewhere.
       3. **Set the floor for deploy** (deploy_eval reads it from the config at start-up; ladder4 has no Learner and RULES no network, so only ladder6 is affected):
          ```
          sed -i 's/"scale_end": 0.0,/"scale_end": 1.0,/' config_warehouse.py && grep -n '"scale_end"' config_warehouse.py
          ```
          It must print `"scale_end": 1.0,`.
       4. **The graduation, four runs in parallel** (16 cores; same arms in each, so every arm meets the same days):
          ```
          nohup python -u deploy_eval.py --arms ladder6=ladder_run6/flowrra_curriculum.pth:noroute,ladder4=ladder_run4/flowrra_curriculum.pth,RULES --seeds 0,1,2 --failures off --people off --safety on --check --out deploy_eval_clean10.csv > clean10.log 2>&1 &
          nohup python -u deploy_eval.py --arms ladder6=ladder_run6/flowrra_curriculum.pth:noroute,ladder4=ladder_run4/flowrra_curriculum.pth,RULES --seeds 0,1,2,3,4 --failures on --people on --safety on --out deploy_eval_full10.csv > full10.log 2>&1 &
          nohup python -u deploy_eval.py --arms ladder6=ladder_run6/flowrra_curriculum.pth:noroute,ladder4=ladder_run4/flowrra_curriculum.pth,RULES --agents 15 --seeds 0,1,2 --failures off --people off --safety on --out deploy_eval_clean10_k15.csv > clean10_k15.log 2>&1 &
          nohup python -u deploy_eval.py --arms ladder6=ladder_run6/flowrra_curriculum.pth:noroute,ladder4=ladder_run4/flowrra_curriculum.pth,RULES --agents 15 --seeds 0,1,2,3,4 --failures on --people on --safety on --out deploy_eval_full10_k15.csv > full10_k15.log 2>&1 &
          ```
          - A minute in: `grep -E "\[deploy\] arm|\[Bench\]" clean10.log` should show ladder6 with "route OFF for this arm" and `self_path=on`, `learner=on`; ladder4 with `self_path=off` and no route-off note.
          - When clean10 is done: `grep -E "IDENTICAL|DIFFERENT" clean10.log` should say IDENTICAL.
          - The fourth run (full, 15 robots) is new: it asks whether the network copes when the stress failures take out most of a small fleet, as in training.
       5. **Score the graduation** (seconds each):
          ```
          python report_run.py deploy --csv deploy_eval_clean10.csv --ref ladder6 --out report_clean10
          python report_run.py deploy --csv deploy_eval_full10.csv --ref ladder6 --out report_full10
          python report_run.py deploy --csv deploy_eval_clean10_k15.csv --ref ladder6 --out report_clean10_k15
          python report_run.py deploy --csv deploy_eval_full10_k15.csv --ref ladder6 --out report_full10_k15
          ```
          Each prints ladder6 against ladder4 and against RULES, for both maps together and each map alone, ending in a verdict line: "GRADUATES", "held by the safety gate" or "does not graduate".
          - **The decision of record is clean10 and full10 against ladder4, both maps together**, as fixed in entry 42. Both must say GRADUATES for ladder6 to become the champion. The by-map lines and the k15 runs are reported, not scored.
          - **The order leak is still in** (step 3 below is not built). Clean is unaffected: no failures and no people, so nothing random to leak. Full has the leak exactly as full9 did, so it compares fairly with run 9's graduation, but single days are noisier. If full10's verdict hangs on one measure with p near 0.05, repeat it after the fix.
          - **Rohit's hope** (15 October): deploy days may be gentler than training's bad days. Clean has no failures at all. Full uses the same stress failures as training, so the bad days are not gentler there; what differs is exploration, 0.015 at deploy against the training log's `eps` (0.025 and rising in run 10), and that ladder4 meets the very same days. The safety gate (near misses with people, p < 0.05) settles it either way.
       6. **When credits return:** send the `.txt` reports and the PNGs, not the raw CSVs and logs (much smaller to read). Then log the scoring here.

    **Results, read 18 October (Rohit ran everything on his own, 15–17 October).**
    - **Run 10 (ladder6), training, scored by `report_run.py` on episodes 66–80:**
      - P1 (Learner) ✅: 99.4% small, 99.1% large.
      - P2 (chosen moves) ❌ by a hair on the large map: 20.6% against ≤ 20% (small 26.9% ✅).
      - P3 (small map fixed) ✅: lead time 78 against run 9's 173 (−55%); large 113 against 129.
      - P4 (safety): split. ✅ over all fleets; ❌ like for like on L3 (5.5 / 6.5 against 3.9 / 2.0). Near misses were lower (6.4 / 3.0 against 23 / 11). No collisions.
      - P5 (safety value) ❌: −0.185, inside the band, but the loss kept climbing (+0.0011 over the last 30).
      - P6 (heads) ✅: 43.6%.
      - Also: as the scale fell 5 → 1, the large map's heads drifted away from the route (26% → 37%) and its chosen moves rose (10% → 21%).
    - **Run 11 (ladder7)**, Rohit's own follow-up: fleets 25/40/60 only, no 15-robot days. Its router scale ended at **0**, not 1.0 (the graphs; chosen moves equal the heads in its last 15). That means the config's `scale_end` was 0.0 when it started.
      - Its small-map heads improved to 36.9%, the best yet; the large map's stayed at about 29%.
      - Small-map lead time rose to 104 in the last 15, as the nudge faded to 0.
    - **Graduation (`clean10`, `full10`, `clean11`, `full11`; ladder6 or ladder7 with the route off, against ladder4 with the route on and RULES): neither graduates; ladder4 stays the champion.**

      | Both maps | ladder6 clean / full | ladder7 clean / full | ladder4 clean / full |
      | --- | --- | --- | --- |
      | Lead time | 111.5 / 127.2 | 106.4 / 120.8 | **76.2 / 85.8** |
      | Book delivered | 0.992 / 0.927 | 0.997 / 0.926 | 0.999 / 0.955 |
      | L3 fires | 5.9 / 6.4 | 6.7 / 8.6 | **1.9 / 1.2** |
      | Near misses with people (full) | **11.2** | 25.5 | 13.8 |
      | Guard stops | 322 / 300 | 187 / 252 | 78 / 84 |

      - Faster than ladder4 on 0 of 18 clean days and 0 of 30 full days (ladder7: 1 of 30). Wins of 5: ladder6 0 (clean) and 1 (full, near misses); ladder7 0 and 0. Both pass the safety gate: no collisions, and near misses are not firmly worse. ladder6 had fewer near misses than ladder4 on the large map (2.0 against 9.4, p 0.10).
      - Against ladder5 on the same clean days (clean9): small lead 87 against 157, much better; large 136 against 123, worse.
    - **An open question that changes the reading: at what scale did deploy run?** deploy_eval reads `scale_end` from the config at start-up. ladder6 did much worse at deploy than in its own training: in full, lead time small 99 against 78 (+27%) and large 155 against 113 (+38%); guard stops small 529 against 229. ladder7, trained to scale 0, matched its training (99 against 104, 143 against 135; guard stops 422 against 358). Run 11 also shows the config was at 0.0 when it started. If deploy also read 0.0, ladder6 was graduated without the nudge that dry run D showed carries the gain. To check: `grep "router scale" clean10.log full10.log clean11.log full11.log`.
    - **Answered 18 October: deploy ran at scale 0.0.** The `[Bench]` lines of clean10, full10, clean11 and full11 all say `router scale 0.0`. So both graduations tested the heads alone, without the Learner's nudge. That is not the design that was trained (the floor of 1.0), and the dry runs showed the nudge carries the gain. The graduation of the router design is still to be done: the floor sweep below. Rohit started it on 18 October.
    - **The Kiva map (Rohit, with Gemini's help, 15–17 October): the first test on a layout none of the networks has seen.**
      - `Kiva_2D_46_33_1`: 46 × 33 cells, 1,262 nodes, with rack blocks between aisles, a goal bank of 344 cells and start/goal files per seed. There is also a two-floor version, `Kiva_3D_2Floor_5Ladders`: the same floor twice, joined by 5 lifts. It is not tested yet.
      - Tested on the 2D map: seeds 0–1 × fleets 15/25/40/60 (8 days per condition); books of 150 and 200 orders; clean (1,200 steps) and full (1,500 steps). Arms: ladder4 with the route on against RULES, plus ladder7 `:noroute` against RULES in clean, at scale 0 like the rest.
      - **ladder4 beats RULES on the unseen map in all four conditions.** No collisions anywhere.

        | Kiva 2D | Book delivered, ladder4 / RULES | Lead time | L3 fires | Near misses with people | Of 5 |
        | --- | --- | --- | --- | --- | --- |
        | Clean, 150 | **0.98** / 0.91 (8 of 8 days) | **164** / 210 (7 of 8) | **48** / 307 (8 of 8) | – | 3 |
        | Clean, 200 | **0.97** / 0.82 (8 of 8) | 220 / 241 (3 of 8) | **84** / 398 (8 of 8) | – | 3 |
        | Full, 150 | **0.89** / 0.79 (7 of 8) | 192 / 194 | **32** / 321 (8 of 8) | 133 / 80 (p 0.11) | 3, gate passed |
        | Full, 200 | 0.82 / 0.74 (5 of 8) | 203 / 236 | **56** / 405 (8 of 8) | 194 / 198 | 4 |

        - In clean, with 150 orders, ladder4 cleared the whole book with 15, 25 and 40 robots, in about 15–16 minutes (steps 900–1,076). RULES cleared it on no day within the 20 minutes.
        - **Cautions:**
          - only 2 seeds;
          - ladder4's near misses with people were higher in full with 150 orders (not firmly);
          - its guard stops were higher than RULES' at 40–60 robots;
          - its heads differ from the route on 63% of clear moves on this map, against about 30% on the trained maps. What transfers is the route plus the meeting judgement, not the heads' driving.
      - **ladder7 with the route off, at scale 0, does not transfer:** book 0.72 against 0.91 for RULES and 0.98 for ladder4; lead time 307 against 210 and 164. **But its Learner read the route on 85–97% of clear moves on the unseen map (92% on average).** The skill that would steer it transfers; at scale 0 it was switched off.
    - **The floor sweep, scored 18 October (clean, seeds 0–2, both maps, 25/40/60 robots; 18 days paired with clean10's ladder4 rows; route off; floor = `router.scale_end` at deploy):**

      | Against ladder4 (route on) | Lead time | L3 fires | Book | Lead: days better / worse, p |
      | --- | --- | --- | --- | --- |
      | ladder4 | 76.2 | 1.94 | 0.999 | – |
      | ladder6, floor 0 (clean10) | 111.5 | 5.89 | 0.992 | 0 / 18, p 0.000 |
      | ladder6, floor 1 | 91.6 | 6.78 | 0.994 | 2 / 16, p 0.000 |
      | **ladder6, floor 2** | **78.7** | **2.67** (p 0.41) | **0.998** (p 0.79) | 6 / 12, **p 0.47** |
      | ladder6, floor 3 | 79.9 | 3.67 (p 0.06) | 1.000 | 6 / 12, p 0.12 |
      | ladder7, floor 1 / 2 / 3 | 89.1 / 80.9 / 80.1 | 5.0 / 3.9 / 6.0 (all p < 0.01) | 0.999–1.000 | p 0.000 / 0.027 / 0.024 |

      - **ladder6 at a floor of 2 cannot be told apart from ladder4 with the route on.** Lead time is 3% slower (p 0.47); L3 fires and the book are not firmly different; no collisions. With the nudge off it was 46% slower. This is entry 42's research result on clean days: **the network drives the whole floor itself** (route off, the Learner inside it doing the routing) and matches the route-on champion.
      - **By the rule it does not graduate:** 0 wins of the 4 measures present (near misses are absent in clean), because matching is not beating. ladder4 stays the champion for deployment.
      - **Floor 2 is the sweet spot.** Floor 1 under-steers; floor 3 brings more L3 fires (heavier steering into meetings, where the route's move is not the right one).
      - **ladder7** (trained to scale 0 in its tail) is consistently worse on L3 at every floor. **ladder6 is the line to carry.**
      - **Floors 2 and 3 cannot be told apart either** (head to head: lead time 78.7 against 79.9, p 0.42; L3 fires 2.7 against 3.7, p 0.30; book 0.998 against 1.000, p 0.18). Floor 2 missed about 3 of 1,440 orders over the 18 days, floor 3 none, ladder4 one. By map, floor 2 matches ladder4 on the large map (lead time 104.8 against 105.4) and comes within 12% on the small one (52.5 against 47.1, p 0.36).
      - **Decided with Rohit (18 October): settle the floor in the full condition, where failures and people can separate them.** ladder6 at floors 2.0, 2.5 and 3.0, run as separate processes. With keyed draws, every process meets the same days, so one ladder4 + RULES run pairs with all three. Then Kiva clean at floors 2 and 3, paired with the existing Kiva rows for ladder4 and RULES (clean is unaffected by keyed draws).
      - **The rule, fixed before the results:** choose the floor with the most wins of 5 against ladder4 in full, with the safety gate passed. A tie goes to fewer near misses with people, then to more of the book delivered.
      - **full12 scored, 18 October** (keyed days, seeds 0–4, both maps, 25/40/60 robots: 30 paired days; ladder4 and RULES in the floor-2.0 process, paired with all three floors):

        | Against ladder4 | Book | Lead time | L3 fires | Near misses with people | Wins of 5 |
        | --- | --- | --- | --- | --- | --- |
        | ladder4 | 0.947 | 83.3 | 1.57 | 22.0 | – |
        | **ladder6, floor 2.0** | **0.949** | 90.9 (p 0.001) | 3.67 (p 0.001) | **16.3** (p 0.27) | **2**, gate passed |
        | ladder6, floor 2.5 | 0.956 | 95.0 (p 0.000) | 4.47 | 38.5 (p 0.11) | 1; gate FAILED on the small map |
        | ladder6, floor 3.0 | 0.948 | 89.9 (p 0.008) | 4.10 | 27.1 | 1 |

        - **By the rule, the floor is 2.0** (most wins). It also beats 2.5 and 3.0 head to head, 3 of 5 each, mostly on near misses with people. **ladder4 stays the champion:** ladder6 does not graduate against it (2 of 5).
        - **Against RULES, ladder6 at floor 2.0 GRADUATES** in full (3 of 5: book, L3 fires 3.7 against 6.0, near misses 16.3 against 35.4; lead time lost, 90.9 against 82.9). It is the first route-off network to pass that test.
        - **By map:**
          - **Large map: parity with ladder4,** and safer around people. Lead time 121.9 against 118.6 (p 0.21), L3 fires 1.3 against 1.2, book 0.941 against 0.938, and near misses with people **3.3 against 10.7 (p 0.046), firmly fewer**.
          - **Small map: the gap that remains.** Lead time 59.9 against 47.9 (+25%, p 0.001), L3 fires 6.0 against 1.9, guard stops 252 against 88.
        - **Progress:** at floor 0 (full10, the old days), ladder6 was 48% slower than ladder4 in full; at floor 2 it is 9% slower, with no collisions in any arm.
        - **Where to push next:** the small map's meetings. Its robots still leave the route on about a quarter of clear moves (training), and each departure on a crowded floor costs guard stops and L3 fires. Candidates: cause 2, charging a robot for the waiting it causes others (the reward change held since entry 43), or more training from ladder6 at the floor of 2.
    - **On comparing Kiva with the state of the art (18 October).** Rohit shared an outside analysis (Gemini) that put ladder4's Kiva numbers at "the top tier of centralized MAPD solvers" and above physical Kiva drive rates. Reviewed here; the claims do not hold as stated:
      - **Our throughput is demand-limited, theirs is saturated.** Our book releases a fixed number of orders, and orders per hour is capped by the book: 450 per hour is simply 150 orders in 20 minutes. Lifelong MAPF benchmarks (RHCR, League of Robot Runners) give every agent a new task on arrival and measure tasks per timestep under saturation. Per-robot rates from a fixed book also fall as the fleet grows (39.8 at 15 robots, 26.0 at 25) only because the same orders are shared.
      - **The doubled "hop-normalized" figures are not valid.** Our robots move half a cell per step. Doubling the throughput assumes everything scales with speed, but a book released over fixed time does not release orders faster.
      - **Different task, map and rules.** Our orders include dock entry and exit, braking by the intent guard and a safety bubble. The Kiva map here was generated for us, not the benchmark's map file. The published figures in that analysis came without sources and were not checked.
      - **Partly misread.** `router_q_differs` is measured in the clear, where the route drives ladder4, not inside contention. Its 63% says ladder4's heads do not drive this map well; it is not "proactive untangling". The "every metric including guard stops" claim came from the first 5 runs; over all days, ladder4's guard stops are higher at 40–60 robots.
      - **What does stand:** paired days, the same simulator, an unseen layout. ladder4 cleared the 150-order book with 15–40 robots in about 15 minutes with no collisions, where RULES did not finish in 20, with 6–10× fewer L3 fires. That is a generalization result against our own rule baseline, not against the published state of the art.
      - **Found 18 October: a fair harness already exists from v1.** `kiva_instances.py` extracts, from POGEMA itself, the exact lifelong instances Follower (Learn-to-Follow) was evaluated on: RHCR's kiva map (`wfi_warehouse`, 33 × 46, checked cell for cell), goals redrawn on arrival with each agent's goal sequence fixed in advance, 512 timesteps, 32–192 agents, seeds 0–9, throughput in goals per timestep. `run_kiva.py` runs FLOWRRA on them in the core's lifelong mode, counting time in POGEMA's timesteps (2 FLOWRRA steps each) and reporting collisions beside throughput. Its checkpoint loader predates the Learner, the self path and the router floor, so it needs `benchmark_flowrra.build_agent`, arms like deploy_eval's (`:noroute`, RULES), the deploy safety settings, and a braking switch for the MAPF-rules run. With that, ours can be set against the published numbers on the identical protocol.
      - **Built 18 October: `kiva_eval.py`**, which replaces run_kiva.py for this.
        - It runs deploy_eval's own arms on the POGEMA instances: `label=checkpoint[:noroute]`, RULES, and PIBT, with the networks built by `benchmark_flowrra.build_agent`, the deploy safety switches on and greedy play by default.
        - `--braking off` gives the MAPF-rules run.
        - `deploy_eval.run_one` gained a lifelong branch (`inst["lifelong"]`: goal sequences, no order book, no docks), and its rows keep the `lifelong_*` counts.
        - PIBT is our own lifelong version, on the same graph and goal sequences, with vertex and swap conflicts counted (both 0 in tests). It is an in-house yardstick beside the published numbers.
        - **Sandbox checks** (synthetic goal sequences, because POGEMA 1.4.0 does not install here; the real instances are on AWS):
          - With 1 and 4 robots over 100 timesteps, RULES reaches exactly as many goals as PIBT (3 and 17). The time conversion and the motion are right, so any gap is the handling of robots meeting.
          - **An early warning:** over only 60 timesteps, RULES reached 0.30 goals per timestep against PIBT's 0.78 with 32 robots, and 0.43 against 1.92 with 96. Braking off moved 96 robots only to 0.48. L3 fires (10 and 34) and guard stops (110 and 612) were high for so few steps.
          - So on this open Kiva grid, the meeting machinery, not the bubble, is what costs. Short horizons and synthetic tasks: the real instances and the networks will tell.
          - Runtime: about 0.6 s per step with 96 robots (RULES), so 1,024 steps take about 10 minutes, and the 192-robot runs longer.
      - **First real numbers, 18 October (Rohit's AWS, POGEMA instances, seed 0, 32 robots, 512 timesteps, safety rules on):** PIBT 1.006 goals per timestep (515 goals); RULES 0.168 (86 goals, L3 fires 273); ladder6 at floor 2.0: 0.137 (70 goals, L3 64); ladder4: 0.129 (66 goals, L3 175). No collisions anywhere. At 64 robots PIBT reached 1.678. **Ours reach 13–17% of PIBT at only 2.5% floor density.**
      - **Sandbox diagnosis the same morning** (synthetic POGEMA-style tasks, 32 robots, 512 timesteps, RULES; PIBT 1.045 on these tasks):
        - Baseline: 0.186, with **453 L3 fires**.
        - One thing off at a time: corridor-entry rule 0.189; braking (`--braking off`) 0.285; intent guard 0.246 (123 collisions); all six conflict rules 0.064; all safety switches 0.162 (84 collisions).
        - **No single switch closes the gap.** Braking costs the most of the switches (+53% when removed), but even then RULES reaches only 27% of PIBT.
        - With 1 or 4 robots, ours equals PIBT exactly. So the loss comes from robots meeting each other, at only modest density (2.5% of cells with 32 robots), and L3 fires about every two steps say meetings are escalated again and again. Where exactly the time goes is not known yet.
        - **Corrected 18 October (Rohit caught both):** an earlier version of this entry blamed "Kiva's one-cell-wide two-way aisles" and said the meeting machinery "was built and tested on maps with few such aisles". Both were wrong.
          - RHCR's kiva map has no one-cell aisles. Each shelf block is one row of racks with an endpoint row on each side, blocks are separated by fully open rows, and open columns cut through every 11 cells.
          - Meetings were designed for, explicitly: the priority rule (nearest goal first, then seniority, then aging for a long waiter) exists for exactly this situation.
        - **Checked 18 October (Rohit's question: are the instances read as designed, and are robots given several orders at once?):**
          - One goal at a time. Each JSON holds 300 goals per robot, but the env deals them out like a deck: the mission's goal is the first, and `_lifelong_next` hands over the next only when the robot arrives. PIBT gets the same decks.
          - Every POGEMA goal is an RHCR endpoint, and the cache precomputes distance maps for all 480 endpoints, so no robot falls back to Manhattan distance.
          - **What lifelong switches off is our dispatcher, by the benchmark's rules and for every method:** no shared book, no nearest-unclaimed choice or reservation, no leaving the floor at a dock after delivery, no waiting off the floor until work exists. Each next goal is a uniform draw over the whole map. So book-mode and lifelong numbers do not compare one to one (book trips are to the nearest open order and the nearest dock).
          - The meeting handling is the same code in both modes. In book mode on the Gemini map (clean, 150 orders), RULES already fired L3 307 times per 1,200-step day on average over 15–60 robots, and cleared the book on no day within 20 minutes. In lifelong it fires 252–273 times in 1,024 steps with 32 robots. ladder4: 48 per day in book mode, 175 in lifelong (RHCR map).
          - **The Gemini map in lifelong mode (sandbox, synthetic sequences drawn from its 344-cell goal bank, 32 robots, 512 timesteps):** RULES 0.199 goals per timestep (102 goals, L3 252) against PIBT 1.162 (595). The same collapse as on RHCR's map, so the map is not the cause.
        - **Where the priority rule acts on Kiva (counted 18 October).** The back-out (the loser and its followers back out to the junction and pull over) applies only inside corridors, chains of single-lane cells with exactly two neighbours. Kiva has almost none: 20 of RHCR's 1,278 cells, and 28 of the Gemini map's 1,262. Everywhere else the priority rule's only action is "the same cell next: the lower priority waits this step", and a pair that stays stuck is separated by L3 after `l3_stuck_steps`. PIBT's loser never waits in place: it is pushed to any free neighbouring cell in the same timestep, and whoever is there is pushed in turn. The idea is the same; on an open grid the action is not. Whether this is where the time goes is for the trace to say.
        - **Next:** trace one 32-robot day (where robots stall, for how long, and which rule holds them). Design nothing until that trace says where the time goes.
      - **Braking off, 19 October (Rohit's AWS, his edited kiva_eval.py: `--braking off`, which also turns off bubbles by kind; safety on; ladder6 at floor 3.0; seed 0):**

        | Goals per timestep | 32 robots | 64 | 128 |
        | --- | --- | --- | --- |
        | ladder4 | **0.281** (L3 42) | **0.369** (L3 191, 1 collision) | 0.195 (L3 710, 3 collisions) |
        | ladder6, floor 3.0 | 0.250 (L3 19) | 0.225 (L3 393) | 0.094 (L3 908) |
        | RULES | 0.209 (L3 409) | 0.195 (L3 688) | – |
        | PIBT (his edit) | 0.932 | 1.535 | – |
        | PIBT_SAFE | 0.049 | 0.047 | – |

        - **The bubble costs about half here:** ladder4 at 32 robots went from 0.129 (braking on) to 0.281; RULES from 0.168 to 0.209. Without it, ladder4 also collided (1 at 64 robots, 3 at 128).
        - **Still far from PIBT:** ladder4 reaches about 30% of PIBT at 32 robots and 24% at 64.
        - **The network beats our own rules on this map:** ladder4 against RULES, 0.281 / 0.209 at 32 and 0.369 / 0.195 at 64, with about 10× and 3.5× fewer L3 fires. ladder4 improves from 32 to 64 robots; RULES does not.
        - ladder6 ran at floor 3.0, not the chosen 2.0, and two things changed at once against the first run, so its numbers are not attributed.
        - **PIBT_SAFE cannot be cited: it freezes.** It forbids entering any cell occupied at the start of the step, which also removes PIBT's push. Checked in the sandbox (32 robots, synthetic tasks): 20 goals in the first 64 timesteps, then 1 in the next 448. Two robots facing each other wait for each other's cells forever. A fair spaced PIBT must keep the push.
        - **The edit also lowered plain PIBT by 7–9%** (477 against 515 goals at 32 robots, 786 against 859 at 64): its new `else: continue` stops a robot from following into a cell whose occupant already moved away this timestep. Use the original version for anything published, or label this one.
      - **GRAND (Gaber, Alharbi, Zardini and Gammelli, arXiv 2512.03194), read 19 October.**
        - Its setting: a discrete four-connected grid with no physical units, robot footprint or safety distance. Safety constraints are named as future work.
        - It is a scheduler, not a path planner:
          - a GNN trained with SAC sets how many free robots each region should hold (regions from a shortest-path Voronoi partition);
          - a min-cost flow moves idle robots between regions to match;
          - a bipartite matching assigns tasks inside each region.
        - It does not steer orders away from dense areas; it moves idle robots toward where work will be, with congestion as one input.
        - With 200 agents it beat the 2024 League of Robot Runners winner by about 10% (with reassignment) and cut total conflicts by about 20%.
        - It maps onto FLOWRRA_FEDERATED's design: regions are holons, guidance and rebalancing are the federation layer, and local matching is each holon's dispatcher. That is future work, a design and not a result.
      - **A fair comparison with the state of the art would need:** the benchmark's own map file, a lifelong mode (a new goal on arrival, no book), one cell per timestep, throughput over a fixed horizon at 60–200+ agents, and the open-source solvers (PIBT, RHCR) run on the same tasks. Held as a later item.
    - **Next (deploy only, no training): a floor sweep on clean days.** ladder6 and ladder7 with the route off at scale 1, 2 and 3, paired with clean10's ladder4 and RULES rows. Clean replays are exact: clean9, clean10 and clean11 give identical ladder4 and RULES rows. If some floor reaches ladder4's lead time with the safety gate passed, full follows at that floor, all three arms in one invocation.
    **Built: the order-leak fix (`keyed_draws`), 18 October.**
    - **What leaked, found by reading the code:**
      - single failures drew from Python's shared `random`, interleaved with exploration and anything else that draws;
      - waves drew from their own stream in sequence, and how many draws came before depended on how many steps the wave was eligible, which depends on the policy's progress;
      - people and debris drew from the field's stream in sequence, and how many draws came before depended on where robots stood.
      - deploy_eval already pinned the episode counters and re-seeded every arm, so arms started from the same state. They drifted apart as soon as their robots did.
      - Victims also depended on the order fleets were listed in.
    - **The fix:** config `keyed_draws.seed`.
      - `None` (the default, and training): every draw exactly as before.
      - An int: each draw is keyed by (seed, what is drawn, the step, and the robot, debris spot or person) and hashed with crc32, which is the same in every process (Python's own `hash()` is salted per process).
      - Victims are the candidates with the lowest key that step, so a robot that is a candidate in two arms is hit in both, whatever the list order. If one victim is not a candidate in another arm, the rest of the wave is still hit there.
    - **Files:**
      - `core_warehouse.py`: `_keyed_rng`, `_keyed_rank`; single failures and waves use them when keyed; the keyed seed is passed to the obstacle field.
      - `obstacles_warehouse.py`: `_r()` returns the field's own stream when off (so the sequence is the old one exactly) or a keyed stream when on, at all seven draw sites (initial placement, wander moves, debris spawn and cell, a person's entry, side step and exit).
      - `deploy_eval.py`: `--keyed-draws on|off`, on by default, setting the seed to the instance seed for every arm, printed in the header. Use off to reproduce CSVs written before the fix.
      - `config_warehouse.py`: `keyed_draws: {"seed": None}`, for documentation only. The code defaults to None without it, so it was left out of this zip, to avoid resetting Rohit's `scale_end`.
    - **`test_keyed_draws.py`** (new, 11 checks): off is unchanged for single failures, waves and people; on, a victim ignores the history of draws and the list order; off, the same change of history changes the victim (the leak was real); a wave picks the same victims whatever came before; a missing victim spares only itself; seeds differ and repeat; the keyed roll fires at the set rate (0.305 against 0.3 over 4,000 steps); people start and walk the same, and debris appears at the same steps and cells, whatever the field's own stream did. ALL PASS in the sandbox, with a stub in place of the network (no torch here). With the same stub, test_failure_waves, test_obstacles_cleanup and test_stream_failures still pass; test_obstacles and test_obstacles_integration need torch. **Rohit's machine, 18 October: test_keyed_draws, test_failure_waves, test_obstacles, test_obstacles_cleanup, test_stream_failures and test_obstacles_integration all ALL PASS.** The two-process check (ladder4 and RULES, small map, 25 robots, seed 0, 400 steps, failures and people on, run in two separate processes) came back **IDENTICAL**, so keyed days repeat exactly across processes. The sweep was started alongside it on AWS. (The first sweep, on his local Fedora machine, was killed by the out-of-memory killer at about 4 of 36 days.)
    - **Consequence:** full-condition rows written from now on are new days. They will not match full8–full11. Every arm of a comparison must be run again together, which every graduation already does in one invocation.
    3. **The order-leak fix, before run 10's frozen tests.** Built 18 October (above). Seed failure, debris and people draws from the episode, the step and the robot (or the debris spot), so every arm meets the same events on the same day. Then check that the ladder4 and RULES replays still match among themselves; they will no longer match full8's rows, because the days change.
    4. **Run 10's graduation:** clean and full, all three arms in one invocation each.
       - ladder6 `:noroute` against ladder4 (route on) and RULES, by 3 of 5 plus the safety gate.
       - **Set `training.learner.router.scale_end` to 1.0 in the config before deploying.** deploy_eval reads the floor from there, and run 10's design keeps it.
       - The `--check` repeat on the first instance.
       - Also a 15-robot clean test.
    5. **If the small map still lags:** cause 2, charging a robot for the waiting it causes others (a reward change), designed and dry-run like the others.
    6. **Carried:**
       - trace the worst near-miss day (small map, 40 robots, seed 4);
       - the no-work-exit throughput refinement;
       - the capacity test with the top-up book;
       - item 5's deploy-time conditions (realistic failures, technicians, batteries);
       - the destination and boundary planner ideas;
       - a separate `start_off` router option, if a later run wants no warm-up.

44. **The final deploy_eval: the MAPF baselines on our own floor (19 October).**
    - **Why.** The Kiva lifelong comparison set PIBT, under the usual MAPF rules, against a fleet inside a safety boundary. Both treat robots as points on the graph, but PIBT's may enter a cell the moment it is vacated and move nose to tail with no gap; ours keep 1 hop at least, brake from 1.25 hops when following and 1.75 meeting, and move half a hop per step (Rohit: "don't compare apples and oranges"). And baselines_mapf's rolling-horizon runner lives in its own world: no gap, no braking, no people, its own failures, no book and no docks.
    - **Decided with Rohit:**
      - **Arms:** ladder4 (route on, the champion), ladder6 `:noroute`, ladder7 `:noroute`, RULES; the baselines_mapf variants used in earlier articles (RHCR-PP, RHCR-PIBT, RHCR-PIBT+naive) and per-step PIBT.
      - **Config:** the training config as it stands (next_state_edge_features on, which only changes how training stores transitions), assignment greedy, router floor 2.0 for both route-off networks (one floor per process).
      - **Maps:** one small, one medium and one large from the 3D MAPF warehouse dataset (Wang et al., ICAPS 2024), with books of 100; Gemini's Kiva map as it is, named "a Kiva-style layout made for this project". Checked against RHCR's kiva map (Li et al., AAAI 2021) on 20 October: the same 33×46 grid and aisle rhythm (a shelf row every 4 rows, blocks 10 shelves wide), but two banks of shelves per row instead of three, and 12 blocked workstations at the edges where RHCR has open home cells. 1,262 open cells against 1,278: 112 open only in RHCR's, 96 only in ours. The lifelong Kiva tests (the ×3.1) ran on RHCR's map itself, kiva_eval's default.
      - **Fleets** 20, 40, 60 and 80; **seeds** 0–7; clean and full, keyed days.
    - **Design: same floor, different brains** (`mapf_drivers.py`). The planners are drivers inside the real simulator, where RULES and the networks sit.
      - **The world, the same for every arm:** map, book, greedy dispatcher, docks, rescue handover, failures and people on keyed days, half a hop per step.
      - **Physical safety, on for every arm:** gap and intent guard, braking bubbles, direction-aware braking, never into a robot that stays put, people a hard no-go, collision recovery (Tier-3 yields and tabu moves act only after a collision).
      - **Our brain, off for a baseline:** route by default, corridor entry, priority, the conflict ladder, and the livelock escape (after 40 steps without getting closer, the env drives a robot down its shortest path for 20 steps; for a baseline that would put our brain over its plan).
      - **PIBT, per step:** plan_pibt's rule at every step for the robots standing on a node; heuristic the robot's own distance map from the env; priority +1 per timestep away from the goal, reset when the goal changes.
      - **RHCR-PP / RHCR-PIBT:** baselines_mapf's windowed planners unchanged, window 20, replan every 5 timesteps (benchmark_all's defaults) and whenever the floor changes or a robot is found off its plan. A planned wait is kept while a robot is on time; a late robot skips waits it has missed.
      - **What the planners see:** the whole floor (centralised; ours see their vision range). People, debris and robots that will not move are blocked, except a rescuer's own pickup. Planning time is reported, never charged.
      - **A call made, flagged:** dead robots' cells are blocked for the planners, although our world lets a robot pass through a dead one (only people are a hard no-go). No real planner plans through a broken robot. For RHCR-PIBT, a pickup cell is open to everyone (plan_pibt takes one graph).
    - **Built:**
      - `mapf_drivers.py`;
      - `deploy_eval.py`: baseline arm names, our brain off while a baseline's env is built, `--rhcr-window` and `--rhcr-replan-every`, and the header prints assignment, router floor, next_state_edge_features and the switches turned off;
      - new columns `arm_kind`, `planner_ms_per_step`, `planner_replans` and `planner_offplan`, with L3 NaN for a baseline (no ladder);
      - `test_mapf_drivers.py`: a head-on meeting and a follower, every planner, with the gap checked.
    - **Decided with Rohit, later on 19 October:**
      - **Medium map:** `25_10_5_10_5_2`, 10,700 nodes over 10 floors of 890 cells, joined by lift shafts.
      - **The three benchmark maps:** books of 100, released over 500 steps, 1,000 steps a day.
      - **Kiva:** a book of 200 (denser, harder for us, and it covers the 150 numbers).
      - **The naive rescue may send a robot that is carrying an order,** as ours may. So RHCR-PIBT+naive differs from ours only in having no re-dispatch.
      - **The pickup dwell is our floor's own** (4 steps standing on the pickup) for every arm. Planners may not push a rescuer while it transfers (fixed in RHCR, which first allowed it).
    - **test_mapf_drivers.py: ALL PASS** after one fix in the RHCR driver.
      - The bug: robots between nodes were planned from the node they were heading for, so two robots closing on one node from opposite sides were both planned as standing on it, and the guard held them head-on forever. RHCR-PIBT's head-on test never got home.
      - The fix: plan a robot between nodes from the nearer node, never two robots from one, and let it reach a node before turning.
      - **Results:** head-on, home in 48 / 16 / 25 steps for PIBT / RHCR-PP / RHCR-PIBT. Following, 30–32 steps against 10 for one robot alone. No collisions; the gap never under 1 hop (0.999999 is float rounding).
    - **Found by the follower test: two flaws in our world that slow our own robots. Magnitude unknown; to be measured before anything is claimed.**
      1. **The convoy crawl.** In `_kind_brake`, `d <= min_gap` gives the floor (10% speed) before the bubble's far end is checked. A leader's bubble is `bubble_other` = min_gap = 1.0 ("no ramp"), so a leader exactly 1 hop ahead of its follower is braked to 10% as well. The gap then never opens, and the pair crawls at 0.05 hops a step: 20 steps for one hop in the test. The guard parks a catching-up follower at exactly min_gap, so such pairs may be common. Float noise decides 0.999999 (crawl) against 1.000001 (full speed).
      2. **The float stall.** After braked steps, a robot can stop 0.000001 short of a node (twenty 0.05 steps sum to 0.999999 in float32). Our "same cell next" rule then reads that node as its next cell, the robot behind claims the same cell and wins on priority, and "never into a robot that stays put" makes the one behind wait for it. **Reproduced under RULES:** `claim_waits` and `stopped_waits` +1 each per step for 40 steps, until the livelock escape broke it. It hits RULES and the networks (our rules are on), not the baselines (our rules off).
      - **Proposed fixes, switchable so earlier runs reproduce:**
        - snap a robot onto a node when it is within 0.0001 hops of it;
        - a bubble that ends at min_gap does not brake a pair at min_gap or more.
      - **To measure first:** an A/B on lifelong Kiva, RULES, 32 robots, the current world against the fixed one.
    - **Found reading the rescue code:** in our world a stranded order goes back to the book. `_book_order_waiting` counts any released order nobody holds, so any robot can deliver it without fetching the load from the dead robot, and the pickup is only one of two routes. So ours and +naive would differ by almost nothing. Options for Rohit:
      - (A) make it physical for every arm: a stranded order can be delivered only after its load is fetched (his "handover, not reissue", item 5);
      - (B) keep the world as is and drop +naive.
    - **Decided with Rohit (19 October, 13:02):** option A; fix both world flaws; go on the two runs.
    - **Built:**
      - `safety.node_snap_eps` (node_warehouse.SNAP_EPS, applied in `reach_position`, so the move, the mask, the gradient and the guard agree);
      - `safety.bubble_floor_fix` (`_kind_brake` checks the far end first, with 1e-4 tolerance);
      - `errors.orphans_need_pickup` (orders in an open pickup, and orders lost by the naive rescue, are reserved for everyone and do not count as waiting in the book);
      - `errors.naive_rescue` (a dead fetcher's pickup is abandoned and never re-dispatched; a robot that dies carrying a rescued load gets no pickup). New counts: `orders_lost_naive`, `orders_pickups_abandoned`.
      - **All off in the config, so training and old CSVs reproduce.** `deploy_eval --world-fixes on|off` and `--orphans pickup|book` (defaults `on` and `pickup`, for every arm; the header prints them); `kiva_eval --world-fixes`. The `+naive` arms set `naive_rescue` while their env is built.
    - **`test_world_fixes.py`: ALL PASS.** The same 2-robot day under RULES:
      - old world: the leader moves 0.5 hops in the first 10 steps, and the pair is home at steps 71 and 72, after 23 livelock overrides;
      - fixed: 5.0 hops in 10 steps, home at steps 10 and 11 (10 is one robot alone), no livelock override, no collision, closest 1.15 hops.
      - The pickup and naive rules check out both ways.
      - With every switch off, test_keyed_draws, test_failure_waves, test_obstacles_cleanup, test_stream_failures, test_intent_guard and test_mapf_drivers still pass. test_corridor_rules needs the real network (run on AWS).
    - **The A/B on lifelong Kiva** (sandbox, synthetic POGEMA-style tasks, 32 robots, 512 timesteps, RULES):

      | | Goals | Per timestep | L3 fires |
      | --- | --- | --- | --- |
      | old world | 95 | 0.186 | 453 |
      | fixed | 107 | 0.209 | 308 |
      | PIBT (outside the env) | 535 | 1.045 | – |

      - **The old world reproduced the earlier sandbox baseline exactly** (0.186, 453).
      - **The fixes are real but small here:** +13% goals and 32% fewer L3 fires. They are not the main cause of the lifelong gap; RULES still reaches 20% of PIBT.
    - **Integration days** (sandbox, Gemini's Kiva map, 20 robots, book of 40 over 200 steps, world fixes on, orphans pickup; working checks, not results):
      - **Clean, seed 0, 400 steps:**

        | Arm | Delivered | Lead time | Guard stops | Planning |
        | --- | --- | --- | --- | --- |
        | RULES | 36/40 | 44.8 | 415 | L3 15 |
        | PIBT | 40/40 | 40.8 | 0 | 0.27 ms/step |
        | RHCR-PP | 40/40 | 39.7 | 54 | 15 ms/step, 76 replans |
        | RHCR-PIBT | 39/40 | 42.7 | 262 | 2.9 ms/step, 92 replans |

        No collisions.
      - **Full, seeds 0–1, 500 steps, with RHCR-PIBT+naive:**
        - every arm met the same failures (12 on seed 0) and completed handovers, with no collisions;
        - naive lost 5 orders and abandoned 1 pickup on seed 0;
        - on seed 1, where no rescuer died, naive was identical to RHCR-PIBT, as it must be.
        - The summary's crash at the end is scipy meeting the sandbox's fake torch.
    - **Fixed after the full day: PIBT's priority reset.** The driver reset a robot's priority whenever its goal changed; PIBT resets it only when the goal is reached. A rescuer diverted mid-route lost its priority. Now a change right after arrival resets it, and a mid-route change keeps it. PIBT's handovers on seed 0 went from 4 to 8, and its stranded orders from 9 to 4. test_mapf_drivers still ALL PASS.
    - **Rohit's claim, tested (19 October, 13:30): the closeness rules are the main cost.** He had said it repeatedly (warning zones, approach, how close two robots may be), and the evidence already leaned his way (braking off: ladder4 0.129 to 0.281). The test: `kiva_eval` arms `FLOOR-<baseline>`, a MAPF baseline driving on our floor (our half-hop moves, gap, bubbles and guard; our brain off). Lifelong Kiva, sandbox synthetic tasks, 32 robots, 512 timesteps, world fixes on, seed 0:

      | | Goals | Per timestep | Guard stops |
      | --- | --- | --- | --- |
      | PIBT, usual MAPF rules (outside the env) | 535 | 1.045 | – |
      | **PIBT on our floor** | **170** | **0.332** | 0 |
      | RULES on our floor | 107 | 0.209 | – |
      | RHCR-PP on our floor | 42 | 0.082 | 8,725 |

      - **The floor costs PIBT two thirds of its throughput (535 to 170, ×3.1).** Our coordination costs a further ×1.6 (PIBT 170 against RULES 107 on the same floor). On a log scale, about 70% of the 5× gap is the closeness rules any planner pays, and about 30% is ours.
      - **RHCR-PP suffers most:** its plans assume the usual MAPF rules on exact timings, our guard stopped them 8,725 times. Read it as "plans with exact timings break inside a safety boundary", not as RHCR being worse than RULES in general (one day, our adaptation of it).
      - **Caveats:** one seed, synthetic tasks, 32 robots, RULES not the networks.
      - **For the article and AWS:** run `FLOOR-PIBT` (and `FLOOR-RHCR-PP`) on the real POGEMA instances beside ladder4, ladder6, ladder7 and RULES. That is the like-for-like comparison. PIBT's 1.045 is the figure under the usual MAPF rules, a different problem.
      - **What to learn from PIBT on our floor:** it gets 1.6× RULES under identical physics, with no guard stops. Its loser steps aside at once; ours waits in place outside corridors.
    - **AWS, 19 October: all six tests ALL PASS with the real network** (test_world_fixes, test_mapf_drivers, test_corridor_rules, test_keyed_draws, test_intent_guard, test_obstacles_integration).
    - **New scenario set `all_scens_final` (Rohit, 19 October).**
      - **Needed:** the held-out files have 60 rows, so every 80-robot day would be skipped.
      - **Book mode uses the scenario goals:** each robot's first order is its scenario goal, and the rest of the book comes from the goal bank. With 80 robots and a book of 100, 80 orders come from the scenario file.
      - **Settings,** matched to the old set (measured on its medium-map files: hop floors spread about 4–40 and ceilings about 15–120 across seeds, goal bank 400): `--agents 200 --seeds 8 --seed-offset 9000 --min-hops-range 4,40 --max-hops-range 15,120`.
      - Offset 9000 keeps the files apart from earlier sets. deploy_eval's seeds 0–7 read Seed9000–9007 in order, short journeys to long.
      - **Kiva:** `generate_scenarios.py --goal-bank-file` (new) reuses Gemini's 344 shelf cells as the bank. Tested in the sandbox: the bank is identical, 200 rows per file, every goal on the bank, starts and goals distinct.
      - **The replay check still reads the old folder** (all_scens_v2), to reproduce clean10.
    - **Replay check, AWS, 19 October: IDENTICAL.** `--world-fixes off --orphans book`, small map, 25 robots, seed 0, clean: ladder4 and RULES matched clean10 on every count (ladder4: 80 delivered, lead time 64.55, L3 6, guard stops 182; RULES: 80, 47.025, L3 2, 108; no collisions). With every switch off, the new code is the old world exactly.
    - **Assignment (found in the timing header, 19 October):** Rohit's AWS config says `assignment.mode: hungarian`, so every earlier deploy run (clean10 included) matched goals optimally at spawn. Our design is greedy (Rohit). New `deploy_eval --assignment config|greedy|hungarian` (default `config`, which is what the replay reproduced). It acts only at spawn; every later pick was always greedy (nearest open order, reserved).
      - **Decided with Rohit: keep Hungarian.** Every earlier result used it (clean10, full12, the Kiva book runs, the graduations, the replay), and the networks were trained with it. It is the same for every arm.
      - It matters in book runs: each robot's first order is its scenario goal, matched at spawn, so with 80 robots, 80 of 100 orders go through that match.
      - The final run passes `--assignment hungarian` explicitly, so it is on the command line and in the header.
      - The dispatcher, as the article describes it: one optimal match for the orders on the floor at the start of the day, then nearest open order with reservations.
    - **Timing check (AWS, 19 October): all eight arms on the heaviest day** (Kiva, 80 robots, full, 1,500 steps, seed 0; 3.0 CPU-hours in all). One day: a first look, not a result.

      | Arm | Time (s) | Delivered /200 | Lead time (delivered only) | Guard stops | Near misses with people | Rescues | Robots on floor |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | ladder4 | 1,522 | 187 | 220 | 5,660 | 112 | 14/14 | 55 |
      | ladder6 | 1,190 | **191** | 185 | 5,031 | 203 | 14/14 | 53 |
      | ladder7 | 2,361 | 181 | 273 | 10,044 | 198 | 14/14 | 76 |
      | RULES | 837 | 176 | 176 | 3,290 | 117 | 14/14 | 54 |
      | PIBT | 804 | 162 | 48 | **0** | **21** | 13/14 | 45 |
      | RHCR-PP | 1,669 | 134 | 89 | 24,665 | 78 | 11/14 | 77 |
      | RHCR-PIBT | 1,262 | 103 | 101 | 28,419 | 81 | 10/14 | 79 |
      | RHCR-PIBT+naive | 1,262 | 103 | 101 | 29,175 | 78 | 9/14, 1 lost | 79 |

      - No collisions in any arm.
      - **Lead time counts delivered orders only,** so an arm that delivers less can look faster (PIBT 48). Scoring must count undelivered orders, for example as waiting until the end of the day.
      - **PIBT on our floor:** no guard stops and the fewest near misses with people. But its robots waited about 149 steps at the docks to re-enter (45 on the floor on average), and it delivered 162.
      - **RHCR jams:** 25,000–28,000 guard stops, almost every robot on the floor and unfinished at the end. Its plans assume the usual MAPF rules on exact timings (following nose to tail), which our safety layer will not execute. It may be partly our adaptation; to decide before the run.
      - **Planning time:** RHCR-PP 99 ms a step, RHCR-PIBT 8.5, PIBT 1.0.
    - **Decided with Rohit: RHCR-PP gets a one-step buffer (k-robust, k = 1).** A cell stays reserved for one timestep after a robot leaves it, so its plans never follow nose to tail and convoys keep one empty cell, a spacing our floor drives at full speed.
      - PIBT keeps the standard rule (it follows at once). It re-decides every step, so our braking slows it without breaking anything; its push needs following, so a buffer would freeze it (as PIBT_SAFE did).
      - RHCR-PIBT stays as in the earlier articles, for the same reason.
      - Rows record `planner_robust_k`.
      - **Checks:** test_mapf_drivers ALL PASS (the follower day: 11 and 12 steps, against 30 and 31 without the buffer). Sandbox Kiva, 20 robots, clean: 40/40 delivered, lead time 37.8 against 39.7, guard stops 0 against 54. Whether it helps at 80 robots, the run will show.
    - **Predictions and scoring, fixed before the final run (draft, 19 October; Rohit to add his):**
      - **Measures:**
        - book delivered;
        - lead time with every undelivered order counted as waiting until the end of the day (censored, so an arm cannot look fast by delivering less);
        - collisions;
        - near misses with people (full);
        - guard stops.
        - Planning time is reported, not scored.
      - **Pairing and "firm":** each comparison is paired on map × fleet × seed; firm means Wilcoxon p < 0.05. Each condition is scored on all its days, then by map and by fleet size.
      - **P1.** Collisions zero, or within 1 a day on average, for every arm (the guard is on for all).
      - **P2.** In full at 60–80 robots, ladder4 and ladder6 each deliver firmly more of the book than every baseline arm.
      - **P3.** PIBT has the fewest guard stops and the fewest near misses with people of all arms, in both conditions.
      - **P4.** PIBT beats RULES on censored lead time at 20 and 40 robots, clean.
      - **P5.** The buffer works: RHCR-PP has at least 5× fewer guard stops than RHCR-PIBT and delivers firmly more at 60–80 robots.
      - **P6.** RHCR-PIBT+naive never delivers more than RHCR-PIBT. It is identical on days no rescuer dies, and it is the only arm with lost orders.
      - **Fixed at P1–P6 (Rohit, 17:26):** no prediction for clean between PIBT and our arms. ladder4 is the network plus the route, RULES is something else again, and nobody knows; so it is left open, not guessed.
      - **Launched on AWS:** `final_jobs.py --run --parallel 14`, 240 jobs.
    - **The final run finished (20 October, about 16 hours).** 896 clean and 1,024 full arm-days, every arm on the same 128 days per condition. `final_report.py` writes the tables, figures and prediction score. **The predictions, scored by the rule fixed before the run:**
      - **P1 PASS:** the highest arm mean was 0.016 collisions a day.
      - **P2 FAIL:**
        - ladder4 firmly beats every baseline at 60–80 robots in full (90.5% delivered, against PIBT 85.3, RHCR-PP 89.1 with p 0.003, and RHCR-PIBT 77.5);
        - ladder6 does not beat RHCR-PP (89.0 against 89.1, p 0.54).
      - **P3 PASS:** PIBT had no guard stops and the fewest near misses with people (47.2 a day; next RHCR-PP 52.4).
      - **P4 PASS:** PIBT beats RULES on lead time on clean days, 114.8 against 150.7 at 20 robots (p 0.017) and 95.8 against 141.8 at 40.
      - **P5 FAIL** on its 5× threshold: RHCR-PP had 3.0× (clean) and 3.7× (full) fewer guard stops than RHCR-PIBT. It does deliver firmly more at 60–80 robots (95.4 against 76.7 clean, 89.1 against 77.5 full).
      - **P6 FAIL:** naive delivered more than RHCR-PIBT on 44 of 128 days. On the 29 days with nothing lost or abandoned it was identical, as required. Once a rescuer dies, the days diverge, and re-dispatch is not a clear gain.
      - **First reading (detail to follow with Rohit):**
        - ladder4 delivers the most in both conditions.
        - The planners are faster per order and keep further from people, most clearly on the Kiva-style map.
        - The route-off networks (ladder6, ladder7) trail ladder4, worst on Kiva with 20 robots.
      - **Rohit's reading (20 October): a milestone for an unfinished system.**
        - FLOWRRA, decentralised, is up there with centralised planners, and it is evaluated closer to the real world than they usually are: points kept apart by a safety boundary (a gap, warning zones, braking), failures and people.
        - With more fine-tuning, the network alone (route off) could become an alternative in its own right, and fleets of it can hold their own against planners.
        - Working closer with the distances (a tighter safe gap, where safety allows) should do better still.
        - The detailed article comes next, with every claim stated exactly as measured.
      - **Measured, for that article:**
        - ladder4 (network + route, decentralised) delivers the most in both conditions, and firmly the most at 60–80 robots with failures and people.
        - Collisions near zero for every arm: at most 0.016 a day.
        - The planners' advantages are lead time and distance from people.
        - The network alone trails ladder4.
        - Under the usual MAPF rules, PIBT's throughput is 3.1× what it is inside our safety boundary (lifelong Kiva, entry 44): the cost of the boundary, which any planner pays.
    - **Hours for the final run, from these times.** Time grows with robots on the floor × steps. Kiva about 100 CPU-hours (the full condition about 60 of them). The benchmark maps 50–95 CPU-hours: books end early and idle robots wait off the floor; earlier deploy days ran at 65–90 s per arm-day. **Total about 150–200 CPU-hours: 11–16 hours on 14 processes.**
    - **The queue: `final_jobs.py`** (Rohit: run in parallel).
      - **Why it can split freely:** days are keyed and every arm is re-seeded per day, so an arm meets the same days in any process.
      - **One job:** one arm × one condition × one seed × one map group (the group named `ours` in the code is the three benchmark maps × 4 fleets = 12 rows; kiva: 4 rows). That is 240 jobs (+naive in full only), heaviest first, run N at a time with `OMP_NUM_THREADS=1`.
      - **Resume:** a job is done when its CSV has every row, so `--run` resumes after a stop.
      - `--merge` writes final_clean.csv and final_full.csv.
      - Tested in the sandbox on two toy jobs.
    - **Next, on AWS:**
      - the tests with the real network (done);
      - the replay check (done, IDENTICAL);
      - the replay check (`--world-fixes off --orphans book` must reproduce earlier clean rows exactly);
      - the load check and the timing check;
      - then predictions, the run, scoring and the article.
    - **After the run (21 October): replaying a day, and an open item.**
      - **Replay tools.** `record_day.py` runs one day through `deploy_eval` with the final run's flags and records every step; `render_day.py` turns the recording into an HTML animation and a GIF.
        - `--check` compares a replay with its scored row.
        - Variants (`--book`, `--max-failures`, `--extra-steps`) are labelled as variants in every frame's title.
        - Two fixes found on the way:
          - a longer run must still tell the env the scored length (throughput pressure T and the failure window are computed from it), or the day changes from step 1;
          - replays run on one thread, as the jobs did.
      - **What a replay showed.** The day: ladder4, Kiva-style map, 60 fleets, seed 3, full, a variant with a book of 150 and at most 9 failures.
        - 142 of 150 were delivered.
        - At the end, 8 orders were still marked as assigned, with no fleet carrying them and no pickup open, so they could never be delivered.
      - **It fits the scored data:**
        - no arm finished the book on any full Kiva-style day;
        - on clean Kiva-style days, ladder4 finished it on 30% of days;
        - across ladder4's full Kiva-style days, more failures go with more undelivered orders (correlation 0.34).
      - **Rohit's hypothesis:** the hop limit on reassigning an order, set for our arms. If so, our arms under-scored, and the gap in the hardest conditions would widen once it is fixed.
      - **Open, to test.** Record who holds each claim (a small addition to `record_day.py`), find which arms leave orders stranded, and fix the cause.
        - If confirmed, it goes into the article's Scope and assumptions.
        - Until then the article does not mention it.

45. **Collapsing jams by sampling: `thrml` offline, then the collapse rung (21–22 October).** Design, numbers and caveats in `JAM_COLLAPSE_DESIGN.md`.
    - **Why.** The ladder resolves a jam fleet by fleet. FLOWRRA's founding idea, and the article's §4, call for a joint collapse: hold the ways the fleets could move in superposition, then settle on one consistent plan. That is sampling from an energy-based model, which is what `thrml` does. It is Extropic's open-source library for block Gibbs sampling on discrete models, built for prototyping ahead of their sampling hardware.
    - **The first attempt, dropped.**
      - It used one hop of look-ahead (`jam_energy.py`), compared with where the ladder had each fleet two steps later.
      - That was unfair because of braking. Of 320 fleet moves, 168 had slowed, 96 had stopped, and only 55 had made a full hop.
    - **Rohit's two-boundary design (22 October).**
      - Start: the cluster's cells when it forms.
      - Exit: where each fleet actually came out of it.
      - `thrml` fills in everything between, under the same rules, with the fleets outside the jam clamped to the recording. The plan is compared with what the fleets really drove.
    - **The tools.** All are in the `thrml` folder; they read a day recorded by `record_day.py`.
      - `extract_jams.py`: clusters of warned fleets within 3 hops, chained over steps into jams.
      - `jam_collapse.py`: the two boundaries, windows inside the standing congestion zones, and an exact A* for up to 4 fleets.
      - `jam_thrml.py`: the `thrml` model.
      - `jam_figures.py`: each fleet's distance to its exit over time, recorded against planned.
    - **The day:** ladder4, Kiva-style map, 60 fleets, seed 3, full, replayed on Rohit's laptop. It delivered 183; the scored run delivered 192 (the open item below).
    - **Fixes on the way.**
      - Fleets had to stay parked on their exit until the window ended. Now they leave once out, the same rule in both solvers.
      - Chains start from shortest-path plans.
      - Half the chains keep the rules hard; half pay a finite penalty while hot.
      - A jam with no plan is retried with twice the chains and steps.
      - Zone windows were counted twice; that is fixed.
      - Jams that start with two fleets on one cell (mid-hop rounding) are skipped.
    - **Results (Rohit's laptop, 22 October: 103 windows in 11 minutes on the CPU).**
      - `thrml` found a plan in 98 of the 99 windows that have one. The miss is jam 140: 2 fleets, a 23-tick window.
      - Its plan was the exact best in 90 of 95 small jams.
      - Distance driven: 777 hops in the plans against 1,723 recorded (shortest 707).
      - Driven away from the exit: 35 against 538.
      - Jam by jam: less in 81, about the same in 15, more in 2.
      - Exact search on all jams of up to 4 fleets: 657 hops against 1,546.
      - The big windows, all inside congestion zones:
        - 7 fleets: 34 hops, the shortest possible;
        - 8 fleets: 37 (shortest 35);
        - 4 fleets: 13, the shortest possible;
        - 6 fleets: 53 on the retry (shortest 43). It varies from run to run: 49, 55, 65, or no plan.
    - **What it does not show.**
      - The plans know where each fleet came out and what its neighbours did next.
      - No claim about deliveries.
      - Time is not compared, because of braking.
      - It is one day.
    - **A finding for FLOWRRA itself: the dithering.**
      - Jam 154: two fleets rocked half a hop back and forth for 28 steps, where one hop would have separated them.
      - Jam 351: two fleets advanced and backed off in opposite phase.
      - **Hypothesis (Rohit and Claude):** each fleet reacts on its own to a field that adds everyone's paths up, so neither can settle who goes first. A coupled (joint) density can.
      - To try: hysteresis on yielding, and a coupled future-density field for path awareness.
    - **Part B, the live rung (decided with Rohit, 22 October).**
      1. **Far boundary:** hops to goal now, the network's values later.
      2. **Solvers, in two steps.**
         - B1: exact search up to 4 fleets, `thrml` for 5–8.
         - B2: `thrml` for every jam, padded to one size so JAX compiles once.
      3. **Commitment:** the members follow the plan in lockstep. Tick k+1 starts only when every member stands on its tick-k cell. The plan is cancelled when the cluster grows (re-planned at once), a member fails, leaves, gets a new goal or meets L3, or a tick takes more than 24 steps.
      4. **Braking and the intent guard stay on.**
      5. **Position:** above L1, with L1, L2 and L3 underneath as the fallback.
    - **B1, built.**
      - **`collapse_warehouse.py`, the rung.**
        - A cluster of warned fleets that lasts 2 steps gets one joint plan for the next 8 hops (the path-awareness horizon).
        - The cost is every fleet's hops to goal, added over the ticks: the area under its distance-to-goal line, so waiting costs, backing off costs more, and a zigzag costs most.
        - Fleets outside the cluster, within 10 hops, are clamped: at tick k they may be anywhere from their cell to k hops along their route ahead. Failed, waiting and parked fleets stay put.
        - People and debris are no-go.
        - A cluster with no plan is left to the ladder and tried again after 10 steps.
        - Nothing in `core_warehouse.py` is edited. `install()` wraps four FLOWRRA methods:
          - the constructor;
          - `_ladder_plan_retreats`, so no retrace is ordered for a fleet on a plan;
          - `_ladder_retrace_actions`, so plans are made first and their moves outrank other orders;
          - `get_error_statistics`, which adds the rung's counters and every fleet's distance driven to each CSV row.
      - **`collapse_eval.py`, the paired test.** It runs the same day through `deploy_eval` with the final run's flags, once with the rung off and once on. It writes one CSV and prints the paired table: deliveries, distance driven by all fleets, collisions, near misses, L3, L1 orders, guard stops and lead time.
      - **Smoke test in the sandbox.** The sandbox has no torch and no checkpoints, so this ran on the RULES arm, with no network. Kiva-style map, 60 fleets, seed 0, full, a 300-step test day; one run per variant.
        - **No collisions in any run.**
        - **Rung off:**
          - 92 delivered;
          - L3 73, L1 orders 142, guard stops 902;
          - 29.0 cells driven per delivery;
          - 63 s.
        - **Rung on, exact search plus `thrml` for 5–8 fleets** (16 chains, 300 steps live):
          - 110 delivered;
          - L3 26, L1 74, guard stops 154;
          - 32.8 cells per delivery;
          - 494 s, of which `thrml` took 445 s for 42 plans.
        - **Rung on, exact only** (`--big ladder`):
          - 98 delivered;
          - L3 39, L1 107, guard stops 446;
          - 33.1 cells per delivery;
          - 58 s.
        - **A 120-step check** on seed 1, exact only: 65 delivered off, 71 on.
        - **Bugs found and fixed.**
          - The env's `waiting_nodes` means "held still with a peer close by", not parked, so plans were being cancelled in the middle of jams. Waiting fleets now stay in plans.
          - A tiny cost per hop moved now breaks the tie between waiting and a sideways move.
        - **Reading:** this checks that the mechanism works; it is not a result.
          - Every "on" run had fewer L3, L1 orders and guard stops, and more deliveries.
          - Distance driven per delivery rose 13–14 %, to watch in the real test.
          - Short days, one seed, and the RULES arm, not ladder4.
      - **Decision time, tracked (22 October, Rohit's ask).**
        - **For every day, rung on or off:**
          - the time per simulated step for the whole floor;
          - the fleets' own decision inside it (the network and route, or RULES' driver);
          - the rung's share.
        - **For every planning attempt, found or not:** the time by solver (median, 95th percentile and longest), with `thrml`'s split into JAX compiling and sampling. Its two halves of chains run the same compiled program, so the first call minus the second is the compile.
        - **Offline, `jam_thrml.py` now prints each jam's decision time,** split into building the model, compiling and sampling.
        - **Sandbox, RULES arm, a 150-step test day (2 CPU cores):**
          - a step took 168 ms for the whole floor, 8.6 ms of it the fleets' own decision;
          - exact plans took a median of 2.1 ms (95th percentile 300 ms, longest 544 ms);
          - `thrml` plans took a median of 9.6 s: about 5.0 s compiling and 5.3 s sampling;
          - the rung added 1.5 s per step on average, almost all of it `thrml`.
          - Offline, the 7-fleet window took 44 s: 0.24 s building, 5.6 s compiling and 38 s sampling (32 chains, 600 steps).
        - **Against a real-time budget:**
          - Half a hop (one step) takes about half a second for an AGV at about 1 m/s on 1 m cells.
          - Exact plans fit that budget; `thrml` on a CPU does not.
          - B2's fixed sizes remove the compile.
          - The sampling is the part sampling hardware is meant to speed up. That is untested here, not a claim.
      - **The paired test moves to AWS (22 October).** The laptop froze with four runs at once, so the test runs on the 16-core SageMaker box: 4 seeds × three versions (rung off, rung with `thrml`, exact only), one per core. A 50-step check there: 57 delivered against 47.
      - **A bug found on AWS, and fixed (22 October).**
        - **What happened:** two of the four `thrml` runs died partway through the day with "LLVM compilation error: Cannot allocate memory".
        - **The cause:** every jam has its own shape, so every `thrml` call compiles a new program, and JAX kept every compiled program loaded. Measured in the sandbox, each call added about 350 memory maps and 20–100 MB. After a couple of hundred calls the process reaches the system's limit on memory maps (65,530 by default), and the compiler fails.
        - **The fix, in `jam_thrml.collapse()`:**
          - the weight tables go into the compiled program as arguments, not as constants baked into it;
          - after each call, the compiled programs are dropped (`jax.clear_caches()`).
        - **Measured over 25 calls:** memory maps flat at about 413 (they had grown to 8,145) and memory about 540 MB (it had grown to 880 MB). The plans are unchanged, checked on jams 102, 154 and 283.
        - All four `thrml` runs are rerun with the fix. The off and exact-only runs never call `thrml`, so they are unaffected.
      - **First paired result: the rung with exact search only (AWS, 22 October).** ladder4, Kiva-style map, 60 fleets, full, seeds 0–3: four full days, each with the rung off and on. Clusters of 5 or more were left to the ladder in this version.

        | Per day (mean) | Rung off | Rung on | Days better |
        |---|---|---|---|
        | Orders delivered (book of 200) | 190.0 | 191.5 | 3 of 4 |
        | Mean lead time per order (steps) | 233.0 | 148.2 (−36 %) | 4 of 4 |
        | Near-miss steps with people | 298.8 | 155.8 (−48 %) | 4 of 4 |
        | Cells driven per delivery | 74.4 | 67.0 (−10 %) | |
        | Distance driven, all fleets | 14,127 | 12,831 | 3 of 4 |
        | L1 retrace orders | 836.8 | 478.5 | 4 of 4 |
        | Intent-guard stops | 5,887 | 3,312 | 3 of 4 |
        | L3 fired | 104.2 | 79.0 | 2 of 4 (mixed) |
        | Collisions | 0 | 0 | |

        - **Decision time:**
          - exact plans: median 10.2 ms; 95th percentile 1.0 s (the time limit, hit by searches that found no plan);
          - the rung: 64 ms per step on average, within a step of about 600 ms for the whole floor;
          - runtime per day: about the same either way (937 s off, 898 s on).
        - **The rung's day:** 450 plans, 31 run to their last tick, 325 re-planned as clusters grew, and 170 searches found no plan. 376 cluster attempts were too big for this version.
        - **Reading:**
          - Deliveries barely move: the book is nearly finished either way.
          - Orders arrive much faster, and fleets come about half as close to people, on all four days.
          - Fleets drive about 10 % less per delivery.
        - **Not firm yet:** four days are too few for the agreed test. Even 4 of 4 in one direction gives p = 0.125 two-sided; 8 days could reach p < 0.05.
        - One map, one fleet size; the `thrml` variant is still running.
      - **Eight days: the rung with exact search only (AWS, 22 October).** Seeds 0–7, the same setup as above. Paired Wilcoxon tests on the per-day differences; "firm" means p < 0.05.

        | Per day (mean) | Rung off | Rung on | Days better | p |
        |---|---|---|---|---|
        | Mean lead time per order (steps) | 229.3 | 135.5 (−41 %) | 8 of 8 | **0.008, firm** |
        | Intent-guard stops | 5,976 | 3,162 (−47 %) | 7 of 8 | **0.016, firm** |
        | L1 retrace orders | 818 | 431 | 8 of 8 | 0.008, firm (partly by design: the rung takes over from L1 for planned fleets) |
        | Near-miss steps with people | 246.9 | 145.8 (−41 %) | 7 of 8 | 0.055, not quite |
        | Distance driven, all fleets | 13,530 | 12,529 (−7 %) | 6 of 8 | 0.078 |
        | Cells driven per delivery | 71.9 | 65.6 | | |
        | Orders delivered (book of 200) | 188.1 | 190.9 | 6 of 8 | 0.195 |
        | L3 fired | 118.6 | 74.5 | 5 of 8 | 0.25 |
        | Collisions | 0 | 0 | | |

        - **Decision time:**
          - exact plans: median 9 ms;
          - the rung: 54 ms per step on average;
          - runtime per day: about the same either way (779 s off, 746 s on).
        - **Reading:** a joint plan for clusters of 2–4 fleets gets orders delivered about 40 % sooner, firmly, at no cost in deliveries or collisions, with about half the guard stops. Near misses with people fall on 7 of 8 days, just short of firm.
        - **Scope:** one map, 60 fleets, the full condition, 8 days.
      - **Four days: the rung with `thrml` for clusters of 5–8 (AWS, 22 October).** Seeds 0–3.

        | Per day (mean) | Rung off | Rung on | Days better |
        |---|---|---|---|
        | Mean lead time per order (steps) | 233.0 | 126.5 (−46 %) | 4 of 4 |
        | Distance driven, all fleets | 14,127 | 11,623 (−18 %) | 4 of 4 |
        | Cells driven per delivery | 74.4 | 60.5 | |
        | L3 / L1 / guard stops | 104 / 837 / 5,887 | 52 / 329 / 1,923 | 4 of 4 each |
        | Orders delivered | 190.0 | 192.0 | 3 of 4 |
        | **Near-miss steps with people** | 298.8 | **738.0** | **2 of 4: +662 and +1,494 on seeds 2 and 3** |
        | Runtime per day (s) | 937 | 4,423 (4.7×) | |

        - **Decision time:**
          - `thrml` plans: median 15.5 s (10.5 s compiling, 5.7 s sampling) on one busy CPU core, with up to 16 runs on the box;
          - 255 attempts a day;
          - the rung: 2.6 s per step on average.
        - **Reading:**
          - Handling the bigger clusters adds to the gains in lead time, distance and L3.
          - But on two days fleets were near people far more often: a safety regression, to understand before any claim.
          - A likely cause: plans for big clusters hold fleets in lockstep for many steps, and they avoid only the cells people stand on when the plan is made. People who walk up afterwards find fleets parked beside them.
          - To try: keep plans a hop away from people, and cancel a plan when a person comes within 2 hops of a member.
        - Four days cannot be firm (p ≥ 0.125).
      - **B2 built: `thrml` compiled once per size (22 October).**
        - `collapse_bucketed()` in `jam_thrml.py` pads a jam's choices up to one of 32, 40, 48, 64, 80, 96, 128, 160 or 192 per variable, keeping the exact number of fleets.
        - Each size is compiled once per run and reused by every later jam of that size. The live rung now uses it.
        - Padding fleets as well (up to 4, 6 or 8) was tried first and dropped: padding 5 fleets to 8 tripled the sampling on the CPU, because every fleet's variables are tied to every other fleet's.
        - **Checked (sandbox, 2 CPU cores):**
          - the same plans as before on jams 154, 102 and 10012;
          - a size's second jam compiles in about 0 s, and memory maps stay flat;
          - a 3-fleet jam's decision went from 8.2 s to 2.2 s, and a 7-fleet window's from 42.7 s to 32.4 s.
        - **What is left is sampling.** It grows with the jam (a 7-fleet window is 112 variables with up to 90 choices each). On the laptop GPU it was 0.33 s (median).
      - **The people fix (22 October).**
        - **Why:** the `thrml` days had near misses with people far up on 2 of 4 days.
        - **The mechanism, as the code counts it:** a near miss is a fleet on a cell next to a person, counted per person per step. People wait when a fleet blocks their way. So a plan that holds fleets beside a person's path piles near misses up.
        - **What the rung does now:**
          - a plan keeps a hop away from every person, and from the next 4 cells of each person's path;
          - when a person comes within 2 hops of a fleet on a plan, the plan ends and FLOWRRA's own rules take over;
          - no plan is made for a cluster with a person within 2 hops.
        - New counters: `cancel_people` and `near_people`.
        - **Checks in the sandbox.**
          - A unit test confirms the rung sees each person, the path ahead of them, and the 2-hop zone around them.
          - A 600-step test day (RULES arm, exact only) ran cleanly: 151 plans, no collisions. On that day no planned fleet came within 2 hops of a person, so the new rule never fired. The real test is the AWS rerun.
      - **With the people fix and B2: the paired days again (AWS, 23 October).** Same setup; the rung-off rows are reused (same code path, same numpy 2.5.4).
        - **Eight days, exact search only:**

          | Per day (mean) | Rung off | Rung on | Days better | p |
          |---|---|---|---|---|
          | Orders delivered (of 200) | 188.1 | 191.5 | 7 of 8 (1 equal) | **0.016, firm** |
          | Mean lead time per order (steps) | 229.3 | 132.4 (−42 %) | 8 of 8 | **0.008, firm** |
          | Intent-guard stops | 5,976 | 2,561 (−57 %) | 8 of 8 | **0.008, firm** |
          | Distance driven, all fleets | 13,530 | 12,458 (−8 %) | 7 of 8 | **0.016, firm** |
          | Cells driven per delivery | 71.9 | 65.0 (−10 %) | 7 of 8 | **0.016, firm** |
          | L1 retrace orders | 818 | 395 | 8 of 8 | 0.008 (partly by design) |
          | L3 fired | 118.6 | 66.1 | 5 of 8 | 0.109 |
          | Near-miss steps with people | 246.9 | 309.5 | 5 of 8 | 0.641, mixed |
          | Collisions | 0 | 0 | | |

          - **Near misses with people are driven by one day.** On seed 6 they rose by 1,168. Without that day the mean change is about −95, close to the −101 before the fix. One person held up beside fleets for a long time can add hundreds of steps. To look at: a replay of seed 6.
          - **Decision time:** exact plans took a median of 10 ms, and the rung 58 ms per step. Runtime per day was about the same (779 s off, 689 s on).
          - **The people rule:** 2.2 plans ended because a person came close, and 41 clusters a day were left to FLOWRRA's own rules.
        - **Four days, with `thrml` for clusters of 5–8:**

          | Per day (mean) | Rung off | Rung on | Days better |
          |---|---|---|---|
          | Orders delivered | 190.0 | 193.0 | 4 of 4 |
          | Mean lead time per order (steps) | 233.0 | 116.7 (−50 %) | 4 of 4 |
          | Near-miss steps with people | 298.8 | 85.0 (−72 %) | 4 of 4 |
          | Distance driven, all fleets | 14,127 | 11,489 (−19 %) | 4 of 4 |
          | Cells driven per delivery | 74.4 | 59.5 (−20 %) | 4 of 4 |
          | Intent-guard stops | 5,887 | 2,007 (−66 %) | 4 of 4 |
          | L3 fired | 104.2 | 63.2 | 2 of 4 (two +2) |
          | Collisions | 0 | 0 | |

          - **The people regression is gone:** near misses are now down on all four days, against up sharply on two days before the fix.
          - **B2 worked live:** compiling took a median of 27 ms (it had been 10.5 s). A `thrml` plan took 6.5 s (median), almost all of it sampling on one CPU core.
          - **Runtime per day:** 2.6× the rung-off run, down from 4.7×.
          - **Not firm:** four days can give p = 0.125 at best. Seeds 4–7 would settle it.
        - **Reading:**
          - The exact-only rung now firmly raises deliveries, shortens lead time, cuts driving and halves guard stops, with zero collisions. Its effect on near misses with people is not settled.
          - Adding `thrml` for the big clusters is the strongest version on every measure, now including people, but it is measured on four days only, and it costs decision time.
      - **Eight days with `thrml` for clusters of 5–8 (AWS, 23 October).** Seeds 0–7, with the people fix and B2.

        | Per day (mean) | Rung off | Rung on | Days better | p |
        |---|---|---|---|---|
        | Orders delivered (of 200) | 188.1 | 191.4 | 7 of 8 (1 equal) | **0.016, firm** |
        | Mean lead time per order (steps) | 229.3 | 129.8 (−43 %) | 8 of 8 | **0.008, firm** |
        | Distance driven, all fleets | 13,530 | 11,649 (−14 %) | 8 of 8 | **0.008, firm** |
        | Cells driven per delivery | 71.9 | 60.9 (−15 %) | 8 of 8 | **0.008, firm** |
        | Intent-guard stops | 5,976 | 2,492 (−58 %) | 8 of 8 | **0.008, firm** |
        | L1 retrace orders | 818 | 343 | 8 of 8 | 0.008 (partly by design) |
        | Near-miss steps with people | 246.9 | 114.2 (−54 %) | 6 of 8 | 0.078 |
        | L3 fired | 118.6 | 80.8 | 5 of 8 | 0.180 |
        | Collisions | 0 | 0 | | |

        - **Decision time:**
          - exact plans: median 7.4 ms;
          - `thrml` plans: median 5.7 s on one CPU core, of which 25 ms compiling and 5.0 s sampling;
          - the rung: 1.1 s per step on average. A day took 2.8× as long as with the rung off (779 s against 2,189 s).
        - **Head to head with the exact-only version, on the same eight days** (each "on" day minus the other):
          - `thrml` drove less: −808 cells a day, 6 of 8 days, p = 0.109;
          - near misses with people were fewer: −195 a day, 6 of 8 days, p = 0.055;
          - deliveries and lead time came out the same (p 0.86 and 0.74).
        - **Reading:**
          - Both versions firmly raise deliveries, about halve lead time and guard stops, and cut driving, with zero collisions.
          - Handling the big clusters with `thrml` drives less again and comes closer to people less often, but neither difference is firm.
          - Its cost is decision time: 5 s of sampling per big-cluster plan on a CPU core, against a real-time budget of about half a second.
        - **Scope:** one map (Kiva-style), 60 fleets, the full condition, the ladder4 network.
      - **A plan log: decision time by jam size (23 October).**
        - With the rung on, `collapse_eval.py` now writes `<run>_plans.csv`: one row per planning attempt. Each row gives the day, the step, the solver, the number of fleets, the variables, the most choices per variable, the compiled size, whether a plan was found, and the milliseconds (for `thrml`, split into compiling and sampling).
        - `collapse_eval.py --plans` summarises it by solver and fleets, and for `thrml` by compiled size.
        - Fixed on the way: an attempt that stopped before sampling was picking up the previous jam's timing.
        - **The 8 days again, with the log (AWS `ml.g5.4xlarge`, one CPU core per day, 23 October).** Same seeds, same settings.
          - Every firm result held: 191.6 orders (+3.5, p 0.016), lead time 130.2 (−43 %), cells per delivery 62.8 (−13 %), guard stops 2,549 (−57 %), all firm; zero collisions.
          - Near misses with people moved: 188.0 against 246.9 (−24 %), better on 5 of 8 days, p 0.31 (the first run had −54 %, 6 of 8). Exact search stops at 1 s of wall-clock time, so two runs of the same day can part ways, and near misses are the noisiest measure.
          - **Decision time, 4,307 attempts:**

            | Solver, fleets | Attempts | Found | Median | p95 |
            |---|---|---|---|---|
            | exact, 2 | 1,191 | 85 % | 2 ms | 10 ms |
            | exact, 3 | 761 | 84 % | 22 ms | 244 ms |
            | exact, 4 | 577 | 58 % | 262 ms | 1,016 ms |
            | `thrml`, 5 | 463 | 71 % | 4.2 s | 12.8 s |
            | `thrml`, 6 | 348 | 70 % | 6.4 s | 17.9 s |
            | `thrml`, 7 | 286 | 68 % | 9.3 s | 22.3 s |
            | `thrml`, 8 | 313 | 70 % | 12.1 s | 28.0 s |

          - The 3–4 fleet jams exact search could not settle within its 1 s went to `thrml` too: 368 attempts, found 5 % (3 fleets) and 50 % (4 fleets).
          - `thrml` sampling grows roughly as fleets² × choices^1.5 (within 20 % across 39 sizes): from 1.0 s at 5 fleets × 32 choices to 23 s at 8 × 128.
          - Compiling: 300 first-of-a-size compiles over the 8 days, median 8.6 s each; every other attempt spent 4 ms compiling. (The 25 ms in the first run was the median over all plans.) On a floor, all sizes could be compiled at start-up.
        - **What the wait would cost: an estimate, not a test.** The simulator freezes time while the rung decides, so the gains assume instant decisions. Suppose a cluster stood still while its plan was found, with one step at 0.5 s and one hop at 1 s:
          - exact search: about 37 s of deciding a day at about 4 fleets each, some 150 fleet-seconds, against about 1,000 hops it saves on its own. Worth the stop;
          - `thrml` on one CPU core: 222 attempts a day, failures included, about 9,000–11,000 fleet-seconds (a fifth to a quarter of the floor's 45,000), against the 400–800 hops it adds beyond exact search. More than ten times the saving;
          - break-even is about 0.3–0.5 s per decision, roughly one step. Left out: stopped clusters blocking others, and plans going stale while they wait. Both make the cost larger.
      - **Replay, rung off and on (23 October).**
        - `record_day.py --collapse off|on` records a day with the rung installed as the paired test runs it. With the rung on, each snapshot also holds the joint plans: their fleets, the solver (exact or `thrml`), and the cells left in each plan. Every snapshot now also carries running counts of guard stops, near-miss steps with people and cells driven.
        - `render_pair.py` draws the two days side by side, at the same step: the whole day, a close-up of the rung's first plan (the days are identical up to it, so both panels show the same jam), and a close-up of the first `thrml` plan (by then the days have parted, and the title says so).
        - `collapse_warehouse.py` now notes each plan's solver; nothing else in the rung changed.
        - Checked here on 120 steps of a RULES day (seed 1): both recordings and all three views came out right.
      - **What the seed-1 replay showed (23 October).** The rung-on recording is the scored day (`--check` identical, 193 delivered); so is the off one (187).
        - **When the gain comes.** The two days are level for the first 300 steps; the rung is even behind early on (by up to 5 orders near step 36). From about step 300 it pulls away: +27 by step 500, 191 delivered by step 750. Without the rung the day reaches 187 only at step 1,443.
        - **Near misses with people follow time on the floor.** At step 1,000, 45 fleets are still carrying orders without the rung, against 16 with it. Up to step 800, the rung-on day had more near misses (69 against 46). Its lower total (180 against 382) comes from finishing sooner, not from being safer per step of work.
        - **Most plans are not followed.** 449 plans: the median was held for 2 steps and got through 0 of its 8 hops. About 356 were remade when their cluster changed (any new fleet in a path warning with a member), 66 were cancelled, and 27 ran all 8 hops: 24 from exact search, 3 of the 197 that `thrml` sampled. In practice the rung plans 8 hops ahead, uses about the first move, and plans again: rolling-horizon planning, not one plan followed in lockstep. Each `thrml` decision is used for about 2 steps.
        - **The open question:** is the gain the joint moves, or the side effects of the rung? A fleet on a plan gets no L1 retrace order (one authority; retraces fell 56 % over the 8 days), and plan members mostly stand still while their plan is remade.
      - **The hold test (built 23 October).** `--solver hold`: the same clusters, grace, people rule, one authority and lockstep, but the "plan" is to stay put. No search, no sampling. A fleet is held at most 8 steps in a row, then its own rules get 10 steps (without that cap a cluster that keeps changing would be held for ever). `collapse_eval.py --head A B` compares two versions of the rung on the same days.
        - Reading: if hold scores about the same as the rung, the gain is holding and one authority, and the outreach notes need reframing. If it scores clearly worse, the joint plans earn it. The outreach notes are held until then.
        - Checked here on 150 steps of a RULES day: holds made, completed, remade and released as intended.
        - **Result (AWS, the same 8 days, 23 October).** Hold against no rung, and against the rung (`on3`):

          | Per day (mean) | No rung | Hold | Rung | Hold vs rung |
          |---|---|---|---|---|
          | Orders delivered | 188.1 | 183.5 (p 0.094) | 191.6 | −8.1, worse 8 of 8, **firm** |
          | Mean lead time (steps) | 229.3 | 242.8 (p 0.31) | 130.2 | +113, worse 8 of 8, **firm** |
          | Cells driven per delivery | 71.9 | 64.5 (**firm**) | 62.8 | +1.7, p 0.31 |
          | Intent-guard stops | 5,976 | 4,250 (p 0.11) | 2,549 | +1,701, **firm** |
          | L3 fired | 118.6 | 182.5 (**firm**, worse) | 76.0 | +106.5, **firm** |
          | L1 retrace orders | 818 | 577 (**firm**, by design) | 356 | +220, **firm** |
          | Near-miss steps with people | 246.9 | 255.5 (p 0.31) | 188.0 | +67.5, p 0.74 |

          - Seed 3 went badly with hold: −20 orders, +843 near-miss steps and +203 L3 against no rung.
        - **Reading.**
          - Holding a cluster still and giving it one authority does not give the rung's gains. Deliveries and lead time are firmly worse than with the rung on all 8 days, and no better than with no rung. So the plans' moves earn those gains, even though each plan is used for only about its first move.
          - Holding alone does give most of the cut in driving per delivery (−10 % against the rung's −13 %). So "less driving" cannot be credited to the joint plans.
          - Not settled: what `thrml` adds. Its extra driving saving over exact-only (400–800 cells a day) may be holding too: with `thrml` the big clusters sit on plans that are remade every couple of steps; with exact-only they are left to the ladder.
      - **The exact-plus-hold test (built and run 23 October; result below).** `--big hold`: exact search for 2–4 fleets as in the rung, and hold wherever `thrml` would be asked (5–8 fleets, and 3–4 that exact search could not settle in time). Against the rung (`on3`), it asks whether sampling the big clusters adds anything beyond holding them. One difference: hold always "succeeds", where `thrml` found a plan about 63 % of the time.
        - Checked here on 150 steps of a RULES day: 44 exact plans and 15 holds, as intended. The hold-only test, refactored onto the same code, repeats its earlier short run exactly.
        - **To run it** (AWS, in `~/flowrra_aws`, with the 23 October `collapse_warehouse.py` and `collapse_eval.py`; `grep -c '"hold"' collapse_warehouse.py` should print more than 0; about 15 minutes for all 8 days in parallel):
          ```
          c=0; for s in 0 1 2 3 4 5 6 7; do
            taskset -c $c nohup python collapse_eval.py --arm ladder4 --fleets 60 --seeds $s --cond full --collapse on --big hold --out aws_s${s}_bighold.csv > aws_s${s}_bighold.log 2>&1 &
            c=$((c+1)); done
          pgrep -fc collapse_eval.py          # 8 while running, 0 when done
          python collapse_eval.py --compare "aws_s*_off.csv" "aws_s*_bighold.csv"
          python collapse_eval.py --head "aws_s*_on3.csv" "aws_s*_bighold.csv"
          ```
        - **Reading** (`--head`: A is the rung, B is exact-plus-hold, "diff" is B − A):
          - B about equal to A on deliveries, lead time and driving: sampling the big clusters adds nothing beyond holding them. The Extropic note's "what `thrml` adds" and its wait estimate must then be withdrawn.
          - A firmly better: the sampled moves earn their place in the big clusters, and by how much is the number for the notes.
          - Then stop the AWS instance.
        - **Result (AWS `ml.g5.4xlarge`, the same 8 days, one CPU core per day, 23 October).** `--head`: A = the rung with `thrml` for big clusters (`on3`), B = exact-plus-hold (`bighold`); "diff" is B − A. Paired Wilcoxon, "firm" means p < 0.05.

          | Per day (mean) | A: sampling | B: hold | B − A | Days A better | p |
          |---|---|---|---|---|---|
          | Orders delivered (of 200) | 191.6 | 189.8 | −1.9 | 7 of 8 (1 equal) | **0.016, firm** |
          | Mean lead time per order (steps) | 130.2 | 167.5 | +37.3 | 7 of 8 | **0.039, firm** |
          | Distance driven, all fleets | 12,036 | 11,320 | −715 | B less on 7 of 8 | 0.109 |
          | Cells driven per delivery | 62.8 | 59.7 | −3.1 | B less on 6 of 8 | 0.195 |
          | Near-miss steps with people | 188.0 | 206.4 | +18 | A fewer on 6 of 8 | 0.383 |
          | L3 fired | 76.0 | 100.5 | +24.5 | A fewer on 6 of 8 | 0.148 |
          | Intent-guard stops | 2,549 | 2,689 | +139 | mixed | 0.844 |
          | L1 retrace orders | 356 | 362 | +6 | mixed | 0.844 |
          | Collisions | 0 | 0 | | | |
          | Runtime per day (s) | 2,274 | 922 | −1,352 | | **0.008, firm** |
          | The rung's time per step (ms) | 1,164 | 47 | −1,118 | | **0.008, firm** |

        - **Exact-plus-hold against no rung (`--compare`, same 8 days):** lead time 229.3 → 167.5 (−27 %, p 0.016, firm); distance driven 13,530 → 11,320 (−16 %, firm); cells per delivery 71.9 → 59.7 (firm); guard stops 5,976 → 2,689 (−55 %, firm); L1 retrace orders 818 → 362 (firm, partly by design); deliveries 188.1 → 189.8 (p 0.125); near misses 246.9 → 206.4 (p 0.48) and L3 118.6 → 100.5 (p 0.38) not firm; collisions 0. Per day the rung made about 337 exact plans and 253 holds; 24.5 holds were released by the 8-step cap.
        - **Reading.**
          - **Sampling earns its place against holding.** For the clusters `thrml` would be asked about, sampled plans deliver about 2 more orders a day (+1.9, 7 of 8 days) and cut lead time by about 22 % (−37 steps, 7 of 8 days), both firm. These are the numbers for the outreach notes.
          - **Not separable on anything else.** Near misses with people, L3, guard stops and L1 orders are not firmly different. Hold drove less (−715 cells a day, 7 of 8 days, p 0.109, not firm). So "less driving" is not credited to sampling, as in the hold-only test.
          - **What it costs.** A day took 2.4× as long with sampling (2,274 s against 922 s); the rung's share of a step was 1,164 ms against 47 ms.
          - **Hold costs something, and that matters for the wait.** Hold stands a cluster still for up to 8 steps (4 s), about what a CPU `thrml` decision takes (median 4.2–12.1 s, 8–24 steps), but with no plan at the end of it. So B shows what unproductive standing costs; it is not a wait-then-plan test.
          - **Against leaving big clusters to the ladder, sampling is level.** In the earlier head-to-head, `thrml` against exact-only was the same on deliveries and lead time (p 0.86 and 0.74); exact-only's lead time was 132.4 against hold's 167.5 (separate batches, not paired). Read together: holding a big cluster costs lead time, and sampling avoids that cost; sampling beating the ladder itself on delivery or lead time is not shown. The paired exact-only against exact-plus-hold run would settle it (`--head` on the two sets of CSVs).
          - **Caveats.** One map, 60 fleets, full condition, 8 days. Exact search stops at 1 s of wall-clock time, so reruns of a day can part ways (near misses most). The lead-time p of 0.039 is the marginal one, among many measures looked at. Hold always "succeeds"; `thrml` found a plan about 70 % of the time for 5–8 fleets, and the clusters it missed went back to the ladder.
    - **Next.**
      - ~~Make the plans people-aware, then rerun~~ (done, above).
      - ~~The `thrml` version on seeds 4–7~~ (done, above).
      - The same paired test on other conditions: clean days, 40 and 80 fleets, another map.
      - Sampling time: the GPU, and then the on-board version (one sampler per fleet, the energy vote).
      - A replay of seed 6 with the exact-only rung, to see what held a person up.
      - ~~B2: one fixed size for `thrml`, so it compiles once~~ (done, above).
      - The exact-only rung on more conditions (clean; 80 fleets), if credits allow.
      - The rung against PIBT on the same days, once sampling is fast enough to pay for its wait (Rohit: with sampling working, FLOWRRA could move ahead of PIBT rather than level with it). To test, not yet tested.
      - Later: the collapse built into the core (`core_warehouse.py`), with its energies taken from the density field (`density_warehouse.py`) and the network trained with the collapse in the loop.
      - ~~**First, when resuming:** the exact-plus-hold test~~ (done, above: sampling beats holding on deliveries and lead time, firmly).
      - Pair exact-only against exact-plus-hold on the same 8 days, to see whether holding big clusters is worse than leaving them to the ladder (needs the exact-only CSV names for `--head`).
      - Let plans run further: remake a plan only when a newcomer actually conflicts with it, and see whether the gain grows.
    - **Open item: the laptop replay.** It delivers 183 where the scored run delivered 192.
      - A lead, checked and closed (22 October): `config_warehouse.py` in the repo differed from the sandbox copy in `next_state_edge_features` and the router floor `scale_end`.
      - The laptop's `[deploy]` header shows `scale_end 2.0` and `next_state_edge_features True`, the same as the AWS final run (Rohit checked), so those two settings are not the cause. The sandbox copy was the odd one out.
      - Still open: library versions, or the machine.

**Next.**

1. **`deploy_eval.py` runs** (entries 24 and 26 to 28): all scored.
2. **ladder_run2, cold: done and scored (entry 31).** Safety by construction held for 150 episodes; the network did not learn its job, most likely through value inflation in routed states (ladder_run1's metrics agree: its values stayed under the ceiling). Frozen tests scored (entries 32 and 33): the safety layer took collisions to about zero at no cost in throughput, and the champion ties RULES under full stress. Then, to design and check before any build: ladder_run3, warm from the champion, with two fixes, the honest target (entry 31) and parked robots as obstacles (entry 33), plus lower exploration and possibly fewer failures in training. The heavier-day test (no training needed) can run while that is designed. **Designed in entry 34, with a step-by-step checklist; that checklist is the plan of record.** Next, in this order (Rohit, 12 October): score it; then the frozen test against shake3, the champion and RULES, run as before in two conditions, clean (no failures, no people) and full (both on), with `--safety on` for every arm; then a heavier-day test: the frozen networks on 120 and 160 orders over 400 steps (`deploy_eval --book`), to find where each one breaks. Train on the fixed movement (entry 25) with three changes, then measure on these same seeded days against RULES and champion+route:
   - the approach-aware safety bubble (item 3);
   - a reward that charges approach and collisions, but not presence;
   - route by default during training as well, so the network learns only where robots meet: the one place it still decides, and the place it currently costs collisions.

   The bar to clear is RULES' 0.87 collisions per shift at RULES' delivery.
3. **A smaller, approach-aware safety bubble** (Rohit's proposal, after the runs). Units in the code: one hop is one cell (a whole edge), a robot moves half a hop per step, and run 5's config counts a collision at 0.5 hops. The warning distance is 2 hops today, which is 4 steps of driving.
   - **Closing on a still robot, or following one:** the gap shrinks at most half a hop per step: 0.5 hops to the crash, plus half a hop of braking, gives 1 hop.
   - **Head-on:** the gap shrinks a whole hop per step, so it needs 1.5 hops.
   - **A margin above both.** A robot exactly on the edge of the bubble still drives at full speed, and robots often stand on half-hop marks. At exactly 1.0 (following) or 1.5 (head-on), a pair on the edge crashes in one step, so the minimums are about 1.25 and 1.75 hops. A test should prove it before any run: two robots at every half-hop distance, head-on and following, stepped forward, never crashing. Real robots need more again for reaction time and stopping distance.

   **Decided (12 October): the full, approach-aware version.** About 1.25 hops when the other robot is not closing head-on, and about 1.75 head-on, proved by the test before any training. A fixed 1.25 everywhere would let a head-on pair cross from 1.25 to 0.25 apart in one step. Built before the cold run, together with the reward switch (approach and collisions, not presence) and route by default as a core config switch. Reissue with technician retrieval, the low-battery protocol and failures as a rate per robot-hour (entry 28) were to be built while the cold run trains; on 12 October they were held instead, as deploy-time conditions only (item 5).

   **Known issue, to fix with the per robot-hour failures (found 12 October by the corridor test).** Failure injection draws from Python's shared `random`, so which robot fails, and when, can depend on the order the robots are listed in and on anything else that drew from `random` earlier in the step. In the corridor test a failure wave fired at step 6 in the reversed run only. Training and evaluation always list the robots in the same order, so their results are repeatable (`deploy_eval --check` says IDENTICAL), but a failure is not tied to a robot. The fix is the one used for the tabu draws: seed each draw from the episode seed, the robot and the step.

   The warning distance sets four things at once: when braking starts, the presence and approach charges in the reward, the network's "peer nearby" input, and the ladder's warning set. Change it with a new training run, not under a frozen checkpoint. The cheap first test is route by default with a smaller radius only (1 to 1.5 hops).
4. **Two realistic maps** from the 3D MAPF warehouse generator: an Amazon-style single-floor layout that warehouse teams recognise, and a three-level layout to show the complexity.
5. **Failures, technicians and batteries: held, not built (decided 12 October).** None of this is for training. Training keeps the stress failures, harsher on ourselves on purpose. If time allows, these become deploy-time conditions for a final comparison of the frozen networks against RULES. The network needs nothing new for them: dispatch, exits and re-entries are rules, and a technician on the floor is a person, which it already trains with.
   - **Failures as a rate per robot-hour**, with two settings: realistic (from published figures for warehouse robots, to be looked up) and stress (today's). A shift is 800 steps, about 13 minutes, so at realistic rates a shift sees almost no failures, which is why the stress setting stays. Each draw is seeded from the episode, the robot and the step, which also fixes the known issue in item 3.
   - **A technician for every dead robot.** The people model, sent to a dead robot instead of to debris. Half the time the robot is reset in place and goes back into service; half the time it is towed out at the nearest dock (Rohit: 50-50). Either way its cell is freed. Today a dead robot stays on the floor as an obstacle for the rest of the shift: in episode 19 of run 6, 12 of 25 robots failed on the small map, and the floor jammed (10 recoveries, 10 L3, no collisions).
   - **Human-assisted handover, not reissue** (Rohit). Reissuing an order means re-picking and repacking the goods, which is extra load on production. Instead the technician walks to the dead robot and lifts its load onto the next robot, which FLOWRRA has already sent there. That is all the person does, and no new robot hardware is needed. The measure is the time from failure to the load moving again, against the manual sequence (the technician arrives first, then a replacement is called).
   - **Low battery, and the dying robot** (Rohit). Robots start the shift at mixed charge levels, as on a real floor mid-shift. Below a threshold, a robot finishes its order if the charge left covers the order plus the trip to a dock. Otherwise it hands the order back to the book and heads for the dock now, charges off the floor, and comes back. A failure can be read the same way: a robot that knows its battery is dying heads for the nearest transfer point, where a person or a machine moves its load to a fresh robot.
   - **Real-site practice to compare against:** how warehouses handle robot faults today (who responds, how long it takes, how loads are recovered), so the comparison shows what FLOWRRA automates.

Later: L2 pull-over outside corridors (the "no room" case); anchoring for continual learning (L2-SP or EWC) with a champion/challenger gate, so nothing learned live reaches the floor untested; a sensitivity test of the density field's decay rate; the small-map diagnostic; corridor back-outs chosen by retreat cost; and a calibration test that hops stay hops.

## Code map

Most of the ladder extends code that already exists; the genuinely new parts are the wait-for graph, the retreat-cost choice and the per-fleet trail.

| File | Change | New or extended |
| --- | --- | --- |
| `conflict_warehouse.py` | Build W from the pair verdicts; find cycles per group | Extended |
| `corridor_warehouse.py` | `_new_meetings` generalised from corridors to any 2-cycle; loser chosen by R(S), not priority key; retrace order type alongside pull-over | Extended |
| `core_warehouse.py` | Remove the `and not _held` lock-out; per-fleet trail; rung counters; one authority in warning zones; ladder-context features in each fleet's state and pair relations on the attention connections (3) | Extended |
| `recovery_warehouse.py` | Tiers 1–3 called only at L3; Tier 2 kept as the safety net | Narrowed |
| `agent_warehouse.py` | Recovery head frozen under the ladder (2b, done); Learner head (3) | New heads |
| `config_warehouse.py` | `conflict.ladder`, `conflict.ladder.shadow`, `training.rung_costs`, `training.learner`, all off | New switches |
| `main_runner_warehouse.py` | Log rung counts, flips and retreat hops per episode | Extended |