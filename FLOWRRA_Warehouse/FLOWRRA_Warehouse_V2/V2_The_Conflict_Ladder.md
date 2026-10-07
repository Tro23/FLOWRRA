# FLOWRRA v2 — The Conflict Ladder

Rohit Tamidapati · 5 October 2026 · updated 6 October 2026

> **Status, 7 October 2026.** Steps 1–3 are built and tested. Runs 2 and 3 and the shock benchmark are complete, with their scores in the build log: on the large map, policy + RULES matches RULES alone; on the small map it falls well short. Run 3 showed the Learner's nudge was too loud, so it was made **bounded** and recalibrated (β = 0.1), and next-state edge features earned their place in a paired test. **Run 4 (learner_run2) is under way**, with its predictions written below before its results. The build log at the end records each step and what its runs showed.

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
| 3 | Learner head, silent outside conflict groups, and the ladder-context signal in every fleet's observation, plus pair relations on the attention connections | `training.learner` | `test_learner_head.py` |

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

    No prediction is made that run 4 closes the small-map gap. The Learner speaks only inside conflicts, while the small-map shortfall looks like routing in free traffic (detours of 2–3 times RULES' distance), which the Learner never touches. That gets its own diagnostic, run alongside.

**Next.** Run 4 is judged against the predictions above, paired with runs 2 and 3, using the replicate's noise band. In parallel: a small-map diagnostic separating corridor avoidance from longer exposure. Later: L2 pull-over outside corridors (the "no room" case), corridor back-outs chosen by retreat cost, and a calibration test that hops stay hops across every unit the system uses. Then stream mode.

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
