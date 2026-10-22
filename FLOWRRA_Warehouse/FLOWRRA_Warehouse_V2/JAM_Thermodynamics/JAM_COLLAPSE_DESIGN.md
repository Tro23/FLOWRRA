# FLOWRRA × thrml: collapsing a jam by sampling

*Design and first results, 22 October 2026. Everything runs on Rohit's laptop. The scripts are in this folder.*

## FLOWRRA in brief

FLOWRRA is a decentralised controller for warehouse fleets (AGVs), built at DhaaRn. There is no central planner computing every path. Instead, the set of all fleets on the floor is treated as one **holon**:
- its first task is to keep itself whole (environment A);
- meanwhile it works the floor, the orders, failures and people (environment B).

**It runs on density.** Each fleet reads a local field around itself, a diamond of 231 cells within 5 hops. The field holds:
- which cells are reachable through the floor graph;
- conflict memory: collisions and warnings splatted into space and time, fading on a fast and a slow timescale;
- its own route ahead, and its neighbours' routes ahead, added up.

Where fleets and their intended paths pile up, the field's density rises. A jam is a place where density piles up faster than it dissolves.

The field is read in a Gibbs frame, P ∝ exp(−E/T):
- every fleet carries its local action entropy: how open its choices still are;
- the temperature is the holon's throughput pressure. Many orders and little time give a low T: push, and accept risk. Slack gives a high T: hold, and stay clean.

Ray tracing makes the reading local and measured: each fleet's rays are its own reading of its contribution to the shared distribution.

**A graph network decides, and learns from what actually happened.**
- One graph neural network is shared by all fleets. A gated 3D convolution reads each fleet's field, and graph attention lets neighbours inform each other.
- Heads with one job each forecast the consequences of a move: safety, delivery and efficiency, weighted 6 : 4 : 1. A Learner head carries the judgement of the rules.
- Values are learned by reinforcement learning (SARSA), from the moves the system actually made.
- A ladder of path-aware rules resolves the conflicts that remain: retrace (L1), pull over (L2), and the fundamental tiers (L3) as the safety net.

**Where FLOWRRA stands.** We ran it against reimplemented PIBT and RHCR on the same floor, inside the same safety boundary: four maps, 20–80 fleets, eight seeds, 1,920 arm-days in all.
- Its best network delivered the most of the order book: 98.9 % without disruptions and 87.6 % with failures and people.
- At 60–80 fleets under disruption, it delivered 90.5 % against 89.1 % for the closest planner.
- The planners were faster per order and kept further from people.

The full article is *FLOWRRA: Harmonious dance.*

## Why `thrml` fits

FLOWRRA's founding idea was **wave-function collapse**: hold every way the fleets could move in superposition, then let it settle into one consistent plan. Two parts of that idea are still unbuilt, and both are what `thrml` does.

- **Jams are still resolved fleet by fleet.** Each fleet reads the summed density field on its own, and the ladder untangles the rest by rule.
  - Inside jams, this study finds the fleets drove more than twice what a joint plan needed, mostly dithering back and forth.
  - A plausible reason: separate decisions on a summed field cannot settle who goes first (see *A finding for FLOWRRA itself*). A coupled, joint density can.
- **The full Gibbs field was left unbuilt because of its partition function Z,** which costs compute on every step.
  - A Gibbs sampler never needs Z. It only compares the energies of a few local options at a time.
  - So sampling is the natural way to use the exponential field without having to compute it.

`thrml` is Extropic's open-source library for **block Gibbs sampling on discrete energy-based models**: categorical variables, factors between pairs of them, and blocks updated in turn. A FLOWRRA jam is exactly that kind of model:
- the variables are where each fleet is, at each tick;
- the rules are energies between pairs of variables;
- the cost is one tick of delay at a time;
- FLOWRRA's temperature is the sampler's temperature.

Nothing in the problem had to be bent to fit the tool. The FLOWRRA article (§4) had already planned the collapse as a two-boundary problem: fix where the fleets are and where they leave, and solve between. This prototype builds the collapse half, with the exits taken from the recording in place of a learned boundary.

## A pathway to real deployments

Extropic is developing hardware that aims to make sampling from discrete energy-based models far more energy-efficient. We think warehouse robots could be one of the first real-world places where that pays off.

- **The workload fits.**
  - A jam is small, discrete and local: in this study, 16 to 114 variables, each with tens of possible cells, linked only to nearby fleets.
  - Jams recur all day, on every floor.
  - Each one must be settled in about the time a fleet takes to drive a hop.
- **Robots run on batteries.** On a robot, computing draws on the same battery that drives it, so energy-efficient sampling matters there in a way it does not on a server plugged into the wall.
- **The value shows up in numbers operators already track:** deliveries per hour, distance driven, near misses.
  - Fewer hops driven inside jams means less energy and less wear.
  - If a live collapse raises throughput, the hardware's advantage reaches the operator's figures directly.
- **It is decentralised by design.** FLOWRRA already decides per fleet, and Gibbs sampling is local by nature. Each robot could sample its jam on board and agree with its neighbours by comparing energies (see *An on-board version*). There is no central server in the loop.
- **The path is staged, and each stage is measurable:**
  1. the offline collapse on recorded days with `thrml` (this study: done);
  2. a live collapse rung in the simulator, to answer whether deliveries rise;
  3. the on-board version in simulation, with per-fleet samplers, the energy vote and stale information;
  4. a hardware pilot: a few robots, or a simulator with Extropic's chips in the loop doing the sampling.

**What we cannot claim yet.**
- Only stage 1 is done, and its gain is distance driven inside jams, with boundaries taken from the recording.
- Whether deliveries rise is not yet known.
- How well jams map onto the chips' connectivity and size is not known either.

Those are the questions we would like to work on together.

## Where it stands

- **Built:**
  - find the jams in a recorded day;
  - fix each jam's two boundaries from the recording: where the fleets were when the jam formed, and where each one came out of it;
  - fill in everything between with `thrml`;
  - check `thrml` against an exact search on small jams;
  - compare the plans with what the fleets actually drove, and draw it.
- **The main finding:** inside a jam, a joint plan between the same two boundaries needs roughly **half the driving** the fleets actually did:
  - the exact search on jams of up to 4 fleets: 657 hops against 1,546;
  - `thrml` across 98 jams of 2–8 fleets: 777 hops against 1,723.

  Most of the difference is fleets rocking back and forth in path warnings, which this write-up calls **dithering**.
- **`thrml` itself:**
  - It found a plan in 98 of the 99 jams that have one.
  - On jams of up to 4 fleets, its plan was the exact best in 90 of 95.
  - It found legal plans for all four windows of 4–8 fleets inside the congestion zones, two of them at exactly the shortest distance. The 6-fleet window still varies from run to run.
  - Those big windows are where the exact search can no longer go, so they are where `thrml` earns its place.
- **What it does not show yet:** whether deliveries would rise. Only running the collapse live in the simulator can show that (see *Next steps*).

## How we got here: from one hop to two boundaries

**The first attempt: one hop of look-ahead** (`jam_energy.py`, kept for reference)
- The plan: score each jam fleet's next hop against its goal, and compare that with where the ladder put it two recorded steps later.
- **It was unfair, and we dropped it.** FLOWRRA brakes in warning zones, so two steps later is often not a full hop.
  - Of 320 fleet moves, 168 had slowed, 96 had stopped, and only 55 had made a full hop.
  - Comparing one-hop plans with braking fleets measured the brakes, not the decisions.

**Rohit's design: two boundaries** (the wave-function-collapse idea)
- Fix both ends from the real day, and let the sampler fill in everything between:
  - **start:** every jam fleet's cell when the cluster forms;
  - **exit:** where each fleet actually came out of the cluster.
- The question then becomes: *given where the fleets got out, could they have got there more directly, deciding together?*
- That is a clean comparison of a holon deciding jointly against fleets deciding separately, on the same floor and between the same boundaries.

## The pieces, in plain language

### 1. Recording a day (`record_day.py`)

- Records every fleet at every step: position, state, load, path warning, goal cell, and ladder rung (0 none, 1 retrace or pull over, 3 the fundamental tiers).
- Also records the people and debris.
- The day used here:
  - ladder4, Kiva-style map 46 × 33, 60 fleets, seed 3, full conditions;
  - recorded on Rohit's laptop: 183 deliveries;
  - the scored AWS run delivered 192, so the replay differs from it. That is an open item, not a problem for this study, which only needs *a* real day.

### 2. Finding the jams (`extract_jams.py`)

**What counts as a jam**
- At every step, take the fleets in a path warning.
- Link two of them when they are within 3 hops of each other, counted through the floor graph, so a rack in between means they are not near.
- A linked group of 2 or more fleets is a **cluster**.
- A cluster that persists from step to step, sharing at least one fleet, is one **jam**, from the step it forms to the step it dissolves.

**On this day**
- 395 jams. Most are fleeting: 233 lasted one step and 37 lasted two.
- 144 jams lasted 3 or more steps or made the ladder act.
- 11 jams lasted over 100 steps. These are the **standing congestion zones**: the longest ran 1,352 steps and drew in 49 fleets.

### 3. The two boundaries (`jam_collapse.py`)

**For an ordinary jam (up to 100 steps long)**
- **start:** each fleet's cell at the step the jam forms. A fleet between two cells (mid-hop) counts as in the cell it is driving into.
- **exit:** each fleet's cell at the first step it is more than 3 hops from every other jam fleet.
- A jam is used only if every fleet comes out within 100 steps and none fails or leaves meanwhile.

**For a standing congestion zone**
- A zone never really dissolves, so we take **windows** inside it: at a step where the ladder starts acting on a fleet in the zone, the cluster around that fleet becomes a jam of its own.
- Within a zone, windows are at least 20 steps apart, and there are at most 40 in all. The same cluster at the same step is taken only once.

**Ticks**
- A plan moves in **ticks** of one hop.
- Tick k is matched to recorded step t0 + 2k: two steps per hop at full speed.

**Everything outside the jam stays as it really was**
- Fleets outside the jam are **clamped** to their recorded cells at those steps. In `thrml`'s terms they are fixed nodes the plan must respect.
- People and debris are clamped the same way.
- So the plan is stitched into the real day.

**The window length T**
- T is the longest shortest path from start to exit, plus 8 ticks of slack, capped at 24 ticks.

**Which jams survive**
- Of 166 candidate jams and windows, 103 survive:
  - 79 of 2 fleets, 20 of 3, and one each of 4, 6, 7 and 8 fleets;
  - 11 of the 103 are inside the congestion zones.
- Skipped: in 36 a fleet failed or left the floor; in 27 the fleets did not separate within 100 steps.

### 4. The rules and the cost

Every plan, exact or sampled, keeps these rules:
- **Moving:** each tick a fleet stays put or moves one hop.
- **No sharing:** no two fleets in one cell, whether jam fleets or clamped neighbours.
- **No passing through:** no fleet enters a cell another fleet held one tick earlier.
  - This rules out swaps and passing through each other.
  - It is stricter than FLOWRRA, which lets a fleet follow one hop behind. So it handicaps the plan, not the recorded fleets.
- **No-go cells:** no people, debris or racks.
- **Leaving:** a fleet that reaches its exit may drive on. From then on it holds no cell, just as the recording stops following it once it is out of the cluster. The plan ends when every fleet is at its exit or out.

**The cost:** every tick a fleet has not yet reached its exit costs 1. A plan's cost is the fleets' arrival times added up.

### 5. The exact answer (A* search, in `jam_collapse.py`)

- For up to 4 fleets, an A* search over all the fleets' cells together, tick by tick, finds the plan with the lowest cost.
- It takes about a second for 100 jams.
- This is the referee: before trusting `thrml`, we check that it finds the same plans.

### 6. The collapse with `thrml` (`jam_thrml.py`)

**The superposition**
- There is one variable per fleet per tick: **where the fleet is at that tick**.
- Its choices are the cells the fleet could be in then: reachable from its start in k hops, and able to reach its exit in the ticks left. That is the fleet's **light cone**.
- One more choice, **OUT**, means the fleet has reached its exit and driven on.
- Before sampling, every choice is open: that is the superposition. Sampling collapses it into one plan.

**The rules as energy tables**
- Each rule involves at most two variables, so each one becomes a table of energies:
  - one fleet from tick to tick: stay or move one hop, and OUT only from its exit;
  - two fleets at the same tick: not the same cell;
  - two fleets at consecutive ticks: not into the cell the other just held.
- The cost (1 per tick not yet out) is a table on each single variable.

**How it maps onto `thrml`** (version 0.1.4, Apache-2.0)

| Our piece | `thrml` piece |
|---|---|
| each variable | `CategoricalNode` |
| each rule or cost table | `CategoricalEBMFactor` |
| re-sampling a variable given its neighbours | `CategoricalGibbsConditional` |
| the plan's structure | `FactorSamplingProgram` and `BlockGibbsSpec` |

`thrml`'s codon-optimisation example was the template.

**Block Gibbs**
- Each fleet's variables split into two **blocks**: its even ticks and its odd ticks.
- No two variables in a block touch, so a whole block is re-sampled at once, given everything else.

**Annealing**
- All the energies are multiplied by β, raised from 0.1 to 10 over 600 steps, with 2 sweeps per step.
- Hot (small β), the sampler explores; cold (large β), it settles on the cheapest legal plan.

**32 chains in parallel**
- Half start from shortest-path plans: each fleet drives a shortest path to its exit, after a random wait at its start. Chain 0 waits for nobody.
- The other half start from random cells.

**Hard and soft rules**
- Half the chains keep the rules hard from the start. They polish plans near where they began.
- The other half pay a finite penalty for a broken rule (20 × β). While hot, they can pass through a broken rule on the way out of a tight spot, such as one fleet backing into a side aisle. At β = 10 a break costs 200, so they end up legal too.

**The plan and the retry**
- The plan is the final state with no broken rule and the lowest cost, over all chains.
- A jam with no legal plan is run once more, with twice the chains and twice the steps.

### 7. Measuring

- **Distance driven, in hops.**
  - For the recorded fleets it comes from their raw positions, half-hops included, so a fleet rocking inside one edge counts at its true length.
  - A recorded fleet can start or exit mid-hop, so its distance can be off by up to half a hop at each end. A jam counts as "about the same" when the two distances are within half a hop per fleet.
- **Distance driven away from the exit:** the part of the driving that took a fleet further from where it was going. This is where dithering shows.
- **Not time.** The plans run at full speed; FLOWRRA brakes near other fleets. So the plans' times are shown but never compared.

### 8. Pictures (`jam_figures.py`)

Each jam gets two panels:
- **Left:** the floor around the jam: racks in grey, each fleet's recorded path as a thin line, the plan dashed, a dot at its start and a square at its exit.
- **Right:** each fleet's distance to its exit over time. Dithering shows as a zigzag; the plan runs straight down.

## Results

### Is `thrml` finding the best plans?

**First version, on 73 small jams** (2–3 fleets), with the same window length as the exact plans:
- It found the exact best plan in 70 at the first try.
- With a longer run (64 chains, 1,000 steps) it found 72 of 73.
- In the last one it found a legal plan of cost 11 against the best 9.

**The first full run on the big set** (102 distinct jams and windows) exposed a mistake in my model:
- A fleet that reached its exit had to stay parked on that cell until the window ended. If a clamped neighbour later drove over that cell, no plan was possible.
- Of the 29 jams with no plan: 4 have no legal plan under our rules (the exact search agrees), 16 were this mistake (including the 4-fleet window), and in 9 the sampler missed a plan that exists.
- One 8-fleet window was solved twice by accident, a duplicate since removed. It drove 65 hops in one run and 41 in the other, so the sampler was not settling.

**The fixes** (the current version):

| Fix | What it did |
|---|---|
| Fleets leave at their exit (the OUT choice) | The parking failures are gone. Every jam still without a plan either has none (the exact search agrees) or is a sampler miss. The exact search uses the same rule. |
| Shortest-path starts | 8-fleet window: 37–39 hops across runs (shortest 35), against 65 or 41 before. 7-fleet window: 34 hops, exactly the shortest, against 54. 4-fleet window: 13 hops, the shortest, where before it was blocked. |
| Half the chains with soft rules | Found plans in 4 of the 5 small jams the sampler had missed (jams 275, 276, 280, 283), matching the exact best in 3 of them. |
| Retry with twice the chains and steps | 6-fleet window: 49 hops in one retry-length run (shortest 43). Shorter runs gave 55 and 65 hops, or no plan at all. |

**Tests of the current model on the 46 hardest jams.** These are all the earlier failures, the plans that were not the best, and the 4–8 fleet windows.
- With all chains hard: a legal plan in 39 of 46. In 31 of the 34 that the exact search also solved, it was the exact best.
- The mixed hard-and-soft version was then re-tested on the 16 hardest of them:
  - jam 140 (2 fleets, a 23-tick window) is still missed, even with 64 chains and 1,200 steps;
  - jams 9 and 225 have no legal plan (the exact search agrees);
  - jams 197 and 344 start with a fleet on a cell a neighbour also occupies. That is an artefact of mid-hop rounding, and both methods now skip it.

### The comparison with the recorded day

**Exact search, all jams of up to 4 fleets, current rules** (a plan exists in 96 of 100):

| | Plan | Recorded | Shortest possible |
|---|---|---|---|
| Distance driven (hops) | 657 | 1,546 | 623 |
| Driven away from the exit (hops) | 17 | 489 | — |

- Jam by jam: the plan drove less in 79, about the same in 15, and more in 2.
  - Jam 280: our no-following rule forced a detour, 9 hops against 7.7.
  - Jam 283: the recorded distance is under the shortest path because of half-hop rounding at the ends.
- The plans have everyone out in a median of 4 ticks, against a median of 19 recorded steps, braking included.

**`thrml`, the full run of the current version** (on Rohit's laptop, 22 October: 103 jams in 11 minutes on the CPU):
- A legal plan in 98.
- 2 skipped (jams 197 and 344): a fleet starts on a cell a neighbour also occupies, because of mid-hop rounding.
- 3 without a plan:
  - jams 9 and 225 have no legal plan at all (the exact search agrees);
  - jam 140 is the one real miss: a 2-fleet jam with a 23-tick window, missed even on the longer retry.
- So `thrml` found a plan in 98 of the 99 jams that have one.
- Against the exact search, on the 95 jams of up to 4 fleets that both solved: the same best cost in 90. The other 5, all of 3 fleets (jams 35, 67, 275, 342, 10009), got legal plans that cost a little more.

| Distance driven (hops) | Plan | Recorded | Shortest |
|---|---|---|---|
| All 98 | 777 | 1,723 | 707 |
| 2 fleets (75) | 399 | 984 | 377 |
| 3 fleets (19) | 241 | 466 | 205 |
| 4–8 fleets (4) | 137 | 273 | 125 |
| Inside congestion zones (11) | 184 | 369 | 170 |

- Driven away from the exit: 35 hops against 538.
- Jam by jam: the plan drove less in 81, about the same in 15, and more in 2. Those 2 are jams 280 and 283, the same two as for the exact search, for the same reasons.
- For the record, the first version had found 73 plans: 624 hops against 1,227.

**The big windows** (all inside congestion zones):

| Window | Fleets | Plan (hops) | Recorded | Shortest |
|---|---|---|---|---|
| 10012 | 7 | 34 | 74.8 | 34 |
| 10004 | 8 | 37 (39 in two test runs) | 69.2 | 35 |
| 10008 | 6 | 53, found on the longer retry (other runs: 49, 55, 65, or no plan) | 87.0 | 43 |
| 10028 | 4 | 13 | 41.6 | 13 |

### Two jams, in pictures (`jam_figure_small.png`)

- **Jam 154:** two fleets were one hop from separating. They spent 28 steps rocking half a hop back and forth, 25.5 hops in all. The plan needs 1 hop.
- **Jam 102:** in the plan, one fleet steps one hop aside, lets another pass straight through, then comes back: 8 hops against 29.4 recorded. That is joint coordination that fleets deciding separately cannot do.

Two more from the full run:
- **Jam 350** (3 fleets): 25.8 hops recorded against 7.
- **Jam 351** (2 fleets): 10.6 against 2. For about 8 steps the two fleets move in opposite phase: each time one edges towards its exit, the other backs off, and then they swap. That pattern matters for *A finding for FLOWRRA itself*, below.

`jam_figure_zones.png` shows windows 10012 (7 fleets) and 10004 (8 fleets).

## What this shows, and what it doesn't

**It shows**
- Inside a jam, between the same two boundaries, deciding together needs a little under half the driving: 45 % in the full `thrml` run, 42 % for the exact search on jams of up to 4 fleets.
- The extra driving is mostly dithering, back and forth in path warnings, not long detours. The distance driven away from the exits is 17–35 hops in the plans against 489–538 recorded.
- `thrml` can do this collapse on real FLOWRRA jams. It found a plan in 98 of the 99 jams that have one, the exact best in 90 of 95 small ones, and good legal plans for all four 4–8 fleet windows.

**It does not show**
- **That deliveries would rise.** The plan is stitched into the real day only inside its window; past it, the day would have unfolded differently. Only a live rung in the simulator can show the knock-on effect.
- **A fair fight on knowledge.** The plans know where each fleet came out and what the clamped neighbours did next. The ladder knew neither. This is an oracle boundary: it measures the price of deciding separately, not what a live sampler would achieve.
- **Anything about time,** because of braking (see *Measuring*).
- **Every jam.** Jams that did not separate within 100 steps, or where a fleet failed, are left out: 63 of 166.
- **More than one day.** It is one day, one map and one seed.
- **That `thrml` beats exact search on small jams.** It doesn't: the exact search is faster there. `thrml`'s place is the big windows, and the form of the problem, which is the form Extropic's hardware runs.

**The open question for `thrml`:** reliability on big windows. The 6-fleet window gave 49, 53, 55 and 65 hops across runs, and once no plan. More chains and slower cooling help, at a cost in time. That cost is the one sampling hardware is meant to cut. We have not tested that, but it is the natural question to bring to Extropic.

## A finding for FLOWRRA itself

The dithering is worth fixing in FLOWRRA whether or not `thrml` ever runs live. Jam 154 is the clearest case: two fleets rock half a hop back and forth for 28 steps, where one hop would have separated them.

**A likely cause** (a hypothesis, not tested): each fleet reacts on its own to a density field that adds up everyone's predicted paths.

Take two fleets meeting in a narrow aisle. There are two good futures: A goes first while B waits, or B goes first while A waits.
- **The summed field** shows each fleet only "someone may be coming". Each decides separately, so both may yield, then both go, then both yield again.
- **The coupled (joint) density** of the collapse holds the two futures as separate options. Choosing one fixes both fleets at once: if A goes, B waits.

Jam 351 looks like what the summed version would produce: two fleets advancing and backing off in opposite phase.

**Two things to try in FLOWRRA:**
- **Hysteresis:** once a fleet yields, it keeps yielding for a few steps.
- **A coupled field:** feed path awareness the fleets' future densities from a warm, not fully cooled, joint sample instead of each fleet's own predicted path. That field would already account for who gives way.

Both are ideas, not yet tested.

## An on-board version (ideas, not tested)

Gibbs sampling is local by design, and the model's blocks are already one fleet's even ticks and one fleet's odd ticks. So the collapse could run on the robots themselves, each with its own sampling chip.

**Two ways to split the work:**

1. **Each robot samples only its own blocks.** It re-plans given its neighbours' current plans, passing plans over the network.
   - Neighbours must take turns, using a colouring of who touches whom. Otherwise two fleets that update at once, each reacting to the other's old plan, can flip back and forth: dithering again, inside the sampler.
   - Every sweep needs a round of messages, and a collapse needs hundreds of sweeps.
2. **Every robot holds the whole holon's model** and samples full joint plans locally, from what it senses and is told.
   - Robots may settle on different plans, so they vote: each broadcasts its best plan and its energy, and the holon adopts the lowest-energy legal plan, with ties going to the lowest fleet id. Finding a plan is hard, but checking one is easy.
   - This is the script's "best of 32 chains", with the chains spread over the robots.
   - It needs one round of messages per decision instead of one per sweep, so it is the more practical of the two.

**Who is in the holon.** A fleet joins a robot's model when its light cone overlaps the others' within the window, which the ray-traced density already shows. A fleet arriving from outside the first cluster joins as its density reaches the others. A fleet left out of the winning plan treats that plan's fleets as fixed neighbours, as the offline study clamps the outside fleets.

**Temperature.**
- **Different temperatures for different copies.** This is parallel tempering: hot chains explore, cold chains polish, and they swap states now and then. Robots could hold different temperatures. It would help most on the big windows.
- **Urgency belongs in the cost, not the temperature.** A loaded fleet, or one with a tight order, pays more per tick of delay, so others give way to it. That keeps one shared energy, which the vote needs to be fair.

**Questions for Extropic:**
- How densely can sampling cells be linked? Inside a jam, every fleet's tick is tied to every other fleet's same and next tick.
- How large a model does one chip hold, and could a chip be small and frugal enough to ride on a battery-powered robot?

**Testing it before any hardware:** simulate option 2 in `thrml` on the same jams, with each fleet holding its own chains, the vote, and slightly stale information.

## How to run it

Put the scripts in the same folder as the day file:

```
DAY=day_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz

# 1. Find the jams (seconds)
python extract_jams.py $DAY

# 2. Build the jams and zone windows (seconds)
python jam_collapse.py $DAY jams_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz --min-size 2 --max-size 8 --max-steps 100 --zones 40 --no-exact --out problems_big.json.gz

# 3. The collapse with thrml (11 minutes on Rohit's laptop, CPU; nohup keeps it running)
nohup python jam_thrml.py $DAY problems_big.json.gz --out thrml_big.json.gz > thrml_big.log 2>&1 &
tail -f thrml_big.log

# 4. Pictures (seconds each; any jam number from the log works)
python jam_figures.py $DAY problems_big.json.gz thrml_big.json.gz --jams 154,102 --out jam_figure_small.png
python jam_figures.py $DAY problems_big.json.gz thrml_big.json.gz --jams 10012,10004 --out jam_figure_zones.png

# Optional: the exact search on small jams, and thrml checked against it (about 5 minutes)
python jam_collapse.py $DAY jams_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz --max-size 3
python jam_thrml.py $DAY collapse_exact_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz --horizon exact
```

**`jam_thrml.py` options:**
- `--chains 32 --steps 600` set the sampling budget. More is steadier and slower.
- `--penalty 20` sets the soft chains' rule penalty; `0` makes every chain hard.
- `--no-retry` skips the longer second try.
- `--jams 140,10008` runs only those jams.

## Next steps

1. ~~The full run of the current version on the laptop.~~ Done on 22 October; the numbers are above.
2. **A live rung** (option B). Put the collapse into the simulator as a new rung: when a cluster forms, sample a joint plan for its fleets for the next few ticks. This is the only test of whether deliveries rise. The boundary then has to come from the fleets' goals rather than from a recorded exit.
3. **Reach out to Extropic,** with the figures, these numbers and the honest limits: *"I built this on `thrml` for robot fleets. Here is a real jam from a fleet that ran 1,920 days against PIBT and RHCR, here is the sampled collapse, here is how it compares. Would you be interested in going further?"*
4. **Optional:**
   - try the dithering fix in FLOWRRA;
   - find out why the laptop replay delivers 183 where the scored run delivered 192 (start by comparing the configuration lines at the top of each run).

## Decisions (Rohit, 22 October)

1. **Horizons from 1 to 8 steps**, with 8 matching path awareness, which looks 8 route cells ahead. *Superseded* by the two-boundary design: the window now runs from the jam forming to every fleet being out, up to 24 ticks.
2. **Only the clustered fields.** A jam is a cluster of fleets in path warnings within 3 hops of each other. Free-moving fleets are left alone.
3. **Stitch, then compare.** The jam's fleets are the free variables; nearby fleets keep their recorded positions (clamped); the plan must be legal against them. Stitching holds only inside the window.
4. **People and debris are hard no-go cells,** as in FLOWRRA.
5. **The day:** ladder4, Kiva-style map, 60 fleets, seed 3, full, re-recorded with the scored settings.
6. **Two boundaries** (Rohit's wave-function-collapse design): start where the cluster formed, exit where each fleet really came out. The sampler fills in the rest, and the plan is compared with what the fleets actually drove. This replaced the one-hop comparison, which braking made unfair.
7. **Bigger jams and the congestion zones** (4–8 fleets, longer windows, windows inside the standing zones), and the **space-time picture.**
8. **Fleets leave at their exit, warm starts, and mixed hard and soft chains:** the fixes after the first full run.
9. **The on-board version** to be written up as ideas for the Extropic conversation: separate blocks per fleet, coupled rather than summed.

## Files

| File | What it does |
|---|---|
| `record_day.py` | records a day, every fleet at every step |
| `render_day.py` | a recorded day as an interactive page and a GIF |
| `extract_jams.py` | finds the jams and saves each one's surroundings |
| `jam_collapse.py` | the two boundaries; the exact search for small jams |
| `jam_thrml.py` | the collapse with `thrml` |
| `jam_figures.py` | the pictures |
| `jam_energy.py` | the first, one-hop attempt (kept for reference, not used) |