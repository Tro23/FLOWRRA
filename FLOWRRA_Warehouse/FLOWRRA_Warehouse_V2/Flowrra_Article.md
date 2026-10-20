# FLOWRRA: A Decentralised Holon of Warehouse Fleets That Learns to Keep Itself Whole

Rohit Tamidapati · DhaaRn · October 2026

*Draft 1.*

---

## Abstract

Warehouse automated guided vehicles (AGVs) are usually coordinated by a central planner that computes a path for every vehicle. FLOWRRA takes a different stance. Each vehicle is a *fleet*, and the set of all fleets on the floor forms one holon, which treats itself as its primary environment (A): a loop whose integrity it must keep. The floor, the orders, failures and people form the secondary environment (B).

**Perception.** Each fleet perceives a local affordance field that combines three things:
- ray tracing;
- conflicts splatted into space and time;
- the paths ahead of its neighbours.

The field is read in a Gibbs frame, which puts two measured numbers in every fleet's state: its local action entropy, and the holon's throughput pressure in the place of temperature.

**Decision.** One shared graph network scores each fleet's moves with three consequence heads (safety, delivery and efficiency, weighted 6 : 4 : 1) and an imitation head. A ladder of path-aware rules resolves the conflicts that remain. Values are learned from the moves the system actually made.

**Evaluation.** We ran FLOWRRA against reimplemented PIBT and RHCR on the same simulated floor, inside the same safety boundary: four maps, 20–80 fleets, eight seeds, with and without failures and people, 1,920 arm-days in all.
- The best FLOWRRA network delivered the most of the order book in both conditions: 98.9% without disruptions, and 87.6% with failures and people.
- With failures and people at 60–80 fleets, it delivered more than every planner: 90.5% against 89.1% for the closest (paired Wilcoxon, p = 0.003).
- With little training, a few hundred episodes, the network alone (the route switched off) was level with the best planner at 60–80 fleets under failures: 89.0% against 89.1%.
- Collisions stayed at or below 0.016 per arm-day for every arm.
- The planners were faster per order and kept further from people.

---

## 1. Introduction

**The problem.** A warehouse floor is a graph: aisles, junctions, shelf faces and stations, which fleets traverse one cell at a time. Coordinating many fleets on such a graph is multi-agent path finding (MAPF). Its strongest practical solvers are central:
- **PIBT** decides every fleet's next step each timestep by priority inheritance.
- **RHCR** replans windowed paths for all fleets at intervals.

Both assume one planner with the whole floor in view, and fleets that execute what they are told. Learned decentralised methods, such as Follower, instead give each fleet a local view and a trained policy, and let coordination emerge.

**The reorientation.** FLOWRRA starts from a different question. It does not ask how to route each fleet to its goal. It asks how the set of fleets keeps itself intact while it works. So the set of fleets, one holon, is its own primary environment, environment A. The floor, the order book, failures and people are environment B, the world the holon works in. Every fleet carries a reading of the holon's integrity in its state, and every step is judged as one joint move of the whole loop.

FLOWRRA began as a swarm controller built from federated holons: self-similar groups held together by a spring-loop topology. This paper describes its warehouse instance.

**Contributions**
1. **The two-environment formulation.** The holon's integrity is part of every fleet's state, and each step's moves are judged together, broken into independent chunks only where they truly are independent (§3.1, §3.7).
2. **A local perception.** It combines:
   - an affordance field built from ray tracing and graph reachability;
   - conflicts splatted into space and time;
   - the paths ahead of each fleet's neighbours;
   - two measured Gibbs quantities, local action entropy and throughput pressure (§3.3–§3.6).
3. **A decision stack.** Three consequence heads and an imitation head sit over a ladder of path-aware rules, and values are learned from the moves the system actually executed (§3.8–§3.10).
4. **A same-floor evaluation against PIBT and RHCR.** It includes a measurement of what the safety boundary costs a planner (§5).

**What we claim, and what we do not.** We tested under conditions modelled on real operation: a safety boundary around every fleet, driving physics, hardware failures and people on the floor. Under those conditions, a decentralised holon with a learned policy delivered as much as central planners overall, and more at high density under disruption, where coordination is hardest. That is evidence that decentralised, learned control is a serious option for physical fleets.

We do not claim that learning beats planning in general:
- the planners were faster per order and kept further from people;
- our rules alone came within 0.1 points of the champion under disruption;
- the baselines were implemented from their papers, not the authors' code;
- all results are from simulation.

Nor do we claim physical retrocausality or quantum collapse. FLOWRRA's founding ideas use those words as metaphors, and §4 maps each one to the mechanism that now carries it.

---

## 2. Related work

**MAPF.** Stern et al. (2019) set out the standard definitions: agents occupy vertices of a graph, and a solution must avoid two agents in one vertex at one time, or two agents swapping along one edge.
- **PIBT** (Okumura et al., 2019) resolves each timestep by priority with backtracking. The fleet that loses a contest is pushed aside at once.
- **RHCR** (Li et al., 2021) makes lifelong MAPF tractable at warehouse scale by planning in windows and replanning periodically.
- **k-robust MAPF** (Atzmon et al., 2018) adds a time buffer between agents to absorb execution delays. We use k = 1 for one RHCR variant.
- **Size and execution.** Other lines of work model physical agents:
  - MAPF-POST (Hönig et al., 2016) post-processes plans for kinematic limits and safety distances;
  - MAPF for large agents (Li et al., 2019) gives agents a footprint;
  - continuous-time MAPF (Andreychuk et al., 2022) drops the discrete clock.

  The headline results of PIBT and RHCR, and the standard benchmarks, use the point model with no gap. §5.3 measures what a gap costs.

**Learned decentralised MAPF.** Follower (Skrynnik et al., 2024) combines a learned local policy with heuristic routing for lifelong MAPF. POGEMA (Skrynnik et al.) provides the lifelong protocol on RHCR's Kiva map that our boundary-cost test follows (§5.3).

**Scheduling across the floor.** GRAND (2025) learns where free agents should be distributed across regions, then rebalances them by min-cost flow. It is close in spirit to the holon-level rebalancing we leave for future work (§7).

**Statistical mechanics in learning.** Jaynes (1957) treats temperature as a Lagrange multiplier on an entropy-constrained distribution, with no gas required. Maximum-entropy reinforcement learning (Haarnoja et al., 2017; 2018) optimises reward minus temperature times entropy, and tunes the temperature automatically. That entropy is over actions. FLOWRRA's frame also has a spatial term, to which those results do not transfer automatically.

**Credit and collapse.** Hindsight credit assignment (Harutyunyan et al., 2019) learns from what actually followed a decision. In procedural generation, WaveFunctionCollapse (Gumin, 2016; Karth and Smith, 2017) holds every cell in superposition, fixes the most constrained cell first, and propagates constraints. Diffusion planners (Janner et al., 2022) settle whole trajectories from noise into ones consistent with their start and goal. §4 relates FLOWRRA's mechanisms to these.

---

## 3. The model

Throughout, a *fleet* is one vehicle, a single robot. The set of all fleets on the floor forms one *holon*: environment A. Distances are in hops, one hop being one edge of the floor graph.

### 3.1 Two environments

**Environment A, the holon: the set of all fleets**
- the fleets' positions and intended routes;
- the shared affordance field and its memory (§3.3–§3.4);
- the integrity of the whole: the *holon integrity*, recomputed before every look. It is 0 if any collision occurred, 0.5 if any warning, and 1 if the floor is clear. It sits in every fleet's state, so each fleet's decision is conditioned on the condition of the whole.

**Environment B, the world the holon works in**
- the floor graph, with its shelf faces, docks and stations;
- an order book released over time;
- hardware failures, after which a stopped fleet's load is handed over to a rescuer;
- people, who walk onto the floor to clear debris.

### 3.2 The world and its safety boundary

**Movement.** Fleets move along graph edges at half a hop per step, so one hop takes two steps. Every move happens at driving speed, retreats included; there are no jumps.

**Fleets are points on the graph in every system we compare, FLOWRRA included.** What FLOWRRA adds is a boundary around each point, which acts as the fleet's footprint:
- a minimum gap of 1 hop between fleets;
- warning bubbles of 1.75 hops for fleets meeting, 1.25 for one following another, and 1.0 otherwise;
- directional braking when the gap to the fleet ahead closes below 1.5 hops;
- an intent guard: every fleet declares its next move before anyone moves, and a move that would breach the gap is stopped;
- people as hard no-go cells.

**Fairness.** In the evaluation (§5), every arm runs inside this same boundary, the planners included. In their own benchmarks, PIBT and RHCR forbid only two fleets in one cell and two fleets swapping. §5.3 measures what the difference costs.

### 3.3 Perception: the local field

Each fleet sees a Manhattan diamond of radius 5 around it, 231 cells, in five channels:
1. **mask**, marking cells that are real floor and reachable within 5 hops *through the graph*. A cell one aisle over may be 2 cells away in a straight line but 20 hops by the graph, so it does not count as near.
2. **repulsion**, the fast memory of conflicts (§3.4);
3. **slow memory**, the same events remembered over a longer timescale (§3.4);
4. **the fleet's own route ahead** (§3.5);
5. **its neighbours' routes ahead**, accumulated (§3.5).

**The affordance** of a cell is A = mask / (1 + R), where R is its repulsion. It falls steadily as repulsion rises and never saturates, so even in a dense jam neighbouring cells can still be ranked.

**The base features.** Beside the field, each fleet has 84 base features:
- its heading and goal offset;
- six axis-aligned rays that report how far each direction is clear and what stopped the ray: a waiting fleet, a permanently stopped one, a wall, or something unregistered such as a person;
- the velocities and offsets of the peers the rays hit;
- situation flags;
- the holon integrity;
- the two Gibbs quantities (§3.6).

The state is 84 + 5 × 231 = 1,239 numbers per fleet. **Ray tracing is what makes the field local and measured:** each fleet's rays and field are its own reading of its contribution to the shared distribution.

### 3.4 Spatio-temporal splatting

Conflicts are written into the field where they happen, so the field remembers trouble in space and in time.

**Events.** A collision or a warning stamps repulsion around its location: a fatal splat at 1.5× severity, a warning at 0.6×. Each real warning pair gets one splat of its own. Splats take the maximum with what is already there rather than adding to it, so repeated warnings cannot pile up into a permanent no-go zone.

**Fading.** Memory fades on two timescales:
- the fast channel decays by a factor of 0.7 per step, so an event is materially gone within about 8 steps;
- the slow channel has a half-life of 32 steps, long enough to learn that a corridor keeps jamming, short enough to forget it once it clears.

**Peers.** Each active peer is stamped as a swept path along the next 3 cells it will occupy, with severity falling along the trail. A symmetric blob cannot express the most important fact in a crossing: a fleet moving away is not a threat. A swept path lights up head-on convergence, while receding traffic barely registers.

### 3.5 Path awareness

**The routes.** Each fleet's route follows its goal-distance map downhill: the hop count from every cell to its destination, computed backwards from the destination. Two channels of the field carry routes ahead:
- the fleet's own route, over the cells ahead within its diamond;
- its neighbours' routes, accumulated.

**Warnings.** A warning is raised by paths that are closing, not by mere presence. Two fleets whose routes converge within a 3-hop radius over an 8-hop route horizon are warned. A convoy following at a steady gap is not penalised.

So a fleet sees not only where others are, but where they are about to be.

### 3.6 The Gibbs frame and the temperature

**The frame.** FLOWRRA reads its field in the frame of the Gibbs measure:

```
P(cell) = exp(−E/T) / Z,        F = E − T·S
```

Stress E is minimised while entropy S is paid for at an exchange rate T. The exponential form is chosen for one property: scale covariance. Doubling every stress and halving T leaves the distribution unchanged, so one policy can carry across densities and floor sizes without retuning. An earlier rational form, 1/(1 + R)^β, has no such invariance.

**Replacing the temperature.** The first design put the holon's integrity in the temperature slot, exp(−S / k·C), with C the global integrity in {0, 0.5, 1}. It failed twice:
- **a singularity at C = 0.** One collision anywhere would set the temperature for every fleet on the floor.
- **underflow.** Low integrity and high stress occur together, so in a jam, the moment the gradient matters most, the exponent collapses to exactly zero.

The fix was to measure coherence locally instead. Ray tracing gives each fleet its own reading of its contribution to the distribution, and the entropy below comes from that. That frees the temperature slot for what the system actually trades against safety: throughput.

**What is built: two measured quantities in every fleet's state**
- **S, local action entropy:** the Shannon entropy of the affordances of the six neighbouring cells, normalised by the log of the number of live neighbours. A value of 1 means every option is equally good. Lower values mean contention has concentrated and the choice is being made for the fleet.
- **T, throughput pressure:** T = 1 / (1 + orders left / steps left).
  - Many orders and little time give a low T: push, and accept risk.
  - Slack gives a high T: hold, and stay clean.

  The fixed ratio between the idle penalty and the movement reward was already an implicit, constant T. Making it explicit and state-dependent is the same change at a different altitude.

**What is not built yet.** The field itself is still the rational form A = mask / (1 + R) of §3.3. Three pieces remain:
- the exponential field with a partition function shared across the floor (computed stably by log-sum-exp);
- the global spatial entropy;
- F as the training objective.

The partition function costs compute on every step. It matters most where fleets have real alternatives, in aisles wide enough for two to pass side by side; in single-lane aisles the choices are few.

### 3.7 Wholeness of state and action

**State.** FLOWRRA treats state and action as one whole:

```
H(S, A) = H(S) + H(A | S)
```

- **H(S), spatial entropy:** how concentrated stress is on the floor.
- **H(A | S), conditional action entropy:** given where fleets stand, how undecided they are. It distinguishes a fleet undecided in a hotspot from one undecided in an empty aisle; H(A) alone cannot.

The local term of H(A | S) is built (§3.6); the global H(S) awaits the shared partition function.

**Action, carried into chunks: decide, guard, move**
1. **Decide.** Every fleet proposes its move from the same snapshot of the floor.
2. **Guard.** Conflicting proposals are grouped into independent components (union-find). Each contested component is resolved jointly by Hungarian assignment (Kuhn, 1955), with tie-breaks that do not depend on processing order. Then the intent guard checks every landing against the gap.
3. **Move.** All fleets move at once. Waits, goal claims and arrivals are then judged against one frozen snapshot.

The whole is broken into chunks only where they are truly independent, and no fleet gains by being processed first.

**Learning.** At the end of each step, the orchestrator pushes the whole step to the learner as one transition: every fleet's state, action, reward, next state and next action together. The loop is learned as a loop.

### 3.8 Decision: one network, heads with one job each

**The network.** A single graph neural network is shared by all fleets:
- a gated 3D convolution reads each fleet's five-channel diamond;
- graph attention (Veličković et al., 2018) lets neighbours inform each other, over edges carrying closeness, conflict, head-on approach and arrival order;
- the hidden width is 128.

**The heads**

| Head | Its job | Trained by | Used where |
| --- | --- | --- | --- |
| Safety, Delivery, Efficiency | Forecast the consequences of a fleet's own move | Temporal-difference learning on real rewards only | At choice time, weighted 6 : 4 : 1 |
| Learner | Learn the system's judgement: what the rules (later, the route) would do here | Supervised, on the moves the system made | At choice time only, as a bounded preference, never inside a value or a target |
| Recovery | Decide when the ladder has failed | Parked | L3 is a fixed rule (§3.9) |

**The weighting.** The 6 : 4 : 1 weights mean safety outranks delivery, and delivery outranks efficiency. Each head's reward is divided by a fixed scale, so the weights mean what they say. Safety charges closing speed inside a shrinking gap, not mere proximity.

**The route.** In free traffic, a fleet can simply follow its route (§3.5).
- **ladder4 (route on).** The route drives every fleet that is not in a warned pair, and the network decides only for warned pairs.
- **ladder6 and ladder7 (route off).** The network drives every fleet, with the Learner trained to carry the route's judgement.

### 3.9 The conflict ladder

When proposals conflict on a path, a ladder of rules resolves them. Each rung is tried only when the one above cannot resolve the group, and ordinary traffic never leaves L0.

| Rung | What it does | Safety cost |
| --- | --- | --- |
| L0 | Ordering, braking and corridor holds: the rules of normal traffic | 0 |
| L1 | Retrace: the fleet that gives way backs off 1–2 hops along its trail | −2.4 per hop retreated |
| L2 | Pull over: retreat to a bay off the corridor | −2.4 per hop, plus −6 |
| L3 | Safety net: the fundamental recovery tiers (spatial escape, temporal rewind, right-of-way yield) | −50 per fleet involved |

**How the ladder works**
- **Triggers are structural:** a cycle, a blocked trail, or no progress within a set number of steps. They are never a learned head's guess.
- **Commitment:** a fleet granted passage keeps it until it has cleared the conflict.
- **Rung cost, paid by consequence:** a rung cost is the safety penalty charged once, to the fleets involved, when a conflict escalates to that rung. It prices the disruption their moves made necessary. Following a rule is never charged.
- **The fundamental tiers** at L3 are the floor beneath everything. They will carry more of the load once aisles let two fleets pass side by side.
- **Calibration:** giving way for one hop (two steps at half speed) costs what lingering in the worst warning would cost for the same time, and L3 costs what a collision does.

### 3.10 Learning

**SARSA on the move made.** Values are learned by SARSA (Rummery and Niranjan, 1994) from the move the system actually executed. Each transition is held for one step, until the next move is known.
- Where the route or a rule overrode the network, the target uses the executed move, so nothing but real consequences ever moves a value.
- Experience is replayed from a buffer of 20,000 transitions in batches of 64, with discount 0.99.
- A target network is copied from the online network every 1,000 learning steps.

**Training.** The networks are lightly trained.
- **The shared base:** all three start from one network, trained from scratch for 150 episodes of 800 steps.
- **ladder4, the champion:** 50 more episodes, with one setting changed (SARSA everywhere).
- **ladder6:** two further stages of 80 episodes, adding the Learner and progressively switching the route off.
- **ladder7:** a variant of ladder6's last stage, on 25–60 fleets only.

Each stage ran its own exploration schedule end to end. In all, these are a few hundred episodes. The full history is in the build log (V2_The_Conflict_Ladder.md).


---

## 4. Retrocausality and collapse, made concrete

FLOWRRA was founded on two principles taken from physics:
- **Retrocausality:** the future informs the present.
- **Wave function collapse:** a superposition of possibilities settles into one consistent outcome.

They were conceived independently, as the design's starting principles, before the mechanisms below existed. The papers cited in this section were found afterwards, to see whether the ideas existed elsewhere and to put them in established academic terms. In the system described above, each principle now has a working counterpart, built and measured.

**Deciding from predicted futures.** At decision time, FLOWRRA already looks ahead:
- the route reads the distance to the destination;
- the field carries every neighbour's route ahead (§3.5);
- the guard lets every fleet declare its move, computes where all of them would land, and only then lets anyone move (§3.7).

This is predictive control. Only predictions are available at decision time, never the future itself.

**Learning from actual futures.** In learning, information from what actually followed flows back legitimately: every value is learned from what came after. SARSA on the move made goes one step further. It holds each transition until the system's next executed move is known, so the target reflects what the system actually did (§3.10). The defensible name for this "retrocausality" is hindsight credit assignment (Harutyunyan et al., 2019). Decisions are made from predicted futures; learning is done from actual futures.

**Working backwards from the goal.** Each fleet's goal-distance map is computed backwards, from its destination to every cell, and the route follows it downhill. Where a fleet is now and where it must end up fix the path between them, like a two-point boundary problem in physics. The "future" here is a present fact: the destinations are known now. This is backward reasoning from known goals, not physical retrocausality, and that is what keeps the language honest.

**Collapse.** When fleets bunch up, the ladder resolves the group before contact:
- **temporal:** L1 retraces a fleet along the path it came by;
- **spatial:** L2 moves it into a bay off the corridor.

This is the discrete, built form of the spatial-temporal collapse in FLOWRRA's earlier designs, and it is measured. On a clean day the champion needed the safety net (L3) 36 times. The rules alone needed it 140 times (§5.2).

**The next step: a learned boundary, then an exact collapse.** For several fleets in a jam, the full form is local space-time planning, and it has the shape of wave function collapse in procedural generation (Gumin, 2016; Karth and Smith, 2017):
- each fleet's possible positions over the next few steps start in superposition;
- the most constrained fleet is fixed first;
- the constraints propagate: one fleet per cell per step, no swaps, progress towards each destination;
- the first step of the consistent plan is the move taken now.

The design has two steps:
1. **The boundary.** Assign each fleet in the zone an exit cell, no two the same. Minimise the moves inside the zone plus the cost-to-go from each exit; the network's values price the traffic beyond the zone.
2. **The collapse.** Solve the zone's moves exactly between the two fixed boundaries: where the fleets are, and where they leave.

The ladder's L2 is a crude version of the first step: one retreat cell, chosen by rule. A learned boundary is only as good as the values behind it, which is why honest values (§3.10) came first.

---

## 5. Evaluation: same floor, different brain

### 5.1 Setup

**Conditions modelled on real operation.** The test floor includes what standard MAPF benchmarks leave out:
- the safety boundary of §3.2 around every fleet;
- driving at half a hop per step, with no jumps, and speed that changes with the situation: slowing in warning zones, braking behind a fleet closing ahead, holding speed on the final approach to a goal, and picking up again once clear;
- hardware failures, about 11.5 a day, modelled on VDA 5050, the open standard for communication between AGVs and a master control. A failed fleet stops dead mid-mission and reports an error state, and its load is handed over to a rescuer. What a fleet reads about its stopped and immobile peers is what a VDA 5050 state report would carry;
- people walking onto the floor to clear debris;
- orders released over time from a book, with docks where fleets enter and leave.

It leaves out:
- communication delay and loss;
- localisation error;
- batteries. Charging is designed to reuse the leave-and-return dynamics fleets already have at the docks: undock, travel, dock, and return to work.

**Maps**
- **Three multi-level warehouses from the 3D MAPF benchmark set** (Peter Wang and Rishi Veerapaneni, listed on mapf.info): small (25_5_2_5_2_1), medium (25_10_5_10_5_2) and large (50_20_5_10_5_2). Their floors are joined by vertical shafts.
- **A Kiva-style layout made for this project:** 46 × 33 cells, two banks of shelves per row and 12 workstations, 1,262 open cells. It shares RHCR's grid and aisle rhythm, but it is not RHCR's published map.

**Days**
- **Fleets on the floor:** 20, 40, 60 and 80.
- **Seeds:** 0–7, with scenario files generated per map and seed. Each fleet's first order is its scenario goal, assigned at spawn by Hungarian assignment.
- **Demand:**
  - on the benchmark maps, a book of 100 orders released over 500 steps, in days of 1,000 steps;
  - on the Kiva-style map, 200 orders over 500 steps, in days of 1,200 steps (clean) or 1,500 (full).
- **Two conditions:**
  - *clean*: no failures, no people;
  - *full*: both.
- **Identical days:** failures and people are drawn from keys (instance, step, fleet), so every arm meets exactly the same day.

**Arms**

| Arm | What drives the fleets |
| --- | --- |
| ladder4 | FLOWRRA's champion: the network, with the route on (§3.8) |
| ladder6, ladder7 | FLOWRRA networks with the route off: the network drives every fleet |
| RULES | FLOWRRA's rules and ladder alone, with no network |
| PIBT | PIBT's rule applied every step to fleets on nodes |
| RHCR-PP | RHCR with prioritised planning: a 20-timestep window, replanned every 5 timesteps or when the floor changes, with a one-timestep buffer (k-robust, k = 1) |
| RHCR-PIBT | RHCR with PIBT as its planner, without a buffer, since PIBT's push needs to follow at once |
| RHCR-PIBT+naive | RHCR-PIBT with naive rescue: a dead fleet's pickup is abandoned and never re-dispatched (full only) |

**Same floor, different brain**
- **The same for every arm:**
  - the world: map, book, dispatcher, docks, rescue handovers, failures, people;
  - the physical safety layer: gap, bubbles, braking, intent guard, yielding to stopped fleets, people as no-go cells, collision recovery.
- **Off for the baselines:** FLOWRRA's brain, meaning the route, corridor entry, priority, the ladder and the livelock escape.
- **Adapting the planners:** they are drivers inside our world and see the whole floor.
  - Fleets between two nodes are planned from the nearer node.
  - Stopped fleets and people are obstacles, except a rescuer's own pickup.
  - Planning time is reported, not scored.
- **Implemented from their papers**, not the authors' code:
  - PIBT's per-step rule, with priority rising 0.5 a step and reset on arrival;
  - RHCR with a 20-timestep window, replanned every 5 timesteps or when the floor changes.

**World fixes.** Three flaws in our world were found while preparing the baselines. Their fixes apply to every arm:
- **node snap:** a fleet stuck at a floating-point position just short of a node no longer stalls;
- **bubble floor:** a leader exactly 1 hop ahead of its follower is no longer braked to a crawl;
- **orphans wait for a pickup:** a stopped fleet's order waits for a rescuer to fetch its load, instead of returning to the book.

**Measures**
- **Book delivered:** the share of the day's book that was delivered.
- **Lead time, censored:** every undelivered order counts as waiting until the end of the day, so an arm cannot look fast by delivering less.
- **Collisions.**
- **Near misses with people** (full).
- **Guard stops:** moves the intent guard refused.

**Statistics.** Every comparison is paired on map × fleet × seed, 128 days per arm per condition, with Wilcoxon signed-rank tests. "Firmly" below means p < 0.05 in that direction. In all: 896 arm-days clean and 1,024 full, 1,920 in total, run as 240 jobs on 14 processes in about 16 hours.

### 5.2 Results

**Table 1. Headline, means per day.** Lead time is in steps.

| Arm | Delivered, clean | Delivered, full | Lead, clean | Lead, full | Near misses with people, full | Guard stops, full | Planning ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ladder4 | **98.9%** | **87.6%** | 112 | 192 | 77.4 | 1,175 | – |
| ladder6 | 94.2% | 82.1% | 149 | 234 | 74.7 | 1,138 | – |
| ladder7 | 93.6% | 82.8% | 155 | 241 | 61.5 | 1,476 | – |
| RULES | 95.5% | 87.5% | 126 | 191 | 84.0 | 500 | – |
| PIBT | 98.4% | 79.2% | **86** | 188 | **47.2** | **0** | 0.9 |
| RHCR-PP | 94.8% | 86.3% | 103 | **165** | 52.4 | 1,284 | 115–119 |
| RHCR-PIBT | 75.2% | 74.6% | 210 | 270 | 79.4 | 4,694 | 52 |
| RHCR-PIBT+naive | – | 75.3% | – | 265 | 77.3 | 4,732 | 50 |

**Table 2. Book delivered by number of fleets, full condition.**

| Arm | 20 | 40 | 60 | 80 |
| --- | --- | --- | --- | --- |
| ladder4 | 80.5% | **89.2%** | **91.3%** | **89.6%** |
| ladder6 | 64.5% | 86.0% | 88.7% | 89.4% |
| ladder7 | 74.4% | 83.2% | 86.5% | 86.9% |
| RULES | **81.8%** | 89.1% | 89.9% | 89.0% |
| PIBT | 62.4% | 83.6% | 84.6% | 86.1% |
| RHCR-PP | 77.8% | 89.1% | 89.2% | 89.0% |
| RHCR-PIBT | 69.3% | 74.0% | 76.9% | 78.1% |
| RHCR-PIBT+naive | 70.8% | 74.5% | 76.7% | 79.0% |

**Table 3. Book delivered by map, full condition** (clean, on the Kiva-style map, in brackets).

| Arm | Small | Medium | Large | Kiva-style |
| --- | --- | --- | --- | --- |
| ladder4 | 90.1% | 89.8% | **86.3%** | **84.3%** (95.6%) |
| ladder6 | 90.5% | 86.2% | 81.6% | 70.3% (76.9%) |
| ladder7 | **91.5%** | **90.5%** | 84.5% | 64.5% (74.9%) |
| RULES | 90.6% | 89.8% | 85.6% | 84.0% (83.5%) |
| PIBT | 78.1% | 82.4% | 79.1% | 77.2% (93.5%) |
| RHCR-PP | 88.3% | 89.4% | 84.4% | 83.0% (84.8%) |
| RHCR-PIBT | 81.8% | 87.1% | 82.9% | 46.5% (33.4%) |
| RHCR-PIBT+naive | 81.1% | 87.9% | 85.2% | 46.7% (–) |

**What the tables show**

1. **ladder4 delivered the most of the book in both conditions.**
   - Clean: 98.9% against PIBT's 98.4%, which is no firm difference (p = 0.18).
   - Full: 87.6% against RULES' 87.5% (p = 0.40) and RHCR-PP's 86.3%, firmly ahead of RHCR-PP (p = 0.001).
   - On clean days, most arms delivered nearly the whole book on the three benchmark maps. The Kiva-style map, with twice the book, separated them: ladder4 95.6%, PIBT 93.5%, RHCR-PP 84.8%, RULES 83.5%.

2. **Where coordination is hardest, ladder4 delivered firmly more than every planner.** With failures and people at 60–80 fleets:
   - ladder4 90.5%;
   - RHCR-PP 89.1% (p = 0.003);
   - PIBT 85.3% (p < 0.001);
   - RHCR-PIBT 77.5% (p < 0.001).

   FLOWRRA's rules alone delivered 89.5% there.

3. **The network alone is already a serious option, with little training.** In the route-off networks, the network drives every fleet. They had only one or two stages of 80 training episodes beyond the shared base (§3.10).
   - With failures and people at 60–80 fleets, ladder6 delivered 89.0%. That is level with the best planner, RHCR-PP at 89.1% (p = 0.54), and firmly above PIBT (85.3%) and RHCR-PIBT (77.5%).
   - With failures and people overall, both (82.1%, 82.8%) delivered more than PIBT (79.2%) and RHCR-PIBT (74.6%).
   - ladder7 had the highest mean of any arm on the small map (91.5%) and the medium map (90.5%).
   - ladder7 had fewer near misses with people (61.5 a day) than ladder4 (77.4) or RULES (84.0).
   - Their weaknesses were concentrated on the Kiva-style map (64.5–70.3% under failures) and on 20-fleet days. That is why both trail ladder4 overall (by 4.9 and 5.5 points, firmly). They are also slower per order (234–241 steps in full).

4. **What the learned policy added to the rules.**
   - Clean: ladder4 delivered 3.4 points more than RULES and was faster per order (112 steps against 126), both firmly.
   - Full: the two were level.
   - The champion needed the safety net (L3) far less: 36 times a clean day against 140, and 42 a full day against 117.
   - It had more guard stops than the rules (1,175 a day against 500 in full): the network proposes moves the guard refuses more often than the rules do.

5. **The planners were faster per order and kept further from people.**
   - Fastest clean: PIBT, 86 steps against ladder4's 112, firmly.
   - Fastest full: RHCR-PP, 165 against 192, firmly.
   - Fewest near misses with people in full: PIBT (47 a day) and RHCR-PP (52) against ladder4's 77. Firm for RHCR-PP; PIBT's daily figures varied too much to be firm (p = 0.19).
   - PIBT re-decides every step from where the fleets actually are, and the guard never had to stop it.

6. **PIBT lost ground under failures.**
   - It fell from 98.4% clean to 79.2% full, and to 62.4% at 20 fleets.
   - It completed the fewest rescue handovers of any arm: 7.8 a day, against 8.3 for the naive arm (which abandons pickups by design) and 10.3–11.3 for the rest.

7. **Rescue re-dispatch made no consistent difference.** RHCR-PIBT delivered 74.6% with re-dispatch and 75.3% without, though only the naive arm lost orders (2.5 a day).

8. **Safety.** No arm averaged more than 0.016 collisions a day.

**Computation.** RHCR-PP's planning took 115–119 ms per step, and PIBT's under 1 ms. FLOWRRA's network was not timed on its own. Measured as ladder4's extra wall time over RULES, which runs the same world without the network, it cost about 100–135 ms per step for all fleets together, on one CPU thread (§6).

### 5.3 What a safety boundary costs: the price of physical operation

PIBT's and RHCR's published results, like the standard MAPF benchmarks, assume the point model: two fleets may not share a cell or swap places, and nothing more. A physical fleet must also keep a safety distance from the next. To measure what that costs a planner, and to separate it from the cost of coordination, we ran a lifelong test on RHCR's own Kiva map:
- synthetic lifelong tasks drawn in POGEMA's style, with a new goal on arrival;
- 32 fleets;
- 512 timesteps;
- seed 0.

**Table 4. Lifelong Kiva (RHCR's map), 32 fleets, 512 timesteps, seed 0.**

| | Goals | Goals per timestep |
| --- | --- | --- |
| PIBT, usual MAPF rules | 535 | 1.045 |
| PIBT inside the safety boundary | 170 | 0.332 |
| FLOWRRA's rules inside the safety boundary | 107 | 0.209 |

**What it shows**
- Keeping a safety boundary costs PIBT a factor of 3.1 in throughput (1.045 to 0.332 goals per timestep). This is the price of physical operation, which results under the point model do not show.
- Inside the same boundary, coordination accounts for a further factor of 1.6, PIBT over our rules.
- On a log scale, about 70% of the gap between standard PIBT and our rules is the boundary, and about 30% is coordination.
- PIBT's advantage here is that the fleet losing a contest steps aside at once, while ours waits in place outside corridors. That is a lesson for FLOWRRA's rules.
- The price grows with the size of the boundary, so the boundary itself is worth shrinking. A physical fleet needs a gap of about its stopping distance, which shrinks with speed (§7).

This is one seed. It shows the size of the effect, not a tight estimate.

### 5.4 Further detail

The per-day results, the scoring script and the full run history, including the predictions written before the run and how each scored, are in the build log (V2_The_Conflict_Ladder.md, entry 44) and the repository.

---

## 6. Scope and assumptions

- **The safety boundary.** Every result is for one setting, chosen for safety at hop scale before working at a finer one:
  - a minimum gap of 1 hop;
  - warning bubbles of 1.0–1.75 hops;
  - braking from 1.5 hops;
  - the intent guard.

  Every arm ran inside it, FLOWRRA's included. A tighter boundary would move every arm, and not necessarily equally.
- **The baselines** were implemented from their papers, not the authors' code:
  - PIBT's per-step rule;
  - RHCR with a 20-timestep window, replanned every 5 timesteps, with k = 1 for RHCR-PP.

  They drove inside the same world and boundary as FLOWRRA, with FLOWRRA's own decision rules switched off.
- **Maps and seeds.** Four maps, eight seeds each (0–7), at 20, 40, 60 and 80 fleets: 128 days per arm per condition.
  - The Kiva-style map shares RHCR's 33 × 46 grid and aisle rhythm, but not its shelf layout (1,262 open cells against 1,278).
  - The boundary-cost test (§5.3) used RHCR's own map, with one seed.
- **Training.** A shared base of 150 episodes, then 50 more for ladder4, and one or two stages of 80 for the route-off networks (§3.10).
- **Service level, not capacity.**
  - The fleets work lifelong: each takes the next order on arrival, orders are released continuously, and a stream mode removes the end of the book.
  - The evaluation measured how much of a set demand was served in a fixed day: 100 orders over 500 steps in 1,000-step days on the benchmark maps, and 200 over 500 in 1,200- or 1,500-step days on the Kiva-style map. It did not measure the most the holon can move at saturation.
  - On clean days the leading arms sit near the 100% ceiling, which compresses the differences between them.
- **Computation.**
  - In simulation, on one CPU thread for all fleets together, ladder4 took about 100–135 ms per step more than RULES, which runs the same world without the network. RHCR-PP's planning alone took 115–119 ms per step.
  - In deployment each fleet runs its own forward pass on its own hardware, so its cost depends on its neighbourhood, not on the whole floor as a central planner's does.
- **Where the planners lead:** speed per order (RHCR-PP 165 steps against ladder4's 192 in full) and distance from people (PIBT 47 and RHCR-PP 52 near misses a day, against ladder4's 77).
- **Where the route-off networks trail:** the Kiva-style map and 20-fleet days.
- **Simulation:** no communication faults or localisation error, and no batteries yet.

---

## 7. Future work

- **Federated holons.** Holons that grow and shrink, and rebalance fleets between zones of the floor, keeping each zone at equilibrium, in the spirit of GRAND's rebalancing. The temperature becomes per zone, so different zones can run at different pressures.
- **The full Gibbs field:**
  - the exponential affordance with a partition function shared across the floor;
  - the global spatial entropy;
  - F = E − T·S as the objective.

  It matters most where fleets have real alternatives, in aisles wide enough for two to pass side by side.
- **The learned boundary and exact collapse** of §4.
- **The network alone.** More training for the route-off networks. They already match the best planner at 60–80 fleets under failures and lead on the smaller maps.
- **Working closer: a boundary that scales with speed.**
  - A physical fleet needs a gap of about its stopping distance: centimetres when creeping, more at speed. Real AGVs' speed-dependent safety fields already work this way.
  - FLOWRRA already slows in warning zones. Tying the boundary to speed would take it from hop scale towards centimetres where that is safe.
  - §5.3 shows how much that is worth to every arm. FLOWRRA's rules were tuned for the current setting.
- **Lifelong capacity.** The boundary-cost test across seeds, with FLOWRRA's networks beside the planners, on POGEMA's own instances.
- **The world.** Fleets that enter and leave the floor; batteries, with charging stations served by the same leave-and-return dynamics as the docks; and hardware.

---

## 8. Conclusion

FLOWRRA treats the set of fleets on a warehouse floor as one holon whose first task is to keep itself whole. Each fleet reads a local field of reachability, conflict memory and paths ahead. Each decides through heads with one job each, under a ladder of rules, and learns from what the system actually did.

We tested it against central planners under conditions modelled on real operation: a safety boundary around every fleet, driving physics, failures and people. It delivered the most of the order book, and firmly more than every planner at high density under disruption. The planners kept their advantages in speed per order and in distance from people.

Keeping the safety distance a physical fleet needs costs a standard planner two-thirds of its throughput, a price that results under the point model do not show.

FLOWRRA's founding metaphors, retrocausality and collapse, now point to mechanisms that run: learning from actual futures, reasoning backwards from known goals, and resolving jams in space and time before contact.

The result is not that learning beats planning. It is that a decentralised, learning holon is a serious option for physical fleets, and the measurements show where it already leads and where it does not yet.

---

## Author's note

- *The principles behind FLOWRRA (environments A and B, learning from the future, collapse) came to me before I knew the literature that names them. I am grateful for them, and for how they arrived; they are not only mine.*
- *I could have not realized this, had it not been for Claude. A tremendous Coder, De-Bugger and Analyzer, kept me inline through coding principles and never once allowed me to overclaim a vision without it being grounded in truth. I would like to pay my gratitude to Gemini and Grok for being tremendous peer reviewers and for their wonderful support.*

## Code and data

Code, the per-day results (final_clean.csv, final_full.csv), the report script and the build log are at github.com/Tro23/FLOWRRA (FLOWRRA_Warehouse_V2), under AGPL-3.0.

## References

- Andreychuk, A., Yakovlev, K., Surynek, P., Atzmon, D., and Stern, R. (2022). Multi-agent pathfinding with continuous time. *Artificial Intelligence*, 305.
- Atzmon, D., Stern, R., Felner, A., Wagner, G., Barták, R., and Zhou, N.-F. (2018). Robust multi-agent path finding. *SoCS*.
- GRAND (2025). GRAND: Guidance, Rebalancing, and Assignment for Networked Dispatch in Multi-Agent Path Finding. arXiv:2512.03194.
- Gumin, M. (2016). WaveFunctionCollapse. GitHub repository.
- Haarnoja, T., Tang, H., Abbeel, P., and Levine, S. (2017). Reinforcement learning with deep energy-based policies. *ICML*.
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*.
- Harutyunyan, A., Dabney, W., Mesnard, T., et al. (2019). Hindsight credit assignment. *NeurIPS*.
- Janner, M., Du, Y., Tenenbaum, J. B., and Levine, S. (2022). Planning with diffusion for flexible behavior synthesis. *ICML*.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4).
- Karth, I., and Smith, A. M. (2017). WaveFunctionCollapse is constraint solving in the wild. *FDG*.
- Hönig, W., Kumar, T. K. S., Cohen, L., Ma, H., Xu, H., Ayanian, N., and Koenig, S. (2016). Multi-agent path finding with kinematic constraints. *ICAPS*.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2.
- Li, J., Surynek, P., Felner, A., Ma, H., Kumar, T. K. S., and Koenig, S. (2019). Multi-agent path finding for large agents. *AAAI*.
- Li, J., Tinka, A., Kiesel, S., Durham, J. W., Kumar, T. K. S., and Koenig, S. (2021). Lifelong multi-agent path finding in large-scale warehouses. *AAAI*.
- Okumura, K., Machida, M., Défago, X., and Tamura, Y. (2019). Priority inheritance with backtracking for iterative multi-agent path finding. *IJCAI*.
- Rummery, G. A., and Niranjan, M. (1994). On-line Q-learning using connectionist systems. Technical report CUED/F-INFENG/TR 166, Cambridge University.
- Skrynnik, A., Andreychuk, A., Nesterova, M., Yakovlev, K., and Panov, A. (2024). Learn to Follow: decentralized lifelong multi-agent pathfinding via planning and learning. *AAAI*.
- Skrynnik, A., et al. (2025). POGEMA: a benchmark platform for cooperative multi-agent pathfinding. *ICLR*.
- Stern, R., Sturtevant, N., Felner, A., et al. (2019). Multi-agent pathfinding: definitions, variants, and benchmarks. *SoCS*.
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., and Bengio, Y. (2018). Graph attention networks. *ICLR*.
- Wang, P., and Veerapaneni, R. 3D MAPF benchmark set. Listed at mapf.info.
