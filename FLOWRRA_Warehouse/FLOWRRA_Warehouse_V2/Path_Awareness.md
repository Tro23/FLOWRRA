# Path awareness: the holistic FLOWRRA design

Status: **approach charging built** (switch off). **Path awareness designed**, to
be built next. The two go into one joint run, each behind its own switch.

## Why

FLOWRRA was conceived as a swarm: many agents, each seeing its surroundings,
coordinating through what they see rather than through a central planner. This is
that design, brought back into the warehouse. Two things are missing today:

- Fleets see where others **are** and **have been** (the Gibbs-inspired density
  field, the rays), but not where they are **about to go**.
- The reward punishes one of the ways out of a tangle: following.

The symptoms are familiar. Recovery separates fleets in space without knowing
where they are heading, so pairs re-met at the same junction (119/56, 22 times)
and ping-ponged. The trained policy declines at the greedy end in every run since
cold_run16; if soft target updates (cold_run21) do not remove that, coordination
between identical greedy fleets is the leading suspect, and this is its fix.

## Half one: path awareness (to build)

- **Paths ahead.** Each fleet's next few intended cells along its route to its
  goal, stamped into a new channel of the density field, fading with distance
  ahead. "Ahead" means along the route, which can wind: round, back, then forward.
  Intended path = the next cells along the fleet's gradient route (its default
  path, already known), not its future actions (which are not).
- **The rays stay exactly as they are**, all 6 directions; other dynamics depend
  on them. Together with the diamond and the paths ahead, each fleet has a
  **spherical eye**: it sees in every direction, and it knows its own and its
  neighbours' routes wherever they lead.
- **Free routes.** Candidate routes, including back-then-around, scored by how
  clear they are in the paths-ahead density.
- **A learned reflex, nothing hard-coded.** Fleets close to each other untangle
  by choosing among options -- wait, follow another fleet for a while, reroute,
  or a combination over time. Which fleet does what is learned from experience,
  not written as a rule. All the options are already expressible with the 7
  actions; what is added is the picture that shows when each pays.
- **What makes it learnable:** intent it can see (paths ahead); neighbours it
  can talk to (graph attention, already there); a reason to untangle (the
  priority reward, where yielding beats pushing near a conflict); and a
  tie-breaker so identical fleets do not mirror each other into the same move
  (the arrival-order edge channel, already there).
- **Recovery becomes the safety floor.** The reflex takes over the front line --
  the preemptive job -- and recovery and overrides should fall. The tiers stay
  behind it for real collisions: the learned layer acts, a rule layer catches.
- **Decentralised by construction.** Each fleet needs only its own view, its
  neighbours' messages and intents, and one broadcast number (the holon's
  integrity): centralised training, decentralised execution on edge machines.
- A route solver (e.g. Conflict-Based Search) is used only as a **yardstick**,
  to measure how close the learned untangles come to the best possible.

Switching it on changes the state width, so its first run is a cold start.

## How the pieces fit

**Three roles, cleanly divided.**

- **The orchestrator says where to go.** It assigns each fleet a goal and supplies
  the BFS route to it; the goal-distance maps give the distance from every cell,
  so the route from wherever a fleet stands is always known.
- **The paths-ahead channel makes those routes visible locally.** Where routes
  overlap in the channel is where fleets will contend -- the orchestrator's plan,
  seen from inside the tangle.
- **The learned policy decides what each fleet actually does** -- follow its
  route, wait, fall in behind another fleet, or reroute -- all at once, each from
  its own diamond. That simultaneous choice is the collapse.

The BFS does not make the collapse: it supplies the **intentions** the collapse
works with. The BFS says what each fleet wants; the diamond shows where those
wants collide; the policy learns the order and the way out.

**How far ahead: the density diamond's reach.** Each fleet relies only on what it
can see itself, so the part of every route that matters for a tangle lies inside
its own view. Much further stops being local; much shorter sees tangles too late.

**One mechanism, from one obstacle to a crowd.** The BFS runs on the static map, so
a route can point straight through a person or a jam. The rays and mask show the
way is blocked; the paths-ahead channel shows the route goes there anyway; the
fleet learns to wait or go round. Replace the obstacle with another fleet and it
is the same situation, with both routes visible; replace it with a crowd and it is
the same again, with more overlaps. Nothing depends on the number of fleets, since
each decides from its own view. After a detour the route is simply recomputed from
the fleet's new cell, so the paths ahead stay current every step.

**The grid keeps it learnable.** Each fleet chooses from just 7 actions (6
directions and waiting), and at most 6 fleets can meet at one junction. The
90-fleet map is 90% corridor and ~10% junctions, mostly three-way, so most real
tangles involve two or three fleets. In a corridor there is little room to yield,
so untangling usually means one fleet backs up or waits -- which is exactly why
following and waiting must be genuine, unpunished options. Dense many-fleet
tangles are harder to learn, with more combinations to get right; that is where
the tie-breaker, arrival order, matters most.

**Goals create the conflict.** Without goals, nothing is contested and fleets
drift apart. Fleets heading to nearby places share the same shortest routes, so
they converge on the same corridors and shafts. Coordination is the hard part of
this problem, and path awareness targets it directly: it makes the shared routes
visible before they collide.

## Half two: charge approach, not presence (built)

Today the warning charge depends on how close a fleet is, for whatever reason.
Two fleets following calmly pay every step as if about to collide, so a policy
learns to avoid following even where it is the best untangle. On the test
fixture, about **90%** of the warning charge was paid for being near someone,
not for moving toward them.

**Rule.** For each pair inside the warning distance after the action: if the
gap between them shrank, that closing is split between the two by how much each
moved toward the other. Charge = warning_zone x closeness x min(1, share /
base_speed). A gap that held or grew costs nothing.

| case (unit-tested) | fleet A | fleet B |
|---|---|---|
| convoy, same speed | 0 | 0 |
| head-on, both at full speed | -0.80 | -0.80 |
| A drives at a stationary B | -0.80 | 0 |
| follower gaining (0.5 vs 0.3) | -0.22 | 0 |
| standoff, neither moves | 0 | 0 |
| A backs away | 0 | 0 |

**Edge cases handled, most of them learned the hard way:**

1. *Convoy mis-charge* -- measuring movement toward the other's old position
   charged convoy followers in full. Fixed by charging the closing of the gap.
2. *Nearest neighbour changing* -- measured per pair, never per nearest.
3. *Recovery teleports* -- measured across the action only: pre-move positions
   are taken after recovery.
4. *Who pays* -- each fleet for its own share; a held fleet never pays for being
   approached.
5. *Charge-then-undo* -- computed once, after the action, so the old warning-undo
   bug cannot recur.
6. *Scale* -- driving at a neighbour at full speed costs exactly the old worst
   warning, so the verified decision checks still hold.
7. *Farming* -- there is no reward for moving away, only a charge for closing.

**Design principle -- judge the joint outcome.** A move can only be judged by
comparing two ends of the same step: the world the fleets acted on and the world
their actions produced (the learner's own s and s'). The convoy mis-charge came
from comparing one fleet's *after* with another's *before*, as if the other had
stood still. The simultaneous step made the decisions simultaneous; judging the
gap made the judgement simultaneous too. Rule: when judging what a joint action
did, compare everyone's before with everyone's after -- never mix the two ends.

**Open risk: standoffs.** Two fleets stopped face to face close no gap and pay
no approach charge. Only the idle penalty presses on them, after the 3-step
mutual-wait cap -- about 0.03 per step weighted, against up to 0.21 under the
presence charge, so roughly 7x weaker. Standoffs already exist under the old
rule (fixture: 427 pair-steps, longest 116 steps). A stalemate charge is ready to
add if the joint run shows them growing.

## Switches

| switch | status | joint run |
|---|---|---|
| `reward_decomposition.approach_warning` | built, default False | True |
| path awareness | to build | on |

With both off, the code reproduces every earlier run exactly.

## Instruments

Built: `rwd_approach_total`, `rwd_standoff_pairsteps`, `rwd_standoff_longest`,
`rwd_standoff_past_cap`, alongside the old presence warning. To add with path
awareness: how often fleets follow, wait or reroute when in conflict, and the
override and recovery rates.

## Predictions (these can fail)

- Overrides and recovery invocations fall; collisions do not rise.
- Recurrence cycles and ping-pong do not reappear.
- Standoffs do not grow past today's levels.
- The greedy end no longer declines -- if cold_run21 has not already fixed it.
