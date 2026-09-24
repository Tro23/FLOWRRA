"""
recovery_Warehouse.py

Discrete Wave Function Collapse (WFC) for FLOWRRA.
Replaces continuous manifold smoothing and random jitter with discrete 
graph-based Spatial escapes, Temporal rewinds, and a Right-of-Way Yield protocol.
"""

from typing import Any, Dict, List, Optional, Set
import numpy as np

class WarehouseRecovery:
    """
    Manages the Spatial-Temporal Collapse recoveries for the discrete warehouse holon.
    """

    def __init__(
        self,
        history_length: int = 50,
        spatial_safe_threshold: float = 0.35,
        collision_threshold: float = 1.0,
        base_yield_steps: int = 5,
        yield_escalation_per_repeat: int = 5,
        max_yield_steps: int = 30,
        escape_clearance: float = 0.0,
        warning_threshold: float = 2.0,
        pair_escalation_threshold: int = 2,
        escalate_on_recurrence: bool = False,
        recurrence_radius: Optional[float] = None,
        frozen_obstacle_severity: float = 1.0,
        frozen_near_goal_radius: float = 3.0,
    ):
        self.history_length = history_length
        # NOTE (2026-08-28): the MEANING of this value changed with the density
        # rewrite. It used to be a SELECTION criterion -- Tier 1 walked the
        # neighbours in arbitrary graph order and took the first one above it.
        # Under the old clip(1-R) transform that was unreachable in practice: a
        # fresh fatal splat pinned every cell within Manhattan distance 3 to
        # exactly 0.0, so no graph neighbour of a crashed fleet could clear 0.7
        # for ~47 steps, which is why Tier 1 almost never fired and Tier 2 was
        # doing nearly all the work. It is now a safety FLOOR: Tier 1 scores
        # every neighbour and takes the BEST one, rejecting it only if even the
        # best is below this. See CONFIG["recovery"]["spatial_safe_threshold"]
        # for reference values on the new 1/(1+R) scale.
        self.spatial_safe_threshold = spatial_safe_threshold
        # BUG THIS FIXES: Tier 2's safety check below used to hardcode 1.0 directly,
        # independent of whatever collision_threshold was actually configured
        # elsewhere -- so lowering collision_threshold (e.g. to 0.5) wouldn't
        # actually change what Tier 2 considers a "safe" historical snapshot to
        # rewind to.
        self.collision_threshold = collision_threshold
        self.base_yield_steps = base_yield_steps
        self.yield_escalation_per_repeat = yield_escalation_per_repeat
        # HARD CEILING on a yield. Without one the escalation is unbounded, and
        # it ran away in the 2026-09-17 cold run: repeat #169 produced a hold of
        # 10 + 5*169 = 845 steps inside a 780-step episode, so eleven of twelve
        # fleets were frozen from step 163 to the end of the episode.
        self.max_yield_steps = int(max_yield_steps)
        # Fleets currently serving a yield. Their continued overlap is the
        # CONSEQUENCE of the remedy, not a new offence -- see
        # _record_colliding_pairs().
        self.currently_yielding: Set[str] = set()
        # Tier-1 escapes that moved a fleet, and ones that did not. The log
        # CANNOT tell these apart -- its relocation line truncates positions with
        # int(), so a genuine 8.5 -> 8.0 move prints identically to standing
        # still. Counted here instead, where the actual floats are in hand.
        self.tier1_real_escapes = 0
        self.tier1_noop_escapes = 0
        # Candidate cells refused because another fleet was within
        # collision_threshold of them. If this is large, Tier 1 was previously
        # relocating fleets straight into each other.
        self.tier1_rejected_occupied = 0
        # Set by the orchestrator from step.simultaneous. When True, escape order
        # and winner tie-breaks follow fleet id rather than list position.
        self.order_independent = False
        # Cells holding a human or debris. Refreshed by the orchestrator before
        # every recovery call, because it REASSIGNS its set when humans move --
        # a reference taken once would go stale.
        self.static_obstacles: Set = set()
        self.tier1_rejected_obstacle = 0
        self.tier2_rejected_obstacle = 0
        # HOW FAR AN ESCAPE MUST GET FROM THE FLEET IT CONFLICTED WITH.
        #
        # THE GEOMETRIC CEILING, which makes most values useless:
        #   a conflicting peer is within collision_threshold (0.5), and one hop
        #   moves at most 1.0 cell, so the best separation any Tier-1 escape can
        #   achieve is about 1.5. Requiring warning_threshold (2.0) is therefore
        #   UNSATISFIABLE -- measured, Tier 1 succeeded 2 times in 500 steps
        #   instead of 50. It was not made selective, it was switched off.
        #
        # 0.0 disables the extra requirement (default, old behaviour). Useful
        # values live between collision_threshold and ~1.4.
        self.escape_clearance = float(escape_clearance)
        # Recovery is constructed with collision_threshold only; the warning band
        # lives on WarehouseLoop. Derived here so the clearance check has it
        # without threading another argument through every call site.
        self.warning_threshold = float(warning_threshold)

        # RECURRENCE: the same pair recovered again at (roughly) the same place.
        #
        # The repeat-offence counter below only counts pairs that are ACTUALLY
        # TOUCHING (dist <= collision_threshold). A preemptive recovery fires in
        # the warning band, where nobody is touching, so it never counts -- and
        # a pair that keeps meeting there is separated forever and never given
        # right-of-way. cold_run15: fleets 119 and 56 met at the top of the x=5
        # shaft (119 climbing, 56 arriving along the top level), were separated
        # to identical positions 22 times in a row, never escalated, and neither
        # delivered. Every recovery cycle in that run -- three pairs, six fleets
        # -- ended with both fleets undelivered. A cycle always traps a PAIR,
        # which is the shape of the "48/50, never 50/50" runs.
        #
        # Recurrence counts preemptive AND fatal recoveries. Escalation takes the
        # larger of the two counts, so fatal escalation can never get weaker.
        #
        # RADIUS, tied to velocity: between two back-to-back recoveries a fleet
        # can drift at most one step at base_speed (0.5) plus one escape hop
        # (1.0) = 1.5 cells. Defaulting to the warning distance (2.0) always
        # contains that. A pair meeting somewhere NEW resets to 1, so separate
        # junctions do not accumulate.
        self.escalate_on_recurrence = bool(escalate_on_recurrence)
        self.recurrence_radius = (float(recurrence_radius)
                                  if recurrence_radius is not None
                                  else self.warning_threshold)
        self.pair_last_site: Dict = {}
        self.pair_recurrence_counts: Dict = {}
        self.recurrence_escalations = 0
        # After a given PAIR of fleets has fatally collided this many times in one
        # episode, stop treating each collision as a fresh independent incident.
        # See _assign_yields() for the full reasoning.
        self.pair_escalation_threshold = pair_escalation_threshold
        # Same CONFIG values the main perception loop uses -- passed through
        # explicitly here rather than left to get_local_affordance()'s own
        # defaults, so there's one source of truth (CONFIG) instead of two
        # numbers that happen to currently match but could silently drift.
        self.frozen_obstacle_severity = frozen_obstacle_severity
        self.frozen_near_goal_radius = frozen_near_goal_radius
        self.history: List[Dict[str, Any]] = []
        
        # Metrics
        self.total_collapses = 0
        self.spatial_recoveries = 0
        self.temporal_recoveries = 0
        self.yield_recoveries = 0

        # Tracks how many times each node has been part of a FATAL collision this
        # episode (reset naturally each episode, since a fresh WarehouseRecovery is
        # constructed per episode). Used to escalate Tier-3 yield duration for
        # repeat offenders -- see collapse_and_reinitialize()'s Tier 3 section.
        self.node_collision_counts: Dict[str, int] = {}

        # How many times each specific PAIR has fatally collided this episode.
        # Keyed by frozenset({id_a, id_b}) so ordering doesn't matter.
        #
        # WHY PAIRS AND NOT NODES: node_collision_counts above says "fleet 9 has
        # been in 5 collisions", which does not distinguish five different
        # conflicts from the same conflict five times. The logs show it is
        # overwhelmingly the latter -- fleets 9 and 24 colliding at steps 107,
        # 142, 177, 189 and 203 within a single episode. Only a per-pair count
        # can see that.
        self.pair_collision_counts: Dict[frozenset, int] = {}

        # Which fleet LOST right-of-way the last time this pair conflicted.
        # Used to alternate the winner on the next offence -- see _assign_yields().
        self.pair_last_loser: Dict[frozenset, str] = {}

    def _record_recurrences(self, conflicted_nodes: List[Any]) -> int:
        """
        Bump each conflicting pair's recurrence count if it is meeting within
        recurrence_radius of where it last met; start it at 1 if somewhere new.
        Returns the highest count involved.

        A PAIR here is two conflicted fleets within warning_threshold of each
        other -- a preemptive recovery can hold several unrelated warning pairs
        at once, and only genuinely close pairs are one conflict.

        Same guard as _record_colliding_pairs: a pair BOTH currently held is not
        counted. Held fleets cannot move, stay close, and would be counted as a
        fresh recurrence every step -- the self-feeding spiral that once held a
        fleet for 845 steps of a 780-step episode.
        """
        worst = 0
        for i in range(len(conflicted_nodes)):
            for j in range(i + 1, len(conflicted_nodes)):
                a, b = conflicted_nodes[i], conflicted_nodes[j]
                if float(np.sum(np.abs(a.current_pos - b.current_pos))) > self.warning_threshold:
                    continue
                key = frozenset((a.id, b.id))
                if a.id in self.currently_yielding and b.id in self.currently_yielding:
                    worst = max(worst, self.pair_recurrence_counts.get(key, 0))
                    continue
                site = (a.current_pos + b.current_pos) / 2.0
                prev = self.pair_last_site.get(key)
                if prev is not None and float(np.sum(np.abs(site - prev))) <= self.recurrence_radius:
                    self.pair_recurrence_counts[key] = self.pair_recurrence_counts.get(key, 0) + 1
                else:
                    self.pair_recurrence_counts[key] = 1
                self.pair_last_site[key] = site.copy()
                worst = max(worst, self.pair_recurrence_counts[key])
        return worst

    def _record_colliding_pairs(self, conflicted_nodes: List[Any]) -> int:
        """
        Reconstructs which PAIRS among the deadlocked fleets are actually within
        collision_threshold of each other, bumps their counters, and returns the
        highest repeat count involved in this incident.

        Recomputed from positions rather than passed in from WarehouseLoop, so no
        signature has to change anywhere upstream -- the same pairwise test the
        loop already ran is cheap to repeat over the handful of conflicted nodes.
        """
        worst = 0
        for i in range(len(conflicted_nodes)):
            for j in range(i + 1, len(conflicted_nodes)):
                a, b = conflicted_nodes[i], conflicted_nodes[j]
                dist = float(np.sum(np.abs(a.current_pos - b.current_pos)))
                if dist <= self.collision_threshold:
                    # DO NOT COUNT AN OVERLAP BETWEEN TWO ALREADY-HELD FLEETS.
                    #
                    # A yielding fleet cannot move. If it was overlapping when
                    # the hold began, it is still overlapping next step, and the
                    # step after, for the whole duration. Counting that as a
                    # fresh offence makes the escalation measure how long the
                    # remedy has been failing, then use that number to hold them
                    # for even longer.
                    #
                    # This is what ran away on 2026-09-17: REPEAT OFFENCE #133
                    # through #170 on consecutive steps, one per step, holds
                    # climbing 665 -> 845 while nothing moved at all. The pair
                    # had not collided 170 times -- it had collided ONCE and been
                    # unable to separate for 170 steps.
                    if (a.id in self.currently_yielding
                            and b.id in self.currently_yielding):
                        worst = max(worst, self.pair_collision_counts.get(
                            frozenset((a.id, b.id)), 0))
                        continue
                    key = frozenset((a.id, b.id))
                    self.pair_collision_counts[key] = self.pair_collision_counts.get(key, 0) + 1
                    worst = max(worst, self.pair_collision_counts[key])
        return worst

    def _assign_yields(self, conflicted_nodes: List[Any], repeat_count: int) -> Dict[str, Any]:
        """
        Picks a right-of-way winner and returns hold durations for everyone else.

        THE PROBLEM THIS SOLVES: Tier 1 was reporting "Spatial Escape Successful"
        and then the same pair collided again 5-40 steps later, over and over --
        fleets 9 and 24 at steps 107, 142, 177, 189, 203 inside one episode, with
        Tier3 recorded as 0 across all 30 episodes. Tier 1 was not failing; it was
        succeeding at the wrong thing. It separates two fleets by one cell, then
        both immediately steer back toward their goals along the same converging
        paths. The missing ingredient was never SPACE, it was TIME.

        So on a repeat offence the two are now combined: the fleets still get
        physically separated by whichever tier resolves the geometry, AND the
        losers are held still afterwards so the winner can clear the contested
        region before they resume. Duration escalates with how many times this
        specific pair has already collided, so a stubborn pair is separated for
        progressively longer rather than retrying instantly forever.

        Winner is the fleet closest to its goal by TRUE graph distance (see
        node_warehouse.get_graph_distance_to_goal()), so the fleet that is nearly
        home finishes and retires, permanently removing it from the conflict.
        """
        # Python's sort is STABLE: on a distance tie the input order survives,
        # and the input order was self.nodes -- so the right-of-way winner of a
        # tied conflict depended on list position. Break ties by id.
        if self.order_independent:
            ordered = sorted(conflicted_nodes,
                             key=lambda n: (n.get_graph_distance_to_goal(), str(n.id)))
        else:
            ordered = sorted(conflicted_nodes, key=lambda n: n.get_graph_distance_to_goal())
        if not ordered:
            return {"winner": None, "yield_durations": {}}

        # ANTI-STARVATION. "Closest to goal wins" is deterministic, so in a
        # REPEATED pairwise conflict the same fleet loses every single time. In
        # the deployment run, fleets 9 and 24 collided four times, fleet 24 won
        # all four, and fleet 9 -- held for 10+15+20 steps and rewound repeatedly
        # -- ended the rollout 14 hops from its goal, FURTHER out than when the
        # conflict started. It was never once allowed through.
        #
        # So on a repeat offence for a two-fleet conflict, right-of-way alternates:
        # whoever yielded last time gets priority this time. Both fleets make
        # progress in turn instead of one being permanently sacrificed. Larger
        # pile-ups keep the plain closest-to-goal ordering, since there is no
        # unambiguous "last loser" to alternate against.
        if len(ordered) == 2:
            key = frozenset((ordered[0].id, ordered[1].id))
            last_loser = self.pair_last_loser.get(key)
            if last_loser is not None and last_loser == ordered[0].id:
                # The fleet that would win again is the one that yielded last
                # time... no change needed, it already lost. Nothing to swap.
                pass
            elif last_loser is not None and last_loser == ordered[1].id:
                # The default winner won last time too -- give way to the fleet
                # that yielded, so it finally gets a turn.
                ordered = [ordered[1], ordered[0]]

        winner, losers = ordered[0], ordered[1:]
        if len(conflicted_nodes) == 2:
            self.pair_last_loser[frozenset((ordered[0].id, ordered[1].id))] = losers[0].id
        durations: Dict[str, int] = {}
        for loser in losers:
            # yield_escalation_per_repeat was accepted by __init__ but never read
            # by anything -- the old code multiplied base_yield_steps by a node
            # count instead. This makes the configured knob live: 1st offence
            # holds for base_yield_steps, each repeat adds yield_escalation_per_repeat.
            durations[loser.id] = min(
                self.max_yield_steps,
                self.base_yield_steps
                + self.yield_escalation_per_repeat * max(0, repeat_count - 1),
            )
        return {"winner": winner.id, "yield_durations": durations}

    def assess_loop_coherence(self, nodes: List[Any], current_integrity: float):
        """Records the discrete grid state for temporal rewinds."""
        snapshot = {
            "integrity": current_integrity,
            "positions": {n.id: n.current_pos.copy() for n in nodes}
        }
        self.history.append(snapshot)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def collapse_and_reinitialize(
        self, 
        nodes: List[Any], 
        warning_nodes: Set[str], 
        deadlocked_nodes: Set[str],
        density_field: Any,
        frozen_node_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the 3-Tier Discrete Recovery Protocol:
        1. Spatial Escape (Move to a safe adjacent node)
        2. Temporal Rewind (Back up to a previous safe node)
        3. Right-of-Way Yield (The User's Freeze-and-Yield Tiebreaker)
        """
        self.total_collapses += 1

        # BUG THIS FIXES: conflicted_ids used to be warning_nodes.union(deadlocked_nodes)
        # -- so ANY fleet merely in the cautious WARNING zone with some UNRELATED pair
        # got swept into Tier 1/2/3 disruption right alongside the fleets ACTUALLY in a
        # fatal collision, even though it had nothing to do with the collision that
        # triggered this specific recovery event. In a warehouse this busy, one fatal
        # collision between fleet A and B could also forcibly rewind (Tier 2) or
        # force-yield (Tier 3) fleet C, D, E... who were just cautiously near some
        # OTHER, unrelated pair -- losing their own genuine progress for a situation
        # they weren't part of. Only genuinely deadlocked fleets get recovered now;
        # everyone else keeps moving under their own policy, uninterrupted. This
        # doesn't reduce safety: Tier 1's escape-spot check below still sees ALL
        # fleets (including warning-zone ones) via the full `nodes` list, which was
        # already independent of conflicted_nodes -- narrowing this only stops
        # FORCING warning-zone fleets to move, it doesn't blind anything to their
        # presence.
        conflicted_ids = set(deadlocked_nodes)
        conflicted_nodes = [n for n in nodes if n.id in conflicted_ids]
        # ORDER. Tier 1 escapes these one at a time, each seeing the escapes
        # already made -- that sequencing is what stops two fleets taking one
        # cell, so it stays. But it used to follow the order of self.nodes, so
        # the first fleet in the list got first pick of the escape cells. With
        # order_independent, the sequence is fixed by fleet id instead: still
        # sequential, no longer dependent on where a fleet sits in the list.
        # (A JOINT assignment -- every fleet's escape chosen at once -- would be
        # the optimal version; this is the order-independent one.)
        if self.order_independent:
            conflicted_nodes.sort(key=lambda n: str(n.id))

        # Record this as another fatal-collision offense for every node that was
        # genuinely deadlocked (not just in the wider warning set) -- this is what
        # Tier 3 uses below to escalate yield duration for repeat offenders.
        for node_id in deadlocked_nodes:
            self.node_collision_counts[node_id] = self.node_collision_counts.get(node_id, 0) + 1
        
        # How many times has this specific PAIR already collided this episode?
        pair_repeats = self._record_colliding_pairs(conflicted_nodes)
        if self.escalate_on_recurrence:
            _recur = self._record_recurrences(conflicted_nodes)
            if (_recur >= self.pair_escalation_threshold
                    and pair_repeats < self.pair_escalation_threshold):
                # escalating ONLY because of recurrence -- the fatal counter
                # alone would not have. This is the case that used to loop.
                self.recurrence_escalations += 1
            pair_repeats = max(pair_repeats, _recur)
        escalate = pair_repeats >= self.pair_escalation_threshold

        uninvolved_warning_count = len(warning_nodes - deadlocked_nodes)
        # Name the fleets, not just the count. "for 2 nodes" tells you a collapse
        # happened but not to whom, so a repeating collapse is indistinguishable
        # from two different pairs colliding once each -- and those need opposite
        # responses. With ids you can see at a glance whether the same pair keeps
        # re-converging (a Tier 1 that succeeds and immediately undoes itself)
        # or whether congestion is spread across the fleet.
        _ids = ", ".join(sorted(str(n.id) for n in conflicted_nodes))
        print(f"[Recovery] Initiating Spatial-Temporal Collapse for "
              f"{len(conflicted_nodes)} fleet(s) [{_ids}] "
              f"({uninvolved_warning_count} nearby warning-zone fleets left undisturbed)...")
        if escalate:
            print(f"[Recovery] REPEAT OFFENCE #{pair_repeats} for this pair -- "
                  f"separation will be enforced in TIME as well as space.")

        # Defaults to empty for callers that don't pass it (e.g. standalone tests) --
        # matches the prior hardcoded set() behavior in that case.
        if frozen_node_ids is None:
            frozen_node_ids = set()

        # On a repeat offence, settle right-of-way BEFORE the tiers run, so the
        # winner can be exempted from being moved at all.
        #
        # BUG THIS FIXES: "right of way" used to mean only "you are not held
        # still". Tier 2 rewinds every node in conflicted_nodes, winner included,
        # so the fleet with priority was still teleported backwards. In the
        # deployment rollout fleet 24 won right-of-way all four times, was rewound
        # on two of them, and still finished 5.5 hops short. Priority that undoes
        # your progress is not priority. The winner now keeps its position and its
        # momentum, and only the losers are displaced, rewound and held.
        escalated_yields = self._assign_yields(conflicted_nodes, pair_repeats) if escalate else None
        winner_id = escalated_yields["winner"] if escalated_yields else None
        tier_nodes = ([n for n in conflicted_nodes if n.id != winner_id]
                      if winner_id is not None else conflicted_nodes)
        winner_node = next((n for n in conflicted_nodes if n.id == winner_id), None)

        # ---------------------------------------------------------
        # TIER 1: SPATIAL ESCAPE
        # ---------------------------------------------------------
        # Snapshot positions so a partial success (some fleets already moved) can be
        # rolled back cleanly if a later fleet in the list has no safe escape -- otherwise
        # a failed Tier 1 could leave the batch in an inconsistent half-moved state.
        pre_tier1_positions = {n.id: n.current_pos.copy() for n in tier_nodes}
        spatial_success = True
        for node in tier_nodes:
            # Get valid adjacent neighbors from the NetworkX graph
            curr_tuple = tuple(np.round(node.current_pos).astype(int))
            if curr_tuple not in node.grid_pos_dict:
                continue
                
            curr_node_id = node.grid_pos_dict[curr_tuple]
            safe_escape_found = False
            # Who this escape is actually for -- the other fleets in this
            # conflict, excluding the one being moved.
            _conflict_ids = {n.id for n in tier_nodes if n.id != node.id}

            # RANKED SELECTION (2026-08-28). This used to take the FIRST neighbour
            # scoring above spatial_safe_threshold, in whatever arbitrary order
            # node.G.neighbors() happened to yield. Two problems with that:
            #
            #   1. Arbitrary order meant that when several neighbours qualified,
            #      the one picked had nothing to do with which was actually
            #      safest -- a fleet could escape one cell TOWARD the peer it had
            #      just collided with, purely because that neighbour came first
            #      in the adjacency list.
            #   2. Combined with the old saturating clip(1-R) affordance, where a
            #      fresh crash zeroed everything within 3 cells, NO neighbour ever
            #      qualified at all, so this loop reliably fell through to Tier 2.
            #
            # Scoring every neighbour and taking the argmax fixes both: the escape
            # is now the genuinely best available cell, and the threshold degrades
            # to a floor that only rejects a move when even the best option is bad.
            candidates = []
            for neighbor_id in node.G.neighbors(curr_node_id):
                # O(1) via the precomputed reverse index (falls back to a full scan
                # only if a node somehow wasn't built with one -- see node_warehouse.py).
                if node.coords_by_id is not None:
                    n_coords = node.coords_by_id.get(neighbor_id)
                    neighbor_coords = [n_coords] if n_coords is not None else []
                else:
                    neighbor_coords = [coords for coords, nid in node.grid_pos_dict.items() if nid == neighbor_id]
                if not neighbor_coords:
                    continue
                    
                n_pos = np.array(neighbor_coords[0], dtype=np.float32)

                # OBSTACLES ARE A FACT, NOT A PREFERENCE.
                # Candidates come from G.neighbors(), and obstacles are overlaid
                # on the graph rather than removed from it, so an obstacle cell
                # is still a neighbour. The action mask vetoes moving onto one;
                # recovery bypasses the action mask entirely. Found by a test
                # that had been passing only by luck: once a trajectory put a
                # fleet into a collision beside an obstacle, Tier 1 teleported it
                # straight onto the obstacle -- which, for a human, is exactly
                # what the obstacle work exists to prevent. Same principle as the
                # occupancy check below: a soft score cannot stand in for a hard
                # constraint.
                if tuple(int(v) for v in np.round(n_pos)) in self.static_obstacles:
                    self.tier1_rejected_obstacle += 1
                    continue
                
                # Check affordance of this neighbor cell.
                # BUG THIS FIXES (earlier): this used to pass set() for
                # frozen_node_ids, meaning a genuinely frozen (parked) fleet
                # sitting in this neighbor cell got treated as an ordinary
                # "active peer" instead of the frozen-obstacle logic --
                # inconsistent with what the main perception loop (and every
                # other affordance check) actually sees. Passing the real frozen
                # set and node's own goal makes "safe to escape to" mean the same
                # thing everywhere.
                #
                # own_id matters especially here: this evaluates a NEIGHBOUR cell,
                # not the fleet's own, so the density field's legacy
                # position-equality self-check never matched and the fleet counted
                # ITSELF as an obstacle in every escape it considered -- suppressing
                # exactly the cells adjacent to where it currently stands, which is
                # all of them.
                local_affordances = density_field.get_local_affordance(
                    n_pos, nodes, frozen_node_ids,
                    own_goal_pos=node.goal_pos,
                    frozen_obstacle_severity=self.frozen_obstacle_severity,
                    near_goal_radius=self.frozen_near_goal_radius,
                    own_id=node.id,
                )
                # The density field exposes the exact centre index; len//2 is the
                # correct fallback for the symmetric grids it returns, and keeps
                # this working against an older density_warehouse.py.
                center_idx = getattr(density_field, "center_index", len(local_affordances) // 2)

                # OCCUPANCY IS A FACT, NOT A PREFERENCE.
                #
                # The affordance score alone does NOT keep a fleet out of an
                # occupied cell. affordance = 1/(1+R), and a peer standing on the
                # cell contributes R = peer_severity = 0.7, which scores 0.588 --
                # comfortably above spatial_safe_threshold = 0.35. Two peers
                # score 0.461 and still pass. It takes THREE stacked peers before
                # the threshold refuses.
                #
                # So Tier 1 has been relocating fleets ONTO other fleets and
                # reporting "Spatial Escape Successful". That is the mechanism
                # behind the 12-18 fleet blobs: recovery was not failing to
                # disperse the pile, it was building it. Every conclusion drawn
                # from those logs was measuring recovery's own damage.
                #
                # The threshold was RECALIBRATED when affordance became 1/(1+R),
                # and the new scale is compressed enough that no single value of
                # it can separate "one fleet here" from "several fleets nearby".
                # A soft score cannot express a hard constraint -- same lesson as
                # obstacles, where repulsion routes around and the action mask
                # forbids. This is the forbid.
                # CLEARANCE REQUIRED OF AN ESCAPE CELL.
                #
                # At collision_threshold the escape only has to stop the fleets
                # OVERLAPPING -- it can leave them one cell apart, still inside
                # warning_threshold, still braking, still about to re-converge.
                #
                # Measured on cold_run10: mean displacement 1.02 cells against a
                # warning_threshold of 2.0, and 41% of two-fleet recovery events
                # involved a pair already seen 5+ times -- one pair 45 times.
                # 968 of 2,732 relocations were at x=29 or x=5, the two shaft
                # columns, with 701 moves along z. A shaft is ONE CELL WIDE, so
                # G.neighbors() offers only up and down the same shaft: a
                # one-hop escape cannot separate two fleets in it, it just
                # slides them along and they re-converge.
                #
                # Requiring warning_threshold clearance makes Tier 1 FAIL in
                # that geometry instead of reporting success, which is what
                # escalates it to the temporal rewind that can actually resolve
                # it. Default stays at collision_threshold, exactly reproducing
                # the old behaviour.
                # TWO DIFFERENT QUESTIONS, TWO DIFFERENT DISTANCES.
                #
                #   every fleet, at collision_threshold -- never land ON anyone.
                #       A hard constraint. Any fleet, conflicted or not.
                #
                #   the CONFLICTING fleets, at warning_threshold -- did this
                #       escape actually separate the pair it was called for?
                #       Only the fleets in THIS conflict. Bystanders who happen
                #       to be nearby are irrelevant to whether the pair is
                #       resolved.
                #
                # The first version applied the wider distance to every fleet,
                # which at 40 fleets means almost any cell has somebody within
                # 2.0 -- Tier 1 succeeded once in 500 steps instead of fifty
                # times. It had effectively been switched off rather than made
                # selective.
                #
                # Scoped to the conflict, it fails where it should: in a
                # one-cell-wide shaft the conflicting peer is on the only
                # neighbours that exist, so no candidate clears and Tier 1
                # escalates to the temporal rewind. In an open corridor a
                # bystander two cells away no longer vetoes a perfectly good
                # escape.
                _too_close = False
                for other in nodes:
                    if other.id == node.id or other.id in frozen_node_ids:
                        continue
                    _d = float(np.sum(np.abs(n_pos - other.current_pos)))
                    if _d <= self.collision_threshold:
                        _too_close = True
                        break
                    if (self.escape_clearance > 0.0
                            and other.id in _conflict_ids
                            and _d <= self.escape_clearance):
                        _too_close = True
                        break
                if _too_close:
                    self.tier1_rejected_occupied += 1
                    continue

                candidates.append((float(local_affordances[center_idx]), n_pos))

            if candidates:
                best_score, best_pos = max(candidates, key=lambda c: c[0])
                if best_score > self.spatial_safe_threshold:
                    # A GENUINE NO-OP IS A FAILURE, NOT A SUCCESS.
                    #
                    # Candidates come from G.neighbors(), so the current cell is
                    # not normally among them -- but a fleet sitting MID-EDGE
                    # rounds to a cell whose neighbour set can contain the very
                    # cell it already occupies, and then "escaping" leaves it
                    # exactly where it was. Tier 1 then reports success,
                    # force_repair() sets integrity back to 1.0, nothing has
                    # moved, and the identical collision fires again next step.
                    #
                    # Measured at 25.7% of relocations in the 2026-09-13 run
                    # (13,047 of 50,720), directly in code. Do NOT try to read
                    # this rate off the log: the relocation line truncates with
                    # int(), so a real 8.5 -> 8.0 move also prints as (8)->(8)
                    # and is indistinguishable from a no-op. Hence the counter.
                    if float(np.sum(np.abs(best_pos - node.current_pos))) < 1e-6:
                        self.tier1_noop_escapes += 1
                        spatial_success = False
                        break
                    node.current_pos = best_pos
                    node.direction = np.zeros(3, dtype=np.float32)  # <--- MOMENTUM KILL-SWITCH
                    safe_escape_found = True
                    self.tier1_real_escapes += 1

            if not safe_escape_found:
                spatial_success = False
                break
                
        if not spatial_success:
            # Undo any moves already made this tier before falling through to Tier 2.
            for node in tier_nodes:
                node.current_pos = pre_tier1_positions[node.id].copy()

        if spatial_success:
            self.spatial_recoveries += 1
            result = {"mode": "spatial", "success": True}
            if escalate:
                # Tier 1 alone has demonstrably failed for this pair already --
                # it keeps "succeeding" and the pair keeps re-converging. Keep the
                # spatial separation it just achieved, but ALSO hold the losers so
                # the winner can clear the contested region first.
                result.update(escalated_yields)
                self.yield_recoveries += 1
                print(f"[Recovery] Tier 1: Spatial Escape Successful + enforced hold "
                      f"(winner {winner_id} untouched, holds {escalated_yields['yield_durations']}).")
            else:
                # ROUND, do not truncate. int(8.5) is 8, so a fleet that really
                # moved from 8.5 to 8.0 used to print as (8)->(8) and look like a
                # no-op. That artifact cost a wrong diagnosis; the log now shows
                # the half-cell.
                def _fmt(p):
                    return tuple(round(float(v), 1) for v in p)
                _moved = ", ".join(
                    f"{n.id}:{_fmt(pre_tier1_positions[n.id])}"
                    f"->{_fmt(n.current_pos)}"
                    for n in tier_nodes)
                print(f"[Recovery] Tier 1: Spatial Escape Successful. Moved {_moved}")
            return result

        # ---------------------------------------------------------
        # TIER 2: TEMPORAL REWIND
        # ---------------------------------------------------------
        # If there's nowhere safe to step forward/sideways, step BACKWARDS.
        print("[Recovery] Spatial failed. Attempting Tier 2 Temporal Rewind...")
        
        # Look back in history for a state where these SPECIFIC nodes were safe
        safe_snapshot = None
        for i in range(len(self.history)-1, -1, -1):
            snap = self.history[i]
            is_safe = True

            # Only tier_nodes are actually rewound, so a candidate snapshot must
            # keep them clear of EACH OTHER at their historical positions...
            for n1 in tier_nodes:
                for n2 in tier_nodes:
                    if n1.id != n2.id and n1.id in snap["positions"] and n2.id in snap["positions"]:
                        dist = np.sum(np.abs(snap["positions"][n1.id] - snap["positions"][n2.id]))
                        if dist <= self.collision_threshold:
                            is_safe = False
                            break
                if not is_safe:
                    break

            # ...and off every obstacle NOW. A remembered position can have been
            # walked onto by a human since; rewinding onto it would teleport the
            # fleet into them. Such a snapshot is simply not safe, and if none
            # qualifies, recovery falls through to Tier 3 as designed.
            if is_safe and self.static_obstacles:
                for n in tier_nodes:
                    if n.id in snap["positions"] and tuple(
                            int(v) for v in np.round(snap["positions"][n.id])) in self.static_obstacles:
                        is_safe = False
                        self.tier2_rejected_obstacle += 1
                        break

            # ...and clear of the right-of-way winner at its CURRENT position,
            # since the winner is staying put rather than rewinding with them.
            if is_safe and winner_node is not None:
                for n in tier_nodes:
                    if n.id in snap["positions"]:
                        dist = np.sum(np.abs(snap["positions"][n.id] - winner_node.current_pos))
                        if dist <= self.collision_threshold:
                            is_safe = False
                            break

            if is_safe:
                safe_snapshot = snap
                break
                
        if safe_snapshot:
            for node in tier_nodes:
                if node.id in safe_snapshot["positions"]:
                    # 1. Teleport physically
                    node.current_pos = safe_snapshot["positions"][node.id].copy()
                    
                    # 2. Kill momentum
                    node.direction = np.zeros(3, dtype=np.float32) 
                    
                    # 3. Synchronize internal time-state
                    node.last_pos = node.current_pos.copy() 
                    node.last_action = 0 
                    
                    # 4. Patch the VDA 5050 logs
                    curr_tuple = tuple(np.round(node.current_pos).astype(int))
                    if curr_tuple in node.grid_pos_dict:
                        node_id_string = node.grid_pos_dict[curr_tuple]
                        node.trajectory_history.append(f"REWIND_{node_id_string}")
            
            self.temporal_recoveries += 1
            result = {"mode": "temporal", "success": True}
            if escalate:
                result.update(escalated_yields)
                self.yield_recoveries += 1
                print(f"[Recovery] Tier 2: Temporal Rewind Successful + enforced hold "
                      f"(winner {winner_id} untouched, holds {escalated_yields['yield_durations']}).")
            else:
                _rw = ", ".join(
                    f"{n.id}->{tuple(int(v) for v in n.current_pos)}"
                    for n in tier_nodes if n.id in safe_snapshot["positions"])
                print(f"[Recovery] Tier 2: Temporal Rewind Successful. Rewound {_rw}")
            return result

        # ---------------------------------------------------------
        # TIER 3: THE RIGHT-OF-WAY YIELD (Fallback)
        # ---------------------------------------------------------
        print("[Recovery] Temporal failed. Executing Tier 3 Right-of-Way Yield...")
        
        # Compare all conflicted nodes based on their TRUE distance to their goals
        # (graph-distance, not straight-line Manhattan -- see
        # node_warehouse.get_graph_distance_to_goal() for why that distinction
        # matters; the same detour-punishing issue applies here: a fleet needing a
        # long detour could look closer by Manhattan distance while actually
        # needing more real steps than a fleet with no detour to navigate).
        # Sort them: Lowest distance (closest to goal) comes first
        # Guard against an empty conflict set: every conflicted node can be skipped
        # above (e.g. all of them mid-move onto non-node coordinates), and indexing
        # [0] on an empty list would raise instead of falling through gracefully.
        if not conflicted_nodes:
            return {"mode": "none", "success": False}
        
        # The node closest to its goal gets "Right of Way" (keeps its position/momentum);
        # every other conflicted node is held still for an escalating number of steps.
        yields = escalated_yields or self._assign_yields(conflicted_nodes, max(pair_repeats, 1))
        winner_id = yields["winner"]
        yield_durations = yields["yield_durations"]
        losers = [n for n in conflicted_nodes if n.id in yield_durations]
        
        # BUG THIS FIXES: this used to just zero the loser's direction for the
        # current step -- but core_warehouse.step() immediately recomputes a fresh
        # GNN action for every node right after this returns, which overwrote that
        # zeroed direction anyway (core_warehouse.py enforces the actual idle via
        # forced_yield_ids/_yield_until, not via this direction reset -- this reset
        # is kept only because other code may still read .direction before the next
        # action is chosen). The REAL fix is duration: previously a loser held still
        # for exactly one step, then was free to immediately retry the same losing
        # move -- which is exactly what the repeated identical collisions in the
        # logs showed (e.g. the same pair colliding 3 times in one episode). Now the
        # yield escalates with repeat offenses: 1st offense holds for
        # base_yield_steps, 2nd for 2x that, 3rd for 3x, and so on, so a fleet that
        # keeps ending up back in the same conflict gets progressively longer forced
        # separation instead of an instant retry.
        #
        # NOTE: yield_escalation_per_repeat is currently accepted but unused --
        # the escalation multiplies base_yield_steps by the repeat count directly
        # (1st=5, 2nd=10, 3rd=15 at base_yield_steps=5), which is what CONFIG's
        # comment describes. Kept in the signature so callers don't break.
        for loser in losers:
            loser.direction = np.zeros(3, dtype=np.float32)  # Kill momentum

        self.yield_recoveries += 1
        print(f"[Recovery] Tier 3: Fleet {winner_id} given Right-of-Way. "
              f"Others yielding for {yield_durations} steps.")

        return {"mode": "yield", "success": True,
                "winner": winner_id, "yield_durations": yield_durations}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Recovery-tier breakdown. Worth logging per episode: the ratio of
        spatial:temporal is the single clearest signal of whether the affordance
        field is giving Tier 1 anything to work with. Before the density rewrite
        this was overwhelmingly temporal, because a saturated field left Tier 1
        with no distinguishable escape.
        """
        return {
            "total_collapses": self.total_collapses,
            "spatial_recoveries": self.spatial_recoveries,
            # A Tier-1 escape that moved the fleet nowhere is a failure wearing a
            # success label. If noop is a large share of real, Tier 1 is spinning
            # and the tangle it reports resolving is still there.
            "tier1_real_escapes": self.tier1_real_escapes,
            "tier1_noop_escapes": self.tier1_noop_escapes,
            "tier1_rejected_occupied": self.tier1_rejected_occupied,
            "tier1_rejected_obstacle": self.tier1_rejected_obstacle,
            "tier2_rejected_obstacle": self.tier2_rejected_obstacle,
            "recurrence_escalations": self.recurrence_escalations,
            "recurrence_pairs_tracked": len(self.pair_recurrence_counts),
            "temporal_recoveries": self.temporal_recoveries,
            "yield_recoveries": self.yield_recoveries,
            "repeat_offenders": {k: v for k, v in self.node_collision_counts.items() if v > 1},
            "repeat_pairs": {
                tuple(sorted(k)): v
                for k, v in self.pair_collision_counts.items() if v > 1
            },
        }