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
        pair_escalation_threshold: int = 2,
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
            durations[loser.id] = (
                self.base_yield_steps
                + self.yield_escalation_per_repeat * max(0, repeat_count - 1)
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

        # Record this as another fatal-collision offense for every node that was
        # genuinely deadlocked (not just in the wider warning set) -- this is what
        # Tier 3 uses below to escalate yield duration for repeat offenders.
        for node_id in deadlocked_nodes:
            self.node_collision_counts[node_id] = self.node_collision_counts.get(node_id, 0) + 1
        
        # How many times has this specific PAIR already collided this episode?
        pair_repeats = self._record_colliding_pairs(conflicted_nodes)
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
                candidates.append((float(local_affordances[center_idx]), n_pos))

            if candidates:
                best_score, best_pos = max(candidates, key=lambda c: c[0])
                if best_score > self.spatial_safe_threshold:
                    node.current_pos = best_pos
                    node.direction = np.zeros(3, dtype=np.float32)  # <--- MOMENTUM KILL-SWITCH
                    safe_escape_found = True

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
                _moved = ", ".join(
                    f"{n.id}:{tuple(int(v) for v in pre_tier1_positions[n.id])}"
                    f"->{tuple(int(v) for v in n.current_pos)}"
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
            "temporal_recoveries": self.temporal_recoveries,
            "yield_recoveries": self.yield_recoveries,
            "repeat_offenders": {k: v for k, v in self.node_collision_counts.items() if v > 1},
            "repeat_pairs": {
                tuple(sorted(k)): v
                for k, v in self.pair_collision_counts.items() if v > 1
            },
        }