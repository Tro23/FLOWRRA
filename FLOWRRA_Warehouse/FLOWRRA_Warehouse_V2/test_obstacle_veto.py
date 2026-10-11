"""
test_obstacle_veto.py -- a fleet must never be allowed to move toward a human.

FOUND BY test_obstacles_integration.py (2026-10-11): the veto in
FleetNode.get_valid_action_mask() rounded the half-step position, and Python
rounds halves to EVEN (round(2.5) == 2). From x = 2, half a step toward 3 was
checked as cell 2 -- the fleet's own -- so the move into a human on cell 3 was
allowed. Now the veto checks the next node along the move (_entering_cell).

WHAT THIS CHECKS, on both parities of coordinate and on every axis:
  1. standing on a node, the move toward an obstacle on the next node is vetoed;
  2. halfway along an edge, the move on into the obstacle is vetoed, the move
     back is not;
  3. with the next cell free, the move is allowed;
  4. the old rounding really did miss the even case (documents the bug).

Run:  python test_obstacle_veto.py      (ends with ALL PASS)
"""

import numpy as np

from node_warehouse import ACTION_DELTAS, FleetNode

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


class Stub:
    """Just what get_valid_action_mask() reads; every edge exists."""
    _entering_cell = FleetNode._entering_cell

    def __init__(self, pos, obstacles):
        self.current_pos = np.array(pos, dtype=np.float32)
        self.static_obstacles = set(obstacles)

    def is_structurally_valid(self, proposed_pos, action):
        return True


def allowed(pos, obstacles, action):
    return bool(FleetNode.get_valid_action_mask(Stub(pos, obstacles), 0.5)[action])


def test_every_axis_both_parities():
    for ax, (up, down) in enumerate([(1, 2), (3, 4), (5, 6)]):
        for start in (2, 3):                       # even and odd coordinates
            pos = [5, 5, 5]
            pos[ax] = start
            nxt_up, nxt_down = list(pos), list(pos)
            nxt_up[ax] += 1
            nxt_down[ax] -= 1
            tag = f"axis{ax}_from{start}"
            check(f"{tag}_veto_up", allowed(pos, {tuple(nxt_up)}, up), False)
            check(f"{tag}_veto_down", allowed(pos, {tuple(nxt_down)}, down), False)
            check(f"{tag}_free_up_allowed", allowed(pos, {tuple(nxt_down)}, up), True)


def test_halfway_along_an_edge():
    check("halfway_on_into_obstacle_vetoed", allowed([2.5, 0, 0], {(3, 0, 0)}, 1), False)
    check("halfway_back_allowed", allowed([2.5, 0, 0], {(3, 0, 0)}, 2), True)
    check("just_left_a_node_on_into_obstacle", allowed([2.05, 0, 0], {(3, 0, 0)}, 1), False)
    check("braked_near_the_obstacle_still_vetoed", allowed([2.95, 0, 0], {(3, 0, 0)}, 1), False)


def test_old_rounding_missed_the_even_case():
    def old_cell(pos, action):
        p = np.array(pos, dtype=np.float32) + ACTION_DELTAS[action] * 0.5
        return tuple(int(round(float(v))) for v in p)
    check("old_veto_from_even_checked_own_cell", old_cell([2, 0, 0], 1), (2, 0, 0))
    check("old_veto_from_odd_was_right", old_cell([3, 0, 0], 1), (4, 0, 0))
    check("new_veto_from_even", Stub([2, 0, 0], ())._entering_cell(1), (3, 0, 0))


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))