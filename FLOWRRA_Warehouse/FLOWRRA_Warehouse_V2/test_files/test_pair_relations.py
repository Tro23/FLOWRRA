"""
test_pair_relations.py -- step 3c: pair relations on the attention connections,
plus the next-state edge fix.

WHAT 3c ADDS (config conflict.ladder.pair_relations; needs ladder.enabled)
  Six numbers after the existing four on every attention connection i -> j:
      5 head-on   6 contested   7 blocked   8 following      (one-hot, symmetric)
      9 "I (i) wait for j"      10 "j waits for me (i)"      (swap with the view)
  From the path verdicts: head-on = both wait; blocked = the mover waits for the
  blocker; following = the follower waits for the leader; contested = no wait.

ALSO (config gnn.next_state_edge_features): the stored NEXT state had kept only
the bare adjacency mask, so value targets never saw a pairwise number while every
choice did. On, the next state gets the same edge features, from the after-move
verdicts. (Exercised end to end by test_smoke_integration; here, the switch.)

WHAT THIS CHECKS
  1. Off: 1 + 4 channels, as before.
  2. On: 1 + 4 + 6 channels, and the existing four are unchanged.
  3. Head-on, blocked, following and contested land in the right channels, with
     the wait direction the right way round from both sides.
  4. Unwarned verdicts add nothing.
  5. The next-state switch is read from the config.

Run:  python test_pair_relations.py      (ends with ALL PASS)
"""
import numpy as np

from config_warehouse import CONFIG
from conflict_warehouse import Verdict
from core_warehouse import FLOWRRA
from test_smoke_integration import build_instance

FAIL = []
H, C, B, F, IW, WM = 5, 6, 7, 8, 9, 10


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_env(pair=True, next_edges=False):
    saved_l = dict(CONFIG["conflict"].get("ladder", {}) or {})
    saved_g = CONFIG["gnn"].get("next_state_edge_features", False)
    CONFIG["conflict"]["ladder"] = {"enabled": True, "shadow": False, "retrace": False,
                                    "context": False, "pair_relations": pair,
                                    "grace_checks": 1, "trail_length": 8,
                                    "aging_reset_steps": 3, "l3_stuck_steps": 30}
    CONFIG["gnn"]["next_state_edge_features"] = next_edges
    try:
        G, gp, missions, gdm, pool = build_instance(24)
        env = FLOWRRA(G, gp, missions, mode="training", goal_distance_maps=gdm,
                      shared_pool_mode=True, goal_pool=pool)
    finally:
        CONFIG["conflict"]["ladder"] = saved_l
        CONFIG["gnn"]["next_state_edge_features"] = saved_g
    env.proximity.refresh(env.nodes, excluded_ids=env.immobile_nodes)
    return env


def V(a, b, kind, warn=True, follower=None, blocker=None):
    return Verdict(a, b, 2.0, kind, warn, "kind", None, follower, blocker)


def edges(env, verdicts=None):
    n = len(env.nodes)
    return env._build_edge_features(np.ones((n, n), dtype=np.float32) - np.eye(n), verdicts)


def test_widths_and_base_unchanged():
    off = make_env(pair=False)
    on = make_env(pair=True)
    e_off, e_on = edges(off), edges(on)
    check("off_channels", e_off.shape[-1], 1 + 4)
    check("on_channels", e_on.shape[-1], 1 + 4 + 6)
    check("existing_four_unchanged", bool(np.allclose(e_off, e_on[..., :5])), True)


def test_relations_and_wait_direction():
    env = make_env(pair=True)
    ids = [n.id for n in env.nodes]
    a, b, c, d = ids[0], ids[1], ids[2], ids[3]
    ia, ib, ic, i_d = 0, 1, 2, 3
    vs = [V(a, b, "head_on"),
          V(a, c, "blocked", blocker=c),
          V(a, d, "following", follower=a),
          V(b, c, "contested"),
          V(c, d, "head_on", warn=False)]
    e = edges(env, vs)
    check("head_on_both_ways", (e[ia, ib, H], e[ib, ia, H]), (1.0, 1.0))
    check("head_on_both_wait", (e[ia, ib, IW], e[ia, ib, WM], e[ib, ia, IW]), (1.0, 1.0, 1.0))
    check("blocked_marked", (e[ia, ic, B], e[ic, ia, B]), (1.0, 1.0))
    check("mover_waits_for_blocker", (e[ia, ic, IW], e[ic, ia, WM], e[ic, ia, IW]), (1.0, 1.0, 0.0))
    check("follower_waits_for_leader", (e[ia, i_d, F], e[ia, i_d, IW], e[i_d, ia, WM]), (1.0, 1.0, 1.0))
    check("contested_no_wait", (e[ib, ic, C], e[ib, ic, IW], e[ib, ic, WM]), (1.0, 0.0, 0.0))
    check("unwarned_adds_nothing", float(e[ic, i_d, H] + e[i_d, ic, H]), 0.0)


def test_next_state_switch():
    check("next_state_edges_off_by_default", make_env().next_state_edges, False)
    check("next_state_edges_on", make_env(next_edges=True).next_state_edges, True)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            print(f"\n--- {fn.__name__} ---")
            fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))