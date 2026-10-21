"""
drive_shortest_path.py -- the harsh test driver, saved and seeded.

    # CONFLICT_DESIGN.md's A/B: 60 fleets, 50_ map, 300 steps, 3 seeds
    python drive_shortest_path.py --arm today \
        --arm rules:conflict.path_warnings=true,conflict.directional_braking=true

    # STREAM_DESIGN.md measurement fix 3: does preemptive recovery prevent anything?
    python drive_shortest_path.py --recovery coin

    # no maps needed: the smoke-test warehouse, for a quick check
    python drive_shortest_path.py --synthetic --agents 24 --steps 80 --seeds 0

    # judge against a chosen arm, or re-judge saved results without rerunning
    python drive_shortest_path.py --baseline aligned --arm version_unrefined \
        --arm aligned:conflict.node_aligned_moves=true --arm rules:...
    python drive_shortest_path.py --judge driver_results/driver_*.csv \
        --baseline aligned --rename today=version_unrefined

NAMES (decided 2026-09-29). The code as it stood before CONFLICT_DESIGN.md is
VERSION_UNREFINED -- a fixed reference for how far things have come, not the
judge. The components are judged against version_unrefined PLUS node-aligned
moves (--baseline), because they cannot work without it.

WHAT IT IS. Every fleet takes one step down its own goal gradient each step and
never yields -- the driver behind the shaft-jam, dock and pair-classification
numbers in STREAM_DESIGN.md. Those came from throwaway scripts in earlier chats;
this is the same idea (measure_order_dependence.py's fixed_actions) as a file,
so the numbers can be reproduced. The environment is the real one, built as
main_runner_warehouse.py builds it: same instance sampling, same order stream,
same order seed formula, same window. No network, no learning, no exploration.

Ties between equally good moves go to the lower action index, so a run is fully
determined by its seed.

RECOVERY (--recovery). The policy head is replaced by a fixed rule. Collisions
are always recovered by the orchestrator whatever the rule; the rule only
decides PREEMPTIVE recovery, at an "opportunity" -- a step with fleets in the
warning band and none colliding.
    never   no preemptive recovery (the default: a clean A/B of traffic rules)
    always  invoke at every opportunity
    coin    invoke with probability --coin-p at each opportunity, independently.
            This is the randomised test (measurement fix 3): because the coin,
            not the situation, decides, invoked and declined opportunities are
            comparable, and the difference in collisions over the next k steps
            is a causal estimate of what one preemptive recovery prevents.

At every opportunity, in every mode, the environment's prevention-window tracker
watches THE WHOLE AT-RISK SET (intention to treat: recovery itself drops convoys
and may move fewer fleets) and records whether, and how many of, those fleets
collide within recovery_policy.prevention_window steps. Labels opp_invoke /
opp_decline.

CAVEATS FOR THE COIN. Windows overlap and at-risk sets share fleets, so events
are not independent: the interval uses a block bootstrap over 25-step blocks,
not the event count. And the estimate is "one extra invocation, against a
background that invokes half the time" -- not "always versus never", which is
what paired --recovery always / never runs measure.

ARMS (--arm). "name" or "name:dotted.key=value,dotted.key=value". Every arm
runs on the same seeds and therefore the same instances and order sequences.
The baseline is --baseline (default: the first arm); every other arm is judged
against it on CONFLICT_DESIGN.md's pre-registered criteria, and a totals table
shows each arm against the baseline and against version_unrefined. Values are JSON (true, 3, 1.5,
"x"); a key that does not already exist in CONFIG is an error, because a typo
would otherwise run the baseline twice under two labels.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import datetime as _dt
import io
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config_warehouse import CONFIG

_PRISTINE = copy.deepcopy(CONFIG)


# ------------------------------------------------------------------ config arms
def parse_arm(spec: str) -> Tuple[str, Dict[str, Any]]:
    name, _, rest = spec.partition(":")
    name = name.strip()
    if not name:
        raise SystemExit(f"--arm {spec!r}: needs a name")
    sets: Dict[str, Any] = {}
    for item in [p for p in rest.split(",") if p.strip()]:
        key, eq, raw = item.partition("=")
        if not eq:
            raise SystemExit(f"--arm {spec!r}: {item!r} is not key=value")
        try:
            val = json.loads(raw.strip())
        except json.JSONDecodeError:
            val = raw.strip()
        sets[key.strip()] = val
    return name, sets


def _restore(dst: dict, src: dict) -> None:
    """Make dst equal src WITHOUT replacing any nested dict: a module that kept a
    reference to a config section must see the restored values too."""
    for k in list(dst):
        if k not in src:
            del dst[k]
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _restore(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply_config(sets: Dict[str, Any]) -> None:
    """Restore CONFIG to its state at import, in place, then apply this arm's
    overrides."""
    _restore(CONFIG, _PRISTINE)
    for dotted, val in sets.items():
        parts = dotted.split(".")
        node = CONFIG
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise SystemExit(f"unknown config key {dotted!r} ({p!r} not found)")
            node = node[p]
        if parts[-1] not in node:
            raise SystemExit(f"unknown config key {dotted!r} -- refusing to create it")
        node[parts[-1]] = val


# ------------------------------------------------------------------ the driver
class ShortestPathDriver:
    """Stands in for GNNAgent. core_warehouse.step() calls exactly these."""

    last_q_values = None     # no forward pass: the tabu veto falls back as it does
                             # when every fleet explored

    def __init__(self, env, recovery: str, coin_p: float, seed: int):
        self.env = env
        self.recovery = recovery
        self.coin_p = float(coin_p)
        self.rng = random.Random(seed * 7919 + 17)
        self.gamma = float(CONFIG["training"]["gamma"])
        self.opportunities = 0
        self.invoked = 0

    def choose_actions(self, node_features=None, adj_matrix=None, episode_number=None,
                       total_episodes=None, node_ids=None, valid_action_masks=None, **_):
        by_id = {n.id: n for n in self.env.nodes}
        out = []
        for nid, mask in zip(node_ids, valid_action_masks):
            grad = by_id[nid].get_goal_gradient()
            cands = [(float(grad[a - 1]), -a) for a in range(1, 7) if mask[a]]
            best = max(cands) if cands else (0.0, 0)
            out.append(-best[1] if best[0] > 0 else 0)
        return np.array(out, dtype=np.int64)

    def choose_recovery(self) -> int:
        env = self.env
        if not (env._step_warning and not env._step_deadlocked):
            return 0
        self.opportunities += 1
        if self.recovery == "always":
            invoke = True
        elif self.recovery == "coin":
            invoke = self.rng.random() < self.coin_p
        else:
            invoke = False
        env._watch_open("opp_invoke" if invoke else "opp_decline", env._step_warning)
        self.invoked += int(invoke)
        return 1 if invoke else 0

    # The environment tells the agent when a fleet parks or resumes; the
    # network uses it, the driver has nothing to update.
    def freeze_node(self, *a, **k):
        pass

    def unfreeze_node(self, *a, **k):
        pass

    def reset_episode_state(self):
        pass


# ------------------------------------------------------------------ instances
def real_instance(args, seed: int):
    from main_runner_warehouse import MapCache, sample_instance, _order_window
    cache = MapCache(args.maps_dir, args.scens_dir)
    if args.map not in cache.discover([args.map]):
        raise SystemExit(f"map {args.map!r} not found in {args.maps_dir} / {args.scens_dir}")
    inst = sample_instance(cache, args.map, args.agents, random.Random(seed))
    if inst is None:
        raise SystemExit(f"could not sample {args.agents} fleets on {args.map}")
    e = cache.get(args.map)
    stream = bool((CONFIG.get("stream") or {}).get("enabled", False))
    kw = dict(mode="training", shared_pool_mode=True, goal_pool=inst["goal_pool"],
              goal_distance_maps=(e["goal_distance_maps"] if stream else inst["gdm"]))
    if stream:
        kw.update(order_bank=e["bank"], order_seed=seed * 1000003 + 1,
                  order_floor_window=window(args, _order_window))
    return inst["G"], inst["pos_dict"], inst["missions"], kw


def synthetic_instance(args, seed: int):
    """The smoke-test warehouse plus an order bank. No files needed."""
    import test_smoke_integration as smoke
    from node_warehouse import precompute_goal_distances
    G, grid, miss, gdm, gp = smoke.build_instance(args.agents, seed)
    kw = dict(mode="training", shared_pool_mode=True, goal_pool=gp, goal_distance_maps=gdm)
    if bool((CONFIG.get("stream") or {}).get("enabled", False)):
        rng = np.random.default_rng(seed + 100)
        nodes = sorted(G.nodes())
        bank = sorted({nodes[int(i)] for i in rng.integers(0, len(nodes), 60)} | set(gp))
        kw.update(goal_distance_maps=precompute_goal_distances(
                      G, [{"goal_node": g} for g in bank]),
                  order_bank=bank, order_seed=seed * 1000003 + 1, order_floor_window=None)
    return G, grid, miss, kw


def window(args, order_window_fn) -> Optional[int]:
    if args.order_window == "none":
        return None
    if args.order_window == "final":
        return order_window_fn(10 ** 6, 10 ** 6)
    return int(args.order_window)


# ------------------------------------------------------------------ one run
def run_one(args, arm: str, sets: Dict[str, Any], seed: int) -> Tuple[Dict[str, Any], list]:
    apply_config(sets)
    random.seed(seed)
    np.random.seed(seed)
    from core_warehouse import FLOWRRA

    quiet = contextlib.redirect_stdout(io.StringIO()) if not args.verbose else contextlib.nullcontext()
    with quiet:
        G, pos, missions, kw = (synthetic_instance(args, seed) if args.synthetic
                                else real_instance(args, seed))
        env = FLOWRRA(G, pos, missions, **kw)
    driver = ShortestPathDriver(env, args.recovery, args.coin_p, seed)
    env.gnn = driver

    t0 = time.perf_counter()
    steps = 0
    for _ in range(args.steps):
        buf = io.StringIO() if not args.verbose else None
        with (contextlib.redirect_stdout(buf) if buf is not None else contextlib.nullcontext()):
            env.step(episode_step=1, total_episodes=1)
        steps += 1
        if env.is_episode_over():
            break
    wall = time.perf_counter() - t0

    est = env.get_error_statistics()
    stream = "stream_deliveries" in est
    row = {
        "arm": arm, "seed": seed, "map": "synthetic" if args.synthetic else args.map,
        "fleets": len(missions), "steps": steps, "recovery": args.recovery,
        "settings": json.dumps(sets, sort_keys=True),
        "ms_per_step": round(1000.0 * wall / max(1, steps), 1),
        "deliveries": est["stream_deliveries"] if stream else len(env.claimed_goals),
        "orders_issued": est.get("stream_orders_issued", len(env.goal_pool)),
        "ideal_deliveries": est.get("stream_ideal_deliveries", float("nan")),
        "efficiency": est.get("stream_efficiency", float("nan")),
        "collisions": env.loop.total_collisions,
        "repeat_offences": est["conflict_repeat_offences"],
        "max_pair_repeat": est["conflict_max_pair_repeat"],
        "holds": est["conflict_holds"],
        "holds_at_cap": est["conflict_holds_at_cap"],
        "hold_steps": est["conflict_hold_steps"],
        "collapses": est["conflict_collapses"],
        "recovery_forced": est["recovery_forced"],
        "recovery_preemptive": est["recovery_preemptive"],
        "preempt_clear_k": est["preempt_clear_k"],
        "preempt_collided_k": est["preempt_collided_k"],
        "risk_steps": est["risk_steps"],
        "warning_steps": est["warning_steps"],
        "errors_injected": est.get("errors_injected", 0),
        "handovers": est.get("handovers_completed", 0),
        "opportunities": driver.opportunities,
        "invoked": driver.invoked,
        **{f"deliv_q{q}": est.get(f"stream_deliv_q{q}", float("nan")) for q in (1, 2, 3, 4)},
    }
    for lab in ("opp_invoke", "opp_decline"):
        for k in ("opened", "collided", "clear", "fleets_watched", "fleets_hit", "pending"):
            row[f"{lab}_{k}"] = est.get(f"watch_{lab}_{k}", 0)
    events = [(arm, seed, lab, opened, n, hit) for (lab, opened, n, hit) in env._watch_log
              if lab.startswith("opp_")]
    return row, events


# ------------------------------------------------------------------ reporting
def criteria(base: List[dict], arm: List[dict]) -> List[Tuple[str, bool, str]]:
    """CONFLICT_DESIGN.md's pre-registered success criteria, arm vs baseline."""
    b = {r["seed"]: r for r in base}
    a = {r["seed"]: r for r in arm}
    seeds = sorted(set(a) & set(b))
    tot = lambda rows, k: sum(rows[s][k] for s in seeds)
    out = []
    rb, ra = tot(b, "repeat_offences"), tot(a, "repeat_offences")
    out.append(("repeat offences fall by at least half", ra <= 0.5 * rb,
                f"{rb} -> {ra}"))
    hb, ha = tot(b, "holds"), tot(a, "holds")
    out.append(("holds fall by at least half", ha <= 0.5 * hb, f"{hb} -> {ha}"))
    rises = sum(1 for s in seeds if a[s]["deliveries"] > b[s]["deliveries"])
    need = math.ceil(2 * len(seeds) / 3)
    out.append((f"deliveries rise on at least {need} of {len(seeds)} seeds", rises >= need,
                " ".join(f"{b[s]['deliveries']}->{a[s]['deliveries']}" for s in seeds)))
    cb, ca = tot(b, "collisions"), tot(a, "collisions")
    out.append(("collisions rise by no more than 10%", ca <= 1.10 * cb, f"{cb} -> {ca}"))
    return out


def block_bootstrap(events: List[tuple], block: int = 25, reps: int = 2000,
                    seed: int = 0) -> Dict[str, Any]:
    """Invoke-minus-decline difference in (a) share of events with any collision
    and (b) share of watched fleets that collided, resampling 25-step blocks."""
    groups: Dict[tuple, list] = {}
    for (arm, s, lab, opened, n, hit) in events:
        groups.setdefault((arm, s, opened // block), []).append((lab, n, hit))
    keys = list(groups)
    if not keys:
        return {}

    def stat(sample_keys):
        acc = {"opp_invoke": [0, 0, 0, 0], "opp_decline": [0, 0, 0, 0]}
        for k in sample_keys:
            for lab, n, hit in groups[k]:
                a = acc[lab]
                a[0] += 1; a[1] += int(hit > 0); a[2] += n; a[3] += hit
        if acc["opp_invoke"][0] == 0 or acc["opp_decline"][0] == 0:
            return None
        ai = acc["opp_invoke"][1] / acc["opp_invoke"][0]
        ad = acc["opp_decline"][1] / acc["opp_decline"][0]
        fi = acc["opp_invoke"][3] / max(1, acc["opp_invoke"][2])
        fd = acc["opp_decline"][3] / max(1, acc["opp_decline"][2])
        return ai, ad, fi, fd, acc

    point = stat(keys)
    if point is None:
        return {}
    rng = random.Random(seed)
    d_any, d_frac = [], []
    for _ in range(reps):
        st = stat([rng.choice(keys) for _ in keys])
        if st is None:
            continue
        d_any.append(st[0] - st[1])
        d_frac.append(st[2] - st[3])
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else float("nan")
    ai, ad, fi, fd, acc = point
    return {"invoke_events": acc["opp_invoke"][0], "decline_events": acc["opp_decline"][0],
            "any_invoke": ai, "any_decline": ad, "any_diff": ai - ad,
            "any_ci": (q(d_any, 0.025), q(d_any, 0.975)),
            "frac_invoke": fi, "frac_decline": fd, "frac_diff": fi - fd,
            "frac_ci": (q(d_frac, 0.025), q(d_frac, 0.975)), "blocks": len(keys)}


def report(rows: List[dict], baseline: Optional[str]) -> None:
    """Totals per arm against the baseline and version_unrefined, then the
    pre-registered criteria for every arm against the baseline."""
    names = list(dict.fromkeys(r["arm"] for r in rows))
    baseline = baseline or names[0]
    if baseline not in names:
        raise SystemExit(f"--baseline {baseline!r} is not one of the arms {names}")
    keys = ("deliveries", "collisions", "repeat_offences", "holds")
    tot = {a: {k: sum(r[k] for r in rows if r["arm"] == a) for k in keys} for a in names}
    ref = "version_unrefined" if "version_unrefined" in names else None

    def pct(a, b):
        return f"{(a - b) / b * 100:+.0f}%" if b else "  n/a"
    print(f"\nTOTALS over seeds (baseline: {baseline}"
          + (f"; reference: {ref})" if ref else ")"))
    print(f"   {'arm':<20}" + "".join(f"{k:>18}" for k in keys))
    for a in names:
        cells = []
        for k in keys:
            c = f"{tot[a][k]}"
            if a != baseline:
                c += f" {pct(tot[a][k], tot[baseline][k])}"
            cells.append(f"{c:>18}")
        print(f"   {a:<20}" + "".join(cells))
    if ref and ref != baseline:
        print(f"   vs {ref}:")
        for a in names:
            if a == ref:
                continue
            print(f"   {a:<20}" + "".join(
                f"{pct(tot[a][k], tot[ref][k]):>18}" for k in keys))
    base = [r for r in rows if r["arm"] == baseline]
    for a in names:
        if a in (baseline, ref):
            continue
        print(f"\nPRE-REGISTERED CRITERIA (CONFLICT_DESIGN.md): {a} vs {baseline}")
        res = criteria(base, [r for r in rows if r["arm"] == a])
        for text, ok, detail in res:
            print(f"   {'PASS' if ok else 'FAIL'}  {text:<44} {detail}")
        print(f"   -> {'ALL MET' if all(ok for _, ok, _ in res) else 'NOT MET'}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v4")
    ap.add_argument("--map", default="50_5_5_10_5_2")
    ap.add_argument("--synthetic", action="store_true",
                    help="the smoke-test warehouse instead of a real map (no files)")
    ap.add_argument("--agents", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--order-window", default="final",
                    help="'final' (the window the judged episodes run at), 'none', or floors")
    ap.add_argument("--recovery", choices=("never", "always", "coin"), default="never")
    ap.add_argument("--coin-p", type=float, default=0.5)
    ap.add_argument("--arm", action="append", default=[],
                    help="'name' or 'name:key=value,...' (repeatable; first is the baseline)")
    ap.add_argument("--out", default="driver_results")
    ap.add_argument("--verbose", action="store_true", help="show the environment's own prints")
    ap.add_argument("--baseline", default=None,
                    help="arm every other arm is judged against (default: the first)")
    ap.add_argument("--judge", nargs="+", default=None,
                    help="re-judge saved driver CSVs instead of running anything")
    ap.add_argument("--rename", action="append", default=[],
                    help="old=new, relabel an arm when judging saved results")
    args = ap.parse_args()

    if args.judge:
        import pandas as pd
        ren = dict(r.split("=", 1) for r in args.rename)
        rows = []
        for path in args.judge:
            rows += pd.read_csv(path).to_dict("records")
        for r in rows:
            r["arm"] = ren.get(r["arm"], r["arm"])
        report(rows, args.baseline)
        return

    arms = [parse_arm(a) for a in (args.arm or ["version_unrefined"])]
    if len({a for a, _ in arms}) != len(arms):
        raise SystemExit("arm names must be unique")
    for _, sets in arms:           # fail on a bad key before hours of running
        apply_config(sets)
    apply_config({})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(args.out, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_path = os.path.join(args.out, f"driver_{stamp}.csv")
    events_path = os.path.join(args.out, f"driver_events_{stamp}.csv")

    print(f"{'synthetic' if args.synthetic else args.map} | {args.agents} fleets | "
          f"{args.steps} steps | seeds {seeds} | recovery={args.recovery}"
          + (f" p={args.coin_p}" if args.recovery == "coin" else "")
          + f" | window={args.order_window} | k={CONFIG['recovery_policy'].get('prevention_window', 5)}")
    rows: List[dict] = []
    events: List[tuple] = []
    fh = None
    for arm, sets in arms:
        for seed in seeds:
            row, ev = run_one(args, arm, sets, seed)
            rows.append(row)
            events += ev
            if fh is None:
                fh = open(runs_path, "w", newline="")
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
            fh.flush()
            eff = row["efficiency"]
            print(f"  {arm:<12} seed {seed}  deliveries {row['deliveries']:>4}"
                  + (f" ({eff * 100:.0f}% of {row['ideal_deliveries']:.0f})" if np.isfinite(eff) else "")
                  + f"  coll {row['collisions']:>4}  repeats {row['repeat_offences']:>4}"
                  f"  holds {row['holds']:>5} (cap {row['holds_at_cap']})"
                  f"  collapses {row['collapses']:>4}  {row['ms_per_step']:.0f} ms/step")
    apply_config({})
    if fh is not None:
        fh.close()
    with open(events_path, "w", newline="") as eh:
        w = csv.writer(eh)
        w.writerow(["arm", "seed", "label", "opened_step", "fleets_watched", "fleets_hit"])
        w.writerows(events)
    print(f"\nwrote {runs_path}\nwrote {events_path}")

    report(rows, args.baseline or arms[0][0])

    if args.recovery in ("coin", "always", "never"):
        k = CONFIG["recovery_policy"].get("prevention_window", 5)
        for arm, _ in arms:
            ev = [e for e in events if e[0] == arm]
            n_inv = sum(1 for e in ev if e[2] == "opp_invoke")
            n_dec = sum(1 for e in ev if e[2] == "opp_decline")
            print(f"\nOPPORTUNITIES ({arm}): {n_inv + n_dec} resolved, "
                  f"{n_inv} invoked, {n_dec} declined; window {k} steps")
            if args.recovery != "coin":
                for lab, n in (("opp_invoke", n_inv), ("opp_decline", n_dec)):
                    if n:
                        hit = sum(1 for e in ev if e[2] == lab and e[5] > 0)
                        print(f"   {lab:<12} any collision within {k}: {hit}/{n} = {hit / n:.1%}")
                continue
            bb = block_bootstrap(ev)
            if not bb:
                print("   too few events in one of the arms to compare")
                continue
            print(f"   any watched fleet collided within {k} steps:  invoked {bb['any_invoke']:.1%}"
                  f"   declined {bb['any_decline']:.1%}   difference {bb['any_diff']:+.1%}"
                  f"   95% block-bootstrap [{bb['any_ci'][0]:+.1%}, {bb['any_ci'][1]:+.1%}]")
            print(f"   share of watched fleets that collided:        invoked {bb['frac_invoke']:.1%}"
                  f"   declined {bb['frac_decline']:.1%}   difference {bb['frac_diff']:+.1%}"
                  f"   95% [{bb['frac_ci'][0]:+.1%}, {bb['frac_ci'][1]:+.1%}]")
            print(f"   ({bb['blocks']} blocks of 25 steps; negative = preemption prevents collisions)")


if __name__ == "__main__":
    main()
