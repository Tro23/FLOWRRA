"""
ablate.py

Answers ONE question: which of the fixes produced the jump from ~25% to ~92%
completion?

WHAT AN ABLATION IS
You take the corrected system, put ONE bug back, run the SAME instances, and see
how much the score drops. The size of the drop is that fix's contribution. It has
nothing to do with fleet failures or handovers -- those are a feature of the
simulator; this is an experiment about the simulator.

WHY IT MATTERS HERE
The 100-episode pilot showed episode 1 already at 0.970 and a completely flat
trend afterwards (slope -0.0002/episode, p=0.449). So the improvement was NOT
produced by retraining -- it was produced by the fixes, and right now it is not
known which ones. "We fixed six things and it got better" is not a result.
"Removing fix X costs Y points of completion" is.

HOW THIS RUNS IT FAIRLY
Every variant sees the IDENTICAL list of (map, scenario, k) instances, chosen up
front from one seed. Without that, variant-to-variant differences would be
swamped by instance-to-instance variance -- completion ranges from 0.62 to 1.00
across instances in the pilot, far more than any plausible ablation effect.
Everything runs in eval mode with a frozen policy, so no learning confounds the
comparison.

Usage:
    python ablate.py --maps-dir all_maps --scens-dir all_scens_v2 \
        --checkpoint checkpoints/flowrra_curriculum.pth --episodes 12
"""

import argparse
import contextlib
import io
import logging
import random
import time

import numpy as np
import pandas as pd
import torch

from config_warehouse import CONFIG
from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from main_runner_warehouse import MapCache, sample_instance

logging.getLogger().setLevel(logging.WARNING)


# A variant is either a set of ablation flags, or {"random_policy": True} which
# ignores the network entirely and acts uniformly at random every step.
#
# WHY THE RANDOM VARIANT IS HERE, and why it may be the most important row:
# during the warm-300 run, gradient agreement fell from 0.789 to 0.359 as the
# epsilon schedule peaked -- roughly two thirds of fleets taking random actions --
# while completion moved only from 0.942 to 0.917. That strongly suggests the
# orders are being delivered by machinery that does not learn: the BFS goal
# gradient, the livelock escape that overrides the policy after 40 stall steps
# and follows that gradient directly, affordance braking, and the three recovery
# tiers. If a FULLY random policy also completes ~0.9, the GNN is contributing
# almost nothing and the honest framing of the whole system changes. This is the
# first question a reviewer will ask, so it is better to answer it here.
VARIANTS = [
    ("all fixes (baseline)",          {}),
    ("RANDOM policy (no network)",    {"random_policy": True}),
    ("GREEDY gradient (no network)",  {"gradient_policy": True}),
    ("braking bug restored",          {"brake_on_immobile": True}),
    ("hardcoded bounds restored",     {"hardcoded_bounds": True}),
    ("both bugs restored",            {"brake_on_immobile": True,
                                       "hardcoded_bounds": True}),
]


def run_variant(label, flags, instances, agent, max_steps):
    flags = dict(flags)
    random_policy = flags.pop("random_policy", False)
    gradient_policy = flags.pop("gradient_policy", False)
    for k in CONFIG["ablation"]:
        CONFIG["ablation"][k] = False
    CONFIG["ablation"].update(flags)

    # eps_min == eps_peak == 1.0 pins epsilon at 1.0 for every episode, so every
    # active fleet samples uniformly from its valid actions and the network's
    # output is never consulted. Everything else -- gradient, livelock escape,
    # braking, recovery -- runs untouched, which is exactly the comparison we
    # want: how much of the result survives with the learned part removed?
    eps_override = {"eps_min": 1.0, "eps_peak": 1.0} if random_policy else {}

    rows = []
    for inst in instances:
        with contextlib.redirect_stdout(io.StringIO()):
            env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="eval",
                          goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                          goal_pool=inst["goal_pool"])
            env.gnn = agent
            agent.reset_episode_state()
            _orig = None
            if eps_override:
                _orig = agent.choose_actions
                agent.choose_actions = (
                    lambda *a, _o=_orig, **kw: _o(*a, **{**kw, **eps_override}))
            if random_policy or gradient_policy:
                # A "no network" baseline must ALSO surrender the graph-level
                # recovery decision, which is a second network output. Leaving it
                # on would compare (heuristic actions + learned recovery) against
                # (learned actions + learned recovery) and attribute the
                # difference entirely to routing. Disabling it leaves the forced
                # fallback, so these baselines still recover from deadlock -- via
                # the orchestrator, exactly as they would with no policy at all.
                env.recovery_policy_enabled = False

            if gradient_policy:
                # THE CONTROL THAT ACTUALLY MATTERS.
                #
                # precompute_goal_distances builds a full BFS distance field per
                # goal, so get_goal_gradient() hands the network the EXACT
                # remaining hop count for every neighbour -- and those six values
                # are part of the state vector. A policy that learned "descend the
                # gradient" is therefore behaviourally indistinguishable from the
                # gradient itself.
                #
                # Random is the wrong baseline: beating noise proves very little
                # when the scaffolding includes a distance oracle, an override
                # that follows it after 40 stall steps, and three recovery tiers.
                # The question a reviewer will ask is whether the GNN beats the
                # three-line heuristic whose output it can already read. This
                # variant is that heuristic: step to the lowest-distance
                # neighbour, idle if none improves. Same action encoding the
                # livelock escape uses (0 = idle, gradient index + 1).
                _orig = agent.choose_actions

                def _grad_actions(*_a, _env=env, **_kw):
                    acts = np.zeros(len(_env.nodes), dtype=np.int64)
                    for i, n in enumerate(_env.nodes):
                        if n.id in _env.immobile_nodes:
                            continue
                        g = np.asarray(n.get_goal_gradient(), dtype=np.float32)
                        acts[i] = int(np.argmax(g)) + 1 if np.any(g > 0) else 0
                    return acts

                agent.choose_actions = _grad_actions
            for _ in range(max_steps):
                env.step(episode_step=1, total_episodes=1)
                if env.is_episode_over():
                    break
        if _orig is not None:
            agent.choose_actions = _orig
        # RESILIENCE COLUMNS. Errors fire during evaluation (prob_per_step is on)
        # but nothing was recording them, so the ablation measured only routing --
        # and routing is the one axis where a three-line distance heuristic is
        # already competitive. The handover machinery is the part with no
        # equivalent in RHCR or PIBT, and "the learned policy rescues better" is
        # a claim that can only be made if it is measured.
        est = env.get_error_statistics()
        rows.append({
            "map": inst["map"],
            "agents": len(env.nodes),
            "completion": len(env.claimed_goals) / max(1, len(env.goal_pool)),
            "collisions": env.loop.total_collisions,
            "steps": env.step_count,
            "errors": est["errors_injected"],
            "handovers": est["handovers_completed"],
            "recalls": est["retired_fleets_recalled"],
            "rec_invocations": est["recovery_invocations"],
            "rec_forced": est["recovery_forced"],
            "rec_preemptive": est["recovery_preemptive"],
            "rec_preempt_success": est["recovery_preemptive_success"],
        })
    df = pd.DataFrame(rows)
    for k in CONFIG["ablation"]:
        CONFIG["ablation"][k] = False
    return df


def main():
    # allow_abbrev=False: argparse otherwise accepts any unambiguous PREFIX of a
    # flag, so "--maps X" was silently swallowed as "--maps-dir X" -- overwriting
    # a --maps-dir given earlier on the same line and sending it looking for a
    # directory named after the map. It fails with FileNotFoundError pointing at
    # listdir, which looks like a missing map rather than a mis-parsed argument.
    # An unknown flag should be an error, not a guess.
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2")
    ap.add_argument("--maps", default="",
                    help="comma-separated map subset to evaluate on; blank = all maps "
                         "found in --maps-dir")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=12,
                    help="instances per variant. Each is replayed under every "
                         "variant, so total episodes = this x 4.")
    ap.add_argument("--agents-min", type=int, default=25)
    ap.add_argument("--agents-max", type=int, default=50)
    ap.add_argument("--agent-sets", default="",
                    help="discrete fleet counts, e.g. '55'. USE THIS. At the density "
                         "the pilot trained on, completion sits at 0.986 with most "
                         "episodes perfect -- every variant returns ~0.98 and the "
                         "ablation cannot discriminate. Run it where there is headroom.")
    ap.add_argument("--max-steps", type=int,
                    default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ablation_results.csv")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    agent_sets = [int(v) for v in args.agent_sets.split(',') if v.strip()]
    cache = MapCache(args.maps_dir, args.scens_dir)
    only = [m.strip() for m in args.maps.split(",") if m.strip()] or None
    names = cache.discover(only)
    print(f"{len(names)} maps. Selecting {args.episodes} instances...")

    rng = random.Random(args.seed)
    instances = []
    while len(instances) < args.episodes:
        m = rng.choice(names)
        k = (rng.choice(agent_sets) if agent_sets
             else rng.randint(args.agents_min, args.agents_max))
        with contextlib.redirect_stdout(io.StringIO()):
            inst = sample_instance(cache, m, k, rng)
        if inst:
            instances.append(inst)
    print(f"Selected: {[(i['map'], len(i['missions'])) for i in instances]}\n")

    # Probe one instance to size the network, then load the frozen policy.
    with contextlib.redirect_stdout(io.StringIO()):
        p = instances[0]
        probe = FLOWRRA(p["G"], p["pos_dict"], p["missions"], mode="eval",
                        goal_distance_maps=p["gdm"], shared_pool_mode=True,
                        goal_pool=p["goal_pool"])
        n0 = probe.nodes[0]
        dim = (len(n0.get_state_vector(probe.nodes))
               + len(probe.density.get_local_affordance(n0.current_pos, probe.nodes, set())))
    _rd = CONFIG["reward_decomposition"]
    agent = GNNAgent(node_feature_dim=dim, edge_feature_dim=0,
                     reward_heads=_rd["heads"], head_weights=_rd["weights"],
                     action_size=CONFIG["gnn"]["action_size"],
                     hidden_dim=CONFIG["gnn"]["hidden_dim"],
                     num_layers=CONFIG["gnn"]["num_layers"],
                     n_heads=CONFIG["gnn"]["num_heads"], dropout=0.0,
                     lr=CONFIG["gnn"]["learning_rate"],
                     gamma=CONFIG["training"]["gamma"],
                     buffer_capacity=1000, batch_size=32)
    agent.load(args.checkpoint)
    print(f"Loaded {args.checkpoint}\n")

    results, frames = [], []
    for label, flags in VARIANTS:
        t0 = time.time()
        df = run_variant(label, flags, instances, agent, args.max_steps)
        df["variant"] = label
        frames.append(df)
        # Handover RATE, not raw count: error injection is stochastic so variants
        # do not see identical error counts, and comparing totals would reward
        # whichever variant happened to get more failures to rescue.
        hrate = df.handovers.sum() / max(1, df.errors.sum())
        results.append({
            "variant": label,
            "completion": df.completion.mean(),
            "sd": df.completion.std(),
            "collisions": df.collisions.mean(),
            "steps": df.steps.mean(),
            "errors": df.errors.sum(),
            "handover_rate": hrate,
            "rec_forced": df.rec_forced.mean(),
        })
        print(f"  {label:<28} completion {df.completion.mean():.3f} "
              f"(sd {df.completion.std():.3f})  coll {df.collisions.mean():5.1f}  "
              f"err {df.errors.sum():3.0f} handover {results[-1]['handover_rate']:.0%}  "
              f"[{time.time()-t0:.0f}s]")

    res = pd.DataFrame(results)
    base = res.completion.iloc[0]
    res["drop_vs_baseline"] = base - res.completion

    print("\n" + "=" * 74)
    print("ABLATION RESULT -- how much completion each fix is worth")
    print("=" * 74)
    print(res.round(3).to_string(index=False))
    print("\nA large drop means that fix carried the improvement. If BOTH single")
    print("ablations drop only slightly but 'both' drops a lot, the fixes interact")
    print("and neither alone explains the result.")

    pd.concat(frames).to_csv(args.out, index=False)
    print(f"\nPer-instance detail written to {args.out}")


if __name__ == "__main__":
    main()