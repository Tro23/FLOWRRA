"""
sweep.py

Searches the reward space with MANY parameters at once, instead of one 12-hour
run per coefficient.

WHY THIS EXISTS
One-variable-at-a-time is the right method for ATTRIBUTION -- ablations, bug
hunts, "did the braking fix cause this". It is the wrong method for TUNING, and
the project crossed from one to the other several runs ago. Four reward
coefficients explored one per overnight run is four bits of information for
two days of compute, and it cannot see interactions at all: invocation_cost and
preemptive_bonus only make sense as a ratio, and OFAT can never find that.

WHAT THIS DOES INSTEAD
  * Optuna TPE over all reward parameters jointly.
  * Multi-fidelity: a trial reports its objective every `report_every` episodes
    and Optuna's median pruner kills it if it is tracking below the median of
    completed trials. Bad configs die in ~10 episodes rather than 150.
  * Cheap proxy: small map, low fleet count, short episodes. Tuning does not
    need the full instance, it needs enough signal to RANK configs.
  * Parallel workers if you have the cores.

THE OBJECTIVE IS THE HARD PART, AND IT IS WHY THIS MAY NOT HELP
completion_rate sits at ~0.98 with two thirds of episodes perfect, and
collisions at ~0.5 per episode. Neither has the dynamic range to rank configs;
optimising against a saturated metric is expensive noise no matter how good the
search is. The default objective is therefore weighted toward the quantities
measured to actually VARY -- handover rate and forced-recovery count -- with
completion as a floor constraint rather than the thing being maximised.

If your objective does not move across trials, the answer is not a better
optimiser. It is a harder task.

Usage:
    python sweep.py --maps-dir all_maps --scens-dir all_scens_v2 \
        --map 25_10_5_10_5_2 --trials 40 --episodes 40 --max-steps 300
"""

import argparse
import contextlib
import io
import json
import os
import random
import time

import numpy as np
import optuna
import torch

from config_warehouse import CONFIG
from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from main_runner_warehouse import MapCache, sample_instance

optuna.logging.set_verbosity(optuna.logging.WARNING)


# The parameters searched, and the ranges. Ratios matter more than absolutes
# here: invocation_cost only means something relative to preemptive_bonus, and
# pickup_reward only relative to mission_complete. Ranges are wide on purpose --
# a narrow range around the current value assumes the current value is roughly
# right, which is exactly what is in question.
SEARCH_SPACE = {
    "fatal_collision":      ("float", -60.0, -2.0),
    "warning_zone":         ("float", -4.0, -0.1),
    "pickup_reward":        ("float", 10.0, 150.0),
    "invocation_cost":      ("float", -10.0, -0.25),
    "preemptive_bonus":     ("float", 1.0, 30.0),
    "resolution_bonus":     ("float", 1.0, 30.0),
    "w_safety":             ("float", 0.25, 4.0),
    "w_integrity":          ("float", 0.25, 4.0),
    "w_rescue":             ("float", 0.25, 4.0),
}


def apply_params(p):
    """Write a trial's parameters into CONFIG before building the env."""
    CONFIG["rewards"]["fatal_collision"] = p["fatal_collision"]
    CONFIG["rewards"]["warning_zone"] = p["warning_zone"]
    CONFIG["errors"]["pickup_reward"] = p["pickup_reward"]
    CONFIG["recovery_policy"]["invocation_cost"] = p["invocation_cost"]
    CONFIG["recovery_policy"]["preemptive_bonus"] = p["preemptive_bonus"]
    CONFIG["recovery_policy"]["resolution_bonus"] = p["resolution_bonus"]
    heads = CONFIG["reward_decomposition"]["heads"]
    w = list(CONFIG["reward_decomposition"]["weights"])
    for name, key in (("safety", "w_safety"), ("integrity", "w_integrity"),
                      ("rescue", "w_rescue")):
        if name in heads:
            w[heads.index(name)] = p[key]
    CONFIG["reward_decomposition"]["weights"] = w
    return heads, w


def objective_value(stats):
    """
    Composite score. Higher is better.

    Deliberately NOT completion-dominated. Completion is saturated at ~0.98 and
    cannot rank configs; it appears here only as a floor, penalised hard if a
    config actually breaks routing. The terms that carry the ranking are the
    ones measured to vary across runs.
    """
    comp = stats["completion"]
    handover = stats["handover_rate"]
    coll = stats["collisions"]
    forced = stats["forced"]

    score = 0.0
    score += 3.0 * handover                      # varies most; the differentiator
    score -= 0.5 * coll                          # rare but expensive
    score -= 0.5 * forced                        # policy failing to act
    score += 10.0 * min(comp, 0.95)              # floor, saturates at 0.95
    score -= 20.0 * max(0.0, 0.90 - comp)        # cliff if routing breaks
    return score


def run_trial(trial, args, cache, names):
    p = {}
    for k, (kind, lo, hi) in SEARCH_SPACE.items():
        p[k] = trial.suggest_float(k, lo, hi)
    heads, weights = apply_params(p)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    probe = sample_instance(cache, names[0], args.agents, rng)
    with contextlib.redirect_stdout(io.StringIO()):
        penv = FLOWRRA(probe["G"], probe["pos_dict"], probe["missions"], mode="init",
                       goal_distance_maps=probe["gdm"], shared_pool_mode=True,
                       goal_pool=probe["goal_pool"])
        n0 = penv.nodes[0]
        dim = (len(n0.get_state_vector(penv.nodes))
               + len(penv.density.get_local_affordance(n0.current_pos, penv.nodes, set())))
        agent = GNNAgent(node_feature_dim=dim, edge_feature_dim=0,
                         action_size=CONFIG["gnn"]["action_size"],
                         hidden_dim=CONFIG["gnn"]["hidden_dim"],
                         num_layers=CONFIG["gnn"]["num_layers"],
                         n_heads=CONFIG["gnn"]["num_heads"],
                         reward_heads=heads, head_weights=weights,
                         dropout=CONFIG["gnn"]["dropout"],
                         lr=CONFIG["gnn"]["learning_rate"],
                         gamma=CONFIG["training"]["gamma"],
                         buffer_capacity=CONFIG["training"]["buffer_capacity"],
                         batch_size=CONFIG["training"]["batch_size"])
        agent.cold_start = True

    acc = {"completion": [], "collisions": [], "forced": [], "err": 0, "hand": 0}
    for ep in range(1, args.episodes + 1):
        inst = sample_instance(cache, rng.choice(names), args.agents, rng)
        if inst is None:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                          goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                          goal_pool=inst["goal_pool"])
            env.gnn = agent
            agent.reset_episode_state()
            for _ in range(args.max_steps):
                env.step(episode_step=ep, total_episodes=args.episodes)
                if len(agent.memory) >= agent.batch_size:
                    agent.learn(node_ids=[n.id for n in env.nodes])
                if env.is_episode_over():
                    break
        est = env.get_error_statistics()
        acc["completion"].append(len(env.claimed_goals) / max(1, len(env.goal_pool)))
        acc["collisions"].append(env.loop.total_collisions)
        acc["forced"].append(est["recovery_forced"])
        acc["err"] += est["errors_injected"]
        acc["hand"] += est["handovers_completed"]

        if ep % args.report_every == 0:
            stats = {"completion": float(np.mean(acc["completion"])),
                     "collisions": float(np.mean(acc["collisions"])),
                     "forced": float(np.mean(acc["forced"])),
                     "handover_rate": acc["hand"] / max(1, acc["err"])}
            v = objective_value(stats)
            trial.report(v, ep)
            # MULTI-FIDELITY PRUNING. This is where the compute saving is: a
            # config tracking below the median of completed trials is killed
            # after ~10 episodes instead of running the full budget. Most of the
            # search space is bad, and the point is to spend nothing on it.
            if trial.should_prune():
                raise optuna.TrialPruned()

    stats = {"completion": float(np.mean(acc["completion"])),
             "collisions": float(np.mean(acc["collisions"])),
             "forced": float(np.mean(acc["forced"])),
             "handover_rate": acc["hand"] / max(1, acc["err"])}
    for k, v in stats.items():
        trial.set_user_attr(k, v)
    return objective_value(stats)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2")
    ap.add_argument("--map", default="", help="single map to tune on; blank = all")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=40,
                    help="episodes PER TRIAL. Tuning does not need the full 150 -- it "
                         "needs enough signal to RANK configs, which is far less.")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="shorter than training on purpose. Most episodes finish well "
                         "before 780 and the tail is mostly idle fleets.")
    ap.add_argument("--agents", type=int, default=25)
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--warmup-trials", type=int, default=6,
                    help="trials run to completion before pruning starts, so the "
                         "pruner has a median to compare against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sweep_results")
    args = ap.parse_args()

    cache = MapCache(args.maps_dir, args.scens_dir)
    only = [args.map] if args.map else None
    names = cache.discover(only)
    if not names:
        raise SystemExit("no usable maps")
    print(f"tuning on {names} | {args.trials} trials x {args.episodes} episodes "
          f"x {args.max_steps} steps, k={args.agents}")

    baseline = {k: None for k in SEARCH_SPACE}
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=args.warmup_trials,
                                           n_warmup_steps=args.report_every))
    # Seed the search with the CURRENT config so the sweep can never do worse
    # than what you already have, and so trial 0 is a like-for-like reference.
    study.enqueue_trial({
        "fatal_collision": CONFIG["rewards"]["fatal_collision"],
        "warning_zone": CONFIG["rewards"]["warning_zone"],
        "pickup_reward": CONFIG["errors"]["pickup_reward"],
        "invocation_cost": CONFIG["recovery_policy"]["invocation_cost"],
        "preemptive_bonus": CONFIG["recovery_policy"]["preemptive_bonus"],
        "resolution_bonus": CONFIG["recovery_policy"]["resolution_bonus"],
        "w_safety": 1.0, "w_integrity": 1.0, "w_rescue": 1.0,
    })

    t0 = time.time()
    study.optimize(lambda t: run_trial(t, args, cache, names),
                   n_trials=args.trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(
                       f"  trial {t.number:>3} {t.state.name:<9} value "
                       f"{('%.3f' % t.value) if t.value is not None else '   -  '}"
                       f"  {json.dumps({k: round(v, 2) for k, v in t.params.items()})}")])

    os.makedirs(args.out, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(args.out, "sweep_trials.csv"), index=False)
    print(f"\nbest value {study.best_value:.3f} after {(time.time()-t0)/60:.1f}m")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k:<20} {v:.3f}")
    print("\nbest trial stats:", study.best_trial.user_attrs)
    with open(os.path.join(args.out, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nWritten to {args.out}/. Importances (which params actually matter):")
    try:
        for k, v in optuna.importance.get_param_importances(study).items():
            print(f"  {k:<20} {v:.3f}")
    except Exception as exc:
        print(f"  (needs more completed trials: {exc})")


if __name__ == "__main__":
    main()