"""
jam_thrml.py -- the collapse, sampled with thrml (Extropic's block Gibbs
sampler), on the two-boundary jams built by jam_collapse.py. On small jams it
is checked against the exact plans before we trust it on bigger ones.

THE MODEL (a discrete energy-based model, one variable per fleet per tick)
    X[i, k] = where fleet i is at tick k, for k = 1 .. T.
    Tick 0 (its start cell) is fixed: the first boundary.
    Each variable's choices are the cells fleet i could be in at tick k:
    reachable from its start in k hops and able to reach its exit in the T-k
    ticks left (the "light cone"). That is the superposition. One more choice,
    OUT, means the fleet has reached its exit and driven on: from then on it
    holds no cell, as the recording stops following a fleet once it is out of
    the cluster. At tick T every fleet must be at its exit or out: the second
    boundary.

    The rules become energies (weights are -energy, as thrml expects):
      one fleet, tick to tick   stay or move one hop; OUT only from its
                                exit, and OUT stays OUT                  (pairwise)
      two fleets, same tick     not the same cell                        (pairwise)
      two fleets, next tick     not into the cell the other held         (pairwise)
      clamped neighbours,
      people and debris         cells forbidden at that tick          (one-variable)
    and the cost: 1 for every tick a fleet is not yet at its exit (one-variable).

    "No entering a cell another fleet held a tick earlier" needs only pairs:
    a fleet can only be in that cell now by having entered it, because two
    fleets could not share it a tick earlier.

THE COLLAPSE
    Block Gibbs with annealing: blocks are (fleet, even ticks) and (fleet, odd
    ticks), so no two variables in a block touch. The weights are multiplied
    by beta, raised from 0.1 to 10 over the schedule.
      * Starts: half the chains start from shortest-path plans (each fleet
        drives a shortest path to its exit after a random wait at its start;
        chain 0 waits for nobody), half from random cells.
      * Rules: half the chains keep the rules hard from the start; the other
        half pay a finite penalty for a broken rule while hot, so they can get
        out of tight spots (one fleet backing into a side aisle, say).
      * The plan: the final state with no broken rule and the lowest cost,
        over all chains. A jam with no legal plan is run once more with twice
        the chains and twice the steps.

USE
    python jam_thrml.py DAY.json.gz PROBLEMS.json.gz --horizon own   [--chains 32] [--steps 600]
    python jam_thrml.py DAY.json.gz COLLAPSE_EXACT.json.gz --horizon exact   (check against exact plans)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

INVALID = -1e9
OUT = "out"
# Timing of the last collapse() call, in seconds. The hard and soft halves of
# the chains run as two calls of the same compiled program: the first call
# includes JAX's compile, the second does not, so compile = first - second.
TIMING: dict = {}          # same token as jam_collapse.OUT


def load(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as fh:
        return json.load(fh)


def build(F, P, nb, no_go):
    """Domains, the cost of each choice, and which pairs of choices keep the
    rules. Returns (model, None), or (None, reason) when no plan is possible."""
    mem = P["members"]
    n, T = len(mem), P["T"]
    start = [P["start"][f] for f in mem]
    exit_ = [P["exit"][f] for f in mem]
    ds = [F.dist(s) for s in start]
    de = [F.dist(e) for e in exit_]
    nbs = [set(x) for x in nb]
    ngs = [set(x) for x in no_go]
    if len(set(start)) < n or any(c in nbs[0] for c in start):
        return None, "start cells clash with recorded neighbours"
    dom = {}
    for i in range(n):
        for k in range(1, T + 1):
            cells = [c for c in ds[i] if ds[i][c] <= k and de[i].get(c, 10 ** 6) <= T - k
                     and c not in ngs[k] and c not in nbs[k] and c not in nbs[k - 1]]
            if k == 1:
                cells = [c for c in cells if c == start[i] or c in F.adj[start[i]]]
                cells = [c for c in cells if c not in {start[j] for j in range(n) if j != i}]
            cells = sorted(cells)
            if ds[i].get(exit_[i], 10 ** 6) <= k - 1:
                cells.append(OUT)           # it could have reached its exit a tick earlier
            if not cells:
                return None, "a fleet has nowhere to be at some tick"
            dom[i, k] = cells
    K = max(len(v) for v in dom.values())
    keys = [(i, k) for i in range(n) for k in range(1, T + 1)]
    where = {key: r for r, key in enumerate(keys)}
    cost = np.zeros((len(keys), K), dtype=np.float32)     # 1 per tick not yet at the exit
    pad = np.ones((len(keys), K), dtype=bool)               # categories beyond a variable's choices
    for r, (i, k) in enumerate(keys):
        for a, c in enumerate(dom[i, k]):
            cost[r, a] = 0.0 if c in (exit_[i], OUT) else 1.0
            pad[r, a] = False
    pairs_l, pairs_r, tables = [], [], []

    def add(kl, kr, ok):
        W = np.zeros((K, K), dtype=bool)                    # True = the pair keeps the rule
        for a, ca in enumerate(dom[kl]):
            for b, cb in enumerate(dom[kr]):
                W[a, b] = ok(ca, cb)
        pairs_l.append(where[kl]); pairs_r.append(where[kr]); tables.append(W)

    def own(i):
        def ok(a, b):
            if a == OUT:
                return b == OUT
            if b == OUT:
                return a == exit_[i]
            return a == b or b in F.adj[a]
        return ok

    apart = lambda a, b: a == OUT or b == OUT or a != b
    for i in range(n):
        for k in range(1, T):
            add((i, k), (i, k + 1), own(i))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(1, T + 1):
                add((i, k), (j, k), apart)
            for k in range(1, T):
                add((i, k + 1), (j, k), apart)       # i may not enter the cell j held
                add((j, k + 1), (i, k), apart)
    return {"keys": keys, "dom": dom, "K": K, "cost": cost, "pad": pad,
            "pl": np.array(pairs_l), "pr": np.array(pairs_r), "ok": np.stack(tables) if tables else None}, None


def energy_of(M, idx):
    """(rule breaks, cost) of an assignment (one index per variable)."""
    breaks = int(sum(M["pad"][r, a] for r, a in enumerate(idx)))
    if M["ok"] is not None:
        breaks += int(sum(not M["ok"][q, idx[l], idx[r]] for q, (l, r) in enumerate(zip(M["pl"], M["pr"]))))
    return breaks, float(sum(M["cost"][r, a] for r, a in enumerate(idx) if not M["pad"][r, a]))


def warm_starts(F, P, M, chains, rng):
    """Starting states. Even chains: every fleet drives a shortest path to its
    exit, after a random wait at its start (chain 0: no waits), then leaves.
    Where that cell is not allowed at that tick, a random allowed one is used.
    Odd chains: random cells, as before."""
    keys, dom, T = M["keys"], M["dom"], P["T"]
    sizes = np.array([len(dom[key]) for key in keys])
    init = (rng.random((chains, len(keys))) * sizes).astype(np.uint8)
    for c in range(0, chains, 2):
        paths = []
        for f in P["members"]:
            s, e = P["start"][f], P["exit"][f]
            de = F.dist(e)
            spare = max(0, T - de.get(s, 0))
            wait = 0 if c == 0 else int(rng.integers(0, spare + 1))
            seq, cur = [s] * (wait + 1), s
            while cur != e:
                nxt = sorted(q for q in F.adj[cur] if de.get(q, 10 ** 6) == de[cur] - 1)
                cur = nxt[0] if c == 0 else nxt[int(rng.integers(len(nxt)))]
                seq.append(cur)
            paths.append(seq)
        for r, (i, k) in enumerate(keys):
            want = paths[i][k] if k < len(paths[i]) else OUT
            if want in dom[i, k]:
                init[c, r] = dom[i, k].index(want)
    return init


def collapse(F, P, M, chains, steps, sweeps, seed, penalty, init=None):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from thrml import Block, BlockGibbsSpec, CategoricalNode, FactorSamplingProgram, SamplingSchedule, sample_states
    from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional

    keys, K = M["keys"], M["K"]
    nodes = [CategoricalNode() for _ in keys]
    # Weights are -energy. A category beyond a variable's choices is never
    # allowed. Half the chains keep the rules hard from the start; the other
    # half pay a finite `penalty` (times beta) for a broken rule, so while hot
    # they can pass through a broken rule on the way to a better legal plan,
    # and at beta 10 a break costs 10 x penalty. Hard chains polish plans near
    # their start; soft chains get out of tight spots. The best of both wins.
    unary = jnp.array(np.where(M["pad"], INVALID, -M["cost"]).astype(np.float32))
    has_pairs = M["ok"] is not None
    if has_pairs:
        ok = jnp.array(M["ok"])
        left = [nodes[i] for i in M["pl"]]
        right = [nodes[i] for i in M["pr"]]
    groups = []                                  # (fleet, parity) -> node positions
    for i in range(len(P["members"])):
        for par in (0, 1):
            g = [r for r, (fi, k) in enumerate(keys) if fi == i and k % 2 == par]
            if g:
                groups.append(g)
    free_blocks = [Block([nodes[r] for r in g]) for g in groups]
    all_block = Block(nodes)

    # The weight tables go into the compiled program as ARGUMENTS, not as
    # constants baked into it: baked in, every jam's tables became part of the
    # machine code JAX generates, which made compiling slower and heavier.
    def program(beta, pen, unary, ok):
        factors = [CategoricalEBMFactor([Block(nodes)], beta * unary)]
        if has_pairs:
            factors.append(CategoricalEBMFactor([Block(left), Block(right)], beta * jnp.where(ok, 0.0, -pen)))
        spec = BlockGibbsSpec(free_blocks, [])
        return FactorSamplingProgram(spec, [CategoricalGibbsConditional(K) for _ in free_blocks], factors, [])

    sched = SamplingSchedule(n_warmup=sweeps, n_samples=1, steps_per_sample=1)
    betas = jnp.logspace(-1, 1, steps)
    # Starting states: given by the caller (the live rung builds its own), or
    # shortest-path and random starts for the offline jams.
    init = jnp.array(init if init is not None else warm_starts(F, P, M, chains, np.random.default_rng(seed)))

    @eqx.filter_jit
    def run(key, init, pen, unary, ok):
        n = init.shape[0]

        def step(state, inp):
            sk, beta = inp
            prog = program(beta, pen, unary, ok)
            blocks = [state[:, jnp.array(g)] for g in groups]
            ck = jax.random.split(sk, n)
            new = jax.vmap(lambda kk, *b: sample_states(kk, prog, sched, list(b), [], [all_block])[0][0])(ck, *blocks)
            return new, None

        final, _ = jax.lax.scan(step, init, (jax.random.split(key, steps), betas))
        return final

    k_hard, k_soft = jax.random.split(jax.random.key(seed))
    half = chains // 2 if penalty > 0 else chains
    TIMING.clear()
    t1 = time.perf_counter()
    ok_arg = ok if has_pairs else None
    parts = [np.array(run(k_hard, init[:half], jnp.float32(-INVALID), unary, ok_arg))]
    TIMING["first"] = time.perf_counter() - t1
    if half < chains:
        t2 = time.perf_counter()
        parts.append(np.array(run(k_soft, init[half:], jnp.float32(penalty), unary, ok_arg)))
        TIMING["second"] = time.perf_counter() - t2
    if half == chains - half and "second" in TIMING:
        TIMING["compile"] = max(0.0, TIMING["first"] - TIMING["second"])
        TIMING["sample"] = 2 * TIMING["second"]
    final = np.concatenate(parts)
    # MEMORY (found in the live rung on AWS, 22 October). Every jam has its own
    # shape, so every call compiles a new program, and JAX keeps every compiled
    # program loaded: about 350 more memory maps and 20-100 MB per call. After
    # a couple of hundred calls the process hits the system's limit on memory
    # maps and the compiler fails ("LLVM compilation error: Cannot allocate
    # memory"). Dropping the compiled programs after each call keeps it flat.
    del run
    jax.clear_caches()
    best, best_e = None, (10 ** 9, 0.0)
    for c in range(chains):
        idx = [int(v) for v in final[c]]
        e = energy_of(M, idx)
        if e < best_e:
            best, best_e = idx, e
    return best, best_e



# ----------------------------------------------------------------- B2: compile once
# THE PROBLEM B2 SOLVES. collapse() compiles a new program for every jam,
# because every jam has its own shape (fleets, ticks, choices per variable).
# Live, that was two-thirds of a thrml decision (AWS: median 10.5 s compiling,
# 5.7 s sampling). B2 pads every jam up to one of a few fixed sizes, so each
# size is compiled ONCE per process and reused by every later jam of that size.
#   fleets:  2 to 8, exactly  (padding fleets cost too much: measured on the
#            CPU, padding 5 fleets to 8 tripled the sampling time, because every
#            fleet's variables are tied to every other fleet's)
#   choices: 32, 40, 48, 64, 80, 96, 128, 160 or 192 per variable (at most
#            about 1.25 x the choices a jam needs; padded choices are never allowed)
# Ticks are not padded: the live rung always plans 8. So a run compiles at most
# 7 x 9 programs, each once, instead of one per jam. The padded program samples
# exactly the same model: a padded choice has weight -1e9.
# (Dummy fleets are still supported, for a size list with gaps: one choice,
# and every rule with them holds.)
BUCKET_FLEETS = (2, 3, 4, 5, 6, 7, 8)
BUCKET_CHOICES = (32, 40, 48, 64, 80, 96, 128, 160, 192)
_RUNNERS: dict = {}


def _structure(n: int, T: int):
    """Variables and pairs for n fleets over ticks 1..T, in the same order as
    build() and the live rung make them."""
    keys = [(i, k) for i in range(n) for k in range(1, T + 1)]
    pairs = []
    for i in range(n):
        for k in range(1, T):
            pairs.append(((i, k), (i, k + 1)))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(1, T + 1):
                pairs.append(((i, k), (j, k)))
            for k in range(1, T):
                pairs.append(((i, k + 1), (j, k)))
                pairs.append(((j, k + 1), (i, k)))
    return keys, pairs


def _runner(nB: int, T: int, KB: int, chains_half: int, steps: int, sweeps: int):
    """One compiled sampler per size, kept for the life of the process."""
    key_ = (nB, T, KB, chains_half, steps, sweeps)
    if key_ in _RUNNERS:
        return _RUNNERS[key_]
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from thrml import Block, BlockGibbsSpec, CategoricalNode, FactorSamplingProgram, SamplingSchedule, sample_states
    from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional

    keys, pairs = _structure(nB, T)
    where = {k: r for r, k in enumerate(keys)}
    nodes = [CategoricalNode() for _ in keys]
    left = [nodes[where[a]] for a, _b in pairs]
    right = [nodes[where[b]] for _a, b in pairs]
    groups = []
    for i in range(nB):
        for par in (0, 1):
            g = [r for r, (fi, k) in enumerate(keys) if fi == i and k % 2 == par]
            if g:
                groups.append(g)
    free_blocks = [Block([nodes[r] for r in g]) for g in groups]
    all_block = Block(nodes)
    sched = SamplingSchedule(n_warmup=sweeps, n_samples=1, steps_per_sample=1)
    betas = jnp.logspace(-1, 1, steps)

    def program(beta, pen, unary, ok):
        factors = [CategoricalEBMFactor([Block(nodes)], beta * unary),
                   CategoricalEBMFactor([Block(left), Block(right)], beta * jnp.where(ok, 0.0, -pen))]
        spec = BlockGibbsSpec(free_blocks, [])
        return FactorSamplingProgram(spec, [CategoricalGibbsConditional(KB) for _ in free_blocks], factors, [])

    @eqx.filter_jit
    def run(key, init, pen, unary, ok):
        def step(state, inp):
            sk, beta = inp
            prog = program(beta, pen, unary, ok)
            blocks = [state[:, jnp.array(g)] for g in groups]
            ck = jax.random.split(sk, chains_half)
            new = jax.vmap(lambda kk, *b: sample_states(kk, prog, sched, list(b), [], [all_block])[0][0])(ck, *blocks)
            return new, None

        final, _ = jax.lax.scan(step, init, (jax.random.split(key, steps), betas))
        return final

    _RUNNERS[key_] = (run, keys, pairs, where)
    return _RUNNERS[key_]


def collapse_bucketed(M, n: int, T: int, chains: int, steps: int, sweeps: int, seed: int, penalty: float,
                      init: np.ndarray):
    """collapse(), padded to a fixed size so the program is compiled once per
    size. M is a model in build()'s format for n fleets over ticks 1..T;
    init is (chains, len(M["keys"])). Returns (idx in M's own key order,
    (rule breaks, cost)), or None when the jam is larger than every size."""
    import jax
    import jax.numpy as jnp
    nB = next((b for b in BUCKET_FLEETS if b >= n), None)
    KB = next((b for b in BUCKET_CHOICES if b >= M["K"]), None)
    if nB is None or KB is None or chains % 2:
        return None
    half = chains // 2
    run, keysB, pairsB, whereB = _runner(nB, T, KB, half, steps, sweeps)
    whereM = {k: r for r, k in enumerate(M["keys"])}
    K = M["K"]
    VB = len(keysB)
    costB = np.zeros((VB, KB), dtype=np.float32)
    padB = np.ones((VB, KB), dtype=bool)
    initB = np.zeros((chains, VB), dtype=np.uint8)
    for rB, key in enumerate(keysB):
        r = whereM.get(key)
        if r is None:                       # a dummy fleet: one choice, cost 0
            padB[rB, 0] = False
            continue
        costB[rB, :K] = M["cost"][r]
        padB[rB, :K] = M["pad"][r]
        initB[:, rB] = init[:, r]
    pairM = {(int(l), int(r)): q for q, (l, r) in enumerate(zip(M["pl"], M["pr"]))} if M["ok"] is not None else {}
    okB = np.ones((len(pairsB), KB, KB), dtype=bool)
    for p, (a, b) in enumerate(pairsB):
        ra, rb = whereM.get(a), whereM.get(b)
        if ra is None or rb is None:
            continue                         # a rule with a dummy fleet always holds
        q = pairM.get((ra, rb))
        if q is not None:
            okB[p] = False
            okB[p, :K, :K] = M["ok"][q]
    unaryB = jnp.array(np.where(padB, INVALID, -costB).astype(np.float32))
    okJ = jnp.array(okB)
    k_hard, k_soft = jax.random.split(jax.random.key(seed))
    TIMING.clear()
    t1 = time.perf_counter()
    a = np.array(run(k_hard, jnp.array(initB[:half]), jnp.float32(-INVALID), unaryB, okJ))
    TIMING["first"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    b = np.array(run(k_soft, jnp.array(initB[half:]), jnp.float32(penalty), unaryB, okJ))
    TIMING["second"] = time.perf_counter() - t2
    TIMING["compile"] = max(0.0, TIMING["first"] - TIMING["second"])   # about 0 once this size has run
    TIMING["sample"] = 2 * TIMING["second"]
    TIMING["bucket"] = (nB, KB)
    final = np.concatenate([a, b])
    rows = [whereB[k] for k in M["keys"]]
    best, best_e = None, (10 ** 9, 0.0)
    for c in range(chains):
        idx = [int(final[c, rB]) for rB in rows]
        e = energy_of(M, idx)
        if e < best_e:
            best, best_e = idx, e
    return best, best_e

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("day")
    ap.add_argument("problems", help="output of jam_collapse.py (with or without exact plans)")
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600, help="annealing steps (beta 0.1 -> 10)")
    ap.add_argument("--sweeps", type=int, default=2, help="Gibbs sweeps per annealing step")
    ap.add_argument("--limit", type=int, default=0, help="only the first N jams (testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-retry", action="store_true",
                    help="do not re-run a jam with no legal plan (the retry uses 2x chains and 2x steps)")
    ap.add_argument("--penalty", type=float, default=20.0,
                    help="energy of a broken rule (times beta) in the soft half of the chains; 0 = all chains hard")
    ap.add_argument("--jams", default="", help="only these jam numbers, e.g. 275,280")
    ap.add_argument("--horizon", choices=["exact", "own"], default="own",
                    help="own: shortest path + slack; exact: the exact plan's length (to check thrml)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from jam_collapse import Day
    day = Day(load(a.day))
    F = day.floor
    E = load(a.problems)
    probs = E["problems"][: a.limit] if a.limit else E["problems"]
    if a.jams:
        want = {int(x) for x in a.jams.split(",")}
        probs = [p_ for p_ in probs if p_["P"]["jam"] in want]
    rows = []
    t_start = time.time()
    for q, pr in enumerate(probs):
        P = dict(pr["P"])
        # Check mode: the same horizon as the exact plan, so the exact plan is
        # one of the plans the sampler can find, and the best cost is known.
        if a.horizon == "exact":
            if "ticks_exact" not in pr:
                continue
            P["T"] = pr["ticks_exact"]
        if P["T"] < 1:
            rows.append({"jam": P["jam"], "size": len(P["members"]), "T": P["T"], "legal": True,
                         "cost_thrml": 0.0, "cost_exact": pr.get("cost_exact"), "trivial": True})
            continue
        t_build = time.perf_counter()
        M, why = build(F, P, pr["nb"], pr["no_go"])
        build_s = time.perf_counter() - t_build
        if M is None:
            rows.append({"jam": P["jam"], "size": len(P["members"]), "status": "no plan possible: " + why})
            print(f"[thrml] jam {P['jam']:>5}: {len(P['members'])} fleets: no plan possible ({why})", flush=True)
            continue
        if M["K"] > 255:
            rows.append({"jam": P["jam"], "size": len(P["members"]), "status": "light cone too wide for uint8"})
            continue
        t_jam = time.perf_counter()
        idx, (breaks, e) = collapse(F, P, M, a.chains, a.steps, a.sweeps, a.seed + q, a.penalty)
        tm = dict(TIMING)
        retried = False
        if breaks and not a.no_retry:            # one more go: twice the chains, twice as slow a cooling
            idx, (breaks, e) = collapse(F, P, M, 2 * a.chains, 2 * a.steps, a.sweeps, a.seed + q + 100_000, a.penalty)
            tm = {k: tm.get(k, 0.0) + v for k, v in TIMING.items()}
            retried = True
        seconds = time.perf_counter() - t_jam
        legal = breaks == 0
        cost_exact = pr.get("cost_exact")
        plan = []
        for i, f in enumerate(P["members"]):
            seq = [P["start"][f]]
            for k in range(1, P["T"] + 1):
                c = M["dom"][i, k][idx[M["keys"].index((i, k))]]
                if c == OUT:
                    break                       # it has left: the rest of its ticks are not tracked
                seq.append(c)
            plan.append(seq)
        rows.append({"jam": P["jam"], "size": len(P["members"]), "T": P["T"], "K": M["K"],
                     "variables": len(M["keys"]), "legal": legal, "cost_thrml": round(e, 3) + 0.0 if legal else None,
                     "cost_exact": cost_exact, "retried": retried, "plan": plan,
                     "seconds": round(seconds + build_s, 3), "build_s": round(build_s, 3),
                     "compile_s": round(tm.get("compile", float("nan")), 3),
                     "sample_s": round(tm.get("sample", float("nan")), 3)})
        r = rows[-1]
        if legal and "recorded" in pr:
            dex = {f: F.dist(P["exit"][f]) for f in P["members"]}
            mv = aw = 0
            for i, f in enumerate(P["members"]):
                seq = plan[i]
                for p_, q_ in zip(seq, seq[1:]):
                    if q_ != p_:
                        mv += 1
                        aw += dex[f].get(q_, 0) > dex[f].get(p_, 0)
            r.update(driven_thrml=mv, away_thrml=aw,
                     driven_recorded=sum(v["driven"] for v in pr["recorded"].values()),
                     away_recorded=sum(v["away"] for v in pr["recorded"].values()),
                     shortest=sum(P["shortest"].values()), zone=pr.get("zone_of") is not None)
        msg = "no legal plan found" + (" (also on the longer retry)" if retried else "")
        if legal:
            msg = f"cost {r['cost_thrml']:.0f}" + (f" vs exact {cost_exact}" if cost_exact is not None else "")
            if "driven_recorded" in r:
                msg += f" | driven {r['driven_thrml']} vs recorded {r['driven_recorded']:.1f} (shortest {r['shortest']})"
            if retried:
                msg += " [found on the longer retry]"
        print(f"[thrml] jam {P['jam']:>5}: {len(P['members'])} fleets, {len(M['keys'])} variables (K {M['K']}): {msg}"
              f" | {rows[-1]['seconds']:.1f} s", flush=True)
    ok = [r for r in rows if r.get("legal")]
    print(f"[thrml] {len(rows)} jams in {time.time() - t_start:.0f} s: legal plan in {len(ok)}, "
          f"no plan possible in {sum('status' in r for r in rows)}, "
          f"not found by the sampler in {sum(r.get('legal') is False for r in rows)}")
    timed = sorted(r["seconds"] for r in rows if "seconds" in r)
    if timed:
        comp = [r["compile_s"] for r in rows if r.get("compile_s") == r.get("compile_s") and "compile_s" in r]
        samp = [r["sample_s"] for r in rows if r.get("sample_s") == r.get("sample_s") and "sample_s" in r]
        med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
        bld = [r["build_s"] for r in rows if "build_s" in r]
        print(f"[thrml]   decision time per jam: median {med(timed):.1f} s, longest {timed[-1]:.1f} s. Medians of "
              f"the parts: building the model {med(bld):.2f} s, JAX compiling {med(comp):.1f} s, "
              f"sampling {med(samp):.1f} s")
    withx = [r for r in ok if r.get("cost_exact") is not None]
    if withx:
        same = sum(abs(r["cost_thrml"] - r["cost_exact"]) < 1e-6 for r in withx)
        print(f"[thrml]   against the exact plan: matched in {same}, worse in {len(withx) - same}")
    cmp_ = [r for r in ok if "driven_recorded" in r]
    for label, sel in (("all", cmp_), ("2 fleets", [r for r in cmp_ if r["size"] == 2]),
                       ("3 fleets", [r for r in cmp_ if r["size"] == 3]),
                       ("4+ fleets", [r for r in cmp_ if r["size"] >= 4]),
                       ("inside congestion zones", [r for r in cmp_ if r.get("zone")])):
        if sel:
            S = lambda k: sum(x[k] for x in sel)
            print(f"[thrml]   {label} ({len(sel)} jams): driven {S('driven_thrml')} hops vs recorded "
                  f"{S('driven_recorded'):.1f} (shortest {S('shortest')}); away from exits "
                  f"{S('away_thrml')} vs {S('away_recorded'):.1f}")
    if cmp_:
        # Per jam. Recorded distances are physical and can be off by up to half
        # a hop at each end per fleet (mid-hop starts and exits), so a jam
        # counts as "about the same" within half a hop per fleet.
        fewer = sum(r["driven_thrml"] < r["driven_recorded"] - 0.5 * r["size"] for r in cmp_)
        more = sum(r["driven_thrml"] > r["driven_recorded"] + 0.5 * r["size"] for r in cmp_)
        print(f"[thrml]   per jam: the plan drove less in {fewer}, about the same in {len(cmp_) - fewer - more}, "
              f"more in {more}")
    out = a.out or (a.problems.replace("collapse_exact_", "collapse_thrml_") if "collapse_exact_" in a.problems
                    else a.problems.replace(".json", "_thrml.json"))
    with gzip.open(out, "wt") as fh:
        json.dump(rows, fh)
    print(f"[thrml] -> {out}")


if __name__ == "__main__":
    main()