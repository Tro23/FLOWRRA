"""
check_scenario_overlap.py -- is the training set really held out from the test set?

READ-ONLY. Compares two scenario folders (the layout generate_scenarios.py
writes: <folder>/<map>/<map>_StartGoalLocations_Seed<n>.csv), map by map:

  1. IDENTICAL INSTANCES. generate_scenarios.py makes a file from (map, seed, hop
     settings) alone, so two folders generated from the same seeds with the same
     settings contain the SAME instances. A training folder like that has the
     benchmark's own test instances in it. Compared on the set of (start, goal)
     pairs, so a reordered copy still counts as identical.
  2. LARGEST SHARED FRACTION between any training file and any test file. Random
     files on the same map always share a few pairs by chance; a large fraction
     means a near-copy.
  3. JOURNEY LENGTHS (needs --maps-dir): hop distance over sampled pairs, for each
     folder, plus the shortest and longest journey in the first, middle and last
     file by seed. A --min-hops-range spread shows up as the shortest journey
     rising from the first file to the last; a fixed --min-hops as a flat floor.
     That is how to read the settings a folder was generated with, so new files
     can match them.

Usage:
    python check_scenario_overlap.py --train-dir all_scens_v3 --test-dir all_scens_v2
    python check_scenario_overlap.py --train-dir all_scens_v3 --test-dir all_scens_v2 \\
        --maps 25_5_2_5_2_1,50_20_5_10_5_2 --maps-dir all_maps

Ends with HELD OUT if no training file reproduces a test file, LEAK otherwise.
"""

import argparse
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

SEED_RE = re.compile(r"_StartGoalLocations_Seed(\d+)\.csv$")


def scenario_files(folder: str, map_name: str) -> Dict[int, str]:
    """seed -> path, for one map in one folder ({} if the map is absent)."""
    d = os.path.join(folder, map_name)
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in os.listdir(d):
        m = SEED_RE.search(f)
        if m and f.startswith(map_name):
            out[int(m.group(1))] = os.path.join(d, f)
    return dict(sorted(out.items()))


def read_pairs(path: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(path, index_col=False)
    return list(zip(df["startNodeId"].astype(str).str.strip(),
                    df["goalNodeId"].astype(str).str.strip()))


def load_graph(maps_dir: str, map_name: str) -> Optional[nx.Graph]:
    nodes = os.path.join(maps_dir, f"{map_name}_Nodes.csv")
    edges = os.path.join(maps_dir, f"{map_name}_Edges.csv")
    if not (os.path.exists(nodes) and os.path.exists(edges)):
        return None
    nd = pd.read_csv(nodes, index_col=False)
    ed = pd.read_csv(edges, index_col=False)
    G = nx.Graph()
    G.add_nodes_from(nd["NodeId"].astype(str).str.strip())
    G.add_edges_from(zip(ed["nodeFrom"].astype(str).str.strip(),
                         ed["nodeTo"].astype(str).str.strip()))
    return G


def seed_span(seeds: List[int]) -> str:
    return f"Seed{seeds[0]}-Seed{seeds[-1]}" if seeds else "none"


def hop_profile(G: nx.Graph, files: Dict[int, str], per_file: int,
                max_files: int, rng: random.Random):
    """Sampled hop distances: all of them, and (seed, min, max) for a spread of
    files from the first seed to the last."""
    seeds = list(files)
    if len(seeds) > max_files:
        step = (len(seeds) - 1) / (max_files - 1)
        seeds = sorted({seeds[round(k * step)] for k in range(max_files)})
    all_d, per_seed = [], []
    for s in seeds:
        pairs = read_pairs(files[s])
        take = rng.sample(pairs, min(per_file, len(pairs)))
        d = []
        for a, b in take:
            try:
                d.append(nx.shortest_path_length(G, a, b))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        if d:
            all_d.extend(d)
            per_seed.append((s, min(d), max(d)))
    return all_d, per_seed


def _hop_lines(label, G, files, args):
    d, per_seed = hop_profile(G, files, args.pairs_per_file,
                              args.files_sampled, random.Random(0))
    if not d:
        print(f"    {label}: no reachable pairs sampled")
        return
    p = np.percentile(d, [10, 50, 90])
    print(f"    {label}  {len(d):4d} pairs                     "
          f"{p[0]:5.0f} {p[1]:7.0f} {p[2]:5.0f} {max(d):5d}")
    picks = [per_seed[0], per_seed[len(per_seed) // 2], per_seed[-1]]
    print("      shortest / longest by file: " + " | ".join(
        f"Seed{s}: {lo}-{hi}" for s, lo, hi in picks))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--train-dir", required=True, help="the folder training reads")
    ap.add_argument("--test-dir", required=True, help="the folder the benchmark ran on")
    ap.add_argument("--maps", default="", help="comma-separated; blank = every map in either")
    ap.add_argument("--maps-dir", default="all_maps",
                    help="map files for the journey-length check; skipped if absent")
    ap.add_argument("--pairs-per-file", type=int, default=10)
    ap.add_argument("--files-sampled", type=int, default=30)
    args = ap.parse_args()

    if args.maps:
        names = [m.strip() for m in args.maps.split(",") if m.strip()]
    else:
        names = sorted({d for root in (args.train_dir, args.test_dir)
                        if os.path.isdir(root)
                        for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))})

    leaks = 0
    for name in names:
        tr, te = scenario_files(args.train_dir, name), scenario_files(args.test_dir, name)
        print(f"\n=== {name} ===")
        print(f"  train {args.train_dir}: {len(tr)} files ({seed_span(list(tr))})"
              f"  |  test {args.test_dir}: {len(te)} files ({seed_span(list(te))})")
        if not tr and not te:
            print("  (map in neither folder)")
            continue
        if not tr or not te:
            # Nothing to compare, but the folder that HAS the map still shows its
            # journey lengths -- how to match its settings when generating the
            # other folder's files.
            print("  (map in one folder only -- no overlap to check)")
            G = load_graph(args.maps_dir, name)
            if G is not None:
                print("  journey length in hops (sampled)      p10  median   p90   max")
                for label, files in (("train", tr), ("test ", te)):
                    if files:
                        _hop_lines(label, G, files, args)
            continue
        shared_seeds = sorted(set(tr) & set(te))
        print(f"  seed numbers in both folders: {len(shared_seeds)}"
              + (f" ({seed_span(shared_seeds)})" if shared_seeds else ""))

        tr_sets = {s: frozenset(read_pairs(p)) for s, p in tr.items()}
        te_sets = {s: frozenset(read_pairs(p)) for s, p in te.items()}
        te_index = {v: s for s, v in te_sets.items()}
        same = [(s, te_index[v]) for s, v in tr_sets.items() if v in te_index]
        if same:
            leaks += len(same)
            shown = ", ".join(f"train Seed{a} = test Seed{b}" for a, b in same[:8])
            print(f"  IDENTICAL INSTANCES: {len(same)} of {len(tr)} training files -- "
                  f"{shown}{' ...' if len(same) > 8 else ''}")
        else:
            print(f"  identical instances: none")

        best, where = 0.0, None
        for a, va in tr_sets.items():
            for b, vb in te_sets.items():
                f = len(va & vb) / max(1, min(len(va), len(vb)))
                if f > best:
                    best, where = f, (a, b)
        if where is not None:
            print(f"  largest shared fraction: {best * 100:.1f}% "
                  f"(train Seed{where[0]} vs test Seed{where[1]})"
                  + ("  <- a copy" if best >= 0.999 else
                     "  <- near-copy, look closer" if best >= 0.5 else "  (chance level)"))

        G = load_graph(args.maps_dir, name)
        if G is None:
            print(f"  journey lengths: skipped (no {name} map files in {args.maps_dir})")
            continue
        print("  journey length in hops (sampled)      p10  median   p90   max")
        for label, files in (("train", tr), ("test ", te)):
            _hop_lines(label, G, files, args)

    print("\n" + ("HELD OUT: no training file reproduces a test file."
                  if leaks == 0 else
                  f"LEAK: {leaks} training file(s) reproduce test files -- "
                  f"regenerate those with --seed-offset."))


if __name__ == "__main__":
    main()