"""
final_jobs.py -- the final deploy_eval as a queue of small jobs (build log entry 44).

WHY JOBS. Days are keyed (failures, waves and people per instance, step and
robot) and every arm is re-seeded per day, so an arm meets the same days in any
process (the two-process check came back IDENTICAL). So the run splits freely:
one job = one arm x one condition x one seed x one map group, written to its own
CSV and log, run N at a time, heaviest first. Pairing happens after, on
(map, k, seed), when the CSVs are merged.

THE RUN (decided with Rohit, 19 October)
  arms        ladder4 (route on), ladder6 and ladder7 (route off, floor 2.0 from
              the config), RULES, PIBT, RHCR-PP, RHCR-PIBT, RHCR-PIBT+naive
              (+naive only in full: with no failures it is RHCR-PIBT exactly)
  maps        ours: 25_5_2_5_2_1, 25_10_5_10_5_2, 50_20_5_10_5_2 -- book 100
              released over 500 steps, 1,000 steps a day
              kiva: Kiva_2D_46_33_1 -- book 200 over 500 steps, 1,200 steps
              clean, 1,500 full
  fleets      20, 40, 60, 80;  seeds 0-7;  scenarios all_scens_final
  conditions  clean (no failures, no people) and full (both on)
  world       --world-fixes on --orphans pickup --assignment hungarian, keyed draws on

USE
  python final_jobs.py --list                     # the jobs, and how many are done
  nohup python final_jobs.py --run --parallel 14 > final_queue.log 2>&1 &
  python final_jobs.py --status                   # done / running / left
  python final_jobs.py --merge                    # final_clean.csv, final_full.csv

  A job is done when its CSV has every row (12 for ours, 4 for kiva). --run skips
  done jobs, so after a stop or a reboot the same command resumes. Each job runs
  with OMP_NUM_THREADS=1 (and MKL / OpenBLAS), so 14 jobs do not fight over the
  16 cores. Paths and checkpoint names are set below; edit them if yours differ.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = "final_runs"
SCENS = "all_scens_final"
FLEETS = "20,40,60,80"
SEEDS = list(range(8))

ARMS = {                      # label -> deploy_eval --arms spec
    "ladder4": "ladder4=ladder_run4/flowrra_curriculum.pth",
    "ladder6": "ladder6=ladder_run6/flowrra_curriculum.pth:noroute",
    "ladder7": "ladder7=ladder_run7/flowrra_curriculum.pth:noroute",
    "RULES": "RULES",
    "PIBT": "PIBT",
    "RHCR-PP": "RHCR-PP",
    "RHCR-PIBT": "RHCR-PIBT",
    "RHCR-PIBT+naive": "RHCR-PIBT+naive",
}
NETWORK = {"ladder4", "ladder6", "ladder7"}

GROUPS = {
    "ours": {"maps": "25_5_2_5_2_1,25_10_5_10_5_2,50_20_5_10_5_2", "book": 100,
             "release": 500, "steps": {"clean": 1000, "full": 1000}, "rows": 12},
    "kiva": {"maps": "Kiva_2D_46_33_1", "book": 200,
             "release": 500, "steps": {"clean": 1200, "full": 1500}, "rows": 4},
}
CONDS = {"clean": ("off", "off"), "full": ("on", "on")}    # failures, people


def jobs(scens: str = SCENS, fleets: str = FLEETS, seeds=None, extra: str = ""):
    out = []
    for cond, (fail, people) in CONDS.items():
        for group, g in GROUPS.items():
            for arm, spec in ARMS.items():
                if cond == "clean" and arm.endswith("+naive"):
                    continue
                for s in (seeds if seeds is not None else SEEDS):
                    name = f"{cond}_{group}_{arm}_s{s}".replace("+", "_")
                    # FINAL_WRAPPER: testing only (the sandbox's stub runner, no torch).
                    pre = [sys.executable, "-u"] + ([os.environ["FINAL_WRAPPER"]]
                                                    if os.environ.get("FINAL_WRAPPER") else [])
                    cmd = pre + ["deploy_eval.py", "--arms", spec,
                           "--maps", g["maps"], "--scens-dir", scens,
                           "--agents", fleets, "--seeds", str(s),
                           "--failures", fail, "--people", people,
                           "--book", str(g["book"]), "--release", str(g["release"]),
                           "--max-steps", str(g["steps"][cond]),
                           "--assignment", "hungarian", "--world-fixes", "on",
                           "--orphans", "pickup", "--keyed-draws", "on",
                           "--out", os.path.join(OUT, name + ".csv")] + extra.split()
                    # Heaviest first: networks, full, our maps (more of them).
                    w = ((3 if arm in NETWORK else 1.5) * (1.2 if cond == "full" else 1.0)
                         * (3.0 if group == "ours" else g["steps"][cond] / 1000.0))
                    out.append({"name": name, "cmd": cmd, "rows": g["rows"], "weight": w,
                                "cond": cond, "group": group, "arm": arm, "seed": s})
    return sorted(out, key=lambda j: -j["weight"])


def done(job) -> bool:
    p = os.path.join(OUT, job["name"] + ".csv")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as fh:
            return sum(1 for _ in fh) - 1 >= job["rows"]
    except OSError:
        return False


def run_one(job) -> tuple:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    log = os.path.join(OUT, job["name"] + ".log")
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(job["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env)
    return job["name"], rc, time.time() - t0, done(job)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--parallel", type=int, default=14)
    ap.add_argument("--only", default="", help="comma-separated substrings; keep jobs whose name has one")
    ap.add_argument("--scens-dir", default=SCENS)
    ap.add_argument("--agents", default=FLEETS)
    ap.add_argument("--seeds", default="", help="e.g. 0,1 (default 0-7)")
    ap.add_argument("--extra", default="", help="extra deploy_eval flags (testing)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()] or None
    js = jobs(args.scens_dir, args.agents, seeds, args.extra)
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        js = [j for j in js if any(k in j["name"] for k in keys)]

    if args.list or args.status:
        n_done = sum(done(j) for j in js)
        if args.list:
            for j in js:
                print(f"{'DONE ' if done(j) else '     '}{j['name']}")
        running = [j["name"] for j in js if not done(j)
                   and os.path.exists(os.path.join(OUT, j["name"] + ".log"))]
        print(f"\n{len(js)} jobs: {n_done} done, {len(js) - n_done} left "
              f"({len(running)} started and not finished)")
        return

    if args.run:
        todo = [j for j in js if not done(j)]
        print(f"[final] {len(js)} jobs, {len(js) - len(todo)} already done, running "
              f"{len(todo)}, {args.parallel} at a time", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(run_one, j) for j in todo]
            for k, f in enumerate(as_completed(futs), 1):
                name, rc, dt, ok = f.result()
                el = (time.time() - t0) / 3600
                # Judged by the CSV (every row there), not the exit code: a crash
                # in deploy_eval's closing summary must not mark a full job failed.
                print(f"[final] {k}/{len(todo)} {name}: "
                      f"{'ok' if ok else 'INCOMPLETE (see its .log), rc=' + str(rc)}"
                      f" in {dt / 60:.0f} min | {el:.1f} h elapsed", flush=True)
        return

    if args.merge:
        import pandas as pd
        for cond in CONDS:
            parts = [pd.read_csv(os.path.join(OUT, j["name"] + ".csv")) for j in js
                     if j["cond"] == cond and done(j)]
            missing = [j["name"] for j in js if j["cond"] == cond and not done(j)]
            if parts:
                df = pd.concat(parts, ignore_index=True)
                df.to_csv(f"final_{cond}.csv", index=False)
                print(f"final_{cond}.csv: {len(df)} rows from {len(parts)} jobs"
                      + (f"; NOT DONE: {len(missing)} ({missing[:3]} ...)" if missing else ""))
        return
    ap.print_help()


if __name__ == "__main__":
    main()