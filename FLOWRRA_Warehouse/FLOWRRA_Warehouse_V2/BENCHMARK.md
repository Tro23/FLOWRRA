# Benchmarking FLOWRRA

Three questions, always in this order: **what kind of map is it**, **what could
N fleets deliver on it**, and **where does FLOWRRA sit between those limits**.

## The tools

| file | what it does |
|---|---|
| `benchmark_report.py` | the report: map anatomy, delivery ceilings, and every run against them. Markdown + summary CSV + two charts |
| `drive_shortest_path.py` | measures the CAPACITY: a shortest-path driver with the conflict rules, at any fleet count, with or without failures |
| `convert_grid_map.py` | published 2D grid maps (RHCR, MovingAI) into FLOWRRA's format -- exactly, or stacked into floors joined by lift shafts |
| `radius_study.py` | which perception radius sees conflicts coming |
| `benchmark_all.py`, `benchmark_flowrra.py`, `baselines_mapf.py` | your one-shot MAPF suite (PP, PIBT, RHCR-PP/PIBT, disturbance and failure waves) -- unchanged, a different question: one trip per fleet, not rolling orders |

## The two ceilings

- **The ideal**: fleets x steps x speed / conflict-free cycle -- what the fleets
  would deliver if none ever met another. The run CSV carries it per episode
  (`stream_ideal_deliveries`). Nothing reaches it in dense traffic.
- **The capacity**: what the shortest-path driver with the conflict rules
  actually delivers -- where the map saturates. A policy that routes around
  congestion can beat it; that is the claim worth making.

## Commands

    # 1. capacity at the fleet counts you care about, with and without failures
    R="conflict.path_warnings=true,conflict.directional_braking=true,conflict.corridor_entry=true,conflict.priority=true,conflict.node_aligned_moves=true,conflict.yield_to_stopped=true"
    for N in 30 45 60; do
      python drive_shortest_path.py --agents $N --steps 800 --seeds 0,1,2 --recovery never \
        --arm rules:$R --arm rules_failures:$R,errors.enabled=true --out capacity
    done

    # 2. the report
    python benchmark_report.py --map 50_5_5_10_5_2 \
        --run cold_run25=cold_run25/curriculum_metrics.csv \
        --run cold_run24=cold_run24/curriculum_metrics.csv \
        --capacity "capacity/driver_2*.csv" --out report_50

    # 3. anatomy of several maps side by side
    python benchmark_report.py --anatomy kiva,kiva_3f,kiva_5f,50_5_5_10_5_2 --out anatomy

    # kiva: download RHCR's published map, then convert (exact, and stacked)
    #   https://raw.githubusercontent.com/Jiaoyang-Li/RHCR/master/maps/kiva.map
    python convert_grid_map.py kiva.map --name kiva
    python convert_grid_map.py kiva.map --name kiva_3f --floors 3 --shafts 10
    python convert_grid_map.py kiva.map --name kiva_5f --floors 5 --shafts 10

The labels in `--run` are yours; the first run is the one every other run is
paired against, episode by episode.

## 50_: what 30, 45 and 60 fleets can deliver (2026-09-29)

Shortest-path driver, all six rules, 800 steps, order window +-3, 3 seeds
(sd about +-20 to +-35). Failures: 2-4 dead fleets per episode, as
`errors.enabled` injects them.

| fleets | deliveries | with failures | of the ideal | per fleet |
|---|---|---|---|---|
| 30 | 124 | 133 | 58% | 4.1 |
| 45 | 152 | 165 | 49% | 3.4 |
| 60 | 176 | 172 | 42% | 2.9 |

- **Failures cost nothing measurable at these sizes**: nearly every stranded
  order is rescued, and the differences are inside seed noise.
- **With failures the map saturates at 45**: each fleet from 30 to 45 adds 2.1
  deliveries, from 45 to 60 only 0.5. **cold_run26 (rescues on): 45 fleets.**
- For scale: cold_run25 (60 fleets, learned policy) ran at 41% of its ideal
  in episodes 11-15 with the order window at +-2 -- compare efficiency, not
  raw deliveries, until its episodes reach the +-3 window.

**Rescue-path fixes (2026-09-29).** Failures had never run in stream mode, and
the rescue path crashed there twice: a rescuer handed an inherited order that
had already left the pool (KeyError), and a rescuer left with nothing to carry
(IndexError; in the original code, a KeyError when its own goal was a dock).
Both fixed in `core_warehouse.py`; `test_stream_failures.py` covers them.
Stress: 40 runs, 320 failures injected, 146 orders rescued by handover, no
crashes.

## Kiva against 50_ (anatomy)

| | kiva | kiva, 3 floors | kiva, 5 floors | 50_ |
|---|---|---|---|---|
| cells | 1,278 | 3,934 | 6,590 | 6,300 |
| single-lane cells | 1.6% | 3.9% | 4.4% | 89.5% |
| longest single lane | 1 | 5 | 5 | 46 |
| cells in lanes of 5+ | 0% | 2.5% | 3.0% | 89.5% |
| mean hops to back out and yield | 0.0 | 0.1 | 0.1 | 5.3 |
| cells where two fleets can pass | 98.7% | 96.2% | 95.8% | 0% |
| grid-near cells far by path | 3.1% | 3.0% | 2.8% | 22.1% |

Kiva is an open grid: almost every conflict can be solved by stepping aside,
and the challenge is weaving many robots. 50_ is single-lane: conflicts are
head-on and are solved by waiting, backing out or routing around. FLOWRRA's
refinements were forced by the second; kiva measures the first. The exact
conversion is checked: 1,278 cells and 2,213 edges, identical to a count
straight from the text grid.

FLOWRRA runs on the converted kiva today (one floor detected, 12 docks, every
cell reaches a dock) -- in its own dock-based stream mode. The like-for-like
RHCR comparison needs RHCR's lifelong mode instead (below).

## Kiva: Follower's protocol (2026-09-29)

From Follower's own evaluation config (learn-to-follow,
`experiments/05-warehouse/05-warehouse.yaml`): map `wfi_warehouse` -- our kiva,
cell for cell -- lifelong (`on_target: restart`), 512 timesteps, `soft`
collisions, 32 / 64 / 96 / 128 / 160 / 192 agents, seeds 0-9. Agents start on
the 192 home cells; goals are drawn from the 480 endpoint cells. Throughput =
goals reached by all agents / 512.

    pip install pogema                 # a separate environment is safest (it pins numpy 1.26)
    python kiva_instances.py           # the 60 instances, extracted by POGEMA's own code
    python run_kiva.py --policy rules  # FLOWRRA's rule-based controller
    python run_kiva.py --policy checkpoint --checkpoint <path>   # a trained FLOWRRA network
    python run_kiva.py --policy rules --set warehouse.braking=false --label no_brake

- **Same tasks.** Each agent's goal sequence is fixed by the seed in advance;
  `kiva_instances.py --validate` runs POGEMA's A* agents for the full episode
  and checks every goal POGEMA assigns against the extracted sequence
  (4/4 instances identical). POGEMA reshuffles its start list IN PLACE on each
  reset, so the draws depend on the reset sequence; one reset is used here.
- **Same units.** FLOWRRA moves 0.5 cells per step, so 512 timesteps are 1,024
  steps and throughput is per timestep of a one-cell-per-step agent
  (`lifelong_throughput`).
- **Different physics -- state it with the numbers.** POGEMA simplifies
  execution: a move into a contested cell simply does not happen and the agent
  loses one timestep; there is no braking, no collision, no recovery. FLOWRRA
  models what a vehicle does: it slows near others (`warehouse.braking`), can
  collide, and recovers. On kiva this costs throughput (first ablation, 64
  agents, 256 timesteps: all rules 0.043, no rules and no braking 0.113; POGEMA's
  naive A* ~0.4). Throughput on their protocol is therefore a conservative
  number for FLOWRRA, and collisions are reported beside it.

## Next

1. **Lifelong baselines on the stream.** PIBT and RHCR as drivers inside the
   same simulator: same map, same order stream, same failures, same collision
   model, same CSV -- so the report places them beside FLOWRRA. Your
   `baselines_mapf.py` has the planners; they need the rolling-order loop.
2. **Kiva's lifelong mode.** "Next goal on arrival" with task generation
   matched to RHCR's code, and the speed question settled (FLOWRRA moves 0.5
   cells per step, RHCR one cell per timestep: compare per hop).
3. **The multi-level benchmark.** Kiva at 1, 3 and 5 floors with PIBT and
   FLOWRRA: how each method's efficiency falls as floors and shafts are added.
