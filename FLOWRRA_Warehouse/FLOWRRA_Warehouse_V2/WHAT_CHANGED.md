# What changed (2026-09-29) -- and how to install it

## Install
1. Unzip `flowrra_updated.zip` INTO your FLOWRRA folder and let it replace
   files. It holds every `.py` and `.md` of the project in its final state --
   the unchanged ones too, so nothing can be half-updated.
2. Your maps, scenarios, checkpoints and run folders are not in the zip and are
   not touched.
3. Run the tests (next section). Every one should end with `ALL PASS`.

## Tests
    python test_measurement_fixes.py     # new
    python test_recovery_charge.py       # new
    python test_conflict_rules.py        # new
    python test_corridor_rules.py        # new
    python test_smoke_integration.py     # and the rest of your existing tests

All 20 test files pass here.

## With every switch off
Behaviour is your uploaded code's, with ONE deliberate difference: the
recovery cost is now charged once (it was charged twice -- a duplicated block).
That changes only the recovery head's reward. Everything else extra is
measurement: new columns and counters, nothing that feeds a decision.

## The switches (config_warehouse.py, section "conflict", all False by default)
| switch | what it does |
|---|---|
| `path_warnings` | 1. warn by where fleets are going (head-on, blocked, contested), not by distance |
| `directional_braking` | 2. a steady convoy no longer brakes |
| `corridor_entry` | 3. don't enter a corridor when a fleet inside is coming at you |
| `priority` | 4. nearest goal first, then seniority in the corridor; the loser backs out and pulls over |
| `node_aligned_moves` | stop and turn exactly on nodes (required by 3 and 4) |
| `yield_to_stopped` | never drive into a fleet that is standing still: queue behind it |

For the next training run: set all six to `True`.
Perception radius stays 5 (see CONFLICT_DESIGN.md, component 5).

## Reading the policy apart from the rules
The episode line now ends its recovery part with
`wait policy X% / done Y%`: among fleets at risk, how often the NETWORK
proposed waiting, against how often a wait was actually executed. The gap is
the rules and recovery holds. Judge the policy on the first number; the CSV
has the full split (`choice_policy_*`, `choice_exec_*`, `choice_over_*`).

## Files
NEW
- `conflict_warehouse.py` -- component 1, the route-based pair classifier
- `corridor_warehouse.py` -- components 3 and 4, and yield-to-stopped
- `drive_shortest_path.py` -- the test driver and A/B tool (judges against any arm)
- `check_identity.py` -- proves a change leaves behaviour identical with its switch off
- `radius_study.py` -- which perception radius sees conflicts coming
- `test_measurement_fixes.py`, `test_recovery_charge.py`,
  `test_conflict_rules.py`, `test_corridor_rules.py`
- `WHAT_CHANGED.md` -- this file

CHANGED
- `core_warehouse.py` -- measurement fixes; recovery charged once; the hooks
  for every switch above; proposed-vs-executed accounting
- `loop_warehouse.py` -- warning set can come from the classifier
- `node_warehouse.py` -- node-aligned moves
- `recovery_warehouse.py` -- counts repeat offences
- `config_warehouse.py` -- the "conflict" section; recovery_policy.prevention_window
- `main_runner_warehouse.py` -- efficiency, prevention and the proposed/done wait
  share in the episode line; new CSV columns (including `corridor_*`); the
  conflict switches logged at startup
- `CONFLICT_DESIGN.md` -- the Status section: everything built, measured and decided
- `Calibration.md` -- the -8.5 explained (the duplicated charge)

UNCHANGED: every other file in the zip.

## Where the numbers are
CONFLICT_DESIGN.md, section "Status" -- every A/B table, the map analysis,
where cold_run24 collapsed, and what is left.
