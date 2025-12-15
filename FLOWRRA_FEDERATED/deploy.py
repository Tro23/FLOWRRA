"""
deploy.py

Runs the Federated System in inference mode using trained checkpoints
and a saved initial state from training.
Generates a JSON file representing the world state for Blender visualization
(optional, can be removed if not needed).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Assuming you have a standard config.py
from config import CONFIG
from federation.manager import FederationManager
from holon.holon_core import Holon
from holon.node import NodePositionND


def convert_to_serializable(obj):
    """Helper to serialize numpy objects for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Federated FLOWRRA Deployment")
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="The training episode number to load (e.g., 2000)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps to run",
    )
    # New argument to specify the initial state file
    parser.add_argument(
        "--state_file",
        type=str,
        # Default path based on the main.py saving convention
        default=None,
        help="Path to the JSON file containing the initial node state (e.g., deployment_states/initial_state_ep2000.json)",
    )
    args = parser.parse_args()

    # Determine state file path
    state_file_path = (
        Path(args.state_file)
        if args.state_file
        else Path("deployment_states") / f"initial_state_ep{args.episode}.json"
    )

    print("=" * 60)
    print(f"=== DEPLOYMENT MODE: EPISODE {args.episode} ===")
    print("=" * 60)

    # 1. Load Initial Node State (New Logic)
    if not state_file_path.exists():
        print(f"[Fatal] Initial state file not found at: {state_file_path}")
        print(
            "Please ensure main.py was run successfully and the state file was created."
        )
        sys.exit(1)

    with open(state_file_path, "r") as f:
        state_data = json.load(f)
        all_node_states = state_data["nodes"]
        # Optionally, check or load config from the state file
        # deployed_config = state_data["config"]

    print(
        f"[Deploy] Loaded initial state for {len(all_node_states)} nodes from {state_file_path.name}"
    )

    # Group nodes by holon_id
    holon_node_map = {}
    for state in all_node_states:
        hid = state["holon_id"]
        if hid not in holon_node_map:
            holon_node_map[hid] = []
        holon_node_map[hid].append(state)

    # 2. Setup Federation Manager (matches main.py)
    fed_manager = FederationManager(
        num_holons=CONFIG["federation"]["num_holons"],
        world_bounds=CONFIG["federation"]["world_bounds"],
        breach_threshold=CONFIG["federation"]["breach_threshold"],
        coordination_mode="positional",
    )

    holons = {}
    partitions = fed_manager.get_partition_assignments()

    # 3. Init Holons, Load Models, and Initialize Nodes from State File
    for pid, part in partitions.items():
        # Initialize Holon Wrapper
        holon = Holon(
            holon_id=pid,
            partition_id=pid,
            spatial_bounds={"x": part.bounds_x, "y": part.bounds_y},
            config=CONFIG,
            mode="deployment",
        )

        try:
            # Reconstruct REAL NodePositionND nodes from saved state
            holon_nodes = []

            for node_state in holon_node_map.get(pid, []):
                node = NodePositionND(
                    id=node_state["id"],
                    pos=np.array(node_state["pos"]),
                    dimensions=node_state["dimensions"],
                )

                # Crucially, set the last_pos to enable velocity calculation
                node.last_pos = np.array(node_state["last_pos"])

                # Set config parameters (assuming they are in CONFIG)
                node.sensor_range = CONFIG["node"]["sensor_range"]
                node.move_speed = CONFIG["node"]["move_speed"]

                holon_nodes.append(node)

            # Initialize Orchestrator with the saved nodes
            holon.initialize_orchestrator_with_nodes(holon_nodes)

            # Load Checkpoint (must match main.py save path)
            checkpoint_path = Path("checkpoints") / f"holon_{pid}_ep{args.episode}.pt"

            if not checkpoint_path.exists():
                print(f"[Error] Checkpoint not found: {checkpoint_path}")
                print("Make sure you ran main.py --mode training first.")
                sys.exit(1)

            holon.load(str(checkpoint_path))
            holons[pid] = holon
            print(
                f"[Success] Loaded Holon {pid} from {checkpoint_path.name} with {len(holon_nodes)} nodes."
            )

        except Exception as e:
            print(f"[Fatal] Failed to initialize Holon {pid}: {e}")
            import traceback

            traceback.print_exc()
            return

    # 4. Run Simulation (Inference)
    # ... (Rest of the simulation logic remains the same as before) ...

    history_frames = []
    steps = args.steps
    print(f"\n[Deploy] Simulating {steps} steps...")

    for t in range(steps):
        # Federation Step Data
        holon_states = {}
        frame_nodes = []

        # Step all holons
        for hid, holon in holons.items():
            # Run a step in deployment mode (no training update)
            holon.step(t, total_episodes=1)

            # Aggregate data for visualization
            for n in holon.nodes:
                frame_nodes.append(
                    {
                        "id": n.id,
                        "pos": n.pos.tolist(),
                        "holon_id": hid,
                    }
                )

            holon_states[hid] = holon.get_state_summary()

        # Federation Checks
        alerts = fed_manager.step(holon_states)
        for hid, alert_list in alerts.items():
            if alert_list:
                holons[hid].receive_breach_alerts(alert_list)

        # Build Frame
        frame = {
            "step": t,
            "nodes": frame_nodes,
            "stats": {
                "breaches": fed_manager.total_breaches,
            },
        }
        history_frames.append(frame)

        if t % 50 == 0 and t > 0:
            print(f"  Step {t}/{steps} complete...")

    # 5. Export
    output_filename = f"deployment_viz_ep{args.episode}_from_trained_state.json"
    output = {
        "config": {
            "world_bounds": CONFIG["federation"]["world_bounds"],
            "num_nodes": CONFIG["node"]["total_nodes"],
            "episode": args.episode,
        },
        "frames": history_frames,
    }

    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2, default=convert_to_serializable)

    print(f"\n[Done] Saved visualization data to: {output_filename}")


if __name__ == "__main__":
    main()
