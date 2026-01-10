import json
import os

def save_history_for_blender(history, filename="flowrra_sim.json"):
    print(f"Exporting {len(history)} frames to {filename}...")
    with open(filename, 'w') as f:
        json.dump(history, f)
    print("Export Complete. Import this JSON in Blender.")