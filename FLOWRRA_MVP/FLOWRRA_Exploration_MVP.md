# FLOWRRA: Exploration Swarm MVP
**Flow Recognition Reconfiguration Agent - Commercial Prototype v1.0**

## 1. Executive Summary
This document outlines the Minimum Viable Product (MVP) architecture for FLOWRRA tailored for **Autonomous Exploration and Mapping**. 

Unlike standard swarm systems that rely on rigid formations, FLOWRRA utilizes a **Hybrid Control Loop** combining decentralized Graph Neural Networks (GNN) for fluid movement with a centralized "Frontier Logic" for strategic exploration. This ensures the swarm behaves like a fluid—pouring into unexplored areas while maintaining internal coherence to prevent collisions.

## 2. System Architecture
The system operates on two distinct time-scales to maximize both reactivity and strategy.

### A. The Hybrid Loop
| Layer | Frequency | Component | Function |
| :--- | :--- | :--- | :--- |
| **Fast Loop** | Every Step | **GNN Agent** | Collision avoidance, formation flying, immediate movement. |
| **Slow Loop** | Every $N$ Steps | **Exploration Map** | Calculates "Frontier Vectors" (Goal direction), updates global coverage. |
| **Safety Loop** | Asynchronous | **WFC Recovery** | Monitors "Flow State". If swarm jams/stalls, it teleports nodes to open frontiers. |

### B. Core Components

#### 1. The Exploration Map (`ExplorationMap.py`)
A centralized voxel grid that acts as the "Shared Memory" of the swarm.
* **Resolution:** 0.02 units (configurable).
* **Logic:** As nodes move, they "stamp" their sensor radius onto the grid, flipping voxels from `0` (Unknown) to `1` (Explored).
* **Frontier Calculation:** Calculates a vector pointing from a node to the nearest cluster of `0`s. This is fed into the GNN as the "Intent" vector.

#### 2. The Fluid GNN Agent (`GNNAgent.py`)
A permutation-invariant Graph Neural Network that dictates movement.
* **Inputs (State Vector):**
    * **Self:** Velocity (2), Frontier Vector (2)
    * **Context:** Local Repulsion Grid (Flattened 5x5)
    * **Neighbors:** Relative positions/velocities of nearby peers (via Graph Attention).
* **Output:** $\Delta x, \Delta y$ (Velocity adjustment).

#### 3. Wave Function Collapse (WFC) Recovery
The "Safety Net" that makes this commercially viable.
* **Trigger:** If `Average Reward` < Threshold OR `Collision Count` > Limit for $T$ steps.
* **Action:** "Collapses" the swarm state by resetting low-coherence nodes to the edge of the explored map (the Frontier), effectively "respawning" them in useful positions.

## 3. Technical Specifications

### The Reward Function
The AI is trained on a composite reward signal designed to balance greed (exploration) with safety (flow).

$$R_{total} = R_{explore} + R_{flow} + R_{safety}$$

1.  **$R_{explore}$**: $+2.0 \times (\text{New Voxels Discovered})$
    * *Incentivizes spreading out and entering new rooms.*
2.  **$R_{flow}$**: $+0.5$ if moving in alignment with neighbors.
    * *Incentivizes smooth formation flying.*
3.  **$R_{safety}$**: $-5.0$ for collisions (Node-Node or Node-Wall).
    * *Critical penalty to ensure hardware safety.*

### Data Flow
1.  **Sensors:** Nodes detect walls and neighbors.
2.  **Density:** `DensityFunctionEstimator` computes local "pressure" (repulsion).
3.  **Intent:** `ExplorationMap` provides the "direction of interest".
4.  **Inference:** GNN aggregates (Sensors + Density + Intent) $\rightarrow$ Action.
5.  **Physics:** Action is applied; Physics engine resolves positions.
6.  **Map Update:** New positions update the global Coverage Map.

## 4. Simulation & Visualization

### Blender Integration
This MVP decouples simulation from rendering for maximum performance.
1.  **Simulation:** Python runs the math (GNN + Physics) and exports a lightweight JSON.
    * `{"time": 0, "nodes": [{"id": 0, "pos": [...]}, ...]}`
2.  **Rendering:** Blender imports the JSON to visualize:
    * **Nodes:** As high-fidelity drone models.
    * **Fog of War:** Can be visualized using volumetric shaders driven by the `ExplorationMap` data.

## 5. Roadmap to Product
To move from this MVP to a sellable hardware solution:

1.  **Phase 1 (Current):** Validated Logic in 2D Simulation.
2.  **Phase 2 (3D Volumetric):** Switch `config.py` to `dimensions: 3`. The code is already compatible. Train on "Tube" environments (tunnels/pipes).
3.  **Phase 3 (Hardware-in-the-Loop):** Replace `NodePositionND.py` with a wrapper for ROS2 (Robot Operating System) to send velocity commands to real Crazyflie/DJI drones.