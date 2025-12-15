# FLOWRRA: Federated Holonic Swarm Architecture
### **Scalable, Constraint-Aware, Recursive Multi-Agent System**

**Version:** 0.2 (Federated R-GNN)
**Architecture Type:** Hierarchical Holonic Multi-Agent System (HMAS)

---

## 1. High-Level System Overview

The FLOWRRA architecture moves beyond a single monolithic swarm into a **Federated Hierarchy**. The system is partitioned spatially into **Holons** (autonomous sub-swarms) managed by a lightweight **Federation Layer**.



### **The Two Layers of Intelligence**

| Layer | Component | Responsibility | Intelligence Type |
| :--- | :--- | :--- | :--- |
| **Layer 1** | **The Federation ("Up Above")** | Global coherence, spatial partitioning (Quadtree), and boundary constraint validation. | **Constraint-Based / Orchestrator** |
| **Layer 2** | **The Holon (Local Cluster)** | Local physics, intra-cluster routing, and predictive pathfinding. | **Recurrent GNN (R-GNN) / Reactive** |

---

## 2. Layer 1: The Federation Manager (`federation.py`)

The Federation Manager does not control individual nodes. Instead, it manages the **Space** and the **Boundaries**.

### **Core Components**
1.  **Dynamic Quadtree Manager:**
    * Continuously partitions the global simulation space into $M$ cells based on node density.
    * **Action:** Assigns specific spatial `bounds` to each Holon instance.
2.  **Boundary Link Manager:**
    * Identifies the "Membrane" between Holons.
    * **Search Algorithm:** For every adjacent pair of Holons $(H_A, H_B)$, it finds the closest pair of nodes $(\mathbf{x}_A, \mathbf{x}_B)$ to serve as the active **Boundary Link**.
3.  **Global Constraint Validator:**
    * **Monitor:** Checks the distance $D(\mathbf{x}_A, \mathbf{x}_B)$.
    * **Constraint:** If $D > D_{max}$, the link is flagged as **Broken**.
    * **Response:** Triggers a **Global Repulsion Event** ($\rho_{global}$) at that boundary, forcing Holons to retreat or re-route.

### **Data Input/Output**
* **Input:** Aggregated state summaries from Holons (Centroids, Bounding Boxes).
* **Output:** `Boundary_Map` (List of active boundary nodes and their neighbors) + `Global_Repulsion_Flags`.

---

## 3. Layer 2: The Holon Controller (`holon_core.py`)

Each Holon is an independent instance of the original `core.py`, but modified to be "boundary-aware."

### **Core Components**
1.  **Local Loop Physics (`loop.py`):**
    * Manages spring forces *only* for nodes within its assigned Quadtree cell.
    * Nodes moving out of bounds are "handed off" to the neighbor Holon (ownership transfer).
2.  **Density Function Estimator (`density.py`):**
    * **Intra-Holon ($\rho_{intra}$):** Standard repulsive field from local neighbors.
    * **Inter-Holon ($\rho_{inter}$):** A **Dampened Density Field** generated *only* at the boundary vectors provided by the Federation Layer.
3.  **WFC Recovery (`recovery.py`):**
    * Local collapse: If a Holon becomes unstable, it collapses *independently* of the Federation, ensuring failure isolation.

---

## 4. The Brain: Recurrent Graph Neural Network (R-GNN)

The intelligent agent (`r_gnn_agent.py`) is upgraded to handle the "Inertia" of the routing decisions using Recurrent Neural Networks.



### **Architecture: `GAT-LSTM`**

The model processes data in three stages:

1.  **Spatial Aggregation (GAT Layers):**
    * **Input:** Node Features (Pos, Vel, Sensor Data).
    * **Mechanism:** Graph Attention Network aggregates information from local neighbors ($k$-hop).
    * **Output:** Spatial Embedding Vector $E_t$.

2.  **Temporal Integration (LSTM/GRU Cell):**
    * **Input:** Spatial Embedding $E_t$ + **Previous Hidden State $H_{t-1}$**.
    * **Mechanism:** The LSTM cell updates its memory to retain the history of path stability.
    * **Role:** This enables **Temporal Damping**. The agent "remembers" that a boundary link was stable 5 frames ago and resists jittery changes.
    * **Output:** Context Vector $C_t$ + New Hidden State $H_t$.

3.  **Action Decoder (MLP):**
    * **Input:** Context Vector $C_t$.
    * **Output:** Action Probabilities (Movement Direction).

---

## 5. The Data Movement Pipeline

How information flows through the system in a single simulation step ($t$):

### **Phase A: The Upward Sync (Bottom-Up)**
1.  **Holons** calculate their internal state (Centroid, Active Node Count).
2.  **Holons** identify their own candidate "Edge Nodes" (nodes near the partition boundary).
3.  **Data Push:** Holons send `[Edge_Node_IDs, Positions]` to the **Federation Manager**.

### **Phase B: The Federation Cycle (Central Processing)**
1.  **Quadtree Update:** Federation checks if any Holon is overloaded; splits or merges cells if necessary.
2.  **Link Discovery:** Federation matches Edge Nodes from $H_A$ with $H_B$ to find the optimal **Boundary Link**.
3.  **Constraint Check:**
    * If `Link_Distance < D_max`: Link is **Active**.
    * If `Link_Distance > D_max`: Link is **Severed**.
4.  **Data Push:** Federation sends `[Neighbor_Boundary_Pos, Link_Status]` back to the relevant Holons.

### **Phase C: The Downward Execution (Top-Down)**
1.  **Holon Perception:**
    * Nodes sense local neighbors ($\rho_{intra}$).
    * Nodes receive the Federation's boundary data ($\rho_{inter}$).
2.  **R-GNN Inference:**
    * Input: `Local_State` + `Boundary_Data` + `Memory_H_{t-1}`.
    * Process: R-GNN fuses the immediate congestion data with the historical path memory.
    * Output: `Action`.
3.  **Physical Step:**
    * Nodes move.
    * `loop.py` validates integrity.
    * `recovery.py` checks for collapse.

---

## 6. Implementation File Structure

```text
/FLOWRRA_Federated
├── /federation
│   ├── manager.py          # The "Up Above" orchestrator
│   ├── quadtree.py         # Dynamic spatial partitioning
│   └── constraints.py      # Global D_max and loop integrity checks
├── /holon
│   ├── holon_core.py       # The Worker (modified core.py)
│   ├── density.py          # Dampened repulsive field logic
│   ├── recovery.py         # WFC (unchanged)
│   └── loop.py             # Local physics (unchanged)
├── /brain
│   ├── r_gnn_agent.py      # NEW: GAT + LSTM architecture
│   └── buffer.py           # Sequence Replay Buffer (for temporal training)
└── main.py                 # Entry point
