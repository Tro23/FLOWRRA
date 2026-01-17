# FLOWRRA-APP: Federated Holonic Middleware

**FLOWRRA-APP** is a domain-agnostic middleware layer designed to bridge the gap between high-level autonomous decision-making and various physical or simulated environments. 

By standardizing state representations and action spaces, FLOWRRA_APP enables complex federated holonic logic to be applied across diverse industries: from **Warehouse Robotics** and **Smart City Traffic** to **Satellite Constellations**.

---

## 🏗️ The Middleware Product Stack

FLOWRRA acts as the "Consciousness Layer" between a customer's existing framework and their control systems. It consumes raw state data, processes it through a multi-agent federated hierarchy, and returns actionable commands.

```text
┌─────────────────────────────────────┐
│    Customer's Existing System       │
│    (ROS, Unity, Custom Framework)   │
└─────────────┬───────────────────────┘
              │ State Data
              ↓
┌─────────────────────────────────────┐
│         FLOWRRA Middleware          │
│  ┌─────────────────────────────┐    │
│  │ Domain Detection/Selection  │    │
│  └──────────┬──────────────────┘    │
│             ↓                       │
│  ┌─────────────────────────────┐    │
│  │ Density Function Library    │    │
│  │ (Gaussian, Von Mises, Beta) │    │
│  └──────────┬──────────────────┘    │
│             ↓                       │
│  ┌─────────────────────────────┐    │
│  │  Wave Function Collapse     │    │
│  └──────────┬──────────────────┘    │
│             ↓                       │
│  ┌─────────────────────────────┐    │
│  │   Action Distribution       │    │
│  └─────────────────────────────┘    │
└─────────────┬───────────────────────┘
              │ Action Commands
              ↓
┌─────────────────────────────────────┐
│    Customer's Control System        │
└─────────────────────────────────────┘
```

## 📐 Internal Architecture

The app uses a Federated Manager to partition the world spatially via a Quadtree, ensuring that localized clusters (Holons) handle their own regional logic and agents independently.

```text
┌─────────────────────────────────────────────────────────┐
│               FLOWRRA Backend Core                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────┐      ┌──────────────┐                   │
│  │  Dataset   │─────▶│    Adapter   │                   │
│  │  Loader    │      │    Layer     │                   │
│  └────────────┘      └──────┬───────┘                   │
│                             │                           │
│                             ▼                           │
│        ┌──────────────────────────────────┐             │
│        │   Federated Manager              │             │
│        │   (Spatial Partition)            │             │
│        └────────────┬─────────────────────┘             │
│                     │                                   │
│          ┌──────────┴──────────┐                        │
│          ▼                     ▼                        │
│     ┌─────────┐           ┌─────────┐                   │
│     │ Holon 0 │   ...     │ Holon N │                   │
│     └────┬────┘           └────┬────┘                   │
│          │                     │                        │
│          └────────┬────────────┘                        │
│                   ▼                                     │
│          ┌────────────────┐                             │
│          │  Orchestrator  │                             │
│          │  (Core Logic)  │                             │
│          └────┬───────────┘                             │
│               │                                         │
│     ┌─────────┼─────────┐                               │
│     ▼         ▼         ▼                               │
│  Agent     Density     WFC                              │
│  (GNN)     (Field)   (Recovery)                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 📂 Folder & Module Breakdown

### 🔌 Root: Connectivity & Adaptation
flowrra_adapter.py: The "Translation Layer." Converts domain-specific data into a standardized FlowrraState.

flowrra_domain_extractors.py: Contains specialized logic for extracting features unique to Robotics, Traffic, or Satellites.

flowrra_backend.py: The main entry point that initializes the federation and manages the training/inference loop.

### 🌐 /federated: Spatial Management
manager.py: The "Consciousness Above." Handles high-level spatial partitioning, detects boundary breaches, and aggregates metrics.

quadtree.py: A dynamic spatial partitioner that divides the environment into manageable cells for each Holon.

### 🧠 /holon: Local Intelligence
flowrra_clean_core.py: The local brain of each partition. Orchestrates agents, density fields, and recovery logic.

flowrra_clean_agent.py: A domain-agnostic GNN (Graph Neural Network) agent that makes localized decisions.

flowrra_clean_density.py: Estimates repulsion/affordance fields (using Gaussian, Von Mises, etc.) to help agents navigate obstacles or achieve goals.


## 🚀 Integration Workflow

The integration is a simple request-response pattern:

**Their System → Flowrra API**: Sends raw state data.
**Flowrra Internal**: Adapts data $\rightarrow$ Partitions via Quadtree $\rightarrow$ Decides via Holons.
**Flowrra → Their System**: Returns optimized Action Commands.

## Usage Example

To run the backend for a specific domain:

```
python flowrra_backend.py --domain warehouse --dataset ./data/warehouse.json
```

    - Note: This app currently in active development focusing on flexible density functions and input adaptability. -
