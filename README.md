# FLOWRRA: Flow Recognition Reconfiguration Agent

**Federated Holonic Multi-Agent System for Dynamic Swarm Coordination**

FLOWRRA is a novel multi-agent reinforcement learning framework that enables swarm systems to dynamically recognize, adapt, and reconfigure in response to changing environmental conditions. Built on a federated holonic architecture, FLOWRRA allows agents to self-organize into hierarchical structures (holons) that can autonomously make decisions while maintaining global coordination.

    - For Conceptual Understanding and Evolution of FLOWRRA visit: https://open.substack.com/pub/rohittamidapati/p/flowrra-flow-recognition-reconfiguration?r=721j7i&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true -


## What is FLOWRRA?

FLOWRRA introduces a paradigm shift in swarm intelligence through a radical reorientation: **the entire swarm treats itself as its primary environment (Environment A)**, while everything external—obstacles, boundaries, external forces—becomes a secondary, interacting entity (Environment B). This Ouroboros principle enables the system to maintain structural and informational coherence through recursive self-awareness.

The framework combines:

- **Flow Recognition**: The swarm continuously monitors its own internal coherence and structural integrity, treating deviations as signals to adapt
- **Flow Reconfiguration**: Real-time self-reorganization through density field manipulation and wave function collapse, enabling the system to "fold time inward" and recover from disruptions
- **Retrocausal-Inspired Learning**: Uses observed disruptions to select optimal past configurations, effectively influencing operational history toward desired future states
- **Density Field Evolution**: A spatial memory system where collision events and failures are "splatted" into the environment as repulsion fields, teaching the swarm to avoid dangerous configurations
- **Wave Function Collapse (WFC)**: Dual-mode recovery mechanism that can:
  - **Spatial Collapse (Forward)**: Project into high-affordance regions using current density gradients
  - **Temporal Collapse (Backward)**: Return to historically coherent states through manifold smoothing
- **Federated Holonic Architecture**: Self-similar hierarchical structures from individual agents to swarm level, enabling scalability without sacrificing coherence

### Key Features

- 🔄 **Dynamic Holon Formation**: Agents autonomously cluster based on spatial density and task requirements
- 🧠 **GAT-based Graph Neural Networks**: Two agent architectures available:
  - **Simple GAT-GNN** (`agent.py`): Graph Attention Network with LSTM for temporal memory and multi-head attention
  - **R-GNN** (`r_gnn_agent.py`): Relational GNN for enhanced inter-agent communication
- 🗺️ **Quadtree-based Federation**: Efficient spatial partitioning for large-scale deployments
- 📊 **Real-time Metrics**: Comprehensive tracking of training, exploration, recovery, and frozen node behaviors
- 🎯 **Active/Frozen Node Management**: Adaptive computation allocation - nodes can "crystallize" as static landmarks once their zone is secured
- 🌊 **Density Field Memory**: Spatial repulsion fields that encode collective failure history, teaching avoidance through retrocausal splatting
- ⚡ **Wave Function Collapse**: Bidirectional recovery mechanism combining forward spatial projection and backward temporal restoration
- 🔁 **Retrocausal-Inspired Intelligence**: System doesn't just predict—it reorganizes recursively across time, using future observations to inform past configurations

## Architecture Overview

```
FLOWRRA_FEDERATED
├── holon/              # Core holon agent implementations
│   ├── core.py         # FLOWRRA orchestrator (Environment A self-awareness)
│   ├── agent.py        # GAT-GNN with multi-head attention
│   ├── r_gnn_agent.py  # Relational GNN alternative
│   ├── density.py      # Density field with retrocausal splatting
│   ├── recovery.py     # Wave Function Collapse (spatial + temporal)
│   ├── loop.py         # Spring topology with non-linear forces
│   └── obstacles.py    # Obstacle detection (Environment B)
├── federation/         # Federation manager and spatial partitioning
│   ├── manager.py      # Quadtree-based coordination
│   └── quadtree.py     # Spatial indexing
├── deployment/         # Saved deployment states and trajectories
├── results/            # Training metrics and visualizations
├── config.py           # Configuration parameters
├── main.py             # Training entry point (--parallel support)
├── deploy.py           # Deployment with frozen node tracking
└── visualization.html  # Web-based trajectory viewer
```

### The Ouroboros Architecture

FLOWRRA inverts the traditional agent-environment paradigm:

1. **Environment A (Primary)**: The swarm's own internal state—node positions, loop integrity, density fields, coherence metrics. The system continuously acts upon itself to preserve flow.

2. **Environment B (Secondary)**: External world—obstacles, boundaries, external forces. These are perturbations that the swarm must navigate while maintaining its internal coherence.

3. **The Critical Loop**:
   - **Sense**: Observe both environments (internal coherence + external obstacles)
   - **Evaluate**: Compute flow coherence Φ(S(t)) and loop integrity
   - **Decide**: If coherent, optimize via policy πθ; if disrupted, trigger WFC
   - **Act**: Apply actions to Environment A (self-reconfiguration)
   - **Learn**: Update policy and density field based on outcomes

This recursive self-awareness enables resilience not through external reactivity, but through internal coherence preservation.

## Quick Start

### Prerequisites

```bash
python >= 3.8
torch >= 1.13.0
numpy
matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd FLOWRRA_FEDERATED
```

2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib
```

### Running FLOWRRA

#### Basic Training

Run the main training loop with default configuration:

```bash
python main.py
```

This will:
- Initialize the federated holonic system
- Train agents across multiple episodes
- Save checkpoints and metrics
- Generate visualization maps in `federated_maps/`

#### Parallel Training (Recommended)

For faster training on multi-core systems using parallel holon execution:

```bash
python main.py --parallel
```

#### Custom Training Runs

Adjust episodes, nodes, holons, and enable parallel execution:

```bash
# Train for 100 episodes with 40 nodes across 4 holons in parallel
python main.py --episodes 100 --nodes 40 --holons 4 --parallel

# Train for 200 episodes with 32 nodes (8 per holon)
python main.py --episodes 200 --nodes 32

# Sequential training (single-threaded)
python main.py --episodes 50 --nodes 24
```

#### Custom Configuration

Modify `config.py` to adjust:
- Number of agents and holons
- Environment dimensions
- Training hyperparameters
- Sensor ranges and capabilities

Example:
```python
# config.py
NUM_AGENTS = 100
GRID_SIZE = 200
EPISODES = 1000
LEARNING_RATE = 0.001
```

### Deployment

Deploy trained models and visualize trajectories:

```bash
# Deploy with default settings (500 steps)
python deploy.py --deployment-file deployment/deployment_ep100.json

# Deploy for specific number of steps
python deploy.py --deployment-file deployment/deployment_ep100.json --steps 1000

# Deploy from later training checkpoint
python deploy.py --deployment-file deployment/deployment_ep2000.json --steps 2000
```

This loads saved checkpoints and generates:
- Deployment trajectories with frozen node tracking
- JSON files containing full system state over time
- Freeze/unfreeze event logs
- Collision and recovery metrics

**Visualization**: After deployment, open `visualization.html` in a web browser to see the animated swarm behavior, frozen nodes (gold stars), active nodes (blue circles), and holon boundaries.

### Visualization

After training, view your results:

1. Open `visualization.html` in a web browser
2. The visualization shows real-time agent movements, holon formations, and density maps
3. Metrics are displayed including coverage, exploration efficiency, and recovery time

Then open `visualization.html` in your browser to see:
- 🔵 Active nodes exploring and coordinating
- ⭐ Gold frozen nodes acting as landmarks
- 🔗 Spring connections maintaining loop integrity
- 📊 Statistics (active vs frozen, frame count, speed)

Tip to open `visualization.html` is to open your own local host:
```bash
python -m http.server 8000
```

Then open the Visualization file in the browser: `http://localhost:8000/visualization.html`

## Project Structure

- The main Folder of Implementable FLOWRRA: `FLOWRRA_FEDERATED`

### Core Modules

- **holon/core.py**: FLOWRRA orchestrator with loop structure, WFC triggers, and frozen node management
- **holon/agent.py**: GAT-based GNN agent with epsilon-greedy exploration and gradient masking for frozen nodes
- **holon/r_gnn_agent.py**: Alternative relational GNN implementation for enhanced communication
- **holon/density.py**: N-dimensional density field with collision splatting and retrocausal repulsion
- **holon/exploration.py**: Exploration strategies and frontier detection
- **holon/recovery.py**: Wave Function Collapse with dual-mode recovery (spatial forward + temporal backward)
- **holon/loop.py**: Spring topology management with non-linear warning forces
- **holon/obstacles.py**: Static and moving obstacle detection with line intersection checks
- **federation/manager.py**: Global coordination and holon management with Quadtree partitioning
- **federation/quadtree.py**: Spatial partitioning for scalable deployment

### Key Files

- `main.py`: Primary training script with episode loop and parallel execution support
- `config.py`: Central configuration for all parameters (nodes, holons, rewards, WFC thresholds)
- `deploy.py`: Deployment script with frozen node event tracking and trajectory generation
- `flowrra_model.py`: Neural network architectures (GAT layers, policy networks)
- `agent.py`: Graph Attention Network agent with multi-head attention and LSTM temporal memory
- `r_gnn_agent.py`: Alternative Relational GNN architecture
- `Noise_Cleanup.py`: Data preprocessing utilities
- `visualization.html`: Web-based trajectory playback with frozen node visualization

## Results & Metrics

Training produces several outputs:

- **federated_maps/**: Episode-by-episode visualization PNGs
- **metrics/**: JSON files with detailed per-holon statistics
- **results/**: Aggregated training curves and comparison plots
- **checkpoints/**: Saved model states for deployment
- **deployment/**: JSON trajectories for playback

### Key Metrics Tracked

- Coverage efficiency
- Inter-agent collision rate
- Holon formation/dissolution frequency
- Active vs frozen node ratios
- Exploration completeness
- Recovery time from failures

## BenchMARL Integration

FLOWRRA includes adapters for BenchMARL benchmarking:

```bash
python run_flowrra_benchmarl.py
```

This runs standardized multi-agent benchmarks and outputs comparative results to `benchmarl_results/`.

## The Core Innovation: Retrocausal-Inspired Intelligence

FLOWRRA's breakthrough lies in treating **disruption as information** that flows backward through time to inform system configuration. This isn't actual time travel—it's a directed inference mechanism inspired by quantum retrocausality.

### How It Works

1. **Disruption Detection**: System detects coherence drop (Φ(S(t)) < threshold)
2. **Wave Function Collapse**: 
   - **Spatial Mode (Forward)**: Sample affordance field to find high-coherence regions ahead
   - **Temporal Mode (Backward)**: Extract coherent tail from historical buffer
3. **Retrocausal Splatting**: Failed trajectory is "splatted" into density field as repulsion
4. **Memory Rewriting**: Bad states are replaced with recovered states in history buffer
5. **Learning Signal**: GNN receives differential rewards based on recovery mode
   - Spatial recovery (forward-looking): **High reward** (+15.0) - system anticipated!
   - Temporal recovery (backward-looking): **Medium reward** (+5.0) - system recovered
   - Random jitter (failure): **Heavy penalty** (-8.0) - system couldn't recover

### The Density Field as Spatial Memory

The density field ρ(x) inverts traditional repulsion:
- **High density (ρ → 1.0)**: Safe, navigable, previously successful paths
- **Low density (ρ → 0.0)**: Dangerous, obstacle-filled, failed trajectories

When events occur:
```python
# Collision/WFC trigger
density.splat_collision_event(
    position=failure_point,
    velocity=attempted_direction,  # Project forward (comet-tail)
    severity=0.5,                   # How bad the failure
    is_wfc_event=True               # WFC vs regular collision
)
```

This creates **spatial scars**—geometric memories of failure encoded in the topology itself. The swarm learns not "don't collide with obstacle at (x,y)" but "avoid this configuration space."

### Why This Matters

Traditional RL: State → Action → Reward → Update Policy  
FLOWRRA: State → Disruption → **Collapse** → **Rewrite History** → Update Density

The system doesn't just learn from failure—it reorganizes its operational history to make the failure "never happened" while encoding the lesson in the density field. This is **intelligence as sustained presence** rather than perfect prediction.



### Graph Neural Network Architectures

FLOWRRA provides two GNN implementations:

**1. GAT-based Agent (agent.py - Recommended)**
```python
# Graph Attention Network with:
# - Multi-head attention (4 heads default)
# - LSTM temporal memory for path stability
# - Epsilon-greedy exploration with Gaussian schedule
# - Gradient masking for frozen nodes

from holon.agent import GNNAgent

agent = GNNAgent(
    node_feature_dim=input_dim,
    action_size=4,  # 2D: left, right, up, down
    hidden_dim=128,
    n_heads=4,
    num_layers=3
)
```

**2. Relational GNN (r_gnn_agent.py - Experimental)**
```python
# Enhanced relational reasoning for complex coordination
from holon.r_gnn_agent import R_GNN_Agent

agent = R_GNN_Agent(
    node_feature_dim=input_dim,
    action_size=4,
    hidden_dim=128
)
```

### Density Field Manipulation

The density field is the spatial memory of the swarm, encoding failure history:

```python
from holon.density import DensityFunctionEstimatorND

density = DensityFunctionEstimatorND(
    dimensions=2,
    local_grid_size=(5, 5),
    global_grid_shape=(60, 60)
)

# When collision occurs, splat repulsion
density.splat_collision_event(
    position=collision_pos,
    velocity=attempted_velocity,
    severity=0.5,  # How bad the collision
    is_wfc_event=False  # Regular collision vs WFC trigger
)

# Get affordance potential (1.0 = safe, 0.0 = dangerous)
affordance = density.get_affordance_potential_for_node(
    node_pos=current_pos,
    repulsion_sources=detected_obstacles
)
```

### Wave Function Collapse Configuration

Tune WFC behavior in `config.py`:

```python
CONFIG = {
    "wfc": {
        "collapse_threshold": 0.6,  # Coherence below this triggers WFC
        "tau": 3,  # Consecutive unstable steps before collapse
        "history_length": 200,  # How many states to remember
        "tail_length": 15,  # Coherent sequence length for recovery
        
        # Spatial collapse tuning (forward-looking)
        "spatial_search_radius_mult": 0.9,
        "spatial_samples": 32,
        "spatial_accept_threshold": 0.65,
        "spatial_improvement_min": 0.7,
    }
}
```

**Spatial vs Temporal Recovery**:
- **Spatial (Forward)**: Uses current density field to project into high-affordance regions
- **Temporal (Backward)**: Returns to historically coherent states via manifold smoothing
- The system tries spatial first (anticipatory), falls back to temporal (safe)

## 💡 Pro Tips for Better Results

### Training Tips

1. **Start Small, Scale Up**
   - Begin with 16-24 nodes and 2 holons to validate your setup
   - Once stable, scale to 32-40 nodes and 4 holons
   - For research: 64+ nodes with 8 holons (requires powerful hardware)

2. **Parallel Mode is Your Friend**
   - Always use `--parallel` for multi-holon systems
   - Expect 10-20x speedup on modern CPUs
   - Monitor with `htop` to see all cores engaged

3. **Watch for the "Breathing" Pattern**
   - Healthy swarms show rhythmic expansion/contraction
   - Look for periodic WFC triggers (not every step!)
   - Coverage should climb steadily: 20% → 50% → 85% → 95%

4. **Tune for Your Hardware**
   - GPU available? Increase `hidden_dim` to 256 for richer representations
   - Limited RAM? Reduce `buffer_capacity` to 10000
   - Fast CPU? Increase `steps_per_episode` to 1500 for longer training

### Configuration Tips

1. **WFC Tuning for Different Environments**
   - **Dense obstacles**: Lower `collapse_threshold` (0.5) for earlier recovery
   - **Open spaces**: Higher `collapse_threshold` (0.7) for more exploration
   - **Moving obstacles**: Increase `spatial_samples` (48) for better prediction

2. **Reward Shaping**
   - High `r_explore` (15.0) = aggressive exploration
   - High `r_loop_integrity` (10.0) = tight formations
   - Balance is key: exploration needs coherence to be valuable

3. **Obstacle Placement**
   - Too easy: No obstacles → swarm just wanders
   - Too hard: >6 obstacles per holon → constant collapse
   - Sweet spot: 4-5 obstacles with varied sizes

### Visualization Tips

1. **Reading the Visualizations**
   - **Cyan/Blue nodes moving**: Healthy exploration
   - **Gold stars appearing**: Strategic freezing (mission success!)
   - **Red broken connections**: Normal during learning (should decrease)
   - **Clusters forming**: Holons self-organizing

2. **What Good Training Looks Like**
   ```
   Episode 10:  Reward: 5.2   | Coverage: 23% | WFC: 15 triggers
   Episode 50:  Reward: 12.8  | Coverage: 67% | WFC: 8 triggers
   Episode 100: Reward: 18.4  | Coverage: 94% | WFC: 3 triggers
   ```
   - Rewards climbing ✅
   - Coverage increasing ✅
   - WFC triggers decreasing ✅

3. **Signs of Problems**
   - Rewards oscillating wildly → reduce learning rate
   - Coverage stuck <40% → increase exploration reward
   - WFC every step → lower coherence threshold or fix obstacles

### Experimentation Ideas

1. **Test Resilience**
   - Add moving obstacles after episode 50
   - Randomly "kill" nodes mid-episode (freeze them)
   - Change obstacle positions between episodes

2. **Compare Architectures**
   - Run with `agent.py` (GAT-GNN) baseline
   - Switch to `r_gnn_agent.py` for comparison
   - Measure: training speed, final coverage, WFC frequency

3. **Ablation Studies**
   - Disable spatial collapse: set `spatial_samples=0`
   - Disable density splatting: comment out collision splatting
   - Remove spring forces: set `stiffness=0`
   - See what breaks! (Understanding through controlled destruction)

---



### Common Issues

**Issue**: Training is slow
- **Solution**: Use `--parallel` flag for multi-core execution
- **Alternative**: Reduce `--nodes` (fewer agents = faster training)
- **Check**: Make sure PyTorch is using GPU if available (`torch.cuda.is_available()`)

**Issue**: Memory errors during training
- **Solution**: Decrease `GRID_SIZE` in `config.py` (default: 80x80 → try 60x60)
- **Alternative**: Reduce `buffer_capacity` in GNN config (default: 15000 → try 10000)
- **Check**: Monitor RAM usage - federated mode uses ~2-4GB per holon

**Issue**: Poor convergence (rewards not improving)
- **Solution**: Adjust `LEARNING_RATE` in config (default: 0.0003 → try 0.001)
- **Check**: Look at `training_metrics.json` - is GNN loss decreasing?
- **Tip**: Increase `EPISODES` - complex swarms need 200+ episodes to learn
- **Try**: Tune WFC parameters - if system collapses too often, increase `collapse_threshold`

**Issue**: Nodes getting stuck or not moving
- **Solution**: The unstick mechanism should handle this automatically (see `core.py`)
- **Check**: Are all nodes frozen? Adjust freeze cycle parameters in config
- **Tip**: Increase `move_speed` in node config (default: 0.015 → try 0.02)

**Issue**: Loop keeps breaking
- **Solution**: Increase `ideal_distance` in loop config (default: 0.6 → try 0.7)
- **Alternative**: Decrease `break_threshold` (default: 0.35 → try 0.4)
- **Check**: Are obstacles too dense? Reduce obstacle count in config

**Issue**: Visualization shows nodes outside bounds
- **Solution**: This is intentional! Nodes use toroidal wrapping (they wrap around edges)
- **Check**: Make sure you're using the correct `deployment_ep*.json` file
- **Tip**: Open the JSON file to verify node positions are in [0, 1] range

**Issue**: `visualization.html` not loading trajectory
- **Solution**: Make sure trajectory JSON is from `deployment/` folder, not raw checkpoints
- **Check**: Run `deploy.py` first to generate proper trajectory files
- **Format**: Trajectory files should contain `{"trajectory": [...], "metadata": {...}}`
- **Tip**: Inside the `<script>` of `visualization.html` around line `655` you update the file. (`1500 steps`)

**Issue**: WFC triggers too frequently
- **Solution**: Increase `collapse_threshold` in config (default: 0.6 → try 0.7)
- **Adjust**: Increase `tau` (consecutive unstable steps, default: 3 → try 5)
- **Tune**: Spatial collapse parameters - see Advanced Usage section

**Issue**: GPU not being utilized
- **Check**: `torch.cuda.is_available()` should return `True`
- **Solution**: Install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **Note**: CPU training works fine for small swarms (<40 nodes)

### Debug Mode

Enable verbose logging:
```bash
# Set environment variable for detailed output
export FLOWRRA_DEBUG=1
python main.py --episodes 10 --nodes 16
```

Check metrics files:
```bash
# View training progress
cat metrics/training_metrics.json | jq '.episode_rewards[-10:]'

# Check WFC triggers
cat metrics/holon_0_detailed.json | jq '.wfc_trigger_events | length'
```

### Still Having Issues?

1. Check `FLOWRRA_JOURNEY/` folder - old experiments might have useful configs
2. Review the Substack article for conceptual understanding.
3. Open an issue on GitHub with:
   - Your config.py settings
   - Training command used
   - Error messages or unexpected behavior
   - System info (OS, Python version, PyTorch version)

---

## License

[CC-BY-4.0 license]

## Contributing
Contributions and Collaborations are welcome: Reach out to the Author.

**Areas for Contribution:**
- 3D environment support (elevation dynamics)
- Additional GNN architectures (Transformer-based attention)
- Real-world deployment adaptations (ROS integration)
- Benchmark comparisons (vs MAPPO, QMIX, etc.)
- Visualization enhancements (3D WebGL viewer)

## Contact

For questions, collaboration, or discussions:
- **Author**: Rohit Tamidapati ft. Claude, Gemini, Grok & ChatGPT (Free Versions)
- **DhaaRn**: [https://dhaarn.com]
- **LinkedIn**: [https://www.linkedin.com/in/rohit-tamidapati/]
- **Email**: rohit.tamidapati@dhaarn.com

## Roadmap & Future Work

### Short Term (Next 3-6 Months)
- [ ] **BenchMARL Integration**: Complete standardized benchmarking suite
- [ ] **Real Robot Testing**: Deploy on actual drone swarms (Crazyflies)
- [ ] **3D Full Support**: Extend all components to true 3D (not just 2.5D)
- [ ] **Memory Optimization**: Structured dictionaries for 40-60% speedup
- [ ] **True Decentralization**: Remove Federation central node entirely

### Medium Term (6-12 Months)
- [ ] **Quantum WFC**: Integrate actual quantum random number generators
- [ ] **Meta-Learning WFC**: Learn the coherence metric itself (Φ becomes learnable)
- [ ] **Gossip Protocol**: Inter-holon knowledge transfer for faster convergence
- [ ] **ROS2 Package**: Production-ready deployment for robotics
- [ ] **Unity/Unreal Sim**: High-fidelity 3D simulation environments

### Long Term Vision
- [ ] **Biological Validation**: Compare with actual insect swarm behavior
- [ ] **Satellite Constellation**: LEO deployment for space debris avoidance
- [ ] **Smart City Pilot**: Traffic/emergency coordination demonstration
- [ ] **Neuromorphic Hardware**: Deploy on brain-inspired chips (Intel Loihi)
- [ ] **Cognitive Architecture**: Multi-swarm "thoughts" via wave interference

## Citation

If you want to use FLOWRRA in your research, reach out to the author:
- [rohit.tamidapati@dhaarn.com]
- [rohittamidapati11@gmail.com]

```

## Acknowledgments

FLOWRRA draws inspiration from:
- **Active Inference** (Karl Friston) - Free energy minimization
- **Holonic Systems** (Arthur Koestler) - Self-similar hierarchies  
- **Quantum Mechanics** - Wave function collapse metaphor
- **Swarm Intelligence** - Emergent coordination without central control
- **Graph Neural Networks** - Relational reasoning in multi-agent systems

Special thanks to the broader AI/ML and multi-agent systems research communities.

Immense gratitude to the LLMs to have enhanced the speed of FLOWRRA's realization.

---

**DhaaRn** | Call from the unknown, answered.

*"Intelligence- A vibrant harmonious dance of the darkness and light."*

---

**DhaaRn** | Weavers of Time.
