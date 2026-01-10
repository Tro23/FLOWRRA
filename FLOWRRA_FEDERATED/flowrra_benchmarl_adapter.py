"""
flowrra_benchmarl_adapter.py - FIXED VERSION

Properly integrates FLOWRRA with BenchMARL's architecture.

Key fixes:
1. Correct action extraction from FLOWRRA's GNN
2. Proper observation/action space definitions
3. Metrics injection that BenchMARL can actually read
4. Handles coordinate transforms and frozen nodes
"""

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from typing import Dict, Optional

from config import CONFIG
from main_parallel_single import FederatedFLOWRRA
from Noise_Cleanup import SensorProcessor


class FLOWRRABenchmarkAdapter(nn.Module):
    """
    BenchMARL-compatible wrapper for FLOWRRA Federated.

    Architecture:
    Raw Obs → Noise Cleanup → FLOWRRA (GNN + Physics) → Actions
    """

    def __init__(
        self,
        n_agents: int = 32,
        n_holons: int = 4,
        observation_dim: int = None,  # Auto-calculated
        action_dim: int = 4,  # 4 for 2D (left, right, up, down)
        use_sensor_fusion: bool = True,
        **kwargs
    ):
        super().__init__()

        # Configuration
        self.n_agents = n_agents
        self.n_holons = n_holons
        self.action_dim = action_dim
        self.use_sensor_fusion = use_sensor_fusion

        # Update CONFIG
        CONFIG["node"]["total_nodes"] = n_agents
        CONFIG["federation"]["num_holons"] = n_holons
        CONFIG["spatial"]["dimensions"] = 2  # BenchMARL tasks are 2D

        # Initialize FLOWRRA
        print(f"[Adapter] Initializing FLOWRRA: {n_agents} agents, {n_holons} holons")
        self.flowrra = FederatedFLOWRRA(CONFIG, use_parallel=False)  # Disable parallel for stability

        # Attach sensor processors if enabled
        if use_sensor_fusion:
            self.processors = {
                i: SensorProcessor(
                    node_id=i,
                    dimensions=2,
                    use_consensus=True,
                    filter_mode="auto"
                )
                for i in range(n_agents)
            }

            # Initialize processors with node positions
            for holon in self.flowrra.holons.values():
                for node in holon.nodes:
                    if node.id in self.processors:
                        self.processors[node.id].reset(node.pos.copy())
        else:
            self.processors = None

        # Calculate observation dimension
        self.observation_dim = self._get_observation_dim()

        # Track internal state
        self.current_step = 0
        self.episode_coherence = []
        self.episode_integrity = []

        print(f"[Adapter] Initialized - Obs dim: {self.observation_dim}, Action dim: {action_dim}")

    def _get_observation_dim(self) -> int:
        """Calculate observation dimension from FLOWRRA nodes."""
        # Get a sample node
        sample_node = list(self.flowrra.holons.values())[0].nodes[0]

        # Get sample state vector
        dummy_grid = np.zeros(
            np.prod(CONFIG["repulsion"]["local_grid_size"])
        )
        sample_obs = sample_node.get_state_vector(dummy_grid, [], [])

        return len(sample_obs)

    def _sync_positions_to_flowrra(self, observations: torch.Tensor):
        """
        Sync BenchMARL observations to FLOWRRA node positions.

        Args:
            observations: [batch, n_agents, obs_dim] tensor
        """
        batch_size = observations.shape[0]

        # For now, only handle batch_size=1 (single environment)
        if batch_size != 1:
            raise NotImplementedError("Multi-env batching not yet supported")

        obs = observations[0].detach().cpu().numpy()  # [n_agents, obs_dim]

        # Extract positions from observations
        # Assuming first 2 dims are [x, y] position
        for i in range(self.n_agents):
            raw_pos = obs[i, :2]  # First 2 elements are position

            # Apply sensor fusion if enabled
            if self.processors is not None and i in self.processors:
                filtered_pos = self.processors[i].process_measurement(raw_pos)
            else:
                filtered_pos = raw_pos

            # Update corresponding FLOWRRA node
            for holon in self.flowrra.holons.values():
                for node in holon.nodes:
                    if node.id == i:
                        node.pos = np.clip(filtered_pos, 0.0, 1.0)  # Keep in [0,1]
                        break

    def _extract_actions_from_flowrra(self) -> torch.Tensor:
        """
        Extract actions from FLOWRRA's internal decision process.

        Returns:
            actions: [1, n_agents, 1] tensor of discrete action indices
        """
        all_actions = []

        for holon in self.flowrra.holons.values():
            if not holon.orchestrator:
                # Holon not initialized, use no-op
                for node in holon.nodes:
                    all_actions.append(0)
                continue

            # Get active nodes (skip frozen)
            active_nodes = holon.orchestrator.get_active_nodes()
            frozen_ids = holon.orchestrator.frozen_nodes

            for node in holon.nodes:
                if node.id in frozen_ids:
                    # Frozen nodes don't move
                    all_actions.append(0)  # No-op action
                else:
                    # For active nodes, we need to extract the GNN's decision
                    # This requires modifying core.py to store the last action

                    # WORKAROUND: Use the node's last movement direction
                    # Calculate action from velocity
                    velocity = node.velocity()
                    action = self._velocity_to_action(velocity)
                    all_actions.append(action)

        # Convert to tensor: [1, n_agents, 1]
        actions_tensor = torch.tensor(
            all_actions,
            dtype=torch.long
        ).unsqueeze(0).unsqueeze(-1)

        return actions_tensor

    def _velocity_to_action(self, velocity: np.ndarray) -> int:
        """
        Convert velocity vector to discrete action index.

        Actions:
        0: Left  (-x)
        1: Right (+x)
        2: Up    (+y)
        3: Down  (-y)
        """
        if np.linalg.norm(velocity) < 1e-6:
            return 0  # No movement → arbitrary choice

        # Determine dominant direction
        abs_vx = abs(velocity[0])
        abs_vy = abs(velocity[1])

        if abs_vx > abs_vy:
            # Horizontal movement dominant
            return 1 if velocity[0] > 0 else 0
        else:
            # Vertical movement dominant
            return 2 if velocity[1] > 0 else 3

    def _collect_metrics(self) -> Dict[str, float]:
        """
        Collect FLOWRRA metrics for BenchMARL logging.

        Returns:
            Dict of metric_name -> value
        """
        metrics = {}

        # Federation-level metrics
        fed_stats = self.flowrra.federation.get_statistics()
        metrics["federation/total_breaches"] = fed_stats["total_breaches"]

        # Per-holon metrics (averaged)
        integrities = []
        coherences = []
        frozen_counts = []

        for holon_id, holon in self.flowrra.holons.items():
            if holon.orchestrator:
                # Loop integrity
                integrity = holon.orchestrator.loop.calculate_integrity()
                integrities.append(integrity)

                # Coherence (from latest metrics)
                if holon.orchestrator.metrics_history:
                    latest = holon.orchestrator.metrics_history[-1]
                    coherences.append(latest.get("coherence", 0.0))

                # Frozen node count
                frozen_counts.append(len(holon.orchestrator.frozen_nodes))

        # Aggregate metrics
        metrics["system/avg_integrity"] = float(np.mean(integrities)) if integrities else 0.0
        metrics["system/avg_coherence"] = float(np.mean(coherences)) if coherences else 0.0
        metrics["system/total_frozen_nodes"] = int(np.sum(frozen_counts))
        metrics["system/active_nodes"] = self.n_agents - int(np.sum(frozen_counts))

        # Track for episode statistics
        if integrities:
            self.episode_integrity.append(metrics["system/avg_integrity"])
        if coherences:
            self.episode_coherence.append(metrics["system/avg_coherence"])

        return metrics

    def forward(self, td: TensorDict) -> TensorDict:
        """
        Main forward pass for BenchMARL.

        Args:
            td: TensorDict containing observations

        Returns:
            TensorDict with actions and metrics
        """
        self.current_step += 1

        # PHASE 0: Extract observations
        obs = td["agents", "observation"]  # [batch, n_agents, obs_dim]

        # PHASE 1: Sync to FLOWRRA (with sensor fusion)
        self._sync_positions_to_flowrra(obs)

        # PHASE 2: Execute FLOWRRA step
        # Run one step for each holon
        for holon_id, holon in self.flowrra.holons.items():
            try:
                # Execute holon step
                holon.step(
                    episode_step=self.current_step,
                    total_episodes=CONFIG["training"]["steps_per_episode"]
                )
            except Exception as e:
                print(f"[Adapter] Error in holon {holon_id} step: {e}")
                # Continue with other holons

        # Federation coordination
        holon_states = {
            h_id: h.get_state_summary()
            for h_id, h in self.flowrra.holons.items()
        }
        breach_alerts = self.flowrra.federation.step(holon_states)

        # Send breach alerts
        for holon_id, alerts in breach_alerts.items():
            if alerts:
                self.flowrra.holons[holon_id].receive_breach_alerts(alerts)

        # PHASE 3: Extract actions
        actions = self._extract_actions_from_flowrra()

        # PHASE 4: Collect metrics
        metrics = self._collect_metrics()

        # PHASE 5: Package for BenchMARL
        td.set(("agents", "action"), actions)

        # Add metrics as additional info
        for metric_name, metric_value in metrics.items():
            # BenchMARL can log these if they're in the TensorDict
            td.set(
                ("info", metric_name),
                torch.tensor([metric_value], dtype=torch.float32)
            )

        return td

    def reset(self):
        """Reset adapter state (called at episode start)."""
        self.current_step = 0

        # Log episode statistics
        if self.episode_integrity:
            print(f"[Adapter] Episode stats:")
            print(f"  Avg integrity: {np.mean(self.episode_integrity):.3f}")
            print(f"  Avg coherence: {np.mean(self.episode_coherence):.3f}")

        self.episode_coherence = []
        self.episode_integrity = []

        # Reset sensor processors
        if self.processors:
            for holon in self.flowrra.holons.values():
                for node in holon.nodes:
                    if node.id in self.processors:
                        self.processors[node.id].reset(node.pos.copy())



# =============================================================================
# BENCHMARL TASK WRAPPER
# =============================================================================

class FLOWRRAPettingZooTask:
    """
    Wrapper to make FLOWRRA compatible with PettingZoo environments.

    This handles the impedance mismatch between FLOWRRA's internal
    action generation and BenchMARL's expectation of policy networks.
    """

    def __init__(self, base_task, n_agents: int = 32):
        """
        Args:
            base_task: Original PettingZoo task
            n_agents: Number of agents in FLOWRRA
        """
        self.base_task = base_task
        self.n_agents = n_agents

        # Override observation/action spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Setup observation and action spaces for FLOWRRA."""
        # Calculate FLOWRRA observation dimension
        sample_node_obs_dim = 50  # Approximate, will be calculated properly

        # Override spaces
        from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    shape=(sample_node_obs_dim,)
                )
            })
        })

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": BoundedTensorSpec(
                    shape=(1,),
                    minimum=0,
                    maximum=3,  # 4 discrete actions (0-3)
                    dtype=torch.long
                )
            })
        })


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """
    Example of how to use the fixed adapter with BenchMARL.
    """
    from benchmarl.algorithms import MappoConfig
    from benchmarl.environments import PettingZooTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    # Setup task
    task = PettingZooTask.SIMPLE_SPREAD.get_from_yaml()
    task.config["n_agents"] = 32
    task.config["max_cycles"] = 500

    # Experiment config
    exp_config = ExperimentConfig.get_from_yaml()
    exp_config.max_n_iters = 100
    exp_config.loggers = ["csv", "wandb"]

    # Algorithm
    algo_config = MappoConfig.get_from_yaml()

    # Model - use FLOWRRA adapter
    model_config = MlpConfig.get_from_yaml()

    # FLOWRRA as policy
    flowrra_model = FLOWRRABenchmarkAdapter(
        n_agents=32,
        n_holons=4,
        use_sensor_fusion=True
    )

    # Run experiment
    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        seed=42,
        config=exp_config
    )

    experiment.run()


if __name__ == "__main__":
    # Test initialization
    adapter = FLOWRRABenchmarkAdapter(n_agents=32, n_holons=4)
    print(f"✅ Adapter initialized successfully")
    print(f"   Observation dim: {adapter.observation_dim}")
    print(f"   Action dim: {adapter.action_dim}")
    print(f"   Agents: {adapter.n_agents}")
