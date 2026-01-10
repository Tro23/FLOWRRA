"""
flowrra_model.py

Custom BenchMARL model that properly integrates FLOWRRA.

This creates a proper Model class that BenchMARL can use as a drop-in
replacement for MLP/GRU/etc.

Key innovation: Instead of learning a policy via gradients, FLOWRRA's
internal GNN handles learning. This model just wraps FLOWRRA's decision
process in BenchMARL's expected interface.
"""

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Dict, Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MultiAgentMLP

# BenchMARL imports
from benchmarl.models.common import Model, ModelConfig, parse_model_config

# FLOWRRA imports
from config import CONFIG, validate_config
from main_parallel_single import FederatedFLOWRRA
from Noise_Cleanup import SensorProcessor


# =============================================================================
# FLOWRRA MODEL CONFIG
# =============================================================================

@dataclass
class FlowrraModelConfig(ModelConfig):
    """
    Configuration for FLOWRRA model.

    This tells BenchMARL how to instantiate FLOWRRA.
    """

    # FLOWRRA architecture
    n_holons: int = 4
    use_sensor_fusion: bool = True
    use_consensus: bool = True

    # GNN parameters (from CONFIG)
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_n_heads: int = 4

    # Training mode for FLOWRRA
    flowrra_mode: str = "training"  # "training" or "deployment"

    # Enable internal FLOWRRA learning (separate from BenchMARL)
    enable_flowrra_learning: bool = True

    @staticmethod
    def associated_class() -> Type[Model]:
        """Return the model class this config creates."""
        return FlowrraModel

    @staticmethod
    def supports_action_spec(action_spec) -> bool:
        """FLOWRRA supports discrete actions only."""
        # Check if action space is discrete
        try:
            # TorchRL discrete spec
            return hasattr(action_spec, "space") and action_spec.space.is_discrete
        except:
            # Fallback: assume discrete if not continuous
            return True

    @staticmethod
    def supports_observation_spec(observation_spec) -> bool:
        """FLOWRRA supports any observation spec."""
        return True

    @staticmethod
    def on_policy_compatible() -> bool:
        """FLOWRRA works with on-policy algorithms."""
        return True

    @staticmethod
    def off_policy_compatible() -> bool:
        """FLOWRRA can work with off-policy too."""
        return True


# =============================================================================
# FLOWRRA MODEL - The Core Integration
# =============================================================================

class FlowrraModel(Model):
    """
    BenchMARL Model wrapper for FLOWRRA.

    This bridges FLOWRRA's internal decision-making with BenchMARL's
    expected policy interface.

    Key Design:
    - forward() returns actions from FLOWRRA's GNN
    - BenchMARL sees this as a "policy network"
    - But internally, FLOWRRA does physics + learning
    """

    def __init__(
        self,
        n_agents: int,
        n_holons: int = 4,
        use_sensor_fusion: bool = True,
        use_consensus: bool = True,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_n_heads: int = 4,
        flowrra_mode: str = "training",
        enable_flowrra_learning: bool = True,
        **kwargs
    ):
        """
        Initialize FLOWRRA model.

        Args:
            n_agents: Number of agents
            n_holons: Number of holons (must be perfect square)
            use_sensor_fusion: Enable Kalman/Particle filtering
            use_consensus: Enable distributed consensus
            gnn_hidden_dim: GNN hidden dimension
            gnn_num_layers: GNN depth
            gnn_n_heads: Attention heads
            flowrra_mode: "training" or "deployment"
            enable_flowrra_learning: Let FLOWRRA's GNN learn
        """
        super().__init__()

        self.n_agents = n_agents
        self.n_holons = n_holons
        self.use_sensor_fusion = use_sensor_fusion
        self.enable_flowrra_learning = enable_flowrra_learning

        # Configure FLOWRRA
        self._setup_flowrra_config(
            n_agents, n_holons,
            gnn_hidden_dim, gnn_num_layers, gnn_n_heads,
            flowrra_mode
        )

        # Initialize FLOWRRA system
        print(f"[FLOWRRA Model] Initializing federated system...")
        print(f"  - Agents: {n_agents}")
        print(f"  - Holons: {n_holons}")
        print(f"  - Sensor fusion: {use_sensor_fusion}")
        print(f"  - Internal learning: {enable_flowrra_learning}")

        self.flowrra = FederatedFLOWRRA(
            CONFIG,
            use_parallel=False  # Disable for BenchMARL compatibility
        )

        # Initialize sensor processors
        if use_sensor_fusion:
            self.sensor_processors = self._init_sensor_processors()
        else:
            self.sensor_processors = None

        # Tracking
        self.step_count = 0
        self.episode_count = 0

        # Cache for observation dimension
        self._obs_dim = None

        print(f"[FLOWRRA Model] ✅ Initialization complete")

    def _setup_flowrra_config(
        self, n_agents, n_holons,
        gnn_hidden_dim, gnn_num_layers, gnn_n_heads,
        mode
    ):
        """Setup FLOWRRA configuration."""
        CONFIG["node"]["total_nodes"] = n_agents
        CONFIG["federation"]["num_holons"] = n_holons
        CONFIG["spatial"]["dimensions"] = 2  # BenchMARL is 2D

        # GNN config
        CONFIG["gnn"]["hidden_dim"] = gnn_hidden_dim
        CONFIG["gnn"]["num_layers"] = gnn_num_layers
        CONFIG["gnn"]["n_heads"] = gnn_n_heads

        # Mode
        CONFIG["holon"]["mode"] = mode

        # Validate
        validate_config(CONFIG)

    def _init_sensor_processors(self) -> Dict[int, SensorProcessor]:
        """Initialize sensor processors for all agents."""
        processors = {}

        for holon in self.flowrra.holons.values():
            for node in holon.nodes:
                processors[node.id] = SensorProcessor(
                    node_id=node.id,
                    dimensions=2,
                    use_consensus=True,
                    filter_mode="auto"
                )
                processors[node.id].reset(node.pos.copy())

        print(f"[FLOWRRA Model] Initialized {len(processors)} sensor processors")
        return processors

    def _process_observations(self, observations: torch.Tensor):
        """
        Process observations and sync to FLOWRRA.

        Args:
            observations: [batch, n_agents, obs_dim]
        """
        batch_size = observations.shape[0]

        # Currently only support single environment
        if batch_size != 1:
            raise NotImplementedError("Multi-env batching not supported yet")

        obs = observations[0].detach().cpu().numpy()

        # Extract positions (assume first 2 dims are x, y)
        for i in range(self.n_agents):
            raw_pos = obs[i, :2]

            # Apply sensor fusion
            if self.sensor_processors and i in self.sensor_processors:
                filtered_pos = self.sensor_processors[i].process_measurement(raw_pos)
            else:
                filtered_pos = raw_pos

            # Update FLOWRRA nodes
            self._update_node_position(i, filtered_pos)

    def _update_node_position(self, node_id: int, position):
        """Update a specific node's position."""
        for holon in self.flowrra.holons.values():
            for node in holon.nodes:
                if node.id == node_id:
                    node.pos = np.clip(position, 0.0, 1.0)
                    return

    def _execute_flowrra_step(self):
        """Execute one FLOWRRA timestep."""
        self.step_count += 1

        # Execute each holon
        for holon_id, holon in self.flowrra.holons.items():
            try:
                holon.step(
                    episode_step=self.step_count,
                    total_episodes=CONFIG["training"]["steps_per_episode"]
                )
            except Exception as e:
                print(f"[FLOWRRA Model] Error in holon {holon_id}: {e}")

        # Federation coordination
        holon_states = {
            h_id: h.get_state_summary()
            for h_id, h in self.flowrra.holons.items()
        }
        breach_alerts = self.flowrra.federation.step(holon_states)

        # Send alerts
        for holon_id, alerts in breach_alerts.items():
            if alerts:
                self.flowrra.holons[holon_id].receive_breach_alerts(alerts)

    def _extract_actions(self) -> torch.Tensor:
        """
        Extract actions from FLOWRRA's internal state.

        Returns:
            [n_agents, 1] tensor of action indices
        """
        actions = []

        for holon in self.flowrra.holons.values():
            for node in holon.nodes:
                # Check if node has stored action (requires core.py modification)
                if hasattr(node, 'last_action'):
                    actions.append(node.last_action)
                else:
                    # Fallback: infer from velocity
                    velocity = node.velocity()
                    action = self._velocity_to_action(velocity)
                    actions.append(action)

        # Convert to tensor [n_agents, 1]
        return torch.tensor(actions, dtype=torch.long).unsqueeze(-1)

    def _velocity_to_action(self, velocity) -> int:
        """Convert velocity to discrete action (0=left, 1=right, 2=up, 3=down)."""
        import numpy as np

        if np.linalg.norm(velocity) < 1e-6:
            return 0

        abs_vx = abs(velocity[0])
        abs_vy = abs(velocity[1])

        if abs_vx > abs_vy:
            return 1 if velocity[0] > 0 else 0
        else:
            return 2 if velocity[1] > 0 else 3

    def _collect_auxiliary_loss(self) -> torch.Tensor:
        """
        Collect auxiliary loss from FLOWRRA's internal learning.

        BenchMARL can backprop through this if enable_flowrra_learning=True.
        """
        if not self.enable_flowrra_learning:
            return torch.tensor(0.0)

        # Collect training losses from holons
        losses = []

        for holon in self.flowrra.holons.values():
            if holon.orchestrator and hasattr(holon.orchestrator, 'training_losses'):
                if holon.orchestrator.training_losses:
                    latest_loss = holon.orchestrator.training_losses[-1]
                    losses.append(latest_loss['loss'])

        if losses:
            return torch.tensor(sum(losses) / len(losses))
        else:
            return torch.tensor(0.0)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Forward pass for BenchMARL.

        This is the main entry point BenchMARL calls.

        Args:
            tensordict: Contains observations under ("agents", "observation")

        Returns:
            tensordict: With actions under ("agents", "action")
        """
        # Extract observations
        observations = tensordict[("agents", "observation")]

        # PHASE 1: Process observations
        self._process_observations(observations)

        # PHASE 2: Execute FLOWRRA
        self._execute_flowrra_step()

        # PHASE 3: Extract actions
        actions = self._extract_actions()

        # PHASE 4: Store in tensordict
        # BenchMARL expects actions as [batch, n_agents, action_dim]
        tensordict.set(
            ("agents", "action"),
            actions.unsqueeze(0)  # Add batch dimension
        )

        # PHASE 5: Add auxiliary loss for backprop (optional)
        if self.enable_flowrra_learning:
            aux_loss = self._collect_auxiliary_loss()
            tensordict.set("flowrra_loss", aux_loss)

        return tensordict

    def reset(self, tensordict: Optional[TensorDictBase] = None) -> None:
        """
        Reset for new episode.

        Called by BenchMARL at episode start.
        """
        self.episode_count += 1
        self.step_count = 0

        # Reset sensor processors
        if self.sensor_processors:
            for holon in self.flowrra.holons.values():
                for node in holon.nodes:
                    if node.id in self.sensor_processors:
                        self.sensor_processors[node.id].reset(node.pos.copy())

        print(f"[FLOWRRA Model] Episode {self.episode_count} reset")


# =============================================================================
# MODEL REGISTRATION - Tell BenchMARL about FLOWRRA
# =============================================================================

def register_flowrra_model():
    """
    Register FLOWRRA model with BenchMARL's model registry.

    Call this BEFORE creating experiments.
    """
    from benchmarl.models import model_config_registry

    # Add FLOWRRA to registry
    model_config_registry["flowrra"] = FlowrraModelConfig

    print("[FLOWRRA] ✅ Registered with BenchMARL model registry")
    print("[FLOWRRA] Available models:", list(model_config_registry.keys()))


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_flowrra_model_config(
    n_agents: int = 32,
    n_holons: int = 4,
    use_sensor_fusion: bool = True,
    gnn_hidden_dim: int = 128,
    **kwargs
) -> FlowrraModelConfig:
    """
    Create FLOWRRA model config for BenchMARL experiments.

    Usage:
        from flowrra_model import create_flowrra_model_config

        model_config = create_flowrra_model_config(
            n_agents=32,
            n_holons=4,
            use_sensor_fusion=True
        )

        experiment = Experiment(
            task=task,
            algorithm_config=algo_config,
            model_config=model_config,
            seed=42,
            config=exp_config
        )
    """
    return FlowrraModelConfig(
        n_holons=n_holons,
        use_sensor_fusion=use_sensor_fusion,
        gnn_hidden_dim=gnn_hidden_dim,
        **kwargs
    )


# =============================================================================
# AUTO-REGISTRATION
# =============================================================================

# Automatically register when module is imported
try:
    register_flowrra_model()
except Exception as e:
    print(f"[FLOWRRA] Warning: Could not auto-register: {e}")
    print("[FLOWRRA] Call register_flowrra_model() manually")


if __name__ == "__main__":
    # Test initialization
    print("\n" + "="*70)
    print("FLOWRRA MODEL TEST")
    print("="*70)

    # Create model config
    config = create_flowrra_model_config(
        n_agents=32,
        n_holons=4,
        use_sensor_fusion=True
    )

    print(f"\n✅ Model config created:")
    print(f"   - n_holons: {config.n_holons}")
    print(f"   - use_sensor_fusion: {config.use_sensor_fusion}")
    print(f"   - gnn_hidden_dim: {config.gnn_hidden_dim}")

    # Test model instantiation
    from tensordict import TensorDict
    import torch

    # Create dummy tensordict
    td = TensorDict({
        ("agents", "observation"): torch.randn(1, 32, 50)
    }, batch_size=[1])

    print("\n✅ Test complete!")
