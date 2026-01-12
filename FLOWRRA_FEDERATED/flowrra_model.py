"""
flowrra_model.py - FIXED

The issue: FlowrraModel didn't implement the abstract _forward() method
required by BenchMARL's Model base class.

Fix: Implement _forward() which is called internally by the Model class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type

import numpy as np
import torch
from benchmarl.models.common import Model, ModelConfig
from tensordict import TensorDictBase
from torch import nn

# FLOWRRA imports
from config import CONFIG, validate_config
from main import FederatedFLOWRRA
from Noise_Cleanup import SensorProcessor


@dataclass
class FlowrraModelConfig(ModelConfig):
    """Configuration for FLOWRRA model."""

    n_holons: int = 4
    use_sensor_fusion: bool = True
    use_consensus: bool = True
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_n_heads: int = 4
    flowrra_mode: str = "training"
    enable_flowrra_learning: bool = True

    @staticmethod
    def associated_class() -> Type[Model]:
        return FlowrraModel

    def get_model(
        self,
        input_spec,
        output_spec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: str,
        action_spec,
        model_index: int = 0,
    ):
        """
        Instantiate the FLOWRRA model with BenchMARL's required arguments.
        """
        return self.associated_class()(
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_group,
            input_has_agent_dim=input_has_agent_dim,
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            action_spec=action_spec,
            model_index=model_index,
            is_critic=False,  # FLOWRRA is the actor (policy network)
            # FLOWRRA specific params
            n_holons=self.n_holons,
            use_sensor_fusion=self.use_sensor_fusion,
            use_consensus=self.use_consensus,
            gnn_hidden_dim=self.gnn_hidden_dim,
            gnn_num_layers=self.gnn_num_layers,
            gnn_n_heads=self.gnn_n_heads,
            flowrra_mode=self.flowrra_mode,
            enable_flowrra_learning=self.enable_flowrra_learning,
        )

    @staticmethod
    def supports_action_spec(action_spec) -> bool:
        return True  # Support all action specs

    @staticmethod
    def supports_observation_spec(observation_spec) -> bool:
        return True

    @staticmethod
    def on_policy_compatible() -> bool:
        return True

    @staticmethod
    def off_policy_compatible() -> bool:
        return True


class FlowrraModel(Model):
    """
    FIXED: BenchMARL Model wrapper for FLOWRRA.

    Key fix: Implements _forward() instead of forward()
    """

    def __init__(
        self,
        # BenchMARL required arguments
        input_spec,
        output_spec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: str,
        action_spec,
        model_index: int,
        is_critic: bool,
        # FLOWRRA specific arguments
        n_holons: int = 4,
        use_sensor_fusion: bool = True,
        use_consensus: bool = True,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_n_heads: int = 4,
        flowrra_mode: str = "training",
        enable_flowrra_learning: bool = True,
        **kwargs,
    ):
        # Call parent constructor with required BenchMARL arguments
        super().__init__(
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_group,
            input_has_agent_dim=input_has_agent_dim,
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            action_spec=action_spec,
            model_index=model_index,
            is_critic=is_critic,
        )

        self.n_agents = n_agents
        self.n_holons = n_holons
        self.use_sensor_fusion = use_sensor_fusion
        self.enable_flowrra_learning = enable_flowrra_learning

        # Configure FLOWRRA
        self._setup_flowrra_config(
            n_agents,
            n_holons,
            gnn_hidden_dim,
            gnn_num_layers,
            gnn_n_heads,
            flowrra_mode,
        )

        print(f"[FLOWRRA Model] Initializing federated system...")
        print(f"  - Agents: {n_agents}")
        print(f"  - Holons: {n_holons}")
        print(f"  - Sensor fusion: {use_sensor_fusion}")

        self.flowrra = FederatedFLOWRRA(CONFIG, use_parallel=False)

        # Initialize sensor processors
        if use_sensor_fusion:
            self.sensor_processors = self._init_sensor_processors()
        else:
            self.sensor_processors = None

        # Tracking
        self.step_count = 0
        self.episode_count = 0

        # CRITICAL FIX: BenchMARL requires at least one learnable parameter
        # for the optimizer. Add a dummy parameter that won't affect FLOWRRA.
        self.dummy_param = nn.Parameter(torch.zeros(1, device=device))

        print(f"[FLOWRRA Model] ✅ Initialization complete")
        print(f"[FLOWRRA Model] Added dummy parameter for BenchMARL optimizer")

    def _setup_flowrra_config(
        self, n_agents, n_holons, gnn_hidden_dim, gnn_num_layers, gnn_n_heads, mode
    ):
        """Setup FLOWRRA configuration."""
        CONFIG["node"]["total_nodes"] = n_agents
        CONFIG["federation"]["num_holons"] = n_holons
        CONFIG["spatial"]["dimensions"] = 2

        CONFIG["gnn"]["hidden_dim"] = gnn_hidden_dim
        CONFIG["gnn"]["num_layers"] = gnn_num_layers
        CONFIG["gnn"]["n_heads"] = gnn_n_heads
        CONFIG["holon"]["mode"] = mode

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
                    filter_mode="auto",
                )
                processors[node.id].reset(node.pos.copy())

        print(f"[FLOWRRA Model] Initialized {len(processors)} sensor processors")
        return processors

    def _process_observations(self, observations: torch.Tensor):
        """Process observations and sync to FLOWRRA."""
        batch_size = observations.shape[0]

        # Handle batched environments by processing only the first environment
        # TODO: Support multiple parallel environments for speedup
        if batch_size > 1:
            print(
                f"[FLOWRRA Model] Warning: Processing only first of {batch_size} parallel envs"
            )
            observations = observations[0:1]  # Take only first env

        obs = observations[0].detach().cpu().numpy()

        # Extract positions (assume first 2 dims are x, y)
        for i in range(self.n_agents):
            if i >= obs.shape[0]:
                print(
                    f"[FLOWRRA Model] Warning: Agent {i} not in obs (shape {obs.shape})"
                )
                continue

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
                    total_episodes=CONFIG["training"]["steps_per_episode"],
                )
            except Exception as e:
                print(f"[FLOWRRA Model] Error in holon {holon_id}: {e}")

        # Federation coordination
        holon_states = {
            h_id: h.get_state_summary() for h_id, h in self.flowrra.holons.items()
        }
        breach_alerts = self.flowrra.federation.step(holon_states)

        # Send alerts
        for holon_id, alerts in breach_alerts.items():
            if alerts:
                self.flowrra.holons[holon_id].receive_breach_alerts(alerts)

    def _extract_action_logits(
        self, batch_size: int = 1, action_dim: int = 5
    ) -> torch.Tensor:
        """
        Extract action parameters from FLOWRRA's internal state.

        For continuous action spaces, we output action values directly.
        For Simple Spread: [force_x, force_y, communication_channel_1, channel_2, channel_3]

        Args:
            batch_size: Number of parallel environments
            action_dim: Action dimension (5 for Simple Spread continuous)

        Returns:
            [batch_size, n_agents, action_dim] tensor of action parameters
        """
        action_vectors = []

        for holon in self.flowrra.holons.values():
            for node in holon.nodes:
                # Get velocity (already computed by FLOWRRA)
                velocity = node.velocity()

                # Convert to continuous action for Simple Spread
                # Actions: [force_x, force_y, comm_1, comm_2, comm_3]
                # Use velocity for movement, zeros for communication
                if len(velocity) >= 2:
                    force_x = float(velocity[0]) * 0.1  # Scale down velocity
                    force_y = float(velocity[1]) * 0.1
                else:
                    force_x = 0.0
                    force_y = 0.0

                # Full action vector
                action_vec = [force_x, force_y, 0.0, 0.0, 0.0]  # No communication
                action_vectors.append(action_vec)

        # Convert to tensor [n_agents, action_dim]
        actions_tensor = torch.tensor(action_vectors, dtype=torch.float32)

        # Expand to [batch_size, n_agents, action_dim]
        if batch_size > 1:
            actions_tensor = actions_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            actions_tensor = actions_tensor.unsqueeze(0)

        return actions_tensor

    def _velocity_to_action(self, velocity) -> int:
        """Convert velocity to discrete action."""
        if np.linalg.norm(velocity) < 1e-6:
            return 0

        abs_vx = abs(velocity[0])
        abs_vy = abs(velocity[1])

        if abs_vx > abs_vy:
            return 1 if velocity[0] > 0 else 0
        else:
            return 2 if velocity[1] > 0 else 3

    # =========================================================================
    # KEY FIX: Implement _forward() instead of forward()
    # =========================================================================

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        FIXED: This is the abstract method that BenchMARL's Model requires.

        The base Model class calls this internally.
        """
        # Extract observations - handle both 'agent' and 'agents' keys
        # BenchMARL inconsistently uses singular vs plural
        if ("agents", "observation") in tensordict.keys(include_nested=True):
            observations = tensordict[("agents", "observation")]
            obs_key = ("agents", "observation")
            logits_key = ("agents", "logits")
        elif ("agent", "observation") in tensordict.keys(include_nested=True):
            observations = tensordict[("agent", "observation")]
            obs_key = ("agent", "observation")
            logits_key = ("agent", "logits")
        elif "observation" in tensordict.keys():
            observations = tensordict["observation"]
            obs_key = "observation"
            logits_key = "logits"
        else:
            # Fallback: print available keys for debugging
            print(f"[FLOWRRA Model] Available keys: {list(tensordict.keys())}")
            raise KeyError("Could not find observation in tensordict")

        # Get batch size
        batch_size = observations.shape[0]

        # PHASE 1: Process observations
        self._process_observations(observations)

        # PHASE 2: Execute FLOWRRA
        self._execute_flowrra_step()

        # PHASE 3: Extract action parameters
        # BenchMARL expects [mean_1, mean_2, ..., log_std_1, log_std_2, ...]
        logits = self._extract_action_logits(batch_size=batch_size)

        # PHASE 4: Store logits in tensordict
        tensordict.set(logits_key, logits)

        return tensordict

    def reset(self, tensordict: Optional[TensorDictBase] = None) -> None:
        """Reset for new episode."""
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
# MODEL REGISTRATION
# =============================================================================


def register_flowrra_model():
    """Register FLOWRRA model with BenchMARL."""
    from benchmarl.models import model_config_registry

    model_config_registry["flowrra"] = FlowrraModelConfig

    print("[FLOWRRA] ✅ Registered with BenchMARL model registry")
    print("[FLOWRRA] Available models:", list(model_config_registry.keys()))


def create_flowrra_model_config(
    n_agents: int = 32,
    n_holons: int = 4,
    use_sensor_fusion: bool = True,
    gnn_hidden_dim: int = 128,
    **kwargs,
) -> FlowrraModelConfig:
    """Create FLOWRRA model config for BenchMARL experiments."""
    return FlowrraModelConfig(
        n_holons=n_holons,
        use_sensor_fusion=use_sensor_fusion,
        gnn_hidden_dim=gnn_hidden_dim,
        **kwargs,
    )


# Auto-register
try:
    register_flowrra_model()
except Exception as e:
    print(f"[FLOWRRA] Warning: Could not auto-register: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FLOWRRA MODEL TEST")
    print("=" * 70)

    config = create_flowrra_model_config(n_agents=8, n_holons=2, use_sensor_fusion=True)

    print(f"\n✅ Model config created:")
    print(f"   - n_holons: {config.n_holons}")
    print(f"   - use_sensor_fusion: {config.use_sensor_fusion}")
    print(f"   - gnn_hidden_dim: {config.gnn_hidden_dim}")

    print("\n✅ Test complete!")
