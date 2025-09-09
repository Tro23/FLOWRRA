"""
WaveFunctionCollapse_RL.py

Implements the loop-level collapse mechanism. It maintains a history of coherent
loop states. When the system's overall coherence drops below a threshold for a
sustained period, this class identifies a recent, high-quality "tail" of states
and uses it to re-initialize the system, pulling it out of a failing configuration.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from EnvironmentA_RL import Environment_A

class Wave_Function_Collapse:
    """
    Manages the loop-level collapse and re-initialization process.

    Attributes:
        history_length (int): Max number of historical states to store.
        tail_length (int): The length of the "coherent tail" to search for.
        collapse_threshold (float): Coherence value below which a state is considered unstable.
        tau (int): Number of consecutive unstable steps required to trigger a collapse.
        history (list): A buffer of recent loop snapshots and their coherence scores.
    """
    def __init__(self,
                 history_length: int = 200,
                 tail_length: int = 15,
                 collapse_threshold: float = 0.25,
                 tau: int = 5):
        self.history_length = history_length
        self.tail_length = tail_length
        self.collapse_threshold = collapse_threshold
        self.tau = tau
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        """
        Clears the history buffer.
        """
        self.history.clear()

    def assess_loop_coherence(self, coherence: float, env_snapshot: List[Dict[str, Any]]):
        """
        Records the overall coherence of the entire loop state.
        
        Note: The coherence calculation itself is now handled by the FLOWRRA_RL class.
        This method is purely for history management.

        Args:
            coherence (float): The coherence score for the current loop.
            env_snapshot (List[Dict[str, Any]]): A snapshot of the node states.
        """
        # Add the current state and its coherence to the history
        self.history.append({
            'state_snapshot': env_snapshot,
            'coherence': coherence
        })
        
        # Trim history to the specified length
        if len(self.history) > self.history_length:
            self.history.pop(0)
            
    def collapse_and_reinitialize(self, env: Environment_A, rewards) -> Dict[str, Any]:
        """
        Triggers a collapse if the system is in an unstable state and re-initializes
        the environment from a previously stable state in its history.
        
        Args:
            env (Environment_A): The environment to be reset.
            rewards (np.ndarray): The rewards for the current step. Unused here, but
                                  passed for compatibility with other methods.

        Returns:
            dict: A dictionary containing information about the collapse event.
        """
        # Look for a streak of unstable states at the end of the history
        # We assume this check is done by the calling class (FLOWRRA_RL)
        
        # If a collapse is triggered, search for a coherent tail in the history
        coherent_tail = None
        for i in range(len(self.history) - self.tail_length):
            tail = self.history[i:i + self.tail_length]
            # Check if all states in the tail are above the collapse threshold
            is_coherent_tail = all(h['coherence'] >= self.collapse_threshold for h in tail)
            if is_coherent_tail:
                # We found a suitable tail. The most recent one is usually best.
                coherent_tail = tail
                break

        if coherent_tail is None:
            # Fallback: If no good history exists, apply a small random perturbation.
            for node in env.nodes:
                node.pos = np.mod(node.pos + np.random.randn(2) * 0.05, 1.0)
            return {'reinit_from': 'random_jitter'}

        # --- Manifold Smoothing ---
        # As per the design, we smoothen the possibilities across the comet tail.
        # Here, we use a weighted average of the positions in the tail.
        num_nodes = len(coherent_tail[0]['state_snapshot'])
        positions_over_time = np.array([[node['pos'] for node in snapshot['state_snapshot']] for snapshot in coherent_tail]) # Shape: (tail_length, num_nodes, 2)

        # Use a Gaussian kernel for weighted averaging, giving more weight to recent states in the tail.
        weights = np.exp(-0.5 * np.arange(self.tail_length)[::-1]**2 / (self.tail_length/4)**2)
        weights /= weights.sum() # Normalize

        # Calculate the smoothed target positions
        smoothed_positions = np.einsum('t,tni->ni', weights, positions_over_time)

        # Apply the new, stabilized state to the environment
        for i in range(num_nodes):
            env.nodes[i].pos = smoothed_positions[i]
            # Keep the old position for velocity calculation in the next step
            if coherent_tail[-1] is not None:
                env.nodes[i].last_pos = coherent_tail[-1]['state_snapshot'][i]['pos']

        return {'reinit_from': 'coherent_tail'}
