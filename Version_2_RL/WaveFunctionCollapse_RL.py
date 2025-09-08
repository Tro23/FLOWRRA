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
        Resets the history buffer to prepare for a new episode.
        """
        self.history = []

    def record_step(self, nodes_snapshot: List[Dict[str, Any]], coherence: float, t: int):
        """
        Adds a new step's data to the history buffer, maintaining max length.
        """
        self.history.append({
            't': t,
            'nodes': nodes_snapshot,
            'coherence': coherence
        })
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def check_for_collapse(self) -> bool:
        """
        Checks if the system has been in a low-coherence state for too long.
        """
        if len(self.history) < self.tau:
            return False
        
        recent_coherences = [h['coherence'] for h in self.history[-self.tau:]]
        unstable_streak = all(c < self.collapse_threshold for c in recent_coherences)
        return unstable_streak
        
    def compute_coherence(self, nodes: List[Any]) -> float:
        """
        Calculates the average deviation from the loop center, normalized.
        Lower values = higher coherence. This is a proxy for how tightly the loop is formed.
        """
        if not nodes:
            return 1.0
            
        positions = np.array([n.pos for n in nodes])
        center = np.mean(positions, axis=0)
        distances_from_center = np.linalg.norm(positions - center, axis=1)
        coherence = np.mean(distances_from_center)
        
        # Simple normalization based on expected range of distances
        max_possible_coherence = 0.5 * np.sqrt(2) # Max distance from center in a unit square
        return coherence / max_possible_coherence

    def collapse_and_reinitialize(self, env: Environment_A) -> Dict[str, str]:
        """
        Performs a "wave function collapse" by restoring the system to a
        recent, high-coherence state, or by a random jitter if no such state
        can be found.

        Returns:
            A dictionary with metadata about the re-initialization.
        """
        # Find the most coherent "tail" of states in recent history
        coherent_tail = None
        for i in range(len(self.history) - self.tail_length, -1, -1):
            tail = self.history[i:i + self.tail_length]
            if all(h['coherence'] >= self.collapse_threshold for h in tail):
                coherent_tail = [h['nodes'] for h in tail]
                break

        if coherent_tail is None:
            # Fallback: If no good history exists, apply a small random perturbation.
            for node in env.nodes:
                node.pos = np.mod(node.pos + np.random.randn(2) * 0.05, 1.0)
            return {'reinit_from': 'random_jitter'}

        # --- Manifold Smoothing ---
        # As per the design, we smoothen the possibilities across the comet tail.
        # Here, we use a weighted average of the positions in the tail.
        num_nodes = len(coherent_tail[0])
        positions_over_time = np.array([[node['pos'] for node in snapshot] for snapshot in coherent_tail]) # Shape: (tail_length, num_nodes, 2)

        # Use a Gaussian kernel for weighted averaging, giving more weight to recent states in the tail.
        weights = np.exp(-0.5 * np.arange(self.tail_length)[::-1]**2 / (self.tail_length/4)**2)
        weights /= weights.sum() # Normalize

        # Calculate the smoothed target positions
        smoothed_positions = np.einsum('t,tni->ni', weights, positions_over_time)

        # Apply the new, stabilized state to the environment
        last_snapshot = coherent_tail[-1]
        for i in range(num_nodes):
            env.nodes[i].pos = smoothed_positions[i]
            # Keep the angle from the end of the tail to preserve orientation
            env.nodes[i].angle_idx = last_snapshot[i]['angle_idx']
        
        return {'reinit_from': 'coherent_tail_smooth'}
