"""
WaveFunctionCollapse.py

Implements the loop-level collapse mechanism. It maintains a history of coherent
loop states. When the system's overall coherence drops below a threshold for a
sustained period, this class identifies a recent, high-quality "tail" of states
and uses it to re-initialize the system, pulling it out of a failing configuration.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from EnvironmentA import Environment_A

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

    def record_step(self, snapshot: List[Dict], coherence: float, t: int):
        """
        Adds a new record of the system's state to the history buffer.

        Args:
            snapshot (list): The node snapshot from Environment_A.
            coherence (float): The calculated coherence for this snapshot.
            t (int): The current timestep.
        """
        self.history.append({'t': t, 'snapshot': snapshot, 'coherence': coherence})
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def check_for_collapse(self) -> bool:
        """
        Determines if a collapse should be triggered.

        A collapse occurs if the last `tau` steps have all been below the coherence threshold.

        Returns:
            True if a collapse should occur, False otherwise.
        """
        if len(self.history) < self.tau:
            return False
        # Check the coherence of the last `tau` recorded steps
        recent_window = self.history[-self.tau:]
        return all(step['coherence'] < self.collapse_threshold for step in recent_window)

    def _find_best_coherent_tail(self) -> Optional[List[List[Dict]]]:
        """
        Scans the history to find the best contiguous window of states (a "tail").

        The "best" tail is defined as the one with the highest average coherence. A recency
        bonus is added to prefer newer solutions.

        Returns:
            A list of snapshots representing the best tail, or None if no valid tail is found.
        """
        if len(self.history) < self.tail_length:
            return None

        best_avg_coherence = -1.0
        best_tail_start_idx = -1

        # Slide a window of size `tail_length` across the history
        for i in range(len(self.history) - self.tail_length):
            window = self.history[i : i + self.tail_length]
            avg_coherence = np.mean([step['coherence'] for step in window])
            # Add a small recency bonus to favor more recent stable patterns
            recency_bonus = 0.01 * (i / len(self.history))
            score = avg_coherence + recency_bonus

            if score > best_avg_coherence:
                best_avg_coherence = score
                best_tail_start_idx = i

        if best_tail_start_idx == -1:
            return None

        best_tail_snapshots = [
            self.history[i]['snapshot'] for i in range(best_tail_start_idx, best_tail_start_idx + self.tail_length)
        ]
        return best_tail_snapshots

    def collapse_and_reinitialize(self, env: Environment_A) -> dict:
        """
        Executes the collapse and re-initialization procedure.

        It finds the best coherent tail, smoothens it to create a new target state,
        and applies this new state to the environment's nodes.

        Args:
            env (Environment_A): The environment to apply the new state to.

        Returns:
            A dictionary containing metadata about the collapse event.
        """
        coherent_tail = self._find_best_coherent_tail()

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
            # Keep the angle from the end of the tail to preserve orientation momentum
            env.nodes[i].angle_idx = last_snapshot[i]['angle_idx']

        return {'reinit_from': 'tail_smoothing', 'source_t_start': self.history[0]['t']}
