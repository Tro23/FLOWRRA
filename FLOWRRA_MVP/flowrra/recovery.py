"""
recovery.py

Implements the loop-level collapse mechanism for FLOWRRA.

FIXES:
- Removed undefined Environment_A reference
- Fixed node reference to use passed parameter
- Corrected velocity attribute access
- Improved snapshot creation logic
"""
import numpy as np
from typing import List, Dict, Any, Optional


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
                 collapse_threshold: float = 0.15,
                 tau: int = 2):
        self.history_length = history_length
        self.tail_length = tail_length
        self.collapse_threshold = collapse_threshold
        self.tau = tau
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        """Clears the history buffer."""
        self.history.clear()

    def assess_loop_coherence(self, coherence: float, nodes: List[Any]):
        """
        Records the overall coherence of the entire loop state.
        
        FIXED: Now properly accepts nodes as parameter.

        Args:
            coherence (float): The coherence score for the current loop.
            nodes (List[Any]): List of NodePositionND objects to snapshot.
        """
        # Create snapshot from node states
        snapshot = [{
            'id': n.id,
            'pos': n.pos.copy(),
            'velocity': n.velocity()  # FIXED: Call velocity() method
        } for n in nodes]

        # Add the current state and its coherence to the history
        self.history.append({
            'state_snapshot': snapshot,
            'coherence': coherence
        })
        
        # Trim history to the specified length
        if len(self.history) > self.history_length:
            self.history.pop(0)
    
    def needs_recovery(self) -> bool:
        """
        Checks if the last 'tau' steps have been unstable.
        
        Returns:
            bool: True if recovery is needed, False otherwise.
        """
        if len(self.history) < self.tau:
            return False
            
        # Check the last 'tau' entries
        recent_history = self.history[-self.tau:]
        is_unstable = all(h['coherence'] < self.collapse_threshold for h in recent_history)
        return is_unstable
            
    def collapse_and_reinitialize(self, nodes: List[Any]) -> Dict[str, Any]:
        """
        Triggers a collapse if the system is in an unstable state and re-initializes
        from a previously stable state in its history.
        
        FIXED: Now properly accepts nodes as parameter instead of Environment_A.
        
        Args:
            nodes (List[Any]): List of NodePositionND objects to reinitialize.

        Returns:
            dict: A dictionary containing information about the collapse event.
        """
        # 1. Find a coherent tail in the history
        coherent_tail = None
        
        # Search backwards for a stable sequence
        for i in range(len(self.history) - self.tail_length, -1, -1):
            tail = self.history[i : i + self.tail_length]
            if len(tail) < self.tail_length:
                continue
            
            # Check if all states in this tail are stable
            if all(h['coherence'] >= self.collapse_threshold for h in tail):
                coherent_tail = tail
                break

        # 2. Fallback: Random Jitter if no stable history found
        if coherent_tail is None:
            print("[WFC] No stable history found. Applying random jitter.")
            for node in nodes:
                # Simple random perturbation
                jitter = np.random.randn(node.dimensions) * 0.05
                node.pos = np.clip(node.pos + jitter, 0.0, 1.0)
                # Reset velocity tracking
                node.last_pos = node.pos.copy()
            return {'reinit_from': 'random_jitter', 'tail_length': 0}

        # 3. Manifold Smoothing: Weighted average across the coherent tail
        print(f"[WFC] Stable tail found (length={len(coherent_tail)}). Reconfiguring...")
        num_nodes = len(nodes)
        dimensions = nodes[0].dimensions
        
        # Extract positions: Shape (tail_len, num_nodes, dimensions)
        positions_over_time = np.array([
            [n_data['pos'] for n_data in step['state_snapshot']] 
            for step in coherent_tail
        ])

        # Use a Gaussian kernel for weighted averaging
        # Give more weight to recent states in the tail
        tail_indices = np.arange(self.tail_length)
        weights = np.exp(-0.5 * (tail_indices[::-1]**2) / ((self.tail_length / 4)**2))
        weights /= weights.sum()  # Normalize

        # Calculate smoothed target positions
        # weights shape: (tail_len,)
        # positions_over_time shape: (tail_len, num_nodes, dimensions)
        # Result shape: (num_nodes, dimensions)
        smoothed_positions = np.einsum('t,tnd->nd', weights, positions_over_time)

        # 4. Apply the new, stabilized state to the nodes
        for i, node in enumerate(nodes):
            node.pos = smoothed_positions[i].copy()
            # Reset velocity tracking to the recovered state
            node.last_pos = coherent_tail[-1]['state_snapshot'][i]['pos'].copy()

        return {
            'reinit_from': 'coherent_tail',
            'tail_length': len(coherent_tail),
            'tail_coherence_mean': np.mean([h['coherence'] for h in coherent_tail])
        }